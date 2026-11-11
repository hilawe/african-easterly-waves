#!/usr/bin/env python3
"""The stage sample of the continuity effort, declared in section 8 of
`CONTINUITY_ACROSS_AFRICA_2026-11-09.md` before it was run.

For two coast-reaching QTrack tracks per census class in each of 1990 and 2006 (the earliest by
first observation, ties by identifier), at every observation up to and including the first
Atlantic-side observation, the detector's stages are recomputed from the season's retained
campaign case under the run's thresholds, through the replay's own preparation and staged
detection, and the staged final candidates must reproduce `detect_troughs` at that step or the run
refuses. Each step is classed by the first stage at which nothing of the record lies within 500 km
of QTrack's position. The classes locate where the record loses that position and decide nothing
about whether QTrack's feature is real or whether a change would recover it.

    python3 scripts/continuity_stage_sample.py --out-dir DIR
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import continuity_census as C  # noqa: E402

with open(os.path.abspath(__file__), "rb") as _fh:
    SCRIPT_SHA256_AT_LOAD = hashlib.sha256(_fh.read()).hexdigest()

YEARS = (1990, 2006)
PER_CLASS = 2
CENSUS = "docs/aewc_v2/artifacts/continuity_census/continuity_census.json"
CAMPAIGN = "data/protocol_runs/campaign"
EVIDENCE = "docs/aewc_v2/evidence/validation/protocol_campaign"
STAGE_CLASSES = ("tracked", "candidate_not_tracked", "placed_away", "signal_without_axis", "masked_by_westerlies", "below_threshold")


def select(census_rows, artifact_rows, per_class=PER_CLASS):
    """Section 8's rule: per census class, the coast-reaching QTrack tracks with the earliest first
    observation, ties by the lower identifier."""
    first = {r["id"]: r["first"]["time"] for r in artifact_rows}
    out = []
    for c in C.Q_CLASSES:
        rows = sorted((r for r in census_rows if r["outcome"] == "atlantic_side" and r["class"] == c), key=lambda r: (first[r["id"]], r["id"]))
        out += [(c, r["id"]) for r in rows[:per_class]]
    return out


def stage_class(stored_km, final_km, axis_km, westerly, above, radius=C.RADIUS_KM):
    """The first stage at which nothing of the record lies within `radius` of the position."""
    if stored_km <= radius:
        return "tracked"
    if final_km <= radius:
        return "candidate_not_tracked"
    if axis_km <= radius:
        return "placed_away"
    if westerly:
        return "masked_by_westerlies"
    return "signal_without_axis" if above else "below_threshold"


def nearest_km(lat, lon, lats, lons):
    if len(lats) == 0:
        return float("inf")
    return float(np.min(np.atleast_1d(C.km(lat, lon, np.asarray(lats, float), np.asarray(lons, float)))))


def read_position(st, grids, ct, lat, lon, stored):
    """The stages' content near one position, from one step's staged detection `st`."""
    lat_c, lon_c = grids["lat_c"], grids["lon_c"]
    i, j = int(np.argmin(np.abs(lat_c - lat))), int(np.argmin(np.abs(lon_c - lon)))
    westerly = bool(st["westerly"][i, j])
    curv = float(st["curvature_unmasked"][i, j])
    above = (not westerly) and np.isfinite(curv) and curv >= ct
    vla = np.concatenate([np.asarray(a, float) for a, _ in st["axes"]]) if st["axes"] else np.array([])
    vlo = np.concatenate([np.asarray(b, float) for _, b in st["axes"]]) if st["axes"] else np.array([])
    axis_km = nearest_km(lat, lon, vla, vlo)
    holder = [w for w in st["coarse"] if bool(np.asarray(w["region"])[i, j])]
    holder_km = nearest_km(lat, lon, [w["lat_mean"] for w in holder], [w["lon_mean"] for w in holder]) if holder else None
    final_km = nearest_km(lat, lon, [w["lat_mean"] for w in st["final"]], [w["lon_mean"] for w in st["final"]])
    return {"cell": [float(lat_c[i]), float(lon_c[j])], "westerly": westerly, "curvature_over_threshold": round(curv / ct, 3) if np.isfinite(curv) else None,
            "above": bool(above), "axis_km": round(axis_km, 1), "coarse_holder_center_km": None if holder_km is None else round(holder_km, 1),
            "final_km": round(final_km, 1), "stored_km": round(stored, 1),
            "class": stage_class(stored, final_km, axis_km, westerly, above)}


def staged(case, k, ct, ft, grids, absorb):
    """One step's stages through the replay's functions, refusing unless the final candidates
    reproduce `detect_troughs`."""
    import pilot_alledge_replay as A
    import pilot_event_continuity as EC
    import pilot_margin_replay as MR
    from aew.v1port.detection import detect_troughs
    allc = (np.ones(case["lat_c"].size, bool), np.ones(case["lon_c"].size, bool))
    allf = (np.ones(case["latgrid"].shape[0], bool), np.ones(case["longrid"].shape[1], bool))
    t = float(case["time"][k])
    st = EC.stages_from_prepared(A.prepared_fields(case, k, *allc, *allf), t, ct, ft, grids, absorb=absorb)
    oracle = detect_troughs(t, grids["latgrid_c"], grids["longrid_c"], case["u_c"][k], case["currv_anom_c"][k], case["advcurrv_anom_c"][k],
                            grids["latgrid_f"], grids["longrid_f"], case["currv_anom"][k], coarse_threshold=ct, fine_threshold=ft, absorb=absorb)
    if MR.signature(st["final"]) != MR.signature(oracle):
        raise SystemExit(f"REFUSED: at step {k} the staged detection does not reproduce detect_troughs")
    return st


def season_sample(year, census, census_path=CENSUS, campaign=CAMPAIGN, evidence=EVIDENCE):
    if year not in YEARS:
        raise SystemExit(f"REFUSED: {year} is not a development season of section 8")
    import pilot_margin_replay as MR
    import pilot_stage_diagnostic as SD
    import qtrack_pairing_pilot as M
    srec = next(s for s in census["seasons"] if s["year"] == year)
    art_path = srec["artifact"]["path"]
    if C.sha256_of(art_path) != srec["artifact"]["sha256"]:
        raise SystemExit(f"REFUSED: {art_path} is not the artifact the census read")
    art = json.load(open(art_path, encoding="utf-8"))
    for key in ("this_record", "qtrack"):
        if C.sha256_of(art["inputs"][key]["path"]) != art["inputs"][key]["sha256"]:
            raise SystemExit(f"REFUSED: {art['inputs'][key]['path']} is not the file the {year} artifact names")
    run_record = json.load(open(os.path.join(evidence, f"era5_{year}", f"tracking_era5_{year}.json"), encoding="utf-8"))
    if run_record["dataset_specific"]["tracks_sha256"] != art["inputs"]["this_record"]["sha256"]:
        raise SystemExit(f"REFUSED: the {year} run record does not name the tracks the census read")
    case = SD.load_case(os.path.join(campaign, f"era5_{year}", "tracker_case.mat"), run_record)
    ds = run_record["dataset_specific"]; ct, ft = float(ds["coarse_threshold"]), float(ds["fine_threshold"])
    absorb = bool(ds["tracker_flags"]["absorb"])
    grids = MR.control_grids(case)
    step_of = {C.time_key(t): k for k, t in enumerate(case["time"])}
    q = {t["id"]: t for t in M.load_qtrack(art["inputs"]["qtrack"]["path"])[0]}
    stored_idx = C.index_by_time(M.load_ours(art["inputs"]["this_record"]["path"])[0])
    a_rows = {r["id"]: r for r in art["sides"]["qtrack"]["tracks"]}
    cache, tracks = {}, []
    for cls, qid in select(srec["qtrack"], art["sides"]["qtrack"]["tracks"]):
        r = a_rows[qid]; t = q[qid]; C.check_track(r, t, "qtrack", year)
        steps = []
        for jj in range(r["first_atlantic_side"]["index"] + 1):
            key = C.time_key(t["time"][jj])
            if key not in step_of:
                raise SystemExit(f"REFUSED: QTrack {qid} of {year} has a time the case does not hold")
            k = step_of[key]
            if k not in cache:
                cache[k] = staged(case, k, ct, ft, grids, absorb)
            hit = stored_idx.get(key)
            stored = nearest_km(t["lat"][jj], t["lon"][jj], hit[0], hit[1]) if hit is not None else float("inf")
            steps.append({"step": k, "lat": round(float(t["lat"][jj]), 3), "lon": round(float(t["lon"][jj]), 3),
                          **read_position(cache[k], grids, ct, float(t["lat"][jj]), float(t["lon"][jj]), stored)})
        tracks.append({"year": year, "qtrack_id": qid, "census_class": cls, "steps": steps})
    return {"year": year, "case_sha256": case["sha256"], "thresholds": [ct, ft], "steps_bound": len(cache), "tracks": tracks}


def totals(seasons):
    rows = [(tr["census_class"], s) for se in seasons for tr in se["tracks"] for s in tr["steps"]]
    by = {c: {k: sum(1 for cc, s in rows if cc == c and s["class"] == k) for k in STAGE_CLASSES} for c in C.Q_CLASSES}
    allc = {k: sum(1 for _, s in rows if s["class"] == k) for k in STAGE_CLASSES}
    placed = [s["coarse_holder_center_km"] for _, s in rows if s["class"] == "placed_away" and s["coarse_holder_center_km"] is not None]
    return {"tracks": sum(len(se["tracks"]) for se in seasons), "steps": len(rows), "by_census_class": by, "all": allc,
            "placed_away_with_coarse_holder": len(placed),
            "placed_away_holder_center_km_quartiles": [round(float(x), 1) for x in np.percentile(placed, [25, 50, 75])] if placed else None}


def tables_text(rec):
    T = rec["totals"]
    L = ["# Stage sample of the continuity effort (declared section 8)", "",
         f"{T['tracks']} QTrack tracks, {T['steps']} steps up to the first Atlantic-side observation, in {', '.join(map(str, YEARS))}. "
         f"Each step is classed by the first stage at which nothing of the record lies within {C.RADIUS_KM:g} km of QTrack's position.", "",
         "| census class | " + " | ".join(k.replace("_", " ") for k in STAGE_CLASSES) + " |", "| --- |" + " --- |" * len(STAGE_CLASSES)]
    for c, row in T["by_census_class"].items():
        L.append(f"| {c.replace('_', ' ')} | " + " | ".join(str(row[k]) for k in STAGE_CLASSES) + " |")
    L.append("| all | " + " | ".join(str(T["all"][k]) for k in STAGE_CLASSES) + " |")
    L += ["", f"Placed-away steps whose coarse cell lies inside a merged coarse region: {T['placed_away_with_coarse_holder']}, the region's "
              f"candidate center at quartiles {T['placed_away_holder_center_km_quartiles']} km from QTrack's position.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    pj, pm = (os.path.join(a.out_dir, f"continuity_stage_sample.{e}") for e in ("json", "md"))
    for p in (pj, pm):
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and records are never overwritten")
    census = json.load(open(CENSUS, encoding="utf-8"))
    if census["generated_by"]["sha256"] != C.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the census record was not written by the current census script")
    seasons = [season_sample(y, census) for y in YEARS]
    rec = {"generated_by": {"script": "scripts/continuity_stage_sample.py", "sha256": SCRIPT_SHA256_AT_LOAD,
                            "declared": "CONTINUITY_ACROSS_AFRICA_2026-11-09.md section 8", "radius_km": C.RADIUS_KM, "per_class": PER_CLASS,
                            "census": {"path": CENSUS, "sha256": C.sha256_of(CENSUS)}},
           "seasons": seasons, "totals": totals(seasons)}
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump(rec, fh)
    text = tables_text(rec)
    with open(pm, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
