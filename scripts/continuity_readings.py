#!/usr/bin/env python3
"""The field reading and the association reading of the continuity effort, declared in section 10
of `CONTINUITY_ACROSS_AFRICA_2026-11-09.md` before they were run.

THE FIELD READING. At every placed-away and candidate-not-tracked step of the stage sample it
reads the tracker's own 700 hPa inputs at QTrack's position: a TROUGH CROSSING (the
nine-point-smoothed meridional wind turning from zero or northerly on the west to southerly on the
east within 4 degrees of QTrack's longitude, on at least two of the three fine rows nearest QTrack's
latitude) and a LOCAL CURVATURE MAXIMUM (a cell of the prepared fine curvature anomaly within 300 km
that exceeds all eight neighbors and reaches the fine threshold). Each step is supported, partial or
unsupported. For a placed-away step inside a merged coarse region it records whether that maximum
lies 500 km or more from the region's candidate center.

THE ASSOCIATION READING. It replays each season through the production association with a trace at
the candidate-not-tracked steps, refusing unless the replay's finished tracks equal the stored
campaign tracks in content and order, and records for the final candidate nearest QTrack's position
within 500 km the track that claimed it or was seeded from it and that track's fate.

Neither reading decides that QTrack's feature is real.

    python3 scripts/continuity_readings.py --out-dir DIR
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
import continuity_stage_sample as S  # noqa: E402

with open(os.path.abspath(__file__), "rb") as _fh:
    SCRIPT_SHA256_AT_LOAD = hashlib.sha256(_fh.read()).hexdigest()

SAMPLE = "docs/aewc_v2/artifacts/continuity_census/continuity_stage_sample.json"
CROSSING_DEG = 4.0
ROWS_NEEDED = 2
MAXIMUM_KM = 300.0
DISTINCT_KM = 500.0
FATES = ("finished", "pruned_in_loop", "removed_by_final_speed_filter", "removed_as_single_observation", "not_held")


def crossing_on_row(v_row, lons, qlon, max_deg=CROSSING_DEG):
    """The distance in degrees from `qlon` to the nearest place where v turns from <= 0 on the west
    to > 0 on the east (the trough-axis order), within `max_deg`, or None. Longitudes ascend."""
    v = np.asarray(v_row, float); lo = np.asarray(lons, float)
    best = None
    for i in range(v.size - 1):
        if np.isfinite(v[i]) and np.isfinite(v[i + 1]) and v[i] <= 0.0 < v[i + 1]:
            x = lo[i] + (lo[i + 1] - lo[i]) * (-v[i] / (v[i + 1] - v[i]))
            d = abs(x - qlon)
            if d <= max_deg and (best is None or d < best):
                best = d
    return best


def trough_crossing(v, latgrid_f, longrid_f, qlat, qlon):
    lats = latgrid_f[:, 0]; lons = longrid_f[0, :]
    rows = sorted({int(np.argmin(np.abs(lats - (qlat + dl)))) for dl in (-1.0, 0.0, 1.0)})
    dists = [crossing_on_row(v[r], lons, qlon) for r in rows]
    found = [d for d in dists if d is not None]
    return {"rows_read": len(rows), "rows_with_crossing": len(found), "present": len(found) >= ROWS_NEEDED,
            "nearest_deg": None if not found else round(min(found), 2)}


def local_maximum(curv, latgrid_f, longrid_f, ft, qlat, qlon, radius_km=MAXIMUM_KM):
    """The nearest cell within `radius_km` that exceeds all eight neighbors and reaches `ft`."""
    c = np.where(np.isfinite(curv), curv, -np.inf)
    best = None
    near_r = np.flatnonzero(np.abs(latgrid_f[:, 0] - qlat) <= 4.0)            # 300 km lies inside 4 degrees
    near_c = np.flatnonzero(np.abs(longrid_f[0, :] - qlon) <= 4.0 / max(np.cos(np.deg2rad(qlat + 4.0 * np.sign(qlat))), 0.2))
    rows = [i for i in near_r if 0 < i < c.shape[0] - 1]
    cols = [j for j in near_c if 0 < j < c.shape[1] - 1]
    for i in rows:
        for j in cols:
            x = c[i, j]
            if x < ft:
                continue
            nb = c[i - 1:i + 2, j - 1:j + 2].copy(); nb[1, 1] = -np.inf
            if x > nb.max():
                d = float(C.km(qlat, qlon, latgrid_f[i, j], longrid_f[i, j]))
                if d <= radius_km and (best is None or d < best[2]):
                    best = (float(latgrid_f[i, j]), float(longrid_f[i, j]), d)
    return None if best is None else {"lat": best[0], "lon": best[1], "km": round(best[2], 1)}


def support(crossing_present, maximum):
    n = int(bool(crossing_present)) + int(maximum is not None)
    return ("unsupported", "partial", "supported")[n]


def attribute(entry, qlat, qlon, result, finished, radius_km=C.RADIUS_KM):
    """At one traced step, the fate of the track holding the final candidate nearest the position
    within `radius_km`."""
    cands = entry["candidates"]
    if not cands:
        return {"candidate": None, "fate": "no_candidate"}
    d = [float(C.km(qlat, qlon, c["lat_mean"], c["lon_mean"])) for c in cands]
    i = int(np.argmin(d))
    if d[i] > radius_km:
        return {"candidate": None, "fate": "no_candidate", "nearest_km": round(d[i], 1)}
    idx = cands[i]["index"]
    holder = next((c["birth"] for c in entry["claims"] if c["candidate"] == idx), None) or \
        next((s["birth"] for s in entry["seeds"] if s["candidate"] == idx), None)
    out = {"candidate": idx, "candidate_km": round(d[i], 1), "holder": holder}
    if holder is None:
        out["fate"] = "not_held"
        return out
    if holder not in result["fates"]:
        raise SystemExit(f"REFUSED: the track {holder} holding a candidate at step {entry['step']} has no recorded fate")
    out["fate"] = result["fates"][holder]
    if out["fate"] == "pruned_in_loop":
        out["n_obs"] = result["pruned_at"][holder]["n_obs"]
    if out["fate"] == "finished":
        f = finished[result["finished_index"][holder]]
        k = f["steps"].index(entry["step"])
        out["stored_km"] = round(float(C.km(qlat, qlon, f["lat"][k], f["lon"][k])), 1)
        out["n_obs"] = len(f["steps"])
    return out


def season_readings(year, sample, sample_path=SAMPLE, campaign=S.CAMPAIGN, evidence=S.EVIDENCE, replay=True):
    if year not in S.YEARS:
        raise SystemExit(f"REFUSED: {year} is not a development season of section 8")
    import pilot_alledge_replay as A
    import pilot_margin_replay as MR
    import pilot_stage_diagnostic as SD
    import pilot_track_crosswalk as P
    se = next(s for s in sample["seasons"] if s["year"] == year)
    run_dir = os.path.join(campaign, f"era5_{year}")
    run = P.load_run(run_dir, year=year)
    ev = json.load(open(os.path.join(evidence, f"era5_{year}", f"tracking_era5_{year}.json"), encoding="utf-8"))
    if run["tracks_sha256"] != ev["dataset_specific"]["tracks_sha256"]:
        raise SystemExit(f"REFUSED: the {year} campaign run is not the one the evidence records")
    case = SD.load_case(os.path.join(run_dir, "tracker_case.mat"), ev)
    if case["sha256"] != se["case_sha256"]:
        raise SystemExit(f"REFUSED: the {year} case is not the one the stage sample read")
    ds = ev["dataset_specific"]; ct, ft = float(ds["coarse_threshold"]), float(ds["fine_threshold"])
    flags = ds["tracker_flags"]; grids = MR.control_grids(case)
    allc = (np.ones(case["lat_c"].size, bool), np.ones(case["lon_c"].size, bool))
    allf = (np.ones(case["latgrid"].shape[0], bool), np.ones(case["longrid"].shape[1], bool))
    lost = [(tr["qtrack_id"], st) for tr in se["tracks"] for st in tr["steps"] if st["class"] in ("placed_away", "candidate_not_tracked")]
    cache, field = {}, []
    for qid, st in lost:
        k = st["step"]
        if k not in cache:
            fields = A.prepared_fields(case, k, *allc, *allf)
            cache[k] = (fields, S.staged(case, k, ct, ft, grids, bool(flags["absorb"])))
        fields, stg = cache[k]
        cr = trough_crossing(fields["fine_meridional_wind_smoothed"], grids["latgrid_f"], grids["longrid_f"], st["lat"], st["lon"])
        mx = local_maximum(fields["fine_curvature"], grids["latgrid_f"], grids["longrid_f"], ft, st["lat"], st["lon"])
        row = {"qtrack_id": qid, "step": k, "stage_class": st["class"], "crossing": cr, "maximum": mx, "support": support(cr["present"], mx)}
        if st["class"] == "placed_away" and mx is not None:
            i = int(np.argmin(np.abs(grids["lat_c"] - st["lat"]))); j = int(np.argmin(np.abs(grids["lon_c"] - st["lon"])))
            holders = [w for w in stg["coarse"] if bool(np.asarray(w["region"])[i, j])]
            if holders:
                dc = min(float(C.km(mx["lat"], mx["lon"], w["lat_mean"], w["lon_mean"])) for w in holders)
                row["maximum_to_holder_center_km"] = round(dc, 1); row["distinct_from_holder"] = dc >= DISTINCT_KM
        field.append(row)
    assoc = []
    if replay:
        cand_steps = sorted({st["step"] for _, st in lost if st["class"] == "candidate_not_tracked"})
        boxes = {}
        for _, st in lost:
            if st["class"] == "candidate_not_tracked":
                boxes.setdefault(st["step"], []).append({"lat": [st["lat"] - 5.0, st["lat"] + 5.0], "lon": [st["lon"] - 5.0, st["lon"] + 5.0]})
        trace = {"steps": set(cand_steps), "box": boxes[cand_steps[0]][0], "families": [], "boxes_by_step": boxes, "lineage_steps": set(),
                 "point": None, "radius_deg": 5.0}
        finished, _counts, _bound, _ = A.run_season(case, grids, ct, ft, flags, trace=trace)
        gate = MR.archive_gate(finished, run)
        if not gate["passed"]:
            raise SystemExit(f"REFUSED: the {year} replay's finished tracks are not the stored campaign tracks")
        by_step = {e["step"]: e for e in trace["result"]["steps"]}
        for qid, st in lost:
            if st["class"] == "candidate_not_tracked":
                assoc.append({"qtrack_id": qid, "step": st["step"], **attribute(by_step[st["step"]], st["lat"], st["lon"], trace["result"], finished)})
    return {"year": year, "field": field, "association": assoc, "steps_staged": len(cache)}


def totals(seasons):
    f = [r for s in seasons for r in s["field"]]; a = [r for s in seasons for r in s["association"]]
    sup = {c: {k: sum(1 for r in f if r["stage_class"] == c and r["support"] == k) for k in ("supported", "partial", "unsupported")}
           for c in ("placed_away", "candidate_not_tracked")}
    distinct = [r for r in f if "distinct_from_holder" in r]
    fates = {k: sum(1 for r in a if r["fate"] == k) for k in FATES + ("no_candidate",)}
    pruned_n = sorted(r["n_obs"] for r in a if r["fate"] == "pruned_in_loop")
    stored = [r["stored_km"] for r in a if r["fate"] == "finished"]
    return {"field_steps": len(f), "support_by_stage": sup,
            "placed_away_with_maximum_in_holder": len(distinct), "distinct_from_holder": sum(1 for r in distinct if r["distinct_from_holder"]),
            "association_steps": len(a), "fates": fates, "pruned_n_obs": pruned_n,
            "finished_stored_km": sorted(stored)}


def tables_text(rec):
    T = rec["totals"]
    L = ["# Field and association readings of the continuity effort (declared section 10)", "",
         f"Field reading over {T['field_steps']} placed-away and candidate-not-tracked steps of the stage sample. A trough crossing within "
         f"{CROSSING_DEG:g} degrees on {ROWS_NEEDED} of 3 rows, and a local curvature maximum at or above the fine threshold within {MAXIMUM_KM:g} km.", "",
         "| stage class | supported | partial | unsupported |", "| --- | --- | --- | --- |"]
    for c, row in T["support_by_stage"].items():
        L.append(f"| {c.replace('_', ' ')} | {row['supported']} | {row['partial']} | {row['unsupported']} |")
    L += ["", f"Placed-away steps with a local maximum and a merged holder region: {T['placed_away_with_maximum_in_holder']}, of which "
              f"{T['distinct_from_holder']} have the maximum {DISTINCT_KM:g} km or more from the region's candidate center.", "",
          f"Association reading over {T['association_steps']} candidate-not-tracked steps, the fate of the track holding the nearest candidate:", "",
          "| fate | steps |", "| --- | --- |"]
    L += [f"| {k.replace('_', ' ')} | {n} |" for k, n in T["fates"].items()]
    L += ["", f"Observations of the tracks pruned in the loop: {T['pruned_n_obs']}. Stored distances of finished holders (km): {T['finished_stored_km']}.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    pj, pm = (os.path.join(a.out_dir, f"continuity_readings.{e}") for e in ("json", "md"))
    for p in (pj, pm):
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and records are never overwritten")
    sample = json.load(open(SAMPLE, encoding="utf-8"))
    if sample["generated_by"]["sha256"] != S.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the stage sample was not written by the current stage-sample script")
    seasons = [season_readings(y, sample) for y in S.YEARS]
    rec = {"generated_by": {"script": "scripts/continuity_readings.py", "sha256": SCRIPT_SHA256_AT_LOAD,
                            "declared": "CONTINUITY_ACROSS_AFRICA_2026-11-09.md section 10", "crossing_deg": CROSSING_DEG, "rows_needed": ROWS_NEEDED,
                            "maximum_km": MAXIMUM_KM, "distinct_km": DISTINCT_KM, "sample": {"path": SAMPLE, "sha256": C.sha256_of(SAMPLE)}},
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
