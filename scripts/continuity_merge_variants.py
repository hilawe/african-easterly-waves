#!/usr/bin/env python3
"""The first stage of developing the merge-placement change, declared in section 12 of
`CONTINUITY_ACROSS_AFRICA_2026-11-09.md` before it was run.

At the 275 distinct steps of the stage sample, the released detector configuration is recomputed
and required to reproduce `detect_troughs`, and two experimental variants of the coarse merge run
beside it on the same prepared fields: VIN, the input-position assignment, and VCLIP, the clip at
exhaustion with a 5-degree radius. For each variant it counts RECOVERY (placed-away steps with a
variant final candidate within 500 km of QTrack's position), REGRESSION (sampled steps where the
released configuration has such a candidate and the variant has none) and COST over the African
and Atlantic sector (the change in final candidates and in pairs of final candidates closer than
500 km). Nothing here tracks, and nothing changes the released record.

    python3 scripts/continuity_merge_variants.py --out-dir DIR
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
SECTOR = (-40.0, 40.0, 0.0, 35.0)                    # lon_min, lon_max, lat_min, lat_max
CLIP_DEG = 5.0
VARIANTS = {"VIN": {"coarse_position_from_input": True}, "VCLIP": {"coarse_clip_radius_deg": CLIP_DEG}}


def in_sector(waves, sector=SECTOR):
    return [w for w in waves if sector[0] <= float(w["lon_mean"]) <= sector[1] and sector[2] <= float(w["lat_mean"]) <= sector[3]]


def close_pairs(waves, radius_km=C.RADIUS_KM):
    """Pairs of candidates closer than `radius_km` to each other."""
    la = [float(w["lat_mean"]) for w in waves]; lo = [float(w["lon_mean"]) for w in waves]
    n = 0
    for i in range(len(la)):
        for j in range(i + 1, len(la)):
            if float(C.km(la[i], lo[i], la[j], lo[j])) < radius_km:
                n += 1
    return n


def final_km(waves, lat, lon):
    return S.nearest_km(lat, lon, [w["lat_mean"] for w in waves], [w["lon_mean"] for w in waves])


def tally(rows, variant, radius_km=C.RADIUS_KM):
    """Recovery and regression of one variant over the sampled positions."""
    rec = sum(1 for r in rows if r["stage_class"] == "placed_away" and r[variant] <= radius_km)
    reg = sum(1 for r in rows if r["released"] <= radius_km and r[variant] > radius_km)
    return {"recovery": rec, "regression": reg,
            "regression_by_stage": {c: sum(1 for r in rows if r["stage_class"] == c and r["released"] <= radius_km and r[variant] > radius_km)
                                    for c in ("tracked", "candidate_not_tracked", "placed_away")}}


def season_variants(year, sample, campaign=S.CAMPAIGN, evidence=S.EVIDENCE):
    if year not in S.YEARS:
        raise SystemExit(f"REFUSED: {year} is not a development season of section 8")
    import pilot_alledge_replay as A
    import pilot_margin_replay as MR
    import pilot_stage_diagnostic as SD
    se = next(s for s in sample["seasons"] if s["year"] == year)
    ev = json.load(open(os.path.join(evidence, f"era5_{year}", f"tracking_era5_{year}.json"), encoding="utf-8"))
    case = SD.load_case(os.path.join(campaign, f"era5_{year}", "tracker_case.mat"), ev)
    if case["sha256"] != se["case_sha256"]:
        raise SystemExit(f"REFUSED: the {year} case is not the one the stage sample read")
    ds = ev["dataset_specific"]; ct, ft = float(ds["coarse_threshold"]), float(ds["fine_threshold"])
    absorb = bool(ds["tracker_flags"]["absorb"]); grids = MR.control_grids(case)
    allc = (np.ones(case["lat_c"].size, bool), np.ones(case["lon_c"].size, bool))
    allf = (np.ones(case["latgrid"].shape[0], bool), np.ones(case["longrid"].shape[1], bool))
    positions = {}
    for tr in se["tracks"]:
        for st in tr["steps"]:
            positions.setdefault(st["step"], []).append((tr["qtrack_id"], st))
    rows, steps = [], []
    for k in sorted(positions):
        t = float(case["time"][k])
        released = S.staged(case, k, ct, ft, grids, absorb)["final"]
        fields = A.prepared_fields(case, k, *allc, *allf)
        finals = {"released": released}
        for name, opt in VARIANTS.items():
            finals[name] = A.detect_from_prepared(fields, t, ct, ft, grids, absorb=absorb, **opt)
        steps.append({"step": k, **{f"{n}_finals_in_sector": len(in_sector(w)) for n, w in finals.items()},
                      **{f"{n}_close_pairs": close_pairs(in_sector(w)) for n, w in finals.items()}})
        for qid, st in positions[k]:
            rows.append({"qtrack_id": qid, "step": k, "stage_class": st["class"],
                         **{n: round(final_km(w, st["lat"], st["lon"]), 1) for n, w in finals.items()}})
    return {"year": year, "rows": rows, "steps": steps}


def totals(seasons):
    rows = [r for s in seasons for r in s["rows"]]; steps = [x for s in seasons for x in s["steps"]]
    out = {"positions": len(rows), "distinct_steps": len(steps),
           "placed_away": sum(1 for r in rows if r["stage_class"] == "placed_away"),
           "released_within": sum(1 for r in rows if r["released"] <= C.RADIUS_KM)}
    for n in ("released",) + tuple(VARIANTS):
        out[f"{n}_finals_in_sector"] = sum(x[f"{n}_finals_in_sector"] for x in steps)
        out[f"{n}_close_pairs"] = sum(x[f"{n}_close_pairs"] for x in steps)
    out["variants"] = {n: tally(rows, n) for n in VARIANTS}
    return out


def tables_text(rec):
    T = rec["totals"]
    L = ["# Merge-placement variants at the detector (declared section 12)", "",
         f"{T['positions']} sampled QTrack positions on {T['distinct_steps']} distinct steps of 1990 and 2006, {T['placed_away']} of them placed away. "
         f"VIN is the input-position assignment and VCLIP the {CLIP_DEG:g}-degree clip at exhaustion, both coarse merge only.", "",
         "| configuration | recovery of placed-away steps | regression | final candidates in the sector | close pairs under 500 km |",
         "| --- | --- | --- | --- | --- |",
         f"| released | 0 | 0 | {T['released_finals_in_sector']} | {T['released_close_pairs']} |"]
    for n, v in T["variants"].items():
        L.append(f"| {n} | {v['recovery']} | {v['regression']} | {T[f'{n}_finals_in_sector']} | {T[f'{n}_close_pairs']} |")
    L += ["", "Regression by stage class: " + "; ".join(f"{n} {v['regression_by_stage']}" for n, v in T["variants"].items()) + ".", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    pj, pm = (os.path.join(a.out_dir, f"continuity_merge_variants.{e}") for e in ("json", "md"))
    for p in (pj, pm):
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and records are never overwritten")
    sample = json.load(open(SAMPLE, encoding="utf-8"))
    if sample["generated_by"]["sha256"] != S.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the stage sample was not written by the current stage-sample script")
    seasons = [season_variants(y, sample) for y in S.YEARS]
    rec = {"generated_by": {"script": "scripts/continuity_merge_variants.py", "sha256": SCRIPT_SHA256_AT_LOAD,
                            "declared": "CONTINUITY_ACROSS_AFRICA_2026-11-09.md section 12", "variants": {k: v for k, v in VARIANTS.items()},
                            "sector": list(SECTOR), "sample": {"path": SAMPLE, "sha256": C.sha256_of(SAMPLE)}},
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
