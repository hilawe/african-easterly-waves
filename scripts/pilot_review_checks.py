#!/usr/bin/env python3
"""Two checks an independent review of the eastern pilot made from the stored tracks,
retained here so they can be rerun: whether the tropical eastern summer starts appear in
any audit selection, and how well the control's edge-band summer tracks are covered when
ANY treatment position may cover each observation.

1. TROPICAL EASTERN SUMMER STARTS: treatment tracks whose first observation lies east of
   40 E, between 5 and 20 N, in June to September, from the crosswalk report's attributes;
   for each audit cases file given, which of them it selected.
2. UNION COVERAGE: for every control track starting in [30, 40) E in June to September,
   the share of its observations with ANY treatment track's position within 3 degrees at
   the same time, whichever track, so a control split across several treatment tracks is
   still counted as covered. Median share and the number of controls covered at 80
   percent or more. This is more permissive than any identity match.

    python3 scripts/pilot_review_checks.py --crosswalk <report> --control-run <B dir> --treatment-run <C dir> \\
        --year 1990 --cases <audit cases json> [--cases ...] --out <json>
"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402
import pilot_track_crosswalk as P  # noqa: E402

EPOCH = dt.date(1900, 1, 1)
EAST_DEG, LAT_BAND, SEASON, BAND, CLOSE_DEG, COVERED = 40.0, (5.0, 20.0), (6, 7, 8, 9), (30.0, 40.0), 3.0, 0.8


def tropical_eastern_summer_starts(report):
    B = report["attributes_b"]
    return sorted(j for j, x in enumerate(B) if x["start_lon"] > EAST_DEG and LAT_BAND[0] <= x["start_lat"] <= LAT_BAND[1] and x["start_month"] in SEASON)


def selections(cases_paths):
    out = {}
    for path in cases_paths:
        c = json.load(open(path))
        out[os.path.basename(path)] = {"sha256": X.digest(path), "treatment_tracks": sorted({x["b"] for x in c["cases"]})}
    return out


def union_coverage(control_tracks, treatment_tracks):
    pos = {}
    for tr in treatment_tracks:
        for k, tt in enumerate(tr["time"]):
            pos.setdefault(round(float(tt), 6), []).append((float(tr["lat"][k]), float(tr["lon"][k])))
    rows = []
    for i, tr in enumerate(control_tracks):
        month = (EPOCH + dt.timedelta(days=float(tr["time"][0]))).month
        lon0 = float(tr["lon"][0])
        if not (BAND[0] <= lon0 < BAND[1] and month in SEASON):
            continue
        within = 0
        for k, tt in enumerate(tr["time"]):
            cand = pos.get(round(float(tt), 6), [])
            if cand and min(np.hypot(la - tr["lat"][k], lo - tr["lon"][k]) for la, lo in cand) <= CLOSE_DEG:
                within += 1
        rows.append({"control_track": i, "observations": int(tr["time"].size), "share_within_close": round(within / tr["time"].size, 4)})
    shares = np.array([r["share_within_close"] for r in rows]) if rows else np.array([])
    return {"definition": f"share of a control's observations with any treatment position within {CLOSE_DEG:g} degrees at the same time; controls starting in [{BAND[0]:g}, {BAND[1]:g}) E in June to September",
            "controls": len(rows), "median_share": None if not rows else round(float(np.median(shares)), 4),
            "share_percentiles": None if not rows else {str(p): round(float(np.percentile(shares, p)), 4) for p in (25, 50, 75)},
            "controls_covered_at_least_80_percent": int((shares >= COVERED).sum()), "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--crosswalk", required=True)
    ap.add_argument("--control-run", required=True)
    ap.add_argument("--treatment-run", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--cases", action="append", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    report = json.load(open(args.crosswalk))
    a, b = P.load_run(args.control_run, year=args.year), P.load_run(args.treatment_run, year=args.year)
    for side, run in (("a", a), ("b", b)):
        if report["runs"][side]["tracks_sha256"] != run["tracks_sha256"]:
            raise SystemExit(f"REFUSED: the crosswalk was not made from the {side} run given")
    trop = tropical_eastern_summer_starts(report)
    sel = selections(args.cases)
    B = report["attributes_b"]
    out = {"generated_by": "scripts/pilot_review_checks.py", "script_sha256": X.digest(os.path.abspath(__file__)), "year": args.year,
           "crosswalk_sha256": X.digest(args.crosswalk), "runs": {"a": a["tracks_sha256"], "b": b["tracks_sha256"]},
           "tropical_eastern_summer_starts": {"definition": "treatment first observation east of 40 E, 5 to 20 N, June to September", "count": len(trop),
                                              "tracks": [{"track": j, "start_lat_lon": [B[j]["start_lat"], B[j]["start_lon"]], "start_month": B[j]["start_month"],
                                                          "end_lon": B[j]["end_lon"], "observations": B[j]["observations"]} for j in trop],
                                              "selected_in": {name: sorted(set(trop) & set(s["treatment_tracks"])) for name, s in sel.items()},
                                              "selected_in_any": sorted(set(trop) & {t for s in sel.values() for t in s["treatment_tracks"]})},
           "audit_selections": sel, "union_coverage": union_coverage(a["tracks"], b["tracks"])}
    try:
        X.publish_json(args.out, out, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {args.out} exists and artifacts are never overwritten")
    u = out["union_coverage"]
    print(f"{args.year}: {len(trop)} tropical eastern summer starts, {len(out['tropical_eastern_summer_starts']['selected_in_any'])} in any selection; "
          f"union coverage of {u['controls']} edge-band controls: median {u['median_share']}, {u['controls_covered_at_least_80_percent']} at 80 percent or more; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
