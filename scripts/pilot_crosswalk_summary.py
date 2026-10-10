#!/usr/bin/env python3
"""Every count the eastern pilot's assessment derives from a crosswalk report, in one
retained artifact, so each number in the assessment has a field to trace to.

It reads only a retained crosswalk report (`scripts/pilot_track_crosswalk.py crosswalk`)
and, for the Africa-origin edge-band subset, the matching edge-band trace
(`scripts/pilot_edge_band_trace.py`). Nothing is recomputed from tracks.

- ITEM 2 AS WRITTEN. The brief's candidate recovered prefix is a pair whose control (B)
  track starts later and farther west than its treatment (C) partner, any pair class, any
  end. Reported in full, then restricted to C starts east of 40 E (the added strip), then
  to the same end (the superseded "loose" rule), with the in-season pairs (June to
  September by either side's first observation) of each.
- EASTERN STARTS. C tracks whose first observation lies east of 40 E: by month, by start
  latitude, how far west they end, first-detection region, later entry into the Africa
  polygon, for all months and for June to September.
- NO IDENTICAL PARTNER by the control's start band, half-open bands with a row for starts
  at exactly 40 E, all months and June to September.
- START SHIFTS by pair and by control track, and the month-to-month transitions of pairs.
- THE COMPARISON POOL of item 8, the potentially-censored flag's count and definition.
- With a trace, the in-season Africa-origin controls starting in 30 to 40 E and how many
  keep a partner over at least 80 percent of their life within 3 degrees.

    python3 scripts/pilot_crosswalk_summary.py --crosswalk <report> [--trace <edge-band trace>] --out <json>
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402

EAST_DEG = 40.0
COMPARISON_BAND_DEG = 5.0
SEASON = (6, 7, 8, 9)
LAT_BAND = (5.0, 20.0)
REACH_LONS = (40.0, 30.0, 20.0, 0.0)
BANDS = (("west of 20 W", -1e9, -20.0), ("20 W to 0", -20.0, 0.0), ("0 to 20 E", 0.0, 20.0),
         ("20 E to 30 E", 20.0, 30.0), ("30 E to 40 E", 30.0, 40.0))


def in_season(month):
    return int(month) in SEASON


def band_of(lon):
    for name, lo, hi in BANDS:
        if lo <= lon < hi:
            return name
    return "exactly 40 E" if lon == EAST_DEG else "east of 40 E"


def pct(values):
    if not len(values):
        return {"10": None, "50": None, "90": None}
    v = np.asarray(values, float)
    return {"10": round(float(np.percentile(v, 10)), 2), "50": round(float(np.percentile(v, 50)), 2), "90": round(float(np.percentile(v, 90)), 2)}


def pair_row(p, A, B):
    a, b = A[p["a"]], B[p["b"]]
    return {"a": p["a"], "b": p["b"], "class": p["class"], "prefix_steps": p["candidate_prefix_steps"], "prefix_degrees": p["candidate_prefix_degrees"],
            "start_shift_days": p["start_shift_days"], "b_start_lat_lon": [b["start_lat"], b["start_lon"]], "b_start_month": b["start_month"],
            "a_start_lat_lon": [a["start_lat"], a["start_lon"]], "a_start_month": a["start_month"], "same_end": p["class"] != "endpoint changed"}


def item2(report):
    A, B = report["attributes_a"], report["attributes_b"]
    lit = [p for p in report["pairs"] if B[p["b"]]["start_time"] < A[p["a"]]["start_time"] and B[p["b"]]["start_lon"] > A[p["a"]]["start_lon"]]
    east = [p for p in lit if B[p["b"]]["start_lon"] > EAST_DEG]
    same_end = [p for p in east if p["class"] != "endpoint changed"]

    def season_of(ps):
        return [p for p in ps if in_season(A[p["a"]]["start_month"]) or in_season(B[p["b"]]["start_month"])]
    east.sort(key=lambda p: (-p["candidate_prefix_steps"], B[p["b"]]["start_time"]))
    return {"definition": "a pair whose control track starts later and farther west than its treatment partner, any class, any end",
            "pairs": len(lit), "control_tracks": len({p["a"] for p in lit}), "by_class": dict(collections.Counter(p["class"] for p in lit)),
            "in_season_pairs": len(season_of(lit)),
            "east_of_40": {"pairs": len(east), "control_tracks": len({p["a"] for p in east}), "treatment_tracks": len({p["b"] for p in east}),
                           "in_season_pairs": len(season_of(east)), "rows": [pair_row(p, A, B) for p in east]},
            "east_of_40_same_end": {"pairs": len(same_end), "control_tracks": len({p["a"] for p in same_end}), "in_season_pairs": len(season_of(same_end))}}


def eastern_starts(report):
    B = report["attributes_b"]
    unpaired = set(report["unpaired_b"])

    def block(js):
        lats = [B[j]["start_lat"] for j in js]
        out = {"tracks": len(js), "unpaired": sum(1 for j in js if j in unpaired), "start_lat_percentiles": pct(lats),
               "start_lat_5_to_20_N": sum(1 for x in lats if LAT_BAND[0] <= x <= LAT_BAND[1]),
               "first_detection_region": dict(collections.Counter(B[j]["first_detection_region"] for j in js)),
               "later_entry_into_africa": sum(1 for j in js if B[j]["first_entry_into_africa_index"] not in (None, 0)),
               "lifetime_observations_percentiles": pct([B[j]["observations"] for j in js])}
        for lon in REACH_LONS:
            out[f"ending_west_of_{lon:g}"] = sum(1 for j in js if B[j]["end_lon"] < lon)
        return out
    east = [j for j, x in enumerate(B) if x["start_lon"] > EAST_DEG]
    season = [j for j in east if in_season(B[j]["start_month"])]
    return {"definition": f"treatment tracks whose first observation lies east of {EAST_DEG:g} E",
            "by_month": {str(m): sum(1 for j in east if B[j]["start_month"] == m) for m in range(1, 13)},
            "all_months": block(east), "june_to_september": block(season)}


def no_identical_partner(report):
    A = report["attributes_a"]
    best = collections.defaultdict(set)
    for p in report["pairs"]:
        best[p["a"]].add(p["class"])
    rows = {}
    for name in [b[0] for b in BANDS] + ["exactly 40 E"]:
        members = [i for i, x in enumerate(A) if band_of(x["start_lon"]) == name]
        season = [i for i in members if in_season(A[i]["start_month"])]
        rows[name] = {"changed": sum(1 for i in members if "identical" not in best[i]), "all": len(members),
                      "season_changed": sum(1 for i in season if "identical" not in best[i]), "season_all": len(season)}
    return {"bands": "half-open [lo, hi) in the control's start longitude, with starts at exactly 40 E their own row", "rows": rows}


def start_shifts(report):
    pairs = report["pairs"]
    by_a = collections.defaultdict(list)
    for p in pairs:
        by_a[p["a"]].append(p["start_shift_days"])
    A, B = report["attributes_a"], report["attributes_b"]
    months = collections.Counter(f"{A[p['a']]['start_month']}->{B[p['b']]['start_month']}" for p in pairs if A[p["a"]]["start_month"] != B[p["b"]]["start_month"])
    return {"pairs": {"negative": sum(1 for p in pairs if p["start_shift_days"] < 0), "zero": sum(1 for p in pairs if p["start_shift_days"] == 0),
                      "positive": sum(1 for p in pairs if p["start_shift_days"] > 0)},
            "control_tracks_with_a_partner": len(by_a),
            "control_tracks": {"every_partner_same_start": sum(1 for v in by_a.values() if all(s == 0 for s in v)),
                               "some_partner_earlier": sum(1 for v in by_a.values() if any(s < 0 for s in v)),
                               "some_partner_later": sum(1 for v in by_a.values() if any(s > 0 for s in v))},
            "month_transitions_of_pairs": dict(sorted(months.items(), key=lambda kv: [int(x) for x in kv[0].split("->")])),
            "note": "every pair's own signed shift is the field start_shift_days of the crosswalk report"}


def comparison_pool(report):
    A, B = report["attributes_a"], report["attributes_b"]
    pool = [p for p in report["pairs"] if EAST_DEG - COMPARISON_BAND_DEG <= A[p["a"]]["start_lon"] < EAST_DEG
            and B[p["b"]]["start_time"] == A[p["a"]]["start_time"] and B[p["b"]]["start_lat"] == A[p["a"]]["start_lat"]
            and B[p["b"]]["start_lon"] == A[p["a"]]["start_lon"]]
    return {"definition": "a control start within 5 degrees west of 40 E whose treatment partner starts at the same time, latitude and longitude",
            "pairs": len(pool), "rows": [pair_row(p, A, B) for p in pool]}


def africa_origin_edge_band(report, trace):
    A = report["attributes_a"]
    rows = {r["a"]: r for r in trace["rows"]}
    members = [i for i in rows if in_season(A[i]["start_month"]) and A[i]["first_detection_region"] == "AFR"]
    covered = [i for i in members if rows[i]["partner"]["steps_within_close"] / rows[i]["life"] >= 0.8]
    whole = [{"a": i, "b": rows[i]["partner"]["b"]} for i in covered if rows[i]["partner"]["steps_within_close"] == rows[i]["life"]]
    return {"definition": "controls starting in the trace's band, first observation in June to September, first detection AFR",
            "controls": len(members), "covered_80_percent_within_3_deg": len(covered), "covered_over_whole_life": whole}


def summarize(report, trace=None):
    out = {"generated_by": "scripts/pilot_crosswalk_summary.py", "script_sha256": X.digest(os.path.abspath(__file__)),
           "year": report.get("year"), "runs": report.get("runs"),
           "item2": item2(report), "eastern_starts": eastern_starts(report), "no_identical_partner": no_identical_partner(report),
           "start_shifts": start_shifts(report), "comparison_pool": comparison_pool(report),
           "potentially_censored": {"count": sum(1 for x in report["attributes_b"] if x["potentially_censored_start"]),
                                    "definition": f"first observation within {report['boundary_deg']:g} degrees of the treatment's east edge, one coarse cell"}}
    if trace is not None:
        # BOTH RUNS ARE BOUND, because the trace's rows index the crosswalk's control attributes
        # and name partners in its treatment run, and a missing digest never matches
        for side, name in (("a", "control"), ("b", "treatment")):
            t_digest = (trace.get("runs") or {}).get(side, {}).get("tracks_sha256")
            c_digest = (report.get("runs") or {}).get(side, {}).get("tracks_sha256")
            if not t_digest or not c_digest or t_digest != c_digest:
                raise SystemExit(f"REFUSED: the trace was not made from the crosswalk's {name} run")
        out["africa_origin_edge_band"] = africa_origin_edge_band(report, trace)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--crosswalk", required=True)
    ap.add_argument("--trace", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    report = json.load(open(args.crosswalk))
    trace = json.load(open(args.trace)) if args.trace else None
    out = summarize(report, trace)
    out["inputs"] = {"crosswalk": args.crosswalk, "crosswalk_sha256": X.digest(args.crosswalk),
                     "trace": args.trace, "trace_sha256": X.digest(args.trace) if args.trace else None}
    try:
        X.publish_json(args.out, out, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {args.out} exists and artifacts are never overwritten")
    print(f"{out['year']}: item 2 {out['item2']['pairs']} pairs, {out['item2']['east_of_40']['pairs']} east of 40 E; "
          f"{out['eastern_starts']['all_months']['tracks']} eastern starts; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
