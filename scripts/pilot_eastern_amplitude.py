#!/usr/bin/env python3
"""The amplitude comparison of `EASTERN_AMPLITUDE_COMPARISON_2026-10-11.md`. It reads the
retained 700 hPa vorticity maxima and compares, within each scored map's fixed case box,
the peak amplitudes within the tolerance of the read geometry (near) with the box's other
peaks (away), and reports each standing control's box amplitudes. Descriptive only.

    python3 scripts/pilot_eastern_amplitude.py --maxima-2006 <json> --maxima-1990 <json> --reference <json> --out <json>
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_alledge_replay as A  # noqa: E402
import pilot_eastern_maxima as MX  # noqa: E402
import pilot_eastern_measurement as EM  # noqa: E402

TOLS = (3.0, 5.0)
MOVING = ("1137_moving", "R8_east", "R9_first")
STANDING = ("1137_standing",)
THRESHOLD_UNITS = 1.0                                                    # 1.0e-5 per second in the stored units


def overlap(near, away):
    """Share of near-away pairs in which the near peak is larger, ties counted as half."""
    if not near or not away:
        return None
    s = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in near for b in away)
    return s / (len(near) * len(away))


def describe(vals):
    return None if not vals else {"n": len(vals), "median": round(statistics.median(vals), 3), "max": round(max(vals), 3), "min": round(min(vals), 3)}


def summarize(rows, t):
    ov = [r[t]["overlap"] for r in rows if r[t]["overlap"] is not None]
    nm = [r[t]["near"]["median"] for r in rows if r[t]["near"]]
    return {"maps": len(rows), "available": len(ov), "unavailable": [r["date"] for r in rows if r[t]["overlap"] is None],
            "overlap_median": round(statistics.median(ov), 3) if ov else None, "overlap_range": [round(min(ov), 3), round(max(ov), 3)] if ov else None,
            "maps_above": sum(1 for x in ov if x > 0.5), "maps_at": sum(1 for x in ov if x == 0.5), "maps_below": sum(1 for x in ov if x < 0.5),
            "near_median_over_maps": round(statistics.median(nm), 3) if nm else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--maxima-2006", "--maxima-1990", "--reference", "--out"):
        ap.add_argument(name, required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    r6, r9 = json.load(open(args.maxima_2006)), json.load(open(args.maxima_1990))
    for r in (r6, r9):
        vals = [p["value"] for d in r["maxima_by_date"].values() for p in d["B"]]
        if min(vals) <= THRESHOLD_UNITS:
            raise SystemExit("REFUSED: a stored value is not above the threshold in units of 1e-5 per second")
    ref = json.load(open(args.reference))["reference"]
    maps = {(m["date"], m["segment"]): m for m in ref["maps"] if m["scored"]}
    rows = []
    for row in r6["scored_maps"]:
        m = maps[(row["date"], row["segment"])]
        box = ref["segments"][row["segment"]]["case_box"]
        pts = r6["maxima_by_date"][row["date"]]["B"]
        out = {"date": row["date"], "segment": row["segment"]}
        for t in TOLS:
            inbox = [p for p in pts if MX.in_box(p["lat"], p["lon"], box)]
            near = [p["value"] for p in inbox if EM.ref_distance(p["lat"], p["lon"], m) <= t]
            away = [p["value"] for p in inbox if EM.ref_distance(p["lat"], p["lon"], m) > t]
            outside_near = sum(1 for p in pts if not MX.in_box(p["lat"], p["lon"], box) and EM.ref_distance(p["lat"], p["lon"], m) <= t)
            ov = overlap(near, away)
            out[str(t)] = {"near": describe(near), "away": describe(away), "near_values": [round(x, 3) for x in sorted(near)],
                           "away_values": [round(x, 3) for x in sorted(away)], "overlap": None if ov is None else round(ov, 3), "near_outside_box": outside_near}
        rows.append(out)
    groups = {s: [r for r in rows if r["segment"] == s] for s in MOVING + STANDING}
    groups["moving"] = [r for r in rows if r["segment"] in MOVING]
    summary = {g: {str(t): summarize(rs, str(t)) for t in TOLS} for g, rs in groups.items()}
    controls = {}
    for r in (r9, r6):
        for name in sorted({c["control"] for c in r["controls"]}):
            per = []
            for c in r["controls"]:
                if c["control"] != name:
                    continue
                v = [p["value"] for p in r["maxima_by_date"][c["date"]]["B"] if MX.in_box(p["lat"], p["lon"], c["box"])]
                per.append({"date": c["date"], "box": describe(v)})
            med = [x["box"]["median"] for x in per if x["box"]]
            controls[name] = {"maps": len(per), "maps_without_peaks": [x["date"] for x in per if not x["box"]],
                              "median_box_amplitude_over_maps": round(statistics.median(med), 3) if med else None,
                              "max_box_amplitude": max((x["box"]["max"] for x in per if x["box"]), default=None), "per_map": per}
    def higher(g, t):
        s = summary[g][t]
        return s["overlap_median"] is not None and s["overlap_median"] > 0.5 and s["maps_above"] > s["maps_below"]
    seg_higher = {g: {t: higher(g, t) for t in ("3.0", "5.0")} for g in MOVING}
    above_controls = {t: all(summary["moving"][t]["near_median_over_maps"] > c["median_box_amplitude_over_maps"] for c in controls.values()) for t in ("3.0", "5.0")}
    if all(seg_higher[g][t] for g in MOVING for t in ("3.0", "5.0")) and all(above_controls.values()):
        verdict = "positive"
    elif not any(seg_higher[g][t] for g in MOVING for t in ("3.0", "5.0")):
        verdict = "negative"
    else:
        verdict = "mixed"
    res = {"generated_by": "scripts/pilot_eastern_amplitude.py", "script_sha256": A.X.digest(os.path.abspath(__file__)),
           "inputs": {k: {"path": p, "sha256": A.X.digest(p)} for k, p in (("maxima_2006", args.maxima_2006), ("maxima_1990", args.maxima_1990), ("reference", args.reference))},
           "units": "stored values are relative vorticity times 1e5, in units of 1e-5 per second; every value above 1.0",
           "summary": summary, "controls": {k: {x: v for x, v in c.items() if x != "per_map"} for k, c in controls.items()},
           "segment_near_higher": seg_higher, "moving_near_above_every_control": above_controls, "verdict": verdict,
           "maps": rows, "controls_per_map": {k: c["per_map"] for k, c in controls.items()},
           "limits": "descriptive; successive maps and peaks are not independent samples; neither outcome validates the references or establishes physical separation"}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: res[k] for k in ("summary", "controls", "segment_near_higher", "moving_near_above_every_control", "verdict")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
