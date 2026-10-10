#!/usr/bin/env python3
"""Where the control's tracks starting in a longitude band went in the treatment, whatever
the crosswalk's pairing rule says.

A post hoc diagnostic of the eastern pilot, declared in EASTERN_PILOT_ASSESSMENT_2026-10-05.md
after the crosswalk showed the control's 30 to 40 E starts falling from 175 to 89 (1990) and
183 to 80 (2006) while only a handful of pairs were eastern recoveries. For every control
track whose first observation lies in the band, the treatment track sharing the most time
steps within 3 degrees (ties by the smaller mean separation) is its BEST PARTNER, with no
separation cutoff, so a track displaced by more than the pairing rule's 1 degree is still
found. The report gives, for the band and for its June to September subset, the
distribution of that partner's mean separation, the share of the control's life the
partner matches within 1 and within 3 degrees, where the partner starts relative to the
control, and how many partners are earlier eastern starts covering most of the control.
It answers "extended, displaced, or replaced", and nothing about physical waves.

    python3 scripts/pilot_edge_band_trace.py --a <control run dir> --b <treatment run dir> --year 1990 \\
        --band 30 40 --out <json>
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402
import pilot_track_crosswalk as P  # noqa: E402

NEAR_DEG, CLOSE_DEG, COVERED_SHARE = 1.0, 3.0, 0.8
EAST_START_DEG = 40.0


def best_partner(tr, tb, by_time):
    """The treatment track sharing the most time steps within CLOSE_DEG of this control
    track, ties by the smaller mean separation over every shared step. None when no
    treatment track observes any of the control's time steps."""
    cand = {}
    for k, tt in enumerate(tr["time"]):
        for j, kk in by_time.get(round(float(tt), 6), []):
            d = float(np.hypot(tb[j]["lat"][kk] - tr["lat"][k], tb[j]["lon"][kk] - tr["lon"][k]))
            cand.setdefault(j, []).append(d)
    if not cand:
        return None
    j, ds = min(cand.items(), key=lambda kv: (-sum(1 for d in kv[1] if d <= CLOSE_DEG), float(np.mean(kv[1])), kv[0]))
    return {"b": int(j), "shared_steps": len(ds), "mean_separation_deg": round(float(np.mean(ds)), 3),
            "steps_within_near": int(sum(1 for d in ds if d <= NEAR_DEG)), "steps_within_close": int(sum(1 for d in ds if d <= CLOSE_DEG))}


EMPTY_PCT = {"10": None, "50": None, "90": None}


def _summary(rows, tb):
    if not rows:
        # A STABLE EMPTY SUMMARY, every key present with zero or None, so a band with no
        # control or no partner never fails after its report is published
        return {"controls": 0, "partner_mean_separation_deg_percentiles": dict(EMPTY_PCT),
                "share_of_life_matched_within_near_percentiles": dict(EMPTY_PCT), "share_of_life_matched_within_close_percentiles": dict(EMPTY_PCT),
                "controls_covered_within_close": 0, "controls_covered_within_near": 0,
                "partner_start_lon_minus_control_percentiles": dict(EMPTY_PCT), "partners_starting_east_of_edge": 0,
                "partners_starting_earlier": 0, "partners_starting_same_time": 0, "partners_starting_later": 0,
                "partners_east_and_earlier_covering_within_close": 0, "partners_east_and_earlier_covering_within_near": 0,
                "partner_life_minus_control_life_percentiles": dict(EMPTY_PCT)}
    life = np.array([r["life"] for r in rows], float)
    sep = np.array([r["partner"]["mean_separation_deg"] for r in rows])
    near = np.array([r["partner"]["steps_within_near"] for r in rows]) / life
    close = np.array([r["partner"]["steps_within_close"] for r in rows]) / life
    pstart = np.array([tb[r["partner"]["b"]]["lon"][0] for r in rows])
    pshift = np.array([tb[r["partner"]["b"]]["time"][0] - r["start_time"] for r in rows])
    plife = np.array([tb[r["partner"]["b"]]["time"].size for r in rows], float)
    pct = lambda v: {"10": round(float(np.percentile(v, 10)), 3), "50": round(float(np.percentile(v, 50)), 3), "90": round(float(np.percentile(v, 90)), 3)}
    east_earlier = (pstart > EAST_START_DEG) & (pshift < 0)
    return {"controls": len(rows),
            "partner_mean_separation_deg_percentiles": pct(sep),
            "share_of_life_matched_within_near_percentiles": pct(near),
            "share_of_life_matched_within_close_percentiles": pct(close),
            "controls_covered_within_close": int(np.sum(close >= COVERED_SHARE)),
            "controls_covered_within_near": int(np.sum(near >= COVERED_SHARE)),
            "partner_start_lon_minus_control_percentiles": pct(pstart - np.array([r["start_lon"] for r in rows])),
            "partners_starting_east_of_edge": int(np.sum(pstart > EAST_START_DEG)),
            "partners_starting_earlier": int(np.sum(pshift < 0)), "partners_starting_same_time": int(np.sum(pshift == 0)), "partners_starting_later": int(np.sum(pshift > 0)),
            "partners_east_and_earlier_covering_within_close": int(np.sum(east_earlier & (close >= COVERED_SHARE))),
            "partners_east_and_earlier_covering_within_near": int(np.sum(east_earlier & (near >= COVERED_SHARE))),
            "partner_life_minus_control_life_percentiles": pct(plife - life)}


def trace(a, b, year, band):
    ta, tb = a["tracks"], b["tracks"]
    by_time = {}
    for j, t in enumerate(tb):
        for k, tt in enumerate(t["time"]):
            by_time.setdefault(round(float(tt), 6), []).append((j, k))
    rows, unmatched = [], []
    for i, tr in enumerate(ta):
        if not band[0] <= float(tr["lon"][0]) < band[1]:
            continue
        partner = best_partner(tr, tb, by_time)
        month = P.S.date_of(float(tr["time"][0])).month
        if partner is None:
            unmatched.append(i)
            continue
        rows.append({"a": i, "start_lon": float(tr["lon"][0]), "start_lat": float(tr["lat"][0]), "start_time": float(tr["time"][0]),
                     "start_month": month, "life": int(tr["time"].size), "partner": partner})
    season = [r for r in rows if 6 <= r["start_month"] <= 9]
    return {"generated_by": "scripts/pilot_edge_band_trace.py", "script_sha256": X.digest(os.path.abspath(__file__)), "year": year,
            "band_lon": list(band), "east_edge_lon": EAST_START_DEG,
            "rule": {"near_deg": NEAR_DEG, "close_deg": CLOSE_DEG, "covered_share": COVERED_SHARE,
                     "partner": "the treatment track sharing the most time steps within close_deg, ties by the smaller mean separation, no separation cutoff"},
            "runs": {"a": {"dir": a["dir"], "tracks_sha256": a["tracks_sha256"]}, "b": {"dir": b["dir"], "tracks_sha256": b["tracks_sha256"]}},
            "controls_in_band": len(rows) + len(unmatched), "controls_with_no_treatment_track_at_any_of_their_times": unmatched,
            "all_months": _summary(rows, tb), "june_to_september": _summary(season, tb), "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True, help="the control run directory")
    ap.add_argument("--b", required=True, help="the treatment run directory")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--band", type=float, nargs=2, default=(30.0, 40.0), help="the control start-longitude band, lower inclusive")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    a, b = P.load_run(args.a, year=args.year), P.load_run(args.b, year=args.year)
    a["dir"], b["dir"] = args.a, args.b
    out = trace(a, b, args.year, tuple(args.band))
    try:
        X.publish_json(args.out, out, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {args.out} exists and artifacts are never overwritten")
    s, j = out["all_months"], out["june_to_september"]
    print(f"{out['controls_in_band']} control tracks start in {args.band}; best partner covers >= {COVERED_SHARE:.0%} within {CLOSE_DEG:.0f} deg for "
          f"{s['controls_covered_within_close']} (June to September {j['controls_covered_within_close']} of {j['controls']}); wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
