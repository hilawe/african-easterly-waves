#!/usr/bin/env python3
"""The first-pass sample of the 60 E candidate evaluation, chosen by the frozen rules of
EASTERN_60E_EVALUATION_2026-10-07.md section 2 and never by inspection.

From one season's replay and its screen record (the eastern population, first detection
in June through September between 5 and 20 N and strictly east of 40 E), the eastern
cases are the n histories with the most stored observations, ties to the earlier first
detection. The comparison case is the finished history first detected in the same months
and latitude band between 20 and 40 E whose number of stored observations is the lower
median of that population, ties to the earlier first detection. The event history, when
a point is given, is the finished history with an observation within 5 degrees and 24
hours of it, the nearest by distance then by time, or none, with the nearest observation
of all recorded either way. The record binds the replay and the screen by digest.

    python3 scripts/pilot_case_sample.py --replay <replay json.gz> --screen <screen json> --out <fresh json> [--event 10.5 29.0 38960.0]
"""
import argparse
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402
import pilot_margin_compare as MC  # noqa: E402
import season_metrics as S  # noqa: E402


def in_months_and_band(track, lat_range):
    day = S.date_of(float(track["time"][0]))
    return 6 <= day.month <= 9 and lat_range[0] <= float(track["lat"][0]) <= lat_range[1]


def sample(finished, eastern, *, start_east_of, lat_range, n_eastern=3, comparison_west_of=20.0):
    order = sorted(eastern, key=lambda i: (-len(finished[i]["lon"]), float(finished[i]["time"][0]), i))
    population = [i for i, t in enumerate(finished) if in_months_and_band(t, lat_range) and comparison_west_of <= float(t["lon"][0]) <= start_east_of]
    counts = sorted(len(finished[i]["lon"]) for i in population)
    median = counts[(len(counts) - 1) // 2] if counts else None
    candidates = [i for i in population if len(finished[i]["lon"]) == median]
    comparison = min(candidates, key=lambda i: (float(finished[i]["time"][0]), i)) if candidates else None
    return {"eastern": order[:n_eastern], "eastern_population": sorted(eastern), "comparison_population": population,
            "comparison_observations_median": median, "comparison": comparison}


def event_history(finished, *, lat, lon, day, max_deg=5.0, max_hours=24.0):
    best, best_any = None, None
    for i, t in enumerate(finished):
        for la, lo, ti in zip(t["lat"], t["lon"], t["time"]):
            d, h = math.hypot(float(la) - lat, float(lo) - lon), abs(float(ti) - day) * 24.0
            if best_any is None or (d, h) < (best_any[1], best_any[2]):
                best_any = (i, d, h)
            if d <= max_deg and h <= max_hours and (best is None or (d, h) < (best[1], best[2])):
                best = (i, d, h)
    if best is not None:
        return {"index": best[0], "distance_deg": best[1], "hours": best[2]}
    return {"index": None, "nearest": None if best_any is None else {"index": best_any[0], "distance_deg": best_any[1], "hours": best_any[2]}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--replay", "--screen", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--event", nargs=3, type=float, metavar=("LAT", "LON", "DAYS_SINCE_1900"))
    ap.add_argument("--n-eastern", type=int, default=3)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    rec, sha = MC.load_gz(args.replay)
    with open(args.screen, "rb") as fh:
        blob = fh.read()
    screen = json.loads(blob)
    if screen.get("replay", {}).get("sha256") != sha:
        raise SystemExit("REFUSED: the screen record was not made from this replay")
    sel = screen["selection"]
    chosen = sample(rec["finished"], screen["indices"], start_east_of=float(sel["first_longitude_strictly_east_of"]), lat_range=tuple(sel["first_latitude_range"]), n_eastern=args.n_eastern)
    event = None
    if args.event:
        lat, lon, day = args.event
        event = {"point": {"lat": lat, "lon": lon, "days_since_1900": day}, **event_history(rec["finished"], lat=lat, lon=lon, day=day)}

    def describe(i):
        t = rec["finished"][i]
        return {"index": i, "first_detection": t["birth"], "first": {"time": float(t["time"][0]), "date": str(S.date_of(float(t["time"][0]))), "lat": float(t["lat"][0]), "lon": float(t["lon"][0])},
                "observations": len(t["lon"]), "westernmost_lon": float(min(float(x) for x in t["lon"])), "last": {"time": float(t["time"][-1]), "lat": float(t["lat"][-1]), "lon": float(t["lon"][-1])}}
    out = {"generated_by": "scripts/pilot_case_sample.py", "script_sha256": X.digest(os.path.abspath(__file__)), "year": rec["year"],
           "replay": {"path": args.replay, "sha256": sha}, "screen": {"path": args.screen, "sha256": hashlib.sha256(blob).hexdigest()},
           "rules": {"eastern": f"the {args.n_eastern} eastern histories with the most stored observations, ties to the earlier first detection",
                     "comparison": "the history first detected in the same months and band between 20 E and the eastern mark with the lower-median number of stored observations, ties to the earlier first detection",
                     "event": "the history with an observation within 5 degrees and 24 hours of the point, nearest by distance then by time"},
           **chosen, "eastern_cases": [describe(i) for i in chosen["eastern"]], "comparison_case": None if chosen["comparison"] is None else describe(chosen["comparison"]),
           "event": event, "event_case": None if not event or event.get("index") is None else describe(event["index"])}
    X.publish_json(args.out, out, exclusive=True)
    print(f"{rec['year']}: eastern population {len(chosen['eastern_population'])}, eastern cases {chosen['eastern']}, comparison population {len(chosen['comparison_population'])} "
          f"(median observations {chosen['comparison_observations_median']}), comparison case {chosen['comparison']}, event {event}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
