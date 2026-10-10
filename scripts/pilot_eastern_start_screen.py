#!/usr/bin/env python3
"""A bounded, read-only screen of one retained replay's finished tracks, selecting those
whose first observation is in June through September, between 5 N and 20 N inclusive,
and strictly east of a declared longitude (40 E), with how many of them reach a declared
western mark (20 E) or farther west and the westernmost stored longitude among them. It
reproduces the external review's screen of 2026-10-07 on the consistently prepared 60 E
runs.

WHAT IT IS AND IS NOT. The selection follows the earlier tropical summer audit
population and applies no Africa-polygon filter, so it includes starts outside Africa.
It reads each finished track's stored arrays (`time[0]`, `lat[0]`, `lon[0]` for the
selection, `min(lon)` over the whole stored history for the reach). It describes the
stored histories under the pair the replay applied. It is not an acceptance test, the
western mark is not a target, and a count of starts says nothing about whether the
physical disturbances continue, whether later fragments belong to them, or what the
extension does for starts west of the declared longitude.

    python3 scripts/pilot_eastern_start_screen.py --replay <replay json.gz> --out <fresh json> [--start-east-of 40 --reach 20 --lat 5 20]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402
import pilot_margin_compare as MC  # noqa: E402
import season_metrics as S  # noqa: E402


def screen(finished, *, start_east_of, reach, lat_range, region_of=None):
    """The selection and, when a region function is given, each history's first entry into
    Africa (the first stored observation whose region is AFR), an attribute kept apart
    from the first detection."""
    rows = []
    for i, t in enumerate(finished):
        t0, lat0, lon0 = float(t["time"][0]), float(t["lat"][0]), float(t["lon"][0])
        day = S.date_of(t0)
        if 6 <= day.month <= 9 and lat_range[0] <= lat0 <= lat_range[1] and lon0 > start_east_of:
            westernmost = float(min(float(x) for x in t["lon"]))
            row = {"index": i, "first_detection": t.get("birth"), "first": {"time": t0, "date": str(day), "lat": lat0, "lon": lon0},
                   "observations": len(t["lon"]), "westernmost_lon": westernmost, "last_lon": float(t["lon"][-1]), "reaches": bool(westernmost <= reach)}
            if region_of is not None:
                hits = [k for k, (lo, la) in enumerate(zip(t["lon"], t["lat"])) if region_of(float(lo), float(la)) == "AFR"]
                row["first_entry_into_africa"] = None if not hits else {"index": hits[0], "time": float(t["time"][hits[0]]), "date": str(S.date_of(float(t["time"][hits[0]]))), "lat": float(t["lat"][hits[0]]), "lon": float(t["lon"][hits[0]])}
            rows.append(row)
    return {"selection": {"first_observation_months": [6, 9], "first_latitude_range": list(lat_range), "first_longitude_strictly_east_of": start_east_of, "africa_polygon_filter": False},
            "reach_mark_lon": reach, "selected": len(rows), "reaching": int(sum(r["reaches"] for r in rows)),
            "westernmost_lon_among_selected": min((r["westernmost_lon"] for r in rows), default=None), "indices": [r["index"] for r in rows], "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--replay", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--start-east-of", type=float, default=40.0)
    ap.add_argument("--reach", type=float, default=20.0)
    ap.add_argument("--lat", nargs=2, type=float, default=(5.0, 20.0))
    ap.add_argument("--regions-dir", help="the archive's region polygons. When given, each selected history carries its first entry into Africa")
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    rec, sha = MC.load_gz(args.replay)
    region_of, region_digests = None, None
    if args.regions_dir:
        regions, region_digests = S.load_regions(args.regions_dir)
        region_of = lambda lon, lat: S.source_region({"lon": [lon], "lat": [lat]}, regions)
    result = screen(rec["finished"], start_east_of=args.start_east_of, reach=args.reach, lat_range=tuple(args.lat), region_of=region_of)
    out = {"generated_by": "scripts/pilot_eastern_start_screen.py", "script_sha256": X.digest(os.path.abspath(__file__)), "year": rec["year"],
           "replay": {"path": args.replay, "sha256": sha, "preparation": rec["preparation"], "configuration": rec.get("configuration"), "thresholds": rec["thresholds"], "runs": rec["runs"]},
           "regions": None if region_digests is None else {"dir": args.regions_dir, "sha256": region_digests},
           "reading": "a description of the stored histories under the replay's pair, not an acceptance test. The reach mark is not a target.", **result}
    X.publish_json(args.out, out, exclusive=True)
    print(f"{rec['year']}: {result['selected']} finished histories start in June to September between {args.lat[0]} and {args.lat[1]} N strictly east of {args.start_east_of} E, "
          f"{result['reaching']} reach {args.reach} E or farther west, westernmost stored longitude {result['westernmost_lon_among_selected']}. Indices {result['indices']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
