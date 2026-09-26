#!/usr/bin/env python3
"""The season-selection rule, executed and printed, never applied by hand.

THE RULE, approved 2026-09-25. From the published record's June to September Africa
track counts per year over the declared years, excluding the pilot year and any year
whose file is absent, take the year with the HIGHEST count and the year with the LOWEST,
ties broken toward the EARLIER year. A track belongs to a season by its first
observation's calendar month, the same membership the season metrics use. The script
prints the whole table with the two rows marked, digests every file it read, and
publishes the artifact exclusively. The choice is whatever it prints.
"""

import argparse
import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import exact_tracks as X  # noqa: E402
import season_metrics as M  # noqa: E402
from aew.v1port import validate as V  # noqa: E402


def choose(counts, exclude):
    """(highest_year, lowest_year) from {year: count}, ties to the earlier year."""
    eligible = {y: c for y, c in counts.items() if y not in exclude}
    if not eligible:
        return None, None
    highest = min(y for y, c in eligible.items() if c == max(eligible.values()))
    lowest = min(y for y, c in eligible.items() if c == min(eligible.values()))
    return highest, lowest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record-dir", default="data/aewc")
    ap.add_argument("--years", default="1983-2007")
    ap.add_argument("--exclude", default="1990", help="comma-separated years excluded, the pilot year")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    y0, y1 = (int(x) for x in args.years.split("-"))
    exclude = {int(x) for x in args.exclude.split(",") if x.strip()}
    counts, digests, absent = {}, {}, []
    for y in range(y0, y1 + 1):
        path = os.path.join(args.record_dir, f"ERA-Int_ew_700hPa_{y}_AFR.nc")
        if not os.path.exists(path):
            absent.append(y)
            continue
        with open(path, "rb") as fh:
            blob = fh.read()
        digests[str(y)] = hashlib.sha256(blob).hexdigest()
        tracks = M.read_record_bytes(blob)     # parsed from the hashed bytes, not the path
        counts[y] = sum(1 for t in tracks if M.in_season(t, y))
    highest, lowest = choose(counts, exclude)
    payload = {"generated_by": "scripts/select_seasons.py",
               "rule": ("highest and lowest June to September Africa track count by first "
                        "observation over the declared years, excluding the pilot year and "
                        "absent years, ties to the earlier year"),
               "script_sha256": X.digest(__file__),
               "git_head_at_launch": X.repository_head(
                   os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
               "launched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "years_declared": [y0, y1], "excluded": sorted(exclude), "absent": absent,
               "counts": {str(y): counts[y] for y in sorted(counts)},
               "record_sha256": digests,
               "selected": {"highest": highest, "lowest": lowest}}
    try:
        X.publish_json(args.out, payload, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {args.out} exists and artifacts are never overwritten")
    for y in sorted(counts):
        mark = "  <- highest" if y == highest else ("  <- lowest" if y == lowest else
                                                    ("  (excluded)" if y in exclude else ""))
        print(f"  {y}  {counts[y]:4d}{mark}")
    print(f"selected: highest {highest}, lowest {lowest}; absent {absent}; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
