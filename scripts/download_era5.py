#!/usr/bin/env python
"""Download daily-mean ERA5 winds for the AEW analysis.

Full years are downloaded (not JAS-only) so the band-pass filter can run on the
continuous record. Usage:

    python scripts/download_era5.py --var v700 --years 2000          # one-year test
    python scripts/download_era5.py --var v700 --years 1984-2007     # full record
"""

import argparse

from aew.data.era5 import DEFAULT_AREA, build_daily_series


def parse_years(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", default="v700", help="v700, u700, v600, u600, v850, u850")
    ap.add_argument("--years", default="2000", help="e.g. 2000 or 1984-2007 or 1984,1985")
    ap.add_argument("--grid", type=float, default=0.5)
    ap.add_argument("--out-dir", default="data/era5")
    args = ap.parse_args()
    years = parse_years(args.years)
    print(f"Downloading ERA5 {args.var} for {years[0]}-{years[-1]} at {args.grid} deg")
    print(f"Area (N,W,S,E): {DEFAULT_AREA}")
    path = build_daily_series(
        args.var, years, grid=(args.grid, args.grid), out_dir=args.out_dir
    )
    print("Wrote:", path)


if __name__ == "__main__":
    main()
