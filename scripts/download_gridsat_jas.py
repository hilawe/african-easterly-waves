#!/usr/bin/env python
"""Download GridSat-B1 Tb for JAS of multiple years over the AEW box (S3 + crop)."""
import argparse
from aew.data.gridsat import download_month

ap = argparse.ArgumentParser()
ap.add_argument("--years", default="1984-2007")
ap.add_argument("--months", default="7,8,9")
ap.add_argument("--out-dir", default="data/gridsat_jas")
a = ap.parse_args()
if "-" in a.years:
    lo, hi = a.years.split("-"); years = range(int(lo), int(hi) + 1)
else:
    years = [int(y) for y in a.years.split(",")]
months = [int(m) for m in a.months.split(",")]
total = 0
for y in years:
    for m in months:
        paths = download_month(y, m, out_dir=a.out_dir)
        total += len(paths)
        print(f"{y}-{m:02d}: {len(paths)} files (cumulative {total})", flush=True)
print("DONE", total)
