#!/usr/bin/env python
"""Download one month of GridSat-B1 Tb cropped to the AEW box (S3 + local crop)."""
import argparse
from aew.data.gridsat import download_month

ap = argparse.ArgumentParser()
ap.add_argument("--year", type=int, default=2000)
ap.add_argument("--month", type=int, default=7)
ap.add_argument("--out-dir", default="data/gridsat")
a = ap.parse_args()
paths = download_month(a.year, a.month, out_dir=a.out_dir)
print(f"downloaded {len(paths)} files for {a.year}-{a.month:02d}")
