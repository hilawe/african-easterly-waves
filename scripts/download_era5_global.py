#!/usr/bin/env python
"""Download global-longitude 6-hourly ERA5 v700 (tropical band) for wk_bandpass."""
import argparse
from aew.data.era5 import build_6hourly_global

ap = argparse.ArgumentParser()
ap.add_argument("--years", default="2000-2004")
a = ap.parse_args()
lo, hi = (a.years.split("-") + [a.years])[:2] if "-" in a.years else (a.years, a.years)
years = list(range(int(lo), int(hi) + 1)) if "-" in a.years else [int(x) for x in a.years.split(",")]
print("downloading global 6-hourly v700 for", years)
p = build_6hourly_global("v700", years)
print("wrote", p)
