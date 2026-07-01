#!/usr/bin/env python
"""Download the ERA5 trough-environment fields (700 hPa RH, TCWV) for the AEW domain.

June-September, 6-hourly synoptic hours, 0.5 deg, on the analysis domain. These are the
full-coverage replacements for the sparse SSM/I trough-mean TCWV in the developing vs
non-developing composite (700 hPa relative humidity is the primary variable, total column
water vapour the cross-check). One CDS request per year per variable; each request skips
if its output file already exists, so the script is resumable after a queue timeout.
"""

import argparse
import time

from aew.data.era5 import download_year_6hourly_region


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2000-2004")
    ap.add_argument("--vars", default="r700,tcwv")
    a = ap.parse_args()
    lo, hi = a.years.split("-") if "-" in a.years else (a.years, a.years)
    years = list(range(int(lo), int(hi) + 1))
    var_keys = a.vars.split(",")

    for var_key in var_keys:
        for year in years:
            t0 = time.time()
            print(f"requesting {var_key} {year} ...", flush=True)
            path = download_year_6hourly_region(year, var_key)
            print(f"  -> {path} ({time.time() - t0:.0f} s)", flush=True)
    print("all done")


if __name__ == "__main__":
    main()
