#!/usr/bin/env python
"""Validate the wave-extraction engine against the published targets.

Reproduces the basepoint composite-date selection from daily-mean ERA5 v700:
  1. band-pass filter the CONTINUOUS daily v700 record (Lanczos 2-10 day)
  2. take the basepoint grid cell (default 10N, 0E) and subset JAS
  3. threshold at n_sigma * std(JAS filtered series)
  4. composite_dates = local maxima above threshold

Published target (West Africa 10N/0E, 2 sigma): 272 dates, filtered-v700 threshold
3.26136 m/s. NOTE: that target is from ERA-Interim; ERA5 will give close but not
identical numbers (different reanalysis). A count near ~272 and threshold near ~3.3
confirms the engine; exact match is not expected. See docs/VALIDATION_TARGETS.md.

Usage:
    python scripts/validate_basepoint.py --file data/era5/era5_v700_1984-2007_daily.nc
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr

from aew.events import composite_dates, std_threshold
from aew.filtering import lanczos_bandpass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="daily-mean v700 NetCDF")
    ap.add_argument("--var", default="v700")
    ap.add_argument("--base-lat", type=float, default=10.0)
    ap.add_argument("--base-lon", type=float, default=0.0)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--months", default="7,8,9")
    args = ap.parse_args()

    ds = xr.open_dataset(args.file)
    da = ds[args.var].squeeze(drop=True)  # drop singleton pressure_level / number
    latname = "latitude" if "latitude" in da.coords else "lat"
    lonname = "longitude" if "longitude" in da.coords else "lon"

    # nearest basepoint grid cell, full continuous time series
    series = da.sel({latname: args.base_lat, lonname: args.base_lon}, method="nearest")
    time = pd.DatetimeIndex(series["time"].values)
    values = np.asarray(series.values, dtype=float)

    # 1) band-pass filter the continuous record (axis 0 = time)
    filt = lanczos_bandpass(values, period_low=2, period_high=10, obs_per_day=1, axis=0)

    # 2) subset JAS
    months = [int(m) for m in args.months.split(",")]
    jas = np.isin(time.month, months) & np.isfinite(filt)
    base_series = filt[jas]
    base_time = time[jas]

    # 3) threshold, 4) composite dates
    thr = std_threshold(base_series, args.sigma)
    cd = composite_dates(base_series, thr, time=base_time.values)

    print(f"basepoint: {args.base_lat}N / {args.base_lon}E   sigma: {args.sigma}")
    print(f"JAS days (finite, filtered): {base_series.size}")
    print(f"std(filtered v700) = {np.std(base_series, ddof=1):.5f} m/s")
    print(f"threshold ({args.sigma} sigma) = {thr:.5f} m/s   (ERA-Interim target: 3.26136)")
    print(f"number of composite dates = {len(cd)}   (ERA-Interim target: 272)")
    ds.close()


if __name__ == "__main__":
    main()
