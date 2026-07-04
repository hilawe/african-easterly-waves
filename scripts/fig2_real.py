#!/usr/bin/env python
"""Reproduce the real Figure-2 MCS Hovmoller from the ORIGINAL data.

Base series : v700.anom.waves (td), westward wavenumber -20..0, 2.5-10 day (Frank &
              Roundy 2006). Basepoint 10N/0E, 2 sigma -> 272 composite dates.
Shading     : MCS count anomaly from csct_africa_cs245.nc (ISCCP cloud systems <245 K).
Contours    : composite of unfiltered v700 (climo_unfiltered_v700mb.nc), 5-15N average.
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr

from aew.composites import (anomaly, composite_xt_preread, hovmoller_event_counts,
                            lag_axis)
from aew.events import composite_dates, std_threshold
from aew.plotting import hovmoller, save

BASE = "data/original"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-lat", type=float, default=10.0)
    ap.add_argument("--base-lon", type=float, default=0.0)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--csct", default="csct/csct_africa_cs245.nc",
                    help="csct file under data/original (cs245=MCS, cs190_cs245_50_100=WDCS)")
    ap.add_argument("--panel", default="a) MCS", help="panel label, e.g. 'a) MCS' or 'b) WDCS'")
    ap.add_argument("--shaded-label", default="MCS count anomaly (count - lag mean)")
    ap.add_argument("--out", default="fig2_real.png")
    ap.add_argument("--save-npz", default=None,
                    help="also write the panel arrays (shaded, contour, coordinates, "
                         "provenance numbers) for the composed validation figure")
    a = ap.parse_args()
    minLat, maxLat, minLon, maxLon, lon_scale = 5.0, 15.0, -40.0, 80.0, 4.0

    # 1) base wave series -> composite dates
    wav = xr.open_dataset(f"{BASE}/eraint/v700_waves_basebox.nc")["td"]
    s = wav.sel(lat=a.base_lat, lon=a.base_lon, method="nearest")
    base_vals = np.asarray(s.values, dtype=float)
    base_time = pd.DatetimeIndex(s["time"].values)
    ok = np.isfinite(base_vals)
    thr = std_threshold(base_vals[ok], a.sigma)
    cd = composite_dates(base_vals[ok], thr, time=base_time[ok].values)
    print(f"basepoint {float(s.lat):.1f}N/{float(s.lon):.1f}E  "
          f"std={np.nanstd(base_vals, ddof=1):.5f}  thr={thr:.5f}  n_dates={len(cd)}")

    # 2) contour: unfiltered v700, 5-15N mean, lag composite (daily matching)
    clim = xr.open_dataset(f"{BASE}/eraint/climo_unfiltered_v700mb.nc")["v700"]
    clim_band = clim.sel(lat=slice(minLat, maxLat)).mean("lat")
    if clim.lat.values[0] > clim.lat.values[-1]:  # descending lat
        clim_band = clim.sel(lat=slice(maxLat, minLat)).mean("lat")
    clim_time = pd.DatetimeIndex(clim_band["time"].values)
    cd_daily = pd.DatetimeIndex(pd.DatetimeIndex(cd.time).floor("D").unique())
    comp = composite_xt_preread(clim_band.values, clim_time.values, cd_daily.values,
                                -6, 6, 1, n_tests=0)
    clon = clim.lon.values

    # 3) shading: MCS counts about the 272 dates, lon-lag, anomaly
    cs = xr.open_dataset(f"{BASE}/{a.csct}")
    cs_time = pd.DatetimeIndex(cs["time"].values)
    cs_lon = np.asarray(cs["lon"].values, dtype=float)
    cs_lat = np.asarray(cs["lat"].values, dtype=float)
    lon_centers = np.arange(minLon, maxLon + 1e-6, lon_scale)
    lag = lag_axis(-6, 6, 1).astype(float)
    counts = hovmoller_event_counts(cd.time, cs_time.values, cs_lon, lon_centers, lag,
                                    cs_lat=cs_lat, min_lat=minLat, max_lat=maxLat)
    shaded = anomaly(counts, "anomaly")

    if a.save_npz:
        np.savez(a.save_npz, shaded=shaded, lon_centers=lon_centers, lag=lag,
                 contour=comp.values, contour_lon=clon, contour_lag=comp.lag,
                 base_lon=a.base_lon, n_dates=len(cd), thr=thr, sigma=a.sigma,
                 lon_scale=lon_scale)
        print("cached panel arrays at", a.save_npz)

    # 4) plot
    prov = (f"Number of dates: {len(cd)}\nStd. Dev. threshold: {a.sigma:g}\n"
            f"Filt. v-700 threshold: {thr:.5f} m/s\nBin degree scale: {lon_scale:g}")
    fig, ax = hovmoller(
        shaded, lon_centers, lag, contour=comp.values, contour_lon=clon,
        contour_lag=comp.lag, base_lon=a.base_lon, lon_range=(minLon, maxLon),
        title=f"{a.panel}  -  Averaged {minLat:g}N-{maxLat:g}N  (basepoint {a.base_lon:g}E)",
        shaded_label=a.shaded_label, provenance=prov,
    )
    out = save(fig, a.out)
    print("wrote", out)


if __name__ == "__main__":
    main()
