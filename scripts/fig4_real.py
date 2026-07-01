#!/usr/bin/env python
"""Reproduce a real Figure-4 basepoint composite MAP from the ORIGINAL data.

Same 272 composite dates as Fig 2. Shaded: MCS count anomaly on a lon-lat map at a chosen
lag (count at that lag minus the lag-window mean per cell). Contours: composite of
unfiltered v700 at that lag (composite_xy). Rendered on a cartopy map (basepoint_map).
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr

from aew.composites import composite_xy_preread, map_event_counts
from aew.events import composite_dates, std_threshold
from aew.plotting import basepoint_map, save

BASE = "data/original"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-lat", type=float, default=10.0)
    ap.add_argument("--base-lon", type=float, default=0.0)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--lag", type=float, default=0.0)
    ap.add_argument("--csct", default="csct/csct_africa_cs245.nc")
    ap.add_argument("--panel", default="MCS")
    ap.add_argument("--out", default="fig4_real.png")
    a = ap.parse_args()
    west, east, south, north = -40.0, 60.0, -5.0, 30.0

    # 1) composite dates (same engine as Fig 2)
    wav = xr.open_dataset(f"{BASE}/eraint/v700_waves_basebox.nc")["td"]
    s = wav.sel(lat=a.base_lat, lon=a.base_lon, method="nearest")
    v = np.asarray(s.values, dtype=float)
    t = pd.DatetimeIndex(s["time"].values)
    ok = np.isfinite(v)
    thr = std_threshold(v[ok], a.sigma)
    cd = composite_dates(v[ok], thr, time=t[ok].values)
    print(f"n_dates={len(cd)}  thr={thr:.5f}")

    # 2) contour: composite_xy of unfiltered v700 at the chosen lag
    clim = xr.open_dataset(f"{BASE}/eraint/climo_unfiltered_v700mb.nc")["v700"]
    clim_time = pd.DatetimeIndex(clim["time"].values)
    cd_daily = pd.DatetimeIndex(pd.DatetimeIndex(cd.time).floor("D").unique())
    cxy = composite_xy_preread(clim.values, clim_time.values, cd_daily.values,
                               lags=[a.lag], n_tests=0)
    clat, clon = clim.lat.values, clim.lon.values

    # 3) shading: MCS count anomaly map = count(lag) - mean over lags, per cell
    cs = xr.open_dataset(f"{BASE}/{a.csct}")
    cs_time = pd.DatetimeIndex(cs["time"].values).values
    cs_lon = np.asarray(cs["lon"].values, dtype=float)
    cs_lat = np.asarray(cs["lat"].values, dtype=float)
    lon_c = np.arange(west, east + 1e-6, 2.0)
    lat_c = np.arange(south, north + 1e-6, 2.0)
    lags = np.arange(-6.0, 7.0)
    stack = np.stack([
        map_event_counts(cd.time, cs_time, cs_lon, cs_lat, lon_c, lat_c,
                         lag=L, half_window=0.5, statistic="count")
        for L in lags
    ])
    shaded = stack[np.argmin(np.abs(lags - a.lag))] - stack.mean(axis=0)

    # 4) cartopy map
    prov = f"n dates: {len(cd)}   lag: {a.lag:g} d   sigma: {a.sigma:g}"
    fig, ax = basepoint_map(
        shaded, lon_c, lat_c, contour=cxy.values[0], contour_lon=clon, contour_lat=clat,
        base_lon=a.base_lon, base_lat=a.base_lat, extent=(west, east, south, north),
        title=f"{a.panel} count anomaly + v700, lag {a.lag:g} d (basepoint "
              f"{a.base_lat:g}N/{a.base_lon:g}E)",
        shaded_label=f"{a.panel} count anomaly", provenance=prov,
    )
    out = save(fig, a.out)
    print("wrote", out)


if __name__ == "__main__":
    main()
