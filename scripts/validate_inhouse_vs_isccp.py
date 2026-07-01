#!/usr/bin/env python
"""Validate the in-house GridSat tracker against the ORIGINAL ISCCP CS (and Huang).

Three independent infrared MCS products for the same month/region/size-cut:
  1. Original ISCCP CS (>=90 km)   csct_africa_cs245_csize-ge-90.nc   [the reference]
  2. In-house GridSat tracker      (245/220 K, overlap+projection, 90 km)
  3. Huang et al. 2018             (233 K, overlap+Kalman, 90 km)
Bins each onto a common lon-lat grid, reports pairwise spatial pattern correlations,
and renders three count maps. The in-house product matching ISCCP's spatial pattern is
the validation; absolute counts differ by the known factor 2-3 across trackers.
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr

from aew.binning import bin_sum
from aew.data.huang import load_huang_month
from aew.data.pyflextrkr import from_pyflextrkr
from aew.tracks import Tracks


def count_map(lon, lat, lon_c, lat_c):
    g, _ = bin_sum(lon_c, lat_c, lon, lat, z=np.ones(lon.size), variant="fixed")
    return g


def load_isccp(path, year, month):
    ds = xr.open_dataset(path)
    t = pd.DatetimeIndex(ds["time"].values)
    lat = np.asarray(ds["lat"].values, dtype=float)
    lon = np.asarray(ds["lon"].values, dtype=float)
    sel = (t.year == year) & (t.month == month)
    return Tracks(time=t[sel].values, lat=lat[sel], lon=lon[sel], variables={})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2000)
    ap.add_argument("--month", type=int, default=7)
    ap.add_argument("--min-radius", type=float, default=90.0)
    ap.add_argument("--out", default="validate_inhouse_vs_isccp.png")
    a = ap.parse_args()
    west, east, south, north = -40.0, 40.0, 0.0, 25.0
    lon_c = np.arange(west, east + 1e-6, 2.0)
    lat_c = np.arange(south, north + 1e-6, 2.0)

    isccp = load_isccp("data/original/csct/csct_africa_cs245_csize-ge-90.nc",
                       a.year, a.month).filter_region(south, north, west, east)
    gs = from_pyflextrkr("data/gridsat/tracks_2000_07.nc",
                         min_radius_km=a.min_radius).filter_region(south, north, west, east)
    hu = load_huang_month(
        f"data/huang/MCS_record_1985_02_2008-11/MCS_record_{a.year}-{a.month:02d}.txt",
        min_radius_km=a.min_radius).filter_region(south, north, west, east)

    products = [("ISCCP CS (original, >=90 km)", isccp),
                ("In-house GridSat (245/220 K)", gs),
                ("Huang 2018 (233 K)", hu)]
    maps = {}
    for name, tr in products:
        maps[name] = count_map(tr.lon, tr.lat, lon_c, lat_c)
        print(f"{name:32s}: {len(tr):6d} system-time pts")

    ref = maps[products[0][0]]
    print(f"\nspatial pattern correlation vs ISCCP CS reference:")
    for name, _ in products[1:]:
        r = np.corrcoef(ref.ravel(), maps[name].ravel())[0, 1]
        print(f"  {name:32s}: r = {r:.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), subplot_kw={"projection": proj})
    vmax = max(m.max() for m in maps.values())
    for ax, (name, _) in zip(axes, products):
        pc = ax.pcolormesh(lon_c, lat_c, maps[name], cmap="YlOrRd", vmin=0, vmax=vmax,
                           transform=proj, shading="auto")
        ax.coastlines(resolution="110m", linewidth=0.5)
        ax.set_extent([west, east, south, north], crs=proj)
        ax.set_xticks(np.arange(west, east + 1, 20), crs=proj)
        ax.set_yticks(np.arange(south, north + 1, 10), crs=proj)
        ax.tick_params(labelsize=8)
        if name != products[0][0]:
            r = np.corrcoef(ref.ravel(), maps[name].ravel())[0, 1]
            name = f"{name}   (r={r:.2f} vs ISCCP)"
        ax.set_title(name, fontsize=10)
        fig.colorbar(pc, ax=ax, shrink=0.85, label="count")
    fig.suptitle(f"In-house tracker validation vs original ISCCP CS  "
                 f"({a.year}-{a.month:02d}, >={a.min_radius:.0f} km)", fontsize=11)
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
