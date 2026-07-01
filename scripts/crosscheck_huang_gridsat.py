#!/usr/bin/env python
"""Independent cross-check: in-house GridSat tracker vs Huang et al. (2018) MCS dataset.

Both are infrared-tracked MCS products. This bins each onto a common longitude-latitude
grid over the Africa box for the same month, at a common >=90 km equivalent-radius cut,
and renders side-by-side count maps plus summary statistics. Differences of a factor of
~2-3 in absolute counts are expected between trackers (MCSMIP), so the comparison is
about spatial pattern agreement, not identical numbers.
"""

import argparse

import numpy as np

from aew.binning import bin_sum
from aew.data.huang import load_huang_month
from aew.data.pyflextrkr import from_pyflextrkr


def count_map(lon, lat, lon_centers, lat_centers):
    gbin, _ = bin_sum(lon_centers, lat_centers, lon, lat,
                      z=np.ones(lon.size), variant="fixed")
    return gbin  # (nlat, nlon)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gridsat", default="data/gridsat/tracks_2000_07.nc")
    ap.add_argument("--huang", default="data/huang/MCS_record_1985_02_2008-11/MCS_record_2000-07.txt")
    ap.add_argument("--min-radius", type=float, default=90.0)
    ap.add_argument("--out", default="crosscheck_huang_gridsat.png")
    a = ap.parse_args()

    west, east, south, north = -40.0, 40.0, 0.0, 25.0
    lon_centers = np.arange(west, east + 0.01, 2.0)
    lat_centers = np.arange(south, north + 0.01, 2.0)

    # in-house GridSat tracker (already 90 km cut at tracking, re-applied to be safe)
    gs = from_pyflextrkr(a.gridsat, min_radius_km=a.min_radius)
    gs = gs.filter_region(min_lat=south, max_lat=north, min_lon=west, max_lon=east)

    # Huang
    hu = load_huang_month(a.huang, min_radius_km=a.min_radius)
    hu = hu.filter_region(min_lat=south, max_lat=north, min_lon=west, max_lon=east)

    gs_map = count_map(gs.lon, gs.lat, lon_centers, lat_centers)
    hu_map = count_map(hu.lon, hu.lat, lon_centers, lat_centers)

    print("=== July 2000, Africa box, >=%.0f km radius ===" % a.min_radius)
    for name, tr in [("GridSat (in-house)", gs), ("Huang 2018", hu)]:
        ntrk = len(set(tr.variables["track_id"]))
        rmed = float(np.median(tr.variables["radius_km"]))
        print(f"{name:22s}: {len(tr):5d} system-time pts | {ntrk:4d} tracks | "
              f"median radius {rmed:5.1f} km")

    # spatial pattern agreement (pattern correlation of the two count maps)
    a1, a2 = gs_map.ravel(), hu_map.ravel()
    r = np.corrcoef(a1, a2)[0, 1]
    print(f"spatial pattern correlation of count maps: r = {r:.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(2, 1, figsize=(9, 7),
                             subplot_kw={"projection": proj})
    vmax = max(gs_map.max(), hu_map.max())
    for ax, m, title in [(axes[0], gs_map, "In-house GridSat tracker (245/220 K, overlap+projection)"),
                         (axes[1], hu_map, "Huang et al. 2018 (233 K, overlap+Kalman)")]:
        pc = ax.pcolormesh(lon_centers, lat_centers, m, cmap="YlOrRd",
                           vmin=0, vmax=vmax, transform=proj, shading="auto")
        ax.coastlines(resolution="110m", linewidth=0.5)
        ax.set_extent([west, east, south, north], crs=proj)
        ax.set_xticks(np.arange(west, east + 1, 20), crs=proj)
        ax.set_yticks(np.arange(south, north + 1, 10), crs=proj)
        ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=10)
        fig.colorbar(pc, ax=ax, shrink=0.8, label="MCS count (Jul 2000)")
    fig.suptitle(f"Independent cross-check, >={a.min_radius:.0f} km radius   "
                 f"(pattern r={r:.2f})", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
