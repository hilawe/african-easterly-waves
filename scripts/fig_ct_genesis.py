#!/usr/bin/env python
"""CT family genesis density over Africa, stratified by depth and longevity.

Where AEW-season convective-system families originate (genesis), all families vs the deep
(<200 K core) and long-lived (cslife>=8, ~24 h) subsets. Uses the ISCCP CT family data.
"""

import argparse

import numpy as np

from aew.binning import bin_sum
from aew.data.ct import from_ct_genesis

WEST, EAST, SOUTH, NORTH = -40.0, 50.0, 0.0, 30.0
LON_C = np.arange(WEST, EAST + 1e-6, 2.0)
LAT_C = np.arange(SOUTH, NORTH + 1e-6, 2.0)


def density(tr):
    g, _ = bin_sum(LON_C, LAT_C, tr.lon, tr.lat, z=np.ones(len(tr)), variant="fixed")
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/original/CT/ct_genesis.nc")
    ap.add_argument("--out", default="fig_ct_genesis.png")
    a = ap.parse_args()

    allg = from_ct_genesis(a.path).filter_region(SOUTH, NORTH, WEST, EAST)
    deep = from_ct_genesis(a.path, deep_core_K=200.0).filter_region(SOUTH, NORTH, WEST, EAST)
    longl = from_ct_genesis(a.path, min_lifetime=8).filter_region(SOUTH, NORTH, WEST, EAST)
    panels = [("All CT family genesis", allg),
              ("Deep genesis (core < 200 K)", deep),
              ("Long-lived genesis (lifetime >= 8 systems)", longl)]
    for name, tr in panels:
        print(f"{name:42s}: {len(tr):6d} families | median radius "
              f"{np.median(tr.variables['csize_km']):.0f} km | median lifetime "
              f"{np.median(tr.variables['cslife']):.0f}")

    import matplotlib
    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), subplot_kw={"projection": proj})
    for ax, (name, tr) in zip(axes, panels):
        m = density(tr)
        pc = ax.pcolormesh(LON_C, LAT_C, m, cmap="YlOrRd", vmin=0, transform=proj,
                           shading="auto")
        ax.coastlines(resolution="110m", linewidth=0.5)
        ax.set_extent([WEST, EAST, SOUTH, NORTH], crs=proj)
        ax.set_xticks(np.arange(WEST, EAST + 1, 20), crs=proj)
        ax.set_yticks(np.arange(SOUTH, NORTH + 1, 10), crs=proj)
        ax.tick_params(labelsize=8)
        ax.set_title(f"{name}  (n={len(tr)})", fontsize=10)
        fig.colorbar(pc, ax=ax, shrink=0.85, label="genesis count")
    fig.suptitle("ISCCP CT family genesis density, JAS 1984-2007", fontsize=11)
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
