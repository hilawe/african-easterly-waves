#!/usr/bin/env python
"""Wave-following (trough-relative) composite of MCS about the moving AEW trough.

Uses AEWC (NCEI C00784) ERA-Interim curvature-vorticity trough trajectories over West
Africa and the original ISCCP MCS (csct_africa_cs245.nc). For every trough observation,
MCS within +/- 3 h are binned by longitude RELATIVE to the trough (east positive) and
latitude, accumulated over all troughs. The anomaly (count minus the trough-relative-
longitude mean per latitude) shows where convection sits relative to the moving trough,
a wave-following alternative to the fixed-basepoint composite.
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr

from aew.composites import anomaly, wave_relative_counts
from aew.data.aewc import load_aewc_troughs
from aew.plotting import panel_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aewc-glob", default="data/aewc/ERA-Int_ew_700hPa_*_AFR.nc")
    ap.add_argument("--csct", default="data/original/csct/csct_africa_cs245.nc")
    ap.add_argument("--out", default="fig_wave_following.png")
    a = ap.parse_args()

    # AEWC troughs: West African AEW region, JAS
    tr = (load_aewc_troughs(a.aewc_glob)
          .filter_region(min_lat=5, max_lat=20, min_lon=-40, max_lon=40)
          .filter_months([7, 8, 9]))
    print(f"AEWC trough observations (West Africa, JAS): {len(tr)}")

    cs = xr.open_dataset(a.csct)
    cs_time = pd.DatetimeIndex(cs["time"].values).values
    cs_lon = np.asarray(cs["lon"].values, dtype=float)
    cs_lat = np.asarray(cs["lat"].values, dtype=float)

    rel_c = np.arange(-30.0, 30.1, 2.0)
    lat_c = np.arange(0.0, 25.1, 2.0)
    counts, n = wave_relative_counts(tr.time, tr.lon, cs_time, cs_lon, cs_lat,
                                     rel_c, lat_c, time_tol_hours=3.0)
    print(f"trough observations matched to MCS: {n}; total MCS binned: {int(counts.sum())}")

    # matched-null control: keep trough TIMES and LATITUDES, randomize trough
    # LONGITUDES. If the peak survives vs this null, it is a real trough-relative signal,
    # not the Sahel MCS climatology projecting into the relative frame.
    rng = np.random.default_rng(0)
    n_null = 20
    null_stack = np.empty((n_null,) + counts.shape)
    for i in range(n_null):
        shift = rng.uniform(-180, 180, size=tr.lon.size)
        lon_shift = (tr.lon + shift + 180) % 360 - 180
        null_stack[i], _ = wave_relative_counts(tr.time, lon_shift, cs_time, cs_lon,
                                                cs_lat, rel_c, lat_c, time_tol_hours=3.0)
    null_mean = null_stack.mean(axis=0)
    null_std = null_stack.std(axis=0)
    # signal above the shifted-trough null (per cell), and per-latitude display anomaly
    anom = counts - null_mean
    band = (lat_c >= 5) & (lat_c <= 15)
    prof = anom[band].mean(axis=0)
    prof_null2s = 2 * null_std[band].mean(axis=0)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    amax = np.nanmax(np.abs(anom))
    pc = ax1.contourf(rel_c, lat_c, anom, levels=np.linspace(-amax, amax, 21),
                      cmap="RdBu_r", extend="both")
    ax1.axvline(0, color="green", lw=2)
    ax1.set_ylabel("Latitude (N)")
    ax1.set_title("MCS excess over shifted-trough null, relative to the moving AEW trough")
    panel_label(ax1, "a", 15)
    # give the colorbar its own axis appended to ax1, and append a matching invisible
    # spacer to ax2, so both panels keep the same width and the shared longitude axis
    # lines up vertically (the trough axis at 0 sits at the same position in both)
    cax = make_axes_locatable(ax1).append_axes("right", size="3%", pad=0.15)
    fig.colorbar(pc, cax=cax, label="MCS count minus shifted-trough null")
    make_axes_locatable(ax2).append_axes("right", size="3%", pad=0.15).set_axis_off()
    # latitude-averaged (5-15N) profile with the null +/-2 sigma band
    ax2.fill_between(rel_c, -prof_null2s, prof_null2s, color="grey", alpha=0.25,
                     label="shifted-trough null +/-2 sigma")
    ax2.plot(rel_c, prof, color="tab:red", label="observed - null")
    ax2.axvline(0, color="green", lw=2)
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xlabel("Longitude relative to trough (deg; east positive)")
    ax2.set_ylabel("MCS excess, 5-15N mean")
    panel_label(ax2, "b", 15)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.suptitle(f"Wave-following composite  ({n} trough obs, JAS)", fontsize=12)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
