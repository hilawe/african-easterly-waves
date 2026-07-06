#!/usr/bin/env python
"""Wave-following (trough-relative) composite of MCS about the moving AEW trough.

Uses AEWC (NCEI C00784) ERA-Interim curvature-vorticity trough trajectories over West
Africa and the original ISCCP MCS (csct_africa_cs245.nc). For every trough observation,
MCS within +/- 3 h are binned by longitude RELATIVE to the trough (east positive) and
latitude, accumulated over all troughs. The anomaly (count minus the trough-relative-
longitude mean per latitude) shows where convection sits relative to the moving trough,
a wave-following alternative to the fixed-basepoint composite.

Three panels share the longitude axis so they line up for the eye:
  (a) the latitude-longitude excess over the shifted-trough null,
  (b) its 5-15 N mean with the null +/-2 sigma band,
  (c) the same 5-15 N mean split into weak/strong curvature-vorticity terciles.
Panels (a) and (b) use an independent 20-member null (seed 0); panel (c) uses the
15-member null per tercile (seed 0), matching the standalone stratified computation so
the reported peaks are unchanged.
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr

from aew.composites import wave_relative_counts
from aew.data.aewc import load_aewc_troughs
from aew.plotting import panel_label


def excess_profile(sub, cs_time, cs_lon, cs_lat, rel_c, lat_c, band, rng, n_null):
    """5-15 N mean MCS excess over a shifted-trough null, and the null +/-2 sigma band."""
    counts, n = wave_relative_counts(sub.time, sub.lon, cs_time, cs_lon, cs_lat,
                                     rel_c, lat_c, time_tol_hours=3.0)
    null = np.empty((n_null,) + counts.shape)
    for i in range(n_null):
        sh = (sub.lon + rng.uniform(-180, 180, sub.lon.size) + 180) % 360 - 180
        null[i], _ = wave_relative_counts(sub.time, sh, cs_time, cs_lon, cs_lat,
                                          rel_c, lat_c, time_tol_hours=3.0)
    exc = (counts - null.mean(0))[band].mean(0)
    null2s = 2 * null.std(0)[band].mean(0)
    return exc, null2s, n


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
    band = (lat_c >= 5) & (lat_c <= 15)
    counts, n = wave_relative_counts(tr.time, tr.lon, cs_time, cs_lon, cs_lat,
                                     rel_c, lat_c, time_tol_hours=3.0)
    print(f"trough observations matched to MCS: {n}; total MCS binned: {int(counts.sum())}")

    # (a)/(b) matched-null control: keep trough TIMES and LATITUDES, randomize trough
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
    anom = counts - null_mean
    prof = anom[band].mean(axis=0)
    prof_null2s = 2 * null_std[band].mean(axis=0)

    # (c) stratify by curvature vorticity (wave amplitude) into weak/strong terciles, each
    # with its own 15-member null. A fresh seed-0 stream reproduces the standalone
    # stratified figure exactly (weak draws first, then strong).
    crv = tr.variables["crv"]
    good = np.isfinite(crv)
    tr_s = tr.filter(good)
    crv = crv[good]
    lo_t, hi_t = np.nanpercentile(crv, [33.3, 66.7])
    weak = tr_s.filter(crv <= lo_t)
    strong = tr_s.filter(crv >= hi_t)
    print(f"stratify by curvature vorticity: weak n={len(weak)} (<= {lo_t:.3g}), "
          f"strong n={len(strong)} (>= {hi_t:.3g})")
    rng_c = np.random.default_rng(0)
    ew, nw, _ = excess_profile(weak, cs_time, cs_lon, cs_lat, rel_c, lat_c, band, rng_c, 15)
    es, _, _ = excess_profile(strong, cs_time, cs_lon, cs_lat, rel_c, lat_c, band, rng_c, 15)
    print(f"peak MCS excess (5-15N): weak {np.max(ew):.0f}, strong {np.max(es):.0f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(8, 11), sharex=True, gridspec_kw={"height_ratios": [1.5, 1, 1]})

    # (a) latitude-longitude excess, colorbar in its own appended axis. Round the fill
    # levels and colorbar ticks to a nice interval (the 2.5/5/10/25/... family) rather
    # than amax-derived values like 537 or 1074.
    amax = np.nanmax(np.abs(anom))
    nice = np.array([2.5, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000])
    j = int(np.argmax(amax / nice <= 4))           # smallest step with <= 4 ticks per side
    tick_step = float(nice[j])
    fill_step = float(nice[max(0, j - 2)])          # finer bands for a smooth fill
    vmax = np.ceil(amax / fill_step) * fill_step
    tmax = np.floor(amax / tick_step) * tick_step
    levels = np.arange(-vmax, vmax + fill_step * 0.5, fill_step)
    ticks = np.arange(-tmax, tmax + tick_step * 0.5, tick_step)
    pc = ax1.contourf(rel_c, lat_c, anom, levels=levels, cmap="RdBu_r", extend="both")
    ax1.axvline(0, color="green", lw=2)
    ax1.set_ylabel("Latitude (N)")
    ax1.set_title("MCS excess over shifted-trough null, relative to the moving AEW trough")
    panel_label(ax1, "a", 15)
    cax = make_axes_locatable(ax1).append_axes("right", size="3%", pad=0.15)
    fig.colorbar(pc, cax=cax, label="MCS count minus shifted-trough null", ticks=ticks)
    # matching invisible spacers so (b) and (c) keep (a)'s width; the shared longitude
    # axis then lines up vertically and the trough axis at 0 sits at one x-position
    make_axes_locatable(ax2).append_axes("right", size="3%", pad=0.15).set_axis_off()
    make_axes_locatable(ax3).append_axes("right", size="3%", pad=0.15).set_axis_off()

    # (b) 5-15 N mean profile with the null +/-2 sigma band
    ax2.fill_between(rel_c, -prof_null2s, prof_null2s, color="grey", alpha=0.25,
                     label="shifted-trough null +/-2 sigma")
    ax2.plot(rel_c, prof, color="tab:red", label="observed - null")
    ax2.axvline(0, color="green", lw=2)
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_ylabel("MCS excess, 5-15N mean")
    ax2.set_title("All troughs", fontsize=10)
    panel_label(ax2, "b", 15)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # (c) same 5-15 N mean, split by wave amplitude (curvature-vorticity terciles)
    ax3.fill_between(rel_c, -nw, nw, color="grey", alpha=0.2,
                     label="shifted-trough null +/-2 sigma")
    ax3.plot(rel_c, ew, color="tab:blue", label=f"weak waves (n={len(weak)})")
    ax3.plot(rel_c, es, color="tab:red", label=f"strong waves (n={len(strong)})")
    ax3.axvline(0, color="green", lw=2)
    ax3.axhline(0, color="k", lw=0.6)
    ax3.set_xlabel("Longitude relative to trough (deg; east positive)")
    ax3.set_ylabel("MCS excess, 5-15N mean")
    ax3.set_title("Stratified by wave amplitude (curvature-vorticity terciles)", fontsize=10)
    panel_label(ax3, "c", 15)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    fig.suptitle(f"Wave-following composite  ({n} trough obs, JAS)", fontsize=12)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
