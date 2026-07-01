#!/usr/bin/env python
"""Amplitude-preconditioning lead-lag: does the AEW trough precede the convection?

Addresses the causality objection (troughs and MCS as two faces of one convectively
coupled disturbance). For each AEWC trough observation, the same wave's curvature-vorticity
amplitude 48 h earlier (along its own trajectory) is the PRECONDITIONING amplitude, set
before the present convection exists. Two tests:

  1. Split troughs by preconditioning amplitude (T-48 h) into weak/strong terciles and
     composite MCS in trough-relative longitude for each, vs a shifted-trough null. If the
     strong-precursor waves carry the larger convective excess, the wave led the convection.
  2. Lag regression: local MCS count at T=0 (trough-relative box) vs amplitude at T-48 h.

A positive lead relationship is the forcing direction and is hard to explain by convection
merely spinning up the wave, because the amplitude is measured before the convection.
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr

from aew.composites import wave_relative_counts
from aew.data.aewc import load_aewc_trajectories

REL_C = np.arange(-30.0, 30.1, 2.0)
LAT_C = np.arange(0.0, 25.1, 2.0)
BAND = (LAT_C >= 5) & (LAT_C <= 15)


def excess_profile(tr, cst, csx, csy, rng, n_null=15):
    counts, _ = wave_relative_counts(tr.time, tr.lon, cst, csx, csy, REL_C, LAT_C, 3.0)
    null = np.empty((n_null,) + counts.shape)
    for i in range(n_null):
        sh = (tr.lon + rng.uniform(-180, 180, tr.lon.size) + 180) % 360 - 180
        null[i], _ = wave_relative_counts(tr.time, sh, cst, csx, csy, REL_C, LAT_C, 3.0)
    return (counts - null.mean(0))[BAND].mean(0), 2 * null.std(0)[BAND].mean(0)


def local_counts(tr, cst, csx, csy, tol_h=3.0, dlon=8.0, lat_lo=5.0, lat_hi=15.0):
    """Per-trough-observation MCS count in a trough-relative box within +/- tol_h."""
    ct = pd.DatetimeIndex(cst).values.astype("datetime64[ns]").astype("int64")
    o = np.argsort(ct); ct = ct[o]; csx = np.asarray(csx)[o]; csy = np.asarray(csy)[o]
    tol = int(tol_h * 3.6e12)
    tt = pd.DatetimeIndex(tr.time).values.astype("datetime64[ns]").astype("int64")
    out = np.zeros(tr.time.size)
    for i, (tobs, L) in enumerate(zip(tt, tr.lon)):
        lo = np.searchsorted(ct, tobs - tol, "left")
        hi = np.searchsorted(ct, tobs + tol, "left")
        if hi <= lo:
            continue
        rel = (csx[lo:hi] - L + 180) % 360 - 180
        m = (np.abs(rel) <= dlon) & (csy[lo:hi] >= lat_lo) & (csy[lo:hi] <= lat_hi)
        out[i] = m.sum()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aewc-glob", default="data/aewc/ERA-Int_ew_700hPa_*_AFR.nc")
    ap.add_argument("--csct", default="data/original/csct/csct_africa_cs245.nc")
    ap.add_argument("--out", default="fig_precondition.png")
    a = ap.parse_args()

    tr = (load_aewc_trajectories(a.aewc_glob)
          .filter_region(min_lat=5, max_lat=20, min_lon=-40, max_lon=40)
          .filter_months([7, 8, 9]))
    pre = tr.variables["crv_lag48"]          # amplitude 48 h earlier (preconditioning)
    tr = tr.filter(np.isfinite(pre)); pre = pre[np.isfinite(pre)]
    lo_t, hi_t = np.nanpercentile(pre, [33.3, 66.7])
    weak = tr.filter(pre <= lo_t)
    strong = tr.filter(pre >= hi_t)
    print(f"troughs with 48 h history: {len(tr)}; weak-precursor {len(weak)}, "
          f"strong-precursor {len(strong)}")

    cs = xr.open_dataset(a.csct)
    cst = pd.DatetimeIndex(cs["time"].values).values
    csx = np.asarray(cs["lon"].values, float)
    csy = np.asarray(cs["lat"].values, float)

    rng = np.random.default_rng(0)
    ew, nw = excess_profile(weak, cst, csx, csy, rng)
    es, ns = excess_profile(strong, cst, csx, csy, rng)
    print(f"peak trough-relative MCS excess: weak-precursor {np.max(ew):.0f}, "
          f"strong-precursor {np.max(es):.0f}")

    # lag regression: local MCS at T=0 vs amplitude at T-48 h
    lc = local_counts(tr, cst, csx, csy)
    x = pre * 1e6  # curvature vorticity in 1e-6 /s
    r = np.corrcoef(x, lc)[0, 1]
    b, a0 = np.polyfit(x, lc, 1)
    print(f"lag regression  local MCS(T0) vs amplitude(T-48h): r = {r:.2f}, "
          f"slope = {b:.2f} systems per 1e-6/s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.fill_between(REL_C, -nw, nw, color="grey", alpha=0.2, label="shifted-trough null +/-2 sigma")
    ax1.plot(REL_C, es, color="tab:red", label=f"strong precursor at T-48h (n={len(strong)})")
    ax1.plot(REL_C, ew, color="tab:blue", label=f"weak precursor at T-48h (n={len(weak)})")
    ax1.axvline(0, color="green", lw=2); ax1.axhline(0, color="k", lw=0.6)
    ax1.set_xlabel("Longitude relative to trough (deg; east positive)")
    ax1.set_ylabel("MCS excess over null, 5-15N mean")
    ax1.set_title("MCS organization by wave amplitude 48 h BEFORE the convection")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # binned means for the scatter (raw scatter is dense)
    bins = np.quantile(x, np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(x, bins) - 1, 0, 9)
    bx = np.array([x[idx == k].mean() for k in range(10)])
    by = np.array([lc[idx == k].mean() for k in range(10)])
    ax2.scatter(x, lc, s=4, alpha=0.15, color="grey")
    ax2.plot(bx, by, "o-", color="tab:red", label="decile means")
    xs = np.linspace(x.min(), x.max(), 50)
    ax2.plot(xs, a0 + b * xs, "k--", lw=1, label=f"fit r={r:.2f}")
    ax2.set_xlabel("wave curvature-vorticity amplitude 48 h earlier (1e-6 /s)")
    ax2.set_ylabel("local MCS count at T=0 (trough box)")
    ax2.set_title("Preconditioning: earlier wave amplitude vs later convection")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
