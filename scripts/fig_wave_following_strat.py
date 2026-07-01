#!/usr/bin/env python
"""Wave-following MCS composite STRATIFIED by wave property (amplitude or wavelength).

Splits AEWC troughs into weak/strong terciles by curvature vorticity (wave amplitude) or
into short/long terciles by wavelength, and composites MCS in trough-relative coordinates
for each subset, each with its own shifted-trough null. Tests whether stronger (or longer)
waves organize convection more strongly at/ahead of the trough.
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr

from aew.composites import wave_relative_counts
from aew.data.aewc import load_aewc_troughs

REL_C = np.arange(-30.0, 30.1, 2.0)
LAT_C = np.arange(0.0, 25.1, 2.0)
BAND = (LAT_C >= 5) & (LAT_C <= 15)


def excess_profile(tr, cs_time, cs_lon, cs_lat, rng, n_null=15):
    counts, n = wave_relative_counts(tr.time, tr.lon, cs_time, cs_lon, cs_lat,
                                     REL_C, LAT_C, time_tol_hours=3.0)
    null = np.empty((n_null,) + counts.shape)
    for i in range(n_null):
        sh = (tr.lon + rng.uniform(-180, 180, tr.lon.size) + 180) % 360 - 180
        null[i], _ = wave_relative_counts(tr.time, sh, cs_time, cs_lon, cs_lat,
                                          REL_C, LAT_C, time_tol_hours=3.0)
    exc = (counts - null.mean(0))[BAND].mean(0)
    null2s = 2 * null.std(0)[BAND].mean(0)
    return exc, null2s, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by", choices=["crv", "wavelength"], default="crv")
    ap.add_argument("--aewc-glob", default="data/aewc/ERA-Int_ew_700hPa_*_AFR.nc")
    ap.add_argument("--csct", default="data/original/csct/csct_africa_cs245.nc")
    ap.add_argument("--out", default="fig_wave_following_strat.png")
    a = ap.parse_args()

    tr = (load_aewc_troughs(a.aewc_glob)
          .filter_region(min_lat=5, max_lat=20, min_lon=-40, max_lon=40)
          .filter_months([7, 8, 9]))
    val = tr.variables[a.by]
    good = np.isfinite(val)
    tr = tr.filter(good); val = val[good]
    lo_t, hi_t = np.nanpercentile(val, [33.3, 66.7])
    weak = tr.filter(val <= lo_t)
    strong = tr.filter(val >= hi_t)
    label = "curvature vorticity" if a.by == "crv" else "wavelength"
    print(f"stratify by {label}: weak/low n={len(weak)} (<= {lo_t:.3g}), "
          f"strong/high n={len(strong)} (>= {hi_t:.3g})")

    cs = xr.open_dataset(a.csct)
    cs_time = pd.DatetimeIndex(cs["time"].values).values
    cs_lon = np.asarray(cs["lon"].values, float)
    cs_lat = np.asarray(cs["lat"].values, float)

    rng = np.random.default_rng(0)
    ew, nw, _ = excess_profile(weak, cs_time, cs_lon, cs_lat, rng)
    es, ns, _ = excess_profile(strong, cs_time, cs_lon, cs_lat, rng)
    print(f"peak MCS excess (5-15N): weak {np.max(ew):.0f}, strong {np.max(es):.0f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    hi_name = "long" if a.by == "wavelength" else "strong"
    lo_name = "short" if a.by == "wavelength" else "weak"
    ax.fill_between(REL_C, -nw, nw, color="grey", alpha=0.2)
    ax.plot(REL_C, ew, color="tab:blue", label=f"{lo_name} waves (n={len(weak)})")
    ax.plot(REL_C, es, color="tab:red", label=f"{hi_name} waves (n={len(strong)})")
    ax.axvline(0, color="green", lw=2); ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Longitude relative to trough (deg; east positive)")
    ax.set_ylabel("MCS excess over shifted-trough null, 5-15N mean")
    ax.set_title(f"Wave-following MCS composite stratified by {label}  (JAS, terciles)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
