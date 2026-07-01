#!/usr/bin/env python
"""Cross-check the basepoint composite dates against the NCEI AEW Climatology (C00784).

Two independent AEW definitions:
  - ours: local maxima of filtered v700 (`td`) at the basepoint above 2 sigma (the 272-date
    method), here over 2000-2004.
  - AEWC: curvature-vorticity wave-trough trajectories (Belanger et al.), ERA-Interim member.

For each of our composite dates, find the signed time to the nearest AEWC trough passage at
the basepoint longitude. If the two definitions capture the same waves, the composite dates
should coincide with (or sit at a consistent phase offset from) AEWC trough passages far more
than a random null would.
"""

import argparse
import glob

import numpy as np
import pandas as pd
import xarray as xr

from aew.events import composite_dates, std_threshold


def load_aewc(files, base_lon, lon_tol, lat_lo, lat_hi):
    times = []
    for f in files:
        ds = xr.open_dataset(f)
        t = pd.DatetimeIndex(ds["time"].values)
        lat = np.asarray(ds["lat"].values, float)
        lon = np.asarray(ds["lon"].values, float)
        lon = np.where(lon > 180, lon - 360, lon)
        sel = (np.abs(lon - base_lon) <= lon_tol) & (lat >= lat_lo) & (lat <= lat_hi)
        times.append(t[sel])
        ds.close()
    return pd.DatetimeIndex(np.concatenate([t.values for t in times])).sort_values()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aewc-glob", default="data/aewc/ERA-Int_ew_700hPa_*_AFR.nc")
    ap.add_argument("--td", default="data/original/eraint/v700_waves_basebox.nc")
    ap.add_argument("--base-lat", type=float, default=10.0)
    ap.add_argument("--base-lon", type=float, default=0.0)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--lon-tol", type=float, default=2.5)
    ap.add_argument("--tol-days", type=float, default=1.5)
    ap.add_argument("--out", default="aewc_crosscheck.png")
    a = ap.parse_args()

    files = sorted(glob.glob(a.aewc_glob))
    years = sorted(int(f.split("_")[-2]) for f in files)
    passages = load_aewc(files, a.base_lon, a.lon_tol, a.base_lat - 5, a.base_lat + 5)
    print(f"AEWC trough passages near basepoint ({a.base_lon:g}E, "
          f"{a.base_lat-5:g}-{a.base_lat+5:g}N, +/-{a.lon_tol:g} lon): {len(passages)} "
          f"over {years[0]}-{years[-1]}")

    # our composite dates from td, restricted to the AEWC window
    td = xr.open_dataset(a.td)["td"].sel(lat=a.base_lat, lon=a.base_lon, method="nearest")
    tt = pd.DatetimeIndex(td["time"].values)
    tv = np.asarray(td.values, float)
    keep = (tt.year >= years[0]) & (tt.year <= years[-1])
    tt, tv = tt[keep], tv[keep]
    thr = std_threshold(tv[np.isfinite(tv)], a.sigma)
    cd = pd.DatetimeIndex(composite_dates(tv, thr, time=tt.values).time)
    print(f"our composite dates ({a.sigma:g} sigma): {len(cd)}")

    # signed nearest-passage lag (days) for each composite date
    pv = passages.values.astype("datetime64[ns]").astype("int64")
    def nearest_lag(dates):
        dv = pd.DatetimeIndex(dates).values.astype("datetime64[ns]").astype("int64")
        out = np.empty(dv.size)
        for i, d in enumerate(dv):
            out[i] = (d - pv[np.argmin(np.abs(pv - d))]) / 8.64e13
        return out

    lag = nearest_lag(cd)
    hit = np.mean(np.abs(lag) <= a.tol_days)

    # null: random JAS-season dates in the same window
    rng = np.random.default_rng(0)
    jas = tt[np.isin(tt.month, [7, 8, 9])]
    null_hits = []
    for _ in range(200):
        rd = pd.DatetimeIndex(rng.choice(jas.values, size=len(cd), replace=False))
        null_hits.append(np.mean(np.abs(nearest_lag(rd)) <= a.tol_days))
    null = np.mean(null_hits)

    print(f"composite dates within +/-{a.tol_days:g} d of an AEWC trough passage: "
          f"{hit*100:.0f}%   (random null {null*100:.0f}% +/- {np.std(null_hits)*100:.0f}%)")
    print(f"median |lag| = {np.median(np.abs(lag)):.2f} d ; median signed lag = "
          f"{np.median(lag):+.2f} d")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lag, bins=np.arange(-5, 5.1, 0.5), color="tab:red", alpha=0.8,
            label=f"our dates ({hit*100:.0f}% within +/-{a.tol_days:g} d)")
    ax.axvline(0, color="k", lw=0.8)
    ax.axvspan(-a.tol_days, a.tol_days, color="grey", alpha=0.15)
    ax.set_xlabel("signed time to nearest AEWC trough passage (days)")
    ax.set_ylabel("count of our composite dates")
    ax.set_title(f"Basepoint composite dates vs AEWC curvature-vorticity troughs "
                 f"({years[0]}-{years[-1]})\nrandom null {null*100:.0f}% within window")
    ax.legend()
    fig.tight_layout(); fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
