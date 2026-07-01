#!/usr/bin/env python
"""Close the wind-side independence loop: wk_bandpass on global ERA5 v700 vs original td.

Applies the ported space-time filter (westward wavenumber -20..0, 2.5-10 day) to global
6-hourly ERA5 v700, extracts the basepoint (10N/0E) wave series, and compares it against
the original ERA-Interim wave series (v700.anom.waves.nc `td`) over the same window:
  - correlation of the two filtered basepoint series
  - composite-date count and std at 2 sigma from each
This shows the wind side can be reproduced from public ERA5 alone, without the original
wave file. Exact equality is not expected (ERA5 vs ERA-Interim, different grid).
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr

from aew.events import composite_dates, std_threshold
from aew.filtering import wk_bandpass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--era5", default="data/era5/era5_v700_2000-2004_6h_global.nc")
    ap.add_argument("--td", default="data/original/eraint/v700_waves_basebox.nc")
    ap.add_argument("--base-lat", type=float, default=10.0)
    ap.add_argument("--base-lon", type=float, default=0.0)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--out", default="wk_independence.png")
    a = ap.parse_args()

    ds = xr.open_dataset(a.era5).squeeze(drop=True)
    latn = "latitude" if "latitude" in ds.coords else "lat"
    lonn = "longitude" if "longitude" in ds.coords else "lon"
    v = ds["v700"]
    lon = ds[lonn].values
    # order dims (time, lat, lon) and wk-filter (westward wavenumbers -20..0, 2.5-10 day)
    v = v.transpose("time" if "time" in v.dims else "valid_time", latn, lonn)
    tname = "time" if "time" in v.dims else "valid_time"
    filt = wk_bandpass(v.values, obs_per_day=4, period_min=2.5, period_max=10.0,
                       wavenum_min=-20, wavenum_max=0, time_axis=0, lon_axis=-1,
                       lon=lon)
    fda = xr.DataArray(filt, dims=("time", latn, lonn),
                       coords={"time": ds[tname].values, latn: ds[latn].values,
                               lonn: ds[lonn].values})
    era_series = fda.sel({latn: a.base_lat, lonn: a.base_lon}, method="nearest")
    et = pd.DatetimeIndex(era_series["time"].values)
    ev = np.asarray(era_series.values, float)

    # original td at basepoint over the same window
    td = xr.open_dataset(a.td)["td"].sel(lat=a.base_lat, lon=a.base_lon, method="nearest")
    tt = pd.DatetimeIndex(td["time"].values)
    tv = np.asarray(td.values, float)
    lo, hi = et.min(), et.max()
    keep = (tt >= lo) & (tt <= hi)
    tt, tv = tt[keep], tv[keep]

    # align on common timestamps
    common = et.intersection(tt)
    ea = pd.Series(ev, index=et).reindex(common).to_numpy()
    ta = pd.Series(tv, index=tt).reindex(common).to_numpy()
    ok = np.isfinite(ea) & np.isfinite(ta)
    r = np.corrcoef(ea[ok], ta[ok])[0, 1]

    # composite dates from each
    thr_e = std_threshold(ev[np.isfinite(ev)], a.sigma)
    cd_e = composite_dates(ev, thr_e, time=et.values)
    thr_t = std_threshold(tv[np.isfinite(tv)], a.sigma)
    cd_t = composite_dates(tv, thr_t, time=tt.values)

    print(f"window {str(lo)[:10]}..{str(hi)[:10]}  basepoint "
          f"{float(era_series[latn]):.1f}N/{float(era_series[lonn]):.1f}E")
    print(f"series correlation (ERA5 wk_bandpass vs original td): r = {r:.2f}  (n={ok.sum()})")
    print(f"ERA5 wk_bandpass: std={np.nanstd(ev, ddof=1):.3f}  thr={thr_e:.3f}  "
          f"n_dates={len(cd_e)}")
    print(f"original td     : std={np.nanstd(tv, ddof=1):.3f}  thr={thr_t:.3f}  "
          f"n_dates={len(cd_t)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    seg = slice(0, min(480, ok.sum()))  # ~first 120 days
    ax1.plot(common[seg], ea[seg], color="tab:red", lw=0.9, label="ERA5 wk_bandpass")
    ax1.plot(common[seg], ta[seg], color="tab:blue", lw=0.9, alpha=0.7, label="original td")
    ax1.set_title(f"Basepoint v700 wave series (first 120 d)  r={r:.2f}")
    ax1.set_ylabel("m/s"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.scatter(ta[ok], ea[ok], s=3, alpha=0.3)
    lim = np.nanmax(np.abs([ta[ok], ea[ok]]))
    ax2.plot([-lim, lim], [-lim, lim], "k--", lw=0.7)
    ax2.set_xlabel("original td (m/s)"); ax2.set_ylabel("ERA5 wk_bandpass (m/s)")
    ax2.set_title(f"r={r:.2f}   ERA5 n_dates={len(cd_e)} vs td {len(cd_t)}")
    ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
