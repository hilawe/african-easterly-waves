#!/usr/bin/env python
"""Developing vs non-developing trough composite: the pre-trough moisture precondition.

Conditions the causality question on the environment instead of chasing linear wave forcing.
Each AEWC trough is given a convective response from the independent csct cloud-system record
(count of systems in a trough-relative box over the next 24 h), and troughs are split into
developing (top response tercile) and non-developing (bottom tercile). The test is whether
the developing troughs sat in a moister environment BEFORE that convection existed: the
pre-trough total column water vapour (TCWV) sampled ~24 h earlier along each trough's own
track. A higher pre-trough TCWV for developing troughs is a moisture precondition set ahead
of the convection, which the wave-convection covariation alone cannot show.

Moisture source and its limit: TCWV here is the AEWC trough-mean SSM/I precipitable water
(meantpw), which is only ~40% populated (satellite overpass gaps), so only troughs with a
real earlier sample contribute. The primary variable in the plan is ERA5 700 hPa relative
humidity (full coverage); this script is the framework and a first result on the on-disk
TCWV, and the 700 hPa RH pull is the next data step. A longitude-stratified panel checks that
any developing/non-developing moisture difference is not just the moist-west/dry-east gradient.

Inputs: AEWC trajectories (data/aewc), csct CS-245 systems (data/original/csct).
Writes fig_developing.png.
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr

from aew.data.aewc import load_aewc_trajectories
from aew.environment import (
    cluster_bootstrap_diff,
    forward_response,
    lead_value,
    terciles,
)

LEAD_H = 24.0
TOL_H = 6.0
RESP_WIN_H = 24.0
DLON = 8.0
LAT_LO, LAT_HI = 5.0, 15.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aewc-glob", default="data/aewc/ERA-Int_ew_700hPa_*_AFR.nc")
    ap.add_argument("--csct", default="data/original/csct/csct_africa_cs245.nc")
    ap.add_argument("--out", default="fig_developing.png")
    a = ap.parse_args()

    tr = (load_aewc_trajectories(a.aewc_glob)
          .filter_region(min_lat=5, max_lat=20, min_lon=-30, max_lon=40)
          .filter_months([7, 8, 9]))
    print(f"JAS corridor troughs: {len(tr)}")

    cs = xr.open_dataset(a.csct)
    cst = pd.DatetimeIndex(cs["time"].values).values
    csx = np.asarray(cs["lon"].values, float)
    csy = np.asarray(cs["lat"].values, float)
    cs.close()

    resp = forward_response(tr.time, tr.lon, cst, csx, csy, RESP_WIN_H, DLON, LAT_LO, LAT_HI)
    lead_tcwv = lead_value(tr.time, tr.variables["traj_id"], tr.variables["tpw"],
                           LEAD_H, TOL_H)

    keep = np.isfinite(lead_tcwv)
    print(f"troughs with a real pre-trough (t-{LEAD_H:.0f}h +/-{TOL_H:.0f}h) TCWV: "
          f"{keep.sum()} ({100 * keep.mean():.0f}%)")
    resp_k = resp[keep]
    tcwv_k = lead_tcwv[keep]
    lon_k = tr.lon[keep]

    tid_k = tr.variables["traj_id"][keep]
    # Terciles are computed among the analyzable (finite pre-trough TCWV) troughs, not all
    # troughs; the split is "developing vs non-developing among SSM/I-sampled troughs".
    low, high = terciles(resp_k)           # non-developing, developing
    from scipy import stats
    rng = np.random.default_rng(0)

    # pooled comparison -- reported but flagged: it mixes the moist-west/dry-east gradient
    # with the developing/non-developing split, so it is geography-confounded (Simpson).
    ndp, dvp = tcwv_k[low], tcwv_k[high]
    print(f"\nPOOLED (geography-confounded, do not headline): non-dev {ndp.mean():.1f} mm "
          f"(n={low.sum()}), developing {dvp.mean():.1f} mm (n={high.sum()}), "
          f"diff {dvp.mean()-ndp.mean():+.1f} mm")

    # geography-controlled: developing/non-developing pre-trough TCWV within longitude bins,
    # spanning the full loaded range; a per-group minimum sample size flags where the sparse
    # TCWV leaves a bin (mostly the east) under-sampled. Per-bin p is observation-level.
    MIN_N = 15
    edges = np.arange(-30, 41, 10.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    nd_lon, dv_lon, reliable = [], [], []
    print("\nlon-bin(E)  n_nd  n_dv   non-dev  developing   diff(mm)   p(obs-level)")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (lon_k >= lo) & (lon_k < hi)
        nd, dv = tcwv_k[m & low], tcwv_k[m & high]
        ok = nd.size >= MIN_N and dv.size >= MIN_N
        reliable.append(ok)
        nd_lon.append(nd.mean() if nd.size else np.nan)
        dv_lon.append(dv.mean() if dv.size else np.nan)
        if ok:
            _, pb = stats.ttest_ind(dv, nd, equal_var=False)
            print(f"{lo:+.0f}..{hi:+.0f}   {nd.size:4d}  {dv.size:4d}   {nd.mean():6.1f}   "
                  f"{dv.mean():8.1f}   {dv.mean()-nd.mean():+7.1f}   {pb:.3f}")
        else:
            print(f"{lo:+.0f}..{hi:+.0f}   {nd.size:4d}  {dv.size:4d}   under-sampled (need "
                  f"{MIN_N}/group), not tested")
    nd_lon = np.array(nd_lon); dv_lon = np.array(dv_lon); reliable = np.array(reliable)

    # primary result: western corridor (lon < 0), both groups sampled. Significance uses a
    # CLUSTER bootstrap over trajectories, because a wave contributes many correlated 6-hourly
    # observations -- treating each observation as independent (a plain t-test) overstates it.
    west = lon_k < 0
    ndw, dvw = tcwv_k[west & low], tcwv_k[west & high]
    diff, lo_ci, hi_ci, nwa, nwb = cluster_bootstrap_diff(
        tid_k[west & low], ndw, tid_k[west & high], dvw, rng)
    _, pw_obs = stats.ttest_ind(dvw, ndw, equal_var=False)
    sig = "not significant (CI crosses 0)" if lo_ci <= 0 <= hi_ci else "significant"
    print("\n(developing vs non-developing are the response terciles AMONG the SSM/I-sampled "
          "troughs, not all troughs)")
    print(f"PRIMARY -- western corridor (lon<0E): non-dev {ndw.mean():.1f} mm "
          f"(n={ndw.size} obs, {nwa} waves), developing {dvw.mean():.1f} mm "
          f"(n={dvw.size} obs, {nwb} waves), difference {diff:+.1f} mm. Trajectory "
          f"cluster-bootstrap 95% CI [{lo_ci:+.1f}, {hi_ci:+.1f}] -- {sig}. "
          f"(observation-level t-test p={pw_obs:.1e} is over-optimistic: it ignores that "
          "each wave gives many correlated samples.)")
    print("Reading: at most a weak hint that developing troughs in the west sit in moister "
          "pre-trough air; the wave-clustered test does not establish it. East of ~0E the "
          "non-developing sample is too small to test (sparse SSM/I coverage).")

    print("\nCAVEAT: TCWV here is the ~40%-populated AEWC SSM/I trough-mean precipitable "
          "water and is sampled at the trough's own shifting location, not a fixed meridian. "
          "The planned primary variable is full-coverage ERA5 700 hPa relative humidity "
          "sampled in a fixed box ahead of the trough. A moisture precondition is consistent "
          "with, but does not by itself prove, convection organizing the wave.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    parts = ax1.violinplot([ndw, dvw], positions=[0, 1], showmeans=True, showextrema=False)
    for pc, col in zip(parts["bodies"], ("tab:blue", "tab:red")):
        pc.set_facecolor(col); pc.set_alpha(0.5)
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(
        [f"non-developing\n(n={ndw.size})", f"developing\n(n={dvw.size})"])
    ax1.set_ylabel("pre-trough TCWV, 24 h before passage (mm)")
    ax1.set_title(f"Western corridor (lon<0E): developing {diff:+.1f} mm\n"
                  f"cluster-bootstrap 95% CI [{lo_ci:+.1f}, {hi_ci:+.1f}] ({sig})")
    ax1.grid(alpha=0.3, axis="y")

    # Panel B: per-bin means; reliable bins solid, under-sampled bins greyed/open
    for arr, col, lab in ((nd_lon, "tab:blue", "non-developing"),
                          (dv_lon, "tab:red", "developing")):
        ax2.plot(centers[reliable], arr[reliable], "o-", color=col, label=lab)
        if (~reliable).any():
            ax2.plot(centers[~reliable], arr[~reliable], "x", color=col, alpha=0.4)
    ax2.plot([], [], "x", color="grey", label=f"under-sampled (<{MIN_N}/group)")
    ax2.set_xlabel("longitude bin (deg E)")
    ax2.set_ylabel("pre-trough TCWV (mm)")
    ax2.set_title("By longitude: precondition well-sampled only in the west")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(a.out, dpi=150)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
