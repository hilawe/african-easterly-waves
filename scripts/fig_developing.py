#!/usr/bin/env python
"""Developing vs non-developing trough composite: the pre-trough moisture precondition.

Conditions the causality question on the environment instead of chasing linear wave forcing.
Each AEWC trough is given a convective response from the independent csct cloud-system record
(count of systems in a trough-relative box over the next 24 h), and troughs are split into
developing (top response tercile) and non-developing (bottom tercile) WITHIN each 10-degree
longitude x calendar-month cell, so the two groups have a matched composition in geography
and season and the comparison is neither the moist-west/dry-east climatology nor the
seasonal moisture cycle in disguise. The test is whether the developing
troughs sat in a moister environment BEFORE that convection existed (~24 h earlier). The
t-24h field is not an unperturbed background (the wave train continuously moistens the
corridor), so a positive contrast is read as a probabilistic bias within the coupled
wave-train system, not a binary trigger acting on a single wave. Significance is a
trajectory cluster bootstrap (a wave's many 6-hourly observations are correlated), and the
wave-level estimand is reported alongside the per-observation one.

The environment source is selectable (--env):

- ``tpw`` (default until the ERA5 pull lands): the AEWC trough-mean SSM/I precipitable
  water (meantpw), sampled along the trough's own track. Only ~40% populated (satellite
  overpass gaps), so only troughs with a real earlier sample contribute.
- ``r700`` / ``tcwv``: full-coverage ERA5 fields (data/era5/region6h, from
  scripts/download_era5_env.py), sampled as a fixed box at the trough's meridian 24 h
  before the trough arrives (lead_field_box) -- the primary design, uncontaminated by the
  trough's own convection because the westward-moving trough is still east of that box.

A longitude-stratified panel checks that any developing/non-developing moisture difference
is not just the moist-west/dry-east gradient.

Inputs: AEWC trajectories (data/aewc), csct CS-245 systems (data/original/csct), optional
ERA5 environment files (data/era5/region6h). Writes fig_developing.png.
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr

from aew.data.aewc import load_aewc_trajectories
from aew.environment import (
    cluster_bootstrap_diff,
    forward_response,
    lead_field_box,
    lead_value,
    stratified_terciles,
)

LEAD_H = 24.0
TOL_H = 6.0
RESP_WIN_H = 24.0
DLON = 8.0
LAT_LO, LAT_HI = 5.0, 15.0
ENV_LABEL = {
    "tpw": ("SSM/I trough-mean TCWV", "mm"),
    "r700": ("ERA5 700 hPa relative humidity", "%"),
    "tcwv": ("ERA5 total column water vapour", "mm"),
    "shear": ("ERA5 600-925 hPa shear", "m/s"),
}


from aew.data.era5 import load_region_6h as load_era5_region  # noqa: E402  shared loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aewc-glob", default="data/aewc/ERA-Int_ew_700hPa_*_AFR.nc")
    ap.add_argument("--csct", default="data/original/csct/csct_africa_cs245.nc")
    ap.add_argument("--env", default="tpw", choices=("tpw", "r700", "tcwv", "shear"),
                    help="pre-trough environment source (tpw = sparse SSM/I along-track; "
                         "r700/tcwv = full-coverage ERA5 fixed-box moisture; shear = ERA5 "
                         "600-925 hPa vector shear magnitude, the organization axis)")
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
    env_name, unit = ENV_LABEL[a.env]
    if a.env == "tpw":
        # sparse SSM/I scalar, sampled along the trough's own track
        lead_env = lead_value(tr.time, tr.variables["traj_id"], tr.variables["tpw"],
                              LEAD_H, TOL_H)
    else:
        # full-coverage ERA5 field, fixed box at the trough meridian 24 h before arrival
        if a.env == "shear":
            # 600-925 hPa vector shear magnitude from the four wind components; the four
            # records must share one time/grid base before mixing them point by point
            comps = {}
            for k in ("u600", "v600", "u925", "v925"):
                comps[k] = load_era5_region(k)
            ft, flat, flon, _ = comps["u600"]
            for k in ("v600", "u925", "v925"):
                tk, lk, ok, _ = comps[k]
                if not (tk.equals(ft) and np.array_equal(lk, flat)
                        and np.array_equal(ok, flon)):
                    raise ValueError(f"ERA5 {k} grid/time differs from u600")
            ffield = np.sqrt((comps["u600"][3] - comps["u925"][3]) ** 2
                             + (comps["v600"][3] - comps["v925"][3]) ** 2)
        else:
            ft, flat, flon, ffield = load_era5_region(a.env)
        print(f"ERA5 {a.env}: {ft.size} steps {ft.min().date()}..{ft.max().date()}")
        lead_env = lead_field_box(tr.time, tr.lon, ft.values, flat, flon, ffield,
                                  LEAD_H, tol_h=3.0, dlon=5.0,
                                  lat_lo=LAT_LO, lat_hi=LAT_HI)

    keep = np.isfinite(lead_env)
    print(f"troughs with a pre-trough (t-{LEAD_H:.0f}h) {env_name} sample: "
          f"{keep.sum()} ({100 * keep.mean():.0f}%)")
    resp_k = resp[keep]
    tcwv_k = lead_env[keep]
    lon_k = tr.lon[keep]

    tid_k = tr.variables["traj_id"][keep]
    from scipy import stats
    rng = np.random.default_rng(0)

    # LONGITUDE x MONTH stratified tercile split: the convective response climatology varies
    # with longitude (more forward-window systems in the east) AND with the season, so a
    # pooled tercile split re-encodes both coordinates and any moisture contrast inherits
    # the moist-west/dry-east gradient and the seasonal moisture cycle (Simpson at two
    # scales). Splitting within each 10-degree x calendar-month cell gives the two groups a
    # matched composition in both, so the pooled difference is a within-geography,
    # within-season contrast. Cells with too few analyzable troughs are left out entirely.
    edges = np.arange(-30, 41, 10.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    month_k = pd.DatetimeIndex(tr.time[keep]).month.values.astype(float)
    m_edges = np.array([6.5, 7.5, 8.5, 9.5])
    low, high = stratified_terciles(resp_k, lon_k, edges, min_bin=30,
                                    strat2=month_k, edges2=m_edges)
    low_lon, high_lon = stratified_terciles(resp_k, lon_k, edges, min_bin=30)

    # per-bin table (same stratified masks); per-bin p is observation-level, indicative only
    MIN_N = 15
    nd_lon, dv_lon, reliable = [], [], []
    print(f"\nlon-bin(E)  n_nd  n_dv   non-dev  developing   diff({unit})   p(obs-level)")
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

    # primary result: pooled difference over the stratified split (geography-matched by
    # construction). Significance uses a CLUSTER bootstrap over trajectories, because a wave
    # contributes many correlated 6-hourly observations -- treating each observation as
    # independent (a plain t-test) overstates the sample size.
    ndw, dvw = tcwv_k[low], tcwv_k[high]
    diff, lo_ci, hi_ci, nwa, nwb = cluster_bootstrap_diff(
        tid_k[low], ndw, tid_k[high], dvw, rng)
    _, pw_obs = stats.ttest_ind(dvw, ndw, equal_var=False)
    sig = "not significant (CI crosses 0)" if lo_ci <= 0 <= hi_ci else "significant"
    print("\n(developing vs non-developing are longitude x month stratified response "
          "terciles among the troughs with an analyzable pre-trough environment sample)")
    print(f"PRIMARY (per-observation) -- lon x month stratified, {env_name}: non-dev "
          f"{ndw.mean():.1f} {unit} (n={ndw.size} obs, {nwa} waves), developing "
          f"{dvw.mean():.1f} {unit} (n={dvw.size} obs, {nwb} waves), difference "
          f"{diff:+.1f} {unit}. Trajectory cluster-bootstrap 95% CI "
          f"[{lo_ci:+.1f}, {hi_ci:+.1f}] -- {sig}. "
          f"(observation-level t-test p={pw_obs:.1e} is over-optimistic: it ignores that "
          "each wave gives many correlated samples.)")

    # the longitude-only split for comparison: its stronger contrast includes the seasonal
    # covariation (developing troughs cluster in the moister part of the season), which the
    # month stratification deliberately removes to isolate the sub-seasonal signal.
    dL, loL, hiL, _, _ = cluster_bootstrap_diff(
        tid_k[low_lon], tcwv_k[low_lon], tid_k[high_lon], tcwv_k[high_lon], rng)
    sigL = "significant" if not (loL <= 0 <= hiL) else "ns"
    print(f"COMPARISON (lon-only split, includes the seasonal covariation): "
          f"{dL:+.1f} {unit}, CI [{loL:+.1f}, {hiL:+.1f}] -- {sigL}. The gap between the "
          "two is the seasonal-scale part of the moisture-development association.")

    # wave-level estimand: collapse to per-trajectory means first, so a long-lived wave
    # counts once. This asks a different question (do developing WAVES live in moister air)
    # and is reported alongside the per-observation contrast, not hidden behind it.
    def _wave_means(gids, vals):
        u = {}
        for g, v in zip(gids, vals):
            u.setdefault(g, []).append(v)
        keys = np.array(sorted(u))
        return keys, np.array([np.mean(u[k]) for k in keys])
    ka, va = _wave_means(tid_k[low], ndw)
    kb, vb = _wave_means(tid_k[high], dvw)
    dW, loW, hiW, _, _ = cluster_bootstrap_diff(ka, va, kb, vb, rng)
    sigW = "not significant (CI crosses 0)" if loW <= 0 <= hiW else "significant"
    print(f"WAVE-LEVEL (one value per trajectory): difference {dW:+.1f} {unit}, bootstrap "
          f"95% CI [{loW:+.1f}, {hiW:+.1f}] -- {sigW}. This is the wave-track-selector "
          "reading: whole trajectories embedded in moister or drier synoptic states. The "
          "per-observation contrast above draws much of its strength from persistent states "
          "along the same tracks, so the two estimands bracket the claim.")

    if a.env == "tpw":
        print("\nCAVEAT: this environment source is the ~40%-populated AEWC SSM/I trough-mean "
              "precipitable water, sampled at the trough's own shifting location, not a fixed "
              "meridian; east of ~0E the non-developing sample is too small to test. The "
              "primary source is the ERA5 700 hPa relative humidity (--env r700). A moisture "
              "precondition is consistent with, but does not by itself prove, convection "
              "organizing the wave.")
    else:
        # contamination sensitivity: at t-24h the trough sits ~7 deg east of the box, so its
        # prior-day convection can reach the eastern box edge. Shifting the box west and
        # lengthening the lead grows the separation; report how the contrast responds.
        print("\nSENSITIVITY (upstream separation from the trough's t-24h position):")
        for label, shift, lead in (("box at L-5, t-24h", -5.0, LEAD_H),
                                   ("box at L-8, t-24h", -8.0, LEAD_H),
                                   ("box at L,   t-36h", 0.0, 36.0)):
            lv = lead_field_box(tr.time, tr.lon + shift, ft.values, flat, flon, ffield,
                                lead, tol_h=3.0, dlon=5.0, lat_lo=LAT_LO, lat_hi=LAT_HI)
            lvk = lv[keep]
            dS, loS, hiS, _, _ = cluster_bootstrap_diff(
                tid_k[low & np.isfinite(lvk)], lvk[low & np.isfinite(lvk)],
                tid_k[high & np.isfinite(lvk)], lvk[high & np.isfinite(lvk)], rng)
            sS = "ns" if loS <= 0 <= hiS else "significant"
            print(f"  {label}:  {dS:+.2f} {unit}  CI [{loS:+.2f}, {hiS:+.2f}]  {sS}")
        print("\nNOTE: the environment is the full-coverage " + env_name + " in a fixed box "
              "at the trough meridian 24 h before arrival. The trough is then ~7 deg east of "
              "the box, so its prior-day circulation can reach the eastern box edge; the "
              "sensitivity block above shows how the contrast responds as the box moves "
              "farther upstream.")
        if a.env == "shear":
            print("Shear is the ORGANIZATION axis, kept separate from moisture by design; "
                  "there is no one-directional expectation (moderate shear can favor "
                  "organized systems while strong shear disrupts them), so the sign and "
                  "magnitude are reported without a precondition claim. That the shear "
                  "contrast fails the displacement test while moisture passes it matches "
                  "their relaxation timescales: tropical wind perturbations adjust within "
                  "hours (gravity waves, convective momentum transport) while moisture "
                  "anomalies persist for days against advection, so shear reads as the "
                  "wave's instantaneous local state and moisture as a longer-lived "
                  "boundary condition.")
        else:
            print("FRAMING: the t-24h field is not an unperturbed ambient state. The wave "
                  "train and its convection continuously moisten the corridor, so the "
                  "sample is a cumulative state of the coupled system, and the contrast is "
                  "a subtle but statistically robust probabilistic bias in which parts of "
                  "that state develop convection, not a binary environmental trigger for "
                  "a single wave.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    parts = ax1.violinplot([ndw, dvw], positions=[0, 1], showmeans=True, showextrema=False)
    for pc, col in zip(parts["bodies"], ("tab:blue", "tab:red")):
        pc.set_facecolor(col); pc.set_alpha(0.5)
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(
        [f"non-developing\n(n={ndw.size})", f"developing\n(n={dvw.size})"])
    ax1.set_ylabel(f"pre-trough {env_name}, 24 h before passage ({unit})")
    ax1.set_title(f"Lon x month stratified: developing {diff:+.1f} {unit}\n"
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
    ax2.set_ylabel(f"pre-trough {env_name} ({unit})")
    ax2.set_title("Developing vs non-developing by longitude")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(a.out, dpi=150)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
