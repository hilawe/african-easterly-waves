#!/usr/bin/env python
"""Response-definition sensitivity for the Lagrangian moisture-supply contrast.

The developing/non-developing split rests on one construction: the count of CS-245
systems in a trough-relative box (half-width 8 deg, 5-15N) over the next 24 h, split into
terciles within longitude x month cells. This harness varies each of those choices (box
half-width 5/8/12 deg, forward window 12/24/48 h, split quantiles terciles/median
halves/outer quartiles) and recomputes the headline Lagrangian statistic under each: the
developing-minus-non-developing 700 hPa relative humidity along the tracked inflow at
-72 h relative to trough passage (48 h back along the trajectory from the -24 h sample),
with the trajectory cluster bootstrap.

The trajectories do not depend on the response definition, so all 5033 corridor troughs
are integrated once and each variant only recomputes the split and the contrast. The
baseline row must reproduce the reported +1.8 (interval +0.4 to +3.3).
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr

from aew.data.aewc import load_aewc_trajectories
from aew.data.era5 import load_region_6h
from aew.environment import cluster_bootstrap_diff, forward_response, terciles
from aew.trajectory import Gridded, back_trajectories

LAT_LO, LAT_HI = 5.0, 15.0
LEAD_H = 24.0
BACK_H = 48.0
SEED_DLON = (-4.0, 0.0, 4.0)
SEED_LATS = (7.0, 10.0, 13.0)
EDGES = np.arange(-30, 41, 10.0)
M_EDGES = np.array([6.5, 7.5, 8.5, 9.5])
MIN_BIN = 30


def stratified_quantile_split(x, lon, month, q_lo, q_hi):
    """Bottom/top quantile masks within longitude x month cells (generalizes terciles)."""
    x = np.asarray(x, dtype=float)
    low = np.zeros(x.size, dtype=bool)
    high = np.zeros(x.size, dtype=bool)
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        for mlo, mhi in zip(M_EDGES[:-1], M_EDGES[1:]):
            m = (lon >= lo) & (lon < hi) & (month >= mlo) & (month < mhi)
            if m.sum() < MIN_BIN:
                continue
            idx = np.where(m)[0]
            xt_lo, xt_hi = np.nanpercentile(x[m], [q_lo, q_hi])
            low[idx[x[m] <= xt_lo]] = True
            high[idx[x[m] >= xt_hi]] = True
    return low, high


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aewc-glob", default="data/aewc/ERA-Int_ew_700hPa_*_AFR.nc")
    ap.add_argument("--csct", default="data/original/csct/csct_africa_cs245.nc")
    a = ap.parse_args()

    tr = (load_aewc_trajectories(a.aewc_glob)
          .filter_region(min_lat=5, max_lat=20, min_lon=-30, max_lon=40)
          .filter_months([7, 8, 9]))
    n_all = len(tr)
    cs = xr.open_dataset(a.csct)
    cst = pd.DatetimeIndex(cs["time"].values).values
    csx = np.asarray(cs["lon"].values, float)
    csy = np.asarray(cs["lat"].values, float)
    cs.close()
    month = pd.DatetimeIndex(tr.time).month.values.astype(float)
    gids = tr.variables["traj_id"]

    # integrate ALL troughs once; the response definition does not touch the trajectories
    tw, wlat, wlon, uu = load_region_6h("u700")
    _, _, _, vv = load_region_6h("v700")
    u = Gridded(tw.values, wlat, wlon, uu)
    v = Gridded(tw.values, wlat, wlon, vv)
    t7, rlat, rlon, rfield = load_region_6h("r700")
    rh = Gridded(t7.values, rlat, rlon, rfield)

    seed_time = tr.time.astype("datetime64[ns]") - np.timedelta64(int(LEAD_H * 3600), "s")
    gd, gl = np.meshgrid(SEED_DLON, SEED_LATS)
    npar = gd.size
    seeds_t = np.repeat(seed_time, npar)
    seeds_lon = (tr.lon[:, None] + gd.ravel()[None, :]).ravel()
    seeds_lat = np.broadcast_to(gl.ravel()[None, :], (n_all, npar)).ravel().copy()
    print(f"integrating {seeds_t.size} parcels once for all {n_all} troughs ...")
    elapsed, plat, plon = back_trajectories(u, v, seeds_t, seeds_lat, seeds_lon,
                                            hours=BACK_H, dt_hours=1.0)
    k = int(round(BACK_H / (elapsed[1] - elapsed[0])))
    t_abs = (seeds_t.astype("datetime64[ns]").astype("int64") / 3.6e12) - BACK_H
    rh72 = np.nanmean(rh.sample(t_abs, plat[k], plon[k]).reshape(n_all, npar), axis=1)

    # variant grid: one axis varied at a time about the baseline
    variants = [
        ("baseline  (dlon 8, win 24 h, terciles)", 8.0, 24.0, (33.3, 66.7)),
        ("box       (dlon 5, win 24 h, terciles)", 5.0, 24.0, (33.3, 66.7)),
        ("box       (dlon 12, win 24 h, terciles)", 12.0, 24.0, (33.3, 66.7)),
        ("window    (dlon 8, win 12 h, terciles)", 8.0, 12.0, (33.3, 66.7)),
        ("window    (dlon 8, win 48 h, terciles)", 8.0, 48.0, (33.3, 66.7)),
        ("split     (dlon 8, win 24 h, median halves)", 8.0, 24.0, (50.0, 50.0)),
        ("split     (dlon 8, win 24 h, outer quartiles)", 8.0, 24.0, (25.0, 75.0)),
    ]

    rng = np.random.default_rng(0)
    print("\n-72 h (passage-relative) Lagrangian RH700 contrast under each response "
          "definition\n(developing minus non-developing, wave-cluster bootstrap 95% CI):\n")
    print(f"{'variant':47s} {'n_nd':>5s} {'n_dv':>5s} {'diff':>7s}  {'95% CI':>16s}  sig")
    for label, dlon, win, (q_lo, q_hi) in variants:
        resp = forward_response(tr.time, tr.lon, cst, csx, csy, win, dlon, LAT_LO, LAT_HI)
        if (q_lo, q_hi) == (50.0, 50.0):
            # median halves: strict below / at-or-above so the groups partition the cell
            low = np.zeros(resp.size, bool)
            high = np.zeros(resp.size, bool)
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                for mlo, mhi in zip(M_EDGES[:-1], M_EDGES[1:]):
                    m = (tr.lon >= lo) & (tr.lon < hi) & (month >= mlo) & (month < mhi)
                    if m.sum() < MIN_BIN:
                        continue
                    idx = np.where(m)[0]
                    med = np.nanmedian(resp[m])
                    low[idx[resp[m] < med]] = True
                    high[idx[resp[m] >= med]] = True
        else:
            low, high = stratified_quantile_split(resp, tr.lon, month, q_lo, q_hi)
        ok = np.isfinite(rh72)
        d, lo_ci, hi_ci, _, _ = cluster_bootstrap_diff(
            gids[low & ok], rh72[low & ok], gids[high & ok], rh72[high & ok], rng)
        sig = "significant" if not (lo_ci <= 0 <= hi_ci) else "ns"
        print(f"{label:47s} {int(low.sum()):5d} {int(high.sum()):5d} {d:+7.2f}  "
              f"[{lo_ci:+6.2f}, {hi_ci:+6.2f}]  {sig}")

    # the exceedance-probability bridge for the magnitude paragraph: how much does the
    # mean shift move the odds of moist-tail inflow? Baseline split, pooled inflow RH.
    resp = forward_response(tr.time, tr.lon, cst, csx, csy, 24.0, 8.0, LAT_LO, LAT_HI)
    low, high = stratified_quantile_split(resp, tr.lon, month, 33.3, 66.7)
    ok = np.isfinite(rh72)
    pooled70 = np.nanpercentile(rh72[(low | high) & ok], 70.0)
    p_nd = float((rh72[low & ok] > pooled70).mean())
    p_dv = float((rh72[high & ok] > pooled70).mean())
    print(f"\nEXCEEDANCE BRIDGE: P(inflow RH at -72 h above the pooled 70th percentile, "
          f"{pooled70:.1f}%):\n  non-developing {100 * p_nd:.1f}%  developing "
          f"{100 * p_dv:.1f}%  (ratio {p_dv / p_nd:.2f})")


if __name__ == "__main__":
    main()
