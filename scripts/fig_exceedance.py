#!/usr/bin/env python
"""Exceedance view of the Lagrangian moisture-supply contrast (supplementary figure).

The magnitude paragraph argues that a small mean shift in inflow moisture converts into a
substantial change in the odds of crossing the moist threshold, because deep convection
responds nonlinearly to free-tropospheric humidity. This figure shows that conversion
directly: the cumulative distributions of the -72 h along-inflow 700 hPa relative
humidity for developing and non-developing troughs, with the exceedance probabilities at
the pooled 70th percentile annotated. The distributions are built with the FROZEN
baseline (identical constants to the development analysis) on any season set; the pooled
25-season record is the default.
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr

from aew.data.aewc import load_aewc_trajectories
from aew.environment import cluster_bootstrap_diff, forward_response, stratified_terciles
from aew.trajectory import Gridded, back_trajectories
from validate_heldout import load_era5_years, parse_years

LAT_LO, LAT_HI = 5.0, 15.0
LEAD_H = 24.0
BACK_H = 48.0
RESP_WIN_H = 24.0
DLON = 8.0
SEED_DLON = (-4.0, 0.0, 4.0)
SEED_LATS = (7.0, 10.0, 13.0)
EDGES = np.arange(-30, 41, 10.0)
M_EDGES = np.array([6.5, 7.5, 8.5, 9.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="1983-2007")
    ap.add_argument("--csct", default="data/original/csct/csct_africa_cs245.nc")
    ap.add_argument("--pctl", type=float, default=70.0,
                    help="exceedance threshold as a pooled percentile")
    ap.add_argument("--out", default="fig_exceedance.png")
    a = ap.parse_args()
    years = parse_years(a.years)
    print(f"exceedance figure on {len(years)} seasons: {years[0]}..{years[-1]}")

    aewc_paths = [f"data/aewc/ERA-Int_ew_700hPa_{y}_AFR.nc" for y in years]
    tr = (load_aewc_trajectories(aewc_paths)
          .filter_region(min_lat=5, max_lat=20, min_lon=-30, max_lon=40)
          .filter_months([7, 8, 9]))
    cs = xr.open_dataset(a.csct)
    cst_all = pd.DatetimeIndex(cs["time"].values)
    csx = np.asarray(cs["lon"].values, float)
    csy = np.asarray(cs["lat"].values, float)
    cs.close()
    inyr = np.isin(cst_all.year, years)
    cst, csx, csy = cst_all[inyr].values, csx[inyr], csy[inyr]

    resp = forward_response(tr.time, tr.lon, cst, csx, csy, RESP_WIN_H, DLON,
                            LAT_LO, LAT_HI)
    month = pd.DatetimeIndex(tr.time).month.values.astype(float)
    low, high = stratified_terciles(resp, tr.lon, EDGES, min_bin=30,
                                    strat2=month, edges2=M_EDGES)
    sel_idx = np.where(low | high)[0]
    n_case = sel_idx.size

    tw, wlat, wlon, uu = load_era5_years("u700", years)
    _, _, _, vv = load_era5_years("v700", years)
    tr7, rlat, rlon, rfield = load_era5_years("r700", years)
    u = Gridded(tw.values, wlat, wlon, uu)
    v = Gridded(tw.values, wlat, wlon, vv)
    rh = Gridded(tr7.values, rlat, rlon, rfield)

    seed_time = (tr.time[sel_idx].astype("datetime64[ns]")
                 - np.timedelta64(int(LEAD_H * 3600), "s"))
    gd, gl = np.meshgrid(SEED_DLON, SEED_LATS)
    npar = gd.size
    seeds_t = np.repeat(seed_time, npar)
    seeds_lon = (tr.lon[sel_idx][:, None] + gd.ravel()[None, :]).ravel()
    seeds_lat = np.broadcast_to(gl.ravel()[None, :], (n_case, npar)).ravel().copy()
    print(f"integrating {seeds_t.size} parcels ...")
    elapsed, plat, plon = back_trajectories(u, v, seeds_t, seeds_lat, seeds_lon,
                                            hours=BACK_H, dt_hours=1.0)
    k = int(round(BACK_H / (elapsed[1] - elapsed[0])))
    t_abs = (seeds_t.astype("datetime64[ns]").astype("int64") / 3.6e12) - BACK_H
    rh72 = np.nanmean(rh.sample(t_abs, plat[k], plon[k]).reshape(n_case, npar), axis=1)

    low_s = low[sel_idx]
    high_s = high[sel_idx]
    ok = np.isfinite(rh72)
    nd = np.sort(rh72[low_s & ok])
    dv = np.sort(rh72[high_s & ok])
    thr = np.nanpercentile(rh72[ok], a.pctl)
    p_nd = float((nd > thr).mean())
    p_dv = float((dv > thr).mean())
    print(f"threshold (pooled {a.pctl:.0f}th pctl): {thr:.1f}%")
    print(f"P(exceed): non-developing {100 * p_nd:.1f}%  developing {100 * p_dv:.1f}%  "
          f"(ratio {p_dv / p_nd:.2f})")

    # cluster-bootstrap interval on the exceedance-probability DIFFERENCE (wave unit,
    # same as elsewhere); the ratio is reported as a point value only
    gids = tr.variables["traj_id"][sel_idx]
    rng = np.random.default_rng(0)
    exc_nd = (rh72 > thr).astype(float)
    d, lo_ci, hi_ci, _, _ = cluster_bootstrap_diff(
        gids[low_s & ok], exc_nd[low_s & ok], gids[high_s & ok], exc_nd[high_s & ok], rng)
    print(f"exceedance-probability difference: {100 * d:+.1f} points "
          f"[{100 * lo_ci:+.1f}, {100 * hi_ci:+.1f}]  (this script's bootstrap stream)")

    # one-number-one-source: annotate the CANONICAL interval from the deposit table when
    # it is available, so the in-figure numbers match the caption and the text (this
    # script's own bootstrap stream differs from the driver's in the 2nd decimal)
    try:
        import pandas as _pd
        _row = _pd.read_csv("deposit/canonical_numbers.csv").query(
            "tier == 'pooled' and statistic == 'exceedance_p70' and level == 700 "
            "and time_rel_h == -72").iloc[0]
        d, lo_ci, hi_ci = _row["diff"] / 100, _row["ci_lo"] / 100, _row["ci_hi"] / 100
        print(f"annotating the canonical interval: {100 * d:+.1f} points "
              f"[{100 * lo_ci:+.1f}, {100 * hi_ci:+.1f}] (deposit/canonical_numbers.csv)")
    except Exception:
        print("deposit/canonical_numbers.csv not available; annotating this script's "
              "bootstrap interval")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(nd, np.linspace(0, 1, nd.size), color="tab:blue",
            label=f"MCS-quiet (n={nd.size})")
    ax.plot(dv, np.linspace(0, 1, dv.size), color="tab:red",
            label=f"MCS-active (n={dv.size})")
    ax.axvline(thr, color="k", lw=1, ls="--")
    ax.text(thr - 0.7, 0.42, f"pooled {a.pctl:.0f}th pctl ({thr:.1f}%)", fontsize=8,
            rotation=90, ha="right", va="bottom")
    ax.axhline(1 - p_nd, color="tab:blue", lw=0.8, ls=":")
    ax.axhline(1 - p_dv, color="tab:red", lw=0.8, ls=":")
    ax.text(0.02, 0.92,
            f"P(exceed): {100 * p_dv:.1f}% vs {100 * p_nd:.1f}%  "
            f"(ratio {p_dv / p_nd:.2f})\n"
            f"difference {100 * d:+.1f} points [{100 * lo_ci:+.1f}, {100 * hi_ci:+.1f}]",
            transform=ax.transAxes, fontsize=9, va="top")
    ax.set_xlabel("700 hPa RH along the inflow, 72 h prior to trough passage (%)")
    ax.set_ylabel("cumulative fraction of troughs")
    ax.set_title(f"Inflow-moisture distributions, {years[0]}-{years[-1]}")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
