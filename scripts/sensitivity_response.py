#!/usr/bin/env python
"""Response-definition sensitivity for the Lagrangian moisture-supply contrast.

The MCS-active/MCS-quiet split rests on one construction: the count of CS-245 detections
in a trough-relative box (half-width 8 deg, 5-15 N) over the next 24 h, split into
terciles within longitude x calendar-month cells. This harness varies each of those
choices (box half-width 5/8/12 deg, forward window 12/24/48 h, split quantiles
terciles/median halves/outer quartiles) and recomputes the headline Lagrangian statistic
under each: the MCS-active minus MCS-quiet 700 hPa relative humidity along the tracked
inflow at -72 h relative to trough passage (48 h back along the trajectory from the -24 h
sample), with the wave-cluster bootstrap.

This harness runs on the REPAIRED pipeline (REPAIR_SPEC.md), so it shares build_deposit's
estimators exactly: complete-window eligibility (R3) before the response and split,
6-hourly surface-pressure terrain-validity masks (R2) on the humidity and wind fields,
the valid-parcel aggregation rule (mean over valid parcels, missing below five), and the
exact 1/3, 2/3 linear quantiles for the tercile boundary. The trajectories do not depend
on the response definition, so the eligible corridor troughs are integrated once and each
variant only recomputes the split and the contrast. The baseline row (8 deg, 24 h,
terciles) reproduces the canonical lagrangian_rh -72 h point estimate; only its interval
differs, because this harness draws a separate bootstrap stream.
"""

import argparse
import os

import numpy as np
import pandas as pd

from aew.data.aewc import load_aewc_trajectories
from aew.data.era5 import load_region_6h
from aew.environment import (cluster_bootstrap_diff, complete_window_mask,
                             forward_response)
from aew.terrain import DELTA_HPA, mask_level_inplace
from aew.trajectory import Gridded, aggregate_parcels, back_trajectories
from validate_heldout import parse_years

LAT_LO, LAT_HI = 5.0, 15.0
LEAD_H = 24.0
BACK_H = 48.0
SEED_DLON = (-4.0, 0.0, 4.0)
SEED_LATS = (7.0, 10.0, 13.0)
EDGES = np.arange(-30, 41, 10.0)
M_EDGES = np.array([6.5, 7.5, 8.5, 9.5])
MIN_BIN = 30
LEVEL = 700.0

# one axis varied at a time about the baseline; q as fractions (R3 exact quantiles)
VARIANTS = [
    ("baseline", 8.0, 24.0, (1 / 3, 2 / 3)),
    ("box_dlon5", 5.0, 24.0, (1 / 3, 2 / 3)),
    ("box_dlon12", 12.0, 24.0, (1 / 3, 2 / 3)),
    ("window_12h", 8.0, 12.0, (1 / 3, 2 / 3)),
    ("window_48h", 8.0, 48.0, (1 / 3, 2 / 3)),
    ("split_median", 8.0, 24.0, (0.5, 0.5)),
    ("split_quartiles", 8.0, 24.0, (0.25, 0.75)),
]


def stratified_quantile_split(x, lon, month, q_lo, q_hi):
    """Bottom/top quantile masks within longitude x month cells, matching the canonical
    tercile convention (aew.environment.terciles): exact linear quantiles, inclusive
    boundaries (x <= lo_t, x >= hi_t). When q_lo == q_hi (median halves) the cell is
    partitioned strictly (x < med, x >= med) so the groups do not overlap."""
    x = np.asarray(x, dtype=float)
    low = np.zeros(x.size, dtype=bool)
    high = np.zeros(x.size, dtype=bool)
    partition = q_lo == q_hi
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        for mlo, mhi in zip(M_EDGES[:-1], M_EDGES[1:]):
            m = (lon >= lo) & (lon < hi) & (month >= mlo) & (month < mhi)
            if m.sum() < MIN_BIN:
                continue
            idx = np.where(m)[0]
            finite = np.isfinite(x[m])
            if not finite.any():
                continue
            xt_lo, xt_hi = np.nanquantile(x[m][finite], [q_lo, q_hi])
            xm = x[m]
            if partition:
                low[idx[np.isfinite(xm) & (xm < xt_lo)]] = True
                high[idx[np.isfinite(xm) & (xm >= xt_hi)]] = True
            else:
                low[idx[np.isfinite(xm) & (xm <= xt_lo)]] = True
                high[idx[np.isfinite(xm) & (xm >= xt_hi)]] = True
    return low, high


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aewc-glob", default="data/aewc/ERA-Int_ew_700hPa_*_AFR.nc")
    ap.add_argument("--csct", default="data/original/csct/csct_africa_cs245.nc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="deposit")
    a = ap.parse_args()

    years = parse_years("1983-2007")
    tr = (load_aewc_trajectories(a.aewc_glob)
          .filter_region(min_lat=5, max_lat=20, min_lon=-30, max_lon=40)
          .filter_months([7, 8, 9]))
    # R3: complete-window eligibility BEFORE the response and the split
    tr = tr.filter(complete_window_mask(tr.time))
    n_all = len(tr)

    import xarray as xr
    cs = xr.open_dataset(a.csct)
    cst = pd.DatetimeIndex(cs["time"].values).values
    csx = np.asarray(cs["lon"].values, float)
    csy = np.asarray(cs["lat"].values, float)
    cs.close()
    inyr = np.isin(pd.DatetimeIndex(cst).year, years)
    cst, csx, csy = cst[inyr], csx[inyr], csy[inyr]
    month = pd.DatetimeIndex(tr.time).month.values.astype(float)
    gids = tr.variables["traj_id"]

    # R2 surface pressure for the terrain masks, shared by the RH and wind fields
    spt, splat, splon, spf = load_region_6h("sp", years=years)

    # integrate ALL eligible troughs once; the response definition never touches them
    tw, wlat, wlon, uu = load_region_6h("u700", years=years)
    u = Gridded(tw.values, wlat, wlon, mask_level_inplace(uu, spf, LEVEL, DELTA_HPA))
    del uu
    tv, _, _, vv = load_region_6h("v700", years=years)
    v = Gridded(tv.values, wlat, wlon, mask_level_inplace(vv, spf, LEVEL, DELTA_HPA))
    del vv
    t7, rlat, rlon, rfield = load_region_6h("r700", years=years)
    rh = Gridded(t7.values, rlat, rlon, mask_level_inplace(rfield, spf, LEVEL, DELTA_HPA))

    seed_time = tr.time.astype("datetime64[ns]") - np.timedelta64(int(LEAD_H * 3600), "s")
    gd, gl = np.meshgrid(SEED_DLON, SEED_LATS)
    npar = gd.size
    seeds_t = np.repeat(seed_time, npar)
    seeds_lon = (tr.lon[:, None] + gd.ravel()[None, :]).ravel()
    seeds_lat = np.broadcast_to(gl.ravel()[None, :], (n_all, npar)).ravel().copy()
    print(f"integrating {seeds_t.size} parcels once for all {n_all} eligible troughs ...",
          flush=True)
    elapsed, plat, plon = back_trajectories(u, v, seeds_t, seeds_lat, seeds_lon,
                                            hours=BACK_H, dt_hours=1.0)
    del u, v
    k = int(round(BACK_H / (elapsed[1] - elapsed[0])))
    t_abs = (seeds_t.astype("datetime64[ns]").astype("int64") / 3.6e12) - BACK_H
    # R2 valid-parcel rule: mean over valid parcels, missing below five
    rh72, _ = aggregate_parcels(rh.sample(t_abs, plat[k], plon[k]), n_all, npar)
    del rh
    ok = np.isfinite(rh72)

    rng = np.random.default_rng(a.seed)
    rows = []
    print("\n-72 h Lagrangian RH700 contrast under each response definition "
          "(repaired sample)\n(MCS-active minus MCS-quiet, wave-cluster bootstrap "
          "95% CI):\n")
    print(f"{'variant':17s} {'dlon':>5s} {'win':>4s} {'split':>10s} "
          f"{'n_q':>5s} {'n_a':>5s} {'diff':>7s}  {'95% CI':>16s}  sig")
    for label, dlon, win, (q_lo, q_hi) in VARIANTS:
        resp = forward_response(tr.time, tr.lon, cst, csx, csy, win, dlon,
                                LAT_LO, LAT_HI)
        low, high = stratified_quantile_split(resp, tr.lon, month, q_lo, q_hi)
        d, lo_ci, hi_ci, na, nb = cluster_bootstrap_diff(
            gids[low & ok], rh72[low & ok], gids[high & ok], rh72[high & ok], rng)
        sig = not (lo_ci <= 0 <= hi_ci)
        split = ("terciles" if (q_lo, q_hi) == (1 / 3, 2 / 3)
                 else "median" if q_lo == q_hi else "quartiles")
        rows.append(dict(variant=label, dlon=dlon, win_h=win, split=split,
                         n_quiet=int((low & ok).sum()), n_active=int((high & ok).sum()),
                         diff=d, ci_lo=lo_ci, ci_hi=hi_ci, significant=bool(sig)))
        print(f"{label:17s} {dlon:5.0f} {win:4.0f} {split:>10s} "
              f"{int((low & ok).sum()):5d} {int((high & ok).sum()):5d} {d:+7.2f}  "
              f"[{lo_ci:+6.2f}, {hi_ci:+6.2f}]  {'sig' if sig else 'ns'}", flush=True)

    out = os.path.join(a.outdir, "response_sensitivity.csv")
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6f")
    npos = sum(r["diff"] > 0 for r in rows)
    nsig = sum(r["significant"] for r in rows)
    print(f"\n{npos}/{len(rows)} variants positive, {nsig}/{len(rows)} significant; "
          f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
