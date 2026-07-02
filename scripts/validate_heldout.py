#!/usr/bin/env python
"""Held-out replication of the Lagrangian moisture-supply contrast.

The analysis was designed and refined on 2000-2004 (the development sample), which raises
the adaptive-analysis question: would the same pipeline, frozen, find the same signal in
seasons it has never seen? The convection record spans 1983-2007, so 20 JAS seasons
(1983-1999 and 2005-2007) are available as held-out data. This script runs the EXACT
frozen baseline on any year set, with every constant identical to the development
analysis: forward response in a trough-relative box (half-width 8 deg, 5-15 N, 24 h),
terciles within 10-degree longitude x calendar-month cells (minimum 30 per cell), 9
parcels per trough integrated 48 h backward at 700 hPa through 0.5-degree ERA5 winds, and
the wave-cluster bootstrap. Nothing is tuned here; there are no free parameters.

Reported per run: the Eulerian box contrast at -24 h, the Lagrangian along-inflow
contrast at -72 h relative to trough passage (the headline), the wave-level estimand, and
a year-block interval (which becomes real inference at 20 clusters rather than the
5-cluster stress check of the development sample).

Development-sample reference values (2000-2004): Eulerian +0.7 (-0.1..+1.5) ns;
Lagrangian -72 h +1.82 (+0.40..+3.18); wave-level +0.5 (-0.1..+1.2) ns.
"""

import argparse
import glob as _glob
import os

import numpy as np
import pandas as pd
import xarray as xr

from aew.environment import (
    cluster_bootstrap_diff,
    forward_response,
    lead_field_box,
    stratified_terciles,
)
from aew.data.aewc import load_aewc_trajectories
from aew.trajectory import Gridded, back_trajectories

LAT_LO, LAT_HI = 5.0, 15.0
LEAD_H = 24.0
BACK_H = 48.0
RESP_WIN_H = 24.0
DLON = 8.0
SEED_DLON = (-4.0, 0.0, 4.0)
SEED_LATS = (7.0, 10.0, 13.0)
EDGES = np.arange(-30, 41, 10.0)
M_EDGES = np.array([6.5, 7.5, 8.5, 9.5])


def parse_years(spec):
    years = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            years.extend(range(int(lo), int(hi) + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def load_era5_years(var_key, years):
    ts, blocks, lat, lon = [], [], None, None
    for y in years:
        pg = f"data/era5/region6h/era5_{var_key}_{y}_6h_region.nc"
        paths = sorted(_glob.glob(pg))
        if not paths:
            raise FileNotFoundError(f"missing {pg}")
        ds = xr.open_dataset(paths[0])
        tname = "valid_time" if "valid_time" in ds.coords else "time"
        ts.append(pd.DatetimeIndex(ds[tname].values))
        lat = np.asarray(ds["latitude"].values, float)
        lon = np.asarray(ds["longitude"].values, float)
        name = [v for v in ds.data_vars if v in ("r", "u", "v")]
        blocks.append(np.asarray(ds[name[0]].squeeze().values, dtype=np.float32))
        ds.close()
    t = pd.DatetimeIndex(np.concatenate([x.values for x in ts]))
    field = np.concatenate(blocks, axis=0)
    o = np.argsort(t.values)
    t, field = t[o], field[o]
    uniq = np.concatenate([[True], t.values[1:] != t.values[:-1]])
    t, field = t[uniq], field[uniq]
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        field = field[:, ::-1, :]
    return t, lat, lon, field


def report(label, gid_a, va, gid_b, vb, rng, unit="%"):
    d, lo, hi, na, nb = cluster_bootstrap_diff(gid_a, va, gid_b, vb, rng)
    sig = "significant" if not (lo <= 0 <= hi) else "ns"
    print(f"  {label:34s} {d:+6.2f} {unit}  [{lo:+6.2f}, {hi:+6.2f}]  {sig}  "
          f"(clusters {na}/{nb})")
    return d, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="1983-1999,2005-2007",
                    help="season set, e.g. 1983-1999,2005-2007 or 2000-2004")
    ap.add_argument("--csct", default="data/original/csct/csct_africa_cs245.nc")
    a = ap.parse_args()
    years = parse_years(a.years)
    print(f"held-out run on {len(years)} seasons: {years[0]}..{years[-1]}")

    aewc_paths = [f"data/aewc/ERA-Int_ew_700hPa_{y}_AFR.nc" for y in years]
    missing = [p for p in aewc_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} AEWC files missing, first: {missing[0]}")
    tr = (load_aewc_trajectories(aewc_paths)
          .filter_region(min_lat=5, max_lat=20, min_lon=-30, max_lon=40)
          .filter_months([7, 8, 9]))
    print(f"JAS corridor troughs: {len(tr)}")

    cs = xr.open_dataset(a.csct)
    cst_all = pd.DatetimeIndex(cs["time"].values)
    csx = np.asarray(cs["lon"].values, float)
    csy = np.asarray(cs["lat"].values, float)
    cs.close()
    inyr = np.isin(cst_all.year, years)
    cst, csx, csy = cst_all[inyr].values, csx[inyr], csy[inyr]
    print(f"CS-245 systems in these seasons: {inyr.sum()}")

    resp = forward_response(tr.time, tr.lon, cst, csx, csy, RESP_WIN_H, DLON,
                            LAT_LO, LAT_HI)
    month = pd.DatetimeIndex(tr.time).month.values.astype(float)
    low, high = stratified_terciles(resp, tr.lon, EDGES, min_bin=30,
                                    strat2=month, edges2=M_EDGES)
    sel = low | high
    sel_idx = np.where(sel)[0]
    n_case = sel_idx.size
    print(f"split: {n_case} troughs (non-dev {low.sum()}, developing {high.sum()})")

    tw, wlat, wlon, uu = load_era5_years("u700", years)
    _, _, _, vv = load_era5_years("v700", years)
    tr7, rlat, rlon, rfield = load_era5_years("r700", years)
    u = Gridded(tw.values, wlat, wlon, uu)
    v = Gridded(tw.values, wlat, wlon, vv)
    rh = Gridded(tr7.values, rlat, rlon, rfield)

    gids = tr.variables["traj_id"][sel_idx]
    low_s = low[sel_idx]
    high_s = high[sel_idx]
    yrs = pd.DatetimeIndex(tr.time[sel_idx]).year.values
    rng = np.random.default_rng(0)

    print("\nFROZEN BASELINE on the held-out seasons "
          "(development-sample values in the docstring):")

    # 1. Eulerian box contrast at -24 h
    lead = lead_field_box(tr.time, tr.lon, tr7.values, rlat, rlon, rfield,
                          LEAD_H, tol_h=3.0, dlon=5.0, lat_lo=LAT_LO, lat_hi=LAT_HI)
    lv = lead[sel_idx]
    ok = np.isfinite(lv)
    report("Eulerian box (-24 h)", gids[low_s & ok], lv[low_s & ok],
           gids[high_s & ok], lv[high_s & ok], rng)

    # 2. Lagrangian -72 h contrast (the headline)
    seed_time = (tr.time[sel_idx].astype("datetime64[ns]")
                 - np.timedelta64(int(LEAD_H * 3600), "s"))
    gd, gl = np.meshgrid(SEED_DLON, SEED_LATS)
    npar = gd.size
    seeds_t = np.repeat(seed_time, npar)
    seeds_lon = (tr.lon[sel_idx][:, None] + gd.ravel()[None, :]).ravel()
    seeds_lat = np.broadcast_to(gl.ravel()[None, :], (n_case, npar)).ravel().copy()
    print(f"  integrating {seeds_t.size} parcels {BACK_H:.0f} h backward at 700 hPa ...")
    elapsed, plat, plon = back_trajectories(u, v, seeds_t, seeds_lat, seeds_lon,
                                            hours=BACK_H, dt_hours=1.0)
    k = int(round(BACK_H / (elapsed[1] - elapsed[0])))
    t_abs = (seeds_t.astype("datetime64[ns]").astype("int64") / 3.6e12) - BACK_H
    rh72 = np.nanmean(rh.sample(t_abs, plat[k], plon[k]).reshape(n_case, npar), axis=1)
    ok = np.isfinite(rh72)
    report("Lagrangian inflow (-72 h)", gids[low_s & ok], rh72[low_s & ok],
           gids[high_s & ok], rh72[high_s & ok], rng)

    # 3. wave-level estimand
    def wave_means(g, vals):
        u_ = {}
        for gg, vv_ in zip(g, vals):
            u_.setdefault(gg, []).append(vv_)
        keys = np.array(sorted(u_))
        return keys, np.array([np.mean(u_[kk]) for kk in keys])
    ka, va = wave_means(gids[low_s & ok], rh72[low_s & ok])
    kb, vb = wave_means(gids[high_s & ok], rh72[high_s & ok])
    report("wave-level (-72 h)", ka, va, kb, vb, rng)

    # 4. year-block interval (real inference at ~20 clusters)
    report(f"year-block (-72 h, {np.unique(yrs).size} clusters)",
           yrs[low_s & ok], rh72[low_s & ok], yrs[high_s & ok], rh72[high_s & ok], rng)

    # epoch split, since ISCCP-era satellite coverage differs across decades
    for lo_y, hi_y in ((1983, 1993), (1994, 2007)):
        m = (yrs >= lo_y) & (yrs <= hi_y)
        if (low_s & ok & m).sum() > 100 and (high_s & ok & m).sum() > 100:
            report(f"epoch {lo_y}-{hi_y} (-72 h)",
                   gids[low_s & ok & m], rh72[low_s & ok & m],
                   gids[high_s & ok & m], rh72[high_s & ok & m], rng)


if __name__ == "__main__":
    main()
