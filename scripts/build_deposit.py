#!/usr/bin/env python
"""One-source deposit driver: every reported statistic from one code path and one seed.

The manuscript's numbers policy is one number, one source. The frozen harness
(validate_heldout.py) and the budget script (fig_moisture_budget.py) share their
estimators but draw separate bootstrap streams, so their intervals differ in the second
decimal for shared statistics. This driver recomputes the full statistics family once,
with a single code path and a documented random-stream layout, and the manuscript quotes
only this table. It also emits the per-trough, per-case, and per-parcel tables for the
dataset deposit.

Tiers: the analysis was designed on 2000-2004 (dev), replicated frozen on the 20
held-out seasons (1983-1999 and 2005-2007), and pooled over 1983-2007 (primary). Every
constant is the frozen baseline: forward response in a trough-relative box (half-width
8 deg, 5-15 N, 24 h forward), terciles within 10-degree longitude x calendar-month cells
(minimum 30 per cell), 9 parcels per trough integrated 48 h backward at the analysis
level through 0.5-degree ERA5 winds, and the wave-cluster bootstrap. Class labels follow
the manuscript terminology: MCS-active (upper tercile) and MCS-quiet (lower tercile).

Clock convention: the environment sample sits 24 h before trough passage and the
trajectories extend 48 h back from it, so trajectory-elapsed hours 0..48 are
passage-relative -24..-72 h. The table's time_rel_h column is passage-relative.

Random-stream layout (what makes any single number reproducible): each (tier, level)
block consumes a fresh generator seeded with (seed, tier_index, level), and each tier's
column-and-shear block (TCWV, 600-925 hPa shear) one seeded with (seed, tier_index, 1),
with the statistics computed in the fixed order of this file. Partial invocations
(--tiers, --levels) therefore reproduce the same numbers as the full run.

Outputs under --outdir:
  canonical_numbers.csv        every reported statistic, all tiers
  troughs_<tier>.csv           every JAS corridor trough observation with its response
                               and class label
  cases_<tier>_<level>.csv     per selected trough: box environment, along-inflow RH,
                               vapor/temperature decomposition, theta-e, origin
                               fractions, antecedent convection
  parcels_<tier>_<level>.csv   per parcel: seed and origin positions, origin sector
"""

import argparse
import gc
import os
import time as _time

import numpy as np
import pandas as pd
import xarray as xr

from aew.data.aewc import load_aewc_trajectories
from aew.data.era5 import load_region_6h
from aew.environment import (
    cluster_bootstrap_diff,
    forward_response,
    lead_field_box,
    stratified_terciles,
)
from aew.thermo import saturation_vapor_pressure, theta_e
from aew.trajectory import Gridded, back_trajectories, classify_origin
from validate_heldout import parse_years

# the frozen baseline constants (identical to validate_heldout.py)
LAT_LO, LAT_HI = 5.0, 15.0
LEAD_H = 24.0
BACK_H = 48.0
RESP_WIN_H = 24.0
DLON = 8.0
BOX_DLON = 5.0
TOL_H = 3.0
SEED_DLON = (-4.0, 0.0, 4.0)
SEED_LATS = (7.0, 10.0, 13.0)
EDGES = np.arange(-30, 41, 10.0)
M_EDGES = np.array([6.5, 7.5, 8.5, 9.5])
ELAPSED_H = (0.0, 12.0, 24.0, 36.0, 48.0)   # passage-relative -24 .. -72 h
DECOMP_H = (0.0, 24.0, 48.0)
EPOCHS = ((1983, 1993), (1994, 2007))
EXC_PCTL = 70.0

TIERS = {"dev": "2000-2004", "heldout": "1983-1999,2005-2007", "pooled": "1983-2007"}
TIER_ORDER = ("dev", "heldout", "pooled")


def wave_means(gid, vals):
    """Collapse observations to one mean per wave (the wave-level estimand input)."""
    acc = {}
    for g, v in zip(gid, vals):
        acc.setdefault(g, []).append(v)
    keys = np.array(sorted(acc))
    return keys, np.array([np.mean(acc[k]) for k in keys])


def theta_e_from_mix(T, mix_gkg, p_hpa):
    """Theta-e from temperature and mixing ratio (the budget's perturbation form)."""
    e = p_hpa * mix_gkg / (622.0 + mix_gkg)
    rh = 100.0 * e / saturation_vapor_pressure(T)
    return theta_e(T, rh, p_hpa)


def mixing_ratio(rh_pct, T, p_hpa):
    e = (rh_pct / 100.0) * saturation_vapor_pressure(T)
    return 622.0 * e / (p_hpa - e)


class Deposit:
    """Accumulates canonical-number rows and prints each as it lands."""

    def __init__(self):
        self.rows = []

    def contrast(self, tier, level, stat, time_rel, unit, gq, vq, ga, va, rng,
                 scale=1.0, note=""):
        d, lo, hi, na, nb = cluster_bootstrap_diff(gq, vq, ga, va, rng)
        row = dict(tier=tier, level=level, statistic=stat, time_rel_h=time_rel,
                   unit=unit, kind="contrast",
                   n_quiet=int(vq.size), n_active=int(va.size),
                   clusters_quiet=na, clusters_active=nb,
                   mean_quiet=float(np.mean(vq)) * scale,
                   mean_active=float(np.mean(va)) * scale,
                   diff=d * scale, ci_lo=lo * scale, ci_hi=hi * scale,
                   significant=bool(not (lo <= 0.0 <= hi)), note=note)
        self.rows.append(row)
        print(f"  {tier:7s} {str(level):4s} {stat:34s} {row['diff']:+8.3f} "
              f"[{row['ci_lo']:+8.3f}, {row['ci_hi']:+8.3f}] {unit:7s} "
              f"{'sig' if row['significant'] else 'ns'}", flush=True)
        return row

    def point(self, tier, level, stat, time_rel, unit, value, note=""):
        row = dict(tier=tier, level=level, statistic=stat, time_rel_h=time_rel,
                   unit=unit, kind="point",
                   n_quiet=None, n_active=None, clusters_quiet=None,
                   clusters_active=None, mean_quiet=None, mean_active=None,
                   diff=float(value), ci_lo=None, ci_hi=None, significant=None,
                   note=note)
        self.rows.append(row)
        print(f"  {tier:7s} {str(level):4s} {stat:34s} {row['diff']:+8.3f} "
              f"{'':22s} {unit:7s} point", flush=True)
        return row


def load_troughs_and_systems(years, csct_path):
    aewc_paths = [f"data/aewc/ERA-Int_ew_700hPa_{y}_AFR.nc" for y in years]
    missing = [p for p in aewc_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} AEWC files missing, first: {missing[0]}")
    tr = (load_aewc_trajectories(aewc_paths)
          .filter_region(min_lat=5, max_lat=20, min_lon=-30, max_lon=40)
          .filter_months([7, 8, 9]))
    cs = xr.open_dataset(csct_path)
    cst_all = pd.DatetimeIndex(cs["time"].values)
    csx = np.asarray(cs["lon"].values, float)
    csy = np.asarray(cs["lat"].values, float)
    cs.close()
    inyr = np.isin(cst_all.year, years)
    return tr, cst_all[inyr].values, csx[inyr], csy[inyr]


def run_level(dep, tier, level, years, tr, cst, csx, csy, sel_idx, low_s, high_s,
              seed, outdir):
    """The per-level statistics family. The contrast/point call order below IS the
    random-stream order; do not reorder without bumping the documented layout."""
    tier_idx = TIER_ORDER.index(tier)
    rng = np.random.default_rng([seed, tier_idx, level])
    t0 = _time.time()
    n_case = sel_idx.size
    gids = tr.variables["traj_id"][sel_idx]
    yrs = pd.DatetimeIndex(tr.time[sel_idx]).year.values

    # ---- Eulerian family on the level's relative-humidity field ----
    rt, rlat, rlon, rfield = load_region_6h(f"r{level}", years=years)
    rt_vals = rt.values

    def box(lon_shift, lead_h):
        lv = lead_field_box(tr.time, tr.lon + lon_shift, rt_vals, rlat, rlon, rfield,
                            lead_h, tol_h=TOL_H, dlon=BOX_DLON,
                            lat_lo=LAT_LO, lat_hi=LAT_HI)[sel_idx]
        ok = np.isfinite(lv)
        return lv, ok

    box0, ok0 = box(0.0, LEAD_H)
    r_eul = dep.contrast(tier, level, "eulerian_box", -24, "%",
                         gids[low_s & ok0], box0[low_s & ok0],
                         gids[high_s & ok0], box0[high_s & ok0], rng,
                         note="fixed box at the trough meridian, 24 h before passage")
    ka, va_ = wave_means(gids[low_s & ok0], box0[low_s & ok0])
    kb, vb_ = wave_means(gids[high_s & ok0], box0[high_s & ok0])
    dep.contrast(tier, level, "eulerian_box_wave_level", -24, "%",
                 ka, va_, kb, vb_, rng, note="one mean per wave")
    for shift, lead_h, stat, trel in ((-5.0, LEAD_H, "eulerian_box_L-5", -24),
                                      (-8.0, LEAD_H, "eulerian_box_L-8", -24),
                                      (0.0, 36.0, "eulerian_box_-36h", -36),
                                      (8.0, 72.0, "eulerian_control_L+8", -72),
                                      (12.0, 72.0, "eulerian_control_L+12", -72)):
        lv, ok = box(shift, lead_h)
        dep.contrast(tier, level, stat, trel, "%",
                     gids[low_s & ok], lv[low_s & ok],
                     gids[high_s & ok], lv[high_s & ok], rng)

    # ---- trajectories (loads freed aggressively; 25-season fields are large) ----
    seed_time = (tr.time[sel_idx].astype("datetime64[ns]")
                 - np.timedelta64(int(LEAD_H * 3600), "s"))
    gd, gl = np.meshgrid(SEED_DLON, SEED_LATS)
    npar = gd.size
    seeds_t = np.repeat(seed_time, npar)
    seeds_lon = (tr.lon[sel_idx][:, None] + gd.ravel()[None, :]).ravel()
    seeds_lat = np.broadcast_to(gl.ravel()[None, :], (n_case, npar)).ravel().copy()
    seeds_h = seeds_t.astype("datetime64[ns]").astype("int64") / 3.6e12

    tu, wlat, wlon, uu = load_region_6h(f"u{level}", years=years)
    u = Gridded(tu.values, wlat, wlon, uu)
    del uu
    gc.collect()
    tv, _, _, vv = load_region_6h(f"v{level}", years=years)
    v = Gridded(tv.values, wlat, wlon, vv)
    del vv
    gc.collect()
    print(f"  integrating {seeds_t.size} parcels {BACK_H:.0f} h backward at "
          f"{level} hPa ...", flush=True)
    elapsed, plat, plon = back_trajectories(u, v, seeds_t, seeds_lat, seeds_lon,
                                            hours=BACK_H, dt_hours=1.0)
    del u, v
    gc.collect()
    dt_step = elapsed[1] - elapsed[0]

    # parcel-level samples along the inflow, then the big fields are freed
    rh = Gridded(rt_vals, rlat, rlon, rfield)
    del rfield
    gc.collect()
    rh_par, sep = {}, {}
    for eh in ELAPSED_H:
        k = int(round(eh / dt_step))
        rh_par[eh] = rh.sample(seeds_h - eh, plat[k], plon[k])
        sep[eh] = float(np.nanmean(np.hypot(plat[k] - seeds_lat, plon[k] - seeds_lon)))
    del rh
    gc.collect()
    tt_, tlat, tlon, tfield = load_region_6h(f"t{level}", years=years)
    temp = Gridded(tt_.values, tlat, tlon, tfield)
    del tfield
    gc.collect()
    t_par = {}
    for eh in ELAPSED_H:
        k = int(round(eh / dt_step))
        t_par[eh] = temp.sample(seeds_h - eh, plat[k], plon[k])
    del temp
    gc.collect()

    def case_mean(par_vals):
        return np.nanmean(par_vals.reshape(n_case, npar), axis=1)

    # ---- along-inflow relative humidity (the supply-contrast profile) ----
    rh_case = {eh: case_mean(rh_par[eh]) for eh in ELAPSED_H}
    for eh in ELAPSED_H:
        cm = rh_case[eh]
        ok = np.isfinite(cm)
        dep.contrast(tier, level, "lagrangian_rh", -(24 + int(eh)), "%",
                     gids[low_s & ok], cm[low_s & ok],
                     gids[high_s & ok], cm[high_s & ok], rng,
                     note=f"mean parcel separation {sep[eh]:.1f} deg")

    # the endpoint estimands: wave level, year block, epochs
    rh72 = rh_case[BACK_H]
    ok72 = np.isfinite(rh72)
    ka, va_ = wave_means(gids[low_s & ok72], rh72[low_s & ok72])
    kb, vb_ = wave_means(gids[high_s & ok72], rh72[high_s & ok72])
    dep.contrast(tier, level, "lagrangian_rh_wave_level", -72, "%",
                 ka, va_, kb, vb_, rng, note="one mean per wave")
    dep.contrast(tier, level, "lagrangian_rh_year_block", -72, "%",
                 yrs[low_s & ok72], rh72[low_s & ok72],
                 yrs[high_s & ok72], rh72[high_s & ok72], rng,
                 note=f"{np.unique(yrs).size} year clusters")
    spans_both = (yrs.min() <= EPOCHS[0][1]) and (yrs.max() >= EPOCHS[1][0])
    for lo_y, hi_y in (EPOCHS if spans_both else ()):
        m = (yrs >= lo_y) & (yrs <= hi_y)
        if (low_s & ok72 & m).sum() > 100 and (high_s & ok72 & m).sum() > 100:
            dep.contrast(tier, level, f"lagrangian_rh_epoch_{lo_y}_{hi_y}", -72, "%",
                         gids[low_s & ok72 & m], rh72[low_s & ok72 & m],
                         gids[high_s & ok72 & m], rh72[high_s & ok72 & m], rng)

    # exceedance of the tier's pooled 70th percentile at -72 h
    thr = float(np.nanpercentile(rh72[ok72], EXC_PCTL))
    exc = (rh72 > thr).astype(float)
    p_q = float(exc[low_s & ok72].mean())
    p_a = float(exc[high_s & ok72].mean())
    dep.contrast(tier, level, "exceedance_p70", -72, "points",
                 gids[low_s & ok72], exc[low_s & ok72],
                 gids[high_s & ok72], exc[high_s & ok72], rng, scale=100.0,
                 note=f"threshold {thr:.1f}% (pooled {EXC_PCTL:.0f}th pctl of the "
                      "selected-case inflow)")
    dep.point(tier, level, "exceedance_threshold", -72, "%", thr)
    dep.point(tier, level, "exceedance_ratio", -72, "ratio",
              p_a / p_q if p_q > 0 else np.nan,
              note=f"P(exceed) active {100 * p_a:.1f} vs quiet {100 * p_q:.1f}")

    # ---- route attribution from the parcel origins ----
    sector = classify_origin(seeds_lat, seeds_lon, plat[-1], plon[-1])
    dep.point(tier, level, "parcels_lost_fraction", -72, "frac",
              float((sector == "lost").mean()))
    sec_case = {s: (sector == s).astype(float).reshape(n_case, npar).mean(axis=1)
                for s in ("south", "north", "east", "west", "local")}
    dlat_case = np.nan_to_num(
        np.nanmean((plat[-1] - seeds_lat).reshape(n_case, npar), axis=1), nan=0.0)
    for s in ("south", "north", "east", "west", "local"):
        dep.contrast(tier, level, f"route_{s}_fraction", -72, "frac",
                     gids[low_s], sec_case[s][low_s],
                     gids[high_s], sec_case[s][high_s], rng)
    dep.contrast(tier, level, "route_displacement", -72, "deg",
                 gids[low_s], dlat_case[low_s],
                 gids[high_s], dlat_case[high_s], rng,
                 note="origin minus seed latitude, lost parcels as 0")
    if level == 850:
        # 850 hPa intersects the eastern highlands (ERA5 extrapolates below ground),
        # so the west-of-25E subset is the terrain-safe version of the route test
        west = tr.lon[sel_idx] < 25.0
        dep.contrast(tier, level, "route_south_fraction_west25", -72, "frac",
                     gids[west & low_s], sec_case["south"][west & low_s],
                     gids[west & high_s], sec_case["south"][west & high_s], rng,
                     note="terrain-safe west-of-25E subset")
        dep.contrast(tier, level, "route_displacement_west25", -72, "deg",
                     gids[west & low_s], dlat_case[west & low_s],
                     gids[west & high_s], dlat_case[west & high_s], rng,
                     note="terrain-safe west-of-25E subset")

    # ---- vapor/temperature decomposition and theta-e along the inflow ----
    mix_case, t_case, d_mix, d_T = {}, {}, {}, {}
    for eh in DECOMP_H:
        mix_case[eh] = case_mean(mixing_ratio(rh_par[eh], t_par[eh], float(level)))
        t_case[eh] = case_mean(t_par[eh])
    for eh in DECOMP_H:
        for name, cm, unit in (("mix", mix_case[eh], "g/kg"), ("T", t_case[eh], "K")):
            ok = np.isfinite(cm)
            row = dep.contrast(tier, level, f"decomp_{name}", -(24 + int(eh)), unit,
                               gids[low_s & ok], cm[low_s & ok],
                               gids[high_s & ok], cm[high_s & ok], rng)
            if name == "mix":
                d_mix[eh] = row
            else:
                d_T[eh] = row

    te_case = {}
    for eh in ELAPSED_H:
        te = theta_e(t_par[eh], rh_par[eh], float(level))
        te_case[eh] = case_mean(te)
        ok = np.isfinite(te_case[eh])
        dep.contrast(tier, level, "thetae", -(24 + int(eh)), "K",
                     gids[low_s & ok], te_case[eh][low_s & ok],
                     gids[high_s & ok], te_case[eh][high_s & ok], rng)

    # north-origin arm only (conditional diagnostic; retention differs by group)
    te48 = theta_e(t_par[BACK_H], rh_par[BACK_H], float(level))
    te_n = case_mean(np.where(sector == "north", te48, np.nan))
    okn = np.isfinite(te_n)
    dep.contrast(tier, level, "thetae_north_origin", -72, "K",
                 gids[low_s & okn], te_n[low_s & okn],
                 gids[high_s & okn], te_n[high_s & okn], rng,
                 note=f"conditional on a north-origin parcel; retained "
                      f"{int((low_s & okn).sum())}/{int((high_s & okn).sum())} of "
                      f"{int(low_s.sum())}/{int(high_s.sum())}")

    # the theta-e budget at -72 h: perturb the MCS-quiet mean state by the measured
    # vapor and temperature contrasts and compare the net against the observed theta-e
    ok = np.isfinite(mix_case[BACK_H]) & np.isfinite(t_case[BACK_H])
    T_q = float(t_case[BACK_H][low_s & ok].mean())
    r_q = float(mix_case[BACK_H][low_s & ok].mean())
    base = float(theta_e_from_mix(T_q, r_q, float(level)))
    vap = float(theta_e_from_mix(T_q, r_q + d_mix[BACK_H]["diff"], float(level))) - base
    cool = float(theta_e_from_mix(T_q + d_T[BACK_H]["diff"], r_q, float(level))) - base
    dep.point(tier, level, "thetae_budget_vapor_term", -72, "K", vap,
              note=f"from {d_mix[BACK_H]['diff']:+.3f} g/kg about the quiet mean state")
    dep.point(tier, level, "thetae_budget_cooling_term", -72, "K", cool,
              note=f"from {d_T[BACK_H]['diff']:+.3f} K about the quiet mean state")
    dep.point(tier, level, "thetae_budget_predicted", -72, "K", vap + cool,
              note="vapor plus cooling term; compare the observed thetae contrast")

    # ---- antecedent convection (canonical from the 700 hPa block only) ----
    antecedent = None
    if level == 700:
        prior_start = tr.time[sel_idx].astype("datetime64[ns]") - np.timedelta64(48, "h")
        antecedent = forward_response(prior_start, tr.lon[sel_idx], cst, csx, csy,
                                      win_h=24.0, dlon=DLON + 2.0,
                                      lat_lo=LAT_LO, lat_hi=LAT_HI)
        dep.contrast(tier, level, "antecedent_convection", -48, "systems",
                     gids[low_s], antecedent[low_s], gids[high_s], antecedent[high_s],
                     rng, note="CS-245 count near the box over t-48h..t-24h")

    # the advective attenuation factor (derived, no random stream)
    lag72 = next(r for r in dep.rows
                 if r["tier"] == tier and r["level"] == level
                 and r["statistic"] == "lagrangian_rh" and r["time_rel_h"] == -72)
    if r_eul["diff"] != 0:
        dep.point(tier, level, "attenuation_factor", None, "ratio",
                  lag72["diff"] / r_eul["diff"],
                  note="lagrangian -72 h contrast over the meridian Eulerian box")

    # ---- per-case and per-parcel tables ----
    case = pd.DataFrame({
        "case": np.arange(n_case),
        "traj_id": gids,
        "time": pd.DatetimeIndex(tr.time[sel_idx]),
        "lat": tr.lat[sel_idx],
        "lon": tr.lon[sel_idx],
        "year": yrs,
        "month": pd.DatetimeIndex(tr.time[sel_idx]).month,
        "label": np.where(high_s, "MCS-active", "MCS-quiet"),
        "box_rh_m24": box0,
    })
    for eh in ELAPSED_H:
        case[f"rh_m{24 + int(eh)}"] = rh_case[eh]
        case[f"thetae_m{24 + int(eh)}"] = te_case[eh]
    for eh in DECOMP_H:
        case[f"mix_m{24 + int(eh)}"] = mix_case[eh]
        case[f"T_m{24 + int(eh)}"] = t_case[eh]
    for s in ("south", "north", "east", "west", "local"):
        case[f"frac_{s}"] = sec_case[s]
    case["dlat_origin"] = dlat_case
    case["exceed_p70"] = np.where(ok72, exc, np.nan)
    if antecedent is not None:
        case["antecedent"] = antecedent
    case_path = os.path.join(outdir, f"cases_{tier}_{level}.csv")
    case.to_csv(case_path, index=False, float_format="%.4f")

    parcels = pd.DataFrame({
        "case": np.repeat(np.arange(n_case), npar),
        "parcel": np.tile(np.arange(npar), n_case),
        "traj_id": np.repeat(gids, npar),
        "seed_time": pd.DatetimeIndex(seeds_t),
        "seed_lat": seeds_lat,
        "seed_lon": seeds_lon,
        "origin_lat": plat[-1],
        "origin_lon": plon[-1],
        "sector": sector,
    })
    parcel_path = os.path.join(outdir, f"parcels_{tier}_{level}.csv")
    parcels.to_csv(parcel_path, index=False, float_format="%.3f")
    print(f"  wrote {case_path} and {parcel_path} "
          f"({_time.time() - t0:.0f} s for the {level} hPa block)", flush=True)


def run_extras(dep, tier, years, tr, sel_idx, low_s, high_s, seed):
    """Column water vapour and the 600-925 hPa shear control (Eulerian box family)."""
    tier_idx = TIER_ORDER.index(tier)
    rng = np.random.default_rng([seed, tier_idx, 1])
    gids = tr.variables["traj_id"][sel_idx]

    def family(name, unit, ft_vals, flat, flon, ffield):
        def box(shift):
            lv = lead_field_box(tr.time, tr.lon + shift, ft_vals, flat, flon, ffield,
                                LEAD_H, tol_h=TOL_H, dlon=BOX_DLON,
                                lat_lo=LAT_LO, lat_hi=LAT_HI)[sel_idx]
            return lv, np.isfinite(lv)
        lv0, ok0 = box(0.0)
        dep.contrast(tier, None, f"{name}_box", -24, unit,
                     gids[low_s & ok0], lv0[low_s & ok0],
                     gids[high_s & ok0], lv0[high_s & ok0], rng)
        for shift, stat in ((-5.0, f"{name}_box_L-5"), (-8.0, f"{name}_box_L-8")):
            lv, ok = box(shift)
            dep.contrast(tier, None, stat, -24, unit,
                         gids[low_s & ok], lv[low_s & ok],
                         gids[high_s & ok], lv[high_s & ok], rng)
        ka, va_ = wave_means(gids[low_s & ok0], lv0[low_s & ok0])
        kb, vb_ = wave_means(gids[high_s & ok0], lv0[high_s & ok0])
        dep.contrast(tier, None, f"{name}_wave_level", -24, unit,
                     ka, va_, kb, vb_, rng, note="one mean per wave")

    ft, flat, flon, ffield = load_region_6h("tcwv", years=years)
    family("tcwv", "mm", ft.values, flat, flon, ffield)
    del ffield
    gc.collect()

    # shear magnitude built pairwise to hold at most three season-set arrays at once
    ft, flat, flon, u6 = load_region_6h("u600", years=years)
    t2, l2, o2, u9 = load_region_6h("u925", years=years)
    if not (t2.equals(ft) and np.array_equal(l2, flat) and np.array_equal(o2, flon)):
        raise ValueError("ERA5 u925 grid/time differs from u600")
    du = u6 - u9
    del u6, u9
    gc.collect()
    t3, l3, o3, v6 = load_region_6h("v600", years=years)
    t4, l4, o4, v9 = load_region_6h("v925", years=years)
    for nm, (tk, lk, ok_) in (("v600", (t3, l3, o3)), ("v925", (t4, l4, o4))):
        if not (tk.equals(ft) and np.array_equal(lk, flat) and np.array_equal(ok_, flon)):
            raise ValueError(f"ERA5 {nm} grid/time differs from u600")
    dv = v6 - v9
    del v6, v9
    gc.collect()
    shear = np.sqrt(du * du + dv * dv)
    del du, dv
    gc.collect()
    family("shear", "m/s", ft.values, flat, flon, shear)
    del shear
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="dev,heldout,pooled",
                    help="comma-separated subset of dev,heldout,pooled")
    ap.add_argument("--levels", default="700,850",
                    help="comma-separated subset of 700,850")
    ap.add_argument("--csct", default="data/original/csct/csct_africa_cs245.nc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="deposit")
    a = ap.parse_args()
    tiers = [t.strip() for t in a.tiers.split(",") if t.strip()]
    levels = [int(x) for x in a.levels.split(",") if x.strip()]
    for t in tiers:
        if t not in TIERS:
            raise SystemExit(f"unknown tier {t!r} (choose from {TIER_ORDER})")
    os.makedirs(a.outdir, exist_ok=True)
    dep = Deposit()
    t_start = _time.time()

    for tier in TIER_ORDER:
        if tier not in tiers:
            continue
        years = parse_years(TIERS[tier])
        print(f"\n=== tier {tier}: {len(years)} seasons "
              f"{years[0]}..{years[-1]} ===", flush=True)
        tr, cst, csx, csy = load_troughs_and_systems(years, a.csct)
        resp = forward_response(tr.time, tr.lon, cst, csx, csy, RESP_WIN_H, DLON,
                                LAT_LO, LAT_HI)
        month = pd.DatetimeIndex(tr.time).month.values.astype(float)
        low, high = stratified_terciles(resp, tr.lon, EDGES, min_bin=30,
                                        strat2=month, edges2=M_EDGES)
        sel = low | high
        sel_idx = np.where(sel)[0]
        low_s = low[sel_idx]
        high_s = high[sel_idx]
        print(f"corridor troughs {len(tr)}, systems {cst.size}, split "
              f"{sel_idx.size} (quiet {low.sum()}, active {high.sum()})", flush=True)
        dep.point(tier, None, "n_corridor_troughs", None, "count", len(tr))
        dep.point(tier, None, "n_waves", None, "count",
                  np.unique(tr.variables["traj_id"]).size)
        dep.point(tier, None, "n_systems", None, "count", cst.size,
                  note="CS-245 systems in the tier's calendar years, all months")
        dep.point(tier, None, "n_systems_jas", None, "count",
                  int(np.isin(pd.DatetimeIndex(cst).month, [7, 8, 9]).sum()))
        dep.point(tier, None, "n_selected", None, "count", sel_idx.size)
        dep.point(tier, None, "n_quiet", None, "count", int(low.sum()))
        dep.point(tier, None, "n_active", None, "count", int(high.sum()))

        troughs = pd.DataFrame({
            "traj_id": tr.variables["traj_id"],
            "time": pd.DatetimeIndex(tr.time),
            "lat": tr.lat,
            "lon": tr.lon,
            "year": pd.DatetimeIndex(tr.time).year,
            "month": pd.DatetimeIndex(tr.time).month,
            "response": resp,
            "label": np.where(high, "MCS-active",
                              np.where(low, "MCS-quiet", "mid-tercile-or-thin-cell")),
        })
        tp = os.path.join(a.outdir, f"troughs_{tier}.csv")
        troughs.to_csv(tp, index=False, float_format="%.3f")
        print(f"wrote {tp}", flush=True)

        for level in levels:
            run_level(dep, tier, level, years, tr, cst, csx, csy, sel_idx,
                      low_s, high_s, a.seed, a.outdir)
        run_extras(dep, tier, years, tr, sel_idx, low_s, high_s, a.seed)

    out = os.path.join(a.outdir, "canonical_numbers.csv")
    pd.DataFrame(dep.rows).to_csv(out, index=False, float_format="%.6f")
    readme_src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "deposit_readme_template.md")
    if os.path.exists(readme_src):
        import shutil
        shutil.copyfile(readme_src, os.path.join(a.outdir, "README.md"))
    print(f"\nwrote {out} ({len(dep.rows)} rows, "
          f"{(_time.time() - t_start) / 60:.1f} min total)", flush=True)


if __name__ == "__main__":
    main()
