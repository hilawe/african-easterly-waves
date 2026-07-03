#!/usr/bin/env python
"""Lagrangian moisture-source attribution for the developing/non-developing contrast.

The developing/non-developing composite shows developing troughs in moister air 24 h
before arrival. This asks where that moisture came from. For every trough in the
longitude x month stratified split (matched geography AND season), 9 parcels are seeded
in the t-24h box
(3 longitudes x 3 latitudes) and integrated 48 h BACKWARD at 700 hPa through the 6-hourly
ERA5 wind. Three diagnostics separate the two candidate sources, each contrasted
developing vs non-developing with the trajectory cluster bootstrap:

1. Origin sector at t-72h (south / north / east / west / local). A higher south-origin
   fraction for developing troughs is the monsoonal-advection signal.
2. Along-track 700 hPa relative humidity at 0, 12, 24, 36, 48 h before the seed. A
   contrast already present at -48 h means the parcels were moist before approaching the
   box (advection); a contrast that only opens near the box means local moistening.
3. Antecedent convection, the CS-245 count near the box during the PRIOR day
   (t-48h..t-24h). A higher count for developing troughs supports local evaporation from
   antecedent rainfall as a contributor.

Isobaric limitation (see aew.trajectory): parcels stay on 700 hPa, so monsoon inflow that
ascends from lower levels is seen only where it has already reached 700 hPa. Winds prefer
the regional 0.5-degree files and fall back to the global 1.5-degree band with a printed
note. Writes fig_moisture_budget.png.
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr

from aew.data.aewc import load_aewc_trajectories
from aew.data.era5 import load_region_6h
from aew.environment import cluster_bootstrap_diff, forward_response, stratified_terciles
from aew.thermo import theta_e
from aew.trajectory import Gridded, back_trajectories, classify_origin

LEAD_H = 24.0
RESP_WIN_H = 24.0
DLON = 8.0
LAT_LO, LAT_HI = 5.0, 15.0
BACK_H = 48.0
SEED_DLON = (-4.0, 0.0, 4.0)
SEED_LATS = (7.0, 10.0, 13.0)
RH_ELAPSED = (0.0, 12.0, 24.0, 36.0, 48.0)


def load_winds(level, years=None):
    uk, vk = f"u{level}", f"v{level}"
    try:
        tu, lat, lon, uu = load_region_6h(uk, years=years)
        tv, latv, lonv, vv = load_region_6h(vk, years=years)
        src = "regional 0.5 deg"
    except FileNotFoundError:
        tu, lat, lon, uu = load_region_6h(
            uk, f"data/era5/global6h/era5_{uk}_*_6h_global.nc", years=years)
        tv, latv, lonv, vv = load_region_6h(
            vk, f"data/era5/global6h/era5_{vk}_*_6h_global.nc", years=years)
        src = "global 1.5 deg fallback"
    if not (tu.equals(tv) and np.array_equal(lat, latv) and np.array_equal(lon, lonv)):
        raise ValueError(f"{uk} and {vk} do not share the same time/lat/lon grid")
    return tu, lat, lon, uu, vv, src


def group_stats(vals, gids, low, high, rng, unit, name):
    d, lo, hi, na, nb = cluster_bootstrap_diff(gids[low], vals[low], gids[high], vals[high],
                                               rng)
    sig = "significant" if not (lo <= 0 <= hi) else "ns"
    print(f"  {name}: non-dev {vals[low].mean():.2f}, developing {vals[high].mean():.2f}, "
          f"diff {d:+.2f} {unit}, cluster-bootstrap 95% CI [{lo:+.2f}, {hi:+.2f}]  {sig}")
    return d, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aewc-glob", default="data/aewc/ERA-Int_ew_700hPa_*_AFR.nc")
    ap.add_argument("--csct", default="data/original/csct/csct_africa_cs245.nc")
    ap.add_argument("--level", type=int, default=700, choices=(700, 850),
                    help="trajectory/sampling pressure level. 700 is the level of the "
                         "moisture anomaly (Saharan dry-air side); 850 tests the low-level "
                         "monsoon-inflow channel. At 850 the level intersects elevated "
                         "terrain in the east (ERA5 extrapolates below ground), so eastern "
                         "origins are indicative only.")
    ap.add_argument("--years", default=None,
                    help="season set, e.g. 2000-2004 (the development tier) or "
                         "1983-2007. Default: every AEWC/ERA5 year on disk, which is "
                         "the pooled record once all files are present. Passing the "
                         "set explicitly makes each tier's numbers regenerable.")
    ap.add_argument("--out", default="fig_moisture_budget.png")
    a = ap.parse_args()
    level = a.level
    years = None
    if a.years:
        from validate_heldout import parse_years
        years = parse_years(a.years)
        print(f"season set: {len(years)} seasons {years[0]}..{years[-1]}")
        aewc_source = [f"data/aewc/ERA-Int_ew_700hPa_{y}_AFR.nc" for y in years]
    else:
        aewc_source = a.aewc_glob

    tr = (load_aewc_trajectories(aewc_source)
          .filter_region(min_lat=5, max_lat=20, min_lon=-30, max_lon=40)
          .filter_months([7, 8, 9]))
    cs = xr.open_dataset(a.csct)
    cst = pd.DatetimeIndex(cs["time"].values).values
    csx = np.asarray(cs["lon"].values, float)
    csy = np.asarray(cs["lat"].values, float)
    cs.close()
    if years is not None:
        inyr = np.isin(pd.DatetimeIndex(cst).year, years)
        cst, csx, csy = cst[inyr], csx[inyr], csy[inyr]

    # the same stratified split as fig_developing (longitude x month, same response, bins,
    # minimum), so the two groups are matched in geography AND season
    resp = forward_response(tr.time, tr.lon, cst, csx, csy, RESP_WIN_H, DLON, LAT_LO, LAT_HI)
    month = pd.DatetimeIndex(tr.time).month.values.astype(float)
    low, high = stratified_terciles(resp, tr.lon, np.arange(-30, 41, 10.0), min_bin=30,
                                    strat2=month, edges2=np.array([6.5, 7.5, 8.5, 9.5]))
    sel = low | high
    print(f"troughs in the split: {sel.sum()} of {len(tr)} "
          f"(non-dev {low.sum()}, developing {high.sum()})")

    tw, wlat, wlon, uu, vv, wsrc = load_winds(level, years)
    print(f"winds: {level} hPa, {wsrc}, {tw.size} steps "
          f"{tw.min().date()}..{tw.max().date()}")
    u = Gridded(tw.values, wlat, wlon, uu)
    v = Gridded(tw.values, wlat, wlon, vv)
    tr7, rlat, rlon, rfield = load_region_6h(f"r{level}", years=years)
    rh = Gridded(tr7.values, rlat, rlon, rfield)
    try:
        tt7, tlat, tlon, tfield = load_region_6h(f"t{level}", years=years)
        temp = Gridded(tt7.values, tlat, tlon, tfield)
    except FileNotFoundError:
        temp = None
        print(f"(t{level} not on disk; theta-e diagnostic skipped)")

    # seeds: 9 parcels per selected trough, in the t-24h box
    sel_idx = np.where(sel)[0]
    n_case = sel_idx.size
    seed_time = (tr.time[sel_idx].astype("datetime64[ns]")
                 - np.timedelta64(int(LEAD_H * 3600), "s"))
    grid_dlon, grid_lat = np.meshgrid(SEED_DLON, SEED_LATS)          # (3, 3)
    npar = grid_dlon.size
    seeds_t = np.repeat(seed_time, npar)
    seeds_lon = (tr.lon[sel_idx][:, None] + grid_dlon.ravel()[None, :]).ravel()
    seeds_lat = np.broadcast_to(grid_lat.ravel()[None, :], (n_case, npar)).ravel().copy()

    print(f"integrating {seeds_t.size} parcels {BACK_H:.0f} h backward at {level} hPa ...")
    elapsed, plat, plon = back_trajectories(u, v, seeds_t, seeds_lat, seeds_lon,
                                            hours=BACK_H, dt_hours=1.0)

    # 1. origin sectors at t - (LEAD_H + BACK_H)
    sector = classify_origin(seeds_lat, seeds_lon, plat[-1], plon[-1])
    lost = float((sector == "lost").mean())
    print(f"parcels lost from the domain/record: {100 * lost:.1f}%")
    sec_case = {}
    for s in ("south", "north", "east", "west", "local"):
        sec_case[s] = (sector == s).astype(float).reshape(n_case, npar).mean(axis=1)
    dlat_case = np.nanmean((plat[-1] - seeds_lat).reshape(n_case, npar), axis=1)

    gids = tr.variables["traj_id"][sel_idx]
    low_s = low[sel_idx]
    high_s = high[sel_idx]
    rng = np.random.default_rng(0)

    print("\nORIGIN SECTORS at t-72h (per-case parcel fractions; the meridional axis takes "
          "precedence, so 'north' includes the northeasterly origins along the jet):")
    for s in ("south", "north", "east", "west", "local"):
        nd = sec_case[s][low_s].mean()
        dv = sec_case[s][high_s].mean()
        print(f"  {s:6s}: non-dev {100 * nd:5.1f}%   developing {100 * dv:5.1f}%")
    print("\nGROUP CONTRASTS (trajectory cluster bootstrap):")
    group_stats(sec_case["south"], gids, low_s, high_s, rng, "frac", "south-origin fraction")
    group_stats(np.nan_to_num(dlat_case, nan=0.0), gids, low_s, high_s, rng, "deg",
                "meridional displacement (origin minus seed lat)")
    if level == 850:
        # 850 hPa intersects the eastern highlands, where ERA5 extrapolates below ground,
        # so full-domain 850 origin statistics are terrain-contaminated in the east. The
        # west-of-25E subset is the safe version of the monsoon-import test.
        west_ok = tr.lon[sel_idx] < 25.0
        print(f"  WEST-OF-25E SUBSET (terrain-safe 850 origins, "
              f"{int((low_s & west_ok).sum())}/{int((high_s & west_ok).sum())} cases):")
        group_stats(sec_case["south"][west_ok], gids[west_ok], low_s[west_ok],
                    high_s[west_ok], rng, "frac", "  south-origin fraction")
        group_stats(np.nan_to_num(dlat_case, nan=0.0)[west_ok], gids[west_ok],
                    low_s[west_ok], high_s[west_ok], rng, "deg",
                    "  meridional displacement")

    # 2. along-track RH at fixed hours before the seed. The 0 h point is essentially the
    # box contrast itself and the -12/-24 h parcels are still near the box, so only the
    # -48 h point (parcels ~10-12 deg away) is independent evidence of an upstream origin;
    # the mean separation is printed with each row so the reader can see this.
    print(f"\nALONG-TRACK {level} hPa RH (per-case parcel mean at hours before the seed; "
          "lean on -48 h, the earlier points are progressively closer to the box):")
    rh_nd, rh_dv, rh_dd = [], [], []
    for eh in RH_ELAPSED:
        k = int(round(eh / (elapsed[1] - elapsed[0])))
        t_abs = (seeds_t.astype("datetime64[ns]").astype("int64") / 3.6e12) - eh
        vals = rh.sample(t_abs, plat[k], plon[k]).reshape(n_case, npar)
        case = np.nanmean(vals, axis=1)
        sep = np.nanmean(np.hypot(plat[k] - seeds_lat, plon[k] - seeds_lon))
        ok = np.isfinite(case)
        d, lo, hi, _, _ = cluster_bootstrap_diff(
            gids[low_s & ok], case[low_s & ok], gids[high_s & ok], case[high_s & ok], rng)
        sig = "significant" if not (lo <= 0 <= hi) else "ns"
        rh_nd.append(case[low_s & ok].mean())
        rh_dv.append(case[high_s & ok].mean())
        rh_dd.append((d, lo, hi))
        print(f"  -{eh:4.0f} h (mean separation {sep:4.1f} deg): non-dev {rh_nd[-1]:5.1f}%  "
              f"developing {rh_dv[-1]:5.1f}%  diff {d:+.2f} [{lo:+.2f}, {hi:+.2f}]  {sig}")
        if eh == RH_ELAPSED[-1]:
            # effective-sample-size check: the wave cluster already contains the parcel
            # autocorrelation (a wave's cases and parcels resample together); the year
            # block is the strictest exchangeable unit and absorbs regime-scale dependence
            years = pd.DatetimeIndex(tr.time[sel_idx]).year.values
            dy, loy, hiy, _, _ = cluster_bootstrap_diff(
                years[low_s & ok], case[low_s & ok], years[high_s & ok],
                case[high_s & ok], rng)
            sigy = "significant" if not (loy <= 0 <= hiy) else "ns"
            print(f"          year-block stress check of the -{eh:.0f} h contrast "
                  f"({np.unique(years).size} clusters, not primary inference): "
                  f"{dy:+.2f} [{loy:+.2f}, {hiy:+.2f}]  {sigy}")

    # theta-e along track: relative humidity is temperature-dependent, so a moisture
    # contrast alone cannot distinguish a distinct airmass from a temperature fluctuation.
    # Theta-e is conserved in dry and pseudoadiabatic displacements, and the Saharan Air
    # Layer carries a mid-level theta-e minimum, so a theta-e deficit along the more
    # northerly non-developing inflow marks genuine dry-airmass import.
    if temp is not None:
        print(f"\nALONG-TRACK theta-e at {level} hPa (K; airmass identity, "
              "conserved under dry/pseudoadiabatic displacement):")
        for eh in RH_ELAPSED:
            k = int(round(eh / (elapsed[1] - elapsed[0])))
            t_abs = (seeds_t.astype("datetime64[ns]").astype("int64") / 3.6e12) - eh
            rh_p = rh.sample(t_abs, plat[k], plon[k])
            tt_p = temp.sample(t_abs, plat[k], plon[k])
            te = theta_e(tt_p, rh_p, float(level)).reshape(n_case, npar)
            case_te = np.nanmean(te, axis=1)
            ok = np.isfinite(case_te)
            d, lo, hi, _, _ = cluster_bootstrap_diff(
                gids[low_s & ok], case_te[low_s & ok],
                gids[high_s & ok], case_te[high_s & ok], rng)
            sig = "significant" if not (lo <= 0 <= hi) else "ns"
            print(f"  -{eh:4.0f} h: non-dev {case_te[low_s & ok].mean():6.1f}  developing "
                  f"{case_te[high_s & ok].mean():6.1f}  diff {d:+.2f} "
                  f"[{lo:+.2f}, {hi:+.2f}]  {sig}")

        # vapor/temperature decomposition: a relative-humidity contrast can be more vapor,
        # cooler air, or both, and theta-e NETS the two (vapor raises it, cooling lowers
        # it). Saharan Air Layer air is warm AND dry (elevated mixed layer), so a moister-
        # and-cooler developing inflow with a near-zero theta-e contrast is itself the SAL
        # signature; this table shows the two components separately.
        from aew.thermo import saturation_vapor_pressure
        print(f"\nVAPOR/TEMPERATURE DECOMPOSITION at {level} hPa (mixing ratio g/kg, T K):")
        for eh in (0.0, 24.0, 48.0):
            k = int(round(eh / (elapsed[1] - elapsed[0])))
            t_abs = (seeds_t.astype("datetime64[ns]").astype("int64") / 3.6e12) - eh
            rp = rh.sample(t_abs, plat[k], plon[k])
            tp = temp.sample(t_abs, plat[k], plon[k])
            e = (rp / 100.0) * saturation_vapor_pressure(tp)
            mix = 622.0 * e / (float(level) - e)
            for name, arr, unit in (("r", mix, "g/kg"), ("T", tp, "K")):
                case_x = np.nanmean(arr.reshape(n_case, npar), axis=1)
                ok = np.isfinite(case_x)
                d, lo, hi, _, _ = cluster_bootstrap_diff(
                    gids[low_s & ok], case_x[low_s & ok],
                    gids[high_s & ok], case_x[high_s & ok], rng)
                sig = "significant" if not (lo <= 0 <= hi) else "ns"
                print(f"  -{eh:2.0f} h  {name}: non-dev {case_x[low_s & ok].mean():7.2f}  "
                      f"developing {case_x[high_s & ok].mean():7.2f}  diff {d:+.3f} {unit} "
                      f"[{lo:+.3f}, {hi:+.3f}]  {sig}")

        # the SAL fingerprint: theta-e of the NORTH-ORIGIN parcels at -48 h. If the
        # non-developing inflow imports more Saharan air, its northern parcels carry a
        # deeper theta-e minimum, not just a drier RH.
        k = int(round(RH_ELAPSED[-1] / (elapsed[1] - elapsed[0])))
        t_abs = (seeds_t.astype("datetime64[ns]").astype("int64") / 3.6e12) - RH_ELAPSED[-1]
        te48 = theta_e(temp.sample(t_abs, plat[k], plon[k]),
                       rh.sample(t_abs, plat[k], plon[k]), float(level))
        te_n = np.where(sector == "north", te48, np.nan).reshape(n_case, npar)
        case_n = np.nanmean(te_n, axis=1)
        okn = np.isfinite(case_n)
        d, lo, hi, _, _ = cluster_bootstrap_diff(
            gids[low_s & okn], case_n[low_s & okn],
            gids[high_s & okn], case_n[high_s & okn], rng)
        sig = "significant" if not (lo <= 0 <= hi) else "ns"
        print(f"  NORTH-ORIGIN parcels only, theta-e at -48 h: non-dev "
              f"{case_n[low_s & okn].mean():6.1f}  developing "
              f"{case_n[high_s & okn].mean():6.1f}  diff {d:+.2f} "
              f"[{lo:+.2f}, {hi:+.2f}]  {sig}  "
              f"(CONDITIONAL on cases with at least one north-origin parcel, retained "
              f"{int(okn[low_s].sum())}/{int(okn[high_s].sum())} of "
              f"{int(low_s.sum())}/{int(high_s.sum())}; differential retention means this "
              "is not a group-wide fingerprint)")

    # Eulerian control: a fixed box at the climatological source offset (t-72h, east of the
    # meridian) instead of the tracked parcels. If the trajectory contrast were just the box
    # contrast advected, this would reproduce it; a much weaker Eulerian contrast means the
    # Lagrangian tracking is finding the actual moist supply.
    from aew.environment import lead_field_box
    print("\nEULERIAN CONTROL (fixed box at L+shift, t-72h, same split):")
    for shift in (8.0, 12.0):
        lv = lead_field_box(tr.time, tr.lon + shift, tr7.values, rlat, rlon, rfield,
                            72.0, tol_h=3.0, dlon=5.0, lat_lo=LAT_LO, lat_hi=LAT_HI)
        lvs = lv[sel_idx]
        ok = np.isfinite(lvs)
        d, lo, hi, _, _ = cluster_bootstrap_diff(
            gids[low_s & ok], lvs[low_s & ok], gids[high_s & ok], lvs[high_s & ok], rng)
        sig = "significant" if not (lo <= 0 <= hi) else "ns"
        print(f"  L+{shift:.0f} deg: diff {d:+.2f}%  CI [{lo:+.2f}, {hi:+.2f}]  {sig}")

    # 3. antecedent convection near the box in the PRIOR day (t-48h .. t-24h)
    prior_start = tr.time[sel_idx].astype("datetime64[ns]") - np.timedelta64(48, "h")
    antecedent = forward_response(prior_start, tr.lon[sel_idx], cst, csx, csy,
                                  win_h=24.0, dlon=DLON + 2.0,
                                  lat_lo=LAT_LO, lat_hi=LAT_HI)
    print("\nANTECEDENT CONVECTION (CS-245 count near the box, t-48h..t-24h; read as a "
          "prior-convection ASSOCIATION, since convective regimes persist over 48-72 h "
          "and the count is not a direct rainfall-evaporation measurement):")
    group_stats(antecedent, gids, low_s, high_s, rng, "systems", "prior-day count")

    print("\nREADING GUIDE: an RH contrast already present at -48 h means the parcels "
          "arrive already moist (advective origin); a contrast opening only near the box "
          "plus more antecedent convection means local moistening from prior rainfall. "
          "The advective case itself splits by the displacement contrast: more southerly "
          "origins mean enhanced monsoonal import, while equal south fractions with a "
          "reduced NORTHWARD displacement mean less Saharan dry-air import (at 700 hPa "
          "the moisture gradient is dry air to the north). Isobaric 700 hPa trajectories "
          "cannot see moisture ascending from below that level (stated limitation).")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

    # panel A: origin-point density (2D histogram difference, developing minus non-dev)
    ok = np.isfinite(plat[-1]) & np.isfinite(plon[-1])
    case_of_parcel = np.repeat(np.arange(n_case), npar)
    for msk, col, lab in ((low_s, "tab:blue", "MCS-quiet"),
                          (high_s, "tab:red", "MCS-active")):
        pm = msk[case_of_parcel] & ok
        ax1.scatter(plon[-1][pm], plat[-1][pm], s=1.5, alpha=0.12, color=col, label=lab)
    ax1.axhline(LAT_LO, color="k", lw=0.5); ax1.axhline(LAT_HI, color="k", lw=0.5)
    ax1.set_xlim(-45, 50); ax1.set_ylim(-12, 32)
    ax1.set_xlabel("origin longitude (deg E)"); ax1.set_ylabel("origin latitude (deg N)")
    leg = ax1.legend(fontsize=8, markerscale=8)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    ax1.set_title("Parcel origins, 48 h before the t-24h box")
    ax1.grid(alpha=0.3)

    # panel B: along-track RH
    eh = np.array(RH_ELAPSED)
    ax2.plot(-eh, rh_nd, "o-", color="tab:blue", label="MCS-quiet")
    ax2.plot(-eh, rh_dv, "o-", color="tab:red", label="MCS-active")
    ax2.set_xlabel("hours before the t-24h box sample")
    ax2.set_ylabel(f"{level} hPa RH along track (%)")
    ax2.set_title("Along-track moisture history")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # panel C: RH difference with CIs
    dd = np.array([d for d, _, _ in rh_dd])
    lo_ = np.array([lo for _, lo, _ in rh_dd])
    hi_ = np.array([hi for _, _, hi in rh_dd])
    ax3.axhline(0, color="k", lw=0.8)
    ax3.fill_between(-eh, lo_, hi_, color="grey", alpha=0.3, label="cluster-bootstrap 95% CI")
    ax3.plot(-eh, dd, "o-", color="tab:purple")
    ax3.set_xlabel("hours before the t-24h box sample")
    ax3.set_ylabel("MCS-active minus MCS-quiet RH (%)")
    ax3.set_title("When does the moisture contrast open?")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(a.out, dpi=150)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
