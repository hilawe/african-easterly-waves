#!/usr/bin/env python
"""Time-domain lead-lag between the AEW trough and convection, resolved by longitude.

Addresses the ordering objection directly in time. At each reference meridian a continuous
6-hourly wave series (700 hPa curvature vorticity, 5-15N band mean, from ERA5) and a
convection series (cold-cloud CS-245 count in a fixed longitude/latitude box, from csct) are
each band-passed to the 2-6 day AEW band, then cross-correlated. The lag at peak correlation
is the temporal phase relationship: a positive lag means convection lags the trough (the
wave leads), a negative lag means convection leads the trough. Sweeping the meridian from the
eastern genesis region to the coast tests whether that phase relationship changes along the
corridor.

Interpretation caveat printed with the result: at a single meridian this lag is the
trough-convection phase offset carried by the propagating wave, so it is read as a phase
relationship and its change with longitude, not as evidence for a forcing direction on its
own. The convection record (csct) exists only for the July-September season, so the band-pass
is applied within each JAS block; the reported significance folds the band-pass
autocorrelation into an effective sample size and is approximate.

Inputs: ERA5 u/v 700 hPa 6-hourly global (data/era5/global6h), csct CS-245 systems
(data/original/csct). Writes fig_leadlag.png.
"""

import argparse
import glob

import numpy as np
import pandas as pd
import xarray as xr

from aew.filtering import lanczos_bandpass
from aew.leadlag import band_mean_series, box_count_series, lag_cross_correlation, peak_lag
from aew.waves import curvature_vorticity

OBS_PER_DAY = 4          # 6-hourly
STEP_HOURS = 6.0
PERIOD_LOW, PERIOD_HIGH = 2.0, 6.0
LAT_LO, LAT_HI = 5.0, 15.0
DLON = 5.0               # convection box half-width in longitude
MAX_LAG = 12             # +/- 12 samples = +/- 3 days (full curve for the plot)
PEAK_WIN = 30.0          # search the peak within +/- 30 h (the near-zero-lag lobe)
JAS = (7, 8, 9)


def load_era5_curv(u_glob, v_glob):
    """Load ERA5 u/v 700 6-hourly, return (times, lat, lon, band-mean curvature vorticity).

    Curvature vorticity is computed per time step on the full grid, then averaged over the
    5-15N band. Returns W of shape (ntime, nlon).
    """
    up = sorted(glob.glob(u_glob))
    vp = sorted(glob.glob(v_glob))
    if not up or not vp:
        raise FileNotFoundError(f"no ERA5 files for {u_glob!r} / {v_glob!r}")

    def _concat(paths, var):
        ts, lat, lon, blocks = [], None, None, []
        for p in paths:
            ds = xr.open_dataset(p)
            ts.append(pd.DatetimeIndex(ds["valid_time"].values))
            lat = np.asarray(ds["latitude"].values, float)
            lon = np.asarray(ds["longitude"].values, float)
            blocks.append(np.asarray(ds[var].squeeze().values, float))  # (time, lat, lon)
            ds.close()
        t = pd.DatetimeIndex(np.concatenate([x.values for x in ts]))
        arr = np.concatenate(blocks, axis=0)
        o = np.argsort(t.values)
        return t[o], lat, lon, arr[o]

    times, lat, lon, u = _concat(up, "u")
    tv, latv, lonv, v = _concat(vp, "v")
    # the u and v records must share the same time base and grid (curvature vorticity mixes
    # them point by point); a mismatched or missing file would silently corrupt the wave.
    if not (times.equals(tv) and np.array_equal(lat, latv) and np.array_equal(lon, lonv)):
        raise ValueError("ERA5 u and v files do not share the same time/lat/lon grid")
    # ascending latitude for the spherical-gradient routine
    if lat[0] > lat[-1]:
        lat = lat[::-1]; u = u[:, ::-1, :]; v = v[:, ::-1, :]
    crv = curvature_vorticity(u, v, lat, lon)         # (time, lat, lon), NaN in calm cells
    w = band_mean_series(crv, lat, LAT_LO, LAT_HI)    # (time, nlon)
    return times, lon, w


def _lag1(x):
    """Lag-1 autocorrelation of the finite part of x (for the effective sample size)."""
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0
    x = x - x.mean()
    d = (x * x).sum()
    return float((x[:-1] * x[1:]).sum() / d) if d > 0 else 0.0


def seasonal_lag_correlation(w_raw, c_raw, times, max_lag, months=JAS, buffer_days=15):
    """Per-year season lead-lag with the band-pass applied INSIDE each season block.

    The convection record (csct) exists only for the July-September season, so a full-record
    band-pass would filter the JAS edges against artificial off-season zeros. Instead each
    year's JAS block is extracted from the raw (unfiltered) wave and convection series and
    band-passed on its own; ``lanczos_bandpass`` sets the first/last ``buffer_days`` to NaN,
    so the cross-correlation (which drops NaN pairs) uses only the clean season core and
    never pairs across the seasonal gap. Returns the lag axis, the stack of per-year
    correlation curves, and the pooled season-core (w, c) for a significance estimate.
    """
    years = np.unique(times.year)
    R_years, w_cores, c_cores = [], [], []
    lags = np.arange(-max_lag, max_lag + 1)
    for y in years:
        m = np.asarray((times.year == y) & np.isin(times.month, months))
        if m.sum() < 2 * buffer_days * OBS_PER_DAY + 4 * max_lag:
            continue
        wf = lanczos_bandpass(w_raw[m], PERIOD_LOW, PERIOD_HIGH, OBS_PER_DAY, buffer_days)
        cf = lanczos_bandpass(c_raw[m], PERIOD_LOW, PERIOD_HIGH, OBS_PER_DAY, buffer_days)
        _, R = lag_cross_correlation(wf, cf, max_lag)
        if np.isfinite(R).sum() < max_lag:
            continue
        R_years.append(R)
        w_cores.append(wf); c_cores.append(cf)
    if not R_years:
        raise ValueError("no season blocks with enough data")
    return (lags, np.vstack(R_years),
            np.concatenate(w_cores), np.concatenate(c_cores))


def bootstrap_peak_ci(lags, R_years, peak_win, rng, n_boot=500):
    """Year-block bootstrap CI for the peak lag of the mean correlation curve.

    Resamples whole years with replacement, re-averages their correlation curves, and takes
    the peak within +/- ``peak_win`` hours -- the SAME estimator as the reported central
    value (peak of the mean curve), so the interval is consistent with the point estimate.
    """
    win = np.abs(lags * STEP_HOURS) <= peak_win
    n = R_years.shape[0]
    peaks = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        Rb = np.nanmean(R_years[idx], axis=0)
        peaks[b] = peak_lag(lags[win], Rb[win])[0] * STEP_HOURS
    return float(np.nanstd(peaks)), np.nanpercentile(peaks, [16, 84])


def effective_dof_p(w_core, c_core, r):
    """Two-sided p-value for correlation ``r`` with the band-pass autocorrelation folded
    into an effective sample size N_eff = N (1 - a b) / (1 + a b), a, b the lag-1
    autocorrelations. Rough but honest: a 2-6 day band-passed series is far from N
    independent 6-hourly samples."""
    from scipy import stats

    n = np.isfinite(w_core).sum()
    a, b = _lag1(w_core), _lag1(c_core)
    n_eff = n * (1 - a * b) / (1 + a * b)
    if n_eff <= 3 or abs(r) >= 1:
        return n_eff, np.nan
    t = r * np.sqrt((n_eff - 2) / (1 - r * r))
    return n_eff, float(2 * stats.t.sf(abs(t), n_eff - 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u-glob", default="data/era5/global6h/era5_u700_200*_6h_global.nc")
    ap.add_argument("--v-glob", default="data/era5/global6h/era5_v700_200*_6h_global.nc")
    ap.add_argument("--csct", default="data/original/csct/csct_africa_cs245.nc")
    ap.add_argument("--lon-lo", type=float, default=-15.0)
    ap.add_argument("--lon-hi", type=float, default=35.0)
    ap.add_argument("--lon-step", type=float, default=2.5)
    ap.add_argument("--out", default="fig_leadlag.png")
    a = ap.parse_args()

    times, era_lon, w_all = load_era5_curv(a.u_glob, a.v_glob)
    print(f"ERA5 wave field: {times.size} steps {times.min().date()}..{times.max().date()}, "
          f"{era_lon.size} lons")

    cs = xr.open_dataset(a.csct)
    # csct carries two datetime coordinates: `time` (the per-observation time, ~70k/yr over
    # 1983-2007) and a corrupt `num_hits` coordinate (datetimes on the lat/lon dimension, 0
    # systems in 2000). `time` is the physical observation time and is row-parallel to
    # lat/lon; `num_hits` is ignored.
    cst = pd.DatetimeIndex(cs["time"].values)
    csx = np.asarray(cs["lon"].values, float)
    csy = np.asarray(cs["lat"].values, float)
    cs.close()
    # restrict systems to the record span for speed
    span = (cst >= times.min()) & (cst <= times.max())
    cst, csx, csy = cst[span], csx[span], csy[span]
    cs_months = np.unique(cst.month)
    print(f"csct systems in span: {span.sum()}, months present {sorted(cs_months)} "
          f"(band-pass is applied within each season block)")

    grid = times.values  # uniform 6-hourly grid for the convection series

    meridians = np.arange(a.lon_lo, a.lon_hi + 0.1, a.lon_step)
    peak_h, peak_sd, r_at_peak, p_at_peak = [], [], [], []
    rep = {"eastern Sahel": 15.0, "central": 5.0, "west": -10.0}
    rep_curves = {}
    R_MIN = 0.20  # below this the curvature-vorticity wave is too weak for a reliable lag
    win = np.abs(np.arange(-MAX_LAG, MAX_LAG + 1) * STEP_HOURS) <= PEAK_WIN
    rng = np.random.default_rng(0)

    for lam in meridians:
        j = int(np.argmin(np.abs(era_lon - lam)))
        w_raw = w_all[:, j]                                           # unfiltered wave
        c_raw = box_count_series(cst.values, csx, csy, grid, lam, DLON, LAT_LO, LAT_HI)
        lags, R_years, w_core, c_core = seasonal_lag_correlation(w_raw, c_raw, times, MAX_LAG)
        Rbar = np.nanmean(R_years, axis=0)
        # locate the peak within the near-zero-lag lobe (+/- PEAK_WIN h): with a 2-6 day
        # band the correlation has side lobes near +/- one wave period, which are the wave's
        # own periodicity, not the trough-convection phase offset we want.
        lag_star, r_star = peak_lag(lags[win], Rbar[win])
        sd, _ = bootstrap_peak_ci(lags, R_years, PEAK_WIN, rng)
        _, pval = effective_dof_p(w_core, c_core, r_star)
        peak_h.append(lag_star * STEP_HOURS)
        peak_sd.append(sd)
        r_at_peak.append(r_star)
        p_at_peak.append(pval)
        for name, target in rep.items():
            if abs(lam - target) < a.lon_step / 2:
                rep_curves[name] = (lags * STEP_HOURS, Rbar, lam)

    peak_h = np.array(peak_h); peak_sd = np.array(peak_sd)
    r_at_peak = np.array(r_at_peak); p_at_peak = np.array(p_at_peak)

    print("\nlon(E)   peak-lag(h)   boot +/-(h)   R@peak   p(eff-dof)   reading")
    for lam, ph, sd, r, pv in zip(meridians, peak_h, peak_sd, r_at_peak, p_at_peak):
        reading = "wave leads" if ph > 0 else "convection leads"
        print(f"{lam:6.1f}   {ph:+8.1f}   {sd:10.1f}   {r:6.2f}   {pv:9.1e}   {reading}")
    reliable = r_at_peak >= R_MIN
    if reliable.any():
        lo, hi = meridians[reliable].min(), meridians[reliable].max()
        print(f"\nreliable corridor (R>={R_MIN:.2f}): {lo:.1f}..{hi:.1f}E, "
              f"mean peak lag {np.nanmean(peak_h[reliable]):+.1f} h "
              f"(convection leads the trough); mean R {np.nanmean(r_at_peak[reliable]):.2f}, "
              f"max p {np.nanmax(p_at_peak[reliable]):.1e}")
        print(f"east of {hi:.1f}E the curvature-vorticity wave amplitude is small (R<{R_MIN:.2f}); "
              "lead-lag there is not resolved -- consistent with the 40E AEWC/eastern-tracker limit")
    print("\nCAVEAT: at a fixed meridian the lag is the trough-convection phase offset carried "
          "by the propagating wave, so it is read as a phase relationship and its change with "
          "longitude, not as evidence for a forcing direction on its own. The effective-DOF "
          "p-value folds in the band-pass autocorrelation but is only approximate.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = {"eastern Sahel": "tab:red", "central": "tab:green", "west": "tab:blue"}
    for name in ("eastern Sahel", "central", "west"):
        if name in rep_curves:
            lh, R, lam = rep_curves[name]
            ax1.plot(lh, R, color=colors[name], label=f"{name} ({lam:.0f}E)")
    ax1.axvline(0, color="k", lw=0.8)
    ax1.axhline(0, color="k", lw=0.4)
    ax1.set_xlabel("lag of convection relative to trough (h; + = convection later)")
    ax1.set_ylabel("band-passed cross-correlation")
    ax1.set_title("AEW trough vs convection lead-lag (2-6 day band)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.axhline(0, color="k", lw=0.8)
    rel = r_at_peak >= R_MIN
    ax2.fill_between(meridians[rel], (peak_h - peak_sd)[rel], (peak_h + peak_sd)[rel],
                     color="grey", alpha=0.25, label="+/- 1 sigma across years")
    ax2.plot(meridians[rel], peak_h[rel], "o-", color="tab:purple",
             label=f"reliable (R>={R_MIN:.2f})")
    if (~rel).any():
        ax2.plot(meridians[~rel], peak_h[~rel], "x", color="lightgrey",
                 label="wave too weak")
    ax2.set_ylim(-36, 36)
    ax2.set_xlabel("longitude (deg E)")
    ax2.set_ylabel("peak lag (h); + = wave leads, - = convection leads")
    ax2.set_title("Lead-lag along the corridor")
    ax2.text(0.02, 0.04, "above 0: wave leads convection\nbelow 0: convection leads wave",
             transform=ax2.transAxes, fontsize=8, va="bottom")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(a.out, dpi=150)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
