"""Lag composites and CS/CT Hovmoller binning.

Two engines, ported from the surviving Carl Schreck NCL source:

- ``composite_xt`` / ``composite_xt_preread`` -- longitude-lag composite of a gridded
  field about a set of base (event) dates, with a Monte Carlo significance test whose
  null draws come from the SAME calendar days in OTHER years (dates_from_other_years).
  Ported from composites/composite_xt.ncl. Used for the unfiltered-v700 contours and
  for field (wind/shear/size) shading in the Hovmollers.

- ``hovmoller_event_counts`` -- the CS/CT shading: for every base date, the lag of each
  cloud system (cs_time - event) is binned into a (lag, lon) grid via aew.binning.bin_sum,
  summed over all base dates. Ported from the binning loop in
  legacy_ncl/hov/v_csct_comp_wave_hov.ncl. ``anomaly`` turns the raw counts into the
  published anomaly (count minus its lag-window mean) or percentage.

Time handling uses pandas datetime64 (ERA5 via xarray decodes to datetime64). A
non-standard-calendar (cftime) path can be added later if a dataset needs it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .binning import bin_sum

__all__ = [
    "LagComposite",
    "dates_from_other_years",
    "composite_xt_preread",
    "composite_xy_preread",
    "LagCompositeXY",
    "hovmoller_event_counts",
    "hovmoller_event_field",
    "wave_relative_counts",
    "map_event_counts",
    "anomaly",
    "lag_axis",
]


def lag_axis(min_lag, max_lag, delta_lag=1.0):
    """Lag values from min_lag..max_lag inclusive, matching NCL's
    ``1 + round((maxLag-minLag)/deltaLag)`` count."""
    n = 1 + int(round((max_lag - min_lag) / delta_lag))
    return min_lag + delta_lag * np.arange(n)


@dataclass
class LagComposite:
    """Result of composite_xt_preread."""

    values: np.ndarray  # (nlag, nlon) composite; NaN where insignificant (if tested)
    lag: np.ndarray  # (nlag,) lag values in days
    lon: np.ndarray  # (nlon,)
    n_dates: np.ndarray  # (nlag,) number of base dates contributing at each lag
    n_samp: int  # median of n_dates (sample size used for the null)
    p_value: np.ndarray = field(default=None)  # (nlag, nlon) two-sided, or None


@dataclass
class LagCompositeXY:
    """Result of composite_xy_preread (longitude-latitude maps per lag)."""

    values: np.ndarray  # (nlag, nlat, nlon)
    lag: np.ndarray  # (nlag,)
    lat: np.ndarray  # (nlat,)
    lon: np.ndarray  # (nlon,)
    n_dates: np.ndarray  # (nlag,)
    n_samp: int
    p_value: np.ndarray = field(default=None)  # (nlag, nlat, nlon) or None


def dates_from_other_years(base_times, all_times):
    """Population dates with the same month/day/hour as some base date but a different
    year FROM THAT base date, excluding the base dates themselves. Exact port of
    dates_from_other_years.ncl (the per-sample different-year test, not just m/d/h
    membership).
    """
    from collections import defaultdict

    base_times = pd.DatetimeIndex(base_times)
    all_times = pd.DatetimeIndex(all_times)

    # (month, day, hour) -> set of base years
    base_years = defaultdict(set)
    for m, d, h, y in zip(base_times.month, base_times.day, base_times.hour, base_times.year):
        base_years[(int(m), int(d), int(h))].add(int(y))

    am, ad, ah, ay = (
        all_times.month.values, all_times.day.values,
        all_times.hour.values, all_times.year.values,
    )
    mask = np.zeros(all_times.size, dtype=bool)
    for i in range(all_times.size):
        years = base_years.get((int(am[i]), int(ad[i]), int(ah[i])))
        if years is not None and any(y != int(ay[i]) for y in years):
            mask[i] = True
    mask &= ~all_times.isin(base_times)  # eliminate the sample dates themselves
    if not mask.any():
        return all_times.values  # NCL warns and returns the whole population
    return all_times.values[mask]


def composite_xt_preread(
    data, time, base_times, min_lag, max_lag, delta_lag=1.0, n_tests=0, p_thresh=0.0,
    rng=None,
):
    """Longitude-lag composite about base dates, with optional Monte Carlo significance.

    Parameters
    ----------
    data : ndarray (ntime, nlon)
        Field already averaged over the latitude band.
    time : array of datetime64 (ntime,)
        Times aligned with ``data`` rows.
    base_times : array of datetime64
        Event/base dates (e.g. from composite_dates).
    min_lag, max_lag, delta_lag : float
        Lag window in days.
    n_tests : int
        Monte Carlo iterations. 0 disables significance (returns raw composite).
    p_thresh : float
        Two-sided significance threshold (e.g. 0.95). Insignificant points -> NaN.
    rng : np.random.Generator, optional
        For reproducible Monte Carlo. NCL used an unseeded RNG, so exact p-values will
        differ run to run / vs NCL; the composite values themselves are deterministic.

    Returns
    -------
    LagComposite
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("composite_xt_preread expects (time, lon) data")
    lags = lag_axis(min_lag, max_lag, delta_lag)
    comp, n_dates, n_samp, p_value = _lag_composite_nd(
        data, time, base_times, lags, n_tests, p_thresh, rng
    )
    return LagComposite(
        values=comp, lag=lags, lon=np.arange(data.shape[1]), n_dates=n_dates,
        n_samp=n_samp, p_value=p_value,
    )


def _lag_composite_nd(data, time, base_times, lags, n_tests, p_thresh, rng):
    """Shared lag-composite + Monte Carlo core for any data of shape (ntime, *spatial).

    Returns (comp, n_dates, n_samp, p_value) with comp/p_value shaped (nlag, *spatial).
    Ported from composite_xt.ncl / composite_xy.ncl (identical algorithm, differing only
    in the number of trailing spatial dimensions).
    """
    data = np.asarray(data, dtype=float)
    time = pd.DatetimeIndex(time)
    base_times = pd.DatetimeIndex(base_times)
    nlag = lags.size
    spatial = data.shape[1:]

    time_pos = pd.Series(np.arange(time.size), index=time)  # exact time -> row index

    comp = np.full((nlag,) + spatial, np.nan)
    n_dates = np.zeros(nlag, dtype=int)
    for li, lag in enumerate(lags):
        target = base_times + pd.to_timedelta(lag, unit="D")
        present = target[target.isin(time)]
        if present.size == 0:
            continue
        rows = time_pos.loc[present].to_numpy()
        comp[li] = np.nanmean(data[rows], axis=0)
        n_dates[li] = present.size

    n_samp = int(np.median(n_dates)) if np.any(n_dates) else 0

    p_value = None
    if n_tests > 0 and n_samp > 0:
        if rng is None:
            rng = np.random.default_rng()
        population = pd.DatetimeIndex(dates_from_other_years(base_times, time))
        pop_rows = time_pos.loc[population[population.isin(time)]].to_numpy()
        percentile = np.zeros((nlag,) + spatial)
        for _ in range(n_tests):
            idx = rng.integers(0, pop_rows.size, n_samp)
            rand_comp = np.nanmean(data[pop_rows[idx]], axis=0)  # (*spatial,)
            percentile += np.where(comp > rand_comp, 1.0 / n_tests, 0.0)
        p_value = -1.0 + 2.0 * np.maximum(percentile, 1.0 - percentile)
        comp = np.where(p_value >= p_thresh, comp, np.nan)

    return comp, n_dates, n_samp, p_value


def composite_xy_preread(
    data, time, base_times, lags, n_tests=0, p_thresh=0.0, rng=None,
):
    """Latitude-longitude composite about base dates, with Monte Carlo significance.

    Map analogue of ``composite_xt_preread`` (ported from composite_xy.ncl). Used for the
    basepoint-map figures (paper Fig 4; dissertation Figs 4.4, 5.4, 5.7).

    Parameters
    ----------
    data : ndarray (ntime, nlat, nlon)
    time : array of datetime64 (ntime,)
    base_times : array of datetime64
    lags : sequence of lag values in days (e.g. [0], or a range). Unlike composite_xt
        this takes an explicit lag list (the maps are usually drawn at a single lag).
    n_tests, p_thresh, rng : as in composite_xt_preread.

    Returns
    -------
    LagCompositeXY  (values shaped (nlag, nlat, nlon))
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 3:
        raise ValueError("composite_xy_preread expects (time, lat, lon) data")
    lags = np.atleast_1d(np.asarray(lags, dtype=float))
    comp, n_dates, n_samp, p_value = _lag_composite_nd(
        data, time, base_times, lags, n_tests, p_thresh, rng
    )
    return LagCompositeXY(
        values=comp, lag=lags, lat=np.arange(data.shape[1]),
        lon=np.arange(data.shape[2]), n_dates=n_dates, n_samp=n_samp, p_value=p_value,
    )


def _lat_filter(cs_time, cs_lon, z, cs_lat, min_lat, max_lat):
    if cs_lat is None or (min_lat is None and max_lat is None):
        return cs_time, cs_lon, z
    cs_lat = np.asarray(cs_lat, dtype=float)
    keep = np.ones(cs_lat.size, dtype=bool)
    if min_lat is not None:
        keep &= cs_lat >= min_lat
    if max_lat is not None:
        keep &= cs_lat <= max_lat
    return cs_time[keep], cs_lon[keep], (None if z is None else z[keep])


def _bin_each_event(event_times, cs_time, cs_lon, z, lon_centers, lag_centers, variant):
    """Accumulate bin_sum (sum, count) over all base dates, lag = cs_time - event."""
    cs_time = pd.DatetimeIndex(cs_time)
    cs_dt = np.asarray(cs_time.values, dtype="datetime64[ns]")
    one_day = np.timedelta64(1, "D")
    lon_centers = np.asarray(lon_centers, dtype=float)
    lag_centers = np.asarray(lag_centers, dtype=float)
    sum_z = np.zeros((lag_centers.size, lon_centers.size), dtype=float)
    cnt = np.zeros((lag_centers.size, lon_centers.size), dtype=float)
    zz = np.ones(cs_dt.size) if z is None else np.asarray(z, dtype=float)
    for ts in pd.DatetimeIndex(event_times):
        ev_dt = np.datetime64(ts.value, "ns")
        lag = (cs_dt - ev_dt) / one_day  # float days
        gbin, gcnt = bin_sum(lon_centers, lag_centers, cs_lon, lag, z=zz, variant=variant)
        sum_z += gbin
        cnt += gcnt
    return sum_z, cnt


def hovmoller_event_counts(
    event_times, cs_time, cs_lon, lon_centers, lag_centers, variant="fixed",
    cs_lat=None, min_lat=None, max_lat=None,
):
    """Bin cloud-system COUNTS into a (lag, lon) Hovmoller about base dates.

    For each base date, lag = (cs_time - event) in days is binned together with cs_lon
    onto (lag_centers, lon_centers), and the COUNT per bin is accumulated over all base
    dates. This is the published CS/CT-frequency shading (NCL ``ans = tempCount`` in
    v_csct_comp_wave_hov.ncl).

    Parameters
    ----------
    event_times, cs_time : arrays of datetime64
    cs_lon : array
    lon_centers, lag_centers : 1-D arrays (lag in days, e.g. -6..6)
    variant : {"fixed", "buggy"}  -- bin_sum variant
    cs_lat, min_lat, max_lat : optional latitude-band restriction

    Returns
    -------
    counts : ndarray (nlag, nlon)
    """
    cs_lon = np.asarray(cs_lon, dtype=float)
    cs_time, cs_lon, _ = _lat_filter(
        pd.DatetimeIndex(cs_time), cs_lon, None, cs_lat, min_lat, max_lat
    )
    _, cnt = _bin_each_event(
        event_times, cs_time, cs_lon, None, lon_centers, lag_centers, variant
    )
    return cnt


def hovmoller_event_field(
    event_times, cs_time, cs_lon, z, lon_centers, lag_centers, statistic="mean",
    variant="fixed", cs_lat=None, min_lat=None, max_lat=None,
):
    """Bin a per-system VALUE z (e.g. storm radius, shear) into a (lag, lon) Hovmoller.

    Unlike ``hovmoller_event_counts``, this is for the storm-size / field Hovmollers
    where the shaded quantity is a per-bin mean (or sum), not a count.

    statistic="mean" -> sum(z)/count per bin (bins with no systems -> NaN)
    statistic="sum"  -> sum(z) per bin
    """
    cs_lon = np.asarray(cs_lon, dtype=float)
    z = np.asarray(z, dtype=float)
    cs_time, cs_lon, z = _lat_filter(
        pd.DatetimeIndex(cs_time), cs_lon, z, cs_lat, min_lat, max_lat
    )
    sum_z, cnt = _bin_each_event(
        event_times, cs_time, cs_lon, z, lon_centers, lag_centers, variant
    )
    if statistic == "sum":
        return sum_z
    if statistic == "mean":
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(cnt > 0, sum_z / cnt, np.nan)
        return out
    raise ValueError("statistic must be 'mean' or 'sum'")


def map_event_counts(
    event_times, cs_time, cs_lon, cs_lat, lon_centers, lat_centers, lag=0.0,
    half_window=0.5, z=None, statistic="count", variant="fixed",
):
    """Bin cloud systems into a (lat, lon) map at a given lag about base dates.

    For each base date, systems whose lag = cs_time - event falls within
    [lag-half_window, lag+half_window] days are binned into (lat_centers, lon_centers)
    via bin_sum, accumulated over all base dates. This is the spatial (map) analogue of
    hovmoller_event_counts and produces the CS-count shading for the basepoint maps
    (paper Fig 4; dissertation Figs 4.4, 5.4, 5.7).

    statistic="count" -> number of systems per cell (z ignored)
    statistic="mean"  -> mean of z per cell (e.g. storm radius); empty cells NaN
    statistic="sum"   -> sum of z per cell

    Returns ndarray (nlat, nlon). Note bin_sum's first axis is the y axis, so we pass
    lat as the y (first) grid -> output is (nlat, nlon).
    """
    cs_time = pd.DatetimeIndex(cs_time)
    cs_lon = np.asarray(cs_lon, dtype=float)
    cs_lat = np.asarray(cs_lat, dtype=float)
    lon_centers = np.asarray(lon_centers, dtype=float)
    lat_centers = np.asarray(lat_centers, dtype=float)
    zz = np.ones(cs_lon.size) if z is None else np.asarray(z, dtype=float)

    cs_dt = np.asarray(cs_time.values, dtype="datetime64[ns]")
    one_day = np.timedelta64(1, "D")
    sum_z = np.zeros((lat_centers.size, lon_centers.size), dtype=float)
    cnt = np.zeros((lat_centers.size, lon_centers.size), dtype=float)
    for ts in pd.DatetimeIndex(event_times):
        lagd = (cs_dt - np.datetime64(ts.value, "ns")) / one_day
        sel = np.abs(lagd - lag) <= half_window
        if not sel.any():
            continue
        # bin into (lat, lon): pass lon as x-grid, lat as y-grid -> output (nlat, nlon)
        gbin, gcnt = bin_sum(lon_centers, lat_centers, cs_lon[sel], cs_lat[sel],
                             z=zz[sel], variant=variant)
        sum_z += gbin
        cnt += gcnt
    if statistic == "count":
        return cnt
    if statistic == "sum":
        return sum_z
    if statistic == "mean":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(cnt > 0, sum_z / cnt, np.nan)
    raise ValueError("statistic must be 'count', 'sum', or 'mean'")


def wave_relative_counts(
    trough_times, trough_lon, cs_time, cs_lon, cs_lat, rel_lon_centers, lat_centers,
    time_tol_hours=3.0, variant="fixed",
):
    """Composite cloud-system counts in TROUGH-RELATIVE coordinates (wave-following).

    Instead of compositing about a fixed basepoint, this composites convection about the
    MOVING wave trough: for each trough observation (time, longitude), every cloud system
    within +/- time_tol_hours of that time is binned by its longitude RELATIVE to the
    trough (cs_lon - trough_lon) and its latitude, accumulated over all trough
    observations. rel_lon = 0 is the trough; positive = east of the trough.

    The result is a count PER TROUGH OBSERVATION: at a time with several coexisting AEW
    troughs, an MCS is counted once for each trough whose window contains it. That is the
    intended trough-relative estimand. To separate a genuine trough-relative signal from
    the background MCS climatology, compare against a null that shifts trough_lon randomly
    (preserving times and latitudes) -- see scripts/fig_wave_following.py.

    Parameters
    ----------
    trough_times : array of datetime64        trough observation times (e.g. AEWC)
    trough_lon : array                          trough longitude at each observation
    cs_time, cs_lon, cs_lat : arrays            cloud systems
    rel_lon_centers, lat_centers : 1-D arrays   bin centers (rel lon in deg, lat in deg)
    time_tol_hours : match cloud systems within this many hours of a trough observation
    variant : bin_sum variant

    Returns
    -------
    counts : ndarray (nlat, nrellon)   summed cloud-system counts per bin
    n_troughs : int                    number of trough observations used
    """
    tt = pd.DatetimeIndex(trough_times).values.astype("datetime64[ns]").astype("int64")
    tlon = np.asarray(trough_lon, dtype=float)
    order = np.argsort(tt)
    tt, tlon = tt[order], tlon[order]

    ct = pd.DatetimeIndex(cs_time).values.astype("datetime64[ns]").astype("int64")
    clon = np.asarray(cs_lon, dtype=float)
    clat = np.asarray(cs_lat, dtype=float)
    corder = np.argsort(ct)
    ct, clon, clat = ct[corder], clon[corder], clat[corder]  # sorted for searchsorted

    rel_lon_centers = np.asarray(rel_lon_centers, dtype=float)
    lat_centers = np.asarray(lat_centers, dtype=float)
    tol = int(time_tol_hours * 3.6e12)  # hours -> ns
    counts = np.zeros((lat_centers.size, rel_lon_centers.size), dtype=float)
    n_used = 0
    for tobs, L in zip(tt, tlon):
        # half-open window [tobs-tol, tobs+tol) so an MCS exactly at the midpoint between
        # two consecutive trough times (e.g. +3 h from one 6-hourly trough = -3 h from the
        # next) is matched to only one of them, not double-counted.
        lo = np.searchsorted(ct, tobs - tol, side="left")
        hi = np.searchsorted(ct, tobs + tol, side="left")
        if hi <= lo:
            continue
        rel = clon[lo:hi] - L
        # wrap trough-relative longitude to [-180, 180]
        rel = (rel + 180.0) % 360.0 - 180.0
        gbin, _ = bin_sum(rel_lon_centers, lat_centers, rel, clat[lo:hi],
                          z=np.ones(hi - lo), variant=variant)
        counts += gbin
        n_used += 1
    return counts, n_used


def anomaly(counts, kind="anomaly", lag_centers=None, min_lag=None, max_lag=None):
    """Convert raw (lag, lon) counts/field to the published shading.

    The baseline is the mean over the lag dimension. NCL averages over the composite
    window ``ans({minLag:maxLag},:)``; pass ``lag_centers`` with ``min_lag``/``max_lag``
    to restrict the baseline to that window (default: average over all rows, which is
    correct when the grid already spans exactly the window, e.g. -6..6).

    kind="total"   -> counts unchanged
    kind="anomaly" -> counts - lag-mean (per lon)          [NCL anom_type=1]
    kind="pct"     -> 100 * (counts - lagmean) / lagmean    [NCL anom_type=2]
    """
    counts = np.asarray(counts, dtype=float)
    if lag_centers is not None and (min_lag is not None or max_lag is not None):
        lag_centers = np.asarray(lag_centers, dtype=float)
        sel = np.ones(lag_centers.size, dtype=bool)
        if min_lag is not None:
            sel &= lag_centers >= min_lag
        if max_lag is not None:
            sel &= lag_centers <= max_lag
        lag_mean = np.nanmean(counts[sel], axis=0, keepdims=True)
    else:
        lag_mean = np.nanmean(counts, axis=0, keepdims=True)
    if kind == "total":
        return counts
    if kind == "anomaly":
        return counts - lag_mean
    if kind == "pct":
        with np.errstate(divide="ignore", invalid="ignore"):
            return 100.0 * (counts - lag_mean) / lag_mean
    raise ValueError("kind must be 'total', 'anomaly', or 'pct'")
