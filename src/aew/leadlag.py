"""Time-domain lead-lag between the AEW wave and convection.

The wave-following composite shows where convection sits relative to the trough, but a
spatial phase offset alone does not settle the ordering objection (trough and mesoscale
convective system as two faces of one convectively coupled disturbance). This module builds
the temporal version: at a fixed meridian, a continuous wave-amplitude series (700 hPa
curvature vorticity, band-mean over a latitude band) and a convection series (cold-cloud
system count in a longitude/latitude box) are each band-passed to the AEW period band, then
cross-correlated. The sign of the lag at peak correlation says which field leads in time.

Sign convention (verified in the tests): ``lag_cross_correlation(w, c, ...)`` returns
``R[k] = corr(w[t], c[t + k])``. A peak at POSITIVE lag means convection best matches the
EARLIER wave, so convection lags the wave (the wave leads). A peak at NEGATIVE lag means
convection leads the wave. At a single meridian this lag reflects the trough-convection phase
offset carried by the propagating wave, so it is read as a phase relationship and its change
with longitude, not as evidence for a forcing direction on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "box_count_series",
    "band_mean_series",
    "lag_cross_correlation",
    "peak_lag",
]


def _to_ns(times):
    return pd.DatetimeIndex(times).values.astype("datetime64[ns]").astype("int64")


def box_count_series(cs_time, cs_lon, cs_lat, grid_times, ref_lon, dlon,
                     lat_lo, lat_hi):
    """Count cloud systems in a fixed box centered on ``ref_lon`` at each grid time.

    Each cloud-system observation is split between its two bracketing grid times in
    proportion to distance (linear, area-preserving binning), so a finer-cadence system
    record (e.g. 3-hourly) is placed onto a coarser regular grid (e.g. 6-hourly) without a
    systematic time shift: an observation exactly halfway between two grid times contributes
    half a count to each, rather than being rounded to one side. The total assigned weight
    equals the number of in-box observations whose two bracketing grid nodes both lie within
    the grid; an observation past the last node loses the fraction that would fall beyond it
    (with a grid spanning the record this affects nothing). Only systems with trough-relative
    longitude
    within ``+/- dlon`` of ``ref_lon`` (wrapped to [-180, 180)) and latitude in
    ``[lat_lo, lat_hi]`` are counted.

    Parameters
    ----------
    cs_time : array of datetime64
    cs_lon, cs_lat : arrays (degrees, lon in -180..180)
    grid_times : array of datetime64, uniformly spaced and sorted
    ref_lon : float          box center longitude
    dlon : float             half-width of the box in longitude
    lat_lo, lat_hi : float   latitude band

    Returns
    -------
    counts : ndarray (n_grid,) float
    """
    grid = _to_ns(np.asarray(grid_times))
    if grid.size < 2:
        raise ValueError("grid_times needs at least two points to infer spacing")
    dt = int(np.median(np.diff(grid)))
    if dt <= 0:
        raise ValueError("grid_times must be sorted ascending with positive spacing")

    cs = _to_ns(cs_time)
    clon = np.asarray(cs_lon, dtype=float)
    clat = np.asarray(cs_lat, dtype=float)
    rel = (clon - ref_lon + 180.0) % 360.0 - 180.0
    inbox = (np.abs(rel) <= dlon) & (clat >= lat_lo) & (clat <= lat_hi)

    counts = np.zeros(grid.size, dtype=float)
    if not inbox.any():
        return counts
    sel = cs[inbox]
    # linear (area-preserving) split between the two bracketing grid nodes
    p = (sel - grid[0]) / dt                 # float grid position
    lo = np.floor(p).astype(np.int64)
    frac = p - lo                            # 0 on a node, 0.5 exactly halfway
    for node, wt in ((lo, 1.0 - frac), (lo + 1, frac)):
        keep = (node >= 0) & (node < grid.size)
        np.add.at(counts, node[keep], wt[keep])
    return counts


def band_mean_series(field, lat, lat_lo, lat_hi):
    """Latitude-band mean of a (time, lat, lon) field over ``[lat_lo, lat_hi]``.

    NaN-aware (calm-wind cells in curvature vorticity are NaN). Returns (time, lon).
    """
    lat = np.asarray(lat, dtype=float)
    band = (lat >= lat_lo) & (lat <= lat_hi)
    if not band.any():
        raise ValueError(f"no latitudes in [{lat_lo}, {lat_hi}]")
    return np.nanmean(np.asarray(field, dtype=float)[:, band, :], axis=1)


def lag_cross_correlation(w, c, max_lag):
    """Normalized lead-lag cross-correlation ``R[k] = corr(w[t], c[t + k])``.

    For each integer lag ``k`` in ``-max_lag .. max_lag`` the paired samples ``(w[t],
    c[t + k])`` over the valid overlap are used, dropping any pair with a NaN in either
    series, and a Pearson correlation is computed on that overlap (each lag demeaned by its
    own paired means). A positive peak lag means convection lags the wave (wave leads).

    Parameters
    ----------
    w, c : 1-D arrays, same length, uniform time step
    max_lag : int   maximum lag in samples

    Returns
    -------
    lags : ndarray (2*max_lag + 1,) int
    R : ndarray (2*max_lag + 1,) float   NaN where fewer than 3 valid pairs
    """
    w = np.asarray(w, dtype=float)
    c = np.asarray(c, dtype=float)
    if w.shape != c.shape or w.ndim != 1:
        raise ValueError("w and c must be 1-D arrays of the same length")
    n = w.size
    lags = np.arange(-max_lag, max_lag + 1)
    R = np.full(lags.size, np.nan)
    for j, k in enumerate(lags):
        if k >= 0:
            a, b = w[: n - k], c[k:]
        else:
            a, b = w[-k:], c[: n + k]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3:
            continue
        aa = a[m] - a[m].mean()
        bb = b[m] - b[m].mean()
        denom = np.sqrt((aa * aa).sum() * (bb * bb).sum())
        if denom > 0:
            R[j] = float((aa * bb).sum() / denom)
    return lags, R


def peak_lag(lags, R):
    """Lag of the maximum of ``R`` with parabolic sub-sample refinement.

    Returns ``(lag_at_peak, R_at_peak)``. The refinement fits a parabola to the peak sample
    and its two neighbors; at the array ends it returns the discrete peak. NaN samples are
    ignored for locating the discrete maximum.
    """
    lags = np.asarray(lags, dtype=float)
    R = np.asarray(R, dtype=float)
    if not np.isfinite(R).any():
        return np.nan, np.nan
    i = int(np.nanargmax(R))
    if 0 < i < R.size - 1 and np.isfinite(R[i - 1]) and np.isfinite(R[i + 1]):
        y0, y1, y2 = R[i - 1], R[i], R[i + 1]
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        delta = float(np.clip(delta, -1.0, 1.0))
        step = lags[1] - lags[0]
        return float(lags[i] + delta * step), float(y1 - 0.25 * (y0 - y2) * delta)
    return float(lags[i]), float(R[i])
