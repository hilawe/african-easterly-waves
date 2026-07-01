"""African easterly wave trough identification from the 700 hPa wind.

A focused, ERA5-based reimplementation of the curvature-vorticity trough diagnostic behind
the NCEI African Easterly Wave Climatology (Belanger et al.). It exists to cover the two
gaps in that product for this project: it can run to the present, and it can reach east of
40 E to include the Ethiopian Highlands and Darfur genesis region.

Curvature vorticity isolates the turning of the flow (the trough axis) from the shear, and
is less sensitive to the local wind distortion of convection than relative vorticity, which
is why the wave catalogue uses it. For a Northern Hemisphere AEW trough the curvature
vorticity is cyclonic (positive).

This module identifies trough POINTS (time, lat, lon), which is what the trough-relative
composite needs; it does not link them into trajectories. Trough points are local maxima of
the curvature-vorticity anomaly along longitude that exceed a percentile threshold and lie
in weak or easterly zonal flow, following the AEWC criteria.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_R = 6.371e6  # m


def _grads(f, lat, lon):
    """Spherical gradients d f / dx (eastward) and d f / dy (northward) for f on
    (..., nlat, nlon) with lat, lon in degrees (ascending)."""
    lat_r = np.deg2rad(np.asarray(lat, float))
    lon_r = np.deg2rad(np.asarray(lon, float))
    df_dlon = np.gradient(f, lon_r, axis=-1)
    df_dlat = np.gradient(f, lat_r, axis=-2)
    coslat = np.cos(lat_r).reshape(lat_r.size, 1)
    dfdx = df_dlon / (EARTH_R * coslat)
    dfdy = df_dlat / EARTH_R
    return dfdx, dfdy


def relative_vorticity(u, v, lat, lon):
    """Relative vorticity dv/dx - du/dy."""
    dvdx, _ = _grads(v, lat, lon)
    _, dudy = _grads(u, lat, lon)
    return dvdx - dudy


def curvature_vorticity(u, v, lat, lon, calm=0.5):
    """Curvature vorticity of the horizontal wind, the part from turning of the flow.

    zeta_curv = (u^2 v_x - u v u_x + u v v_y - v^2 u_y) / (u^2 + v^2), derived from
    zeta_curv = u dphi/dx + v dphi/dy with phi = atan2(v, u). NaN where wind speed is below
    ``calm`` m/s (direction undefined).
    """
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    dudx, dudy = _grads(u, lat, lon)
    dvdx, dvdy = _grads(v, lat, lon)
    sp2 = u * u + v * v
    num = u * u * dvdx - u * v * dudx + u * v * dvdy - v * v * dudy
    with np.errstate(invalid="ignore", divide="ignore"):
        zc = np.where(sp2 >= calm * calm, num / sp2, np.nan)
    return zc


def anomaly_from_timemean(field):
    """Curvature-vorticity anomaly as the departure from the time mean (axis 0)."""
    field = np.asarray(field, float)
    return field - np.nanmean(field, axis=0, keepdims=True)


def trough_points(crv_anom, u, time, lat, lon, pctl=66.0, u_max=2.5,
                   lat_range=None, min_anom=None):
    """Identify AEW trough points as (time, lat, lon).

    A trough point is a local maximum of the curvature-vorticity anomaly along longitude
    that (a) exceeds the ``pctl`` percentile of all valid anomalies (or ``min_anom`` if
    given) and (b) lies in weak or easterly zonal flow, u <= ``u_max``. This mirrors the
    AEWC trough criteria (curvature-vorticity anomaly above its 66th percentile, weak
    zonal wind) without the advection-of-anomaly refinement.

    Parameters
    ----------
    crv_anom, u : ndarray (ntime, nlat, nlon)
    time : array of datetime64 (ntime,)
    lat, lon : 1-D degree coordinates
    pctl : percentile threshold on the anomaly (Belanger uses 66)
    u_max : maximum zonal wind (m/s) for a trough point
    lat_range : optional (lo, hi) latitude band to restrict the search
    min_anom : optional absolute anomaly threshold (overrides the percentile)

    Returns
    -------
    (times, lats, lons) parallel 1-D arrays of trough points.
    """
    crv_anom = np.asarray(crv_anom, float)
    u = np.asarray(u, float)
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    thr = min_anom if min_anom is not None else np.nanpercentile(crv_anom, pctl)

    lat_ok = np.ones(lat.size, dtype=bool)
    if lat_range is not None:
        lat_ok = (lat >= lat_range[0]) & (lat <= lat_range[1])

    left = crv_anom[:, :, :-2]
    mid = crv_anom[:, :, 1:-1]
    right = crv_anom[:, :, 2:]
    um = u[:, :, 1:-1]
    is_trough = (mid > left) & (mid >= right) & (mid >= thr) & (um <= u_max)
    is_trough &= lat_ok[None, :, None]
    is_trough &= np.isfinite(mid)

    ti, li, xi = np.where(is_trough)
    xi = xi + 1  # shift back to full-longitude index
    return time[ti], lat[li], lon[xi]


def trough_axis(crv_anom, u, time, lat, lon, pctl=66.0, u_max=2.5, lat_range=None,
                smooth_lat=3, smooth_lon=8, min_anom=None):
    """Refined trough detector: 2D-smooth the anomaly to the wave scale, then locate the
    longitudinal maximum with SUB-GRID (parabolic) interpolation so trough longitudes are
    continuous rather than snapped to grid cells (which produced banding in the simple
    ``trough_points``). Same threshold and weak-zonal-wind criteria otherwise.

    smooth_lat, smooth_lon are boxcar widths in grid cells. Returns (times, lats, lons)
    with lons on a continuous axis. Pair with ``coherence_filter`` to drop transient,
    non-propagating maxima.
    """
    from scipy.ndimage import uniform_filter

    crv_anom = np.asarray(crv_anom, float)
    u = np.asarray(u, float)
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    ca = uniform_filter(np.nan_to_num(crv_anom), size=(1, smooth_lat, smooth_lon),
                        mode="nearest")
    thr = min_anom if min_anom is not None else np.nanpercentile(ca, pctl)
    dlon = abs(lon[1] - lon[0])
    lat_ok = (np.ones(lat.size, bool) if lat_range is None
              else (lat >= lat_range[0]) & (lat <= lat_range[1]))

    left, mid, right = ca[:, :, :-2], ca[:, :, 1:-1], ca[:, :, 2:]
    um = u[:, :, 1:-1]
    ismax = (mid > left) & (mid >= right) & (mid >= thr) & (um <= u_max)
    ismax &= lat_ok[None, :, None] & np.isfinite(mid)
    ti, li, xi = np.where(ismax)
    xi = xi + 1
    l = ca[ti, li, xi - 1]; m = ca[ti, li, xi]; r = ca[ti, li, xi + 1]
    denom = l - 2 * m + r
    delta = np.where(np.abs(denom) > 1e-30, 0.5 * (l - r) / denom, 0.0)
    delta = np.clip(delta, -1.0, 1.0)  # sub-grid peak offset in cells
    lon_peak = lon[xi] + delta * dlon
    return time[ti], lat[li], lon_peak, m  # m = peak anomaly amplitude


def collapse_to_axes(times, lats, lons, amps=None, max_gap=5.0):
    """Collapse per-latitude trough points into one trough AXIS centroid per wave per time.

    The detector fires once per latitude row, so a single meridionally-elongated trough
    appears as a vertical string of points at nearly the same longitude. AEWC reports one
    centroid per wave, so this groups, within each time, points whose longitudes are within
    ``max_gap`` of each other. The centroid latitude and longitude are AMPLITUDE-WEIGHTED by
    ``amps`` (the curvature-vorticity peak), so the axis sits at each wave's trough core
    rather than the middle of the search band. Returns (times, lats, lons).
    """
    times = pd.DatetimeIndex(times)
    tv = times.values.astype("datetime64[ns]").astype("int64")
    lons = np.asarray(lons, float)
    lats = np.asarray(lats, float)
    w = np.ones(lons.size) if amps is None else np.clip(np.asarray(amps, float), 0, None)
    out_t, out_la, out_lo = [], [], []
    for tvu in np.unique(tv):
        m = tv == tvu
        lo, la, ww = lons[m], lats[m], w[m]
        if lo.size == 0:
            continue
        order = np.argsort(lo)
        lo, la, ww = lo[order], la[order], ww[order]
        # span-limited grouping: start a new group when the point is more than max_gap from
        # the current group's START (not just its previous point), so a chain of small gaps
        # cannot merge waves that span more than max_gap of longitude.
        splits = []
        start = lo[0]
        for i in range(1, lo.size):
            if lo[i] - start > max_gap:
                splits.append(i)
                start = lo[i]
        for glo, gla, gw in zip(np.split(lo, splits), np.split(la, splits), np.split(ww, splits)):
            wsum = gw.sum()
            if wsum <= 0:
                gw = np.ones_like(gw); wsum = gw.sum()
            out_t.append(tvu)
            out_lo.append(np.sum(glo * gw) / wsum)
            out_la.append(np.sum(gla * gw) / wsum)
    return (np.asarray(out_t, dtype="datetime64[ns]"),
            np.asarray(out_la, float), np.asarray(out_lo, float))


def coherence_filter(times, lats, lons, max_dlon=8.0, max_dlat=3.0):
    """Keep only trough points that have a neighbor at the adjacent time step within
    (max_dlon, max_dlat). This enforces temporal coherence (a propagating wave), removing
    isolated transient maxima. Returns a boolean mask aligned with the inputs.
    """
    times = pd.DatetimeIndex(times)
    lons = np.asarray(lons, float)
    lats = np.asarray(lats, float)
    tvals = times.values.astype("datetime64[ns]").astype("int64")
    uniq = np.unique(tvals)
    idx_by_t = {tv: np.where(tvals == tv)[0] for tv in uniq}
    keep = np.zeros(times.size, dtype=bool)
    for k in range(uniq.size - 1):
        ia = idx_by_t[uniq[k]]
        ib = idx_by_t[uniq[k + 1]]
        if ia.size == 0 or ib.size == 0:
            continue
        dlon = np.abs(lons[ia][:, None] - lons[ib][None, :])
        dlat = np.abs(lats[ia][:, None] - lats[ib][None, :])
        match = (dlon <= max_dlon) & (dlat <= max_dlat)
        keep[ia[np.any(match, axis=1)]] = True
        keep[ib[np.any(match, axis=0)]] = True
    return keep
