"""Kinematic isobaric back-trajectories on a gridded wind field.

The moisture-source question behind the developing/non-developing composite (did the
pre-trough moisture arrive by advection or appear locally) needs parcel histories, not
box statistics. This module integrates parcel positions backward in time through a
6-hourly single-level wind record, with linear interpolation in time and bilinear
interpolation in space, fourth-order Runge-Kutta stepping, and spherical kinematics
(dlat = v dt / R, dlon = u dt / (R cos lat)).

Isobaric limitation, stated once here and echoed by callers: parcels stay on the level
of the wind field (no vertical motion), which is the standard first pass for
moisture-source scoping but cannot represent ascent/descent along the monsoon inflow.

All functions are pure and vectorized over parcels; parcels may have different seed
times (each carries its own clock).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["Gridded", "back_trajectories", "classify_origin"]

M_PER_DEG = 111_190.0  # meters per degree of latitude


def _ns_hours(times):
    return pd.DatetimeIndex(times).values.astype("datetime64[ns]").astype("int64") / 3.6e12


class Gridded:
    """A (time, lat, lon) scalar field with linear-in-time, bilinear-in-space sampling.

    ``lat`` must be ascending. ``lon`` may be regional (ascending) or global; when the
    grid spans ~360 degrees the longitude axis is treated as cyclic. Samples outside the
    time range or a regional longitude/latitude range return NaN.
    """

    def __init__(self, times, lat, lon, values):
        self.t = np.asarray(_ns_hours(times), dtype=float)   # hours since epoch
        if self.t.size > 1 and not np.all(np.diff(self.t) > 0):
            raise ValueError("times must be sorted ascending without duplicates")
        self.lat = np.asarray(lat, dtype=float)
        if self.lat.size > 1 and self.lat[0] > self.lat[-1]:
            raise ValueError("lat must be ascending (flip the field first)")
        self.lon = np.asarray(lon, dtype=float)
        self.v = np.asarray(values, dtype=float)
        if self.v.shape != (self.t.size, self.lat.size, self.lon.size):
            raise ValueError("values must be shaped (ntime, nlat, nlon)")
        dlon = np.median(np.diff(self.lon))
        self.cyclic = abs(self.lon.size * dlon - 360.0) <= 1.5 * dlon

    def _axis_frac(self, coord, x, cyclic=False):
        """Lower index and fraction along one ascending axis; NaN-safe via clipping flag."""
        n = coord.size
        if cyclic:
            step = coord[1] - coord[0]
            pos = (x - coord[0]) / step
            finite = np.isfinite(pos)
            i0 = np.floor(np.where(finite, pos, 0.0)).astype(np.int64)
            frac = np.where(finite, pos - i0, np.nan)
            return i0 % n, (i0 + 1) % n, frac, finite
        i0 = np.searchsorted(coord, x, side="right") - 1
        # the exact final node is inside the domain (frac = 1 on the last interval)
        inside = (x >= coord[0]) & (x <= coord[-1])
        i0c = np.clip(i0, 0, n - 2)
        frac = (x - coord[i0c]) / (coord[i0c + 1] - coord[i0c])
        return i0c, i0c + 1, frac, inside

    def sample(self, t_hours, plat, plon):
        """Field value at per-parcel (time, lat, lon). Arrays broadcast to one shape."""
        t_hours = np.asarray(t_hours, dtype=float)
        plat = np.asarray(plat, dtype=float)
        plon = np.asarray(plon, dtype=float)
        if self.cyclic:
            plon = (plon - self.lon[0]) % 360.0 + self.lon[0]

        k0, k1, ft, t_in = self._axis_frac(self.t, t_hours)
        j0, j1, fy, y_in = self._axis_frac(self.lat, plat)
        i0, i1, fx, x_in = self._axis_frac(self.lon, plon, cyclic=self.cyclic)

        def _plane(k):
            return ((1 - fy) * (1 - fx) * self.v[k, j0, i0]
                    + (1 - fy) * fx * self.v[k, j0, i1]
                    + fy * (1 - fx) * self.v[k, j1, i0]
                    + fy * fx * self.v[k, j1, i1])

        out = (1 - ft) * _plane(k0) + ft * _plane(k1)
        bad = ~(t_in & y_in & x_in)
        if bad.any():
            out = np.where(bad, np.nan, out)
        return out


def back_trajectories(u, v, seed_times, seed_lats, seed_lons, hours=48.0, dt_hours=1.0):
    """Integrate parcels backward for ``hours`` from per-parcel seeds.

    Parameters
    ----------
    u, v : Gridded            wind components (m/s) on the same grid
    seed_times : datetime64 array (nparcel,)
    seed_lats, seed_lons : arrays (nparcel,)
    hours, dt_hours : float   integration span and step

    Returns
    -------
    elapsed : ndarray (nstep+1,)          hours BEFORE the seed (0, dt, 2 dt, ...)
    lat, lon : ndarrays (nstep+1, nparcel) positions; row 0 is the seed. A parcel that
        leaves the grid (regional domain or time range) holds NaN from there on.
    """
    t0 = np.asarray(_ns_hours(np.asarray(seed_times)), dtype=float)
    plat = np.asarray(seed_lats, dtype=float).copy()
    plon = np.asarray(seed_lons, dtype=float).copy()
    nstep = int(round(hours / dt_hours))
    lat_hist = np.full((nstep + 1, plat.size), np.nan)
    lon_hist = np.full((nstep + 1, plat.size), np.nan)
    lat_hist[0], lon_hist[0] = plat, plon

    def tend(t_h, la, lo):
        """(dlat/dt, dlon/dt) in deg per hour at absolute time t_h (backward sign)."""
        uu = u.sample(t_h, la, lo)
        vv = v.sample(t_h, la, lo)
        dlat = -vv * 3600.0 / M_PER_DEG
        coslat = np.cos(np.deg2rad(np.clip(la, -89.0, 89.0)))
        dlon = -uu * 3600.0 / (M_PER_DEG * coslat)
        return dlat, dlon

    dt = dt_hours
    for s in range(nstep):
        t_h = t0 - s * dt  # absolute hours at the current position
        k1a, k1o = tend(t_h, plat, plon)
        k2a, k2o = tend(t_h - dt / 2, plat + k1a * dt / 2, plon + k1o * dt / 2)
        k3a, k3o = tend(t_h - dt / 2, plat + k2a * dt / 2, plon + k2o * dt / 2)
        k4a, k4o = tend(t_h - dt, plat + k3a * dt, plon + k3o * dt)
        plat = plat + dt * (k1a + 2 * k2a + 2 * k3a + k4a) / 6.0
        plon = plon + dt * (k1o + 2 * k2o + 2 * k3o + k4o) / 6.0
        lat_hist[s + 1], lon_hist[s + 1] = plat, plon

    elapsed = dt_hours * np.arange(nstep + 1)
    return elapsed, lat_hist, lon_hist


def classify_origin(seed_lat, seed_lon, origin_lat, origin_lon,
                    dlat_thresh=3.0, dlon_thresh=5.0):
    """Displacement-based origin sector for each parcel.

    The MERIDIONAL displacement decides first (the monsoonal-advection question is
    north-south): an origin more than ``dlat_thresh`` degrees south of the seed is
    "south", north is "north"; otherwise a zonal displacement beyond ``dlon_thresh``
    gives "east"/"west"; anything else is "local". NaN origins (parcel left the domain)
    return "lost".
    """
    dlat = np.asarray(origin_lat, dtype=float) - np.asarray(seed_lat, dtype=float)
    dlon = np.asarray(origin_lon, dtype=float) - np.asarray(seed_lon, dtype=float)
    out = np.full(dlat.shape, "local", dtype=object)
    out[dlat <= -dlat_thresh] = "south"
    out[dlat >= dlat_thresh] = "north"
    zonal = np.abs(dlat) < dlat_thresh
    out[zonal & (dlon >= dlon_thresh)] = "east"
    out[zonal & (dlon <= -dlon_thresh)] = "west"
    out[~np.isfinite(dlat) | ~np.isfinite(dlon)] = "lost"
    return out
