"""Adapter: PyFLEXTRKR track output -> aew.tracks.Tracks.

PyFLEXTRKR writes a "trackstats" NetCDF with per-track statistics on a (tracks, times)
grid: each track is a row, each column a relative time step, and invalid (track, time)
slots are filled. Key variables (names are configurable here because they vary a little
between PyFLEXTRKR configs/versions):

  base_time(tracks, times)   time of each track point (datetime64 after decode, or epoch)
  meanlat(tracks, times)     centroid latitude
  meanlon(tracks, times)     centroid longitude (0-360 or -180..180)
  area(tracks, times)        cold-cloud-shield (CCS) area, km^2
  track_duration(tracks)     number of valid times (lifetime in frames)

This adapter flattens to one row per valid (track, time) cloud-system point -- the form
``aew.composites.hovmoller_event_counts`` / ``hovmoller_event_field`` expect -- computing
the equivalent radius R = sqrt(area/pi) and attaching track_id and duration. Apply the
ISCCP-style size cut (e.g. 90 km radius) via ``min_radius_km``.

See docs/GRIDSAT_CT_PLAN.md. Tested against a synthetic file shaped like PyFLEXTRKR output.
"""

from __future__ import annotations

import numpy as np

from ..tracks import Tracks


def from_pyflextrkr(
    path,
    time_var="base_time",
    lat_var="meanlat",
    lon_var="meanlon",
    area_var="area",
    duration_var="track_duration",
    extra_vars=None,
    min_radius_km=None,
    wrap_lon=True,
):
    """Load a PyFLEXTRKR trackstats NetCDF into a per-system-point ``Tracks``.

    Parameters
    ----------
    path : str
        PyFLEXTRKR trackstats / mcs_tracks NetCDF.
    time_var, lat_var, lon_var, area_var, duration_var : str
        Variable names in the file (override per PyFLEXTRKR config).
    extra_vars : sequence of str, optional
        Additional (tracks, times) variables to carry through (e.g. core_area, min Tb).
    min_radius_km : float, optional
        Keep only systems with equivalent radius >= this (e.g. 90.0 for ISCCP CT).
    wrap_lon : bool
        Convert longitudes > 180 to the -180..180 convention (AEW domain is -45..75).

    Returns
    -------
    Tracks
        time, lat, lon are 1-D over all valid (track, time) points; ``variables`` holds
        radius_km, area_km2, track_id, track_duration, and any extra_vars.
    """
    import xarray as xr

    ds = xr.open_dataset(path, decode_times=True)

    base = ds[time_var]
    lat = ds[lat_var]
    lon = ds[lon_var]
    # valid points: real time AND finite centroid
    time_vals = base.values
    valid = np.isfinite(lat.values) & np.isfinite(lon.values)
    if np.issubdtype(time_vals.dtype, np.datetime64):
        valid &= ~np.isnat(time_vals)
    else:
        valid &= np.isfinite(time_vals)

    ntracks, ntimes = lat.shape
    track_idx = np.broadcast_to(np.arange(ntracks)[:, None], (ntracks, ntimes))

    def flat(a):
        return np.asarray(a).reshape(-1)[valid.reshape(-1)]

    out_time = flat(time_vals)
    out_lat = flat(lat.values).astype(float)
    out_lon = flat(lon.values).astype(float)
    if wrap_lon:
        out_lon = np.where(out_lon > 180.0, out_lon - 360.0, out_lon)

    variables = {"track_id": flat(track_idx).astype(int)}

    if area_var in ds:
        area = flat(ds[area_var].values).astype(float)
        variables["area_km2"] = area
        variables["radius_km"] = np.sqrt(area / np.pi)
    if duration_var in ds:
        dur = np.asarray(ds[duration_var].values)  # per-track (1-D)
        variables["track_duration"] = dur[flat(track_idx).astype(int)]
    if extra_vars:
        for name in extra_vars:
            variables[name] = flat(ds[name].values)

    ds.close()
    tr = Tracks(time=out_time, lat=out_lat, lon=out_lon, variables=variables)

    if min_radius_km is not None and "radius_km" in tr.variables:
        tr = tr.filter(tr.variables["radius_km"] >= min_radius_km)
    return tr
