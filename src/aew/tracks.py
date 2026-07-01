"""Reader for cloud-system (CS) and convective-tracking (CT) point datasets.

The ISCCP CS/CT files used by the PhD store one entry per cloud system (or per
tracked-system time step): 1-D variables ``time``, ``lat``, ``lon`` plus per-system
attributes (radius/size, minimum temperature, number of convective clusters, lifetime,
etc.). The legacy scripts open the file and read ``->time``, ``->lon``, ``->lat`` and
the size/temperature variables directly.

This module loads such a file into a light ``Tracks`` container (numpy arrays), with a
region filter. It feeds ``aew.composites.hovmoller_event_counts`` (which wants
``time``, ``lon``, optionally ``lat`` and a per-system value ``z``).

The original data is not on hand, so the loader is written to the documented layout and
is exercised in tests against a synthetic dataset. When the real CS/CT (or a substitute
like TOOCAN / GridSat+PyFLEXTRKR; see docs/DATA_SOURCES.md) is in hand, point the loader
at it and map the variable names.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Tracks", "load_tracks", "tracks_from_arrays"]


@dataclass
class Tracks:
    """A set of cloud systems as parallel 1-D arrays."""

    time: np.ndarray  # datetime64
    lat: np.ndarray
    lon: np.ndarray
    variables: dict = field(default_factory=dict)  # name -> 1-D array (size, tmin, ...)

    def __len__(self):
        return self.time.size

    def filter_region(self, min_lat=None, max_lat=None, min_lon=None, max_lon=None):
        keep = np.ones(self.time.size, dtype=bool)
        if min_lat is not None:
            keep &= self.lat >= min_lat
        if max_lat is not None:
            keep &= self.lat <= max_lat
        if min_lon is not None:
            keep &= self.lon >= min_lon
        if max_lon is not None:
            keep &= self.lon <= max_lon
        return self._subset(keep)

    def filter(self, mask):
        """Subset by an arbitrary boolean mask (e.g. on a size or temperature variable)."""
        return self._subset(np.asarray(mask, dtype=bool))

    def _subset(self, keep):
        return Tracks(
            time=self.time[keep],
            lat=self.lat[keep],
            lon=self.lon[keep],
            variables={k: v[keep] for k, v in self.variables.items()},
        )


def tracks_from_arrays(time, lat, lon, **variables):
    """Build Tracks from in-memory arrays (used in tests and by substitute readers)."""
    return Tracks(
        time=np.asarray(time),
        lat=np.asarray(lat, dtype=float),
        lon=np.asarray(lon, dtype=float),
        variables={k: np.asarray(v) for k, v in variables.items()},
    )


def load_tracks(
    path, time_var="time", lat_var="lat", lon_var="lon", extra_vars=None
):
    """Load a CS/CT point dataset from NetCDF into a Tracks container.

    Parameters
    ----------
    path : str
        NetCDF file with 1-D time/lat/lon (one entry per system).
    time_var, lat_var, lon_var : str
        Coordinate/variable names.
    extra_vars : sequence of str, optional
        Additional per-system variables to load (e.g. size/radius, tmin, cslife).

    Returns
    -------
    Tracks
    """
    import xarray as xr  # local import so the core has no hard xarray dependency

    ds = xr.open_dataset(path, decode_times=True)
    time = np.asarray(ds[time_var].values)
    lat = np.asarray(ds[lat_var].values, dtype=float)
    lon = np.asarray(ds[lon_var].values, dtype=float)
    variables = {}
    if extra_vars:
        for name in extra_vars:
            variables[name] = np.asarray(ds[name].values)
    ds.close()
    return Tracks(time=time, lat=lat, lon=lon, variables=variables)
