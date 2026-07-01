"""Test the PyFLEXTRKR -> Tracks adapter against a synthetic trackstats file."""

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from aew.data.pyflextrkr import from_pyflextrkr


def _make_trackstats(path):
    """A tiny PyFLEXTRKR-style trackstats file: 3 tracks x 4 times, some invalid slots."""
    ntracks, ntimes = 3, 4
    # PyFLEXTRKR stores base_time as float epoch seconds with NaN for invalid slots
    t0_sec = (np.datetime64("2000-07-10T00:00") - np.datetime64("1970-01-01")) / np.timedelta64(1, "s")
    dt = 3 * 3600.0  # 3-hourly
    base = np.full((ntracks, ntimes), np.nan)
    meanlat = np.full((ntracks, ntimes), np.nan)
    meanlon = np.full((ntracks, ntimes), np.nan)
    area = np.full((ntracks, ntimes), np.nan)
    duration = np.array([4, 2, 3])  # valid lengths per track

    # track 0: 4 valid points, big systems (radius ~ 120 km)
    for j in range(4):
        base[0, j] = t0_sec + j * dt
        meanlat[0, j] = 10.0
        meanlon[0, j] = 5.0 + j
        area[0, j] = np.pi * (120.0 ** 2)
    # track 1: 2 valid points, SMALL systems (radius ~ 50 km -> below a 90 km cut)
    for j in range(2):
        base[1, j] = t0_sec + j * dt
        meanlat[1, j] = 12.0
        meanlon[1, j] = 350.0  # 0-360 lon -> should wrap to -10
        area[1, j] = np.pi * (50.0 ** 2)
    # track 2: 3 valid points, big (radius ~ 100 km)
    for j in range(3):
        base[2, j] = t0_sec + j * dt
        meanlat[2, j] = 8.0
        meanlon[2, j] = 20.0
        area[2, j] = np.pi * (100.0 ** 2)

    ds = xr.Dataset(
        {
            "base_time": (("tracks", "times"), base),
            "meanlat": (("tracks", "times"), meanlat),
            "meanlon": (("tracks", "times"), meanlon),
            "area": (("tracks", "times"), area),
            "track_duration": (("tracks",), duration),
        }
    )
    ds["base_time"].attrs["units"] = "seconds since 1970-01-01"
    ds.to_netcdf(path)


def test_adapter_flattens_valid_points(tmp_path):
    p = tmp_path / "trackstats.nc"
    _make_trackstats(p)
    tr = from_pyflextrkr(str(p))
    # 4 + 2 + 3 = 9 valid (track, time) points
    assert len(tr) == 9
    assert "radius_km" in tr.variables
    assert "track_id" in tr.variables
    # radii recovered from area
    np.testing.assert_allclose(sorted(set(np.round(tr.variables["radius_km"]))), [50, 100, 120])


def test_adapter_wraps_longitude(tmp_path):
    p = tmp_path / "trackstats.nc"
    _make_trackstats(p)
    tr = from_pyflextrkr(str(p), wrap_lon=True)
    # track 1's lon 350 -> -10
    assert tr.lon.min() == pytest.approx(-10.0)
    assert tr.lon.max() <= 180.0


def test_adapter_min_radius_cut(tmp_path):
    p = tmp_path / "trackstats.nc"
    _make_trackstats(p)
    tr = from_pyflextrkr(str(p), min_radius_km=90.0)
    # the small (50 km) track-1 points are dropped: 4 + 3 = 7 remain
    assert len(tr) == 7
    assert tr.variables["radius_km"].min() >= 90.0


def test_adapter_track_id_and_duration(tmp_path):
    p = tmp_path / "trackstats.nc"
    _make_trackstats(p)
    tr = from_pyflextrkr(str(p))
    # track_id repeats per track; duration carried per point
    assert set(tr.variables["track_id"]) == {0, 1, 2}
    # track 0 has duration 4 at all its points
    dur0 = tr.variables["track_duration"][tr.variables["track_id"] == 0]
    assert np.all(dur0 == 4)


def test_adapter_feeds_hovmoller(tmp_path):
    # end-to-end: adapter output drives the CS count Hovmoller
    from aew.composites import hovmoller_event_counts, lag_axis

    p = tmp_path / "trackstats.nc"
    _make_trackstats(p)
    tr = from_pyflextrkr(str(p), min_radius_km=90.0)
    lon_centers = np.arange(-40.0, 80.0, 4.0)
    lag_centers = lag_axis(-6, 6, 1).astype(float)
    events = pd.DatetimeIndex(["2000-07-10"]).values
    counts = hovmoller_event_counts(
        events, tr.time, tr.lon, lon_centers, lag_centers,
        cs_lat=tr.lat, min_lat=5.0, max_lat=15.0,
    )
    assert counts.shape == (lag_centers.size, lon_centers.size)
    assert counts.sum() == 7  # 7 in-band systems at valid lags
