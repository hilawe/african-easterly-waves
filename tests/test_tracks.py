import numpy as np
import pandas as pd

from aew.tracks import Tracks, load_tracks, tracks_from_arrays


def _sample():
    time = pd.date_range("2000-07-01", periods=6, freq="D").values
    lat = np.array([5.0, 12.0, 20.0, -5.0, 10.0, 30.0])
    lon = np.array([-10.0, 0.0, 10.0, 20.0, 50.0, 70.0])
    size = np.array([120.0, 80.0, 250.0, 60.0, 300.0, 90.0])
    return tracks_from_arrays(time, lat, lon, size=size)


def test_tracks_from_arrays_and_len():
    tr = _sample()
    assert len(tr) == 6
    assert "size" in tr.variables
    assert tr.variables["size"].shape == (6,)


def test_filter_region_lat_lon():
    tr = _sample().filter_region(min_lat=5.0, max_lat=15.0, min_lon=-40.0, max_lon=60.0)
    # keep lat in [5,15] and lon in [-40,60]: indices 0(5,-10),1(12,0),4(10,50)
    assert len(tr) == 3
    np.testing.assert_array_equal(tr.lat, [5.0, 12.0, 10.0])
    np.testing.assert_array_equal(tr.variables["size"], [120.0, 80.0, 300.0])


def test_filter_by_variable_mask():
    tr = _sample()
    mcs = tr.filter(tr.variables["size"] > 100.0)  # MCS-like
    assert len(mcs) == 3
    np.testing.assert_array_equal(mcs.variables["size"], [120.0, 250.0, 300.0])


def test_load_tracks_roundtrip(tmp_path):
    import xarray as xr

    tr = _sample()
    ds = xr.Dataset(
        {
            "lat": ("n", tr.lat),
            "lon": ("n", tr.lon),
            "size": ("n", tr.variables["size"]),
        },
        coords={"time": ("n", tr.time)},
    )
    path = tmp_path / "cs.nc"
    ds.to_netcdf(path)
    loaded = load_tracks(str(path), extra_vars=["size"])
    assert len(loaded) == 6
    np.testing.assert_array_equal(loaded.lon, tr.lon)
    np.testing.assert_array_equal(loaded.variables["size"], tr.variables["size"])
