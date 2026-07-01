"""Test the minimal ISCCP-style GridSat tracker on synthetic Tb fields."""

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("scipy")

from aew.data.gridsat_track import (detect_frame, regrid_tb, to_pyflextrkr_netcdf,
                                    track_systems)


def _cold_blob(lat, lon, clat, clon, radius_deg=3.0, depth=60.0, base=290.0):
    """A warm field with one cold (low-Tb) circular blob centered at (clat, clon)."""
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    d2 = (LA - clat) ** 2 + (LO - clon) ** 2
    return base - depth * np.exp(-d2 / (2 * radius_deg ** 2))


def test_detect_frame_finds_blob_with_core():
    lat = np.arange(0.0, 25.0, 0.25)
    lon = np.arange(-20.0, 40.0, 0.25)
    tb = _cold_blob(lat, lon, 10.0, 0.0, radius_deg=3.0, depth=80.0)  # min ~210K core
    labels, props = detect_frame(tb, lat, lon, shield=245.0, core=220.0)
    assert len(props) == 1
    p = next(iter(props.values()))
    assert p["has_core"] is True
    assert p["radius_km"] > 90.0  # a 3-deg blob is a few hundred km
    assert abs(p["lat"] - 10.0) < 1.0 and abs(p["lon"] - 0.0) < 1.0


def test_regrid_halves_then_quarters_grid():
    lat = np.arange(0.0, 8.0, 0.1)
    lon = np.arange(0.0, 8.0, 0.1)
    da = xr.DataArray(
        np.ones((1, lat.size, lon.size)) * 250.0,
        dims=("time", "lat", "lon"),
        coords={"time": [np.datetime64("2000-07-01")], "lat": lat, "lon": lon},
    )
    out = regrid_tb(da, factor=4)
    assert out.sizes["lat"] == lat.size // 4
    assert out.sizes["lon"] == lon.size // 4


def _moving_blob_da():
    lat = np.arange(0.0, 25.0, 0.25)
    lon = np.arange(-30.0, 40.0, 0.25)
    times = pd.date_range("2000-07-10", periods=5, freq="3h").values
    data = np.empty((times.size, lat.size, lon.size))
    for it in range(times.size):
        clon = 20.0 - 4.0 * it  # blob moves westward ~4 deg / 3h
        data[it] = _cold_blob(lat, lon, 10.0, clon, radius_deg=3.0, depth=80.0)
    return xr.DataArray(data, dims=("time", "lat", "lon"),
                        coords={"time": times, "lat": lat, "lon": lon})


def test_track_links_moving_blob_into_one_track():
    da = _moving_blob_da()
    times, tracks = track_systems(da, min_radius_km=90.0, overlap_thresh=0.05)
    # the single westward blob should link into one multi-time track
    durations = sorted(len(p) for p in tracks.values())
    assert durations[-1] == 5  # one track spans all 5 frames
    # and it moves westward: longitudes decrease along the track
    longest = max(tracks.values(), key=len)
    lons = [p["lon"] for _, p in longest]
    assert lons[0] > lons[-1]


def _fast_blob_da(step_deg):
    """A blob moving westward fast enough that consecutive footprints barely overlap."""
    lat = np.arange(0.0, 25.0, 0.25)
    lon = np.arange(-40.0, 40.0, 0.25)
    times = pd.date_range("2000-07-10", periods=5, freq="3h").values
    data = np.empty((times.size, lat.size, lon.size))
    for it in range(times.size):
        clon = 20.0 - step_deg * it
        data[it] = _cold_blob(lat, lon, 10.0, clon, radius_deg=2.0, depth=80.0)
    return xr.DataArray(data, dims=("time", "lat", "lon"),
                        coords={"time": times, "lat": lat, "lon": lon})


def test_projection_links_fast_mover_that_overlap_alone_fragments():
    # blob moves ~7 deg/frame; its ~2-deg-radius footprint does not overlap frame-to-frame
    da = _fast_blob_da(step_deg=7.0)
    # default speed -10 m/s at 10N over 3h ~ -1 deg, too slow; set default to match (~ -7 deg)
    # -7 deg/3h at 10N -> u = -7*111320*cos(10)/10800 ~ -71 m/s
    _, tr_proj = track_systems(da, min_radius_km=90.0, overlap_thresh=0.05,
                               project=True, default_u=-71.0)
    _, tr_noproj = track_systems(da, min_radius_km=90.0, overlap_thresh=0.05,
                                 project=False)
    assert max(len(p) for p in tr_proj.values()) > max(len(p) for p in tr_noproj.values())
    assert max(len(p) for p in tr_proj.values()) == 5  # projection links all 5 frames


def test_tracker_output_roundtrips_through_adapter(tmp_path):
    from aew.data.pyflextrkr import from_pyflextrkr

    da = _moving_blob_da()
    times, tracks = track_systems(da, min_radius_km=90.0, overlap_thresh=0.05)
    path = tmp_path / "gridsat_tracks.nc"
    to_pyflextrkr_netcdf(times, tracks, str(path))
    tr = from_pyflextrkr(str(path))
    assert len(tr) >= 5  # at least the 5 points of the main track
    assert "radius_km" in tr.variables
    assert tr.variables["radius_km"].min() >= 0
