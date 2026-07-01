import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from aew.composites import map_event_counts


def test_map_event_counts_bins_at_lag0():
    lon = np.array([0.0, 4.0, 8.0])
    lat = np.array([5.0, 10.0, 15.0])
    event = pd.DatetimeIndex(["2000-07-10"]).values
    # one system at (lon4, lat10) on the event day -> lag 0 bin (lat idx1, lon idx1)
    cs_time = pd.DatetimeIndex(["2000-07-10"]).values
    counts = map_event_counts(event, cs_time, np.array([4.0]), np.array([10.0]),
                              lon, lat, lag=0.0, half_window=0.5)
    assert counts.shape == (3, 3)
    assert counts[1, 1] == 1
    assert counts.sum() == 1


def test_map_event_counts_lag_window_excludes_far_systems():
    lon = np.array([0.0, 4.0])
    lat = np.array([10.0, 12.0])
    event = pd.DatetimeIndex(["2000-07-10"]).values
    # systems at lag 0 and lag +3 days; window 0.5 keeps only the lag-0 one
    cs_time = pd.DatetimeIndex(["2000-07-10", "2000-07-13"]).values
    cs_lon = np.array([0.0, 0.0])
    cs_lat = np.array([10.0, 10.0])
    counts = map_event_counts(event, cs_time, cs_lon, cs_lat, lon, lat, lag=0.0,
                              half_window=0.5)
    assert counts.sum() == 1


def test_map_event_field_mean_radius():
    lon = np.array([0.0, 4.0])
    lat = np.array([10.0, 12.0])
    event = pd.DatetimeIndex(["2000-07-10"]).values
    cs_time = pd.DatetimeIndex(["2000-07-10", "2000-07-10"]).values
    cs_lon = np.array([0.0, 0.0])
    cs_lat = np.array([10.0, 10.0])
    radius = np.array([200.0, 400.0])
    m = map_event_counts(event, cs_time, cs_lon, cs_lat, lon, lat, z=radius,
                         statistic="mean")
    assert m[0, 0] == 300.0
    assert np.isnan(m[1, 1])


def test_basepoint_map_renders():
    cartopy = pytest.importorskip("cartopy")
    from aew.plotting import basepoint_map

    lon = np.arange(-40.0, 80.0, 4.0)
    lat = np.arange(0.0, 25.0, 2.5)
    shaded = np.outer(np.cos(lat / 8.0), np.sin(lon / 20.0))
    contour = np.outer(np.sin(lat / 8.0), np.cos(lon / 20.0))
    fig, ax = basepoint_map(shaded, lon, lat, contour=contour, base_lon=0.0,
                            base_lat=10.0, extent=(-40, 80, 0, 25), title="t")
    assert fig is not None and ax is not None
