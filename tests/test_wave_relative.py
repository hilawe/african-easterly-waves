import numpy as np
import pandas as pd
from aew.composites import wave_relative_counts


def test_wave_relative_places_systems_relative_to_moving_trough():
    # trough moves westward over 3 times; a cloud system sits 5 deg EAST of the trough each time
    times = pd.date_range("2000-07-10", periods=3, freq="6h")
    trough_lon = np.array([20.0, 15.0, 10.0])          # trough marches west
    cs_time = times.values
    cs_lon = trough_lon + 6.0                            # always 6 deg east of trough
    cs_lat = np.array([10.0, 10.0, 10.0])
    rel_c = np.arange(-30.0, 30.1, 2.0)
    lat_c = np.arange(0.0, 25.1, 2.0)
    counts, n = wave_relative_counts(times.values, trough_lon, cs_time, cs_lon, cs_lat,
                                     rel_c, lat_c, time_tol_hours=1.0)
    assert n == 3
    # all 3 systems land in the rel_lon=+4 bin (nearest to +5) at lat 10
    i_rel = np.argmin(np.abs(rel_c - 6.0))
    i_lat = np.argmin(np.abs(lat_c - 10.0))
    assert counts[i_lat, i_rel] == 3
    assert counts.sum() == 3


def test_wave_relative_time_tolerance_excludes_far_systems():
    times = pd.DatetimeIndex(["2000-07-10T00:00"])
    trough_lon = np.array([0.0])
    # one system same time (kept), one 12 h away (excluded at tol=1h)
    cs_time = pd.DatetimeIndex(["2000-07-10T00:00", "2000-07-10T12:00"]).values
    cs_lon = np.array([3.0, 3.0]); cs_lat = np.array([10.0, 10.0])
    rel_c = np.arange(-30.0, 30.1, 2.0); lat_c = np.arange(0.0, 25.1, 2.0)
    counts, n = wave_relative_counts(times.values, trough_lon, cs_time, cs_lon, cs_lat,
                                     rel_c, lat_c, time_tol_hours=1.0)
    assert counts.sum() == 1


def test_wave_relative_no_double_count_at_window_boundary():
    # two 6-hourly troughs; one MCS exactly midway (+3h from first = -3h from second).
    # With a 3h half-open window it must be counted once, not twice.
    troughs = pd.DatetimeIndex(["2000-07-10T00:00", "2000-07-10T06:00"])
    trough_lon = np.array([0.0, 0.0])
    cs_time = pd.DatetimeIndex(["2000-07-10T03:00"]).values
    cs_lon = np.array([2.0]); cs_lat = np.array([10.0])
    rel_c = np.arange(-30.0, 30.1, 2.0); lat_c = np.arange(0.0, 25.1, 2.0)
    counts, n = wave_relative_counts(troughs.values, trough_lon, cs_time, cs_lon, cs_lat,
                                     rel_c, lat_c, time_tol_hours=3.0)
    assert counts.sum() == 1  # counted for exactly one of the two troughs
