import numpy as np
import pandas as pd

from aew.leadlag import (
    band_mean_series,
    box_count_series,
    lag_cross_correlation,
    peak_lag,
)


def test_box_count_series_linear_binning_and_filters_box():
    grid = pd.date_range("2000-07-01", periods=4, freq="6h").values
    # in-box systems at 04Z, 07Z, 00Z; plus one out of box (lon) and one out of band (lat).
    cs_time = pd.to_datetime(
        ["2000-07-01 04:00", "2000-07-01 07:00", "2000-07-01 06:00",
         "2000-07-01 06:00", "2000-07-01 00:00"]
    ).values
    cs_lon = np.array([0.0, 1.0, 0.0, 40.0, 0.0])   # 4th is far east -> out of box
    cs_lat = np.array([10.0, 10.0, 30.0, 10.0, 10.0])  # 3rd is out of band
    counts = box_count_series(cs_time, cs_lon, cs_lat, grid, ref_lon=0.0, dlon=5.0,
                              lat_lo=5.0, lat_hi=15.0)
    # 04Z -> 4/6 to 06Z, 2/6 to 00Z; 07Z -> 5/6 to 06Z, 1/6 to 12Z; 00Z -> 1 to 00Z.
    expected = np.array([2 / 6 + 1.0, 4 / 6 + 5 / 6, 1 / 6, 0.0])
    np.testing.assert_allclose(counts, expected)
    # area preservation: total assigned weight equals the 3 in-box observations
    assert abs(counts.sum() - 3.0) < 1e-12


def test_box_count_series_on_node_times_are_exact():
    grid = pd.date_range("2000-07-01", periods=3, freq="6h").values
    cs_time = pd.to_datetime(["2000-07-01 06:00", "2000-07-01 12:00"]).values
    cs_lon = np.array([0.0, 0.0]); cs_lat = np.array([10.0, 10.0])
    counts = box_count_series(cs_time, cs_lon, cs_lat, grid, 0.0, 5.0, 5.0, 15.0)
    np.testing.assert_allclose(counts, [0.0, 1.0, 1.0])  # no leakage for on-node times


def test_box_count_series_wraps_longitude():
    grid = pd.date_range("2000-07-01", periods=2, freq="6h").values
    cs_time = pd.to_datetime(["2000-07-01 00:00", "2000-07-01 00:00"]).values
    cs_lon = np.array([179.0, -179.0])  # both within 5 deg of ref_lon=180 across the seam
    cs_lat = np.array([10.0, 10.0])
    counts = box_count_series(cs_time, cs_lon, cs_lat, grid, ref_lon=180.0, dlon=5.0,
                              lat_lo=5.0, lat_hi=15.0)
    assert counts[0] == 2.0


def test_band_mean_series_is_nan_aware():
    field = np.array([[[1.0, 2.0], [np.nan, 4.0], [3.0, 6.0]]])  # (1 time, 3 lat, 2 lon)
    lat = np.array([5.0, 10.0, 15.0])
    out = band_mean_series(field, lat, 5.0, 15.0)
    # lon 0: mean of [1, nan, 3] = 2 ; lon 1: mean of [2, 4, 6] = 4
    np.testing.assert_allclose(out, [[2.0, 4.0]])


def test_lag_cross_correlation_recovers_positive_lag_when_convection_lags():
    # c[t] = w[t-3]: convection is a delayed copy of the wave -> wave leads by 3.
    rng = np.random.default_rng(0)
    w = rng.standard_normal(500)
    c = np.full_like(w, np.nan)
    c[3:] = w[:-3]
    lags, R = lag_cross_correlation(w, c, max_lag=6)
    lag_star, r_star = peak_lag(lags, R)
    assert abs(lag_star - 3.0) < 0.25   # peak at +3 (convection lags the wave)
    assert r_star > 0.99


def test_lag_cross_correlation_recovers_negative_lag_when_convection_leads():
    rng = np.random.default_rng(1)
    w = rng.standard_normal(500)
    c = np.full_like(w, np.nan)
    c[:-2] = w[2:]   # c[t] = w[t+2]: convection leads the wave by 2
    lags, R = lag_cross_correlation(w, c, max_lag=6)
    lag_star, _ = peak_lag(lags, R)
    assert abs(lag_star + 2.0) < 0.25


def test_lag_cross_correlation_handles_all_nan_pairs():
    w = np.arange(10.0)
    c = np.full(10, np.nan)
    lags, R = lag_cross_correlation(w, c, max_lag=3)
    assert np.isnan(R).all()
    assert np.isnan(peak_lag(lags, R)[0])
