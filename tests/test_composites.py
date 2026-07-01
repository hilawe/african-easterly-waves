import numpy as np
import pandas as pd

from aew.composites import (
    anomaly,
    composite_xt_preread,
    composite_xy_preread,
    dates_from_other_years,
    hovmoller_event_counts,
    hovmoller_event_field,
    lag_axis,
)


def test_lag_axis():
    lag = lag_axis(-6, 6, 1)
    assert lag.size == 13
    np.testing.assert_array_equal(lag, np.arange(-6, 7))


def test_dates_from_other_years_same_mdh_other_year():
    # daily times across 3 years
    time = pd.date_range("2000-01-01", "2002-12-31", freq="D")
    base = pd.DatetimeIndex(["2001-07-15", "2001-08-20"])
    pop = pd.DatetimeIndex(dates_from_other_years(base.values, time.values))
    # all returned must share (month,day) with a base date but not be a base date
    assert (pd.DatetimeIndex(pop).isin(base)).sum() == 0
    mmdd = {(t.month, t.day) for t in pop}
    assert mmdd <= {(7, 15), (8, 20)}
    # 3 years, base in 2001 -> the other 2 years contribute for each of 2 base dates
    assert len(pop) == 4


def test_composite_recovers_known_lag_means():
    time = pd.date_range("2000-01-01", periods=100, freq="D")
    nlon = 5
    # field constant across lon, equal to the day index
    data = np.tile(np.arange(100.0)[:, None], (1, nlon))
    base = pd.DatetimeIndex(["2000-01-11", "2000-01-21", "2000-01-31"])  # idx 10,20,30
    res = composite_xt_preread(data, time.values, base.values, -2, 2, 1)
    # lag 0 -> mean(10,20,30)=20; lag +1 -> 21; lag -1 -> 19
    lag0 = np.where(res.lag == 0)[0][0]
    np.testing.assert_allclose(res.values[lag0], 20.0)
    np.testing.assert_allclose(res.values[res.lag == 1][0], 21.0)
    np.testing.assert_allclose(res.values[res.lag == -1][0], 19.0)
    assert res.n_dates[lag0] == 3


def test_composite_drops_out_of_record_lags():
    time = pd.date_range("2000-01-01", periods=20, freq="D")
    data = np.zeros((20, 3))
    base = pd.DatetimeIndex(["2000-01-20"])  # last day, idx 19
    res = composite_xt_preread(data, time.values, base.values, 0, 3, 1)
    # lag 0 present (1 date); lags +1..+3 fall past the record -> 0 dates, NaN
    assert res.n_dates[res.lag == 0][0] == 1
    assert res.n_dates[res.lag == 3][0] == 0
    assert np.all(np.isnan(res.values[res.lag == 3][0]))


def test_monte_carlo_significance_flags_strong_signal():
    # 3 years of daily data; a large positive value ONLY on the base date's calendar
    # day in year 2 -> population (other years, same m/d) is ~0 -> highly significant.
    time = pd.date_range("2000-01-01", "2002-12-31", freq="D")
    nlon = 4
    data = np.zeros((time.size, nlon))
    base = pd.DatetimeIndex(["2001-07-15"])
    data[time == base[0]] = 50.0
    rng = np.random.default_rng(0)
    res = composite_xt_preread(
        data, time.values, base.values, 0, 0, 1, n_tests=500, p_thresh=0.95, rng=rng
    )
    assert res.p_value is not None
    # lag 0 composite is 50 everywhere, population mean ~0 -> survives masking
    assert np.all(np.isfinite(res.values[0]))
    np.testing.assert_allclose(res.values[0], 50.0)


def test_composite_xy_recovers_known_map_at_lag0():
    time = pd.date_range("2000-01-01", periods=100, freq="D")
    nlat, nlon = 4, 5
    # field constant in space, equal to day index
    data = np.tile(np.arange(100.0)[:, None, None], (1, nlat, nlon))
    base = pd.DatetimeIndex(["2000-01-11", "2000-01-21", "2000-01-31"])  # idx 10,20,30
    res = composite_xy_preread(data, time.values, base.values, lags=[0])
    assert res.values.shape == (1, nlat, nlon)
    np.testing.assert_allclose(res.values[0], 20.0)  # mean(10,20,30)
    assert res.n_dates[0] == 3


def test_composite_xy_multiple_lags_shape():
    time = pd.date_range("2000-01-01", periods=60, freq="D")
    data = np.zeros((60, 3, 6))
    base = pd.DatetimeIndex(["2000-01-20", "2000-01-30"])
    res = composite_xy_preread(data, time.values, base.values, lags=[-2, 0, 2])
    assert res.values.shape == (3, 3, 6)
    assert res.lag.tolist() == [-2.0, 0.0, 2.0]


def test_composite_xy_significance_shape():
    time = pd.date_range("2000-01-01", "2002-12-31", freq="D")
    data = np.zeros((time.size, 2, 2))
    base = pd.DatetimeIndex(["2001-07-15"])
    data[time == base[0]] = 50.0
    res = composite_xy_preread(
        data, time.values, base.values, lags=[0], n_tests=300, p_thresh=0.95,
        rng=np.random.default_rng(0),
    )
    assert res.p_value.shape == (1, 2, 2)
    assert np.all(np.isfinite(res.values[0]))  # strong signal survives


def test_hovmoller_counts_land_in_expected_bins():
    lon_centers = np.array([0.0, 4.0, 8.0])
    lag_centers = np.arange(-6.0, 7.0)
    event = pd.DatetimeIndex(["2000-07-10"])
    # one CS at lon 4, two days after the event -> bin (lag=2, lon=4)
    cs_time = pd.DatetimeIndex(["2000-07-12"])
    cs_lon = np.array([4.0])
    counts = hovmoller_event_counts(
        event.values, cs_time.values, cs_lon, lon_centers, lag_centers
    )
    lag2 = np.where(lag_centers == 2.0)[0][0]
    lon4 = 1
    assert counts[lag2, lon4] == 1
    assert counts.sum() == 1


def test_hovmoller_counts_lat_filter_and_multiple_events():
    lon_centers = np.array([0.0, 4.0])
    lag_centers = np.arange(-6.0, 7.0)
    events = pd.DatetimeIndex(["2000-07-10", "2000-07-20"])
    # two systems: one in-band, one out-of-band (should be dropped by lat filter)
    cs_time = pd.DatetimeIndex(["2000-07-10", "2000-07-10"])
    cs_lon = np.array([0.0, 0.0])
    cs_lat = np.array([10.0, 40.0])
    counts = hovmoller_event_counts(
        events.values, cs_time.values, cs_lon, lon_centers, lag_centers,
        cs_lat=cs_lat, min_lat=5.0, max_lat=15.0,
    )
    # only the in-band system counts; it contributes once per event (lag 0 and lag -10)
    # lag -10 is outside the window so dropped -> exactly 1 count at lag 0, lon 0
    assert counts.sum() == 1
    assert counts[np.where(lag_centers == 0)[0][0], 0] == 1


def test_dates_from_other_years_same_year_subhour_excluded():
    # 3-hourly data: a population time at 03:00 must NOT be pulled in for a base at 00:00
    # (different hour), and a same-year same-m/d/h non-base sub-hour case is excluded.
    time = pd.date_range("2000-07-01", "2002-07-31", freq="3h")
    base = pd.DatetimeIndex(["2001-07-15 00:00"])
    pop = pd.DatetimeIndex(dates_from_other_years(base.values, time.values))
    # only 00:00 on 7/15 in 2000 and 2002 qualify (same m/d/h, different year)
    assert set(pop) == {pd.Timestamp("2000-07-15 00:00"), pd.Timestamp("2002-07-15 00:00")}


def test_hovmoller_field_mean_radius():
    lon_centers = np.array([0.0, 4.0])
    lag_centers = np.arange(-6.0, 7.0)
    event = pd.DatetimeIndex(["2000-07-10"])
    # two systems in the same (lag0, lon0) bin with radii 200 and 400 -> mean 300
    cs_time = pd.DatetimeIndex(["2000-07-10", "2000-07-10"])
    cs_lon = np.array([0.0, 0.0])
    radius = np.array([200.0, 400.0])
    field = hovmoller_event_field(
        event.values, cs_time.values, cs_lon, radius, lon_centers, lag_centers,
        statistic="mean",
    )
    i0 = np.where(lag_centers == 0)[0][0]
    assert field[i0, 0] == 300.0
    # empty bins are NaN
    assert np.isnan(field[i0, 1])
    summed = hovmoller_event_field(
        event.values, cs_time.values, cs_lon, radius, lon_centers, lag_centers,
        statistic="sum",
    )
    assert summed[i0, 0] == 600.0


def test_anomaly_window_restriction():
    counts = np.array([[10.0], [2.0], [4.0], [99.0]])  # 4 lags x 1 lon
    lag_centers = np.array([-1.0, 0.0, 1.0, 5.0])
    # restrict baseline to lags [-1,1] -> mean of [10,2,4]=5.333
    anom = anomaly(counts, "anomaly", lag_centers=lag_centers, min_lag=-1, max_lag=1)
    np.testing.assert_allclose(anom[0, 0], 10.0 - (10 + 2 + 4) / 3)


def test_anomaly_modes():
    counts = np.array([[2.0, 0.0], [0.0, 0.0], [4.0, 0.0]])  # (lag=3, lon=2)
    tot = anomaly(counts, "total")
    np.testing.assert_array_equal(tot, counts)
    anom = anomaly(counts, "anomaly")
    # lon0 lag-mean = 2; anomalies -> [0,-2,2]
    np.testing.assert_allclose(anom[:, 0], [0.0, -2.0, 2.0])
    pct = anomaly(counts, "pct")
    np.testing.assert_allclose(pct[:, 0], [0.0, -100.0, 100.0])
