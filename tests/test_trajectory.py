import numpy as np
import pandas as pd

from aew.trajectory import M_PER_DEG, Gridded, back_trajectories, classify_origin


def _grid(nt=17, dt="6h", lat=None, lon=None):
    times = pd.date_range("2000-08-01", periods=nt, freq=dt).values
    lat = np.arange(-10.0, 30.1, 1.0) if lat is None else lat
    lon = np.arange(-40.0, 40.1, 1.0) if lon is None else lon
    return times, lat, lon


def _const_wind(times, lat, lon, u0, v0):
    shape = (len(times), lat.size, lon.size)
    return (Gridded(times, lat, lon, np.full(shape, float(u0))),
            Gridded(times, lat, lon, np.full(shape, float(v0))))


def test_gridded_sample_exact_on_nodes_and_linear_between():
    times, lat, lon = _grid()
    vals = np.zeros((len(times), lat.size, lon.size))
    vals[0] = 1.0
    vals[1] = 3.0
    g = Gridded(times, lat, lon, vals)
    t0 = times[0]
    mid = t0 + np.timedelta64(3, "h")  # halfway between the first two 6-hourly steps
    assert g.sample(_hours(t0), 10.0, 0.0) == 1.0
    np.testing.assert_allclose(g.sample(_hours(mid), 10.0, 0.0), 2.0)


def _hours(t):
    return np.asarray(pd.DatetimeIndex([t]).values.astype("datetime64[ns]").astype("int64"),
                      dtype=float) / 3.6e12


def test_gridded_sample_nan_outside_domain():
    times, lat, lon = _grid()
    g = Gridded(times, lat, lon, np.ones((len(times), lat.size, lon.size)))
    t = _hours(times[2])
    assert np.isnan(g.sample(t, 50.0, 0.0))     # off the lat grid
    assert np.isnan(g.sample(t, 10.0, 100.0))   # off the regional lon grid
    early = _hours(times[0]) - 7.0
    assert np.isnan(g.sample(early, 10.0, 0.0))  # before the record


def test_uniform_westerly_backward_goes_west():
    # u = +10 m/s (westerly): a parcel arrived FROM the west, so backward motion is west.
    times, lat, lon = _grid()
    u, v = _const_wind(times, lat, lon, 10.0, 0.0)
    seed_t = np.array([times[8]])
    elapsed, plat, plon = back_trajectories(u, v, seed_t, np.array([0.0]), np.array([0.0]),
                                            hours=24.0, dt_hours=1.0)
    expect_dlon = -10.0 * 24 * 3600 / M_PER_DEG  # at the equator, cos(lat)=1
    np.testing.assert_allclose(plon[-1, 0], expect_dlon, rtol=1e-3)
    np.testing.assert_allclose(plat[-1, 0], 0.0, atol=1e-6)


def test_uniform_southerly_backward_goes_south():
    # v = +5 m/s (southerly): the parcel came from the south.
    times, lat, lon = _grid()
    u, v = _const_wind(times, lat, lon, 0.0, 5.0)
    seed_t = np.array([times[8]])
    elapsed, plat, plon = back_trajectories(u, v, seed_t, np.array([10.0]), np.array([0.0]),
                                            hours=24.0, dt_hours=1.0)
    expect_dlat = -5.0 * 24 * 3600 / M_PER_DEG
    np.testing.assert_allclose(plat[-1, 0] - 10.0, expect_dlat, rtol=1e-3)


def test_per_parcel_seed_times_use_their_own_clock():
    # wind switches from westerly to southerly halfway through the record; parcels seeded
    # in each regime must feel only their own regime over a short integration.
    times, lat, lon = _grid(nt=17)
    nt = len(times)
    uvals = np.zeros((nt, lat.size, lon.size)); vvals = np.zeros_like(uvals)
    uvals[: nt // 2] = 10.0          # first half: pure westerly
    vvals[nt // 2:] = 10.0           # second half: pure southerly
    u = Gridded(times, lat, lon, uvals); v = Gridded(times, lat, lon, vvals)
    seeds_t = np.array([times[3], times[13]])
    _, plat, plon = back_trajectories(u, v, seeds_t, np.array([0.0, 0.0]),
                                      np.array([0.0, 0.0]), hours=6.0, dt_hours=1.0)
    assert plon[-1, 0] < -0.4 and abs(plat[-1, 0]) < 0.05   # early parcel moved west only
    assert plat[-1, 1] < -0.4 and abs(plon[-1, 1]) < 0.05   # late parcel moved south only


def test_parcel_leaving_regional_domain_becomes_nan_and_stays_nan():
    times, lat, lon = _grid(lon=np.arange(-2.0, 2.1, 1.0))  # tiny lon domain
    u, v = _const_wind(times, lat, lon, 20.0, 0.0)          # fast westerly
    seed_t = np.array([times[8]])
    _, plat, plon = back_trajectories(u, v, seed_t, np.array([0.0]), np.array([0.0]),
                                      hours=24.0, dt_hours=1.0)
    assert np.isnan(plon[-1, 0])
    # once NaN, all later rows stay NaN
    first_bad = np.argmax(~np.isfinite(plon[:, 0]))
    assert not np.isfinite(plon[first_bad:, 0]).any()


def test_solid_body_rotation_returns_after_one_period():
    # rotation about (0N, 0E) in the local tangent plane; at small radius the spherical
    # metric distortion is small, so one full period should nearly close the loop.
    times, lat, lon = _grid(nt=41, dt="6h")
    omega = 2 * np.pi / (48 * 3600.0)  # 48 h period, rad/s
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    x = LON * M_PER_DEG * np.cos(np.deg2rad(LAT))
    y = LAT * M_PER_DEG
    uvals = np.broadcast_to(-omega * y, (len(times),) + y.shape).copy()
    vvals = np.broadcast_to(omega * x, (len(times),) + x.shape).copy()
    u = Gridded(times, lat, lon, uvals); v = Gridded(times, lat, lon, vvals)
    seed_t = np.array([times[30]])
    _, plat, plon = back_trajectories(u, v, seed_t, np.array([2.0]), np.array([0.0]),
                                      hours=48.0, dt_hours=0.5)
    np.testing.assert_allclose(plat[-1, 0], 2.0, atol=0.1)
    np.testing.assert_allclose(plon[-1, 0], 0.0, atol=0.1)


def test_gridded_sample_exact_final_nodes_are_inside():
    times, lat, lon = _grid()
    vals = np.ones((len(times), lat.size, lon.size))
    g = Gridded(times, lat, lon, vals)
    t_last = _hours(times[-1])
    assert g.sample(t_last, lat[-1], lon[-1]) == 1.0   # exact record/domain edges


def test_gridded_cyclic_interpolation_across_the_dateline():
    times = pd.date_range("2000-08-01", periods=3, freq="6h").values
    lat = np.arange(-10.0, 11.0, 5.0)
    lon = np.arange(-180.0, 180.0, 1.5)  # global cyclic grid (240 points)
    vals = np.zeros((3, lat.size, lon.size))
    vals[:, :, 0] = 4.0     # value at -180
    vals[:, :, -1] = 2.0    # value at +178.5
    g = Gridded(times, lat, lon, vals)
    assert g.cyclic
    t = _hours(times[1])
    # halfway between +178.5 and -180 (i.e. 179.25) -> mean of 2 and 4
    np.testing.assert_allclose(g.sample(t, 0.0, 179.25), 3.0)
    # sampling at longitudes outside [-180, 180) wraps
    np.testing.assert_allclose(g.sample(t, 0.0, 539.25), 3.0)  # 539.25 - 360 = 179.25


def test_classify_origin_sectors():
    seed_lat = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    seed_lon = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    org_lat = np.array([5.0, 15.0, 10.5, 10.5, 10.0, np.nan])
    org_lon = np.array([0.0, 2.0, 8.0, -8.0, 1.0, 0.0])
    out = classify_origin(seed_lat, seed_lon, org_lat, org_lon)
    assert list(out) == ["south", "north", "east", "west", "local", "lost"]
