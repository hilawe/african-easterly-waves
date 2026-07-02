import numpy as np
import pandas as pd

from aew.environment import (
    cluster_bootstrap_diff,
    forward_response,
    lead_field_box,
    lead_value,
    stratified_terciles,
    terciles,
)


def test_stratified_terciles_split_within_bins_and_skip_small_bins():
    # bin A (strat ~0) has 30 members with values 0..29; bin B (strat ~10) only 5 members
    x = np.concatenate([np.arange(30.0), np.arange(5.0)])
    strat = np.concatenate([np.zeros(30), np.full(5, 10.0)])
    low, high = stratified_terciles(x, strat, np.array([-5.0, 5.0, 15.0]), min_bin=30)
    assert low[:30].sum() > 0 and high[:30].sum() > 0     # bin A participates
    assert not low[30:].any() and not high[30:].any()     # bin B skipped (< min_bin)
    # within bin A the split is by bin-A terciles, not global ones
    assert low[0] and not low[29] and high[29] and not high[0]


def test_forward_response_counts_forward_window_box_and_wrap():
    trough_times = pd.to_datetime(["2000-08-01 00:00"]).values
    trough_lons = np.array([0.0])
    # cloud systems: two ahead in time and in box; one before the trough (excluded);
    # one ahead but out of the lon box; one ahead but out of the lat band.
    cs_time = pd.to_datetime(
        ["2000-08-01 03:00", "2000-08-01 12:00", "2000-07-31 21:00",
         "2000-08-01 06:00", "2000-08-01 06:00"]
    ).values
    cs_lon = np.array([2.0, -3.0, 0.0, 40.0, 0.0])
    cs_lat = np.array([10.0, 12.0, 10.0, 10.0, 30.0])
    r = forward_response(trough_times, trough_lons, cs_time, cs_lon, cs_lat,
                         win_h=24.0, dlon=8.0, lat_lo=5.0, lat_hi=15.0)
    assert r[0] == 2.0  # only the two in-box, in-band, forward-in-time systems


def test_forward_response_window_is_half_open_forward():
    trough_times = pd.to_datetime(["2000-08-01 00:00"]).values
    cs_time = pd.to_datetime(["2000-08-01 00:00", "2000-08-02 00:00"]).values  # at t and at t+24h
    cs_lon = np.array([0.0, 0.0]); cs_lat = np.array([10.0, 10.0])
    r = forward_response(trough_times, np.array([0.0]), cs_time, cs_lon, cs_lat, win_h=24.0)
    assert r[0] == 1.0  # t included, t+24h excluded (half open)


def test_lead_value_picks_nearest_earlier_real_sample_within_tol():
    # one trajectory, 6-hourly, with a NaN at the exact t-24h slot
    times = pd.to_datetime([
        "2000-08-01 00:00", "2000-08-01 06:00", "2000-08-01 12:00",
        "2000-08-01 18:00", "2000-08-02 00:00",
    ]).values
    tid = np.array([7, 7, 7, 7, 7])
    vals = np.array([40.0, np.nan, 45.0, 50.0, 55.0])  # 06:00 is missing
    lead = lead_value(times, tid, vals, lead_h=24.0, tol_h=6.0)
    # for the last obs (02:00), target is 02:00-24h = previous 02:00; nearest earlier finite
    # sample is 00:00 (value 40, 2 h from target) -> within tol.
    assert lead[-1] == 40.0
    # for the first obs there is no earlier sample -> NaN
    assert np.isnan(lead[0])


def test_lead_value_rejects_when_outside_tolerance():
    times = pd.to_datetime(["2000-08-01 00:00", "2000-08-01 06:00"]).values
    tid = np.array([1, 1])
    vals = np.array([30.0, 31.0])
    # second obs target is 06:00-24h = prev day 06:00; nearest earlier real sample is
    # 00:00 today, ~18 h from target, outside tol_h=6 -> NaN
    lead = lead_value(times, tid, vals, lead_h=24.0, tol_h=6.0)
    assert np.isnan(lead[1])


def test_lead_value_stays_within_trajectory():
    # two trajectories interleaved in time; a lead must not borrow the other wave's sample
    times = pd.to_datetime([
        "2000-08-01 00:00", "2000-08-01 01:00", "2000-08-02 00:00", "2000-08-02 01:00",
    ]).values
    tid = np.array([1, 2, 1, 2])
    vals = np.array([10.0, 99.0, 11.0, 98.0])
    lead = lead_value(times, tid, vals, lead_h=24.0, tol_h=3.0)
    assert lead[2] == 10.0  # traj 1 at day2 leads from traj 1 at day1 (10), not traj 2 (99)
    assert lead[3] == 99.0  # traj 2 at day2 leads from traj 2 at day1 (99), not traj 1


def test_cluster_bootstrap_widens_ci_vs_treating_obs_as_independent():
    # two groups, each one wave (cluster) repeated many times: with only 1 cluster per group
    # the cluster bootstrap cannot resample structure, so the CI collapses to the point diff
    # (every resample is identical). This checks clusters, not observations, drive the CI.
    rng = np.random.default_rng(0)
    gid_a = np.zeros(50, dtype=int); val_a = np.full(50, 40.0)
    gid_b = np.ones(50, dtype=int); val_b = np.full(50, 42.0)
    diff, lo, hi, na, nb = cluster_bootstrap_diff(gid_a, val_a, gid_b, val_b, rng, n_boot=200)
    assert na == 1 and nb == 1
    assert diff == 2.0 and lo == 2.0 and hi == 2.0  # one cluster each -> no resampling spread


def test_cluster_bootstrap_reports_cluster_counts_and_brackets_diff():
    rng = np.random.default_rng(1)
    gid_a = np.repeat(np.arange(20), 5); val_a = rng.normal(40, 1, 100)
    gid_b = np.repeat(np.arange(100, 130), 4); val_b = rng.normal(43, 1, 120)
    diff, lo, hi, na, nb = cluster_bootstrap_diff(gid_a, val_a, gid_b, val_b, rng, n_boot=500)
    assert na == 20 and nb == 30
    assert lo < diff < hi
    assert diff > 0  # group b is ~3 higher


def _synthetic_field():
    # 6-hourly field over 3 days; value = day index so the lead time is checkable
    f_time = pd.date_range("2000-08-01", periods=12, freq="6h").values
    f_lat = np.arange(0.0, 21.0, 5.0)          # 0..20N
    f_lon = np.arange(-20.0, 21.0, 5.0)        # -20..20E
    day = np.arange(12) // 4                   # 0,0,0,0,1,1,1,1,2,...
    field = np.broadcast_to(
        day[:, None, None], (12, f_lat.size, f_lon.size)
    ).astype(float).copy()
    return f_time, f_lat, f_lon, field


def test_lead_field_box_samples_24h_earlier():
    f_time, f_lat, f_lon, field = _synthetic_field()
    # trough at day-2 00Z: 24 h earlier is day-1 00Z, where the field value is 1.0
    t = pd.to_datetime(["2000-08-03 00:00"]).values
    out = lead_field_box(t, np.array([0.0]), f_time, f_lat, f_lon, field,
                         lead_h=24.0, tol_h=3.0)
    assert out[0] == 1.0


def test_lead_field_box_nan_outside_tolerance_and_before_record():
    f_time, f_lat, f_lon, field = _synthetic_field()
    # target time is before the field record starts -> NaN
    t = pd.to_datetime(["2000-08-01 06:00"]).values   # t-24h = Jul 31 06Z, not in record
    out = lead_field_box(t, np.array([0.0]), f_time, f_lat, f_lon, field,
                         lead_h=24.0, tol_h=3.0)
    assert np.isnan(out[0])


def test_lead_field_box_restricts_to_lon_box_and_lat_band():
    f_time, f_lat, f_lon, field = _synthetic_field()
    # make one longitude column inside the box hugely different to prove box selection
    field = field.copy()
    jlat = np.where((f_lat >= 5) & (f_lat <= 15))[0]
    jlon = np.where(f_lon == 0.0)[0]
    field[4, jlat[:, None], jlon] = 100.0             # Aug-2 00Z (index 4), lon 0, in-band
    t = pd.to_datetime(["2000-08-03 00:00"]).values   # 24 h earlier is Aug-2 00Z
    # box centered at 0E with half-width 2.5 -> only lon 0 column, only 5..15N rows
    out = lead_field_box(t, np.array([0.0]), f_time, f_lat, f_lon, field,
                         lead_h=24.0, tol_h=3.0, dlon=2.5)
    assert out[0] == 100.0
    # a box centered far from the tampered column keeps the background value
    out2 = lead_field_box(t, np.array([-15.0]), f_time, f_lat, f_lon, field,
                          lead_h=24.0, tol_h=3.0, dlon=2.5)
    assert out2[0] == 1.0


def test_lead_field_box_is_nan_aware():
    f_time, f_lat, f_lon, field = _synthetic_field()
    field = field.copy()
    field[4] = np.nan                                   # whole Aug-2 00Z step NaN
    t = pd.to_datetime(["2000-08-03 00:00"]).values     # 24 h earlier is Aug-2 00Z
    out = lead_field_box(t, np.array([0.0]), f_time, f_lat, f_lon, field,
                         lead_h=24.0, tol_h=3.0)
    assert np.isnan(out[0])


def test_cluster_bootstrap_handles_waves_shared_across_arms():
    rng = np.random.default_rng(2)
    # wave 5 straddles both arms (some obs low-response, some high-response); the union draw
    # must count it once as a cluster in each arm and route its members accordingly.
    gid_a = np.array([1, 1, 2, 2, 5, 5]); val_a = np.array([40.0, 41, 40, 42, 41, 43])
    gid_b = np.array([3, 3, 4, 4, 5, 5]); val_b = np.array([44.0, 45, 46, 44, 45, 47])
    diff, lo, hi, na, nb = cluster_bootstrap_diff(gid_a, val_a, gid_b, val_b, rng, n_boot=500)
    assert na == 3 and nb == 3      # arm a clusters {1,2,5}; arm b clusters {3,4,5}
    assert lo < diff < hi and diff > 0


def test_terciles_split():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan])
    low, high = terciles(x)
    assert not low[-1] and not high[-1]      # NaN in neither
    assert low[0] and low[1]                 # bottom third
    assert high[-2] and high[-3]             # top third
    assert not (low & high).any()
