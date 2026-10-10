"""The propagation study's contracts: the band-pass passes the declared band and rejects
outside it, the filter never uses or reports samples outside a contiguous supported run,
the terrain rule keeps a band sample only above the declared valid fraction, and the
diagnostic recovers a synthetic traveling signal's westward speed and reports a synthetic
standing signal as no resolved propagation."""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("prop_under_test", os.path.join(ROOT, "scripts", "pilot_eastern_propagation.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _response(w, f):
    k = np.arange(-(len(w) - 1) // 2, (len(w) - 1) // 2 + 1)
    return abs(np.sum(w * np.exp(-2j * np.pi * f * k)))


def test_the_band_pass_passes_the_declared_band_and_rejects_outside_it():
    P = _load()
    w = P.lanczos_bandpass()
    assert len(w) == 81 and abs(_response(w, 0.25 / 4.0) - 1.0) < 0.05          # a 4-day period
    assert _response(w, 0.25 / 1.0) < 0.05 and _response(w, 0.25 / 20.0) < 0.05  # 1-day and 20-day periods


def test_the_filter_keeps_only_whole_windows_inside_supported_runs():
    P = _load()
    w = P.lanczos_bandpass()
    x = np.sin(np.arange(488) * 2 * np.pi / 16.0)
    x[200] = np.nan
    y = P.filter_supported(x, w)
    assert np.isnan(y[:40]).all() and np.isnan(y[-40:]).all() and np.isnan(y[160:241]).all()
    assert np.isfinite(y[40:160]).all() and np.isfinite(y[241:448]).all()
    short = np.full(488, np.nan); short[100:180] = 1.0                            # a run shorter than the window
    assert np.isnan(P.filter_supported(short, w)).all()


def test_a_band_sample_needs_the_declared_valid_fraction_and_averages_valid_cells_only():
    P = _load()
    v = np.ones((2, 20, 1)); v[:, :5, 0] = 100.0
    sp = np.full((2, 20, 1), 90000.0)
    sp[0, :5, 0] = 70000.0                    # 5 of 20 below 700 hPa ground: 75 percent valid, refused
    sp[1, :4, 0] = 70000.0                    # 4 of 20 below ground: 80 percent valid, kept
    series, frac = P.band_series(v, sp, 700)
    assert np.isnan(series[0, 0]) and frac[0, 0] == 0.75
    assert abs(series[1, 0] - (100.0 + 15.0) / 16.0) < 1e-12 and frac[1, 0] == 0.8
    sp_equal = np.full((1, 20, 1), 70000.0)   # equal to the level is below ground
    assert np.isnan(P.band_series(np.ones((1, 20, 1)), sp_equal, 700)[0][0, 0])


def _synthetic(P, kind, seed=0):
    lons = np.arange(P.LON_RANGE[0], P.LON_RANGE[1] + 0.25, 0.5)
    t = np.arange(488) * P.DT_DAYS
    rng = np.random.default_rng(seed)
    w = P.lanczos_bandpass()
    base = 50.0
    j = int(np.argmin(np.abs(lons - base)))
    sums = []
    for s in range(24):
        phase = rng.uniform(0, 2 * np.pi)
        if kind == "traveling":                                  # westward at 7 degrees per day, period 4 days
            field = np.sin(2 * np.pi * (lons[None, :] + 7.0 * t[:, None]) / 28.0 + phase)
        else:                                                    # standing, an antinode at the base
            field = np.cos(2 * np.pi * (lons[None, :] - base) / 28.0) * np.sin(2 * np.pi * t[:, None] / 4.0 + phase)
        field = field + rng.normal(0, 0.5, field.shape)
        f = P.filter_supported(field, w)
        sums.append(P.lag_sums(f[:, j], f))
    return P.estimate(np.stack(sums), lons, base, n_boot=200)


def test_a_synthetic_traveling_signal_is_recovered_as_westward_propagation_at_its_speed():
    P = _load()
    e = _synthetic(P, "traveling")
    assert e["category"] == "westward propagation" and abs(e["speed_deg_per_day"] + 7.0) < 1.0
    assert e["ci_adjusted"][1] < 0 and e["failed_fraction"] == 0.0


def test_a_synthetic_standing_signal_is_reported_as_no_resolved_propagation():
    P = _load()
    e = _synthetic(P, "standing")
    assert e["category"] == "no resolved propagation" and abs(e["speed_deg_per_day"]) < 1.0
    assert e["ci_adjusted"][0] <= 0.0 <= e["ci_adjusted"][1]


def test_the_categories_follow_the_declared_rule():
    P = _load()
    assert P.classify(-8.0, -2.0, 0.0) == "westward propagation" and P.classify(1.0, 3.0, 0.0) == "eastward propagation"
    assert P.classify(-3.0, 2.0, 0.0) == "no resolved propagation"
    assert P.classify(-6.0, 1.0, 0.0) == "inconclusive"                            # zero and AEW-like speeds both inside
    assert P.classify(-8.0, -2.0, 0.2) == "inconclusive" and P.classify(None, None, 0.0) == "inconclusive"
    assert 2005 not in P.SEASONS and len(P.SEASONS) == 24


def test_a_flat_track_has_exactly_zero_speed_so_its_interval_contains_zero():
    P = _load()
    lons = np.arange(40.0, 60.25, 0.5)
    bmap = np.zeros((len(P.LAG_STEPS), lons.size)); j = int(np.argmin(np.abs(lons - 50.0)))
    bmap[:, j] = 1.0                                                    # a lobe fixed at the base at every lag
    speed, track = P.track_speed(bmap, lons, 50.0)
    assert speed == 0.0 and len(track) == 2 * P.SPEED_MAX_STEP + 1
    assert P.classify(-2.0, speed, 0.0) == "no resolved propagation"
