import numpy as np

from aew.events import composite_dates, std_threshold


def test_selects_local_maxima_above_threshold():
    base = np.array([0.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0])
    time = np.arange(base.size)
    res = composite_dates(base, thresh=1.5, time=time)
    # local maxima are at idx 1 (1.0), 3 (3.0), 5 (2.0); keep >=1.5 -> idx 3, 5
    np.testing.assert_array_equal(res.index, [3, 5])
    np.testing.assert_array_equal(res.time, [3, 5])
    np.testing.assert_array_equal(res.amp, [3.0, 2.0])
    assert len(res) == 2


def test_negative_threshold_selects_local_minima():
    base = np.array([0.0, -1.0, 0.0, -3.0, 0.0, -2.0, 0.0])
    res = composite_dates(base, thresh=-1.5)
    # local minima at idx 3 (-3) and 5 (-2); keep <= -1.5
    np.testing.assert_array_equal(res.index, [3, 5])
    np.testing.assert_array_equal(res.amp, [-3.0, -2.0])


def test_no_peaks_above_threshold_returns_empty():
    base = np.array([0.0, 0.5, 0.0, 0.4, 0.0])
    res = composite_dates(base, thresh=2.0)
    assert len(res) == 0


def test_plateau_not_selected_as_strict_max():
    # a flat top is not a strict local maximum (matches local_max_1d delta=0)
    base = np.array([0.0, 2.0, 2.0, 0.0])
    res = composite_dates(base, thresh=1.0)
    assert len(res) == 0


def test_std_threshold_uses_sample_std():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(10000)
    thr = std_threshold(base, 2.0)
    assert abs(thr - 2.0 * np.std(base, ddof=1)) < 1e-12
