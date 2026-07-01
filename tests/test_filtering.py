import os

import numpy as np
import pytest

from aew.filtering import bandpass_weights, filwgts_lanczos, lanczos_bandpass, wgt_runave

REF = os.path.join(os.path.dirname(__file__), "data", "ncl_lanczos_weights_61_0.1_0.5.txt")


def test_weights_symmetric_and_length():
    w = filwgts_lanczos(61, 2, 0.1, 0.5, 1)
    assert w.size == 61
    np.testing.assert_allclose(w, w[::-1], atol=1e-15)


def test_bandpass_weights_sum_near_zero():
    # a band-pass filter has weights summing to ~0 (removes the mean / DC)
    w = bandpass_weights(2, 10, obs_per_day=1, buffer_days=30)
    assert abs(w.sum()) < 1e-2


def test_bandpass_weights_match_legacy_setup():
    # legacy: nWgts=61, filwgts_lanczos(61, 2, freq(1)=0.1, freq(0)=0.5, 1)
    w1 = bandpass_weights(2, 10, obs_per_day=1, buffer_days=30)
    w2 = filwgts_lanczos(61, 2, 0.1, 0.5, 1)
    np.testing.assert_allclose(w1, w2, atol=1e-15)


@pytest.mark.skipif(not os.path.exists(REF), reason="NCL reference weights not present")
def test_weights_close_to_ncl_reference():
    # This implementation is textbook Duchon/Lanczos. NCL 6.x applies a slightly
    # different normalization, so the weights agree only to ~5e-4 (abs), not bit
    # for bit. This is enough to reproduce composite-date COUNTS (peaks above
    # n*sigma are robust to a 0.05% weight change) but NOT to match the printed
    # filtered-v700 thresholds (e.g. 3.26136 m/s) to 6 figures. Bit-exact NCL
    # parity is deferred to the data-validation stage; see docs/VALIDATION_TARGETS.md.
    # When exact parity is needed, use NCL-generated weights from tests/data/.
    ref = np.loadtxt(REF)
    w = filwgts_lanczos(61, 2, 0.1, 0.5, 1)
    assert ref.size == 61
    np.testing.assert_allclose(w, ref, atol=1e-3)


def test_passband_keeps_5day_rejects_30day_daily():
    # Daily sampling (obs_per_day=1). Nyquist is the 2-day period, which is the
    # band's high cutoff, so the meaningful action here is the LOW cut (remove >10d).
    n = 1000
    t = np.arange(n)
    sig5 = np.cos(2 * np.pi * t / 5.0)  # 5-day, inside 2-10 band
    sig30 = np.cos(2 * np.pi * t / 30.0)  # 30-day, below band
    f5 = lanczos_bandpass(sig5)
    f30 = lanczos_bandpass(sig30)
    valid = np.isfinite(f5)
    assert np.std(f5[valid]) > 0.6  # 5-day passes (unit cosine std ~ 0.707)
    assert np.std(f30[valid]) < 0.05  # 30-day removed by the low cut


def test_highcut_rejects_subdaily_signal():
    # Sub-daily sampling (4/day) so a 1-day signal is representable and lies ABOVE
    # the 2-day high cutoff. It should be rejected; a 5-day signal should pass.
    obs = 4
    n = 4000
    t = np.arange(n) / obs  # time in days
    sig1 = np.cos(2 * np.pi * t / 1.0)  # 1-day, above band
    sig5 = np.cos(2 * np.pi * t / 5.0)  # 5-day, in band
    f1 = lanczos_bandpass(sig1, obs_per_day=obs)
    f5 = lanczos_bandpass(sig5, obs_per_day=obs)
    valid = np.isfinite(f1)
    assert np.std(f5[valid]) > 0.6
    assert np.std(f1[valid]) < 0.05


def test_wgt_runave_sets_ends_to_nan():
    x = np.ones(100)
    w = np.ones(5) / 5.0
    out = wgt_runave(x, w)
    assert np.all(np.isnan(out[:2]))
    assert np.all(np.isnan(out[-2:]))
    np.testing.assert_allclose(out[2:-2], 1.0)


def test_wgt_runave_multidim_axis():
    x = np.ones((3, 100))
    w = np.ones(5) / 5.0
    out = wgt_runave(x, w, axis=1)
    assert out.shape == (3, 100)
    assert np.all(np.isnan(out[:, :2]))
    np.testing.assert_allclose(out[:, 2:-2], 1.0)


def test_default_axis_is_time_dim0():
    # Legacy NCL filters along dim 0 (time). A (time, lat, lon) field should filter
    # along time by default, NOT longitude.
    n = 600
    t = np.arange(n)
    field = np.empty((n, 2, 3))
    field[:] = np.cos(2 * np.pi * t / 5.0)[:, None, None]  # 5-day wave at every gridpoint
    out = lanczos_bandpass(field)  # default axis=0
    assert out.shape == field.shape
    valid = np.isfinite(out[:, 0, 0])
    assert np.std(out[valid, 0, 0]) > 0.6  # time-filtered, wave preserved


def test_fill_value_and_masked_become_missing():
    x = np.ones(100)
    x[50] = -999.0
    w = np.ones(5) / 5.0
    out = wgt_runave(x, w, fill_value=-999.0)
    # any window touching index 50 -> NaN (5-wide window => indices 48..52)
    assert np.all(np.isnan(out[48:53]))
    assert np.isfinite(out[40])

    xm = np.ma.array(np.ones(100), mask=False)
    xm[50] = np.ma.masked
    out2 = wgt_runave(xm, w)
    assert np.all(np.isnan(out2[48:53]))

    # masked INTEGER array must not raise (float cast happens before filling NaN)
    xi = np.ma.array(np.ones(100, dtype=int), mask=False)
    xi[50] = np.ma.masked
    out3 = wgt_runave(xi, w)
    assert np.all(np.isnan(out3[48:53]))
    assert np.isfinite(out3[10])
