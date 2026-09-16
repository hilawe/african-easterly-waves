"""Tests for the version 1 curvature-vorticity climatology and threshold port.

THE MUTATION LIST, WRITTEN BEFORE THE ASSERTIONS EXISTED, executable as
tests/mutations_v1port_climatology.py.

    C1   the climatology averages over ALL times rather than per calendar step, giving
         one field instead of a seasonal cycle
    C2   the calendar key drops the hour, so the four steps of a day are pooled
    C3   the calendar key includes the year, so every step has exactly one sample and
         nothing is averaged at all
    C4   the leap day is left as its own sparse mean instead of being interpolated
    C5   the leap-day interpolation uses the wrong endpoints
    C6   the anomaly adds the climatology instead of subtracting it
    C7   the anomaly matches by position rather than by calendar step
    C8   the Southern Hemisphere sign flip is not applied before taking percentiles
    C9   the sign flip is applied to the whole field, not only south of the equator
    C10  the thresholds are percentiles of the RAW curvature vorticity rather than of
         the smoothed anomaly the masks actually compare against
    C11  the smoother is skipped before the percentile
    C12  the nine-point smoother weights the diagonals and edges the same way
    C13  the smoother modifies its outermost ring instead of leaving it alone
    C14  the completeness check never reports a short calendar step
    C15  decimation subsamples without the Gaussian smoothing that precedes it
    C16  both thresholds are taken from the fine grid, so the coarse one is calibrated
         on a distribution its mask never sees
"""

import numpy as np
import pytest

from aew.v1port.climatology import (
    COARSE_PERCENTILE, FINE_PERCENTILE, anomaly_threshold, calendar_key,
    SMOOTH_P, SMOOTH_Q, decimation_shape, subsample_coordinates,
    curvature_anomaly, curvature_climatology, detection_thresholds, gaussian_decimate,
    smooth9, southern_hemisphere_sign,
)


def six_hourly(start="2001-01-01", days=3):
    return np.arange(np.datetime64(start), np.datetime64(start) + np.timedelta64(days, "D"),
                     np.timedelta64(6, "h"))


# --- the nine-point smoother ---------------------------------------------------------

def test_smoother_leaves_a_constant_field_unchanged():
    f = np.full((5, 6), 3.0)
    assert np.allclose(smooth9(f), 3.0)


def test_smoother_leaves_the_outer_ring_untouched():
    """C13. The MATLAB loop runs 2:end-1, so the border keeps its input value."""
    rng = np.random.default_rng(0)
    f = rng.normal(size=(6, 7))
    out = smooth9(f)
    assert np.array_equal(out[0, :], f[0, :])
    assert np.array_equal(out[-1, :], f[-1, :])
    assert np.array_equal(out[:, 0], f[:, 0])
    assert np.array_equal(out[:, -1], f[:, -1])


def test_smoother_matches_the_written_formula_at_one_point():
    """C12. Compute the centre by hand from smth9_f.m's own expression."""
    f = np.arange(9, dtype=float).reshape(3, 3)
    p, q = 0.5, 0.25
    c = f[1, 1]
    expected = (c
                + (p / 4) * (f[0, 1] + f[1, 0] + f[2, 1] + f[1, 2] - 4 * c)
                + (q / 4) * (f[0, 2] + f[0, 0] + f[2, 0] + f[2, 2] - 4 * c))
    assert smooth9(f)[1, 1] == pytest.approx(expected)


def test_smoother_reduces_point_to_point_variance():
    rng = np.random.default_rng(1)
    f = rng.normal(size=(30, 30))
    inner = (slice(1, -1), slice(1, -1))
    assert np.var(smooth9(f)[inner]) < np.var(f[inner])


def test_smoother_propagates_nan_because_the_original_guard_is_dead():
    """FAITHFUL. smth9_f guards on `sum(temp == missing) == 0` with missing = NaN, and
    NaN == NaN is false in MATLAB, so the guard always passes and NaN spreads."""
    f = np.ones((5, 5))
    f[2, 2] = np.nan
    out = smooth9(f)
    assert np.isnan(out[1, 1]) and np.isnan(out[3, 3])
    assert np.isfinite(out[0, 0])          # the untouched border survives


def test_smoother_refuses_a_three_dimensional_field():
    with pytest.raises(ValueError, match="2-D"):
        smooth9(np.zeros((2, 3, 3)))


# --- the calendar grouping -----------------------------------------------------------

def test_calendar_key_extracts_month_day_hour():
    t = np.array(["2001-03-04T12", "2004-02-29T18"], dtype="datetime64[ns]")
    m, d, h = calendar_key(t)
    assert list(m) == [3, 2] and list(d) == [4, 29] and list(h) == [12, 18]


def test_same_calendar_step_in_different_years_shares_a_key():
    """C3. If the year leaked into the key nothing would ever be averaged."""
    t = np.array(["1999-07-04T06", "2005-07-04T06"], dtype="datetime64[ns]")
    m, d, h = calendar_key(t)
    assert (m[0], d[0], h[0]) == (m[1], d[1], h[1])


# --- the climatological mean ---------------------------------------------------------

def test_climatology_averages_across_years_within_a_calendar_step():
    """C1 and C2. Two years of one day; each step must average its own two samples."""
    times = np.array(["2001-06-01T00", "2001-06-01T06",
                      "2002-06-01T00", "2002-06-01T06"], dtype="datetime64[ns]")
    field = np.zeros((4, 2, 2))
    field[0] = 1.0
    field[1] = 10.0
    field[2] = 3.0
    field[3] = 20.0
    clim = curvature_climatology(field, times, interpolate_leap_day=False)
    assert clim["keys"] == [(6, 1, 0), (6, 1, 6)]
    assert np.allclose(clim["mean"][0], 2.0)      # (1 + 3) / 2
    assert np.allclose(clim["mean"][1], 15.0)     # (10 + 20) / 2
    assert list(clim["counts"]) == [2, 2]


def test_climatology_keeps_a_seasonal_cycle_rather_than_one_number():
    """C1 stated the other way round: different calendar steps must differ."""
    times = six_hourly("2001-01-01", days=4)
    field = (np.arange(len(times), dtype=float)[:, None, None]
             * np.ones((1, 3, 3)))
    clim = curvature_climatology(field, times, interpolate_leap_day=False)
    assert len(clim["keys"]) == len(times)
    assert np.ptp([m.mean() for m in clim["mean"]]) > 0


def test_short_calendar_steps_are_reported():
    """C14."""
    times = np.array(["2001-06-01T00", "2002-06-01T00", "2001-06-01T06"],
                     dtype="datetime64[ns]")
    field = np.ones((3, 2, 2))
    clim = curvature_climatology(field, times, expected_years=2,
                                 interpolate_leap_day=False)
    assert ((6, 1, 6), 1) in clim["short_steps"]
    assert ((6, 1, 0), 2) not in clim["short_steps"]


def test_leap_day_is_interpolated_across_the_gap():
    """C4 and C5. The original replaces 29 February with a straight line from the last
    step of 28 February to the first of 1 March, because a thirty-year window holds
    only seven or eight leap days."""
    times = np.array(["2004-02-28T18", "2004-02-29T00", "2004-02-29T06",
                      "2004-02-29T12", "2004-02-29T18", "2004-03-01T00"],
                     dtype="datetime64[ns]")
    field = np.zeros((6, 1, 1))
    field[0] = 0.0        # 28 Feb 18:00
    field[1:5] = 99.0     # the sparse leap-day values, to be discarded
    field[5] = 5.0        # 1 Mar 00:00
    clim = curvature_climatology(field, times, interpolate_leap_day=True)
    values = [float(m[0, 0]) for m in clim["mean"]]
    assert values[0] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(5.0)
    assert values[1:5] == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_leap_day_can_be_left_alone():
    times = np.array(["2004-02-28T18", "2004-02-29T00", "2004-03-01T00"],
                     dtype="datetime64[ns]")
    field = np.zeros((3, 1, 1))
    field[1] = 99.0
    clim = curvature_climatology(field, times, interpolate_leap_day=False)
    assert float(clim["mean"][1][0, 0]) == pytest.approx(99.0)


def test_climatology_refuses_mismatched_times_and_empty_input():
    with pytest.raises(ValueError, match="entries"):
        curvature_climatology(np.zeros((3, 2, 2)), six_hourly(days=1)[:2])
    with pytest.raises(ValueError, match="no timesteps"):
        curvature_climatology(np.zeros((0, 2, 2)),
                              np.array([], dtype="datetime64[ns]"))


# --- the anomaly ---------------------------------------------------------------------

def test_anomaly_subtracts_the_matching_calendar_step():
    """C6 and C7."""
    times = np.array(["2001-06-01T00", "2002-06-01T00"], dtype="datetime64[ns]")
    field = np.zeros((2, 1, 1))
    field[0] = 1.0
    field[1] = 3.0
    clim = curvature_climatology(field, times, interpolate_leap_day=False)
    anom = curvature_anomaly(field, times, clim)
    assert float(anom[0][0, 0]) == pytest.approx(-1.0)   # 1 - mean 2
    assert float(anom[1][0, 0]) == pytest.approx(1.0)    # 3 - mean 2
    assert float(np.mean(anom)) == pytest.approx(0.0)


def test_anomaly_matches_by_calendar_not_by_position():
    """C7 specifically: reverse the input order and the answer must follow the dates."""
    times = np.array(["2001-06-01T00", "2001-06-01T06"], dtype="datetime64[ns]")
    field = np.zeros((2, 1, 1))
    field[0], field[1] = 10.0, 20.0
    clim = curvature_climatology(field, times, interpolate_leap_day=False)
    flipped_times = times[::-1]
    flipped_field = field[::-1]
    anom = curvature_anomaly(flipped_field, flipped_times, clim)
    assert np.allclose(anom, 0.0)


def test_anomaly_refuses_a_calendar_step_the_climatology_does_not_cover():
    times = np.array(["2001-06-01T00"], dtype="datetime64[ns]")
    clim = curvature_climatology(np.ones((1, 1, 1)), times,
                                 interpolate_leap_day=False)
    other = np.array(["2001-07-04T12"], dtype="datetime64[ns]")
    with pytest.raises(ValueError, match="does not cover"):
        curvature_anomaly(np.ones((1, 1, 1)), other, clim)


# --- the hemispheric sign convention and the thresholds ------------------------------

def test_sign_flip_applies_only_south_of_the_equator():
    """C9."""
    lat = np.array([-10.0, 0.0, 10.0])
    field = np.ones((1, 3, 2))
    out = southern_hemisphere_sign(field, lat)
    assert np.all(out[0, 0, :] == -1.0)
    assert np.all(out[0, 1, :] == 1.0)      # the equator itself is not flipped
    assert np.all(out[0, 2, :] == 1.0)


def test_sign_flip_refuses_a_mismatched_latitude_axis():
    with pytest.raises(ValueError, match="latitude axis"):
        southern_hemisphere_sign(np.ones((1, 3, 2)), np.array([1.0, 2.0]))


def test_threshold_is_the_percentile_of_the_smoothed_sign_adjusted_field():
    """C10 and C11. The mask compares against the smoothed, sign-adjusted anomaly, so
    the percentile has to be of exactly that field."""
    rng = np.random.default_rng(4)
    lat = np.linspace(-20, 20, 21)
    anomaly = rng.normal(size=(6, 21, 21)) * 1e-5
    got = anomaly_threshold(anomaly, lat, COARSE_PERCENTILE)
    smoothed = np.stack([smooth9(step) for step in anomaly])
    adjusted = southern_hemisphere_sign(smoothed, lat)
    assert got == pytest.approx(np.percentile(adjusted, COARSE_PERCENTILE))


def test_threshold_moves_when_the_sign_flip_is_omitted():
    """C8."""
    rng = np.random.default_rng(5)
    lat = np.linspace(-20, 20, 21)
    anomaly = rng.normal(size=(4, 21, 21)) * 1e-5
    with_flip = anomaly_threshold(anomaly, lat, COARSE_PERCENTILE)
    smoothed = np.stack([smooth9(s) for s in anomaly])
    without_flip = float(np.percentile(smoothed, COARSE_PERCENTILE))
    assert with_flip != pytest.approx(without_flip)


def test_threshold_ignores_non_finite_points():
    lat = np.array([5.0, 6.0, 7.0])
    anomaly = np.full((1, 3, 3), 2.0)
    anomaly[0, 0, 0] = np.nan
    assert np.isfinite(anomaly_threshold(anomaly, lat, 55.0, smooth=False))


def test_threshold_refuses_an_all_missing_field():
    lat = np.array([5.0, 6.0, 7.0])
    with pytest.raises(ValueError, match="no finite"):
        anomaly_threshold(np.full((1, 3, 3), np.nan), lat, 55.0, smooth=False)


def test_the_cardinal_and_diagonal_weights_are_different_numbers():
    """C12, bound properly at last. smth9_f weights the four EDGE-ADJACENT neighbours by
    p/4 and the four DIAGONAL ones by q/4, and version 1 always calls it with p = 0.5 and
    q = 0.25, so the two are not interchangeable. A catalogue mutation setting q to p
    survived until this test existed: nothing else in the suite could tell a smoother that
    treats all eight neighbours alike from one that does not.

    Each weight is isolated by a field that is zero everywhere except one neighbour, so the
    centre's new value is that weight alone.
    """
    cardinal = np.zeros((3, 3))
    cardinal[0, 1] = 1.0                       # due north of the centre
    assert smooth9(cardinal)[1, 1] == pytest.approx(SMOOTH_P / 4.0)

    diagonal = np.zeros((3, 3))
    diagonal[0, 0] = 1.0                       # northwest of the centre
    assert smooth9(diagonal)[1, 1] == pytest.approx(SMOOTH_Q / 4.0)

    assert SMOOTH_P == 0.5 and SMOOTH_Q == 0.25
    assert SMOOTH_P != SMOOTH_Q, "the whole point of the two-term form"


def test_the_filter_width_and_the_stride_are_computed_separately():
    """P2 and P3, and the defect that made this test worth writing.

    decimate_f.m computes the filter width and the subsampling stride from the two
    resolutions by DIFFERENT rules, and they differ whenever the ratio divides exactly. A
    version of the decimation that took one factor and used `factor + 1` as the width was
    right for every exact ratio the unit tests used and wrong for the first real one:
    ERA-Interim's own 2.5 over 0.75 is 3.33, where the width is 3 and that version used 4.
    """
    assert decimation_shape(0.75, 2.5) == (3, 3), "a non-exact ratio: width equals stride"
    assert decimation_shape(0.5, 1.0) == (3, 2), "an exact ratio: width is one wider"
    assert decimation_shape(2.5, 2.5) == (2, 1), "ratio one still filters"
    with pytest.raises(ValueError, match="finer than the input"):
        decimation_shape(2.5, 1.0)


def test_coarse_coordinates_are_subsampled_not_smoothed():
    """P2. Running a coordinate array through the data filter distorts it at the edges,
    where the convolution has nothing to average against, and the grid stops being evenly
    spaced. A first version of the pipeline did that and produced a negative spacing."""
    lat = np.arange(-35.0, 35.1, 0.75)
    coarse = subsample_coordinates(lat, 3)
    spacing = np.diff(coarse)
    assert np.allclose(spacing, 2.25), "evenly spaced at exactly stride times the input"
    assert coarse[0] == pytest.approx(lat[0]), "and starting where the fine grid starts"


# --- decimation, and the two thresholds coming from two different grids ---------------

def test_decimation_shrinks_the_grid_by_the_stride():
    field = np.zeros((2, 21, 41))
    out = gaussian_decimate(field, 1.0, 3.0)
    assert out.shape == (2, 7, 14)


def test_decimation_at_the_same_resolution_still_smooths():
    """NOT A NO-OP, which an earlier version of this test asserted and decimate_f.m does
    not do. When the ratio divides exactly the filter is one wider than the stride, so a
    ratio of one gives a two-by-two filter and a stride of one: the grid keeps its shape
    and the values change. Reproducing that matters because the width and the stride are
    computed by different rules, and a version that derived one from the other was wrong
    for every ratio that does not divide exactly, including ERA-Interim's own."""
    rng = np.random.default_rng(6)
    field = rng.normal(size=(1, 9, 9))
    out = gaussian_decimate(field, 2.5, 2.5)
    assert out.shape == field.shape
    assert not np.array_equal(out, field)


def test_decimation_preserves_a_constant_field_in_the_interior():
    """A normalized kernel must not change a flat field away from the zero-padded edge."""
    field = np.full((1, 15, 15), 4.0)
    out = gaussian_decimate(field, 1.0, 3.0)
    assert out[0, 2, 2] == pytest.approx(4.0)


def test_decimation_smooths_before_subsampling():
    """C15. Plain subsampling would keep the point-to-point variance; the Gaussian must
    reduce it, which is the whole reason decimate_f convolves first."""
    rng = np.random.default_rng(7)
    field = rng.normal(size=(1, 61, 61))
    smoothed_then_taken = gaussian_decimate(field, 1.0, 3.0)[0][1:-1, 1:-1]
    plainly_taken = field[0, ::3, ::3][1:-1, 1:-1]
    assert np.var(smoothed_then_taken) < 0.6 * np.var(plainly_taken)


def test_decimation_refuses_to_refine():
    """decimate_f coarsens and has no path that refines, so asking it to is a caller error
    rather than something to interpolate around."""
    with pytest.raises(ValueError, match="finer than the input"):
        gaussian_decimate(np.zeros((1, 4, 4)), 2.5, 1.0)


def test_the_two_thresholds_come_from_two_different_grids():
    """C16, and this is the defect the split API exists to prevent. find_ews_f.m gates
    the coarse mask with the decimated field and the fine mask with the input field, so
    taking both percentiles from one array calibrates the coarse threshold on the wrong
    distribution."""
    rng = np.random.default_rng(8)
    lat = np.linspace(-20, 20, 41)
    anomaly = rng.normal(size=(4, 41, 41)) * 1e-5

    coarse, fine = detection_thresholds(anomaly, lat, decimation_factor=2)
    same_grid_coarse = anomaly_threshold(anomaly, lat, COARSE_PERCENTILE)
    assert coarse != pytest.approx(same_grid_coarse), (
        "the coarse threshold was taken from the fine grid")

    expected = anomaly_threshold(gaussian_decimate(anomaly, 1.0, 2), lat[::2],
                                 COARSE_PERCENTILE)
    assert coarse == pytest.approx(expected)
    assert fine == pytest.approx(anomaly_threshold(anomaly, lat, FINE_PERCENTILE))


def test_the_two_percentiles_are_the_ones_the_source_names():
    assert (COARSE_PERCENTILE, FINE_PERCENTILE) == (55.0, 66.0)


def test_the_even_kernel_window_matches_matlab_not_scipy():
    """The decimation's convolution window, anchored to version 1's own output.

    MUTATIONS THIS BINDS, listed before the assertions:
      D1 the offset reverts to scipy's convention, `(width - 1) // 2`
      D2 the offset is applied to rows but not columns
      D3 the window is taken from the wrong end, `-offset` rather than `offset`
      D4 mode="same" is used again instead of an explicit window

    WHY IT EXISTS. For an EVEN-width kernel the "same" window of a convolution is ambiguous
    by one cell, and MATLAB and scipy resolve it in opposite corners. Version 1's
    `decimate_f.m` calls MATLAB `conv2(...,'same')`; the port called scipy's
    `convolve2d(mode='same')`, so every decimated field was shifted one cell. Running the
    archived function under Octave on a real 1990 field, the two differed at every cell,
    median 1.0 against a field rms of 9.8.

    IT WAS UNREACHABLE UNTIL RECENTLY. `decimation_shape` gives an ODD width of 3 for the
    0.75 degree tree this project used for most of its life, and odd kernels are
    unambiguous. Version 1's own one-degree input gives width 2. No test caught the
    difference because none bound the decimated VALUES, only shapes and invariants.

    THE EXPECTED VALUES ARE COMPUTED FROM THE DEFINITION, not copied from a run: the full
    convolution's offset-(1,1) window, which is what Octave's conv2 was measured to return.
    """
    from aew.v1port import climatology as clim

    rng = np.random.default_rng(11)
    field = rng.normal(size=(2, 9, 11))
    got = clim.gaussian_decimate(field, 1.0, 2.5)

    width, stride = clim.decimation_shape(1.0, 2.5)
    assert width == 2 and width % 2 == 0, "this case must have an EVEN kernel to bind"

    from scipy.signal import convolve2d
    kernel = clim.gaussian_kernel(width)
    offset = width // 2
    want = np.stack([
        convolve2d(step, kernel, mode="full")[offset:offset + 9, offset:offset + 11]
        for step in field])[:, ::stride, ::stride]
    assert np.allclose(got, want), "the decimation must take MATLAB's even-kernel window"

    # and it must NOT be scipy's window, which is the defect this replaced
    scipy_same = np.stack([convolve2d(step, kernel, mode="same") for step in field]
                          )[:, ::stride, ::stride]
    assert not np.allclose(got, scipy_same), \
        "scipy's mode='same' window is the wrong corner for an even kernel"

    # an ODD kernel must be unaffected, because there the two conventions agree
    odd_width, _ = clim.decimation_shape(0.75, 2.5)
    assert odd_width % 2 == 1
    odd_field = rng.normal(size=(2, 12, 12))
    odd_got = clim.gaussian_decimate(odd_field, 0.75, 2.5)
    ok = clim.gaussian_kernel(odd_width)
    _, odd_stride = clim.decimation_shape(0.75, 2.5)
    odd_same = np.stack([convolve2d(s, ok, mode="same") for s in odd_field]
                        )[:, ::odd_stride, ::odd_stride]
    assert np.allclose(odd_got, odd_same), \
        "for an odd kernel the two conventions agree and nothing should change"
