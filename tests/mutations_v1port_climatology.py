"""Mutations for the version 1 climatology and threshold port.

The list was written before the assertions existed and is reproduced in the docstring of
tests/test_v1port_climatology.py. Run with the repository's mutation checker against
src/aew/v1port/climatology.py.
"""


def _sub(old, new):
    return lambda s: s.replace(old, new)


MUTATIONS = {
    # C2 the hour dropped from the key, pooling the four steps of a day
    "calendar_key_drops_the_hour": _sub(
        "    return months, days, hours",
        "    return months, days, np.zeros_like(hours)"),
    # C3 the year kept in the key, so nothing is ever averaged
    "calendar_key_keeps_the_year": _sub(
        "    months = times.astype(\"datetime64[M]\").astype(int) % 12 + 1",
        "    months = times.astype(\"datetime64[M]\").astype(int)"),
    # C4 the leap day left as its own sparse mean
    "leap_day_not_interpolated": _sub(
        "    if interpolate_leap_day:\n        mean = _interpolate_leap_day(mean, unique_keys)",
        "    if False:\n        mean = _interpolate_leap_day(mean, unique_keys)"),
    # C5 the leap-day interpolation anchored on the wrong endpoints
    "leap_day_wrong_endpoints": _sub(
        "    before = min(leap) - 1\n    after = max(leap) + 1",
        "    before = min(leap)\n    after = max(leap)"),
    # C6 the anomaly adds instead of subtracts
    "anomaly_adds_the_climatology": _sub(
        "        out[position] = curvature[position] - climatology[\"mean\"][index[key]]",
        "        out[position] = curvature[position] + climatology[\"mean\"][index[key]]"),
    # C7 the anomaly matched by position rather than by calendar step
    "anomaly_matches_by_position": _sub(
        "        out[position] = curvature[position] - climatology[\"mean\"][index[key]]",
        "        out[position] = curvature[position] - climatology[\"mean\"][position]"),
    # C8 the sign flip skipped before the percentile
    "no_southern_sign_flip": _sub(
        "    adjusted = southern_hemisphere_sign(anomaly, lat_values)",
        "    adjusted = anomaly"),
    # C9 the sign flip applied everywhere rather than only south
    "sign_flip_applied_globally": _sub(
        "    south = lat_values < 0",
        "    south = np.ones_like(lat_values, dtype=bool)"),
    # C11 the smoother skipped before the percentile
    "threshold_skips_the_smoother": _sub(
        "    if smooth:\n        anomaly = np.stack([smooth9(step) for step in anomaly])\n"
        "    adjusted = southern_hemisphere_sign(anomaly, lat_values)",
        "    adjusted = southern_hemisphere_sign(anomaly, lat_values)"),
    # C12 the smoother weighting the diagonals like the edges
    "smoother_weights_are_equal": _sub(
        "def smooth9(field, p=SMOOTH_P, q=SMOOTH_Q):",
        "def smooth9(field, p=SMOOTH_P, q=SMOOTH_P):"),
    # C13 the smoother modifying its outermost ring
    "smoother_touches_the_border": _sub(
        "    out = field.copy()",
        "    out = np.zeros_like(field)"),
    # C14 the completeness check never fires
    "short_steps_never_reported": _sub(
        "            if counts[i] != expected_years and (k[0], k[1]) != (LEAP_DAY_MONTH,",
        "            if False and counts[i] != expected_years and (k[0], k[1]) != (LEAP_DAY_MONTH,"),
    # C15 decimation subsamples without smoothing first
    "decimation_skips_the_gaussian": _sub(
        "    smoothed = np.stack([convolve2d(step, kernel, mode=\"same\") for step in field])",
        "    smoothed = field"),
    # C16 both thresholds taken from the fine grid
    "both_thresholds_from_the_fine_grid": _sub(
        "    coarse_field = gaussian_decimate(anomaly_fine, decimation_factor)\n"
        "    lat_coarse = np.asarray(lat_fine, dtype=float)[::decimation_factor]\n"
        "    coarse = anomaly_threshold(coarse_field, lat_coarse, COARSE_PERCENTILE)",
        "    coarse = anomaly_threshold(anomaly_fine, lat_fine, COARSE_PERCENTILE)"),
}
