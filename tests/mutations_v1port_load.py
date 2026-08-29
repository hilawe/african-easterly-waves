"""Mutations for the reanalysis loader.

The list was written before the assertions existed and is reproduced in the docstring of
tests/test_v1port_load.py. Run with the repository's mutation checker against
src/aew/v1port/load.py.
"""


def _sub(old, new):
    return lambda s: s.replace(old, new)


MUTATIONS = {
    # L1 the epoch offset is wrong, which nothing downstream can detect
    "epoch_offset_wrong": _sub(
        "EPOCH_OFFSET_DAYS = 25567",
        "EPOCH_OFFSET_DAYS = 25566"),
    # L1 the offset is not applied at all
    "epoch_offset_not_applied": _sub(
        "    return np.asarray(seconds, dtype=float) / SECONDS_PER_DAY + "
        "EPOCH_OFFSET_DAYS",
        "    return np.asarray(seconds, dtype=float) / SECONDS_PER_DAY"),
    # L1 seconds are read as days
    "seconds_not_converted_to_days": _sub(
        "    return np.asarray(seconds, dtype=float) / SECONDS_PER_DAY + "
        "EPOCH_OFFSET_DAYS",
        "    return np.asarray(seconds, dtype=float) + EPOCH_OFFSET_DAYS"),
    # L2 an unknown time unit is accepted and treated as the common case
    "unknown_time_units_accepted": _sub(
        "        raise ValueError(\n"
        "            f\"the time variable {name!r} has units {units!r}, which this loader "
        "does not \"",
        "        return seconds_since_1970_to_days_since_1900(values)\n"
        "        raise ValueError(\n"
        "            f\"the time variable {name!r} has units {units!r}, which this loader "
        "does not \""),
    # L3 the u and v time axes are not compared
    "time_axes_not_compared": _sub(
        "    if (v_times.shape != times.shape\n"
        "            or not np.allclose(v_times, times, rtol=0.0, atol=one_second)):",
        "    if False:"),
    # L3 the comparison uses a relative tolerance, which on day numbers near thirty
    # thousand is about a third of a day and hides a six-hour offset
    "time_axis_comparison_uses_relative_tolerance": _sub(
        "            or not np.allclose(v_times, times, rtol=0.0, atol=one_second)):",
        "            or not np.allclose(v_times, times)):"),
    # L4 the length-one pressure-level axis is left in place
    "pressure_level_axis_not_dropped": _sub(
        "            while data.ndim > 3:",
        "            while False:"),
    # L5 a missing file is treated as an empty year
    "missing_file_not_refused": _sub(
        "        if not os.path.exists(path):",
        "        if False:"),
    # L6 a year with only one wind file is offered as available
    "half_a_year_counts_as_available": _sub(
        "        if os.path.exists(os.path.join(directory,\n"
        '                                       f"{prefix}_v700_{year}_6h_region.nc")):',
        "        if True:"),
    # L8 a missing value is dropped from the divisor as well as the sum
    "missing_values_excluded_from_the_count": _sub(
        "            self._total[key] += np.nan_to_num(curvature[position], nan=0.0)\n"
        "            self._counts[key] += 1",
        "            self._total[key] += np.nan_to_num(curvature[position], nan=0.0)\n"
        "            self._counts[key] += int(np.isfinite(curvature[position]).all())"),
    # L8 missing values propagate instead of summing as zero
    "missing_values_propagate": _sub(
        "            self._total[key] += np.nan_to_num(curvature[position], nan=0.0)",
        "            self._total[key] += curvature[position]"),
    # L9 the leap-day interpolation is skipped
    "leap_day_not_interpolated": _sub(
        "        if interpolate_leap_day:\n"
        "            mean = clim._interpolate_leap_day(mean, keys)",
        "        if False:\n            mean = clim._interpolate_leap_day(mean, keys)"),
    # L7 the calendar keys are not sorted, so they no longer line up with the batch order
    "keys_not_sorted": _sub(
        "        keys = sorted(self._total)",
        "        keys = list(self._total)"),
    # L10 grids of different shapes are mixed
    "grids_may_be_mixed": _sub(
        "        elif curvature.shape[1:] != self._shape:",
        "        elif False:"),
    # L11 the sampler draws missing values, pulling every percentile toward nothing
    "sampler_draws_missing_values": _sub(
        "    flat = field[np.isfinite(field)]",
        "    flat = np.nan_to_num(field, nan=0.0).ravel()"),
    # L11 the sampler draws with replacement, which biases the percentile
    "sampler_draws_with_replacement": _sub(
        "    return flat[rng.choice(flat.size, size=count, replace=False)]",
        "    return flat[rng.choice(flat.size, size=count, replace=True)]"),
    # L12 the sampling error is reported as a constant, so a noisy estimate looks solid
    "sampling_error_is_a_constant": _sub(
        "    spread = (max(draws) - min(draws)) / abs(estimate) if estimate else "
        'float("inf")',
        "    spread = 0.0"),
    # L12 the estimate is not the percentile it claims to be
    "estimate_is_not_the_percentile": _sub(
        "    estimate = float(np.percentile(sample, percentile))",
        "    estimate = float(np.mean(sample))"),
    # the short-step report stops excluding 29 February, which never has a full count
    "leap_day_reported_as_a_short_step": _sub(
        "                if (self._counts[k] != expected_years\n"
        "                        and (k[0], k[1]) != (clim.LEAP_DAY_MONTH, "
        "clim.LEAP_DAY_DAY)):",
        "                if self._counts[k] != expected_years:"),
}
