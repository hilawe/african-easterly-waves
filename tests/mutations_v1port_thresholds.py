"""Mutations for the threshold estimators.

The list was written before the assertions existed and is reproduced in the docstring of
tests/test_thresholds.py. Run with the repository's mutation checker against
src/aew/v1port/thresholds.py. No entry here hangs by design, so the plain checker is the
right harness, unlike the percentile catalog.
"""


def _sub(old, new):
    return lambda s: s.replace(old, new)


MUTATIONS = {
    # H1 the pass count is ignored, so T1 silently runs T0's transformation
    "smoothing_passes_ignored": _sub(
        "    for _ in range(passes):\n        out = clim.smooth9(out)",
        "    out = clim.smooth9(out)"),
    # H2 the sign adjustment is dropped from the transformed population
    "sign_adjustment_dropped": _sub(
        "    return clim.southern_hemisphere_sign(out[np.newaxis, ...], lats)[0]",
        "    return out"),
    # H3 the coarse scale goes the wrong way round
    "coarse_scale_directions_swapped": _sub(
        '''    if direction == "up":
        return value / 0.9
    if direction == "down":
        return value * 0.9''',
        '''    if direction == "up":
        return value * 0.9
    if direction == "down":
        return value / 0.9'''),
    # H3 the scale silently does nothing
    "coarse_scale_inert": _sub(
        '''    if direction == "up":
        return value / 0.9''',
        '''    if direction == "up":
        return value'''),
    # H4 the median becomes the mean
    "median_becomes_mean": _sub(
        "    median = float(np.median(estimates))",
        "    median = float(np.mean(estimates))"),
    # H5 the absolute value is dropped from the range denominator
    "range_denominator_signed": _sub(
        "    relative_range = (max(estimates) - min(estimates)) / abs(median)",
        "    relative_range = (max(estimates) - min(estimates)) / median"),
    # H6 the gate accepts equality
    "gate_accepts_equality": _sub(
        '            "passes": bool(relative_range < GATE)}',
        '            "passes": bool(relative_range <= GATE)}'),
    # H7 exhaustion returns the last level as if it were a result
    "exhaustion_returns_last_level": _sub(
        "    return None, records",
        "    return records[-1], records"),
    # H8 the first passing level is not honoured
    "first_passing_level_ignored": _sub(
        '        if entry["coarse"]["passes"] and entry["fine"]["passes"]:\n'
        "            return entry, records",
        "        if False:\n            return entry, records"),
    # H9 the generator falls back to the movable default
    "generator_not_pinned": _sub(
        "    return np.random.Generator(np.random.PCG64(seed))",
        "    return np.random.default_rng(seed)"),
    # H10 the exact-draw branch consumes the generator
    "exact_branch_consumes_generator": _sub(
        """                if finite.size <= per_step:
                    drawn.append(finite)
                else:
                    drawn.append(finite[rng.choice(finite.size, size=per_step,
                                                   replace=False)])""",
        """                drawn.append(finite[rng.choice(
                    finite.size, size=min(per_step, finite.size), replace=False)])"""),
    # H11 the upper-bound refusal is dropped
    "upper_bound_refusal_dropped": _sub(
        "    if count > total:",
        "    if False:"),
    # H12 the builder's written-equals-arithmetic assertion is dropped
    "builder_total_check_dropped": _sub(
        "        if written != expected:",
        "        if False:"),
    # H13 selected counts report the request, not the draw
    "selected_counts_report_the_request": _sub(
        "            n_sel = min(per_step, n_finite)",
        "            n_sel = per_step"),
    # H14 a zero median is divided by instead of refused
    "zero_median_divided": _sub(
        "    if median == 0.0 or not np.isfinite(median):",
        "    if not np.isfinite(median):"),
    # H15 the registry hands P2 the source-code periods, collapsing the two cases
    "registry_periods_collapsed": _sub(
        '        "climatology_years": (1981, 2010) if period == "P1" else (1980, 2010),',
        '        "climatology_years": (1981, 2010),'),
    # H15 the registry stops requiring the exact estimator
    "registry_estimator_unbound": _sub(
        '        "estimator": "exact",',
        '        "estimator": None,'),
    # H15 case validation reports nothing regardless of the settings
    "case_validation_silenced": _sub(
        "        if mismatch:\n"
        "            out.append(f\"{key}: the case requires {want!r}, the run supplies "
        "{got!r}\")",
        "        if False:\n"
        "            out.append(f\"{key}: the case requires {want!r}, the run supplies "
        "{got!r}\")"),
    # H16 the full-calendar refusal is dropped from the preflight
    "preflight_calendar_check_dropped": _sub(
        "            if t.size != expected or abs(t[0] - jan1) > one_second:",
        "            if False:"),
    # H17 the u/v coordinate identity check is dropped
    "preflight_uv_check_dropped": _sub(
        '        if not (np.array_equal(coords["u"][0], coords["v"][0])\n'
        '                and np.array_equal(coords["u"][1], coords["v"][1])):',
        "        if False:"),
    # H18 the cross-year drift check is dropped
    "preflight_drift_check_dropped": _sub(
        "        elif not (np.array_equal(lat, ref_lat) and np.array_equal(lon, ref_lon)):",
        "        elif False:"),
    # H19 the timestamp regularity check is dropped
    "preflight_regularity_check_dropped": _sub(
        "        if t.size < 2 or np.any(np.abs(diffs - 0.25) > one_second):",
        "        if False:"),
    # H20 the u/v TIME comparison is dropped, restoring the gap where only u was checked
    "preflight_uv_times_dropped": _sub(
        '        if (times["u"].shape != times["v"].shape\n'
        '                or not np.allclose(times["u"], times["v"], rtol=0.0, '
        "atol=one_second)):",
        "        if False:"),
    # H21 the frozen-coordinate comparison is dropped, so a shifted latitude at the
    # right shape passes into thirty years of computation
    "preflight_expected_vectors_dropped": _sub(
        "            if expected_lat is not None and not (\n"
        "                    np.array_equal(lat, expected_lat)\n"
        "                    and np.array_equal(lon, expected_lon)):",
        "            if False:"),
    # H22 the reserved namespace collapses into free labels, so P1-T6 runs unprotected
    "reserved_namespace_collapsed": _sub(
        '    return "reserved"',
        '    return "free"'),
    # H23 the wrong pressure level is accepted, running a named case at 850 silently
    "preflight_level_check_dropped": _sub(
        '                if value != float(expected_level_hpa):',
        '                if False:'),
    # H23 the missing-level refusal is removed, so expecting 700 hPa stops requiring
    # evidence of 700 hPa, which is the exact contradiction a review executed
    "missing_level_accepted": _sub(
        "                if value is None:\n"
        "                    raise ValueError(",
        "                if False:\n"
        "                    raise ValueError("),
    # H23 the u/v level agreement check is dropped
    "uv_level_agreement_dropped": _sub(
        '        if (levels["u"][0] is not None and levels["v"][0] is not None\n                and levels["u"][0] != levels["v"][0]):',
        '        if False:'),
    # H23 an unestablishable level is given the benefit of the doubt
    "unestablished_level_accepted": _sub(
        "            if reason is not None:\n"
        '                raise ValueError(f"{year} {name}: {reason}")',
        "            if reason is not None:\n                pass"),
}
