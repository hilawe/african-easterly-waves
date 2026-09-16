"""Mutations for the memory-bounded exact percentile.

The list was written before the assertions existed and is reproduced in the docstring of
tests/test_percentile.py. Run with the repository's mutation checker against
src/aew/v1port/percentile.py.

E4 IS THE ONE THIS ORACLE EXISTS FOR. The contract demands bit-equality with
np.percentile(..., method="linear"), and the implementation earns it by delegating the
final interpolation to np.percentile itself over the two order statistics. A plausible
reimplementation, a + (b - a) * t, is correct arithmetic that can differ in the last bit
from NumPy's _lerp, which switches formula at t = 0.5 for rounding symmetry. If the E4
mutation survives, the oracle is comparing with a tolerance somewhere and the "exact" claim
is not being held.

RUN THIS CATALOG WITH A PER-MUTATION TIMEOUT. The `progress_guard_disabled` entry turns a
refusal into an infinite loop, which is exactly what it is for, and the repository's
mutation checker has no timeout, so it will hang on that entry rather than report it. The
harness in the session logs wraps each pytest run in a 90-second subprocess timeout and
treats a hang as CAUGHT, which for this class it is.
"""


def _sub(old, new):
    return lambda s: s.replace(old, new)


MUTATIONS = {
    # E1 the population becomes ~isnan instead of isfinite, so infinities leak in
    "infinities_not_excluded": _sub(
        "    return chunk[np.isfinite(chunk)]",
        "    return chunk[~np.isnan(chunk)]"),
    # E2 the empty population returns NaN instead of refusing
    "empty_population_answered": _sub(
        """    if n == 0:
        raise ValueError(
            "no finite values in the population; a percentile of nothing is refused "
            "rather than answered")""",
        """    if n == 0:
        return float("nan"), 0"""),
    # E3 the rank is off by one
    "rank_off_by_one": _sub(
        "    rank_lo = int(np.floor(f))",
        "    rank_lo = int(np.floor(f)) + 1"),
    # E3 the rank base is n instead of n - 1
    "rank_uses_n_not_n_minus_one": _sub(
        '    f = (q / 100.0) * (n - 1)',
        '    f = (q / 100.0) * n'),
    # E4 the interpolation is reimplemented instead of delegated
    "interpolation_reimplemented": _sub(
        """    frac = f - rank_lo
    value = np.quantile(np.array([x_lo, x_hi]), frac, method="linear")
    return float(value), n""",
        """    frac = f - rank_lo
    return float(x_lo + (x_hi - x_lo) * frac), n"""),
    # E4 the SHIPPED defect: delegation through np.percentile with frac * 100, whose
    # internal division by 100 does not round-trip and moves the last bit. Found by
    # review; the regression carries searched counterexamples at q=55 and q=66.
    "interpolation_percent_round_trip": _sub(
        """    value = np.quantile(np.array([x_lo, x_hi]), frac, method="linear")""",
        """    value = np.percentile(np.array([x_lo, x_hi]), frac * 100.0,
                          method="linear")"""),
    # the q validation is dropped, so the streaming route answers out-of-range requests
    "q_validation_removed": _sub(
        """    if not np.isfinite(q) or q < 0.0 or q > 100.0:""",
        """    if False:"""),
    # the range validation is dropped, so a population spanning the float64 line gets nan
    # from the held route and the finite minimum from the streaming one, two different
    # wrong answers where the contract promises one refusal
    "range_validation_removed": _sub(
        """    if not np.isfinite(hi - lo):""",
        """    if False:"""),
    # E5 the pass-agreement check is dropped, so a drifting bin membership goes unnoticed
    "pass_disagreement_tolerated": _sub(
        """            if candidates.size != in_bin:
                raise AssertionError(""",
        """            if False:
                raise AssertionError("""),
    # E5 the collecting pass drops the clip, so the population maximum lands in a bin
    # the counting pass does not have, the two passes disagree, and only the consistency
    # guard stands between that and a silently wrong order statistic. (A first version of
    # this entry reordered the arithmetic instead, `(v-lo)*(bins/width)` against
    # `(v-lo)/width*bins`, and half a million random probes never made them disagree, so
    # it was a near-equivalent mutation binding nothing and was replaced.)
    "collect_pass_drops_the_clip": _sub(
        """        idx = np.clip(((values - lo) / width * bins).astype(np.int64), 0, bins - 1)
        for b in wanted_bins:""",
        """        idx = ((values - lo) / width * bins).astype(np.int64)
        for b in wanted_bins:"""),
    # the progress guard is disabled, so a non-narrowing refinement loops forever; the
    # test harness treats a hang as a failure, which is the only way to catch this class
    "progress_guard_disabled": _sub(
        "        state = (lo, hi, rank)",
        "        state = (lo, hi, rank, len(filters))"),
    # E6 refinement forgets to re-localise the rank inside the bin
    "refinement_keeps_global_rank": _sub(
        "        rank = rank - below\n        lo, hi = new_lo, new_hi",
        "        lo, hi = new_lo, new_hi"),
    # E6 the within-bin selection ignores the values below the bin
    "below_count_ignored": _sub(
        "            return float(candidates[rank - below])",
        "            return float(candidates[rank])"),
    # E7 the small path takes a different percentile method
    "small_path_method_differs": _sub(
        '        return float(np.percentile(pooled, q, method="linear")), n',
        '        return float(np.percentile(pooled, q, method="lower")), n'),
    # the count returned is the raw count, not the finite count
    "count_includes_nonfinite": _sub(
        "        n += int(values.size)",
        "        n += int(np.asarray(chunk).size)"),
}
