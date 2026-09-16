"""Mutations for the grid-route comparator's coverage check.

The list was written before the assertions existed and is reproduced in the docstring of
tests/test_compare_grid_routes.py. Run with the repository's mutation checker against
scripts/compare_grid_routes.py.

G1 AND G3 ARE THE TWO DEFECTS THAT ACTUALLY SHIPPED, both found by review rather than by
this file, and both of the same shape: a direct tree that agreed everywhere it overlapped
but did not cover the same ground was reported as agreeing. G1 was a missing row and column,
handled by a printed note that did not reach the verdict. G3 was a missing timestep, which
survived the repair for G1 because `shared != min(a, b)` is satisfied by a strict subset.

G4 is the mistake the obvious repair of G3 would introduce next, and G1 is kept as its own
entry because when it was first written the suite did not catch it. Every case above shrank
the DIRECT grid, so the surplus always sat on the strided side and G2's clause caught it
instead. That is why the test file carries a mirror case where the direct grid is larger.
"""


def _sub(old, new):
    return lambda s: s.replace(old, new)


MUTATIONS = {
    # G1 the direct grid's own coverage is never checked
    "direct_grid_coverage_unchecked": _sub(
        "    if shared_rows != rows_d or shared_cols != cols_d:",
        "    if False:"),
    # G2 the strided grid's own coverage is never checked
    "strided_grid_coverage_unchecked": _sub(
        "    if shared_rows != rows_s or shared_cols != cols_s:",
        "    if False:"),
    # G3 the shipped defect: a strict timestamp subset satisfies a comparison against min()
    "timestamps_compared_against_min": _sub(
        "    if n_times_s != n_times_d or n_shared_times != n_times_s:",
        "    if n_shared_times != min(n_times_s, n_times_d):"),
    # G4 the mirror image: checking one side alone passes extras on the other
    "timestamps_compared_on_one_side": _sub(
        "    if n_times_s != n_times_d or n_shared_times != n_times_s:",
        "    if n_shared_times != n_times_s:"),
    # G5 a coverage mismatch is printed but does not change the exit code
    "coverage_mismatch_is_not_fatal": _sub(
        '        for m in mismatch:\n            print(f"  - {m}")\n        return 1',
        '        for m in mismatch:\n            print(f"  - {m}")'),
    # the value comparison stops distinguishing agreement from disagreement
    "every_field_reported_as_agreeing": _sub(
        "    agrees = scale > 0 and normalized <= ROUNDING",
        "    agrees = True"),
    # the agreement tolerance is widened until a real difference passes
    "rounding_tolerance_is_not_rounding": _sub(
        "ROUNDING = 1e-6",
        "ROUNDING = 1.0"),
}
