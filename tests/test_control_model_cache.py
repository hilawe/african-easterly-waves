"""Failure-injection tests for the control-model design-cache guard.

The guard decides whether a precomputed design matrix may be reused instead of rebuilt.
Its specification is docs/DESIGN_CACHE_SPEC.md, written before the fourth repair because
the unit had been repaired three times and each repair introduced the defect the next
pass found.

WHY THE PREVIOUS SUITE DID NOT BIND THE BEHAVIOUR, which is the finding that produced
this file. Its thirteen tests all mutated the cache while leaving the freshness record
STALE. A validator that checked only the record therefore rejected all thirteen while
performing no join, no cohort comparison and no value comparison, and the round-10
exactly that validator was written to prove the suite was vacuous. Every corruption
test below therefore RE-WRITES the freshness record after mutating, with
``resign_after_mutation``, so a record-only validator accepts the corrupted input and
the test fails. Each test also asserts the SPECIFIC rejection reason, because a test
asserting only that ``ok`` is false is satisfied by a validator that refuses everything.

The mutation list in the specification was written before these tests existed, so the
assertions could not be reasoned backwards from.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from control_model import (  # noqa: E402
    COMPARED,
    PREDICTORS,
    complete_case,
    derived_fixed_effects,
    plan_cache_reuse,
    read_design_csv,
    validate_design_cache,
    write_freshness_record,
)

# PER-FIELD COVERAGE IS DERIVED FROM THE CODE'S OWN CONSTANTS, not hand-listed.
# Screen 3 found several rules tested for one member of a set while the others went
# unchecked (one-sided NaN for antecedent only, the dtype gate for four fields of six,
# the infinity rule for one compared column of three). Parametrising from COMPARED and
# PREDICTORS means adding a column to the guard automatically adds its test cases, so
# the gap cannot reopen the way it opened.
CACHE_COMPARED = [c for c, _ in COMPARED]                  # inflow_rh, box_rh, antecedent
DEP_COMPARED = [d for _, d in COMPARED]                    # rh_m72, box_rh_m24, antecedent
DEP_OF = dict(COMPARED)
MODELLED = ["response", *PREDICTORS, "year", "lonmonth"]   # every column the guard requires

N_ELIGIBLE = 60      # every corridor trough that passed the complete-window rule
N_CASES = 40         # the classified subsample, a strict subset of the eligible cohort
N_INCOMPLETE = 10    # eligible troughs carrying a non-finite predictor


def _write_freshness(cache, tmp):
    write_freshness_record(str(cache), str(tmp))


def resign_after_mutation(cache, tmp):
    """Re-write the freshness record so a record-only validator would ACCEPT.

    This is the point of the file. Without it every test below passes against a
    validator that does nothing but compare digests.
    """
    _write_freshness(cache, tmp)


def build_mini_deposit(tmp_path, n_eligible=N_ELIGIBLE, n_cases=N_CASES,
                       n_incomplete=N_INCOMPLETE, seed=0):
    """Build a synthetic deposit, matching eligible-cohort cache, and freshness record.

    Returns ``(tmp_path, cache_path_str, design_df)``. The default arguments reproduce
    the standard fixture exactly (same seed and draw order, so existing tests keep their
    values); callers vary ``n_eligible`` to exercise the smallest and largest accepting
    configurations. The shape mirrors the real one measured on 2026-08-13: the cache
    holds the ELIGIBLE cohort and its key set equals troughs_pooled.csv exactly, the
    classified subsample is a strict subset of it, and some eligible troughs carry a
    non-finite predictor and so fall out of the complete-case cohort.
    """
    assert n_cases <= n_eligible and n_incomplete <= n_eligible
    rng = np.random.default_rng(seed)
    # Spanning several years and months, as the real record does (1983-2007, July to
    # September). A single-month fixture makes the year column constant, and a
    # permutation of a constant column is a no-op, so the fixed-effect check would have
    # looked tested while being unexercised.
    third = n_eligible // 3
    times = pd.DatetimeIndex(np.concatenate([
        pd.date_range("1990-07-01", periods=third, freq="6h").values,
        pd.date_range("1995-08-05", periods=third, freq="6h").values,
        pd.date_range("2003-09-11", periods=n_eligible - 2 * third, freq="6h").values]))
    lon = np.round(rng.uniform(-20, 30, n_eligible), 3)
    # Row 0 sits ON a longitude bin edge, as a real trough does (measured minimum
    # distance to an edge in the real design is 0.0). Without such a row the residual
    # test below cannot flip a bin and silently asserts nothing, which is exactly what
    # was found later: the fixture's nearest longitude was 0.137 away.
    lon[0] = -1e-16
    wave = np.arange(n_eligible) % 12 + 1
    response = rng.integers(0, 40, n_eligible)
    rh72 = np.round(rng.uniform(20, 80, n_eligible), 4)
    box24 = np.round(rng.uniform(20, 80, n_eligible), 4)
    antecedent = np.round(rng.uniform(0, 30, n_eligible), 4)

    pd.DataFrame(dict(traj_id=wave, time=times.astype(str), lat=10.0, lon=lon,
                      year=1990, month=7, response=response,
                      label="MCS-quiet")).to_csv(tmp_path / "troughs_pooled.csv",
                                                 index=False)
    # the classified subsample covers only the first n_cases eligible troughs
    pd.DataFrame(dict(traj_id=wave[:n_cases], time=times[:n_cases].astype(str),
                      lon=lon[:n_cases], rh_m72=rh72[:n_cases],
                      box_rh_m24=box24[:n_cases],
                      antecedent=antecedent[:n_cases])).to_csv(
        tmp_path / "cases_pooled_700.csv", index=False)

    shear = rng.uniform(0, 15, n_eligible)
    if n_incomplete:
        shear[-n_incomplete:] = np.nan      # these leave the complete-case cohort
    # The fixed effects come from the SAME function build_design uses, so the fixture is
    # one the real system can emit. An earlier version of this fixture hardcoded
    # lonmonth="a" and year=1990, which build_design can never produce, and the tests
    # passed against a frame that cannot exist.
    year, lonmonth = derived_fixed_effects(times, lon)
    design = pd.DataFrame(dict(time=times.astype(str), lon=lon, response=response,
                               inflow_rh=rh72, box_rh=box24, antecedent=antecedent,
                               amplitude=rng.uniform(0, 5, n_eligible),
                               shear=shear,
                               tcwv=rng.uniform(20, 60, n_eligible),
                               lonmonth=lonmonth, year=year, wave=wave))
    cache = tmp_path / "control_model_design_eligible.csv"
    design.to_csv(cache, index=False)
    _write_freshness(cache, tmp_path)
    return tmp_path, str(cache), design


@pytest.fixture()
def deposit(tmp_path):
    """The standard 60-row synthetic deposit used by most tests."""
    return build_mini_deposit(tmp_path)


def _reject(msgs, needle):
    assert any(needle in m for m in msgs), f"expected {needle!r} among {msgs}"


# ---------------------------------------------------------------- accepting rows
# The predicate table's accepting rows. A guard that refuses these is a guard that
# gets bypassed, which is worse than no guard at all.

def test_healthy_cache_is_accepted(deposit):
    tmp, cache, design = deposit
    ok, msgs = validate_design_cache(design, cache, str(tmp))
    assert ok, msgs


def test_partial_case_coverage_is_accepted(deposit):
    # binds: [R-CASES-05] [R-PRED-03]
    """Predicate row 3. Only 40 of 60 rows have a deposited predictor counterpart, which
    is the normal state (3,099 of 11,457 uncovered on the real record)."""
    tmp, cache, design = deposit
    ok, msgs = validate_design_cache(design, cache, str(tmp))
    assert ok, msgs


def test_non_finite_predictor_is_accepted(deposit):
    # binds: [R-CACHE-08]
    """The cohort is ELIGIBLE, not complete-case, so a non-finite predictor is expected.
    Refusing it would reject every healthy cache."""
    tmp, cache, design = deposit
    assert design["shear"].isna().sum() == N_INCOMPLETE
    ok, msgs = validate_design_cache(design, cache, str(tmp))
    assert ok, msgs
    assert len(complete_case(design)) == N_ELIGIBLE - N_INCOMPLETE


def test_extra_column_is_accepted(deposit):
    # binds: [R-PRED-04]
    """Predicate row 4. The guard governs the modelled columns, not the file's shape."""
    tmp, cache, design = deposit
    wide = design.copy()
    wide["scratch_note"] = "ignored"
    wide.to_csv(cache, index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(wide, cache, str(tmp))
    assert ok, msgs


# ---------------------------------------------------------------- cohort identity

def test_truncated_cache_is_rejected(deposit):
    # binds: [R-CACHE-11] [R-PRED-05]
    """Truncation is caught by the cohort correspondence, with no self-written row
    count. This is what the eligible-cohort redesign buys."""
    tmp, cache, design = deposit
    short = design.iloc[: N_ELIGIBLE - 10]
    short.to_csv(cache, index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(short, cache, str(tmp))
    assert not ok
    _reject(msgs, "10 deposited trough(s) missing from the cache")


def test_orphan_row_is_rejected(deposit):
    # binds: [R-CACHE-10] [R-PRED-06]
    tmp, cache, design = deposit
    bad = design.copy()
    bad.loc[0, "wave"] = 9999
    bad.to_csv(cache, index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(bad, cache, str(tmp))
    assert not ok
    _reject(msgs, "match no deposited trough")


def test_duplicate_key_is_rejected(deposit):
    # binds: [R-CACHE-05]
    """The previous version called drop_duplicates here, which collapsed conflicting
    rows before any comparison while the model fitted the duplicated frame."""
    tmp, cache, design = deposit
    bad = pd.concat([design, design.iloc[[0]]], ignore_index=True)
    bad.to_csv(cache, index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(bad, cache, str(tmp))
    assert not ok
    _reject(msgs, "duplicate key row(s)")


@pytest.mark.parametrize("anchor", ["troughs_pooled.csv", "cases_pooled_700.csv"])
def test_duplicate_key_in_anchor_is_rejected(deposit, anchor):
    # binds: [R-TROUGH-04] [R-CASES-04]
    """BOTH anchors, which is the gap a repo-access review used to defeat the first
    version of this suite. It covered troughs_pooled.csv only, so a validator that
    deduplicated cases_pooled_700.csv before the one-to-one check passed all 40 tests
    while omitting a rule the specification states."""
    tmp, cache, design = deposit
    t = pd.read_csv(tmp / anchor)
    pd.concat([t, t.iloc[[0]]], ignore_index=True).to_csv(tmp / anchor, index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(design, cache, str(tmp))
    assert not ok
    _reject(msgs, "duplicate key row(s)")


# ---------------------------------------------------------------- fixed effects

@pytest.mark.parametrize("col", ["year", "lonmonth"])
def test_permuted_fixed_effect_is_rejected(deposit, col):
    # binds: [R-FE-01]
    """These enter every fitted model and the within-between model downstream, and the
    first version of this guard checked neither. A repo-access review permuted them in a
    real 11,457-row cache and the pooled primary term moved from 1.013844 to 1.002920
    while the guard reported success. They are exact functions of the row's own key, so
    a disagreement is certain rather than merely suspicious."""
    tmp, cache, design = deposit
    bad = design.copy()
    bad[col] = np.random.default_rng(3).permutation(bad[col].values)
    bad.to_csv(cache, index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(bad, cache, str(tmp))
    assert not ok, f"a permuted {col} was accepted"
    _reject(msgs, f"{col}: ")


@pytest.mark.parametrize("col", ["year", "lonmonth"])
def test_missing_fixed_effect_is_rejected(deposit, col):
    # binds: [R-CACHE-03]
    tmp, cache, design = deposit
    bad = design.drop(columns=[col])
    bad.to_csv(cache, index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(bad, cache, str(tmp))
    assert not ok
    _reject(msgs, f"missing modelled column(s) {col}")


# ---------------------------------------------------------------- values

@pytest.mark.parametrize("col", ["inflow_rh", "box_rh", "antecedent", "response"])
def test_permuted_checkable_column_is_rejected(deposit, col):
    # binds: [R-CASES-06] [R-TROUGH-06]
    """Round 8 checked three of seven columns and shipped. These four have a deposited
    counterpart, so a permutation must be caught even with the record re-signed."""
    tmp, cache, design = deposit
    bad = design.copy()
    bad[col] = np.random.default_rng(1).permutation(bad[col].values)
    bad.to_csv(cache, index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(bad, cache, str(tmp))
    assert not ok, f"a permuted {col} was accepted"
    _reject(msgs, f"{col}: ")


@pytest.mark.parametrize("col", ["amplitude", "shear", "tcwv"])
def test_uncheckable_column_is_accepted_and_declared(deposit, col, capsys):
    # binds: [R-COL-01]
    """THE TRUST BOUNDARY, made executable rather than left in a docstring.

    Nothing in the deposit carries these three, so a wholly wrong column passes. That is
    a real hole and the specification says so. This test exists so the limitation cannot
    be quietly widened or quietly forgotten: if a later change starts claiming these are
    checked, the message assertion fails, and if one starts actually checking them, the
    acceptance assertion fails and the specification gets updated deliberately.
    """
    tmp, cache, design = deposit
    bad = design.copy()
    bad[col] = np.random.default_rng(2).permutation(bad[col].values)
    bad.to_csv(cache, index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(bad, cache, str(tmp))
    assert ok, msgs
    out = capsys.readouterr().out
    assert "NOT CHECKED" in out
    assert col in out.split("NOT CHECKED", 1)[1]


@pytest.mark.parametrize("col", CACHE_COMPARED)
def test_one_sided_nan_is_rejected(deposit, col):
    # binds: [R-CASES-07]
    """The previous version's dropna() turned an injected NaN into a row that was
    silently not compared. EVERY compared column: this was tested for antecedent alone,
    and a guard skipping inflow_rh or box_rh passed the whole suite."""
    tmp, cache, design = deposit
    bad = design.copy()
    bad.loc[0, col] = np.nan                   # row 0 is inside the cases subsample
    bad.to_csv(cache, index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(bad, cache, str(tmp))
    assert not ok, f"a one-sided NaN in {col} was accepted"
    _reject(msgs, f"{col}: 1 row(s) present on one side only")


def test_non_finite_response_is_rejected(deposit):
    # binds: [R-CACHE-09]
    tmp, cache, design = deposit
    bad = design.copy()
    bad.loc[0, "response"] = np.nan
    bad.to_csv(cache, index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(bad, cache, str(tmp))
    assert not ok
    _reject(msgs, "non-finite response")


def test_sub_tolerance_edit_is_accepted(deposit):
    # binds: [R-CASES-06]
    """The accepting counterpart of the comparison, so the tolerance is a stated
    property rather than an accident. The deposit is written at four decimal places, so
    a difference below 1e-3 cannot be distinguished from formatting."""
    tmp, cache, design = deposit
    near = design.copy()
    near["inflow_rh"] = near["inflow_rh"] + 5e-4
    near.to_csv(cache, index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(near, cache, str(tmp))
    assert ok, msgs


# ---------------------------------------------------------------- freshness

def test_changed_deposit_is_rejected(deposit):
    # binds: [R-TROUGH-05]
    """The round-7 failure mode, deposit rebuilt and cache left behind, caught by the
    digest because the record is NOT re-signed here."""
    tmp, cache, design = deposit
    t = pd.read_csv(tmp / "troughs_pooled.csv")
    t.loc[0, "response"] = t.loc[0, "response"] + 7
    t.to_csv(tmp / "troughs_pooled.csv", index=False)
    ok, msgs = validate_design_cache(design, cache, str(tmp))
    assert not ok
    _reject(msgs, "troughs_pooled.csv changed since the cache was built")


@pytest.mark.parametrize("anchor", ["troughs_pooled.csv", "cases_pooled_700.csv"])
def test_changed_anchor_bytes_are_rejected(deposit, anchor):
    # binds: [R-FRESH-05]
    """BOTH digests, which is the second unbound specification rule an audit
    found. The suite checked freshness only for troughs_pooled.csv, so a validator that
    ignored the cases-anchor digest passed all 55 tests. The mutation is a column the
    guard never reads, so ONLY the digest can catch it."""
    tmp, cache, design = deposit
    c = pd.read_csv(tmp / anchor)
    c["a_column_the_guard_never_reads"] = 1
    c.to_csv(tmp / anchor, index=False)
    ok, msgs = validate_design_cache(design, cache, str(tmp))
    assert not ok
    _reject(msgs, f"{anchor} changed since the cache was built")


def test_changed_deposit_is_caught_even_when_resigned(deposit):
    # binds: [R-TROUGH-06]
    """The same change with the record re-signed, so the digest check cannot see it.
    The value comparison must catch it independently. This is the test that fails
    against a validator which only compares digests."""
    tmp, cache, design = deposit
    t = pd.read_csv(tmp / "troughs_pooled.csv")
    t.loc[0, "response"] = t.loc[0, "response"] + 7
    t.to_csv(tmp / "troughs_pooled.csv", index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(design, cache, str(tmp))
    assert not ok
    _reject(msgs, "response: 1 of")


def test_missing_freshness_record_is_rejected(deposit):
    # binds: [R-FRESH-01] [R-PRED-07]
    tmp, cache, design = deposit
    os.remove(cache + ".sources.json")
    ok, msgs = validate_design_cache(design, cache, str(tmp))
    assert not ok
    _reject(msgs, "no freshness record")


@pytest.mark.parametrize("body,needle", [
    ("{not json at all", "unreadable"),
    ("[]", "not a JSON object"),
    ("null", "not a JSON object"),
    ('{"sources": {}}', "carries no sources"),
    ('{"sources": {"troughs_pooled.csv": null, "cases_pooled_700.csv": null}}',
     "no digest for"),
    ('{"columns": ["time"]}', "carries no sources"),
])
def test_malformed_freshness_record_is_rejected(deposit, body, needle):
    # binds: [R-FRESH-02] [R-FRESH-03]
    """All six RAISED out of the previous guard or were accepted. A malformed record is
    a refusal, never an exception and never a pass."""
    tmp, cache, design = deposit
    with open(cache + ".sources.json", "w") as fh:
        fh.write(body)
    ok, msgs = validate_design_cache(design, cache, str(tmp))
    assert not ok
    _reject(msgs, needle)


@pytest.mark.parametrize("name", ["troughs_pooled.csv", "cases_pooled_700.csv"])
def test_absent_anchor_is_rejected(deposit, name):
    # binds: [R-TROUGH-01] [R-CASES-01] [R-PRED-08]
    """An absent anchor recorded a null digest and validated with zero predictors
    compared in the previous version, which is the fail-open shape the global rule
    forbids."""
    tmp, cache, design = deposit
    os.remove(tmp / name)
    ok, msgs = validate_design_cache(design, cache, str(tmp))
    assert not ok
    _reject(msgs, "absent")


@pytest.mark.parametrize("name", ["troughs_pooled.csv", "cases_pooled_700.csv"])
def test_empty_anchor_is_rejected(deposit, name):
    tmp, cache, design = deposit
    pd.read_csv(tmp / name).iloc[:0].to_csv(tmp / name, index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(design, cache, str(tmp))
    assert not ok
    _reject(msgs, "no rows")


def test_cases_sharing_no_row_is_rejected(deposit):
    # binds: [R-PRED-09]
    """Predicate row 9. A comparison covering nothing is not a comparison, so it must
    not read as a pass."""
    tmp, cache, design = deposit
    c = pd.read_csv(tmp / "cases_pooled_700.csv")
    c["traj_id"] = c["traj_id"] + 100000
    c.to_csv(tmp / "cases_pooled_700.csv", index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(design, cache, str(tmp))
    assert not ok
    _reject(msgs, "shares no row with the cache")


# ---------------------------------------------------------------- malformed inputs

@pytest.mark.parametrize("col", MODELLED)
def test_missing_modelled_column_is_rejected(deposit, col):
    # binds: [R-CACHE-03]
    """Every column the guard requires, not the one the original test happened to drop.
    A required-column list omitting any single field passed the suite when only tcwv,
    year and lonmonth were exercised."""
    tmp, cache, design = deposit
    bad = design.drop(columns=[col])
    bad.to_csv(cache, index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(bad, cache, str(tmp))
    assert not ok, f"a cache without {col} was accepted"
    assert any(col in m for m in msgs), (col, msgs)


@pytest.mark.parametrize("col", DEP_COMPARED)
def test_missing_deposited_column_is_rejected(deposit, col):
    # binds: [R-CASES-03]
    """All three deposited counterparts, not rh_m72 alone."""
    tmp, cache, design = deposit
    c = pd.read_csv(tmp / "cases_pooled_700.csv").drop(columns=[col])
    c.to_csv(tmp / "cases_pooled_700.csv", index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(design, cache, str(tmp))
    assert not ok, f"a cases table without {col} was accepted"
    _reject(msgs, f"missing column(s) {col}")


def test_unparseable_time_is_rejected(deposit):
    # binds: [R-CACHE-02]
    tmp, cache, design = deposit
    bad = design.copy()
    bad.loc[0, "time"] = "not a timestamp"
    bad.to_csv(cache, index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(bad, cache, str(tmp))
    assert not ok
    _reject(msgs, "unparseable time")


def test_ok_is_true_only_when_msgs_is_empty(deposit):
    # binds: [R-OUT-01]
    """Mutation 5 of the specification's list. The two halves of the return value are
    not allowed to disagree, on either a healthy or a corrupted input."""
    tmp, cache, design = deposit
    ok, msgs = validate_design_cache(design, cache, str(tmp))
    assert ok is (not msgs)
    bad = design.copy()
    bad.loc[0, "inflow_rh"] = bad.loc[0, "inflow_rh"] + 5.0
    bad.to_csv(cache, index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(bad, cache, str(tmp))
    assert ok is (not msgs)
    assert not ok and msgs


def test_unexpected_error_propagates(deposit, monkeypatch):
    # binds: [R-OUT-03]
    """Mutation 10 of the specification's list, and the one that survived the first
    mutation run with all 37 tests green.

    A validator whose except clause catches Exception rather than _Refuse passes every
    negative test in this file, because a genuine programming error then becomes an
    ordinary refusal and the pipeline merely rebuilds. The guard could be broken for
    good while looking conservative. Only a _Refuse and a one-to-one MergeError are
    refusals. Anything else is a defect and must surface.
    """
    tmp, cache, design = deposit
    import control_model

    def boom(*a, **k):
        raise RuntimeError("a genuine programming error, not a refusal")

    monkeypatch.setattr(control_model, "_read_anchor", boom)
    with pytest.raises(RuntimeError):
        validate_design_cache(design, cache, str(tmp))


def test_unreadable_anchor_is_a_refusal_not_a_crash(deposit, monkeypatch):
    # binds: [R-TROUGH-02]
    """The counterpart. An anchor that exists but cannot be read is a refusal, which is
    the spec's row, so it must NOT propagate."""
    tmp, cache, design = deposit
    import control_model

    def denied(path):
        raise PermissionError(path)

    monkeypatch.setattr(control_model, "_file_digest", denied)
    ok, msgs = validate_design_cache(design, cache, str(tmp))
    assert not ok
    _reject(msgs, "unreadable (PermissionError)")


def test_response_is_compared_exactly(deposit):
    # binds: [R-TROUGH-06]
    """The response is a count deposited at three decimal places, so it round-trips
    exactly and needs no tolerance. Under the 1e-3 the environmental fields need, a
    uniform shift of 0.0005 on every response was accepted and then reported as
    identical."""
    tmp, cache, design = deposit
    bad = design.copy()
    bad["response"] = bad["response"] + 5e-4
    bad.to_csv(cache, index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(bad, cache, str(tmp))
    assert not ok
    _reject(msgs, "response: ")


def test_success_message_matches_what_the_code_does(deposit, capsys):
    # binds: [R-OUT-02]
    """Subcheck 3(a) of the pre-commit playbook applied to an assertion message. The
    message may not claim exact equality where the code allows a tolerance."""
    tmp, cache, design = deposit
    ok, msgs = validate_design_cache(design, cache, str(tmp))
    assert ok, msgs
    out = capsys.readouterr().out
    assert "response EXACTLY equal" in out
    assert "agree within 0.001" in out


@pytest.mark.parametrize("where,col",
                         [("cache", c) for c in ["response", *PREDICTORS, "year"]]
                         + [("anchor", "response")])
def test_non_numeric_value_is_rejected(deposit, where, col):
    # binds: [R-CACHE-06]
    """A plain to_numeric(errors="coerce") turns text into NaN, and NaN then reads as a
    false difference or, when both sides carry text, as agreement. A repo-access review
    reached acceptance four ways through that hole, including a non-numeric deposited
    response reported as identical and non-numeric predictors accepted and then raising
    a TypeError downstream.

    EVERY modelled numeric field. Screen 3 showed the gate could be dropped for a single
    column, box_rh, and still pass the whole suite, because only four of the six
    predictors were exercised.
    """
    tmp, cache, design = deposit
    if where == "cache":
        bad = design.copy()
        bad[col] = bad[col].astype(object)
        bad.loc[0, col] = "not a number"
        bad.to_csv(cache, index=False)
        resign_after_mutation(cache, tmp)
        ok, msgs = validate_design_cache(bad, cache, str(tmp))
    else:
        t = pd.read_csv(tmp / "troughs_pooled.csv")
        t[col] = t[col].astype(object)
        t.loc[0, col] = "not a number"
        t.to_csv(tmp / "troughs_pooled.csv", index=False)
        resign_after_mutation(cache, tmp)
        ok, msgs = validate_design_cache(design, cache, str(tmp))
    assert not ok
    # either refusal is correct: the dtype gate fires first for a whole object column,
    # the value gate for a column that is numeric but carries one unconvertible entry
    assert any(f"non-numeric value(s) in {col}" in m
               or f"{col} is object, not a numeric column" in m for m in msgs), msgs


def test_matching_text_on_both_sides_is_rejected(deposit):
    # binds: [R-CACHE-06]
    """The nastiest of the four: identical text on both sides became NaN on both sides
    and was read as agreement, so the cache was accepted and complete_case() then raised
    a TypeError on a frame the guard had just blessed."""
    tmp, cache, design = deposit
    bad = design.copy()
    bad["antecedent"] = bad["antecedent"].astype(object)
    bad.loc[0, "antecedent"] = "n/a"
    bad.to_csv(cache, index=False)
    c = pd.read_csv(tmp / "cases_pooled_700.csv")
    c["antecedent"] = c["antecedent"].astype(object)
    c.loc[0, "antecedent"] = "n/a"
    c.to_csv(tmp / "cases_pooled_700.csv", index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(bad, cache, str(tmp))
    assert not ok
    assert any("antecedent" in m and "numeric" in m for m in msgs), msgs


def test_string_wave_key_is_rejected(deposit):
    # binds: [R-CACHE-06]
    """This raised an uncaught pandas ValueError out of the merge."""
    tmp, cache, design = deposit
    bad = design.copy()
    bad["wave"] = bad["wave"].astype(str) + "w"
    bad.to_csv(cache, index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(bad, cache, str(tmp))
    assert not ok
    _reject(msgs, "non-numeric value(s) in wave")


def test_null_wave_is_rejected(deposit):
    """_keyed checked time and longitude for nulls but not the wave, so a null wave on
    both sides was accepted."""
    tmp, cache, design = deposit
    bad = design.copy()
    bad.loc[0, "wave"] = np.nan
    bad.to_csv(cache, index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(bad, cache, str(tmp))
    assert not ok
    _reject(msgs, "missing wave value(s)")


def test_timezone_aware_cache_is_accepted(deposit):
    """A timezone-aware cache against a naive anchor raised an uncaught ValueError. These
    are UTC model times throughout, so the two normalize rather than refuse. This is an
    ACCEPTING row: refusing it would be a false positive."""
    tmp, cache, design = deposit
    tz = design.copy()
    tz["time"] = pd.to_datetime(tz["time"]).dt.tz_localize("UTC").astype(str)
    tz.to_csv(cache, index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(tz, cache, str(tmp))
    assert ok, msgs


@pytest.mark.parametrize("col", ["response", "wave"])
def test_value_beyond_exact_integer_range_is_rejected(deposit, col):
    # binds: [R-CACHE-07]
    """Adjacent integers above 2**53 collapse to the same float, so two different
    responses compared equal and the guard printed "EXACTLY equal". Real values here are
    system counts and trajectory identifiers, so the bound costs nothing."""
    tmp, cache, design = deposit
    bad = design.copy()
    bad[col] = bad[col].astype(float)
    bad.loc[0, col] = float(2 ** 53) + 2
    bad.to_csv(cache, index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(bad, cache, str(tmp))
    assert not ok
    _reject(msgs, "too large for exact comparison")


@pytest.mark.parametrize("col", CACHE_COMPARED)
def test_matching_infinities_are_rejected(deposit, col):
    # binds: [R-CASES-09]
    """inf minus inf is NaN, which is not greater than the tolerance, so two matching
    infinities passed and the guard then claimed they agreed within 1e-3. EVERY compared
    column: this covered antecedent alone, and omitting inflow_rh still passed."""
    tmp, cache, design = deposit
    bad = design.copy()
    bad.loc[0, col] = np.inf
    bad.to_csv(cache, index=False)
    c = pd.read_csv(tmp / "cases_pooled_700.csv")
    c.loc[0, DEP_OF[col]] = np.inf
    c.to_csv(tmp / "cases_pooled_700.csv", index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(bad, cache, str(tmp))
    assert not ok, f"matching infinities in {col} were accepted"
    _reject(msgs, f"{col}: 1 infinite value(s)")


def test_duplicate_column_label_is_rejected(deposit):
    # binds: [R-CACHE-04]
    """A duplicated label raised an uncaught TypeError instead of refusing."""
    tmp, cache, design = deposit
    bad = pd.concat([design, design[["shear"]]], axis=1)
    assert bad.columns.duplicated().any()
    ok, msgs = validate_design_cache(bad, cache, str(tmp))
    assert not ok
    _reject(msgs, "duplicate column label(s) shear")


def test_string_typed_year_is_rejected(deposit):
    # binds: [R-CACHE-06]
    """It compared numerically equal while staying a string, and the integer-year tier
    filter downstream then selected zero rows."""
    tmp, cache, design = deposit
    bad = design.copy()
    bad["year"] = bad["year"].astype(str)
    bad.to_csv(cache, index=False)
    resign_after_mutation(cache, tmp)
    # a CSV round-trip parses it back to int, so the defect needs the frame directly
    ok, msgs = validate_design_cache(bad, cache, str(tmp))
    assert not ok
    _reject(msgs, "year is str, not a numeric column")


def test_longitude_below_deposited_precision_is_an_ACCEPTED_residual(deposit):
    # binds: [R-RESID-01]
    """A RESIDUAL, recorded as a passing test so it cannot be silently lost.

    Row 0 sits on a longitude bin edge. Shifting it by 4e-4 preserves the rounded
    comparison key and DOES move it across the edge, so lonmonth legitimately changes and
    the cache stays internally self-consistent. The guard accepts, and no comparison
    against this deposit can refuse it, because troughs_pooled.csv stores longitude at
    three decimals and the information is gone before the guard can read it. The real
    record has this row: 2006-09-09 06:00, wave 11467, longitude -1.11e-16.

    THE ASSERTIONS BELOW ARE WHAT KILL THE REJECTED CANDIDATE FIX. Deriving lonmonth from
    the DEPOSITED longitude would refuse this case, and on the real deposit it refuses 1
    of 9,375 genuine rows, so it is a gate that fires falsely. An earlier version of this
    test used a fixture whose nearest longitude was 0.137 from an edge, so the shift
    changed nothing and the test asserted nothing.
    """
    tmp, cache, design = deposit
    from control_model import LON_EDGES

    assert np.min(np.abs(design.loc[0, "lon"] - LON_EDGES)) < 1e-9, "row 0 must be on an edge"
    below = design.copy()
    below.loc[0, "lon"] = design.loc[0, "lon"] + 4e-4
    y, lm = derived_fixed_effects(pd.to_datetime(below["time"]), below["lon"].values)
    below["year"], below["lonmonth"] = y, lm
    assert lm[0] != design.loc[0, "lonmonth"], "the shift must actually cross a bin edge"
    assert round(float(below.loc[0, "lon"]), 3) == round(float(design.loc[0, "lon"]), 3)
    below.to_csv(cache, index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(below, cache, str(tmp))
    assert ok, f"the recorded residual changed, update the specification: {msgs}"

    # a shift large enough to move the rounded key IS caught, by the cohort check
    above = design.copy()
    above["lon"] = above["lon"] + 0.01
    y, lm = derived_fixed_effects(pd.to_datetime(above["time"]), above["lon"].values)
    above["year"], above["lonmonth"] = y, lm
    above.to_csv(cache, index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(above, cache, str(tmp))
    assert not ok
    _reject(msgs, "match no deposited trough")


def test_orphan_cases_row_is_rejected(deposit):
    # binds: [R-CASES-10]
    """The classified subsample is a strict subset of the eligible cohort, so a cases row
    matching no cached row means the tables describe different records. The inner join
    silently dropped it, so 40 valid rows plus 1 orphan returned success."""
    tmp, cache, design = deposit
    c = pd.read_csv(tmp / "cases_pooled_700.csv")
    extra = c.iloc[[0]].copy()
    extra["traj_id"] = 987654
    pd.concat([c, extra], ignore_index=True).to_csv(
        tmp / "cases_pooled_700.csv", index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(design, cache, str(tmp))
    assert not ok
    _reject(msgs, "match no cached trough")


def test_unexpected_value_error_propagates(deposit, monkeypatch):
    # binds: [R-OUT-03]
    """The contract says only a refusal and a merge-key failure are refusals. An earlier
    version caught ValueError across the whole validator, so a genuine defect anywhere
    inside became a routine refusal. The existing propagation test used RuntimeError and
    did not bind the broader claim."""
    tmp, cache, design = deposit
    import control_model

    def boom(*a, **k):
        raise ValueError("a genuine programming error, not a refusal")

    monkeypatch.setattr(control_model, "_read_anchor", boom)
    with pytest.raises(ValueError):
        validate_design_cache(design, cache, str(tmp))


def test_cache_read_round_trips_exactly(tmp_path):
    # binds: [R-READ-01]
    """A load-bearing property found by measurement, not by design.

    The published control_model_design.csv is written from the cache on the cached path
    and from the fresh build otherwise. If reading the cache is not an exact inverse of
    writing it, those two paths produce numerically different deposited tables and the
    file silently depends on whether a cache happened to exist. The guard's own
    comparisons run at 1e-3 and are indifferent to a last-bit difference, which is why
    this needs its own test rather than riding on the guard.

    The assertion is on the property that is needed (an exact round-trip), not on any
    particular parser being wrong, so it keeps passing if pandas changes its default.
    """
    tricky = [11.323954631638083, 1 / 3, 1e-5 / 3, 52.41693084989534]
    p = tmp_path / "cache.csv"
    pd.DataFrame({"shear": tricky, "lon": tricky}).to_csv(p, index=False)
    back = read_design_csv(str(p))
    assert back.to_csv(index=False) == p.read_text()
    assert [v.hex() for v in back["shear"]] == [v.hex() for v in tricky]


# ------------------------------------------- the eight rules the binding check found
# Written 2026-08-13 after tagging the specification with rule identifiers and checking
# which of them a test actually references. That reported 49 rules declared, 41 claimed
# and 8 unbound. Each test below closes one of those eight. They are the rules NO test
# mentioned. The per-field gaps inside rules that WERE mentioned are the mutation
# catalogue's job, and closing these does not make the suite sound on its own.

def test_absent_cache_rebuilds_without_error(tmp_path):
    # binds: [R-CACHE-01]
    """The ordinary first run. No cache on disk means rebuild, quietly: not an error,
    not an exception, and above all not a reuse. This is a main() branch no test reached,
    which is why the decision was extracted into plan_cache_reuse."""
    build_mini_deposit(tmp_path)
    missing = str(tmp_path / "no_such_cache.csv")
    assert not os.path.exists(missing)
    assert plan_cache_reuse(missing, str(tmp_path)) is None
    # and with no --cache given at all
    assert plan_cache_reuse(None, str(tmp_path)) is None


def test_unreadable_cache_rebuilds_without_raising(tmp_path):
    # binds: [R-CACHE-02]
    """A cache that cannot be parsed is a rebuild, never a crash and never a reuse."""
    _, cache, _ = build_mini_deposit(tmp_path)
    with open(cache, "w") as fh:
        fh.write('a,b\n"unterminated,1\n\x00\x00binary\n')
    assert plan_cache_reuse(cache, str(tmp_path)) is None


def test_header_only_cache_is_rejected(deposit):
    # binds: [R-CACHE-12]
    """An empty cache has no rows, so every deposited trough is missing from it. It must
    refuse rather than accept a comparison over nothing."""
    tmp, cache, design = deposit
    empty = design.iloc[:0]
    empty.to_csv(cache, index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(empty, cache, str(tmp))
    assert not ok
    _reject(msgs, "deposited trough(s) missing from the cache")
    # and the whole-pipeline decision agrees
    assert plan_cache_reuse(cache, str(tmp)) is None


def test_unparseable_cases_anchor_is_rejected(deposit):
    # binds: [R-CASES-02]
    """An anchor that exists but cannot be parsed is a refusal. The freshness digest
    alone would not catch this, because the record is re-signed over the broken file."""
    tmp, cache, design = deposit
    with open(tmp / "cases_pooled_700.csv", "w") as fh:
        fh.write('traj_id,time,lon\n"unterminated,1,2\n')
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(design, cache, str(tmp))
    assert not ok
    assert any("cases_pooled_700.csv" in m for m in msgs), msgs


@pytest.mark.parametrize("col", CACHE_COMPARED)
def test_nan_on_both_sides_is_accepted_and_reported(deposit, capsys, col):
    # binds: [R-CASES-08]
    """An ACCEPTING row, and the one most likely to be broken by a future tightening.
    A predictor absent on BOTH sides is agreement, not a mismatch: refusing it would be
    a false positive against a generation where a deposited predictor is legitimately
    absent. The count must be REPORTED, not silently absorbed."""
    tmp, cache, design = deposit
    bad = design.copy()
    bad.loc[0, col] = np.nan
    bad.to_csv(cache, index=False)
    c = pd.read_csv(tmp / "cases_pooled_700.csv")
    c.loc[0, DEP_OF[col]] = np.nan
    c.to_csv(tmp / "cases_pooled_700.csv", index=False)
    resign_after_mutation(cache, tmp)
    ok, msgs = validate_design_cache(bad, cache, str(tmp))
    assert ok, msgs
    out = capsys.readouterr().out
    assert f"{col}: 1 absent on both sides" in out


def test_null_digest_for_either_anchor_is_rejected(deposit):
    # binds: [R-FRESH-04]
    """Parametrised over BOTH anchors in one test because the loop refuses the troughs
    digest first: a record with both digests null exercises only the troughs branch, and
    a validator that skipped the cases digest entirely passed the whole suite."""
    tmp, cache, design = deposit
    good = json.load(open(cache + ".sources.json"))
    for anchor in ("troughs_pooled.csv", "cases_pooled_700.csv"):
        rec = {"sources": dict(good["sources"])}
        rec["sources"][anchor] = None
        with open(cache + ".sources.json", "w") as fh:
            json.dump(rec, fh)
        ok, msgs = validate_design_cache(design, cache, str(tmp))
        assert not ok, f"a null digest for {anchor} was accepted"
        _reject(msgs, f"no digest for {anchor}")


def test_missing_troughs_column_is_rejected(deposit):
    # binds: [R-TROUGH-03]
    """Every key column and the response, not just the ones a happy path happens to
    touch. Dropping any of them left the cohort check unable to run."""
    tmp, cache, design = deposit
    for col in ("traj_id", "time", "lon", "response"):
        t = pd.read_csv(tmp / "troughs_pooled.csv")
        t.drop(columns=[col]).to_csv(tmp / "troughs_pooled.csv", index=False)
        resign_after_mutation(cache, tmp)
        ok, msgs = validate_design_cache(design, cache, str(tmp))
        assert not ok, f"troughs_pooled.csv without {col} was accepted"
        assert any(col in m for m in msgs), (col, msgs)
        # restore for the next iteration
        build_mini_deposit(tmp)
        resign_after_mutation(cache, tmp)


@pytest.mark.parametrize("n_eligible,n_cases,n_incomplete",
                         [(1, 1, 0), (400, 260, 60)])
def test_smallest_and_largest_valid_configurations_are_accepted(
        tmp_path, n_eligible, n_cases, n_incomplete):
    # binds: [R-PRED-01] [R-PRED-02]
    """The predicate table's first two rows, the accepting ones an author skips.

    A guard that refuses the smallest configuration, or that only works at the size the
    fixture happens to use, is a guard that gets bypassed. The real cohort is 11,457
    rows; 400 is the largest size that stays fast in a unit test, so this binds the
    scale-independence of the checks rather than the exact production size, which is
    covered by the manual real-deposit verification recorded in the specification.
    """
    tmp, cache, design = build_mini_deposit(
        tmp_path, n_eligible=n_eligible, n_cases=n_cases, n_incomplete=n_incomplete)
    assert len(design) == n_eligible
    ok, msgs = validate_design_cache(design, cache, str(tmp))
    assert ok, msgs
    assert plan_cache_reuse(cache, str(tmp)) is not None


def test_freshness_record_carries_only_sources(deposit):
    """The record's narrowness is the design. A row count or a digest of the cache
    would be self-authored and is exactly what round 10 defeated."""
    tmp, cache, design = deposit
    rec = json.load(open(cache + ".sources.json"))
    assert set(rec) == {"sources"}
    assert set(rec["sources"]) == {"troughs_pooled.csv", "cases_pooled_700.csv"}
    assert all(v and len(v) == 64 for v in rec["sources"].values())
