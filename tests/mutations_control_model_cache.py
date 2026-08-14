"""Mutations for the eight newly-bound rules. Each removes exactly one behaviour."""


def _sub(old, new):
    return lambda s: s.replace(old, new)


MUTATIONS = {
    # R-CACHE-01: absent cache must return None (rebuild), not raise or reuse
    "absent_cache_raises": _sub(
        "    if not cache_path or not os.path.exists(cache_path):\n        return None",
        "    if not cache_path or not os.path.exists(cache_path):\n"
        "        raise SystemExit('no cache')"),
    # R-CACHE-02: unreadable cache must rebuild, not propagate
    "unreadable_cache_propagates": _sub(
        "    try:\n        df = read_design_csv(cache_path)\n    except Exception as e:",
        "    try:\n        df = read_design_csv(cache_path)\n    except ZeroDivisionError as e:"),
    # R-CACHE-12 / R-PRED-05: missing deposited troughs must refuse
    "missing_trough_check_deleted": _sub(
        '        if absent:\n'
        '            msgs.append(f"{absent} deposited trough(s) missing from the cache")\n',
        ''),
    # R-CASES-02: an unparseable cases anchor must refuse
    "cases_parse_error_escapes": _sub(
        '    try:\n        f = read_design_csv(path)\n    except Exception as e:\n'
        '        raise _Refuse(f"{name}: unreadable ({e.__class__.__name__})")',
        '    f = read_design_csv(path)\n    if False:\n        pass'),
    # R-CASES-08: both-sides NaN must be ACCEPTED and reported
    "both_sides_nan_refused": _sub(
        '            neither = int((a.isna() & b.isna()).sum())\n'
        '            if neither:\n'
        '                notes.append(f"{col}: {neither} absent on both sides, read as agreement")',
        '            neither = int((a.isna() & b.isna()).sum())\n'
        '            if neither:\n'
        '                msgs.append(f"{col}: {neither} absent on both sides")'),
    "both_sides_nan_not_reported": _sub(
        '                notes.append(f"{col}: {neither} absent on both sides, read as agreement")',
        '                pass'),
    # R-FRESH-04: a null digest for EITHER anchor must refuse
    "null_cases_digest_accepted": _sub(
        '        if not want:\n            raise _Refuse(f"freshness record has no digest for {name}")',
        '        if not want:\n'
        '            if name != "troughs_pooled.csv":\n                continue\n'
        '            raise _Refuse(f"freshness record has no digest for {name}")'),
    # R-TROUGH-03: every required troughs column must be checked
    "troughs_response_column_unchecked": _sub(
        '        if "response" not in tl.columns:\n'
        '            raise _Refuse("troughs_pooled.csv: missing column(s) response")\n',
        ''),
    "troughs_key_columns_unchecked": _sub(
        '    missing = [c for c in ("time", lon_col, wave_col) if c not in frame.columns]\n'
        '    if missing:\n'
        '        raise _Refuse(f"{what}: missing column(s) {\', \'.join(missing)}")',
        '    missing = []'),
    # R-PRED-01 / R-PRED-02: valid configurations of any size must be ACCEPTED
    "one_row_cohort_refused": _sub(
        "        msgs.extend(_check_freshness(cache, dep))",
        "        msgs.extend(_check_freshness(cache, dep))\n"
        "        if len(df) == 1:\n            raise _Refuse('one-row caches unsupported')"),
    "large_cohort_refused": _sub(
        "        msgs.extend(_check_freshness(cache, dep))",
        "        msgs.extend(_check_freshness(cache, dep))\n"
        "        if len(df) > 100:\n            raise _Refuse('large caches unsupported')"),
}


# ---------------------------------------------------------------------------
# PER-FIELD MUTATIONS, added 2026-08-13.
#
# rule_binding.py cannot see these: it reports a rule as CLAIMED as soon as one test
# references it, and screen 3 found several rules tested for one member of a set while
# the other members went unchecked. Each mutation below removes a behaviour for exactly
# ONE column, which is the shape that survived the suite before the tests were
# parametrised over the guard's own COMPARED and PREDICTORS constants.
# ---------------------------------------------------------------------------

def _skip_compared(col):
    """Drop one column from the per-column comparison loop entirely."""
    return _sub("        for col, dep_col in COMPARED:",
                f"        for col, dep_col in [p for p in COMPARED if p[0] != {col!r}]:")


def _skip_one_sided_for(col):
    """Keep the one-sided NaN check for every column except one."""
    return _sub(
        "            one_sided = int((a.isna() ^ b.isna()).sum())",
        f"            one_sided = 0 if col == {col!r} else int((a.isna() ^ b.isna()).sum())")


def _skip_inf_for(col):
    return _sub(
        "            inf = int((np.isinf(a.values) | np.isinf(b.values)).sum())",
        f"            inf = 0 if col == {col!r} else "
        "int((np.isinf(a.values) | np.isinf(b.values)).sum())")


def _skip_both_nan_report_for(col):
    return _sub(
        "            neither = int((a.isna() & b.isna()).sum())",
        f"            neither = 0 if col == {col!r} else int((a.isna() & b.isna()).sum())")


def _skip_dtype_for(col):
    """Weaken the numeric-dtype gate for exactly one predictor."""
    return _sub(
        "        for c in PREDICTORS:\n"
        "            _numeric(cl, c, \"cache\", require_numeric_dtype=True)",
        "        for c in PREDICTORS:\n"
        f"            _numeric(cl, c, \"cache\", require_numeric_dtype=(c != {col!r}))")


def _drop_required_column(col):
    """Omit one field from the required modelled-column list."""
    return _sub(
        '        missing = [c for c in ["response", *PREDICTORS, "year", "lonmonth"]\n'
        "                   if c not in df.columns]",
        '        missing = [c for c in ["response", *PREDICTORS, "year", "lonmonth"]\n'
        f"                   if c not in df.columns and c != {col!r}]")


def _drop_required_dep_column(col):
    return _sub(
        "        absent_cols = [d for _, d in COMPARED if d not in cs.columns]",
        f"        absent_cols = [d for _, d in COMPARED if d not in cs.columns "
        f"and d != {col!r}]")


_COMPARED_COLS = ["inflow_rh", "box_rh", "antecedent"]
_DEP_COLS = ["rh_m72", "box_rh_m24", "antecedent"]
_PREDICTORS = ["inflow_rh", "box_rh", "antecedent", "amplitude", "shear", "tcwv"]
_MODELLED = ["response"] + _PREDICTORS + ["year", "lonmonth"]

for _c in _COMPARED_COLS:
    MUTATIONS[f"skip_compared_column_{_c}"] = _skip_compared(_c)
    MUTATIONS[f"skip_one_sided_nan_{_c}"] = _skip_one_sided_for(_c)
    MUTATIONS[f"skip_infinity_{_c}"] = _skip_inf_for(_c)
    MUTATIONS[f"skip_both_nan_report_{_c}"] = _skip_both_nan_report_for(_c)
for _c in _PREDICTORS:
    MUTATIONS[f"skip_dtype_gate_{_c}"] = _skip_dtype_for(_c)
for _c in _MODELLED:
    MUTATIONS[f"drop_required_column_{_c}"] = _drop_required_column(_c)
for _c in _DEP_COLS:
    MUTATIONS[f"drop_required_dep_column_{_c}"] = _drop_required_dep_column(_c)
