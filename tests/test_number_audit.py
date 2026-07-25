"""Tests for the R6 prose-to-canonical audit (tools/number_audit.py)."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from number_audit import audit, lint_untagged, load_canonical, strip_tags  # noqa: E402


def _canon(tmp_path):
    df = pd.DataFrame([
        dict(tier="pooled", level=700.0, statistic="lagrangian_rh", time_rel_h=-72.0,
             diff=2.993982, ci_lo=2.032387, ci_hi=4.031674),
        dict(tier="pooled", level=None, statistic="n_selected", time_rel_h=None,
             diff=10303.0, ci_lo=None, ci_hi=None),
    ])
    p = tmp_path / "canon.csv"
    df.to_csv(p, index=False)
    rows, dups = load_canonical(p)
    assert dups == []
    return rows


def test_audit_passes_matching_and_flags_stale(tmp_path):
    canon = _canon(tmp_path)
    good = "reaches +2.99<!--n:pooled:700:lagrangian_rh:-72:diff:2--> percent"
    assert audit(good, canon) == []
    stale = "reaches +3.08<!--n:pooled:700:lagrangian_rh:-72:diff:2--> percent"
    errs = audit(stale, canon)
    assert len(errs) == 1 and "STALE" in errs[0]
    missing = "value +1.00<!--n:dev:850:no_such_stat:-72:diff:2-->"
    assert "no canonical row" in audit(missing, canon)[0]


def test_audit_interval_fields_and_unsigned(tmp_path):
    canon = _canon(tmp_path)
    t = ("(+2.03<!--n:pooled:700:lagrangian_rh:-72:ci_lo:2--> to "
         "+4.03<!--n:pooled:700:lagrangian_rh:-72:ci_hi:2-->)")
    assert audit(t, canon) == []
    t2 = "across 10,303<!--n:pooled::n_selected::diff:u0--> troughs"
    assert audit(t2, canon) == []


def test_strip_tags_removes_all():
    t = "x +2.99<!--n:pooled:700:lagrangian_rh:-72:diff:2--> y"
    assert strip_tags(t) == "x +2.99 y"


def test_lint_flags_untagged_but_allows_structure():
    text = ("In 2004 the 700 hPa field at 10 N (Fig. 3) over 24 h shows "
            "a contrast of +2.99 percent.")
    hits = lint_untagged(text)
    assert [h[1] for h in hits] == ["+2.99"]
    tagged = text.replace("+2.99", "+2.99<!--n:pooled:700:lagrangian_rh:-72:diff:2-->")
    assert lint_untagged(tagged) == []


def test_load_canonical_flags_duplicates(tmp_path):
    df = pd.DataFrame([
        dict(tier="pooled", level=700.0, statistic="x", time_rel_h=-72.0, diff=1.0),
        dict(tier="pooled", level=700.0, statistic="x", time_rel_h=-72.0, diff=2.0),
    ])
    p = tmp_path / "dup.csv"
    df.to_csv(p, index=False)
    rows, dups = load_canonical(p)
    assert len(dups) == 1 and "duplicate" in dups[0]


def test_replicate_count_claim_is_audited():
    """Claiming a replicate count the deposit did not use must fail the audit.

    Round 6 found the manuscript claiming 20,000 bootstrap replicates while the driver
    ran 2,000, and nothing caught it: the count sat in prose with no canonical row, and
    the lint allowlisted bare replicate counts. This pins both halves of that fix.
    """
    canon = {("config", "", "bootstrap_replicates", ""): {"diff": 20000.0}}
    good = "bootstraps of 20,000<!--n:config::bootstrap_replicates::diff:u0--> replicates"
    assert audit(good, canon) == []
    bad = good.replace("20,000", "2,000")
    errs = audit(bad, canon)
    assert errs and "bootstrap_replicates" in errs[0], errs


def test_bare_replicate_counts_are_not_allowlisted():
    """An untagged replicate or draw count must be flagged, not waved through."""
    assert lint_untagged("cluster bootstraps of 20,000 replicates.")
    assert lint_untagged("across 1,000 draws.")
    # the hyphenated adjectival form stays allowed
    assert not lint_untagged("a 20,000-replicate bootstrap")
