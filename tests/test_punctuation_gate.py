"""The punctuation gate in tools/check_manuscript_figures.py must actually fire.

WHY THIS EXISTS. The global style rules forbid semicolons and "label: explanation" colons
in body prose and dashes anywhere. That was an author-side habit until 2026-08-13, when
the AMS word cut took the body from 1 semicolon and 1 colon lead-in to 6 and 26. The
mechanism is the point: joining two clauses with ":" or ";" is the cheapest way to save
words, so compressing against a hard word limit actively selects for the forbidden
construction, and the habit is what gets dropped under pressure. A gate does not get
dropped.

The gate is only worth having if a planted violation trips it, so each rule below is
exercised against text that violates it and against text that does not. The accepting
cases matter as much as the refusing ones: a gate that flags a figure caption or a
reference list fires falsely, gets bypassed, and is then worse than no gate at all.
"""

import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))


from check_manuscript_figures import punctuation_hits  # noqa: E402


def scan(text):
    """Rule names only, from the SHIPPED predicate.

    Importing rather than reimplementing is deliberate. An earlier version of this file
    carried its own copy of the rule, and deleting the entire gate from the checker left
    all ten tests passing, because nothing here touched the shipped code.
    """
    return [rule for rule, _n, _ctx in punctuation_hits(text)]


# ------------------------------------------------------------------ it must REFUSE

@pytest.mark.parametrize("prose,expected", [
    ("The two feed each other; the wave organizes the convection.", "semicolon"),
    ("The coupling is two-way: the wave organizes the convection.", "colon"),
    # escapes, not literals: the public-export style gate refuses any unicode dash in a
    # shipped file, and a test fixture is still a shipped file. Python sees the same
    # character either way, so the test is unchanged.
    ("The two feed each other \u2014 mutually.", "dash"),
    ("The two feed each other \u2013 mutually.", "dash"),
])
def test_planted_violation_is_caught(prose, expected):
    assert expected in scan(prose), f"the gate missed a planted {expected}"


# ------------------------------------------------------------------ it must ACCEPT
# A gate that fires falsely gets bypassed. These are the forms the style rules allow.

def test_clean_prose_passes():
    assert scan("The two feed each other, and the wave organizes the convection.") == []


def test_citation_group_semicolon_is_allowed():
    """A reference separator, not prose punctuation."""
    assert scan("Brightness temperature (Knapp et al. 2011; Knapp and NOAA 2014) is used.") == []


def test_figure_caption_is_exempt():
    """The rules permit 'term: definition' and semicolons in caption text."""
    caption = ("- S3. Regime control model. (b) The full panel, pooled: antecedent\n"
               "  convection dominant. Poisson model; wave-cluster-robust intervals.")
    assert scan(caption) == []


def test_reference_list_is_exempt():
    text = "Body prose here.\n\n## References\n\nBolton, D., 1980: Mon. Wea. Rev., 108, 1046;\n"
    assert scan(text) == []


def test_table_and_list_items_are_exempt():
    assert scan("| a | b: c |\n- item: value\n> quoted: thing") == []


# ------------------------------------------------------------- the gate itself fires

def test_the_real_checker_reports_a_planted_violation(tmp_path, monkeypatch, capsys):
    """Bind the SHIPPED gate, not just the predicate copied above.

    Without this, the two could drift and the suite would keep passing against a
    checker that no longer enforces anything.
    """
    import check_manuscript_figures as chk

    src = os.path.join(HERE, "..", "docs", "MANUSCRIPT_DRAFT.md")
    original = open(src).read()
    planted = original.replace("## 1. Introduction",
                               "## 1. Introduction\n\nThis is planted; it must be caught.",
                               1)
    assert planted != original, "could not plant a violation"
    tmp = tmp_path / "planted.md"
    tmp.write_text(planted)
    monkeypatch.setattr(chk, "SRC", str(tmp))
    chk.main()
    out = capsys.readouterr().out
    # A nonzero exit alone proves nothing: this checker refuses for many reasons, and
    # against a temp path several of them fire. Assert the punctuation error itself.
    assert "semicolon in body prose" in out, (
        "the shipped checker did not report the planted semicolon")


def test_the_supplement_is_scanned_too(tmp_path, monkeypatch, capsys):
    """The supplement is a separate document and must be covered.

    A mutation that scanned only the manuscript survived the rest of this file, and the
    supplement is where a real violation lived on 2026-08-13, so its coverage is bound
    here rather than assumed. The assertion is on the "supplement:" label, which only
    appears when that document is actually scanned.
    """
    import check_manuscript_figures as chk

    supp = os.path.join(HERE, "..", "docs", "SUPPLEMENT.md")
    planted = open(supp).read().replace(
        "Several analyses referenced",
        "One point first; several analyses referenced", 1)
    tmp = tmp_path / "planted_supp.md"
    tmp.write_text(planted)
    monkeypatch.setattr(chk, "SUPP", str(tmp))
    chk.main()
    out = capsys.readouterr().out
    assert "supplement: semicolon in body prose" in out, (
        "the checker did not scan the supplement")


# ------------------------------------------------------------------- run-on sentences

from check_manuscript_figures import runon_hits  # noqa: E402

# Fixtures are DOMAIN-NEUTRAL on purpose. This file ships in the public export, and the
# paper is unpublished, so a fixture must not carry a real result. Only the structural
# properties matter here: length, clause joins, and parenthetical groups.
RUNON = ("The first record carries no linkage, so one kind of persistence cannot be "
         "separated from the other kind of persistence, and the second control counts "
         "the same items over the preceding interval, a window disjoint by twenty four "
         "units that bounds but does not remove a single contribution to both of the "
         "two windows under discussion here.")


def test_planted_runon_is_caught():
    hits = runon_hits(RUNON)
    assert hits, "the gate missed a planted run-on"
    prose, joins, _ = hits[0]
    assert prose > 40 and joins >= 2


def test_split_version_passes():
    """The same content as two sentences must pass, or the gate punishes the fix."""
    fixed = RUNON.replace(", and the second control counts",
                          ". The second control counts")
    assert runon_hits(fixed) == []


def test_long_but_single_clause_sentence_passes():
    """Length alone is not a run-on. A number-dense sentence with one idea is fine, and
    flagging it would fire on roughly a quarter of this manuscript."""
    s = ("The measured quantity in the sample is larger in the first group than in the "
         "second by roughly two units of the scale used here, averaged along the sampled "
         "path across the full multi-decade record of the region examined throughout "
         "this particular study here.")
    assert len(s.split()) > 40
    assert runon_hits(s) == []


def test_parentheticals_do_not_inflate_length():
    """A sentence whose length is parenthetical numbers is not breathless."""
    s = ("The quantity is measured in the first frame (+1.0, +0.5 to +1.5), and it falls "
         "in the second frame (+0.5, +0.1 to +0.9, against +1.2 in the other frame), so "
         "the reported value depends on which estimator and which grouping the reader "
         "has in mind.")
    # The sentence MUST straddle the threshold, or it binds nothing: over 40 words as
    # written, at or under 40 once the parenthetical numbers are removed. An earlier
    # version sat under the threshold both ways, and a mutation that stopped stripping
    # parentheticals survived the whole file.
    assert len(s.split()) > 40
    assert len(re.sub(r"\([^()]*\)", "", s).split()) <= 40
    assert runon_hits(s) == []


def test_the_real_checker_reports_a_planted_runon(tmp_path, monkeypatch, capsys):
    import check_manuscript_figures as chk
    src = os.path.join(HERE, "..", "docs", "MANUSCRIPT_DRAFT.md")
    planted = open(src).read().replace("## 1. Introduction",
                                       "## 1. Introduction\n\n" + RUNON, 1)
    tmp = tmp_path / "planted.md"
    tmp.write_text(planted)
    monkeypatch.setattr(chk, "SRC", str(tmp))
    chk.main()
    assert "run-on sentence" in capsys.readouterr().out, (
        "the shipped checker did not report the planted run-on")
