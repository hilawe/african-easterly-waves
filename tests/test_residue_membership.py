"""The residue membership accounting, bound on a synthetic residual artifact and case
artifacts whose claims it must check rather than trust.

MUTATION LIST, written before the assertions:
  R1 a claim naming an index outside the claimed category is accepted;
  R2 an index claimed by two cases is counted twice;
  R3 the remaining lists are not the complement of the explained lists;
  R4 the both-sides-extra pair is counted among the version-1-extra pairs.
Added after a review credited a copied case artifact from an unrelated window:
  R5 a case artifact naming another case identifier is credited;
  R6 a case artifact from a stale port run (different port-output digest) with valid
     category indices is credited;
  R7 a case artifact lacking the binding digests is credited.
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _load():
    spec = importlib.util.spec_from_file_location(
        "residue_membership", os.path.join(HERE, "..", "scripts", "residue_membership.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["residue_membership"] = mod
    spec.loader.exec_module(mod)
    return mod


CASE = "abc123"
HASHES = {"tracker_case.mat": "c" * 64, "tracker_port.mat": "p" * 64,
          "tracker_octave_instrumented.mat": "o1" * 32}


def bound(explains, oracle_hash="o2" * 32):
    """A case artifact bound to the residual artifact's case and files; a different
    reference output digest on purpose, since dump schedules differ."""
    return {"case_id": CASE, "explains": explains,
            "input_sha256": {"tracker_case.mat": HASHES["tracker_case.mat"],
                             "tracker_port.mat": HASHES["tracker_port.mat"],
                             "tracker_octave_instrumented.mat": oracle_hash}}


def residuals():
    return {"case_id": CASE, "input_sha256": dict(HASHES),
            "v1_unmatched": [{"index": 6, "eligible_counterpart_exists": False},
                             {"index": 19, "eligible_counterpart_exists": False},
                             {"index": 31, "eligible_counterpart_exists": True}],
            "pairs": [{"v1_index": 74, "extra_kind": "extra_v1_only"},
                      {"v1_index": 67, "extra_kind": "extra_v1_only"},
                      {"v1_index": 5, "extra_kind": "extra_v1_only"},
                      {"v1_index": 94, "extra_kind": "extra_both_sides"},
                      {"v1_index": 50, "extra_kind": "no_extra"}]}


def test_membership_checks_every_claim_and_derives_the_remaining_lists():
    M = _load()
    cases = {"a.json": bound({"unmatched_v1_tracks": [19], "v1_extra_pairs": []}),
             "b.json": bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74, 67]}),
             "c.json": bound({})}                            # declares nothing
    r = M.membership(residuals(), cases)
    assert r["problems"] == []
    assert r["unmatched_v1_no_eligible_counterpart"] == [6, 19]
    assert r["unmatched_v1_with_eligible_counterpart"] == [31]
    assert r["pairs_by_extra_kind"]["extra_v1_only"] == [5, 67, 74]
    assert r["explained_unmatched"] == [19] and r["remaining_unmatched"] == [6]
    assert r["explained_v1_extra_pairs"] == [67, 74] and r["remaining_v1_extra_pairs"] == [5]
    assert r["counts"] == {"unmatched_no_eligible": 2, "unmatched_explained": 1,
                           "v1_extra_pairs": 3, "v1_extra_pairs_explained": 2,
                           "both_sides_extra_pairs": 1}


def test_wrong_or_duplicate_claims_are_problems(tmp_path):
    M = _load()
    r = M.membership(residuals(), {"a.json": bound({"unmatched_v1_tracks": [31]})})
    assert any("not in the no-eligible-counterpart set" in p for p in r["problems"])
    r = M.membership(residuals(), {"a.json": bound({"v1_extra_pairs": [94]})})
    assert any("not a version-1-extra pair" in p for p in r["problems"])
    r = M.membership(residuals(), {"a.json": bound({"v1_extra_pairs": [74]}),
                                   "b.json": bound({"v1_extra_pairs": [74]})})
    assert any("claimed by more than one case" in p for p in r["problems"])
    assert r["counts"]["v1_extra_pairs_explained"] == 1
    # and the command refuses with nothing written
    pr, pc, out = tmp_path / "r.json", tmp_path / "c.json", tmp_path / "m.json"
    pr.write_text(json.dumps(residuals()))
    pc.write_text(json.dumps(bound({"unmatched_v1_tracks": [31]})))
    assert M.main(["--residuals", str(pr), "--cases", str(pc), "--out", str(out)]) == 2
    assert not out.exists()


def test_a_case_from_another_window_or_a_stale_port_run_is_refused(tmp_path):
    """The claimed indices are valid category members in every case below, so only
    the binding can refuse them."""
    M = _load()
    valid = {"unmatched_v1_tracks": [19], "v1_extra_pairs": [74]}
    other_case = dict(bound(valid), case_id="another-window")
    r = M.membership(residuals(), {"a.json": other_case})
    assert any("is not the residual artifact's" in p for p in r["problems"])
    assert "counts" not in r
    stale = bound(valid)
    stale["input_sha256"]["tracker_port.mat"] = "q" * 64
    r = M.membership(residuals(), {"a.json": stale})
    assert any("tracker_port.mat differs" in p for p in r["problems"])
    unbound = {"case_id": CASE, "explains": valid}
    r = M.membership(residuals(), {"a.json": unbound})
    assert any("digest of tracker_case.mat is missing" in p for p in r["problems"])
    # a different REFERENCE output digest is not a refusal (dump schedules differ)
    r = M.membership(residuals(), {"a.json": bound(valid, oracle_hash="z" * 64)})
    assert r["problems"] == [] and r["counts"]["unmatched_explained"] == 1
    # the command refuses the stale case with nothing written
    pr, pc, out = tmp_path / "r.json", tmp_path / "c.json", tmp_path / "m.json"
    pr.write_text(json.dumps(residuals()))
    pc.write_text(json.dumps(stale))
    assert M.main(["--residuals", str(pr), "--cases", str(pc), "--out", str(out)]) == 2
    assert not out.exists()
