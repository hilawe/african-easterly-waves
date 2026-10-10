"""The record-claims check's own contract. A planted mismatch is reported as MISMATCH with
exit status 1, an unreadable input is UNCHECKED with exit status 2 and never a pass, and
the helpers that turn record values into the document's forms do what they say."""
import importlib.util
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load(name):
    spec = importlib.util.spec_from_file_location(name + "_under_test", os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_planted_mismatch_fails_and_an_unreadable_input_is_unchecked_never_passed():
    C = _load("eval60_claims_check")
    rows, status = C.evaluate([("a", "3", "finished tracks", 1857, 1857), ("b", "4", "cohort", (63, 59), (63, 59))])
    assert status == 0 and [r["status"] for r in rows] == ["MATCH", "MATCH"]
    rows, status = C.evaluate([("a", "3", "finished tracks", 1857, 1856)])
    assert status == 1 and rows[0]["status"] == "MISMATCH" and rows[0]["document"] == 1857 and rows[0]["record"] == 1856
    rows, status = C.evaluate([("a", "3", "finished tracks", 1857, 1857), ("w", "11", "windows inside", True, C.Unchecked("no wind file"))])
    assert status == 2 and rows[1]["status"] == "UNCHECKED" and rows[1]["record"] is None and "no wind file" in rows[1]["reason"]
    rows, status = C.evaluate([("a", "3", "finished tracks", 1857, 1856), ("w", "11", "windows inside", True, C.Unchecked("no wind file"))])
    assert status == 1                                                                                  # a mismatch outranks an unchecked claim


def test_the_helpers_match_the_documents_forms():
    C = _load("eval60_claims_check")
    assert C.round_like(37.333333, 37.3) == 37.3 and C.round_like(28.583333, 28.6) == 28.6 and C.round_like(54.5, 54.5) == 54.5 and C.round_like(31.85, 31.85) == 31.85
    assert C.stamp("1990-10-03 18:00:00", 16.0, 38.0) == "3 Oct 18Z, 16 N 38 E" and C.stamp("2006-06-21 00:00:00", 7.5, 19.0) == "21 Jun 00Z, 7.5 N 19 E"
    bins = {"[50, 55)": 72, "[55, 60)": 22, "[-140, -135)": 20, "[-135, -130)": 21, "[-5, 0)": 22}
    assert C.bins_sum(bins, 50, 999) == 94 and C.bins_sum(bins, -999, -130) == 41 and C.bins_sum(bins, -100, 0) == 22
    assert C._equal((63, 59), [63, 59]) and not C._equal({"retained", "missing"}, {"retained"}) and C._equal(4.6021628338653876e-07, 4.6021628338653876e-07)


def test_a_missing_record_is_unchecked_at_the_evidence_level(tmp_path):
    C = _load("eval60_claims_check")
    ev = C.Evidence(str(tmp_path))
    with pytest.raises(C.Unchecked, match="not under the evidence directory"):
        ev.rec("eval60_screen_1990.json")


def test_an_integer_claim_never_matches_a_fractional_record_value():
    C = _load("eval60_claims_check")
    assert C._equal(0, 0) and C._equal(0, 0.0) and not C._equal(0, 0.9) and not C._equal(63, 63.5) and C._equal(True, True) and not C._equal(True, 1)


def test_a_missing_record_at_the_command_level_is_reported_unchecked_with_exit_2_and_a_written_record(tmp_path):
    C = _load("eval60_claims_check")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    out = tmp_path / "check.json"
    status = C.main(["--evidence", str(evidence), "--out", str(out), "--wind-1990", str(tmp_path / "none.nc"), "--wind-2006", str(tmp_path / "none.nc"), "--run-b", str(tmp_path / "nob"), "--run-c", str(tmp_path / "noc"), "--imagery", str(tmp_path / "noimg")])
    assert status == 2 and out.exists()
    import json
    rec = json.loads(out.read_text())
    assert rec["status"] == "unchecked" and rec["counts"] == {"MATCH": 0, "MISMATCH": 0, "UNCHECKED": 1} and rec["claims"][0]["id"] == "evidence" and rec["claims"][0]["status"] == "UNCHECKED"
