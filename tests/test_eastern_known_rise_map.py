"""Tests for the detectability map of the known rise (`EASTERN_KNOWN_RISE_2026-10-24.md`, section 7)."""

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import eastern_known_rise_map as KM  # noqa: E402
import test_eastern_known_rise as TK  # noqa: E402
import test_eastern_matched_profile as T  # noqa: E402

K, N, M, PS = KM.K, KM.N, KM.M, KM.PS
MAXIMA = {("3", "A"): 0.48, ("3", "B"): 0.13, ("4", "A"): 0.37, ("4", "B"): 0.13}


def test_the_outcome_is_the_noise_comparisons_rule():
    assert KM.outcome({"3": 0.5, "4": 0.4}, MAXIMA) == ("above", 4)
    assert KM.outcome({"3": 0.41, "4": 0.44}, MAXIMA) == ("mixed", 3)                    # the archive's own pattern
    assert KM.outcome({"3": 0.1, "4": 0.1}, MAXIMA) == ("not separated", 0)
    assert KM.outcome({"3": 0.48, "4": 0.37}, MAXIMA) == ("mixed", 2)                    # equal to a largest is not above


def test_the_noise_maxima_and_the_rule_are_the_noise_comparisons_own():
    summ = {"noise": {b: {"rise": {"3": [-1.0, MAXIMA[("3", b)]], "4": [-1.0, MAXIMA[("4", b)]]}} for b in "AB"}}
    assert N.noise_maxima(summ) == MAXIMA
    calls = []
    real = N.exceedance
    KM.N.exceedance = lambda rise, maxima: calls.append(1) or real(rise, maxima)
    try:
        assert KM.outcome({"3": 0.5, "4": 0.4}, MAXIMA) == ("above", 4) and calls == [1]       # the map calls the comparison's rule
    finally:
        KM.N.exceedance = real


def test_a_row_pools_by_the_noise_comparisons_summary_and_counts_outcomes():
    entries = TK.pooled_entries({"A": 0.5, "B": 0.375})
    entries[0]["rise"] = {"3": 0.1, "4": 0.1}; entries[12]["rise"] = {"3": 0.41, "4": 0.44}
    for e in entries:
        e["matched"]["numerator"][1] = 2.0                                                  # day 1 pools apart from day 0
    r = KM.row(1.3, entries, MAXIMA, {"3": 0.4, "4": 0.4})
    assert r["pooled_rise"]["A"]["3"] == pytest.approx(0.5) and r["pooled_rise"]["B"]["4"] == pytest.approx(0.375)
    assert r["reaches_archive"] is False and r["outcomes"] == {"above": 9, "mixed": 10, "not separated": 1} and r["by_placement"][12] == "mixed"
    assert KM.row(1.3, entries, MAXIMA, {"3": 0.375, "4": 0.375})["reaches_archive"] is True                # equal reaches


def run_record(m, rise, built=None, grow=2.0):
    return {"generated_by": {"lobe_mult": m, "grow_days": grow}, "results": {"placements": TK.pooled_entries({"A": rise, "B": rise}), "stored_case_placements": built}}


def test_the_map_orders_multiples_takes_the_built_construction_once_and_names_the_noise_matched_one():
    built = TK.pooled_entries({"A": 0.2, "B": 0.2})
    runs = [run_record(2.0, 0.45, None), run_record(1.3, 0.3, built), run_record(1.5, 0.35, built), run_record(3.0, 0.6, None)]
    mp = KM.build_map(runs, MAXIMA, {"3": 0.41, "4": 0.44})
    assert [r["multiple"] for r in mp["rows"]] == [1.0, 1.3, 1.5, 2.0, 3.0] and mp["noise_matched"] == 2.0
    assert mp["rows"][0]["pooled_rise"]["A"]["3"] == pytest.approx(0.2)
    assert KM.build_map(runs[1:3], MAXIMA, {"3": 0.41, "4": 0.44})["noise_matched"] is None                # only multiples below the archive's
    with pytest.raises(SystemExit, match="disagree on the construction as built"):
        KM.build_map([run_record(1.3, 0.3, built), run_record(1.5, 0.3, TK.pooled_entries({"A": 0.1, "B": 0.1}))], MAXIMA, {"3": 0.4, "4": 0.4})
    with pytest.raises(SystemExit, match="two runs at multiple 1.3"):
        KM.build_map([run_record(1.3, 0.3), run_record(1.3, 0.4)], MAXIMA, {"3": 0.4, "4": 0.4})
    with pytest.raises(SystemExit, match="does not grow over 2 days"):
        KM.build_map([run_record(1.3, 0.3, grow=3.0)], MAXIMA, {"3": 0.4, "4": 0.4})


def test_the_readings_follow_the_map():
    mp = {"rows": [{"multiple": 2.0, "placements": 20, "outcomes": {"above": 5, "mixed": 12, "not separated": 3}}], "noise_matched": 2.0}
    r = KM.readings(mp, "mixed", 3)
    assert r[0] == "under the noise comparison's rule the archive is MIXED, passing 3 of its 4 block-and-day checks"
    assert "is 2, and of its 20 placements 5 come out ABOVE, 12 MIXED and 3 NOT SEPARATED" in r[1] and "not calibrated probabilities" in r[2]
    r = KM.readings(dict(mp, noise_matched=None), "mixed", 3)
    assert "no multiple up to 2 gives a pooled rise on this noise that reaches the archive's" in r[1]


def test_main_gates_and_writes(tmp_path, monkeypatch):
    gen = T.gen_fixture()
    noise_entries = TK.pooled_entries({"A": 0.1, "B": 0.1})
    noise = TK.noise_record(noise_entries)
    noise["generated_by"] = dict(gen, sha256=N.SCRIPT_SHA256_AT_LOAD); noise["results"]["verdict"] = {"archive": "above"}
    mat = {"generated_by": dict(gen, sha256=M.SCRIPT_SHA256_AT_LOAD),
           "results": {"matched": {"by_day": [0.1, 0.2, 0.3, 0.5, 0.6]}, "cohort": ["a"] * 60, "changing": {"histories": 86}}}
    npth, mpth = tmp_path / "noise.json", tmp_path / "mat.json"; npth.write_text(json.dumps(noise)); mpth.write_text(json.dumps(mat))
    def write_run(name, m, rise, **over):
        rec = run_record(m, rise); rec["generated_by"].update(dict(gen, sha256=K.SCRIPT_SHA256_AT_LOAD, noise_archive={"sha256": KM.sha256_of(str(npth))},
                                                                  matched_archive={"sha256": KM.sha256_of(str(mpth))}), **over)
        p = tmp_path / name; p.write_text(json.dumps(rec)); return str(p)
    r1, r2 = write_run("r1.json", 1.3, 0.3), write_run("r2.json", 2.0, 0.6)
    args = ["--noise-archive", str(npth), "--matched-archive", str(mpth), "--out-dir", str(tmp_path), "--runs", r1, r2]
    assert KM.main(args) == 0
    out = json.loads((tmp_path / "eval60_known_rise_map.json").read_text())
    assert out["results"]["noise_matched"] == 2.0 and out["results"]["archive"]["outcome"] == "above" and out["generated_by"]["runs"][r1] == KM.sha256_of(r1)
    assert out["results"]["noise_maxima"] == {f"day {k}, block {b}": pytest.approx(0.1) for k in ("3", "4") for b in "AB"}     # the noise record's own maxima
    assert (tmp_path / "eval60_known_rise_map.md").read_text().startswith("# detectability map")
    with pytest.raises(SystemExit, match="never overwritten"):
        KM.main(args)
    for tag, run, match in (("_a", write_run("b1.json", 1.3, 0.3, sha256="x"), "different version of the known-rise script"),
                            ("_b", write_run("b2.json", 1.3, 0.3, noise_archive={"sha256": "x"}), "another noise or matched-profile record"),
                            ("_c", write_run("b3.json", 1.3, 0.3, xband={"x": "o"}), "the xband input differs")):
        with pytest.raises(SystemExit, match=match):
            KM.main(args[:-2] + [run, "--tag", tag])
    noise["results"]["verdict"] = {"archive": "mixed"}; npth.write_text(json.dumps(noise))
    r3 = write_run("r3.json", 1.3, 0.3)
    with pytest.raises(SystemExit, match="differs from the noise record's verdict"):
        KM.main(args[:-2] + [r3, "--tag", "_d"])
    mat["generated_by"]["sha256"] = "x"; mpth.write_text(json.dumps(mat))
    with pytest.raises(SystemExit, match="different version of its script"):
        KM.main(args[:-2] + [r3, "--tag", "_e"])
