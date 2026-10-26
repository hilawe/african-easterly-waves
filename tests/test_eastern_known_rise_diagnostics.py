"""Tests for the known-rise diagnostics (`EASTERN_KNOWN_RISE_2026-10-24.md`, section 9)."""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import eastern_known_rise_diagnostics as D  # noqa: E402
import test_eastern_known_rise as TK  # noqa: E402
import test_eastern_matched_profile as T  # noqa: E402

K, N, M, H, J, A = D.K, D.N, D.M, D.H, D.J, D.A
MAXIMA = {("3", "A"): 0.48, ("3", "B"): 0.13, ("4", "A"): 0.37, ("4", "B"): 0.13}


def test_fields_keep_every_detected_history_and_count_the_troughs_each_follows(monkeypatch):
    c1 = np.full(A.SEASON_DAYS, np.nan); c1[20:25] = [60.0, 58.0, 56.0, 54.0, 52.0]
    c2 = np.full(A.SEASON_DAYS, np.nan); c2[19:25] = [62.5, 60.5, 58.5, 56.5, 54.5, 52.5]            # half a degree off on the history's days
    monkeypatch.setattr(K, "build", lambda tracks, rng, g, m, pt: (np.zeros((J.N_SAMPLES, len(K.XL))), [c1, c2], [20, 19]))
    both = {"days": [20, 21, 22, 23, 24], "lons": [60.0, 58.0, 56.0, 54.0, 52.0]}
    none = {"days": [40, 41, 42, 43], "lons": [60.0, 58.0, 56.0, 54.0]}
    monkeypatch.setattr(H, "detected", lambda S, month_of: ([both, none], None))
    monkeypatch.setattr(D.P, "b0_rows", lambda Hb, month_of: {y: "b0" for y in Hb})
    b0, follow, every, n = D.fields(K.zero_background, 0, {}, 1.3, {b: 1.0 for b in D.L.BAND_LAT}, None, None)
    assert all(v == [both] for v in follow.values()) and all(v == [both, none] for v in every.values()) and n[id(both)] == 2 and id(none) not in n


def test_the_following_readings_must_reproduce_the_known_rise_record():
    stored = [{"placement": p, "rise": {"3": 0.1, "4": 0.2}, "cohort_histories": 30, "matched": {"by_day": [0.1] * 5}} for p in range(20)]
    D.check_follow(json.loads(json.dumps(stored)), stored)
    relabeled = json.loads(json.dumps(stored)); relabeled[0]["placement"] = 999
    for mine, theirs in (([], stored), (stored, stored[:1]), (stored[:19], stored[:19]), (relabeled, stored), (stored, relabeled), (stored[::-1], stored[::-1])):
        with pytest.raises(SystemExit, match="do not hold every declared placement in order"):
            D.check_follow(json.loads(json.dumps(mine)), theirs)
    for field, value in (("rise", {"3": 0.1, "4": 0.3}), ("cohort_histories", 31)):
        bad = json.loads(json.dumps(stored)); bad[1][field] = value
        with pytest.raises(SystemExit, match="placement 1 does not reproduce"):
            D.check_follow(bad, stored)
    bad = json.loads(json.dumps(stored)); bad[2]["matched"]["by_day"] = [0.2] * 5
    with pytest.raises(SystemExit, match="placement 2 does not reproduce"):
        D.check_follow(bad, stored)


def test_outcomes_and_pooled_rises_use_the_comparisons_own_functions():
    es = TK.pooled_entries({"A": 0.5, "B": 0.1})
    pr = D.pooled_rises(es)
    assert pr["A"]["3"] == pytest.approx(0.5) and pr["B"]["4"] == pytest.approx(0.1)
    assert D.outcomes(es, MAXIMA) == {"above": 10, "mixed": 0, "not separated": 10}


def test_individually_reaching_placements_meet_both_archive_rises():
    es = TK.pooled_entries({"A": 0.5, "B": 0.5})
    es[0]["rise"] = {"3": 0.45, "4": 0.40}; es[1]["rise"] = {"3": 0.41, "4": 0.44}
    r = D.individually_reaching(es, {"3": 0.41, "4": 0.44}, MAXIMA)
    assert r["placements"] == 19 and r["outcomes"] == {"above": 18, "mixed": 1, "not separated": 0}         # equal meets


def test_the_further_reference_uses_its_own_blocks(monkeypatch):
    monkeypatch.setattr(N, "placement", lambda case, p, *a: {"rise": {"3": 0.01 * p, "4": 0.02 * p}})
    monkeypatch.setattr(D, "fields", lambda *a: ("b0", "fh", "ah", {}))
    monkeypatch.setattr(D, "cohort_entry", lambda p, b0, hist: {"placement": p, "rise": {"3": 0.9, "4": 0.9}})
    stored = [{"rise": {"3": 0.35, "4": 0.75}}] * 20
    r = D.fresh({}, {}, None, None, stored, MAXIMA)
    assert r["fresh_noise_maxima"]["day 3, block A"] == pytest.approx(0.29) and r["fresh_noise_maxima"]["day 4, block B"] == pytest.approx(0.78)
    assert r["original_control_on_fresh_noise"] == {"above": 0, "mixed": 20, "not separated": 0}
    assert r["fresh_control_on_original_noise"] == {"above": 20, "mixed": 0, "not separated": 0}                  # each against the other reference
    assert D.FRESH == {"A": tuple(range(20, 30)), "B": tuple(range(30, 40))} and not set(D.FRESH["A"] + D.FRESH["B"]) & set(K.PLACEMENTS)


def test_main_gates(tmp_path, monkeypatch):
    gen = T.gen_fixture()
    noise = TK.noise_record(TK.pooled_entries({"A": 0.1, "B": 0.1})); noise["generated_by"] = dict(gen, sha256=N.SCRIPT_SHA256_AT_LOAD)
    mat = {"generated_by": dict(gen, sha256=M.SCRIPT_SHA256_AT_LOAD), "results": {"matched": {"by_day": [0.1, 0.2, 0.3, 0.5, 0.6]}, "cohort": ["a"] * 60, "changing": {"histories": 86}}}
    npth, mpth = tmp_path / "noise.json", tmp_path / "mat.json"; npth.write_text(json.dumps(noise)); mpth.write_text(json.dumps(mat))
    clean = tmp_path / "clean.json"; clean.write_text(json.dumps({"results": {"placements": []}}))
    def run(m, **over):
        rec = {"generated_by": dict({"sha256": K.SCRIPT_SHA256_AT_LOAD, "lobe_mult": m, "clean": {"path": str(clean), "sha256": D.sha256_of(str(clean))}}, **over),
               "results": {"placements": []}}
        p = tmp_path / f"run_{m}.json"; p.write_text(json.dumps(rec)); return str(p)
    r1, r2 = run(1.3), run(2.5)
    mp = {"generated_by": {"runs": {r1: D.sha256_of(r1), r2: D.sha256_of(r2)}}, "results": {"noise_matched": 2.5}}
    mpp = tmp_path / "map.json"; mpp.write_text(json.dumps(mp))
    monkeypatch.setattr(K.C, "load_inputs", lambda a: ({}, {}, {}, {}, dict(gen)))
    monkeypatch.setattr(D.P, "measured_parts", lambda: None)
    for name in ("fixed_history", "population", "fresh"):
        monkeypatch.setattr(D, name, lambda *a, name=name: {"fixed_history": {"pooled_rise": {}, "cohort_histories_following_several_troughs": 5},
                                                          "population": {}, "fresh": {"fresh_noise_maxima": {}, "original_control_on_fresh_noise": {},
                                                                                      "fresh_control_on_original_noise": {}}}[name])
    monkeypatch.setattr(D, "text", lambda res: "# known-rise diagnostics\n")
    args = ["--shape", "s", "--inject", "i", "--records", "r", "--histories", "h", "--xband", "x", "--matched-archive", str(mpth), "--noise-archive", str(npth),
            "--map", str(mpp), "--out-dir", str(tmp_path)]
    assert D.main(args) == 0
    out = json.loads((tmp_path / "eval60_known_rise_diagnostics.json").read_text())
    assert out["results"]["fixed_history"]["cohort_histories_following_several_troughs"] == 5 and out["generated_by"]["map"]["sha256"] == D.sha256_of(str(mpp))
    with pytest.raises(SystemExit, match="never overwritten"):
        D.main(args)
    open(r1, "a").write(" ")
    with pytest.raises(SystemExit, match="is not the run the map read"):
        D.main(args + ["--tag", "_a"])
    r1 = run(1.3); mp["generated_by"]["runs"] = {r1: D.sha256_of(r1), r2: D.sha256_of(r2)}; mp["results"]["noise_matched"] = 2.0; mpp.write_text(json.dumps(mp))
    with pytest.raises(SystemExit, match="does not hold the control and the noise-matched multiple"):
        D.main(args + ["--tag", "_b"])
    mp["results"]["noise_matched"] = 2.5; mpp.write_text(json.dumps(mp)); clean.write_text(json.dumps({"results": {"placements": [1]}}))
    with pytest.raises(SystemExit, match="clean record has changed"):
        D.main(args + ["--tag", "_c"])
    noise["generated_by"]["xband"] = {"x": "o"}; npth.write_text(json.dumps(noise))
    with pytest.raises(SystemExit, match="the xband input differs"):
        D.main(args + ["--tag", "_d"])
