"""Tests for the daily profile on a matched sample (`EASTERN_MATCHED_PROFILE_2026-10-23.md`)."""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import eastern_matched_profile as M  # noqa: E402

G, P, A = M.G, M.P, M.A
BAND = "15 to 19.5 N"
OTHER = "25 to 29.5 N"

LONG = {"days": [10, 11, 12, 13, 14], "lons": [60.0, 58.0, 56.0, 54.0, 52.0]}      # read on every day
SHORT = {"days": [20, 21, 22], "lons": [50.0, 48.0, 46.0]}                          # three members
EDGE = {"days": [30, 31, 32, 33, 34], "lons": [70.0, 68.0, 66.0, 64.0, 62.0]}       # day 0 reads 76 E, off the grid


def rows_fixture():
    rows = [np.zeros(len(A.LONS)) for _ in range(A.SEASON_DAYS)]
    for h, member, east in ((LONG, -2.0, 1.0), (SHORT, 1.0, 1.0), (EDGE, -2.0, 3.0)):     # a member's anomaly may be positive
        for d, x in zip(h["days"], h["lons"]):
            rows[d][P.col(x)] = member
            if x + P.OFFSET <= A.LONS[-1]:
                rows[d][P.col(x + P.OFFSET)] = east
    return rows


def fixture():
    return {(1983, BAND): rows_fixture(), (1983, OTHER): rows_fixture()}, {(1983, BAND): [SHORT, LONG, EDGE], (1983, OTHER): [LONG]}


def test_the_eligible_days_are_the_joint_readers():
    assert M.eligible_days(LONG) == [0, 1, 2, 3, 4] and M.eligible_days(SHORT) == [0, 1, 2] and M.eligible_days(EDGE) == [1, 2, 3, 4]
    at_edge = {"days": [1, 2, 3, 4, 5], "lons": [69.0, 68.0, 67.0, 66.0, 65.0]}       # 69 E reads 75 E, which lies on the grid
    assert M.eligible_days(at_edge) == [0, 1, 2, 3, 4]


def test_the_cohort_keeps_the_histories_read_on_every_day_with_their_subset_ids():
    b0, hist = fixture()
    cohort, ids = M.matched(hist, (BAND,))
    assert cohort[(1983, BAND)] == [LONG] and cohort[(1983, OTHER)] == [] and ids == ["1983-15N-02"]
    assert M.history_id(2007, "20 to 24.5 N", 3) == "2007-20N-03"


def test_the_kept_sums_give_the_joint_readers_profile():
    b0, hist = fixture()
    c = M.curve(b0, hist, (BAND,))
    assert c["days"] == [2, 3, 3, 2, 2] and c["off_grid"] == [1, 0, 0, 0, 0] and c["histories"] == 3
    assert c["numerator"] == [2.0, 5.0, 5.0, 4.0, 4.0] and c["denominator"] == [3.0, 5.0, 5.0, 4.0, 4.0]
    assert c["by_day"] == G.profile(b0, hist, (BAND,))["by_day"] and np.isclose(c["by_day"][0], 2 / 3)


def test_the_curve_refuses_sums_the_joint_reader_does_not_give(monkeypatch):
    b0, hist = fixture()
    real = G.profile
    for change in ({"by_day": [0.0] * 5}, {"days": [9] * 5}, {"off_grid": [9] * 5}):
        monkeypatch.setattr(G, "profile", lambda *a, change=change: dict(real(*a), **change))
        with pytest.raises(SystemExit, match="kept sums"):
            M.curve(b0, hist, (BAND,))


def test_the_days_outside_the_cohort_come_from_the_difference_of_the_sums():
    b0, hist = fixture()
    o = M.outside(M.curve(b0, hist, (BAND,)), M.curve(b0, M.matched(hist, (BAND,))[0], (BAND,)))
    assert o["days"] == [1, 2, 2, 1, 1] and o["numerator"] == [1.0, 4.0, 4.0, 3.0, 3.0] and o["denominator"] == [1.0, 3.0, 3.0, 2.0, 2.0]
    assert o["by_day"] == [1.0, 4 / 3, 4 / 3, 1.5, 1.5]
    same = {"numerator": [1.0] * 5, "denominator": [2.0] * 5, "days": [1] * 5}
    assert M.outside(same, same)["by_day"] == [None] * 5
    with pytest.raises(SystemExit, match="more history-days"):
        M.outside({**same, "days": [0] * 5}, same)


def test_the_histories_outside_the_cohort_must_add_up_to_the_changing_sample():
    b0, hist = fixture()
    cohort, _ = M.matched(hist, (BAND,))
    rest = M.complement(hist, cohort)
    assert rest[(1983, BAND)] == [SHORT, EDGE] and rest[(1983, OTHER)] == [LONG]
    changing, match = M.curve(b0, hist, (BAND,)), M.curve(b0, cohort, (BAND,))
    r = M.sums(b0, rest, (BAND,))
    assert M.check_additive(changing, match, r) == 0.0
    nudge = lambda s, k, i, v: dict(s, **{k: [x + (v if j == i else 0) for j, x in enumerate(s[k])]})
    assert M.check_additive(changing, match, nudge(r, "numerator", 2, 1e-12)) == pytest.approx(1e-12, abs=1e-14)   # summation order is tolerated and recorded
    for k, i, v, what in (("numerator", 2, 1e-6, "numerator on day 2"), ("denominator", 4, -1e-6, "denominator on day 4")):
        with pytest.raises(SystemExit, match=what):
            M.check_additive(changing, match, nudge(r, k, i, v))
    for k in ("days", "off_grid"):
        with pytest.raises(SystemExit, match="counts are not"):
            M.check_additive(changing, match, nudge(r, k, 0, 1))
    with pytest.raises(SystemExit, match="counts are not"):
        M.check_additive(changing, match, dict(r, histories=r["histories"] + 1))


def test_cohort_sums_scaled_apart_from_the_changing_samples_are_refused(monkeypatch):
    b0, hist = fixture()
    monkeypatch.setattr(M, "BANDS", (BAND,))
    real = M.sums
    def cohort_halved(b0_by, hist_by, bands):                                 # the ratio gate cannot see a paired scaling
        s = real(b0_by, hist_by, bands)
        if [h for (y, b), hs in hist_by.items() if b in bands for h in hs] == [LONG]:
            s = dict(s, numerator=[x / 2 for x in s["numerator"]], denominator=[x / 2 for x in s["denominator"]])
        return s
    monkeypatch.setattr(M, "sums", cohort_halved)
    assert M.curve(b0, M.matched(hist, (BAND,))[0], (BAND,))["by_day"] == [0.5] * 5
    with pytest.raises(SystemExit, match="not the cohort's plus the outside's"):
        M.measure(b0, hist)


def test_the_measure_reads_both_curves_over_the_primary_bands(monkeypatch):
    b0, hist = fixture()
    monkeypatch.setattr(M, "BANDS", (BAND,))
    r = M.measure(b0, hist)
    assert r["cohort"] == ["1983-15N-02"] and r["matched"]["by_day"] == [0.5] * 5 and r["matched"]["days"] == [1] * 5
    assert r["changing"]["histories"] == 3 and r["matched"]["histories"] == 1
    assert r["difference"] == [0.5 - c for c in r["changing"]["by_day"]] and r["outside"]["days"] == [1, 2, 2, 1, 1]
    assert r["additive_max_gap_ms"] == 0.0
    real = M.matched
    monkeypatch.setattr(M, "matched", lambda hist_by, bands: (real(hist_by, bands)[0], ["x", "y"]))
    with pytest.raises(SystemExit, match="not read on every day"):
        M.measure(b0, hist)
    monkeypatch.setattr(M, "matched", lambda hist_by, bands: ({(1983, BAND): [LONG, SHORT]}, ["a", "b"]))
    with pytest.raises(SystemExit, match="not read on every day"):
        M.measure(b0, hist)


def joint_record(gen):
    return {"generated_by": dict(gen), "results": {"measure": {"profile": [0.1] * 5, "profile_days": [3] * 5, "profile_off_grid": [0] * 5, "histories": 3}}}


def gen_fixture():
    return {"shape": {"path": "a", "sha256": "s1"}, "inject": {"path": "b", "sha256": "s2"}, "records": {"m": "s3"},
            "histories": {"path": "c", "sha256": "s4"}, "xband": {"x": "s5"}}


def test_the_inputs_must_be_the_joint_records():
    gen = gen_fixture()
    M.check_same_inputs(gen, joint_record(gen))
    M.check_same_inputs({**gen, "shape": {"path": "elsewhere", "sha256": "s1"}}, joint_record(gen))      # the same file under another path
    for k, changed in (("shape", {"path": "a", "sha256": "x"}), ("inject", {"path": "b", "sha256": "x"}), ("records", {"m": "x"}),
                       ("histories", {"path": "c", "sha256": "x"}), ("xband", {"x": "s5", "y": "s6"})):
        with pytest.raises(SystemExit, match=f"the {k} input"):
            M.check_same_inputs({**gen, k: changed}, joint_record(gen))
    rec = joint_record(gen); del rec["generated_by"]["inject"]
    with pytest.raises(SystemExit, match="the inject input"):
        M.check_same_inputs(gen, rec)


def test_the_curve_over_every_history_must_equal_the_joint_record():
    rec = joint_record(gen_fixture())
    ok = {"by_day": [0.1] * 5, "days": [3] * 5, "off_grid": [0] * 5, "histories": 3}
    M.check_joint(ok, rec)
    for change in ({"by_day": [0.1] * 4 + [0.2]}, {"days": [3] * 4 + [2]}, {"off_grid": [1] + [0] * 4}, {"histories": 4}):
        with pytest.raises(SystemExit, match="differs from the joint record"):
            M.check_joint({**ok, **change}, rec)


def test_the_text_carries_every_sample_and_the_cohort(monkeypatch):
    b0, hist = fixture()
    monkeypatch.setattr(M, "BANDS", (BAND,))
    t = M.text(M.measure(b0, hist))
    assert "| changing | 3 | 0 | 0.667 | 2.000 | 3.000 | 2 | 1 |" in t and "| matched | 1 | 4 | 0.500 |" in t
    assert "| outside the cohort | | 3 | 1.500 | 3.000 | 2.000 | 1 | |" in t and "| 0 | -0.167 |" in t
    assert "The cohort is 1 histories: 1983-15N-02." in t and "the changing sample's within 0.0e+00 m/s." in t


def test_main_refuses_a_joint_record_from_another_script_and_never_overwrites(tmp_path, monkeypatch):
    gen = gen_fixture(); rec = joint_record(gen); rec["generated_by"]["sha256"] = "other"
    jp = tmp_path / "joint.json"; jp.write_text(json.dumps(rec))
    args = ["--shape", "s", "--inject", "i", "--records", "r", "--histories", "h", "--xband", "x", "--joint-archive", str(jp), "--out-dir", str(tmp_path)]
    with pytest.raises(SystemExit, match="different version"):
        M.main(args)
    (tmp_path / "eval60_matched_profile.md").write_text("")
    with pytest.raises(SystemExit, match="never overwritten"):
        M.main(args)


def test_main_writes_the_record_after_its_gates(tmp_path, monkeypatch):
    b0, hist = fixture()
    monkeypatch.setattr(M, "BANDS", (BAND,))
    gen = gen_fixture()
    changing = M.curve(b0, hist, (BAND,))
    rec = joint_record(gen); rec["generated_by"]["sha256"] = G.SCRIPT_SHA256_AT_LOAD
    rec["results"]["measure"].update({"profile": changing["by_day"], "profile_days": changing["days"], "profile_off_grid": changing["off_grid"], "histories": 3})
    jp = tmp_path / "joint.json"; jp.write_text(json.dumps(rec))
    monkeypatch.setattr(M.C, "load_inputs", lambda a: (hist, None, None, None, dict(gen)))
    monkeypatch.setattr(M, "b0_rows", lambda by: b0)
    args = ["--shape", "s", "--inject", "i", "--records", "r", "--histories", "h", "--xband", "x", "--joint-archive", str(jp), "--out-dir", str(tmp_path)]
    assert M.main(args) == 0
    out = json.loads((tmp_path / "eval60_matched_profile.json").read_text())
    assert out["results"]["cohort"] == ["1983-15N-02"] and out["generated_by"]["sha256"] == M.SCRIPT_SHA256_AT_LOAD
    assert out["generated_by"]["joint_archive"]["sha256"] == M.sha256_of(str(jp)) and out["generated_by"]["declared"] == "EASTERN_MATCHED_PROFILE_2026-10-23.md"
    assert (tmp_path / "eval60_matched_profile.md").read_text().startswith("# matched daily profile")
    rec["results"]["measure"]["profile"] = [0.0] * 5; jp.write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="differs from the joint record"):
        M.main(args + ["--tag", "_b"])
    rec["results"]["measure"]["profile"] = changing["by_day"]; rec["generated_by"]["records"] = {"m": "other"}; jp.write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="the records input"):
        M.main(args + ["--tag", "_c"])
