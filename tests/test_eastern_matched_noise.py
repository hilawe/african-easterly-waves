"""Tests for the fixed-cohort daily profile against matched noise (`EASTERN_MATCHED_NOISE_2026-10-23.md`)."""

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import eastern_matched_noise as N  # noqa: E402
import test_eastern_matched_profile as T  # noqa: E402

M, PS = N.M, N.PS


def test_the_rises_are_from_day_0():
    r = N.rises([0.1, 0.2, 0.3, 0.5, 0.75])
    assert set(r) == {"3", "4"} and r["3"] == pytest.approx(0.4) and r["4"] == pytest.approx(0.65)


def test_a_placement_is_the_matched_measure_on_the_prestart_fields(monkeypatch):
    b0, hist = T.fixture()
    monkeypatch.setattr(M, "BANDS", (T.BAND,))
    seen = []
    monkeypatch.setattr(PS, "fields", lambda case, bg, place, tracks, by, std, parts, month_of: (seen.append((case, bg, place)) or (None, b0, hist)))
    e = N.placement(N.CONTROL, 7, {}, {}, {}, None, None)
    assert seen == [(N.CONTROL, "matched", 7)] and e["placement"] == 7 and e["cohort_histories"] == 1
    assert e["matched"]["by_day"] == [0.5] * 5 and e["rise"] == {"3": 0.0, "4": 0.0} and e["changing"]["histories"] == 3
    assert e["additive_max_gap_ms"] == 0.0
    monkeypatch.setattr(PS, "fields", lambda *a: (None, b0, {(1983, T.BAND): [T.SHORT, T.EDGE]}))   # no history read on every day
    with pytest.raises(SystemExit, match="no first day enters"):
        N.placement(None, 0, {}, {}, {}, None, None)


def test_the_run_covers_both_samples_in_every_placement(monkeypatch):
    calls = []
    monkeypatch.setattr(N, "placement", lambda case, p, *a: calls.append((case, p)) or {"placement": p})
    res = N.run({}, {}, {}, None)
    assert list(res) == ["noise", "travel"] and [e["placement"] for e in res["noise"]] == list(range(20))
    assert calls[:20] == [(None, p) for p in range(20)] and calls[20:] == [(N.CONTROL, p) for p in range(20)]
    assert N.BLOCKS == {"A": tuple(range(10)), "B": tuple(range(10, 20))}
    assert N.CONTROL == {"kind": "travel", "growth": "constant"} and PS.case_name(N.CONTROL) == "travel"     # the constant traveler


def entry(p, rise3=0.1, rise4=0.2, n=30, num=None, den=None, base=40):
    return {"placement": p, "cohort_histories": n, "rise": {"3": rise3, "4": rise4},
            "changing": {"by_day": [0.1 * p] * 5, "days": [p] * 5, "off_grid": [0] * 5, "histories": base + p},
            "matched": {"by_day": [0.0] * 5, "numerator": num or [1.0] * 5, "denominator": den or [4.0] * 5}}


def stored_entry(e):
    c = e["changing"]
    return {"profile": c["by_day"], "profile_days": c["days"], "profile_off_grid": c["off_grid"], "histories": c["histories"], "first_day": 0.0}


def test_every_placement_must_reproduce_the_prestart_record():
    res = {"noise": [entry(p) for p in range(20)], "travel": [entry(p, n=50, base=70) for p in range(20)]}
    rec = {"results": {"noise": [stored_entry(e) for e in res["noise"]], "matched": {"travel": [stored_entry(e) for e in res["travel"]]}}}
    N.check_reproduced(res, rec)
    swapped = {"results": {"noise": rec["results"]["noise"], "matched": {"travel": rec["results"]["noise"]}}}
    with pytest.raises(SystemExit, match="travel placement 0 does not reproduce"):         # the traveler is read against its own stored entries
        N.check_reproduced(res, swapped)
    for name in ("noise", "travel"):
        for field, value in (("by_day", [9.0] * 5), ("days", [99] * 5), ("off_grid", [1] * 5), ("histories", 7)):
            bad = json.loads(json.dumps(res)); bad[name][13]["changing"][field] = value
            with pytest.raises(SystemExit, match=f"{name} placement 13 does not reproduce"):
                N.check_reproduced(bad, rec)
    bad = json.loads(json.dumps(res)); bad["travel"][2]["placement"] = 3
    with pytest.raises(SystemExit, match="travel placement 2 does not reproduce"):
        N.check_reproduced(bad, rec)
    short = json.loads(json.dumps(rec)); short["results"]["noise"] = short["results"]["noise"][:19]
    with pytest.raises(SystemExit, match="noise does not hold one entry per placement"):
        N.check_reproduced(res, short)


def test_the_summary_gives_each_blocks_ranges_and_pooled_curve():
    noise = [entry(p, rise3=0.01 * p, rise4=-0.01 * p, n=20 + p, num=[float(p)] * 5, den=[10.0 + p] * 5) for p in range(20)]
    res = {"noise": noise, "travel": [entry(p) for p in range(20)]}
    s = N.summary(res)
    a, b = s["noise"]["A"], s["noise"]["B"]
    assert a["rise"]["3"] == [0.0, pytest.approx(0.09)] and a["rise"]["4"] == [pytest.approx(-0.09), 0.0]
    assert b["rise"]["3"] == [pytest.approx(0.10), pytest.approx(0.19)] and b["cohort_histories"] == [30, 39] and a["cohort_histories"] == [20, 29]
    assert a["pooled"]["by_day"] == [pytest.approx(45.0 / 145.0)] * 5 and a["pooled"]["histories"] == sum(range(20, 30))   # summed sums, not a mean of ratios
    assert b["pooled"]["by_day"] == [pytest.approx(145.0 / 245.0)] * 5


def ranges_fixture(noise_hi=(0.3, 0.3, 0.3, 0.3), trav_lo=(0.5, 0.5, 0.5, 0.5)):
    """Noise maxima and traveler minima for (day 3, A), (day 3, B), (day 4, A), (day 4, B)."""
    def sample(vals, lo):
        out = {}
        for i, b in enumerate("AB"):
            out[b] = {"rise": {"3": [vals[i], 9.0] if lo else [-9.0, vals[i]], "4": [vals[2 + i], 9.0] if lo else [-9.0, vals[2 + i]]}}
        return out
    return {"noise": sample(noise_hi, False), "travel": sample(trav_lo, True)}


def test_the_verdict_follows_the_declared_rule():
    s = ranges_fixture()
    assert N.verdict({"3": 0.4, "4": 0.4}, s)["archive"] == "above"
    assert N.verdict({"3": 0.2, "4": 0.1}, s)["archive"] == "not separated"
    assert N.verdict({"3": 0.3, "4": 0.3}, s)["archive"] == "not separated"                 # equal to the largest is not above
    assert N.verdict({"3": 0.4, "4": 0.2}, s)["archive"] == "mixed"
    v = N.verdict({"3": 0.4, "4": 0.4}, ranges_fixture(noise_hi=(0.3, 0.5, 0.3, 0.3)))
    assert v["archive"] == "mixed" and v["archive_above"] == {"day 3, block A": True, "day 3, block B": False, "day 4, block A": True, "day 4, block B": True}
    assert N.verdict({"3": 0.4, "4": 0.4}, s)["traveler"] == "separates"
    for lo in ((0.3, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.2)):
        assert N.verdict({"3": 0.4, "4": 0.4}, ranges_fixture(trav_lo=lo))["traveler"] == "does not separate"


def test_the_readings_are_chosen_by_the_verdict():
    base = {"archive_above": {}, "traveler_above": {}}
    r = N.readings(dict(base, archive="above", traveler="separates"))
    assert r[0].startswith("the archive's fixed-cohort rise lies above every placement") and "can tell a rise" in r[1] and len(r) == 3
    r = N.readings(dict(base, archive="not separated", traveler="does not separate"))
    assert "neither of the archive's fixed-cohort rises lies above" in r[0] and "cannot tell" in r[1] and "decides nothing" in r[2] and len(r) == 4
    below = N.verdict({"3": -0.2, "4": -0.2}, ranges_fixture())                              # below every noise placement, not within them
    assert below["archive"] == "not separated" and not any("within" in x for x in N.readings(below))
    r = N.readings(dict(base, archive="above", traveler="does not separate"))
    assert "lies above every placement" in r[0] and "cannot tell" in r[1] and len(r) == 3 and not any("decides nothing" in x for x in r)
    r = N.readings(dict(base, archive="mixed", traveler="separates"))
    assert "placement-dependent and unresolved" in r[0] and len(r) == 3
    assert "leave travel unresolved" in r[-1]


def test_the_archive_values_come_from_the_matched_profile_record():
    rec = {"results": {"matched": {"by_day": [0.173, 0.28, 0.42, 0.583, 0.611]}, "cohort": ["a"] * 60, "changing": {"histories": 86}}}
    a = N.archive_values(rec)
    assert a["cohort_histories"] == 60 and a["changing_histories"] == 86 and a["rise"]["4"] == pytest.approx(0.438)


def test_the_text_carries_the_archive_every_placement_and_each_block():
    res = {"noise": [entry(p, rise3=0.01 * p) for p in range(20)], "travel": [entry(p, rise4=0.5) for p in range(20)]}
    s = N.summary(res); v = N.verdict({"3": 0.4, "4": 0.4}, s)
    archive = {"by_day": [0.173, 0.28, 0.423, 0.583, 0.611], "cohort_histories": 60, "changing_histories": 86, "rise": {"3": 0.41, "4": 0.438}}
    t = N.text(res, s, v, archive)
    assert "| archive | | | 86 | 60 | 0.173 | 0.280 | 0.423 | 0.583 | 0.611 | 0.410 | 0.438 |" in t
    assert "| noise | 13 | B | 53 | 30 |" in t and "| travel | 0 | A |" in t and "| noise | B | 30 to 30 | 0.100, 0.190 |" in t
    assert f"Archive: {v['archive']}. Traveler: {v['traveler']}." in t


def test_main_gates_writes_and_never_overwrites(tmp_path, monkeypatch):
    gen = T.gen_fixture()
    res = {"noise": [entry(p) for p in range(20)], "travel": [entry(p, rise3=0.9, rise4=0.9) for p in range(20)]}
    pre = {"generated_by": dict(gen, sha256=PS.SCRIPT_SHA256_AT_LOAD),
           "results": {"noise": [stored_entry(e) for e in res["noise"]], "matched": {"travel": [stored_entry(e) for e in res["travel"]]}}}
    mat = {"generated_by": dict(gen, sha256=M.SCRIPT_SHA256_AT_LOAD),
           "results": {"matched": {"by_day": [0.1, 0.2, 0.3, 0.6, 0.7]}, "cohort": ["a"] * 60, "changing": {"histories": 86}}}
    pp, mp = tmp_path / "pre.json", tmp_path / "mat.json"
    def write(p=pre, m=mat):
        pp.write_text(json.dumps(p)); mp.write_text(json.dumps(m))
    write()
    monkeypatch.setattr(N.C, "load_inputs", lambda a: ({}, {}, {}, {}, dict(gen)))
    monkeypatch.setattr(N.P, "measured_parts", lambda: None)
    monkeypatch.setattr(N, "run", lambda tracks, by, std, parts: json.loads(json.dumps(res)))
    args = ["--shape", "s", "--inject", "i", "--records", "r", "--histories", "h", "--xband", "x", "--prestart-constructions", str(pp),
            "--matched-archive", str(mp), "--out-dir", str(tmp_path)]
    assert N.main(args) == 0
    out = json.loads((tmp_path / "eval60_matched_noise.json").read_text())
    assert out["results"]["verdict"]["archive"] == "above" and out["results"]["verdict"]["traveler"] == "separates"
    assert out["generated_by"]["sha256"] == N.SCRIPT_SHA256_AT_LOAD and out["generated_by"]["declared"] == "EASTERN_MATCHED_NOISE_2026-10-23.md"
    assert out["generated_by"]["prestart_constructions"]["sha256"] == N.sha256_of(str(pp)) and out["results"]["archive"]["cohort_histories"] == 60
    assert (tmp_path / "eval60_matched_noise.md").read_text().startswith("# fixed-cohort daily profile against matched noise")
    with pytest.raises(SystemExit, match="never overwritten"):
        N.main(args)
    for tag, p, m, match in (("_a", dict(pre, generated_by=dict(pre["generated_by"], sha256="x")), mat, "pre-start constructions were made by a different"),
                             ("_b", pre, dict(mat, generated_by=dict(mat["generated_by"], sha256="x")), "matched-profile record was made by a different"),
                             ("_c", dict(pre, generated_by=dict(pre["generated_by"], xband={"x": "other"})), mat, "the xband input differs"),
                             ("_d", pre, dict(mat, generated_by=dict(mat["generated_by"], shape={"path": "a", "sha256": "other"})), "the shape input differs")):
        write(p, m)
        with pytest.raises(SystemExit, match=match):
            N.main(args + ["--tag", tag])
    write()
    bad = json.loads(json.dumps(res)); bad["noise"][4]["changing"]["histories"] = 1
    monkeypatch.setattr(N, "run", lambda tracks, by, std, parts: bad)
    with pytest.raises(SystemExit, match="noise placement 4 does not reproduce"):
        N.main(args + ["--tag", "_e"])
    assert not (tmp_path / "eval60_matched_noise_e.json").exists()
