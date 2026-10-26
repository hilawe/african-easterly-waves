"""Tests for the known rise of the fixed-cohort daily profile (`EASTERN_KNOWN_RISE_2026-10-24.md`)."""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import eastern_known_rise as K  # noqa: E402
import test_eastern_matched_noise as TN  # noqa: E402
import test_eastern_matched_profile as T  # noqa: E402

PS, M, N, H, P, J, A = K.PS, K.M, K.N, K.H, K.P, K.J, K.A


def parts():
    prof = np.zeros(len(H.DX)); prof[H.DX == 0] = -4.0; prof[H.DX == -1.0] = -2.0; prof[H.DX == 6.0] = 1.6; prof[H.DX == 3.0] = 1.0
    H.MEASURED["profile"] = prof.tolist()
    return P.measured_parts()


TRACKS = [{"first_lon": 60.0, "speed": -2.0, "days": 5, "injected": {"measured": 6.0}},
          {"first_lon": 55.0, "speed": -3.0, "days": 4, "injected": {"measured": 4.0}}]


def test_the_stored_growth_is_the_prestart_tests_growing_traveler():
    pt = parts()
    V, centers, starts = K.build(TRACKS, np.random.default_rng([1, 2, 3]), PS.GROW_DAYS, 1.0, pt)
    W, wc = PS.build(TRACKS, np.random.default_rng([1, 2, 3]), {"kind": "travel", "growth": "lobe_zero"}, pt)
    assert np.array_equal(V, W) and all(np.array_equal(a, b, equal_nan=True) for a, b in zip(centers, wc))
    rng = np.random.default_rng([1, 2, 3]); expected = [int(rng.integers(0, A.SEASON_DAYS - tr["days"] + 1)) for tr in TRACKS]
    assert starts == expected and all(np.isfinite(c[s]) and (s == 0 or not np.isfinite(c[s - 1])) for c, s in zip(centers, starts))


def test_the_lobe_grows_from_zero_to_its_multiple_over_the_growth_time():
    pt = parts()
    one = [TRACKS[0]]
    V, _, (t0,) = K.build(one, np.random.default_rng([4]), 2.0, 1.5, pt)
    k0 = t0 * K.L.OBS_PER_DAY; t = J.times()[:, 0]; x0 = TRACKS[0]["first_lon"]
    core = P.core_shape(K.XL, x0, 6.0, pt["core"]); full = H.shape("measured", K.XL, x0, 6.0)
    assert np.allclose(V[k0], core)                                                        # nothing grown at the start
    k = k0 + int(2.0 * K.L.OBS_PER_DAY); x = x0 + TRACKS[0]["speed"] * (t[k] - t0)
    core = P.core_shape(K.XL, x, 6.0, pt["core"]); full = H.shape("measured", K.XL, x, 6.0)
    assert np.allclose(V[k], core + 1.5 * (full - core))                                   # the whole lobe, times 1.5, after 2 days
    half = k0 + K.L.OBS_PER_DAY; x = x0 + TRACKS[0]["speed"] * (t[half] - t0)
    core = P.core_shape(K.XL, x, 6.0, pt["core"]); full = H.shape("measured", K.XL, x, 6.0)
    assert np.allclose(V[half], core + 1.5 * 0.5 * (full - core))


def test_the_clean_background_is_zero_in_every_season():
    bg = K.zero_background(3.0, 7)
    assert len(bg) == len(K.YEARS) and all(b.shape == (J.N_SAMPLES, len(J.LONS)) and not b.any() for b in bg)


def test_fields_keep_the_histories_that_follow_a_trough_with_their_lag(monkeypatch):
    c = np.full(A.SEASON_DAYS, np.nan); c[20:25] = [60.0, 58.0, 56.0, 54.0, 52.0]
    monkeypatch.setattr(K, "build", lambda tracks, rng, g, m, pt: (np.zeros((J.N_SAMPLES, len(K.XL))), [c], [19]))
    on = {"days": [20, 21, 22, 23, 24], "lons": [60.0, 58.0, 56.0, 54.0, 52.0]}
    off = {"days": [40, 41, 42, 43], "lons": [60.0, 58.0, 56.0, 54.0]}
    monkeypatch.setattr(H, "detected", lambda S, month_of: ([on, off], None))
    monkeypatch.setattr(P, "b0_rows", lambda Hb, month_of: {y: "b0" for y in Hb})
    b0, hist, lags = K.fields(K.zero_background, 0, {}, 2.0, 1.0, {b: 1.0 for b in K.L.BAND_LAT}, None, None)
    assert set(hist) == {(y, b) for y in K.YEARS for b in K.BANDS} and all(v == [on] for v in hist.values())
    assert lags[id(on)] == 1 and id(off) not in lags                                      # first member a day after the trough's start
    draws = []
    monkeypatch.setattr(K, "build", lambda tracks, rng, g, m, pt: (draws.append(int(rng.integers(0, 10 ** 9))) or (np.zeros((J.N_SAMPLES, len(K.XL))), [c], [19])))
    K.fields(K.zero_background, 7, {}, 2.0, 1.0, {b: 1.0 for b in K.L.BAND_LAT}, None, None)
    bi = list(K.L.BAND_LAT).index(K.BANDS[0])                                              # the pre-start test's seeds, band then placement then season
    assert draws[0] == int(np.random.default_rng([P.LOBE_SEED, bi, 7, 0]).integers(0, 10 ** 9))
    assert draws[1] == int(np.random.default_rng([P.LOBE_SEED, bi, 7, 1]).integers(0, 10 ** 9))


def test_a_placement_reads_the_matched_measure_and_counts_the_cohorts_lags(monkeypatch):
    b0, hist = T.fixture()
    monkeypatch.setattr(M, "BANDS", (T.BAND,)); monkeypatch.setattr(K, "BANDS", (T.BAND,))
    lags = {id(T.LONG): 0, id(T.SHORT): 2, id(T.EDGE): -1}
    monkeypatch.setattr(K, "fields", lambda *a: (b0, hist, lags))
    e = K.placement(None, 3, {}, 2.0, 1.0, {}, None, None)
    assert e["placement"] == 3 and e["cohort_histories"] == 1 and e["cohort_lag_days"] == {"0": 1} and e["rise"] == {"3": 0.0, "4": 0.0}
    lags[id(T.LONG)] = -2
    assert K.placement(None, 3, {}, 2.0, 1.0, {}, None, None)["cohort_lag_days"] == {"-2": 1}


def pooled_entries(rise_by_block):
    """Twenty placements whose cohort curves pool to a chosen rise in each block."""
    out = []
    for p in range(20):
        r = rise_by_block["A" if p < 10 else "B"]
        out.append({"placement": p, "cohort_histories": 50, "rise": {"3": r, "4": r}, "cohort_lag_days": {"0": 49, "1": 1},
                    "changing": {"by_day": [0.1] * 5, "days": [50] * 5, "off_grid": [0] * 5, "histories": 70},
                    "matched": {"by_day": [0.25, 0, 0, 0.25 + r, 0.25 + r], "numerator": [1.0, 1.0, 1.0, 4.0 * (0.25 + r), 4.0 * (0.25 + r)],
                                "denominator": [4.0] * 5}})
    return out


def test_adequacy_needs_the_pooled_rise_to_reach_the_archives_in_both_blocks_on_both_days():
    arch = {"3": 0.25, "4": 0.5}
    assert K.adequate(arch, K.block_summary(pooled_entries({"A": 0.5, "B": 0.5})))["adequate"] is True     # equal reaches
    ad = K.adequate(arch, K.block_summary(pooled_entries({"A": 0.5, "B": 0.375})))
    assert ad["adequate"] is False and ad["reaches"] == {"day 3, block A": True, "day 3, block B": True, "day 4, block A": True, "day 4, block B": False}
    high = pooled_entries({"A": 0.375, "B": 0.375}); high[3]["rise"] = {"3": 0.9, "4": 0.9}             # one placement above, the pooled curve below
    assert K.adequate(arch, K.block_summary(high))["reaches"]["day 4, block A"] is False


def test_the_stored_growth_must_reproduce_the_prestart_record():
    entries = [TN.entry(p) for p in range(20)]
    rec = {"results": {"matched": {K.STORED_CASE: [TN.stored_entry(e) for e in entries]}}}
    K.check_reproduced(entries, rec)
    for field, value in (("by_day", [9.0] * 5), ("days", [99] * 5), ("off_grid", [1] * 5), ("histories", 7)):
        bad = json.loads(json.dumps(entries)); bad[6]["changing"][field] = value
        with pytest.raises(SystemExit, match="placement 6 does not reproduce"):
            K.check_reproduced(bad, rec)
    with pytest.raises(SystemExit, match="one entry per placement"):
        K.check_reproduced(entries[:19], rec)
    other = {"results": {"matched": {K.STORED_CASE: rec["results"]["matched"][K.STORED_CASE], "travel": []}}}
    K.check_reproduced(entries, other)


def noise_record(entries):
    res = {"noise": entries, "travel": entries}
    return {"results": {"placements": {"noise": entries}, "summary": N.summary(res)}}


def test_the_noise_reading_uses_the_stored_noise_and_this_traveler():
    noise = pooled_entries({"A": 0.1, "B": 0.1})
    ctl = pooled_entries({"A": 0.6, "B": 0.6})
    r = K.noise_reading({"3": 0.4, "4": 0.4}, noise_record(noise), ctl)
    assert r["verdict"]["traveler"] == "separates" and r["verdict"]["archive"] == "above" and r["verdict"]["traveler_rise"] == "reaches the archive's"
    assert r["summary"]["A"]["pooled"]["by_day"][3] == pytest.approx(0.85) and "growing traveler separates" in r["readings"][1]
    rec = noise_record(noise); rec["results"]["summary"]["noise"]["A"]["cohort_histories"] = [1, 2]
    with pytest.raises(SystemExit, match="do not give the noise record's summary"):
        K.noise_reading({"3": 0.4, "4": 0.4}, rec, ctl)


def test_the_readings_for_the_growing_traveler():
    v = TN.verdict_fixture(archive="mixed", traveler="separates", traveler_rise="reaches the archive's")
    r = K.readings(v, "ARCHIVE")
    assert r[0] == "ARCHIVE" and "separates completely from this noise" in r[1] and "in all 4 block-and-day checks" in r[1] and "pass 3 of those checks" in r[1] and len(r) == 3
    r = K.readings(dict(v, traveler="does not separate"), "ARCHIVE")
    assert "fails complete separation" in r[1] and "does not calibrate the archive's physical interpretation" in r[2] and len(r) == 4
    r = K.readings(dict(v, traveler="does not separate", traveler_rise="below the archive's"), "ARCHIVE")
    assert "does not establish insensitivity" in r[1] and "attenuates" in r[1] and not any("does not calibrate" in x for x in r)
    r = K.readings(dict(v, archive="above", traveler="does not separate"), "ARCHIVE")
    assert not any("does not calibrate" in x for x in r) and "constructed traveler" in r[-1] and "leave travel unresolved" in r[-1]
    assert not any("constant" in x for x in K.readings(v, "ARCHIVE"))


def test_no_reading_says_the_comparison_distinguishes_nothing():
    for archive in ("above", "mixed", "not separated"):
        for trav in ("separates", "does not separate"):
            for rise in ("reaches the archive's", "below the archive's"):
                v = TN.verdict_fixture(archive=archive, traveler=trav, traveler_rise=rise)
                for r in (K.readings(v, N.readings(v)[0]), N.readings(v)):
                    assert not any(w in x for x in r for w in ("cannot tell", "decides nothing", "can tell"))


def test_main_clean_and_noise(tmp_path, monkeypatch):
    gen = T.gen_fixture()
    mat = {"generated_by": dict(gen, sha256=M.SCRIPT_SHA256_AT_LOAD),
           "results": {"matched": {"by_day": [0.1, 0.2, 0.3, 0.5, 0.6]}, "cohort": ["a"] * 60, "changing": {"histories": 86}}}
    mp = tmp_path / "mat.json"; mp.write_text(json.dumps(mat))
    monkeypatch.setattr(K.C, "load_inputs", lambda a: ({}, {}, {}, {}, dict(gen)))
    monkeypatch.setattr(P, "measured_parts", lambda: None)
    calls = []
    clean_entries, stored_entries, frozen_entries = pooled_entries({"A": 0.5, "B": 0.5}), pooled_entries({"A": 0.3, "B": 0.3}), pooled_entries({"A": 0.7, "B": 0.7})
    def fake_run(bg_fn, tracks, g, m, std, pt):
        calls.append((bg_fn.__name__, g, m))
        if bg_fn is K.zero_background:
            return json.loads(json.dumps(clean_entries))
        return json.loads(json.dumps(stored_entries if (g, m) == (PS.GROW_DAYS, 1.0) else frozen_entries))
    monkeypatch.setattr(K, "run", fake_run)
    base = ["--shape", "s", "--inject", "i", "--records", "r", "--histories", "h", "--xband", "x", "--matched-archive", str(mp), "--out-dir", str(tmp_path)]
    assert K.main(["clean"] + base + ["--lobe-mult", "1.3"]) == 0
    cj = tmp_path / "eval60_known_rise_clean_g2_m1.3.json"
    out = json.loads(cj.read_text())
    assert out["results"]["adequacy"]["adequate"] is True and out["generated_by"]["lobe_mult"] == 1.3 and calls == [("zero_background", 2.0, 1.3)]
    with pytest.raises(SystemExit, match="never overwritten"):
        K.main(["clean"] + base + ["--lobe-mult", "1.3"])
    with pytest.raises(SystemExit, match="must be positive"):
        K.main(["clean"] + base + ["--lobe-mult", "0"])
    pre = {"generated_by": dict(gen, sha256=PS.SCRIPT_SHA256_AT_LOAD), "results": {"matched": {K.STORED_CASE: [TN.stored_entry(e) for e in stored_entries]}}}
    noise = noise_record(pooled_entries({"A": 0.1, "B": 0.1})); noise["generated_by"] = dict(gen, sha256=N.SCRIPT_SHA256_AT_LOAD)
    pp, np_ = tmp_path / "pre.json", tmp_path / "noise.json"; pp.write_text(json.dumps(pre)); np_.write_text(json.dumps(noise))
    nargs = ["noise"] + base + ["--prestart-constructions", str(pp), "--noise-archive", str(np_), "--clean", str(cj)]
    calls.clear()
    assert K.main(nargs) == 0
    rec = json.loads((tmp_path / "eval60_known_rise_noise.json").read_text()); res = rec["results"]
    assert calls == [("matched_background", 2.0, 1.0), ("matched_background", 2.0, 1.3)] and rec["generated_by"]["lobe_mult"] == 1.3
    assert res["verdict"]["traveler"] == "separates" and res["stored_case_placements"][0]["rise"]["3"] == 0.3 and res["placements"][0]["rise"]["3"] == 0.7
    assert rec["generated_by"]["clean"]["sha256"] == K.sha256_of(str(cj))
    assert (tmp_path / "eval60_known_rise_noise.md").read_text().startswith("# known rise on matched noise, growth over 2 days, lobe times 1.3")
    for tag, edit, match in (("_a", lambda: pp.write_text(json.dumps(dict(pre, generated_by=dict(pre["generated_by"], sha256="x")))), "pre-start constructions"),
                             ("_b", lambda: np_.write_text(json.dumps(dict(noise, generated_by=dict(noise["generated_by"], sha256="x")))), "noise record was made"),
                             ("_c", lambda: np_.write_text(json.dumps(dict(noise, generated_by=dict(noise["generated_by"], xband={"x": "o"})))), "the xband input differs")):
        pp.write_text(json.dumps(pre)); np_.write_text(json.dumps(noise)); edit()
        with pytest.raises(SystemExit, match=match):
            K.main(nargs + ["--tag", tag])
    pp.write_text(json.dumps(pre)); np_.write_text(json.dumps(noise))
    weak = json.loads(cj.read_text()); weak["results"]["adequacy"]["adequate"] = False; wp = tmp_path / "weak.json"; wp.write_text(json.dumps(weak))
    with pytest.raises(SystemExit, match="does not produce a known rise"):
        K.main(nargs[:-1] + [str(wp), "--tag", "_d"])
    stale = json.loads(cj.read_text()); stale["generated_by"]["sha256"] = "x"; sp = tmp_path / "stale.json"; sp.write_text(json.dumps(stale))
    with pytest.raises(SystemExit, match="clean record was made by a different version"):
        K.main(nargs[:-1] + [str(sp), "--tag", "_e"])
    stored_entries[2]["changing"]["histories"] = 1
    with pytest.raises(SystemExit, match="placement 2 does not reproduce"):
        K.main(nargs + ["--tag", "_f"])
    with pytest.raises(SystemExit, match="are required for the noise run"):
        K.main(["noise"] + base)
    mp.write_text(json.dumps(dict(mat, generated_by=dict(mat["generated_by"], sha256="x"))))
    with pytest.raises(SystemExit, match="matched-profile record was made by a different version"):
        K.main(["clean"] + base + ["--lobe-mult", "1.4"])
