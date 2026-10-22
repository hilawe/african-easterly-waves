"""Tests for the injection-recovery test (`EASTERN_INJECTION_RECOVERY_2026-10-21.md`)."""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import eastern_injection_recovery as J  # noqa: E402

L = J.L


def test_the_injection_is_exact_and_a_zero_field_returns_the_background():
    S = np.array([[1.23, -0.07], [12.5, 0.0]])
    H = J.hundredths(S)
    assert np.array_equal(H, np.array([[123.0, -7.0], [1250.0, 0.0]]))
    assert np.array_equal(J.inject(H, np.zeros_like(S)), S)
    assert np.array_equal(J.inject(H, np.array([[0.004, 0.006], [-1.0, 2.345]])), np.array([[1.23, -0.06], [11.5, 2.35]]))
    with pytest.raises(SystemExit, match="not an exact hundredth"):
        J.hundredths(np.array([1.235]))


def test_the_train_moves_west_at_its_speed_and_bursts_never_overlap():
    V = J.train_field(2.0, 7.0, 0.3)
    assert V.shape == (J.N_SAMPLES, len(J.LONS))
    x = J.LONS
    assert np.allclose(V[0], 2.0 * np.sin(2 * np.pi * x / (7.0 * J.TRAIN_PERIOD) + 0.3))
    assert np.allclose(V[4, :-14], V[0, 14:])           # a day later the pattern sits 7 degrees farther west
    assert not np.allclose(V[4, 14:], V[0, :-14])
    V5 = J.train_field(1.0, 5.0, 0.0)
    assert np.allclose(V5[0], np.sin(2 * np.pi * x / (5.0 * J.TRAIN_PERIOD)))
    for kb in J.BURST_COUNTS:
        env = J.burst_envelope(np.random.default_rng(1), kb)
        on = env > 0
        starts = np.flatnonzero(on & ~np.r_[False, on[:-1]])
        assert len(starts) == kb and env.max() <= 1.0
        assert np.isclose(env.sum() / 4.0, kb * J.BURST_DAYS / 2.0)
        assert np.all(np.diff(starts) >= 4 * J.BURST_DAYS)
    env = J.burst_envelope(np.random.default_rng(1), 3)
    V2 = J.train_field(1.0, 7.0, 0.0, envelope=env)
    assert np.array_equal(V2[env == 0], np.zeros_like(V2[env == 0]))


def test_a_packet_runs_its_track_for_its_days_at_its_depth():
    tr = {"first_lon": 60.0, "speed": -2.0, "days": 5, "injected": {"2": 6.0, "4": 7.0}}
    V = J.packet_field([tr], np.random.default_rng(3), 1, 1, 2.0)
    rows = np.flatnonzero(np.abs(V).max(axis=1) > 0)
    assert len(rows) == 5 * 4 and np.all(np.diff(rows) == 1)
    t0 = rows[0] / 4.0
    assert t0 == int(t0)
    for k in (rows[0], rows[0] + 8, rows[-1]):
        tk = k / 4.0
        assert np.isclose(J.LONS[np.argmin(V[k])], 60.0 - 2.0 * (tk - t0), atol=0.25)
        assert np.isclose(V[k].min(), -6.0, atol=0.05)
    V2 = J.packet_field([tr], np.random.default_rng(3), 2, 3, 4.0)  # three copies at twice the wide depth
    active = int((np.abs(V2).max(axis=1) > 0).sum())
    assert 5 * 4 <= active <= 3 * 5 * 4 and V2.min() <= -14.0 + 0.05 and V2.max() <= 0.0
    V1 = J.packet_field([tr], np.random.default_rng(4), 2, 1, 4.0)
    assert np.isclose(V2.sum(), 3 * V1.sum())
    k = int(np.flatnonzero(np.abs(V1).max(axis=1) > 0)[0])
    c = int(np.argmin(V1[k]))
    assert np.isclose(V1[k, c + 8] / V1[k, c], np.exp(-0.5))     # 4 degrees from the center at width 4


def test_the_calibration_is_the_reference_ratio_and_slow_wide_tracks_take_the_floor(monkeypatch):
    assert round(J.calibration_ratio(J.REFERENCE_TRACK, 2.0), 4) == 0.7341
    assert round(J.calibration_ratio(J.REFERENCE_TRACK, 4.0), 4) == 0.6229
    slow = {"first_lon": 53.0, "speed": -7.0 / 6.0, "days": 10}
    assert J.calibration_ratio(slow, 4.0) is None
    assert J.calibration_ratio(slow, 2.0) is not None
    short = {"first_lon": 32.5, "speed": -3.0, "days": 4}       # a minimum on 3 of its 4 days
    assert J.calibration_ratio(short, 2.0) is not None
    h_fast = {"lons": [58.0, 55.5, 53.0, 50.5, 48.0, 45.5], "depths": [-4.0] * 6, "days": list(range(6))}
    h_faster = {"lons": [62.0, 58.5, 55.0, 51.5, 48.0], "depths": [-5.0] * 5, "days": list(range(5))}
    h_slow = {"lons": [53.0 - 7.0 / 6.0 * k for k in range(10)], "depths": [-3.0] * 10, "days": list(range(10))}
    tr = J.band_tracks({(1983, "b"): [h_fast, h_faster], (1984, "b"): [h_slow]})
    fast, faster, slow_t = tr["b"][1983][0], tr["b"][1983][1], tr["b"][1984][0]
    assert not fast["ratio_4_fallback"] and slow_t["ratio_4_fallback"]
    assert fast["ratio"]["4"] != faster["ratio"]["4"]
    assert slow_t["ratio"]["4"] == min(fast["ratio"]["4"], faster["ratio"]["4"])
    assert np.isclose(fast["injected"]["2"], 4.0 / fast["ratio"]["2"])
    monkeypatch.setattr(J, "calibration_ratio", lambda track, width, month_of=None: None)
    with pytest.raises(SystemExit, match="no calibration at the declared width"):
        J.band_tracks({(1983, "b"): [h_fast]})


def test_the_limit_needs_four_of_five_at_every_larger_value():
    assert J.limit([1, 2, 3, 4], [0, 3, 4, 5]) == 3
    assert J.limit([1, 2, 3, 4], [4, 4, 4, 4]) == 1
    assert J.limit([1, 2, 3, 4], [0, 4, 3, 5]) == "not monotone"
    assert J.limit([1, 2, 3, 4], [0, 1, 2, 3]) is None


def entry(cls="standing", slope=0.0, reaches=False, moving=False, r025=False, r02=False):
    return {"class": cls, "slope": slope, "westmost": 50.0, "positive_lag_days": 1.0, "reaches": reaches, "moving": moving,
            "reaches_025": r025, "reaches_02": r02, "class_02": cls}


def test_the_reading_rests_on_60_65_and_70_e_at_each_threshold():
    r = {f"{b:g}": entry() for b in J.BASES}
    r["55"] = entry("westward", -3.0, reaches=True, moving=True, r025=True, r02=True)
    assert J.passes(r) == {"0.3": False, "0.25": False, "0.2": False}
    r["65"] = entry("westward", -3.0, reaches=False, moving=True, r025=True, r02=True)
    assert J.passes(r) == {"0.3": False, "0.25": True, "0.2": True}
    r["70"] = entry("westward", -6.0, reaches=True, moving=True)
    assert J.passes(r)["0.3"]


def test_read_reduces_the_diagnostic_to_the_declared_readers():
    seasons = [L.construction_samples("R2", i) for i in range(len(J.YEARS))]
    r = J.read(seasons, 17.25)
    assert set(r) == {f"{b:g}" for b in J.BASES}
    for v in r.values():
        assert set(v) == {"class", "slope", "westmost", "positive_lag_days", "reaches", "moving", "reaches_025", "reaches_02", "class_02"}
        assert v["moving"] == (v["slope"] is not None and v["slope"] <= J.MOVING_SLOPE)
    assert r["70"]["class"] == "westward" and r["70"]["reaches"]


def test_a_slope_of_exactly_minus_one_is_moving():
    seasons = [L.construction_samples("R6", i) for i in range(len(J.YEARS))]
    real = J.L.summarize
    def fake(fs, bases, lat, null_bases):
        out = real(fs, bases=bases, lat=lat, null_bases=null_bases)
        out["45"]["slope_deg_per_day"] = -1.0
        out["50"]["slope_deg_per_day"] = -0.999
        out["60"]["sensitivity"]["0.25"].update({"slope_deg_per_day": -5.0, "positive_lag_days": 3.0, "westmost": 40.0})
        return out
    J.L.summarize = fake
    try:
        r = J.read(seasons, 17.25)
    finally:
        J.L.summarize = real
    assert r["45"]["moving"] and not r["50"]["moving"]
    assert r["60"]["reaches_025"] and not r["60"]["reaches_02"]


def test_configurations_and_limits_of():
    cfgs = J.configurations()
    assert len(cfgs) == 15 + 8 + 8 and len({c["name"] for c in cfgs}) == len(cfgs)
    def summ(passes03=0, west=None, mov=None):
        west = west or {}; mov = mov or {}
        return {"reading_passes": {"0.3": passes03, "0.25": 0, "0.2": 0},
                "bases": {f"{b:g}": {"westward": west.get(b, 0), "moving": mov.get(b, 0), "median_slope": -1.0, "median_westmost": 50.0} for b in J.BASES}}
    rows = []
    for c in cfgs:
        if c["part"] == "A":
            rows.append({"config": c, "summary": summ(5 if (c["speed"] == 7.0 and c["amp"] >= 2.0) else 0)})
        elif c["part"] == "A2":
            rows.append({"config": c, "summary": summ(0)})
        else:
            m = {"P1": {}, "P2": {}, "P3": {55.0: 4}, "P4": {55.0: 5}, "C2": {60.0: 4}, "C4": {60.0: 4}, "C8": {60.0: 5}, "W1": {}}[c["name"]]
            w = {"P4": {55.0: 4}}.get(c["name"], {})
            rows.append({"config": c, "summary": summ(0, west=w, mov=m)})
    lim = J.limits_of(rows)
    assert lim["train 7"] == 2.0 and lim["train 5"] is None and lim["bursts 3"] is None
    assert lim["packets depth, moving"] == 3 and lim["packets depth, westward"] == 4
    assert lim["packets count, moving"] == 2 and lim["packets count, westward"] is None
    assert lim["packets wide, moving"] is False


def test_the_readings_name_their_limits_and_bases():
    def bg(lim, p1):
        return {"limits": lim, "rows": [{"config": {"name": "P1"}, "summary": p1}]}
    lim = {f"train {c:g}": v for c, v in zip(J.TRAIN_SPEEDS, (None, 2.0, "not monotone"))}
    p1 = {"bases": {f"{b:g}": {"westward": 4 if b == 55.0 else 0, "moving": 5 if b in (50.0, 55.0) else 3} for b in J.BASES}}
    res = {"bands": {"b": {"packets": 7, "backgrounds": {"archive": bg(lim, p1), "matched": bg(lim, p1)}}}}
    s = J.readings(res)["b"]
    assert "a continuous train at 5 degrees per day fires the reading at no amplitude tested" in s[0]
    assert "from 2 m/s" in s[1] and "not monotonically" in s[2]
    assert "band's 7 recorded history tracks" in s[3] and "westward class at 55 E" in s[3] and "moving ridge at 50, 55 E" in s[3]


def construction_fixture():
    base = {f"{b:g}": {"westward": 5, "moving": 5, "median_slope": -2.5, "median_westmost": 45.0} for b in J.BASES}
    return {"K1": {"1": {"0.3": False, "0.25": False, "0.2": False}, "2": {"0.3": True, "0.25": True, "0.2": True}},
            "K2": {"identical": True},
            "K3": {"15 to 19.5 N": {"reading_passes": {}, "bases": json.loads(json.dumps(base))},
                   "20 to 24.5 N": {"reading_passes": {}, "bases": json.loads(json.dumps(base))}},
            "K4": {"2": 0.7341, "4": 0.6229}}


def test_the_construction_rule_and_each_failure():
    c = construction_fixture()
    assert J.apply_rule(c) == {"adequate": True, "failures": []}
    for change, msg in (
        (lambda c: c["K1"]["1"].__setitem__("0.2", True), "K1 reaches at 1 m/s"),
        (lambda c: c["K1"]["2"].__setitem__("0.25", False), "K1 does not reach"),
        (lambda c: c["K2"].__setitem__("identical", False), "K2"),
        (lambda c: [c["K3"]["15 to 19.5 N"]["bases"][b].__setitem__("moving", 3) for b in ("45", "50", "55")], "K3 15 to 19.5 N moving at 3"),
        (lambda c: [v.__setitem__("westward", 3) for v in c["K3"]["20 to 24.5 N"]["bases"].values()], "K3 20 to 24.5 N moving at 6 bases, westward at 0"),
        (lambda c: c["K4"].__setitem__("4", 1.0), "K4 ratio at width 4"),
        (lambda c: c["K4"].__setitem__("2", 0.5), "K4 ratio at width 2"),
    ):
        cc = construction_fixture(); change(cc)
        v = J.apply_rule(cc)
        assert not v["adequate"] and any(msg in f for f in v["failures"]), (msg, v)


def test_the_constructions_gate(tmp_path):
    res = construction_fixture()
    p = tmp_path / "c.json"
    def write(results, verdict, sha=J.SCRIPT_SHA256_AT_LOAD):
        p.write_text(json.dumps({"generated_by": {"sha256": sha}, "results": results, "verdict": verdict}))
    write(res, J.apply_rule(res))
    assert J.check_constructions(str(p), {}, rerun=lambda tr: res)["results"] == res
    write(res, J.apply_rule(res), sha="0" * 64)
    with pytest.raises(SystemExit, match="different version"):
        J.check_constructions(str(p), {}, rerun=lambda tr: res)
    write(res, J.apply_rule(res))
    other = construction_fixture(); other["K4"]["2"] = 0.7
    with pytest.raises(SystemExit, match="differs from a fresh run"):
        J.check_constructions(str(p), {}, rerun=lambda tr: other)
    write(res, {"adequate": True, "failures": ["x"]})
    with pytest.raises(SystemExit, match="stored summary disagrees"):
        J.check_constructions(str(p), {}, rerun=lambda tr: res)
    bad = construction_fixture(); bad["K2"]["identical"] = False
    write(bad, J.apply_rule(bad))
    with pytest.raises(SystemExit, match="do not pass the rule"):
        J.check_constructions(str(p), {}, rerun=lambda tr: bad)


def test_the_zero_injection_gate(monkeypatch):
    seasons = {y: L.construction_samples("R1", i) for i, y in enumerate(J.YEARS)}
    monkeypatch.setattr(L, "BAND_LAT", {"b": 17.25})
    monkeypatch.setattr(L, "season_samples", lambda recs, band: recs)
    by = dict(seasons)
    res = L.annotate(L.summarize([L.filtered(seasons[y]) for y in J.YEARS], bases=J.BASES, lat=17.25, null_bases=()))
    stored = json.loads(json.dumps(res))
    for v in stored.values():
        v["shift_max_r"] = {"median": 0.1, "max": 0.2}
    lag = {"generated_by": {"sha256": L.SCRIPT_SHA256_AT_LOAD}, "results": {"b": {"bases": stored}}}
    J.check_zero_injection(by, lag)
    lag["generated_by"]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="different version of its script"):
        J.check_zero_injection(by, lag)
    lag["generated_by"]["sha256"] = L.SCRIPT_SHA256_AT_LOAD
    stored["60"]["westmost"] = stored["60"]["westmost"] - 0.5
    with pytest.raises(SystemExit, match="b base 60 with nothing injected differs"):
        J.check_zero_injection(by, lag)


def test_each_placement_and_background_draws_its_own_field(monkeypatch):
    seen = []
    monkeypatch.setattr(J, "read", lambda seasons, lat: (seen.append(seasons[0].copy()), {f"{b:g}": entry() for b in J.BASES})[1])
    bg = [[np.zeros((J.N_SAMPLES, len(J.LONS))) for _ in J.YEARS] for _ in range(J.PLACEMENTS)]
    cfg = {"part": "A", "name": "t", "speed": 7.0, "amp": 1.0}
    J.run_config(0, cfg, bg, {}, 0, 17.25, 0)
    J.run_config(0, cfg, bg, {}, 0, 17.25, 1)
    assert len(seen) == 2 * J.PLACEMENTS
    assert all(not np.array_equal(seen[a], seen[b]) for a in range(len(seen)) for b in range(a))
    seen.clear(); J.run_config(0, cfg, bg, {}, 0, 17.25, 0)
    first = list(seen); seen.clear(); J.run_config(0, cfg, bg, {}, 0, 17.25, 0)
    assert all(np.array_equal(a, b) for a, b in zip(first, seen))


def test_matched_noise_is_scaled_and_differs_by_placement():
    u = J.unit_noise_scale()
    assert round(u, 3) == 0.519
    a, b = J.matched_background(2.0, 0), J.matched_background(2.0, 1)
    raw0 = np.round(J.A.noise_field(2.0, seed=J.HI.NOISE_SEEDS[0], n_steps=J.N_SAMPLES, lons=J.LONS), 2)
    assert not np.array_equal(a[0] / 100.0, raw0) and not np.array_equal(a[0], b[0])
    s = J.pooled_filtered_std([L.filtered(H / 100.0) for H in a], [J.col(x) for x in J.SCALE_BASES])
    assert abs(float(np.mean(s)) - 2.0 * u) < 0.1


def test_the_archive_loop_scales_the_matched_background_from_50_to_70_e(monkeypatch):
    seasons = {y: L.construction_samples("R1", i) for i, y in enumerate(J.YEARS)}
    monkeypatch.setattr(L, "BAND_LAT", {"b": 17.25})
    monkeypatch.setattr(L, "season_samples", lambda recs, band: recs)
    monkeypatch.setattr(J, "PLACEMENTS", 1)
    monkeypatch.setattr(J, "configurations", lambda: [{"part": "A", "name": "t", "speed": 7.0, "amp": 1.0}])
    monkeypatch.setattr(J, "limits_of", lambda rows: {"rows": len(rows)})
    tr = {"first_lon": 60.0, "speed": -2.0, "days": 5, "injected": {"2": 6.0, "4": 7.0}}
    res = J.archive_results(seasons, {"b": {1983: [tr, tr], 1990: [tr]}})
    e = res["bands"]["b"]
    fs = J.pooled_filtered_std([L.filtered(seasons[y]) for y in J.YEARS], [J.col(x) for x in J.BASES])
    assert np.allclose(list(e["filtered_std"].values()), fs)
    assert np.isclose(e["scale"], np.mean(fs[1:])) and np.isclose(e["matched_noise_std"], e["scale"] / res["unit_noise_filtered_std"])
    assert e["packets"] == 3 and set(e["backgrounds"]) == {"archive", "matched"}
    assert e["backgrounds"]["archive"]["nothing_injected"] == J.read([seasons[y] for y in J.YEARS], 17.25)
    assert e["backgrounds"]["matched"]["nothing_injected"] != e["backgrounds"]["archive"]["nothing_injected"]
    assert e["backgrounds"]["archive"]["limits"] == {"rows": 1}


def test_summarize_reads_counts_placements_per_base():
    def one(cls55, mov55, passes03):
        bases = {f"{b:g}": entry() for b in J.BASES}
        bases["55"] = entry(cls55, -2.5 if mov55 else 0.0, moving=mov55)
        return {"bases": bases, "passes": {"0.3": passes03, "0.25": False, "0.2": True}}
    reads = [one("westward", True, True), one("other", True, False), one("westward", False, True), one("standing", False, False), one("standing", True, False)]
    s = J.summarize_reads(reads)
    assert s["reading_passes"] == {"0.3": 2, "0.25": 0, "0.2": 5}
    assert s["bases"]["55"]["westward"] == 2 and s["bases"]["55"]["moving"] == 3
    assert s["bases"]["45"]["westward"] == 0 and s["bases"]["55"]["median_slope"] == -2.5


def test_change_from_nothing_reads_each_base_against_the_uninjected_read():
    ref = {f"{b:g}": entry() for b in J.BASES}
    ref["50"] = entry("westward", -2.5, moving=True)
    ref["60"] = entry("other", -1.5, moving=True)
    def summ(west, mov, slope):
        return {"reading_passes": {}, "bases": {f"{b:g}": {"westward": west.get(b, 0), "moving": mov.get(b, 0),
                                                          "median_slope": slope.get(b, 0.0), "median_westmost": 50.0} for b in J.BASES}}
    g = {"nothing_injected": ref, "rows": [
        {"config": {"part": "A", "name": "t"}, "summary": summ({}, {}, {})},
        {"config": {"part": "B", "name": "P1"}, "summary": summ({50.0: 4}, {50.0: 5, 60.0: 4}, {50.0: -2.5, 60.0: -1.5})},
        {"config": {"part": "B", "name": "P3"}, "summary": summ({50.0: 3, 55.0: 4}, {50.0: 5, 55.0: 5, 60.0: 3}, {50.0: -2.4, 55.0: -2.0, 60.0: -0.9})},
    ]}
    c = J.change_from_nothing(g)
    assert set(c) == {"P1", "P3"}
    assert c["P1"] == {"new_westward": [], "lost_westward": [], "new_moving": [], "lost_moving": [], "max_slope_change": 0.0}
    assert c["P3"]["new_westward"] == ["55"] and c["P3"]["lost_westward"] == ["50"]
    assert c["P3"]["new_moving"] == ["55"] and c["P3"]["lost_moving"] == ["60"]
    assert np.isclose(c["P3"]["max_slope_change"], 2.0)
    s = J.post_hoc_sentences({"b": {"archive": {"packet_change": c}}})["b"][0]
    assert s.startswith("on the archive background, P1 changes no base's westward class or moving ridge")
    s3 = J.post_hoc_sentences({"b": {"archive": {"packet_change": {"P1": c["P3"]}}}})["b"][0]
    assert "new westward at 55 E; lost westward at 50 E; new moving at 55 E; lost moving at 60 E" in s3


def test_train_variance_fraction():
    assert np.isclose(J.train_variance_fraction(2.0, 2.0), 0.99926 ** 2 / 2.0)
    res = {"bands": {"b": {"scale": 1.5, "backgrounds": {"archive": {"nothing_injected": {}, "rows": [],
                                                                     "limits": {"train 5": 1.5, "train 7": None, "train 10": "not monotone"}}}}}}
    ph = J.post_hoc(res)["b"]["archive"]["train_variance_fraction"]
    assert np.isclose(ph["train 5"], (1.5 * 0.99926) ** 2 / 2.0 / 1.5 ** 2) and ph["train 7"] is None and ph["train 10"] is None
