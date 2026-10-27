"""Tests for scripts/eastern_phase_envelope.py (`EASTERN_PHASE_ENVELOPE_2026-10-26.md`)."""

import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import eastern_phase_envelope as PE  # noqa: E402

LONS = PE.LONS
T = 360


def col(lon):
    return int(np.flatnonzero(np.isclose(LONS, lon))[0])


def wave(c_phase, wavelength, env=None, seed=0, n=T):
    """An analytic field exp(i(w t - k x)) of a wave at c_phase deg/day, with optional envelope
    function env(x, t) and a little complex noise so every map is finite."""
    t = (np.arange(n) * PE.DT_DAYS)[:, None]
    x = LONS[None, :]
    k = 2.0 * math.pi / wavelength * np.sign(c_phase)       # c = w / k, so westward phase needs k < 0 with w > 0
    w = abs(2.0 * math.pi / wavelength * c_phase)
    A = np.ones((n, len(LONS))) if env is None else env(x, t)
    rng = np.random.default_rng(seed)
    return A * np.exp(1j * (w * t - k * x)) + 0.01 * (rng.standard_normal((n, len(LONS))) + 1j * rng.standard_normal((n, len(LONS))))


def read(Zs, base):
    c = col(base)
    C, E, N = PE.maps(PE.stack([PE.season_sums(Z, c) for Z in Zs]))
    return PE.readings(C, N, c), C, N


# ------------------------------------------------------------------ signs, density and sums

def test_density_is_positive_for_westward_phase_and_negative_for_eastward():
    for c_phase, sign in ((-5.0, 1), (5.0, -1)):
        Z = wave(c_phase, 20.0)
        C, E, N = PE.maps(PE.stack([PE.season_sums(Z, col(50.0))]))
        W = PE.density(C)
        assert np.all(np.sign(W[:, 60:110]) == sign)


def test_density_is_near_zero_for_a_pattern_of_one_phase():
    t = (np.arange(T) * PE.DT_DAYS)[:, None]
    Z = np.exp(-((LONS[None, :] - 50.0) ** 2) / 50.0) * np.exp(1j * 1.5 * t) + 0j
    C, E, N = PE.maps(PE.stack([PE.season_sums(Z, col(50.0))]))
    W = PE.density(C)
    assert np.nanmax(np.abs(W[:, 100:140])) < 1e-9 * np.nanmax(np.abs(C)) ** 2


def test_positive_lag_looks_at_the_field_later_than_the_base():
    Z = np.zeros((T, len(LONS)), complex)
    rng = np.random.default_rng(3)
    s = rng.standard_normal(T + 8) + 1j * rng.standard_normal(T + 8)
    Z[:, col(50.0)] = s[8:]
    Z[:, col(40.0)] = s[:T]          # 40 E carries the base's series 8 samples (2 days) later
    C, E, N = PE.maps(PE.stack([PE.season_sums(Z, col(50.0))]))
    li = int(np.flatnonzero(PE.LAGS == 8)[0])
    assert N[li, col(40.0)] > 0.99
    assert N[int(np.flatnonzero(PE.LAGS == -8)[0]), col(40.0)] < 0.3


def test_resample_weights_of_ones_reproduce_the_pooled_maps_and_counts_sum_to_seasons():
    Zs = [wave(-5.0, 20.0, seed=s) for s in range(4)]
    st = PE.stack([PE.season_sums(Z, col(50.0)) for Z in Zs])
    a = PE.maps(st)
    b = PE.maps(st, np.ones(4))
    for x, y in zip(a, b):
        assert np.allclose(x, y, equal_nan=True)
    W = PE.boot_weights(24, 50)
    assert W.shape == (50, 24) and np.all(W.sum(axis=1) == 24)
    assert np.array_equal(W, PE.boot_weights(24, 50))


def test_a_resample_weight_doubles_one_season():
    Zs = [wave(-5.0, 20.0, seed=s) for s in range(2)]
    st = PE.stack([PE.season_sums(Z, col(50.0)) for Z in Zs])
    two = PE.stack([PE.season_sums(Z, col(50.0)) for Z in (Zs[0], Zs[0], Zs[1])])
    for x, y in zip(PE.maps(st, [2.0, 1.0]), PE.maps(two)):
        assert np.allclose(x, y, equal_nan=True)


def test_incoherent_columns_and_their_neighbors_are_masked():
    C = np.ones((3, 6), complex) * np.exp(1j * np.arange(6))[None, :]
    N = np.ones((3, 6)); N[1, 3] = 0.29
    W = PE.coherent_density(C, N)
    assert np.isnan(W[1, 2]) and np.isnan(W[1, 3]) and np.isnan(W[1, 4])
    assert np.isfinite(W[0, 3]) and np.isfinite(W[1, 1])


# ------------------------------------------------------------------ speeds

@pytest.mark.parametrize("c_phase,wavelength", [(-7.0, 30.0), (-3.0, 12.0), (4.0, 20.0)])
def test_the_phase_speed_of_a_plane_wave_is_recovered(c_phase, wavelength):
    r, C, N = read([wave(c_phase, wavelength, env=lambda x, t: np.exp(-((x - 50.0) ** 2) / 200.0) + 0 * t)], 50.0)
    assert r["phase_speed"] == pytest.approx(c_phase, rel=0.03)


def test_a_wavelength_longer_than_the_limit_reads_no_zonal_progression():
    r, C, N = read([wave(-7.0, 90.0, env=lambda x, t: np.exp(-((x - 50.0) ** 2) / 200.0) + 0 * t)], 50.0)
    assert r["phase_speed"] is None and r["no_zonal_progression"]
    assert abs(2.0 * math.pi / r["k_mean"]) > PE.LONG_WAVE_DEG


@pytest.mark.parametrize("c_env", [-4.0, 0.0, 3.0])
def test_the_envelope_follows_packets_born_everywhere_at_their_speed(c_env):
    rng = np.random.default_rng(5)
    k = 2.0 * math.pi / 20.0
    t = (np.arange(T) * PE.DT_DAYS)[:, None]
    x = LONS[None, :]
    Zs = []
    for s in range(6):
        Z = np.zeros((T, len(LONS)), complex)
        for a, b in zip(rng.uniform(-4, 94, 60), rng.uniform(-10, 75, 60)):
            tau = t - a
            env = np.exp(-((x - b - c_env * tau) ** 2) / 72.0 - tau ** 2 / 8.0)
            Z += env * np.exp(1j * (k * 5.0 * tau + k * (x - b) + rng.uniform(0, 6.28)))
        Zs.append(Z)
    r, C, N = read(Zs, 40.0)
    assert r["envelope_speed"] == pytest.approx(c_env, abs=0.6)
    assert r["phase_speed"] == pytest.approx(-5.0, rel=0.1)


def test_following_stops_where_the_coherence_falls_below_the_floor():
    Z = wave(-5.0, 20.0, env=lambda x, t: np.exp(-((x - 50.0) ** 2) / 200.0) + 0 * t)
    C, E, N = PE.maps(PE.stack([PE.season_sums(Z, col(50.0))]))
    N2 = N.copy()
    N2[np.abs(PE.LAGS) > 4] = 0.29
    pos = PE.follow(PE.coherent_density(C, N2), N2, col(50.0), 1)
    assert sorted(pos) == list(range(-4, 5))
    pos = PE.follow(PE.density(C), N2, col(50.0), 1)          # the floor binds in follow itself
    assert sorted(pos) == list(range(-4, 5))
    pos = PE.follow(PE.coherent_density(C, N), N, col(50.0), 1)
    assert len(pos) > 9


def test_the_analytic_block_drops_two_days_at_each_end():
    S = np.random.default_rng(2).standard_normal((PE.A.SEASON_DAYS * 4, len(LONS)))
    F = PE.LR.filtered(S)
    e0, e1 = PE.LR.valid_block(F)
    assert PE.analytic(S).shape == (e1 - e0 - 2 * PE.TRIM, len(LONS))


def test_the_noise_free_reference_has_no_noise():
    q = PE.clean_readings("D07", {"D07": 3.0}, bases=(50.0,))["50"]
    assert abs(q["envelope_speed"]) < 0.3 and q["envelope_point_class"] == "standing"
    assert q["phase_speed"] == pytest.approx(-7.0, rel=0.03)


def test_the_slope_needs_three_followed_lags():
    assert PE.slope({0: 50.0, 1: 49.0}) is None
    assert PE.slope({-4: 52.0, 0: 50.0, 4: 48.0}) == pytest.approx(-2.0)


# ------------------------------------------------------------------ classes

def test_the_classes_and_their_boundaries():
    assert PE.classify(-1.0, -1.5, -0.1, 0.0, 1.0) == "westward"
    assert PE.classify(-0.99, -1.5, -0.1, 0.0, 1.0) == "unresolved"
    assert PE.classify(-1.0, -1.5, 0.0, 0.0, 1.0) == "unresolved"
    assert PE.classify(-2.0, -3.0, -1.0, 0.0, 0.75) == "unresolved"
    assert PE.classify(1.0, 0.1, 1.5, 0.0, 1.0) == "eastward"
    assert PE.classify(0.0, -1.0, 1.0, 0.0, 0.0) == "standing"
    assert PE.classify(0.0, -1.01, 1.0, 0.0, 0.0) == "unresolved"
    assert PE.classify(-2.0, -3.0, -1.0, 0.051, 1.0) == "unresolved"
    assert PE.classify(-2.0, -3.0, -1.0, 0.05, 1.0) == "westward"
    assert PE.classify(None, None, None, 0.0, 1.0) == "unresolved"


def test_no_zonal_progression_needs_the_point_and_most_resamples():
    r = {"no_zonal_progression": True, "phase_speed": None, "pos_lag_days": 2.0}
    assert PE.phase_class(r, None, None, 1.0, 0.95) == "no zonal phase progression"
    assert PE.phase_class(r, None, None, 1.0, 0.94) == "unresolved"
    assert PE.phase_class(dict(r, no_zonal_progression=False), None, None, 1.0, 1.0) == "unresolved"


def test_the_interval_reports_the_undefined_fraction():
    lo, hi, und = PE.interval([None, 1.0, 2.0, 3.0])
    assert und == pytest.approx(0.25) and lo == pytest.approx(1.05) and hi == pytest.approx(2.95)
    assert PE.interval([None, None]) == (None, None, 1.0)


def test_exactly_five_percent_undefined_stays_admissible_through_interval_and_classify():
    for n_undef, want in ((50, "westward"), (51, "unresolved")):
        lo, hi, und = PE.interval([None] * n_undef + [-2.0] * (1000 - n_undef))
        assert PE.classify(-2.0, lo, hi, und, 1.0) == want


def test_the_half_day_challenge_is_de7_with_only_its_duration_changed(monkeypatch):
    assert PE.CASES["DE7q"]["st"] == 0.5 and PE.CASES["DE7q"]["seed_of"] == "DE7"
    q = PE.signal("DE7q", 3)
    monkeypatch.setattr(PE, "PACKET_ST", 0.5)
    assert np.array_equal(q, PE.signal("DE7", 3))
    assert "DE7q" not in PE.SCORED_CASES


def test_point_class():
    assert PE.point_class(-1.0) == "westward" and PE.point_class(1.0) == "eastward"
    assert PE.point_class(0.99) == "standing" and PE.point_class(None) == "none"


# ------------------------------------------------------------------ the rule

def fake(phase=-7.0, pcls="westward", ecls="westward", e=-5.0):
    return {"phase_class": pcls, "phase_speed": phase, "envelope_class": ecls, "envelope_speed": e}


def fake_results(over=None):
    cases, clean = {}, {}
    want = {"ND7": "westward", "ND3": "westward", "D07": "standing", "D03": "standing", "DE7": "eastward", "DE3": "eastward",
            "U1": "standing", "U2": "standing"}
    for c, cls in want.items():
        cp = PE.CASES[c]["c_phase"]
        e = {"westward": -4.0, "standing": 0.0, "eastward": 2.0}[cls]
        cases[c] = {f"{b:g}": fake(cp, "westward", cls, e) for b in PE.SCORED_BASES}
        clean[c] = {f"{b:g}": {"envelope_speed": e, "envelope_point_class": cls} for b in PE.SCORED_BASES}
    for c in ("S", "N"):
        cases[c] = {f"{b:g}": fake(None, "unresolved", "unresolved", None) for b in PE.BASES}
    res = {"cases": cases, "clean": clean}
    for (c, b, k), v in (over or {}).items():
        res["cases"][c][b][k] = v
    return res


def test_the_rule_passes_the_expected_readings():
    assert PE.apply_rule(fake_results()) == {"adequate": True, "failures": []}


@pytest.mark.parametrize("over", [
    {("ND7", "50", "phase_speed"): -8.8},
    {("ND3", "55", "phase_speed"): -2.2},
    {("D07", "50", "phase_class"): "unresolved"},
    {("DE7", "55", "envelope_class"): "unresolved"},
    {("D03", "50", "envelope_class"): "westward"},
    {("U1", "50", "envelope_class"): "unresolved"},
    {("U2", "50", "envelope_class"): "eastward"},
    {("S", "45", "phase_class"): "eastward"},
    {("N", "0", "envelope_class"): "westward"},
])
def test_the_rule_refuses_each_defect(over):
    assert not PE.apply_rule(fake_results(over))["adequate"]


def test_u2_may_be_unresolved_but_not_moving():
    assert PE.apply_rule(fake_results({("U2", "55", "envelope_class"): "unresolved"}))["adequate"]


def test_the_rule_refuses_a_missing_noise_free_reference():
    res = fake_results()
    res["clean"]["DE3"]["50"]["envelope_speed"] = None
    assert not PE.apply_rule(res)["adequate"]


# ------------------------------------------------------------------ records and refusals

def test_check_constructions_refuses_each_mismatch(tmp_path):
    good = fake_results()
    verdict = PE.apply_rule(good)
    gen = {"sha256": PE.SCRIPT_SHA256_AT_LOAD}
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"generated_by": gen, "results": good, "verdict": verdict}))
    assert PE.check_constructions(str(p), rerun=lambda: good)["verdict"] == verdict
    p.write_text(json.dumps({"generated_by": {"sha256": "0" * 64}, "results": good, "verdict": verdict}))
    with pytest.raises(SystemExit, match="different version"):
        PE.check_constructions(str(p), rerun=lambda: good)
    p.write_text(json.dumps({"generated_by": gen, "results": good, "verdict": verdict}))
    with pytest.raises(SystemExit, match="differs from a fresh run"):
        PE.check_constructions(str(p), rerun=lambda: fake_results({("ND7", "50", "phase_speed"): -7.01}))
    bad = fake_results({("ND7", "50", "phase_speed"): -10.0})
    p.write_text(json.dumps({"generated_by": gen, "results": bad, "verdict": PE.apply_rule(bad)}))
    with pytest.raises(SystemExit, match="do not pass"):
        PE.check_constructions(str(p), rerun=lambda: bad)
    p.write_text(json.dumps({"generated_by": gen, "results": good, "verdict": {"adequate": True, "failures": ["x"]}}))
    with pytest.raises(SystemExit, match="stored verdict"):
        PE.check_constructions(str(p), rerun=lambda: good)


def test_records_are_never_overwritten(tmp_path):
    p = tmp_path / "r.json"
    p.write_text("{}")
    with pytest.raises(SystemExit, match="never overwritten"):
        PE.refuse_existing([str(p)])


def test_the_held_out_season_is_refused_before_any_file_is_opened():
    with pytest.raises(SystemExit, match="held-out"):
        PE.level_band_means(2005, 600)


def test_the_generated_reading_names_speeds_and_no_mechanism():
    r = {"direction": "westward phase tilt", "west_share": 0.9, "phase_class": "westward", "phase_speed": -6.0, "phase_interval": [-7.0, -5.0],
         "envelope_class": "standing", "envelope_speed": 0.1, "envelope_interval": [-0.5, 0.6], "difference_interval": [5.0, 7.0]}
    s = PE.reading_text(r)
    assert "phases westward at -6.00 (-7.00 to -5.00)" in s and "envelope standing at +0.10" in s and "envelope minus phase +6.10" in s
    for word in ("group", "dispers", "packet", "propagat", "energy", "mechanism"):
        assert word not in s.lower()


DATA = os.path.join(os.path.dirname(HERE), "data", "era5", "region6h", "era5_v700_1990_6h_region.nc")
XBAND = os.path.join(os.path.dirname(HERE), "docs", "aewc_v2", "evidence", "validation", "eastern_pilot_2026-10-05", "reports", "extent", "xband")


@pytest.mark.skipif(not (os.path.exists(DATA) and os.path.isdir(XBAND)), reason="the retained fields are not on this machine")
def test_the_band_mean_builder_reproduces_the_stored_700_hpa_records(monkeypatch):
    monkeypatch.chdir(os.path.dirname(HERE))
    files, by = PE.load_xband(XBAND)
    assert PE.check_level_builder(by, years=(1990,))
