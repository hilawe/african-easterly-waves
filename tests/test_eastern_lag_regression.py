"""The lag-regression diagnostic's rules (EASTERN_LAG_REGRESSION_2026-10-21.md, sections 2 to 4)."""

import importlib.util
import json
import math
import os
from fractions import Fraction

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location("lagreg_under_test", os.path.join(ROOT, "scripts", "eastern_lag_regression.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _wave(L, speed=7.0, wavelength=30.0, phase=0.0):
    t = (np.arange(L.A.SEASON_DAYS * L.OBS_PER_DAY) / L.OBS_PER_DAY)[:, None]
    return 3.0 * np.sin(2.0 * np.pi * (L.LONS[None, :] + speed * t) / wavelength + phase)


def test_filter_and_season_samples():
    L = _load()
    n = L.A.SEASON_DAYS * L.OBS_PER_DAY
    t = np.arange(n) / L.OBS_PER_DAY
    S = np.stack([np.sin(2 * np.pi * t / 5.0), np.sin(2 * np.pi * t / 40.0)], axis=1)
    F = L.filtered(S)
    edge = L.BUFFER_DAYS * L.OBS_PER_DAY
    assert np.isnan(F[:edge]).all() and np.isnan(F[-edge:]).all() and np.isfinite(F[edge:-edge]).all(), "fifteen days lost at each end"
    assert abs(np.std(F[edge:-edge, 0]) / np.std(S[:, 0]) - 1.0) < 0.05 and np.std(F[edge:-edge, 1]) / np.std(S[:, 1]) < 0.02, "5 days passes and 40 days does not"
    by = {m: {"band_means_700": {"b": np.full((L.A.MONTH_DAYS[m] * 4, 3), float(m)).tolist()}} for m in L.MONTHS}
    S2 = L.season_samples(by, "b")
    assert S2.shape == (n, 3) and S2[0, 0] == L.MONTHS[0] and S2[-1, 0] == L.MONTHS[-1], "June to September in order"
    gap = {m: {"band_means_700": {"b": [list(r) for r in v["band_means_700"]["b"]]}} for m, v in by.items()}
    gap[L.MONTHS[0]]["band_means_700"]["b"][0][1] = float("nan")
    try:
        L.season_samples(gap, "b"); assert False
    except SystemExit as ex:
        assert "missing sample" in str(ex), "one missing sample refuses the season rather than shortening its filtered block"
    by[L.MONTHS[0]]["band_means_700"]["b"] = by[L.MONTHS[0]]["band_means_700"]["b"][:-1]
    try:
        L.season_samples(by, "b"); assert False
    except SystemExit as ex:
        assert "samples in a season" in str(ex)


def test_lag_convention_and_ridge():
    L = _load()
    seasons = [L.filtered(_wave(L, phase=k)) for k in range(3)]
    col = int(np.flatnonzero(np.isclose(L.LONS, 50.0))[0])
    R = L.lag_correlation(seasons, col)
    zero = int(np.flatnonzero(L.LAGS == 0)[0])
    assert abs(R[zero, col] - 1.0) < 1e-9, "a series with itself at lag 0"
    one_day = int(np.flatnonzero(L.LAGS == L.OBS_PER_DAY)[0])
    near = np.flatnonzero((L.LONS >= 35.0) & (L.LONS <= 57.0))
    assert L.LONS[near[int(np.nanargmax(R[one_day, near]))]] == 43.0, "a positive lag looks at the field later, so a wave moving 7 degrees west a day peaks 7 degrees west of the base"
    tr = L.track_ridge(R, col)
    assert abs(tr["slope_deg_per_day"] + 7.0) < 0.05 and tr["positive_lag_days"] == L.MAX_LAG_DAYS and tr["westmost"] == 22.0
    assert abs(L.speed_ms(-7.0, 0.0) + 7.0 * 111195.0 / 86400.0) < 1e-9 and L.speed_ms(None, 10.0) is None
    shifted = L.lag_correlation(seasons, col, shift=1)
    assert np.nanmax(np.abs(shifted[zero] - R[zero])) > 0.1, "a shift pairs the base of season i with the field of season i + 1"
    flat = np.zeros((len(L.LAGS), len(L.LONS))); flat[zero, col] = 1.0
    assert L.track_ridge(flat, col)["slope_deg_per_day"] is None and L.track_ridge(flat, col)["lags"] == [0], "a ridge of one lag has no slope"
    two = flat.copy(); two[zero + 1, col - 1] = 0.9
    assert L.track_ridge(two, col)["lags"] == [0, 1] and L.track_ridge(two, col)["slope_deg_per_day"] is None, "nor one of two lags"
    stop = np.zeros_like(flat)
    for li, lag in enumerate(L.LAGS):
        c = col - int(round(lag * 0.5))
        stop[li, c] = 0.3 if abs(lag) <= 3 else 0.2999
    stop[zero, col] = 1.0
    tr = L.track_ridge(stop, col)
    assert tr["lags"] == [-3, -2, -1, 0, 1, 2, 3], "0.3 is kept and 0.2999 ends the ridge"
    assert L.track_ridge(stop, col, r_min=0.2)["lags"] == list(range(-16, 17)), "a lower threshold keeps the weaker lags"
    ac = np.zeros_like(flat); ac[zero:zero + 8, col] = [1.0, 0.8, 0.5, 0.1, -0.3, -0.5, -0.4, -0.2]
    assert L.base_autocorrelation(ac, col) == {"first_min_lag_days": 1.25, "first_min_r": -0.5}, "the first local minimum at positive lags"
    flat_bottom = np.zeros_like(flat); flat_bottom[zero:zero + 5, col] = [1.0, 0.5, -0.4, -0.4, 0.0]
    assert L.base_autocorrelation(flat_bottom, col) == {"first_min_lag_days": 0.5, "first_min_r": -0.4}, "a flat bottom's first point is the minimum"
    mono = np.zeros_like(flat); mono[zero:, col] = np.linspace(1.0, -1.0, len(L.LAGS) - zero)
    assert L.base_autocorrelation(mono, col) == {"first_min_lag_days": None, "first_min_r": None}, "no minimum inside the lags"
    far = np.zeros_like(flat); far[zero, col] = 1.0; far[zero + 1, col - 9] = 0.9; far[zero + 1, col - 8] = 0.5
    assert L.track_ridge(far, col)["peak_lon"][1] == 46.0, "4.5 degrees away is outside the search, 4 degrees is inside"
    holed = [F.copy() for F in seasons]; holed[1][200, 5] = np.nan
    try:
        L.lag_correlation(holed, col); assert False
    except SystemExit as ex:
        assert "one finite block" in str(ex), "a missing value inside a season refuses rather than shortening the series silently"
    short = [F.copy() for F in seasons]; short[2][100:150] = np.nan
    try:
        L.lag_correlation(short, col); assert False
    except SystemExit as ex:
        assert "one finite block" in str(ex)
    edge_col = int(np.flatnonzero(np.isclose(L.LONS, -8.5))[0])
    e = np.zeros_like(flat); e[zero, edge_col] = 1.0; e[zero + 1, edge_col - 1] = 0.9
    assert L.track_ridge(e, edge_col)["lags"] == [0], "a peak within a degree of the grid's edge ends the ridge"


def test_classes_and_flags():
    L = _load()
    ex = lambda s: None if s is None else [Fraction(str(s)).numerator, Fraction(str(s)).denominator]
    mk = lambda s, lagd, west, east: {"slope_deg_per_day": s, "slope_exact": ex(s), "positive_lag_days": lagd, "westmost": west, "eastmost": east}
    assert L.classify(mk(-2.0, 1.0, 45.0, 50.0), 50.0) == "westward", "every bar is inclusive"
    assert L.classify(mk(-1.99, 1.0, 45.0, 50.0), 50.0) == "other" and L.classify(mk(-2.0, 0.75, 45.0, 50.0), 50.0) == "other" and L.classify(mk(-2.0, 1.0, 45.5, 50.0), 50.0) == "other"
    assert L.classify(mk(0.99, 3.0, 50.0, 60.0), 50.0) == "standing" and L.classify(mk(None, 0.0, 50.0, 50.0), 50.0) == "standing" and L.classify(mk(-1.0, 3.0, 40.0, 50.0), 50.0) == "other"
    assert L.classify(mk(2.0, 1.0, 50.0, 55.0), 50.0) == "eastward"
    res = L.annotate({"60": mk(-5.0, 3.0, 44.5, 60.0), "65": mk(-5.0, 3.0, 45.0, 65.0), "45": mk(-5.0, 3.0, 40.0, 45.0), "44": mk(-5.0, 3.0, 30.0, 44.0)})
    sens = L.annotate({"60": dict(mk(0.0, 0.75, 60.0, 60.0), sensitivity={"0.2": mk(-5.0, 3.0, 44.0, 60.0)})})
    assert sens["60"]["class"] == "standing" and sens["60"]["sensitivity"]["0.2"]["class"] == "westward" and sens["60"]["sensitivity"]["0.2"]["reaches_s2_west"], "each threshold's ridge is classed on its own"
    assert res["60"]["reaches_s2_west"] and not res["65"]["reaches_s2_west"] and res["45"]["reaches_s2_west"] and not res["44"]["reaches_s2_west"], "44.5 E inclusive, and only from a base east of it"
    assert res["45"]["crosses_40"] and res["44"]["crosses_40"] and not res["60"]["crosses_40"] and not res["65"]["crosses_40"], "40 E inclusive, from any base east of it"
    band = {f"{b:g}": {"reaches_s2_west": False} for b in L.S2_EAST_BASES}
    assert L.band_reading(band) == L.READINGS["stops"]
    band["65"]["reaches_s2_west"] = True
    assert L.band_reading(band) == L.READINGS["passes"]


def test_constructions_rule_and_gate(tmp_path):
    L = _load()
    res = json.loads(json.dumps(L.run_constructions()))
    v = L.apply_rule(res)
    assert v == {"adequate": True, "failures": []}
    assert res["R3"]["55"]["positive_lag_days"] == 1.25 and res["R3"]["55"]["sensitivity"]["0.25"]["positive_lag_days"] == 1.5, "the lower threshold follows the ridge further"
    assert res["R3"]["50"]["class"] == "standing" and res["R3"]["60"]["reaches_s2_west"], "inside the wobbling feature the local signal masks the wave, from east of it the wave passes through"
    assert not any(x["reaches_s2_west"] for x in res["R5"].values()) and all(x["westmost"] > L.S2_WEST for x in res["R5"].values() if x["class"] == "westward")
    assert res["R6"]["50"]["shift_max_r"]["max"] < L.R_MIN and "shift_max_r" not in res["R6"]["40"] and "shift_max_r" not in res["R2"]["50"], "the construction null at three reference bases of R6 only"
    bad = json.loads(json.dumps(res)); bad["R4"]["50"]["class"] = "westward"
    assert L.apply_rule(bad)["failures"] == ["R4 moving at ['50']"]
    slow = json.loads(json.dumps(res)); slow["R2"]["40"]["slope_deg_per_day"] = -5.24
    assert L.apply_rule(slow)["failures"] == ["R2 base 40 not westward at -7 within 25 percent (-5.24)"]
    leak = json.loads(json.dumps(res)); leak["R5"]["60"]["reaches_s2_west"] = True
    assert L.apply_rule(leak)["failures"] == ["R5 base 60 reaches S2's western edge"]
    R4 = np.array([L.construction_samples("R4", i) for i in range(4)])
    t = np.arange(R4.shape[1]) / L.OBS_PER_DAY
    even = (np.floor(t / 4.0) % 2 == 0)
    c47, c54 = (int(np.flatnonzero(np.isclose(L.LONS, x))[0]) for x in (47.0, 54.0))
    assert R4[:, even, c47].mean() - R4[:, ~even, c47].mean() < -2.0 and R4[:, even, c54].mean() - R4[:, ~even, c54].mean() > 2.0, "R4's features take turns as the deeper every four days"
    ok = json.loads(json.dumps(res)); ok["R2"]["40"]["slope_deg_per_day"] = -5.25
    assert L.apply_rule(ok)["adequate"], "25 percent inclusive"
    p = tmp_path / "c.json"
    stale = json.loads(json.dumps(res)); stale["R3"]["60"]["reaches_s2_west"] = False
    for rec, run, msg in (({"generated_by": {"sha256": "x"}, "results": res, "verdict": v}, lambda: res, "different version"),
                          ({"generated_by": {"sha256": L.SCRIPT_SHA256_AT_LOAD}, "results": res, "verdict": v}, lambda: stale, "differs from a fresh run"),
                          ({"generated_by": {"sha256": L.SCRIPT_SHA256_AT_LOAD}, "results": stale, "verdict": v}, lambda: stale, "do not pass the rule"),
                          ({"generated_by": {"sha256": L.SCRIPT_SHA256_AT_LOAD}, "results": res, "verdict": {"adequate": True, "failures": ["x"]}}, lambda: res, "stored summary disagrees")):
        p.write_text(json.dumps(rec))
        try:
            L.check_constructions(str(p), rerun=run); assert False
        except SystemExit as ex:
            assert msg in str(ex), str(ex)
    p.write_text(json.dumps({"generated_by": {"sha256": L.SCRIPT_SHA256_AT_LOAD}, "results": res, "verdict": v}))
    assert L.check_constructions(str(p), rerun=lambda: res)["verdict"]["adequate"] is True
    text = L.constructions_text(res, v)
    assert text.count("\n| R") == 2 * 6 * len(L.BASES) and "adequate under the rule of section 4, yes" in text, "every case and base in the declared table and in the post hoc one"


def test_archive_keeps_bands_apart(monkeypatch):
    L = _load()
    monkeypatch.setattr(L, "YEARS", L.YEARS[:3])
    by = {}
    for i, y in enumerate(L.YEARS):
        wave = _wave(L, phase=i)
        rng = np.random.default_rng(i)
        noise = rng.standard_normal(wave.shape) * 0.01
        start = 0
        for m in L.MONTHS:
            n = L.A.MONTH_DAYS[m] * 4
            bands = {b: (np.round(wave[start:start + n] + noise[start:start + n], 2) if b == "20 to 24.5 N" else np.round(noise[start:start + n] * 100, 2)).tolist() for b in L.BAND_LAT}
            by.setdefault(y, {})[m] = {"band_means_700": bands}
            start += n
    res = L.archive_results(by)
    assert res["20 to 24.5 N"]["reading"] == L.READINGS["passes"] and res["15 to 19.5 N"]["reading"] == L.READINGS["stops"], "each band is read from its own series"
    assert abs(res["20 to 24.5 N"]["bases"]["60"]["speed_ms"] - (-7.0 * 111195.0 * math.cos(math.radians(22.25)) / 86400.0)) < 0.1, "the speed at the band's middle latitude"
    text = L.archive_text(res)
    assert text.count("\n| ") == 2 * (2 + 4 * len(L.BASES)) and "- 20 to 24.5 N, reading under section 2.5, a westward" in text, "two tables, each a header, a rule and a row per band and base"


def _record(L, y, m, lons=None, bands=None):
    import calendar
    times = [f"{y}-{m:02d}-{d:02d}T{h:02d}" for d in range(1, calendar.monthrange(y, m)[1] + 1) for h in (0, 6, 12, 18)]
    lons = L.LONS if lons is None else lons
    return {"year": y, "month": m, "lons": list(map(float, lons)), "bands": {b: v for b, v in L.BAND_LAT.items()} if bands is None else bands, "times": times,
            "band_means_700": {b: np.zeros((len(times), len(lons))).tolist() for b in L.BAND_LAT}}


def test_input_contract_slope_boundary_and_filter_periods():
    L = _load()
    good = _record(L, 1990, 7)
    L.check_inputs([good])
    shifted = _record(L, 1990, 7, lons=L.LONS + 0.5)
    for rec, msg in ((shifted, "declared grid"), (dict(good, bands={"x": [0, 1]}), "declared bands"), (dict(good, times=good["times"][1:] + good["times"][:1]), "six-hourly times"),
                     (dict(good, band_means_700=dict(good["band_means_700"], **{"10 to 14.5 N": good["band_means_700"]["10 to 14.5 N"][:-1]})), "complete array")):
        try:
            L.check_inputs([rec]); assert False
        except SystemExit as ex:
            assert msg in str(ex), str(ex)
    nan = json.loads(json.dumps(good)); nan["band_means_700"]["15 to 19.5 N"][3][7] = float("nan")
    try:
        L.check_inputs([nan]); assert False
    except SystemExit as ex:
        assert "complete array" in str(ex)
    for base in (20.0, 30.0):
        col = int(np.flatnonzero(L.LONS == base)[0]); R = np.zeros((len(L.LAGS), len(L.LONS)))
        for li, lag in enumerate(L.LAGS):
            R[li, int(np.flatnonzero(L.LONS == base - 0.5 * lag)[0])] = 1.0
        tr = L.track_ridge(R, col)
        assert tr["slope_exact"] == [-2, 1] and L.classify(tr, base) == "westward", "an exact -2 degrees per day is westward at every base"
    n = L.A.SEASON_DAYS * L.OBS_PER_DAY
    t = (np.arange(n) / L.OBS_PER_DAY)[:, None]
    phases = [float(np.random.default_rng(L.PHASE_SEED + i).uniform(0, 2 * np.pi)) for i in range(4)]
    col = int(np.flatnonzero(L.LONS == 60.0)[0])
    for period in (3.0, 4.0, 5.0, 6.0):
        series = [L.filtered(np.round(np.broadcast_to(3 * np.sin(2 * np.pi * t / period + p), (n, len(L.LONS))), 2)) for p in phases]
        assert L.base_autocorrelation(L.lag_correlation(series, col), col)["first_min_lag_days"] == period / 2, "through the same filter the first minimum follows the input's period"


def test_a_weak_continuous_train_reads_stops():
    """The review's counterexample (section 8): a westward train of 1 m/s, present in every season
    and crossing 44.5 E from every eastern base, reaches from none of them at any of the three
    thresholds, while at 2 m/s it reaches from all three at every threshold. The reading is a
    statement about the tracked ridge, not about the absence of a traveling signal."""
    L = _load()
    n = L.A.SEASON_DAYS * L.OBS_PER_DAY
    t = (np.arange(n) / L.OBS_PER_DAY)[:, None]; x = L.LONS[None, :]
    phases = [float(np.random.default_rng(L.PHASE_SEED + i).uniform(0, 2 * np.pi)) for i in range(24)]
    out = {}
    for amp in (1.0, 2.0):
        seasons = [L.filtered(np.round(amp * np.sin(2 * np.pi * (x + 7 * t) / 30 + p) + L.A.noise_field(3.03, seed=L.HI.NOISE_SEEDS[i], n_steps=n, lons=L.LONS), 2)) for i, p in enumerate(phases)]
        res = L.annotate(L.summarize(seasons, bases=(60.0, 65.0, 70.0), null_bases=()))
        out[amp] = [v["reaches_s2_west"] or v["sensitivity"]["0.25"]["reaches_s2_west"] or v["sensitivity"]["0.2"]["reaches_s2_west"] for v in res.values()]
        out[(amp, "all")] = [v["reaches_s2_west"] and v["sensitivity"]["0.25"]["reaches_s2_west"] and v["sensitivity"]["0.2"]["reaches_s2_west"] for v in res.values()]
    assert out[1.0] == [False, False, False] and out[(2.0, "all")] == [True, True, True]
