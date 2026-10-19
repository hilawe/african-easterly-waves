"""The background-removal effort must compute each background as section 2 of
`EASTERN_BACKGROUND_REMOVAL_2026-10-19.md` states, place each synthetic feature where it
says, score what it says on hand-built series, and apply the declared pass and selection
rules (section 3)."""
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("anomaly_minima_under_test", os.path.join(ROOT, "scripts", "eastern_anomaly_minima.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_each_background_is_computed_as_stated():
    M = _load()
    mo = M.month_index()
    assert len(mo) == 122 and mo[29] == 6 and mo[30] == 7 and mo[60] == 7 and mo[61] == 8 and mo[92] == 9
    rng = np.random.default_rng(3)
    D = rng.normal(size=(122, 5))
    b1 = M.background("B1", D, mo)
    assert np.allclose(b1[0], D[:30].mean(axis=0)) and np.allclose(b1[29], b1[0]) and np.allclose(b1[30], D[30:61].mean(axis=0)) and not np.allclose(b1[29], b1[30]), "B1 is the own month mean and changes at the month boundary"
    b5 = M.background("B2_5", D, mo)
    assert np.isnan(b5[:2]).all() and np.isnan(b5[-2:]).all() and np.isfinite(b5[2:-2]).all() and np.allclose(b5[2], D[0:5].mean(axis=0)) and np.allclose(b5[60], D[58:63].mean(axis=0))
    b9 = M.background("B2_9", D, mo)
    assert np.isnan(b9[:4]).all() and np.isnan(b9[-4:]).all() and np.isfinite(b9[4]).all() and np.allclose(b9[4], D[0:9].mean(axis=0))
    allmm = {y: {m: np.full(5, float(y)) for m in M.MONTHS} for y in (1990, 1991, 1992)}
    loo = M.loo_month_means(allmm, 1990)
    assert np.allclose(loo[6], 1991.5), "B0's month mean leaves the own season out"
    b0 = M.background("B0", D, mo, clim={m: np.full(5, float(m)) for m in M.MONTHS})
    assert np.allclose(b0[0], 6.0) and np.allclose(b0[121], 9.0)
    try:
        M.background("B3", D, mo); assert False
    except SystemExit as ex:
        assert "unknown background" in str(ex)


def test_the_generator_places_each_feature_where_it_says():
    M = _load()
    D, s_c, tr_c = M.synth_season(standing=lambda t: 50.0, transient=(35.0, 72.0, 2.0, 10.0))
    assert D.shape == (122, len(M.LONS)) and np.allclose(s_c, 50.0)
    assert np.isnan(tr_c[34]) and abs(tr_c[35] - (72.0 - 2.0 * 0.375)) < 1e-12 and abs(tr_c[40] - (72.0 - 2.0 * 5.375)) < 1e-12, "a moving feature's daily-mean center is its position 9 hours into the day"
    east = M.LONS > 55.0
    assert abs(M.LONS[east][np.argmin(D[40][east])] - tr_c[40]) <= 0.5, "the daily-mean minimum lies at the daily-mean center to the grid"
    assert abs(M.LONS[np.argmin(D[10])] - 50.0) < 1e-9
    assert np.allclose(np.rint(D * 400), D * 400), "the samples are rounded to the records' two decimals before the daily mean"
    _, w_c, _ = M.synth_season(standing=lambda t: 50.0 + 3.0 * np.sin(2 * np.pi * t / 14.0))
    t = np.arange(4) / 4.0 + 3.0
    assert abs(w_c[3] - np.mean(50.0 + 3.0 * np.sin(2 * np.pi * t / 14.0))) < 1e-12, "a wobbling center is the mean of its four positions"
    n1, n2 = M.noise_field(2.0), M.noise_field(2.0)
    assert np.array_equal(n1, n2) and abs(n1.std() - 2.0) < 1e-9 and n1.shape == (488, len(M.LONS))
    assert abs(M.noise_field(1.5).std() - 1.5) < 1e-9


def test_the_scores_count_what_they_say():
    M = _load()
    mins = [None, [], [{"lon": 53.0}], [{"lon": 60.0}], [{"lon": 57.0}]]
    sh, n = M.share_near(mins, [52.0, 52.0, 52.0, 52.0, np.nan])
    assert n == 3 and abs(sh - 1 / 3) < 1e-12, "an unavailable day and an undefined center are not counted"
    assert M.share_near([[{"lon": 57.0}]], [52.0]) == (1.0, 1), "5 degrees exactly is near"
    tr = [np.nan, 72.0, 60.0, 51.0, 49.5, 30.0, 23.0]
    mins = [[{"lon": 50.0}], [{"lon": 72.0}], [{"lon": 61.0}], [{"lon": 52.5}], [{"lon": 49.5}], [], [{"lon": 23.0}]]
    r = M.recovery(mins, tr, standing_at=50.0)
    assert r == {"present_days": 4, "share": 0.5, "in_region_days": 2, "in_region_share": 0.5}, r
    r2 = M.recovery([None, [{"lon": 60.0}]], [60.0, 60.0])
    assert r2 == {"present_days": 1, "share": 1.0, "in_region_days": 0, "in_region_share": None}, "a day without a background is not a present day, and no standing center means no in-region days"


def test_anomaly_minima_keep_the_floor_and_the_prominence():
    M = _load()
    lons = np.arange(0.0, 5.0, 1.0)
    D = np.array([[0.0, -1.9, 0.0, 0.0, 0.0], [0.0, -2.0, 0.0, 0.0, 0.0], [0.0, -3.0, -1.0, 0.0, 0.0], [0.0, -3.0, 0.0, 0.0, 0.0]])
    Bg = np.zeros_like(D); Bg[3] = np.nan
    out = M.anomaly_minima(D, Bg, lons, -2.0)
    assert out[0] == [] and [m["lon"] for m in out[1]] == [1.0] and out[3] is None, out
    assert [m["lon"] for m in out[2]] == [1.0] and abs(out[2][0]["prominence"] - 2.0) < 1e-12, "bounded by the lower side ridge at -1, prominence exactly 2 is prominent"
    D2 = np.array([[0.0, -3.0, 0.0, -3.0, 0.0]])
    assert [m["lon"] for m in M.anomaly_minima(D2, np.zeros_like(D2), lons, -2.0)[0]] == [1.0, 3.0]
    Bp = np.zeros_like(D); Bp[1, 4] = np.nan
    assert M.anomaly_minima(D, Bp, lons, -2.0)[1] is None, "a background row missing even one longitude makes the day unavailable"
    D3 = np.array([[0.0, -3.0, -1.5, -3.0, 0.0]])        # minima at -3 split by a ridge at -1.5, prominence 1.5 each
    assert M.anomaly_minima(D3, np.zeros_like(D3), lons, -2.0)[0] == [], "a minimum of prominence below 2 m/s is not prominent"
    iso = np.array([[0.0, -3.83, -1.95, 0.0, 0.0]])
    assert M.anomaly_minima(iso, np.zeros_like(iso), lons, -2.0)[0] == [], "under the declared floor an isolated -3.83 is bounded by the -1.95 just above the floor (section 4)"
    assert [m["lon"] for m in M.anomaly_minima(iso, np.zeros_like(iso), lons, 0.0)[0]] == [1.0], "under the repaired floor the same minimum is bounded by the zero line and is prominent (section 5)"
    shallow = np.array([[0.1, -1.8, 0.1, 0.0, 0.0]])
    assert M.anomaly_minima(shallow, np.zeros_like(shallow), lons, 0.0)[0] == [], "under the repaired floor a minimum 1.9 m/s below its first non-negative cell is not prominent"


def _results(b1_ok, b25_ok, b29_ok, in_region_fail=False, b0_ok=False):
    def disp(ok):
        return {"share_near_standing": 0.0 if ok else 0.5, "days": 100}
    res = {"C1 displaced to 52 E": {"kind": "displaced", "meta": {"shift": 52.0}, "B0": disp(b0_ok), "B1": disp(b1_ok), "B2_5": disp(b25_ok), "B2_9": disp(b29_ok)},
           "C4 depth 5, 2 deg/day": {"kind": "passage", "meta": {"over_standing": True, "depth": 5.0, "speed": 2.0},
                                     **{bg: {"present_days": 20, "share": 0.9, "in_region_days": 4, "in_region_share": (0.5 if in_region_fail and bg == "B2_5" else 1.0)} for bg in ("B0", "B1", "B2_5", "B2_9")}},
           "C4 depth 5, 1 deg/day": {"kind": "passage", "meta": {"over_standing": True, "depth": 5.0, "speed": 1.0},
                                     **{bg: {"present_days": 40, "share": 0.1, "in_region_days": 6, "in_region_share": 0.0} for bg in ("B0", "B1", "B2_5", "B2_9")}},
           "C1 displaced to 52 E, with noise": {"kind": "displaced", "meta": {"shift": 52.0, "variant": True}, **{bg: {"share_near_standing": 0.9, "days": 100} for bg in ("B0", "B1", "B2_5", "B2_9")}}}
    return res


def test_the_rule_and_the_selection_are_applied_as_declared():
    M = _load()
    v, sel = M.apply_rule(_results(True, True, True))
    assert sel == "B1" and v["B0"]["passes"] is False and v["B0"]["failing_cases"] == ["C1 displaced to 52 E"], "B1 is preferred when it passes, B0 fails displacement, the noise variant and the 1 deg/day case do not enter"
    v, sel = M.apply_rule(_results(False, False, True))
    assert sel == "B2_9"
    v, sel = M.apply_rule(_results(False, True, True))
    assert sel == "B2_5", "the shorter passing window"
    v, sel = M.apply_rule(_results(False, True, False, in_region_fail=True))
    assert sel is None and v["B2_5"]["failing_cases"] == ["C4 depth 5, 2 deg/day (in region)"], "the in-region share enters the rule"
    v, sel = M.apply_rule(_results(False, False, False, b0_ok=True))
    assert v["B0"]["passes"] is True and sel is None, "B0 is never selected, even when it alone passes"
    assert M.RULE_SPEEDS == (2.0, 3.0, 5.0) and M.PASS_DISPLACED == 0.10 and M.PASS_RECOVERY == 0.80 and M.PROMINENCE == 2.0
    assert M.FLOOR_RULES == {"declared": (-2.0, "eval60_anom_constructions"), "repaired": (0.0, "eval60_anom_constructions_r1")}


def test_the_archive_crosswalk_counts_survival_by_category():
    M = _load()
    dates = ["1990-07-01", "1990-07-02"]
    per_bg = {bg: [[{"lon": 50.5}], None if bg == "B2_9" else [{"lon": 62.0}]] for bg in M.BACKGROUNDS}
    amin = {(1990, "b"): (dates, per_bg)}
    def cand(date, lon, primary, dist, prom=3.0):
        return {"date": date, "band": "b", "lon": lon, "prominence": prom, "flags": {"primary": primary}, "mask_distance_deg": dist}
    raw = [cand("1990-07-01", 50.0, True, 0.0), cand("1990-07-01", 55.0, False, 3.0), cand("1990-07-02", 61.5, False, 7.0), cand("1990-07-02", 38.0, False, 2.0),
           cand("1990-07-02", 62.0, False, None, prom=1.0), cand("1990-07-02", 64.0, False, None)]
    assert [M.category(c) for c in raw[:3]] + [M.category(raw[5])] == ["on the mask", "off, within 5 degrees", "off, farther", "no mask in the band-month"]
    s = M.survival(raw, amin, 1.0)
    assert s["B1"]["b"]["on the mask"] == {"count": 1, "available": 1, "survived": 1}
    assert s["B1"]["b"]["off, within 5 degrees"] == {"count": 1, "available": 1, "survived": 0}
    assert s["B1"]["b"]["off, farther"] == {"count": 1, "available": 1, "survived": 1} and s["B2_9"]["b"]["off, farther"] == {"count": 1, "available": 0, "survived": 0}, "a day without the background is not available"
    assert s["B1"]["b"]["no mask in the band-month"] == {"count": 1, "available": 1, "survived": 0}
    assert sum(d["count"] for d in s["B1"]["b"].values()) == 4, "west of 40 E and below the prominence class are left out"
    assert M.survival(raw, amin, 2.0)["B1"]["b"]["no mask in the band-month"]["survived"] == 1, "64 E is 2 degrees from 62 E"
    anchors = [{"class": "westward", "date": "1990-07-01", "band": "b", "anchor_lon": 51.0}, {"class": "standing", "date": "1990-07-02", "band": "b", "anchor_lon": 50.0}]
    led = M.ledger_survival(anchors, amin)
    assert led["B1"]["westward"] == {"entry_days": 1, "available": 1, "survived": 1} and led["B1"]["standing"] == {"entry_days": 1, "available": 1, "survived": 0}
    assert led["B2_9"]["standing"]["available"] == 0


def test_the_recurrence_counts_per_available_band_day_with_the_graded_value():
    M = _load()
    lons = np.arange(20.0, 75.01, 0.5)
    class Stub:
        def band_value(self, y, m, b):
            v = np.full(len(lons), 0.3); v[lons == 50.0] = 0.8; return v
    amin = {(1990, "b"): (["1990-07-01", "1990-07-02", "1990-07-03"], {bg: [[{"lon": 50.0}, {"lon": 61.0}], [], None] for bg in M.BACKGROUNDS})}
    rec = M.recurrence(amin, Stub(), lons)
    d = rec["by"]["B1"]["b"]
    edges = np.array(rec["bin_edges_deg"]); k = int(np.searchsorted(edges, 50.0, side="right") - 1)
    assert d["band_days"] == 2 and d["minima"] == 2 and abs(d["per_band_day"][k] - 0.5) < 1e-12 and abs(d["graded_median"] - 0.55) < 1e-12
    assert d["graded_tenths"] == [0, 0, 0, 1, 0, 0, 0, 0, 1, 0]


def test_main_refuses_to_overwrite_and_to_run_the_archive_without_a_selection(tmp_path):
    M = _load()
    try:
        M.main(["constructions", "--out-dir", str(tmp_path)]); assert False
    except SystemExit as ex:
        assert "--floor" in str(ex), "the floor is never implicit"
    (tmp_path / "eval60_anom_constructions_r1.md").write_text("x")
    try:
        M.main(["constructions", "--floor", "repaired", "--out-dir", str(tmp_path)]); assert False
    except SystemExit as ex:
        assert "never overwritten" in str(ex) and "_r1.md" in str(ex), "the repaired run writes its own names and refuses to overwrite them"
    cons = tmp_path / "c.json"; cons.write_text(json.dumps({"selected": None}))
    try:
        M.main(["archive", "--constructions", str(cons), "--records", str(tmp_path), "--flag-ledger", "x", "--out-dir", str(tmp_path)]); assert False
    except SystemExit as ex:
        assert "stops before the archive" in str(ex)
    cons.write_text(json.dumps({"selected": "B1"}))
    try:
        M.main(["archive", "--constructions", str(cons), "--records", str(tmp_path), "--flag-ledger", "x", "--out-dir", str(tmp_path)]); assert False
    except SystemExit as ex:
        assert "names no anomaly floor" in str(ex)


def test_the_production_case_matrix_is_the_declared_one():
    """Review of 2026-10-19, F6: the cases construction_cases() hands to run_constructions()
    are the declared ones, and the noise is the declared recipe rendered independently here."""
    M = _load()
    cases = M.construction_cases()
    names = ["C1 displaced to 52 E", "C1 displaced to 54 E", "C2 drift 48 to 52 E",
             "C3 wobble 1.5 deg, 14 days", "C3 wobble 1.5 deg, 28 days", "C3 wobble 3 deg, 14 days", "C3 wobble 3 deg, 28 days"]
    names += [f"{w} depth {d:g}, {c:g} deg/day" for w in ("C4", "C5") for d in (5.0, 10.0) for c in (1.0, 2.0, 3.0, 5.0)]
    names += ["C0 noise alone", "C1 displaced to 52 E, with noise", "C4 depth 5, 2 deg/day, with noise"]
    assert [c[0] for c in cases] == names
    by = {c[0]: c for c in cases}
    for w in ("C4", "C5"):
        for d in (5.0, 10.0):
            for c in (1.0, 2.0, 3.0, 5.0):
                _, kind, kw, meta = by[f"{w} depth {d:g}, {c:g} deg/day"]
                assert kind == "passage" and kw["transient"] == (35.0, 72.0, c, d) and meta == {"over_standing": w == "C4", "depth": d, "speed": c}
                assert ("standing" in kw) == (w == "C4") and (w == "C5" or kw["standing"](17.3) == 50.0) and "noise" not in kw
    assert by["C1 displaced to 52 E"][2]["standing"](0.0) == 52.0 and by["C1 displaced to 54 E"][2]["standing"](100.0) == 54.0
    drift = by["C2 drift 48 to 52 E"][2]["standing"]
    assert drift(0.0) == 48.0 and abs(drift(61.0) - 50.0) < 1e-12 and abs(drift(122.0) - 52.0) < 1e-12
    for amp in (1.5, 3.0):
        for per in (14.0, 28.0):
            wob = by[f"C3 wobble {amp:g} deg, {per:g} days"][2]["standing"]
            assert abs(wob(per / 4) - (50.0 + amp)) < 1e-12 and abs(wob(per / 2) - 50.0) < 1e-9 and abs(wob(3 * per / 4) - (50.0 - amp)) < 1e-12
    rng = np.random.default_rng(20261019)
    W = rng.standard_normal((488, len(M.LONS)))
    k = np.arange(-16, 17) * 0.5; ker = np.exp(-0.5 * (k / 2.0) ** 2); ker /= ker.sum()
    S = np.array([np.convolve(r, ker, mode="same") for r in W])
    N = np.zeros_like(S); N[0] = S[0]
    for i in range(1, 488):
        N[i] = 0.9 * N[i - 1] + np.sqrt(1.0 - 0.81) * S[i]
    assert np.allclose(by["C0 noise alone"][2]["noise"], N / N.std() * 2.0), "seed 20261019, a six-hourly coefficient of 0.9, a 2-degree Gaussian in longitude, 2 m/s"
    for v in ("C1 displaced to 52 E, with noise", "C4 depth 5, 2 deg/day, with noise"):
        assert np.allclose(by[v][2]["noise"], N / N.std() * 1.5) and by[v][3]["variant"] is True
    assert by["C4 depth 5, 2 deg/day, with noise"][2]["transient"] == (35.0, 72.0, 2.0, 5.0) and by["C1 displaced to 52 E, with noise"][2]["standing"](40.0) == 52.0


def test_the_production_constructions_score_as_recorded():
    """Review of 2026-10-19, F6: production cases through the scoring, against literal
    scores from the committed records under both floors."""
    M = _load()
    dec, rep = M.run_constructions(-2.0), M.run_constructions(0.0)
    assert dec["C4 depth 5, 5 deg/day"]["B1"]["share"] == 0.1 and rep["C4 depth 5, 5 deg/day"]["B1"]["share"] == 1.0
    assert dec["C1 displaced to 52 E"]["B0"]["share_near_standing"] == 1.0 and rep["C1 displaced to 52 E"]["B1"]["share_near_standing"] == 0.0
    assert rep["C5 depth 5, 1 deg/day"]["B2_5"]["share"] == 0.0 and rep["C5 depth 5, 1 deg/day"]["B2_9"]["share"] == 1.0
    assert rep["C3 wobble 1.5 deg, 28 days"]["B2_9"]["share_near_standing"] == 0.0 and rep["C3 wobble 3 deg, 14 days"]["B2_9"]["share_near_standing"] == 1.0
    assert rep["C4 depth 5, 1 deg/day"]["B1"] == {"present_days": 47, "share": 1.0, "in_region_days": 6, "in_region_share": 1.0}
    assert M.apply_rule(dec)[1] is None and M.apply_rule(rep)[1] == "B1"


def test_the_exact_path_decides_stored_precision_thresholds_exactly():
    """Review of 2026-10-19 (minima histories, F1): an exact 2 m/s prominence is prominent,
    and every exact anomaly equals the floating one at its scale."""
    M = _load()
    lons = np.array([40.0, 40.5, 41.0])
    rows, sc = M.exact_anomaly("zero", np.array([[-1440, -2240, -1440]]), np.array([6]))
    got = M.exact_anomaly_minima(rows, sc, lons, -4.0)[0]
    assert [(m["lon"], m["depth"], m["prominence"]) for m in got] == [(40.5, -5.6, 2.0)], "the floating path gives 1.9999999999999996 and drops it"
    rows, sc = M.exact_anomaly("zero", np.array([[0, -1200, 0]]), np.array([6]))
    assert M.exact_anomaly_minima(rows, sc, lons, -4.0)[0] == [], "a -3 m/s minimum is above the -4 m/s floor at every scale"
    rng = np.random.default_rng(11)
    H = rng.integers(-4000, 4000, size=(122, 7)).astype(np.int64)
    D = H / 400.0; mo = M.month_index()
    for kind in ("zero", "B1", "B2_5", "B2_9"):
        rows, sc = M.exact_anomaly(kind, H, mo)
        Bg = np.zeros_like(D) if kind == "zero" else M.background(kind, D, mo)
        for i in range(122):
            if rows[i] is None:
                assert not np.isfinite(Bg[i]).all()
            else:
                assert np.allclose(np.asarray(rows[i]) / sc[i], D[i] - Bg[i], atol=1e-9), (kind, i)
    others = [rng.integers(-4000, 4000, size=(122, 7)).astype(np.int64) for _ in range(3)]
    oms = {m: (sum(o[mo == m].sum(axis=0) for o in others), 3) for m in M.MONTHS}
    rows, sc = M.exact_anomaly("B0", H, mo, oms)
    clim = {m: np.mean([(o / 400.0)[mo == m].mean(axis=0) for o in others], axis=0) for m in M.MONTHS}
    assert all(np.allclose(np.asarray(rows[i]) / sc[i], D[i] - clim[m], atol=1e-9) for i, m in enumerate(mo))
    Hs, s_c, tr_c = M.synth_hundredths(standing=lambda t: 50.0, transient=(35.0, 72.0, 2.0, 5.0))
    D2, s2, t2 = M.synth_season(standing=lambda t: 50.0, transient=(35.0, 72.0, 2.0, 5.0))
    assert np.allclose(Hs / 400.0, D2, atol=1e-12) and np.array_equal(s_c, s2) and np.array_equal(np.nan_to_num(tr_c), np.nan_to_num(t2))
    for bad in (np.array([[1.0, np.nan]] * 4), np.array([[1.001, 0.0]] * 4)):
        try:
            M.hundredths_from_samples(bad); assert False
        except SystemExit as ex:
            assert "missing sample" in str(ex) or "hundredth lattice" in str(ex)


def test_the_retained_1999_minimum_is_prominent_on_the_exact_path():
    """Review of 2026-10-19, F1: on 1999-09-19 in 20 to 24.5 N the 59.5 E raw minimum has an
    exact prominence of 2 m/s, which the floating path computed as 1.9999999999999991."""
    import gzip
    import pytest
    path = os.path.join(ROOT, "docs", "aewc_v2", "evidence", "validation", "eastern_pilot_2026-10-05", "reports", "extent", "climatology", "eval60_regime_1999_09.json.gz")
    if not os.path.exists(path):
        pytest.skip("the retained month record is not in this tree")
    M = _load()
    rec = json.load(gzip.open(path, "rt"))
    H = M.daily_hundredths(rec, "20 to 24.5 N")["1999-09-19"]
    lons = np.array(rec["lons"], float)
    rows, sc = M.exact_anomaly("zero", H[None, :], np.array([9]))
    got = {m["lon"]: m for m in M.exact_anomaly_minima(rows, sc, lons, -4.0)[0]}
    assert 59.5 in got and got[59.5]["prominence"] == 2.0 and got[59.5]["depth"] == -5.6725

