"""Tests for scripts/eastern_energy_flux.py (`EASTERN_ENERGY_FLUX_2026-11-04.md`)."""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import eastern_energy_flux as EF  # noqa: E402

LATS = np.arange(4.0, 16.5, 1.0)
LONS = np.arange(-30.0, 0.5, 1.0)


def fields(seed=0, nlag=15):
    rng = np.random.default_rng(seed)
    shape = (nlag, len(LATS), len(LONS))
    return rng.normal(size=shape), rng.normal(size=shape), 100.0 * rng.normal(size=shape)


# ------------------------------------------------------------------ the estimator

def test_a_field_geostrophic_at_f0_has_no_flux_part_and_moves_with_the_flow():
    _, v, phi = fields()
    u = -EF.ddlat(phi, LATS, axis=-2) / EF.F0
    r = EF.energy_velocity(u, v, phi, -8.0, LATS)
    assert abs(r["c_rel"]) < 1e-12 and r["c_adv"] == pytest.approx(-8.0) and r["c_E"] == pytest.approx(-8.0)


def test_the_flux_part_is_the_weighted_sum_over_the_box_rows_only():
    u, v, phi = fields(1)
    ua = u + EF.ddlat(phi, LATS, axis=-2) / EF.F0
    rows = (LATS >= 6) & (LATS <= 14)
    w = np.cos(np.deg2rad(LATS))[rows][:, None]
    want = (ua[:, rows] * phi[:, rows] * w).sum() / (0.5 * (u[:, rows] ** 2 + v[:, rows] ** 2) * w).sum()
    assert EF.energy_velocity(u, v, phi, 0.0, LATS)["c_rel"] == pytest.approx(want)
    u2 = u.copy(); u2[:, ~rows] = 1e6
    phi2 = phi.copy(); phi2[:, (LATS < 5) | (LATS > 15)] = 1e9
    assert EF.energy_velocity(u2, v, phi2, 0.0, LATS)["c_rel"] == pytest.approx(want), "rows outside the box enter only through the derivative"
    ape = 0.5 * (u ** 2 + v ** 2)
    assert EF.energy_velocity(u, v, phi, 0.0, LATS, ape=ape)["c_rel"] == pytest.approx(want / 2), "change 1 adds the potential energy to the normalization"


def test_the_box_must_leave_a_row_beyond_it_for_the_derivative():
    u, v, phi = fields()
    with pytest.raises(SystemExit, match="beyond the box"):
        EF.energy_velocity(u, v, phi, 0.0, LATS, lat_range=(4.0, 14.0))
    with pytest.raises(SystemExit, match="beyond the box"):
        EF.energy_velocity(u, v, phi, 0.0, LATS, lat_range=(6.0, 16.0))


def test_level_weights_enter_numerator_and_denominator():
    rng = np.random.default_rng(2)
    shape = (15, 3, len(LATS), len(LONS))
    u, v, phi = rng.normal(size=shape), rng.normal(size=shape), 50 * rng.normal(size=shape)
    one = EF.energy_velocity(u[:, :1], v[:, :1], phi[:, :1], 0.0, LATS)["c_rel"]
    stacked = np.concatenate([u[:, :1]] * 3, axis=1), np.concatenate([v[:, :1]] * 3, axis=1), np.concatenate([phi[:, :1]] * 3, axis=1)
    assert EF.energy_velocity(*stacked, 0.0, LATS, weights_p=[1, 2, 3])["c_rel"] == pytest.approx(one)
    a = EF.energy_velocity(u, v, phi, 0.0, LATS, weights_p=[1, 0, 0])["c_rel"]
    assert a == pytest.approx(one)


def test_speeds_in_degrees_per_day_and_point_classes():
    assert EF.M_PER_DEG_DAY == pytest.approx(1.2674, abs=1e-4)
    assert EF.in_deg_per_day({"c_E": EF.M_PER_DEG_DAY * 2.5})["c_E"] == pytest.approx(2.5)
    assert [EF.point_class(c) for c in (1.0, 0.99, -0.99, -1.0)] == ["eastward", "standing", "standing", "westward"]


# ------------------------------------------------------------------ the packets

def test_the_model_conserves_total_energy_and_starts_in_balance():
    t, u, v, phi, pb = EF.run_packet(0.0, 30.0, 300.0, -15.0, days=2.0)
    e = [float(np.sum(0.5 * (pb[None, :, None] * (u[i] ** 2 + v[i] ** 2) + phi[i] ** 2))) for i in (0, -1)]
    assert abs(e[1] / e[0] - 1.0) < 1e-3 and len(t) == 9 and t[-1] == pytest.approx(2.0)
    f = EF.coriolis()
    psi = 1e6 * np.exp(-(EF.SW_LONS[None, :] + 15.0) ** 2 / 200.0) * np.sin(np.pi * EF.SW_LATS[:, None] / 25.0)
    bu, bv, bphi = EF.balanced_state(psi, f)
    lap = EF._ddx(EF._ddx(bphi)) + EF._ddy(EF._ddy(bphi))
    rhs = EF._ddx(f[:, None] * EF._ddx(psi)) + EF._ddy(f[:, None] * EF._ddy(psi))
    inner = slice(2, -2)
    assert np.abs(lap[inner] - rhs[inner]).max() < 0.05 * np.abs(rhs[inner]).max()
    assert np.allclose(bu, -EF._ddy(psi)) and np.allclose(bv, EF._ddx(psi))


def test_the_mean_state_is_in_geostrophic_balance_with_the_flow():
    t, u, v, phi, pb = EF.run_packet(-8.0, 30.0, 300.0, -15.0, days=0.25)
    f = EF.coriolis()
    assert np.allclose(np.gradient(pb, EF.SW_DY)[1:-1], (-f * -8.0)[1:-1], rtol=0.01)
    assert pb[0] == pytest.approx(EF.G * 300.0)


def test_the_energy_centroid_follows_a_translated_field_across_the_periodic_edge():
    lons = EF.SW_LONS
    speeds = []
    for c in (-6.0, 2.5):
        start = 20.0 if c < 0 else 80.0                        # one crosses 0 E, where the angle wraps, the other the channel's edge
        xs = [((start + c * d + 90.0) % 180.0) - 90.0 for d in np.arange(0, 8.25, 0.25)]
        u = np.array([np.exp(-((lons - x + 90) % 180 - 90) ** 2 / 50.0)[None, :] * np.ones((3, 1)) for x in xs])
        xc = EF.energy_centroid(u, 0 * u, 0 * u, np.ones(3), lons)
        speeds.append(np.polyfit(np.arange(0, 8.25, 0.25), xc, 1)[0])
    assert speeds[0] == pytest.approx(-6.0, abs=0.05) and speeds[1] == pytest.approx(2.5, abs=0.05)


def test_the_placement_theory_has_the_rossby_signs():
    assert EF.theory_group_speed(-8.0, 30.0, 300.0) < -1 < EF.theory_group_speed(2.0, 30.0, 300.0) - 2
    assert EF.theory_group_speed(0.0, 30.0, 300.0) > 0 > EF.theory_group_speed(0.0, 60.0, 300.0)


def test_a_construction_reads_its_box_and_window_around_the_base(monkeypatch):
    seen = {}
    real = EF.energy_velocity

    def spy(u, v, phi, U, lats, **kw):
        seen.setdefault("shapes", []).append((u.shape, kw.get("lat_range", EF.BOX_LATS), kw.get("ape") is not None))
        return real(u, v, phi, U, lats, **kw)
    monkeypatch.setattr(EF, "energy_velocity", spy)
    monkeypatch.setattr(EF, "SW_DAYS", 6.0)
    r = EF.construction_reading("x", {"U": -8.0, "wavelength": 30.0, "H": 300.0})
    assert [s for s, _, _ in seen["shapes"]] == [(15, len(EF.SW_LATS), 31)] * 3
    assert [(lr, a) for _, lr, a in seen["shapes"]] == [(EF.BOX_LATS, False), (EF.BOX_LATS, True), (EF.FULL_LATS, True)]
    assert 1.75 <= r["lag0_day"] <= 4.25 and set(r["versions"]) == set(EF.VERSIONS.values())
    assert r["truth"] < -1 and abs(r["energy_drift"]) < 5e-3


def test_adequacy_follows_section_7():
    good = {"a": {"truth": -5.0, "versions": {v: {"c_E": -4.5} for v in EF.VERSIONS.values()}},
            "b": {"truth": 2.0, "versions": {v: {"c_E": 2.9} for v in EF.VERSIONS.values()}}}
    assert EF.adequacy(good)["adequate"]
    off = json.loads(json.dumps(good)); off["a"]["versions"][EF.VERSIONS[0]]["c_E"] = -3.9
    assert not EF.adequacy(off)["adequate"] and EF.adequacy(off, EF.VERSIONS[1])["adequate"]
    flip = json.loads(json.dumps(good)); flip["b"]["truth"] = 1.1; flip["b"]["versions"][EF.VERSIONS[0]]["c_E"] = 0.95
    v = EF.adequacy(flip)
    assert not v["adequate"] and v["failures"] == ["b reads standing, true eastward"], "a class flip within the tolerance still fails"


def test_records_are_never_overwritten_and_tags_name_them(tmp_path, monkeypatch):
    (tmp_path / "eval60_energy_flux_constructions_r1.md").write_text("x")
    monkeypatch.setattr(EF, "run_constructions", lambda: (_ for _ in ()).throw(AssertionError("ran")))
    with pytest.raises(SystemExit, match="never overwritten"):
        EF.main(["constructions", "--out-dir", str(tmp_path), "--tag", "_r1"])


def test_the_constructions_record_holds_every_version_and_verdict(tmp_path, monkeypatch):
    res = {"a": {"spec": {"U": -8.0, "wavelength": 30.0, "H": 300.0}, "theory": -5.0, "truth": -5.2,
                 "versions": {v: {"c_rel": 1.0, "c_adv": -6.3, "c_E": -5.3} for v in EF.VERSIONS.values()}}}
    monkeypatch.setattr(EF, "run_constructions", lambda: res)
    EF.main(["constructions", "--out-dir", str(tmp_path)])
    rec = json.loads((tmp_path / "eval60_energy_flux_constructions.json").read_text())
    assert set(rec["verdicts"]) == set(EF.VERSIONS.values()) and all(v["adequate"] for v in rec["verdicts"].values())
    assert rec["generated_by"]["sha256"] == EF.SCRIPT_SHA256_AT_LOAD
    text = (tmp_path / "eval60_energy_flux_constructions.md").read_text()
    assert "| a | -8 | 30 | 300 | -5.00 | -5.20 | as declared | +1.00 | -6.30 | -5.30 | westward | westward |" in text


# ------------------------------------------------------------------ the ERA5 composites, on synthetic grids

import eastern_phase_envelope as PE  # noqa: E402
import eastern_phase_envelope_maxima as MX  # noqa: E402

SMALL_LEVELS = np.array([950.0, 750.0, 700.0, 650.0, 350.0])
SMALL_LATS = np.arange(-5.0, 30.5, 1.0)
SMALL_LONS = np.arange(-50.0, 40.5, 5.0)


@pytest.fixture
def small_grid(monkeypatch):
    monkeypatch.setattr(EF, "LEVELS", SMALL_LEVELS)
    monkeypatch.setattr(EF, "GRID_LATS", SMALL_LATS)
    monkeypatch.setattr(EF, "GRID_LONS", SMALL_LONS)
    monkeypatch.setattr(EF, "BOX_COLS", (SMALL_LONS >= -30 - 1e-9) & (SMALL_LONS <= 1e-9))
    return SMALL_LEVELS, SMALL_LATS, SMALL_LONS


def pattern(var):
    lev = np.array([1.0, 1.5, 2.0, 1.8, 0.5])[:, None, None]
    lat = np.exp(-((SMALL_LATS - 10.0) ** 2) / 50.0)[None, :, None]
    lon = np.cos(np.deg2rad(SMALL_LONS * (1 + "uvztw".index(var))))[None, None, :]
    return {"u": 3.0, "v": 2.0, "z": 300.0, "t": 0.5, "w": 0.1}[var] * lev * lat * lon


def fake_load(seed_by_year):
    def load(var, year, flux_dir=None):
        if year == 2005:
            raise SystemExit("REFUSED: the 2005 season stays unopened")
        rng = np.random.default_rng(seed_by_year[year])
        r = np.random.default_rng(1000 + seed_by_year[year]).normal(size=488)
        base = 0.0 if var != "v" else 0.0
        pat = pattern(var)
        X = r[:, None, None, None] * pat[None] + 0.002 * np.abs(pat).max() * rng.normal(size=(488,) + pat.shape) + base
        if var == "v":
            k, i, j = 2, int(np.flatnonzero(SMALL_LATS == 10)[0]), int(np.flatnonzero(SMALL_LONS == -15)[0])
            X[:, k, i, j] = r
        if var == "t":
            X = X + 290.0 - 60.0 * (1.0 - SMALL_LEVELS / 1000.0)[None, :, None, None] * 1.5
        return X
    return load


def test_lag_sums_equal_the_maxima_reading_sums_at_its_lags():
    rng = np.random.default_rng(4)
    X, r = rng.normal(size=(300, 5)), rng.normal(size=300)
    a, b = EF.lag_sums(X, r, MX.LAGS), MX.real_sums(X, r)
    assert all(np.allclose(a[k], b[k]) for k in b)


def test_a_planted_pattern_comes_back_from_the_composites(small_grid, monkeypatch):
    years = (1990, 1991, 1992)
    monkeypatch.setattr(EF, "load", fake_load({y: i for i, y in enumerate(years)}))
    totals, box, means = EF.accumulate(years)
    shape = (len(SMALL_LEVELS), len(SMALL_LATS), len(SMALL_LONS))
    lag0 = int(np.flatnonzero(EF.LAGS == 0)[0])
    for var in ("u", "z", "w"):
        cmp = EF.composite(PE.stack([totals[var]]), shape=shape)
        pat = pattern(var)
        ratio = cmp[lag0] / np.where(np.abs(pat) > 0.1 * np.abs(pat).max(), pat, np.nan)
        assert np.nanstd(ratio) < 0.02 * np.nanmean(ratio) and np.nanmean(ratio) > 0, f"{var} keeps its planted pattern up to one factor"
    assert box["u"]["sab"].shape == (3, 15, len(SMALL_LEVELS) * len(SMALL_LATS) * 7)
    mean_t = EF.weighted_mean(means["t"])
    assert mean_t.shape == shape and abs(mean_t.mean() - 290 + 60 * 1.5 * (1 - SMALL_LEVELS / 1000).mean()) < 0.1
    with pytest.raises(SystemExit, match="2005"):
        EF.accumulate((2004, 2005))


def test_the_terms_follow_section_4_on_known_fields(small_grid):
    lats, lons = SMALL_LATS, SMALL_LONS
    nl = len(SMALL_LEVELS)
    shape = (1, nl, len(lats), len(lons))
    one = np.ones(shape)
    phi = 100.0 * np.cos(np.deg2rad(lats))[None, None, :, None] * one
    u = -EF.ddlat(phi, lats, axis=-2) / EF.F0
    cmp = {"u": u, "v": 0 * one, "z": phi, "t": one, "w": 0.5 * one}
    mean = {"u": np.broadcast_to(-0.1 * lats[None, :, None], (nl, len(lats), len(lons))).copy(), "v": np.zeros((nl, len(lats), len(lons)))}
    t = EF.budget_terms(cmp, mean, lats, lons, SMALL_LEVELS)
    assert np.allclose(t["flux_x"], 0, atol=1e-9), "a field geostrophic at f0 has no zonal ageostrophic flux"
    assert np.allclose(t["K"], 0.5 * u ** 2)
    alpha = EF.RD / (SMALL_LEVELS * 100.0)
    assert np.allclose(t["baroclinic"], -0.5 * alpha[None, :, None, None] * one)
    dU = np.gradient(-0.1 * lats, np.deg2rad(lats) * EF.A)
    assert np.allclose(t["barotropic"], 0 * one), "no u v product and no zonal gradient, so no barotropic conversion"
    cmp2 = dict(cmp, v=np.ones(shape))
    t2 = EF.budget_terms(cmp2, mean, lats, lons, SMALL_LEVELS)
    assert np.allclose(t2["barotropic"], -(u * 1.0 * dU[None, None, :, None] + 0.0), rtol=1e-6)
    assert np.allclose(t2["total"], t2["gfc"] + t2["baroclinic"] + t2["barotropic"]) and np.abs(t2["barotropic"]).max() > 0


def test_the_divergence_on_the_sphere_matches_an_analytic_field():
    lats, lons = np.arange(-5.0, 30.5, 0.5), np.arange(-50.0, 40.5, 0.5)
    lam = np.deg2rad(lons)[None, :]
    fx = np.sin(lam) * np.ones((len(lats), 1))
    want = np.cos(lam) / (EF.A * np.cos(np.deg2rad(lats))[:, None])
    got = EF.divergence(fx, 0 * fx, lats, lons)
    assert np.allclose(got[:, 2:-2], want[:, 2:-2], rtol=1e-4)
    th = np.deg2rad(lats)[:, None]
    fy = np.sin(th) * np.ones((1, len(lons)))
    want_y = np.cos(2 * th) / (EF.A * np.cos(th))
    assert np.allclose(EF.divergence(0 * fy, fy, lats, lons)[2:-2], (want_y * np.ones((1, len(lons))))[2:-2], rtol=1e-3)


def test_pressure_weights_static_stability_and_potential_energy():
    w = EF.pressure_weights(np.array([950.0, 900.0, 800.0, 350.0]))
    assert w.sum() == pytest.approx(1.0) and w[0] == pytest.approx(25.0 / 600.0) and w[-1] == pytest.approx(225.0 / 600.0)
    lv = np.array([950.0, 700.0, 500.0, 350.0])
    Tbar = 300.0 * (lv / 1000.0) ** 0.19
    sig = EF.static_stability(Tbar, lv)
    assert (sig > 0).all()
    T = np.ones((1, 4, 1, 1))
    ape = EF.eddy_ape(T, sig, lv)
    assert np.allclose(ape[0, :, 0, 0], (EF.RD / (lv * 100.0)) ** 2 / (2 * sig))


def test_the_box_reading_after_changes_1_and_2_and_its_exact_form(small_grid):
    lats = SMALL_LATS
    nl = len(SMALL_LEVELS)
    shape = (15, nl, len(lats), 7)
    rng = np.random.default_rng(5)
    phi = 50.0 * rng.normal(size=shape)
    u = -EF.ddlat(phi, lats, axis=-2) / EF.F0
    v = rng.normal(size=shape)
    Tbar = 300.0 * (SMALL_LEVELS / 1000.0) ** 0.19
    mean = {"u": -5.0 * np.ones((nl, len(lats), 7)), "t": np.broadcast_to(Tbar[:, None, None], (nl, len(lats), 7)).copy()}
    cmp = {"u": u, "v": v, "z": phi, "t": np.zeros(shape)}
    r = EF.box_reading(cmp, mean)
    assert abs(r["c_rel"]) < 1e-9 and r["ape_share"] == 0.0
    assert r["c_E"] == pytest.approx(-5.0 / EF.M_PER_DEG_DAY)
    assert r["edge_term"] == pytest.approx(-(r["exact"] - r["c_E"]), abs=1e-9), "with no potential energy the two forms differ by the edge term alone"
    r850 = EF.box_reading(cmp, mean, bottom_hpa=850.0)
    assert r850["c_E"] == pytest.approx(-5.0 / EF.M_PER_DEG_DAY)
    cmp["t"] = rng.normal(size=shape)
    r2 = EF.box_reading(cmp, mean)
    assert 0 < r2["ape_share"] < 1 and r2["advective_shortfall"] == pytest.approx(-5.0 / EF.M_PER_DEG_DAY * r2["ape_share"], rel=1e-9)
    with pytest.raises(SystemExit, match="static stability"):
        EF.box_reading(cmp, dict(mean, t=np.broadcast_to((300.0 * (SMALL_LEVELS / 1000.0) ** 0.5)[:, None, None], (nl, len(lats), 7)).copy()))


def test_the_decision_classes_by_its_interval(small_grid, monkeypatch):
    years = (1990, 1991, 1992)
    monkeypatch.setattr(EF, "load", fake_load({y: i for i, y in enumerate(years)}))
    totals, box, means = EF.accumulate(years)
    d = EF.decision(box, means, n_boot=8)
    for key in ("c_E", "c_rel", "c_adv", "exact"):
        x = d[key]
        assert x["class"] == PE.classify(x["point"], x["interval"][0], x["interval"][1], x["undefined"], np.inf)
    assert set(d["point"]) >= {"c_E", "exact", "ape_share", "edge_term", "advective_shortfall"}
    assert d["lat_range"] == [-4.0, 29.0] and d["bottom_hpa"] == 950.0
    assert EF.decision(box, means, n_boot=8, bottom_hpa=650.0)["bottom_hpa"] == 650.0


def test_the_products_read_the_section_at_the_base_over_the_window(small_grid):
    lats, lons = SMALL_LATS, SMALL_LONS
    shape = (len(EF.LAGS), len(SMALL_LEVELS), len(lats), len(lons))
    terms = {k: np.zeros(shape) for k in ("K", "gfc", "baroclinic", "barotropic", "total", "flux_x", "flux_y")}
    j = int(np.flatnonzero(lons == -15)[0])
    k = int(np.flatnonzero(SMALL_LEVELS == 350.0)[0])
    terms["flux_x"][EF.WINDOW, k, :, j] = np.where(lats >= 15, -2.0, 1.0)
    terms["flux_x"][~EF.WINDOW, :, :, j] = 99.0
    i10 = int(np.flatnonzero(lats == 10)[0])
    terms["K"][:, :, i10, :] = 1.0
    terms["K"][:, :, i10, int(np.flatnonzero(lons == -25)[0])] = 5.0
    p = EF.products(terms)
    s = p["section_summary"]
    assert s["650 hPa, 12 to 18 N"] == 0.0 and s["950 to 750 hPa, 6 to 14 N"] == 0.0
    assert s["vertical mean, 15 to 20 N"] < 0 < s["vertical mean, 6 to 14 N"], "only the window's lags enter"
    assert set(p["k_max_longitude"]["6 to 14 N"]) == {-25.0} and len(p["k_max_longitude"]["6 to 14 N"]) == len(EF.LAGS)
    assert set(p["plan"]) == {"0 h", "24 h", "48 h"}


def test_gate_3_refuses_a_record_that_is_not_current_or_not_adequate(tmp_path):
    good = {"a": {"truth": -5.0, "versions": {v: {"c_E": -5.1} for v in EF.VERSIONS.values()}}}
    res = {n: good["a"] for n in EF.CONSTRUCTIONS}
    rec = {"generated_by": {"sha256": EF.SCRIPT_SHA256_AT_LOAD}, "results": res, "verdicts": {v: EF.adequacy(res, v) for v in EF.VERSIONS.values()}}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(rec)); assert EF.check_constructions(str(p))
    for change, why in ((lambda r: r["generated_by"].update(sha256="x"), "different version"),
                        (lambda r: r["results"].pop("U-8 L30 H300"), "six declared"),
                        (lambda r: r["verdicts"][EF.VERSIONS[0]].update(adequate=False), "stored verdict disagrees")):
        bad = json.loads(json.dumps(rec)); change(bad); p.write_text(json.dumps(bad))
        with pytest.raises(SystemExit, match=why):
            EF.check_constructions(str(p))
    bad = json.loads(json.dumps(rec))
    for n in bad["results"]:
        bad["results"][n]["versions"][EF.VERSIONS[2]]["c_E"] = +3.0
    bad["verdicts"] = {v: EF.adequacy(bad["results"], v) for v in EF.VERSIONS.values()}
    p.write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="not adequate"):
        EF.check_constructions(str(p))


def test_the_frozen_gate_and_the_outcome_refuse_what_they_should(monkeypatch):
    assert EF.check_frozen()
    monkeypatch.setitem(EF.FROZEN, "eastern_lag_regression", (EF.LR, "0" * 64))
    with pytest.raises(SystemExit, match="eastern_lag_regression"):
        EF.check_frozen()
    assert EF.outcome("westward") == "westward"
    with pytest.raises(SystemExit, match="not a class"):
        EF.outcome("west")


def test_the_reading_refuses_before_any_field_is_read(tmp_path, monkeypatch):
    def opened(*a, **k):
        raise AssertionError("a field was read")
    monkeypatch.setattr(EF, "load", opened)
    monkeypatch.setattr(EF, "accumulate", opened)
    with pytest.raises(SystemExit, match="required"):
        EF.main(["read", "--out-dir", str(tmp_path)])
    bad = tmp_path / "c.json"; bad.write_text(json.dumps({"generated_by": {"sha256": "x"}}))
    with pytest.raises(SystemExit, match="different version"):
        EF.main(["read", "--out-dir", str(tmp_path), "--constructions", str(bad), "--matched", "m"])
    (tmp_path / "eval60_energy_flux_plan.png").write_text("x")
    with pytest.raises(SystemExit, match="never overwritten"):
        EF.main(["read", "--out-dir", str(tmp_path), "--constructions", str(bad), "--matched", "m"])


def test_the_reading_writes_its_record_and_names_no_mechanism(small_grid, tmp_path, monkeypatch):
    years = (1990, 1991, 1992)
    monkeypatch.setattr(EF, "load", fake_load({y: i for i, y in enumerate(years)}))
    monkeypatch.setattr(EF.D, "YEARS", years)
    monkeypatch.setattr(EF, "check_constructions", lambda p: {})
    monkeypatch.setattr(EF, "gate4", lambda c, y: {"r": 0.999, "max_abs_difference": 0.01})
    monkeypatch.setattr(EF, "terrain_shares", lambda y, *rng: {f"{int(L)} hPa": 0.1 * len(rng) for L in SMALL_LEVELS})
    monkeypatch.setattr(EF, "centroid_partner", lambda: {"envelope_class": "westward", "envelope_speed": -3.0, "envelope_interval": [-3.5, -2.5]})
    weights = PE.boot_weights
    monkeypatch.setattr(PE, "boot_weights", lambda n, n_boot=5, seed=PE.BOOT_SEED: weights(n, 5, seed))
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"results": {"primary and transect": {"700": {"-15": {"envelope_class": "westward", "envelope_speed": -3.57, "envelope_interval": [-4.2, -2.9]}}}}}))
    out = tmp_path / "out"; out.mkdir()
    flux = tmp_path / "flux"; flux.mkdir()
    for v in "uvztw":
        for y in years:
            (flux / f"era5_{v}_{y}_6h_flux.nc").write_bytes(b"synthetic")
    c = tmp_path / "c.json"; c.write_text("{}")
    EF.main(["read", "--out-dir", str(out), "--constructions", str(c), "--matched", str(m), "--flux-dir", str(flux)])
    rec = json.loads((out / "eval60_energy_flux.json").read_text())
    assert rec["outcome"] == rec["decision"]["primary"]["c_E"]["class"] and rec["stored_band_mean_centroid"]["envelope_speed"] == -3.57
    assert set(rec["generated_by"]["inputs"]) == {f"era5_{v}_{y}_6h_flux.nc" for v in "uvztw" for y in years}
    text = (out / "eval60_energy_flux_tables.md").read_text()
    assert f"The outcome row of section 9 is {rec['outcome']}." in text and "| 950 hPa | 0.0000 | 0.2000 |" in text
    est = rec["generated_by"]["estimator"]
    assert est["construction_full_lats"] == [1.0, 24.0] and est["reading_full_lats"] == [-4.0, 29.0] and "full_lats" not in est
    assert rec["decision"]["primary"]["lat_range"] == est["reading_full_lats"] == rec["decision"]["from_850"]["lat_range"]
    assert rec["decision"]["primary"]["bottom_hpa"] == 950.0 and rec["decision"]["from_850"]["bottom_hpa"] == 850.0
    assert "Speeds in deg/day at 10 N (1 deg/day = 1.2674 m/s), read over 4 S to 29 N" in text and "not masked below the surface" in text
    assert all((out / f"eval60_energy_flux_{k}.png").exists() for k in ("section", "hovmoller", "plan"))
    for word in ("group velocity", "dispers", "propagat", "mechanism"):
        assert word not in text.lower()


def test_the_advective_part_is_the_energy_weighted_mean_flow():
    u, v, phi = fields(3)
    prof = np.exp(-((LATS - 13.0) ** 2) / 2.0)[None, :, None]
    u, v = u * prof, v * prof
    U = np.broadcast_to(np.linspace(-10, 2, len(LATS))[None, :, None], u.shape)
    rows = (LATS >= 6) & (LATS <= 14)
    w = np.cos(np.deg2rad(LATS))[rows][:, None]
    K = 0.5 * (u ** 2 + v ** 2)
    want = (U[:, rows] * K[:, rows] * w).sum() / (K[:, rows] * w).sum()
    assert EF.energy_velocity(u, v, phi, U, LATS)["c_adv"] == pytest.approx(want)
    assert abs(want - U[:, rows].mean()) > 0.05, "the weighting matters for this field"


def test_the_meridional_flux_and_the_convergence_have_their_signs(small_grid):
    lats, lons = SMALL_LATS, SMALL_LONS
    nl = len(SMALL_LEVELS)
    shape = (1, nl, len(lats), len(lons))
    lam = np.deg2rad(lons)[None, None, None, :]
    phi = 100.0 * np.sin(lam) * np.ones(shape)
    cmp = {"u": np.ones(shape), "v": np.zeros(shape), "z": phi, "t": np.zeros(shape), "w": np.zeros(shape)}
    mean = {"u": np.zeros((nl, len(lats), len(lons))), "v": np.zeros((nl, len(lats), len(lons)))}
    t = EF.budget_terms(cmp, mean, lats, lons, SMALL_LEVELS)
    assert np.allclose(t["flux_y"], -EF.ddlon(phi, lats, lons) / EF.F0 * phi)
    want_gfc = -EF.ddlon(phi, lats, lons)
    assert np.allclose(t["gfc"][..., 1:-1], want_gfc[..., 1:-1]) and np.abs(want_gfc).max() > 0


def test_the_box_reading_uses_the_widened_box_and_the_level_choice(small_grid):
    lats = SMALL_LATS
    nl = len(SMALL_LEVELS)
    shape = (15, nl, len(lats), 7)
    rng = np.random.default_rng(6)
    north = ((lats >= 20) & (lats <= 25))[None, None, :, None]
    u, v, phi = rng.normal(size=shape) * north, rng.normal(size=shape) * north, 50 * rng.normal(size=shape) * north
    Tbar = 300.0 * (SMALL_LEVELS / 1000.0) ** 0.19
    U = -1.0 * np.ones((nl, len(lats), 7)); U[0] = -9.0
    mean = {"u": U, "t": np.broadcast_to(Tbar[:, None, None], (nl, len(lats), 7)).copy()}
    T = rng.normal(size=shape) * north
    r = EF.box_reading({"u": u, "v": v, "z": phi, "t": T}, mean)
    assert np.isfinite(r["c_E"]), "energy north of 14 N enters the widened box"
    r0 = EF.box_reading({"u": u, "v": v, "z": phi, "t": 0 * T}, mean)
    r850 = EF.box_reading({"u": u, "v": v, "z": phi, "t": 0 * T}, mean, bottom_hpa=850.0)
    assert r850["c_adv"] == pytest.approx(-1.0 / EF.M_PER_DEG_DAY) and r0["c_adv"] < r850["c_adv"], "the 950 hPa flow leaves the reading from 850"
    rows = (lats >= -4) & (lats <= 29)
    w = np.cos(np.deg2rad(lats))[rows][:, None] * EF.pressure_weights(SMALL_LEVELS).reshape(1, -1, 1, 1)
    tot = lambda X: float((X[..., rows, :] * w).sum())
    sig = EF.static_stability(Tbar, SMALL_LEVELS)
    ape = EF.eddy_ape(T, sig, SMALL_LEVELS)
    K = 0.5 * (u ** 2 + v ** 2)
    Ub = np.broadcast_to(U[None], u.shape)
    assert r["exact"] == pytest.approx((tot(Ub * (K + ape)) + tot(u * phi)) / tot(K + ape) / EF.M_PER_DEG_DAY)


def test_the_decision_classes_need_the_interval_not_the_point(monkeypatch):
    seq = iter([{"c_E": 2.0, "c_rel": 2.0, "c_adv": 0.0, "exact": 2.0}] + [{"c_E": x, "c_rel": x, "c_adv": 0.0, "exact": x} for x in np.linspace(-1, 3, 20)])
    monkeypatch.setattr(EF, "box_inputs", lambda box, means, w=None: ({}, {}))
    seen = []
    monkeypatch.setattr(EF, "box_reading", lambda cmp, mean, lat_range=None, bottom_hpa=None: (seen.append(tuple(lat_range)), next(seq))[1])
    d = EF.decision({}, {"u": [None] * 4}, n_boot=20)
    assert d["c_E"]["point"] == 2.0 and d["c_E"]["class"] == "unresolved"
    assert set(seen) == {EF.ERA5_FULL_LATS} and len(seen) == 21, "the point and every resample read the widened box"
    assert d["lat_range"] == [-4.0, 29.0] and d["bottom_hpa"] == 950.0


def test_the_mean_flow_is_weighted_by_the_resample():
    means = [(np.array([10.0]), 2), (np.array([40.0]), 4)]
    assert EF.weighted_mean(means)[0] == pytest.approx(50.0 / 6.0)
    assert EF.weighted_mean(means, [2, 0])[0] == pytest.approx(5.0) and EF.weighted_mean(means, [0, 1])[0] == pytest.approx(10.0)


def test_the_low_levels_of_the_section_stop_at_750_hpa(small_grid):
    lats, lons = SMALL_LATS, SMALL_LONS
    shape = (len(EF.LAGS), len(SMALL_LEVELS), len(lats), len(lons))
    terms = {k: np.zeros(shape) for k in ("K", "gfc", "baroclinic", "barotropic", "total", "flux_x", "flux_y")}
    j = int(np.flatnonzero(lons == -15)[0])
    terms["flux_x"][:, int(np.flatnonzero(SMALL_LEVELS == 700.0)[0]), :, j] = 5.0
    s = EF.products(terms)["section_summary"]
    assert s["950 to 750 hPa, 6 to 14 N"] == 0.0 and s["vertical mean, 6 to 14 N"] > 0
    terms["flux_x"][:, int(np.flatnonzero(SMALL_LEVELS == 750.0)[0]), :, j] = 1.0
    assert EF.products(terms)["section_summary"]["950 to 750 hPa, 6 to 14 N"] > 0


def test_gate_4_refuses_a_composite_that_is_not_the_retained_one(monkeypatch):
    rng = np.random.default_rng(7)
    lons = np.arange(-45.0, 35.5, 1.0)
    rows = {y: rng.normal(size=(488, len(lons))) for y in (1990, 1991)}
    refs = {y: rows[y][:, 30] + 0.1 * rng.normal(size=488) for y in rows}
    monkeypatch.setattr(EF, "retained_row", lambda y, level=700, lat=10.0, lons=lons: rows[y])
    monkeypatch.setattr(EF.PR, "point_series", lambda y, level, lon: refs[y])
    old = EF.composite(EF.PE.stack([EF.lag_sums(*EF.MX.block(rows[y], refs[y])) for y in rows]))
    cmp_v = np.zeros((len(EF.LAGS), len(EF.LEVELS), len(EF.GRID_LATS), len(EF.GRID_LONS)))
    k, i = int(np.flatnonzero(EF.LEVELS == 700.0)[0]), int(np.flatnonzero(EF.GRID_LATS == 10.0)[0])
    cols = [int(np.flatnonzero(EF.GRID_LONS == x)[0]) for x in lons]
    cmp_v[:, k, i, cols] = old
    assert EF.gate4(cmp_v, tuple(rows))["r"] == pytest.approx(1.0)
    cmp_v[:, k, i, cols] = old + 0.45 * old.std() * rng.normal(size=old.shape)
    with pytest.raises(SystemExit, match="below 0.98"):
        EF.gate4(cmp_v, tuple(rows))


def test_terrain_shares_count_levels_at_or_below_the_surface(tmp_path, monkeypatch):
    import netCDF4
    d = tmp_path / "data/era5/region6h"; d.mkdir(parents=True)
    la, lo = np.arange(35.0, -25.5, -0.5), np.arange(-45.0, 75.5, 0.5)
    for y in (1990, 1991):
        f = netCDF4.Dataset(str(d / f"era5_sp_{y}_6h_region.nc"), "w")
        f.createDimension("valid_time", 2); f.createDimension("latitude", len(la)); f.createDimension("longitude", len(lo))
        f.createVariable("latitude", "f8", ("latitude",))[:] = la
        f.createVariable("longitude", "f8", ("longitude",))[:] = lo
        sp = np.full((2, len(la), len(lo)), 101000.0)
        sp[0, la == 10.0, :] = 95000.0                       # exactly the 950 hPa level on one row at one time
        f.createVariable("sp", "f8", ("valid_time", "latitude", "longitude"))[:] = sp
        f.close()
    monkeypatch.chdir(tmp_path)
    sh = EF.terrain_shares((1990, 1991))
    n_rows = int(((EF.GRID_LATS >= -4) & (EF.GRID_LATS <= 29)).sum())
    assert sh["950 hPa"] == pytest.approx(1.0 / (2 * n_rows)) and sh["925 hPa"] == 0.0


def test_gate_1_checks_each_file_against_its_own_season(tmp_path, monkeypatch):
    seen = []

    def check(path, var, year, months=EF.D.MONTHS):
        seen.append((os.path.basename(path), var, year))
        raise SystemExit("REFUSED: checked")
    monkeypatch.setattr(EF.D, "check", check)
    with pytest.raises(SystemExit, match="checked"):
        EF.load("t", 1993, str(tmp_path))
    assert seen == [("era5_t_1993_6h_flux.nc", "t", 1993)]


def _real_grid_terms(seed=3):
    rng = np.random.default_rng(seed)
    shape = (len(EF.LAGS), len(EF.LEVELS), len(EF.GRID_LATS), len(EF.GRID_LONS))
    return {k: rng.normal(size=shape) for k in ("K", "gfc", "baroclinic", "barotropic", "total", "flux_x", "flux_y")}


def test_the_950_to_750_hpa_mean_is_the_trapezoid_over_that_layer():
    terms = _real_grid_terms()
    j = int(np.flatnonzero(EF.GRID_LONS == -15.0)[0])
    terms["flux_x"][..., j] = 0.0
    terms["flux_x"][:, int(np.flatnonzero(EF.LEVELS == 750.0)[0]), :, j] = 1.0
    assert EF.products(terms)["section_summary"]["950 to 750 hPa, 6 to 14 N"] == pytest.approx(12.5 / 200.0)
    terms["flux_x"][..., j] = EF.LEVELS[None, :, None]
    assert EF.products(terms)["section_summary"]["950 to 750 hPa, 6 to 14 N"] == pytest.approx(850.0)


def test_gate_4_refuses_an_undefined_or_missing_correlation(monkeypatch):
    rng = np.random.default_rng(8)
    lons = np.arange(-45.0, 35.5, 1.0)
    rows = {y: rng.normal(size=(488, len(lons))) for y in (1990, 1991)}
    refs = {y: rows[y][:, 30] + 0.1 * rng.normal(size=488) for y in rows}
    monkeypatch.setattr(EF, "retained_row", lambda y, level=700, lat=10.0, lons=lons: rows[y])
    monkeypatch.setattr(EF.PR, "point_series", lambda y, level, lon: refs[y])
    old = EF.composite(EF.PE.stack([EF.lag_sums(*EF.MX.block(rows[y], refs[y])) for y in rows]))
    cmp_v = np.zeros((len(EF.LAGS), len(EF.LEVELS), len(EF.GRID_LATS), len(EF.GRID_LONS)))
    with pytest.raises(SystemExit, match="correlates at nan"):
        EF.gate4(cmp_v, tuple(rows))
    k, i = int(np.flatnonzero(EF.LEVELS == 700.0)[0]), int(np.flatnonzero(EF.GRID_LATS == 10.0)[0])
    cols = [int(np.flatnonzero(EF.GRID_LONS == x)[0]) for x in lons]
    cmp_v[:, k, i, cols] = old
    cmp_v[3, k, i, cols[5]] = np.nan
    with pytest.raises(SystemExit, match="not finite"):
        EF.gate4(cmp_v, tuple(rows))


def test_a_refused_gate_4_writes_nothing(small_grid, tmp_path, monkeypatch):
    years = (1990, 1991, 1992)
    monkeypatch.setattr(EF, "load", fake_load({y: i for i, y in enumerate(years)}))
    monkeypatch.setattr(EF.D, "YEARS", years)
    monkeypatch.setattr(EF, "check_constructions", lambda p: {})

    def refuse(c, y):
        raise SystemExit("REFUSED: gate 4")
    monkeypatch.setattr(EF, "gate4", refuse)
    out = tmp_path / "out"; out.mkdir()
    with pytest.raises(SystemExit, match="gate 4"):
        EF.main(["read", "--out-dir", str(out), "--constructions", "c", "--matched", "m"])
    assert list(out.iterdir()) == []


def test_the_filter_in_column_batches_equals_one_call_and_refuses_unequal_blocks():
    rng = np.random.default_rng(9)
    flat, r = rng.normal(size=(488, 1031)), rng.normal(size=488)
    F1, r1 = EF.MX.block(flat, r)
    F2, r2 = EF.filtered_block(flat, r, batch=512)
    assert F2.shape == F1.shape and np.array_equal(F2, F1) and np.array_equal(r2, r1)
    flat[:40, 700] = np.nan
    with pytest.raises(SystemExit, match="different reference blocks"):
        EF.filtered_block(flat, r, batch=512)


def test_the_figures_say_unmasked_and_the_plan_views_cover_the_counted_area(tmp_path):
    prod = EF.products(_real_grid_terms())
    paths = {k: str(tmp_path / f"{k}.png") for k in ("section", "hovmoller", "plan")}
    drawn = EF.draw(paths, prod)
    assert set(drawn["titles"]) == {"section", "hovmoller", "plan"} and all("unmasked" in s for s in drawn["titles"].values())
    assert drawn["plan_extent"] == [EF.PLAN_LONS[0], EF.PLAN_LONS[1], EF.PLAN_LATS[0], EF.PLAN_LATS[1]]
    assert all(os.path.getsize(p) > 0 for p in paths.values())
