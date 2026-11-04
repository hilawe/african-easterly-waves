"""Tests for scripts/eastern_phase_envelope_matched.py (`EASTERN_PHASE_ENVELOPE_MATCHED_2026-10-27.md`)."""

import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import eastern_anomaly_minima as A  # noqa: E402
import eastern_lag_regression as LR  # noqa: E402
import eastern_phase_envelope as PE  # noqa: E402
import eastern_phase_envelope_matched as M  # noqa: E402

N_T = A.SEASON_DAYS * PE.OBS_PER_DAY
UNION = np.arange(-45.0, 75.0 + 1e-9, 0.5)


def seasons_on(lons, center, c_env, n=4, seed=0):
    """Real band means of westward troughs (-7 deg/day, 30 degrees) inside a Gaussian envelope
    6 degrees wide centered at `center` and moving at `c_env`, defined on UNION and cut to
    `lons`, with a little noise so every map is finite."""
    out = []
    t = (np.arange(N_T) * PE.DT_DAYS)[:, None]
    k = -2.0 * math.pi / 30.0
    w = 2.0 * math.pi / 30.0 * 7.0
    for i in range(n):
        rng = np.random.default_rng([seed, i])
        x = UNION[None, :]
        env = np.exp(-0.5 * ((x - center - c_env * (t % 8.0)) / 6.0) ** 2)
        F = env * np.cos(w * t - k * x + rng.uniform(0, 2 * math.pi)) + 0.05 * rng.standard_normal((N_T, UNION.size))
        cols = np.flatnonzero((UNION >= lons[0] - 1e-9) & (UNION <= lons[-1] + 1e-9))
        out.append(F[:, cols])
    return out


# ------------------------------------------------------------------ gate 5, grid extent

def test_readings_do_not_depend_on_grid_beyond_the_windows_they_use():
    """A field read at 10 E on the frozen grid (10 W to 75 E) and on the matched grid (45 W to
    35 E) gives identical results while every window stays inside both."""
    a = M.results(seasons_on(PE.LONS, 10.0, 0.0), (10.0,), PE.LONS, n_boot=20)
    b = M.results(seasons_on(M.MLONS, 10.0, 0.0), (10.0,), M.MLONS, n_boot=20)
    pos = a["10"]["positions"]                     # every window clear of both grids' end columns, where the density is undefined
    assert pos and min(pos) - PE.TRACK_HALF_DEG > PE.LONS[0] + 0.25 and max(pos) + PE.TRACK_HALF_DEG < M.MLONS[-1] - 0.25
    assert M._same(json.loads(json.dumps(a)), json.loads(json.dumps(b))) == []


def test_the_same_comparison_sees_a_window_cut_by_the_frozen_grid():
    """The negative control of the test above, an envelope at 8 W whose windows run past 10 W."""
    a = M.results(seasons_on(PE.LONS, -8.0, 0.0), (-8.0,), PE.LONS, n_boot=20)
    b = M.results(seasons_on(M.MLONS, -8.0, 0.0), (-8.0,), M.MLONS, n_boot=20)
    assert M._same(json.loads(json.dumps(a)), json.loads(json.dumps(b))) != []


def test_the_reading_path_equals_the_frozen_summary_on_the_frozen_grid():
    seasons = seasons_on(PE.LONS, 30.0, 0.0, n=4)
    got = M.results(seasons, PE.BASES, PE.LONS, n_boot=7)
    want = PE.summarize([PE.analytic(S) for S in seasons], n_boot=7)
    assert M._same(json.loads(json.dumps(got)), json.loads(json.dumps(want))) == [] and list(got) == list(want)


# ------------------------------------------------------------------ gates 1 to 4

def test_the_frozen_digest_is_checked():
    assert M.check_frozen()
    with pytest.raises(SystemExit, match="frozen script's digest"):
        M.check_frozen("0" * 64)


def test_the_comparison_reports_each_kind_of_difference():
    base = {"a": 1.0, "b": [1, 2], "c": "westward", "d": None, "e": True}
    assert M._same(base, json.loads(json.dumps(base))) == []
    assert M._same(base, dict(base, a=1.0 + 1e-13)) == []
    for other in (dict(base, a=1.0 + 1e-9), dict(base, b=[1, 2, 3]), dict(base, c="standing"), dict(base, d=0.0),
                  dict(base, e=False), {k: v for k, v in base.items() if k != "e"}):
        assert M._same(base, other) != []


def test_check_builder_refuses_any_difference_from_the_stored_band_means(monkeypatch):
    monkeypatch.setattr(M, "band_means", lambda y, level, band, span: np.full((3, 2), 1.25))
    monkeypatch.setattr(LR, "season_samples", lambda recs, band: np.full((3, 2), 1.25))
    assert len(M.check_builder({1990: {}, 1991: {}}, years=(1990, 1991))) == 2
    monkeypatch.setattr(LR, "season_samples", lambda recs, band: np.full((3, 2), 1.26))
    with pytest.raises(SystemExit, match="does not reproduce the stored 700 hPa record for 1990"):
        M.check_builder({1990: {}}, years=(1990,))


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    seasons = seasons_on(PE.LONS, 30.0, 0.0, n=4)
    res = json.loads(json.dumps(M.results(seasons, PE.BASES, PE.LONS, n_boot=5)))
    path = tmp_path_factory.mktemp("stored") / "stored.json"
    rec = {"results": {M.RECORD_BAND: res}, "readings": {M.RECORD_BAND: {b: PE.reading_text(r) for b, r in res.items()}}}
    path.write_text(json.dumps(rec))
    return seasons, path, rec


def test_check_reproduction_passes_the_same_readings(stored):
    seasons, path, rec = stored
    assert M.check_reproduction(seasons, str(path), n_boot=5)


@pytest.mark.parametrize("change", ("number", "class", "reading", "base"))
def test_check_reproduction_refuses_each_kind_of_change(stored, tmp_path, change):
    seasons, path, rec = stored
    rec = json.loads(json.dumps(rec))
    r = rec["results"][M.RECORD_BAND]["30"]
    if change == "number":
        r["positions"][0] += 1e-9
    elif change == "class":
        r["envelope_class"] = "eastward" if r["envelope_class"] != "eastward" else "westward"
    elif change == "reading":
        rec["readings"][M.RECORD_BAND]["30"] += " "
    else:
        del rec["results"][M.RECORD_BAND]["70"]
    p = tmp_path / "changed.json"
    p.write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="does not reproduce the stored 700 hPa readings"):
        M.check_reproduction(seasons, str(p), n_boot=5)


def test_the_held_out_season_is_refused_before_any_file_is_opened(monkeypatch):
    netCDF4 = pytest.importorskip("netCDF4")

    def opened(*a, **k):
        raise AssertionError("a file was opened")
    monkeypatch.setattr(netCDF4, "Dataset", opened)
    with pytest.raises(SystemExit, match="held-out season"):
        M.band_means(2005, 850, M.MATCHED_BAND)


# ------------------------------------------------------------------ configuration and rules

def test_the_declared_configuration():
    assert M.SHARED_YEARS == tuple(y for y in range(1990, 2008) if y != 2005) and len(M.SHARED_YEARS) == 17
    assert len(M.ALL_YEARS) == 24 and 2005 not in M.ALL_YEARS
    assert M.MLONS[0] == -45.0 and M.MLONS[-1] == 35.0 and M.MLONS.size == 161
    assert M.BANDS == {"8 to 15 N": (8.0, 15.0, 15), "10 to 14.5 N": (10.0, 14.5, 10)}
    assert M.PRIMARY == ((850, -15.0), (700, -30.0)) and set(b for _, b in M.PRIMARY) <= set(M.TRANSECT)
    assert M.TRANSECT == (-30.0, -20.0, -15.0, -10.0, 0.0, 10.0)


def test_published_speeds_in_degrees_per_day():
    assert round(M.published_deg_per_day(3.3), 1) == 2.6 and round(M.published_deg_per_day(3.6), 1) == 2.9
    assert round(M.published_deg_per_day(-10.4), 1) == -8.2 and round(M.published_deg_per_day(-10.1), 1) == -8.0
    for ms in (3.3, 3.6, -10.4):
        assert abs(LR.speed_ms(M.published_deg_per_day(ms), M.PUBLISHED_LAT) - ms) < 1e-12


def test_direction_agreement_by_class():
    assert M.direction_agreement("eastward") == "agrees in direction"
    assert M.direction_agreement("westward") == M.direction_agreement("standing") == "disagrees in direction"
    assert M.direction_agreement("unresolved") == "decides nothing"


@pytest.mark.parametrize("classes, row", [
    (("eastward", "eastward"), "both eastward"),
    (("westward", "westward"), "both westward, both standing, or one of each"),
    (("standing", "standing"), "both westward, both standing, or one of each"),
    (("westward", "standing"), "both westward, both standing, or one of each"),
    (("eastward", "westward"), "one eastward and the other not, or either unresolved"),
    (("standing", "eastward"), "one eastward and the other not, or either unresolved"),
    (("eastward", "unresolved"), "one eastward and the other not, or either unresolved"),
    (("westward", "unresolved"), "one eastward and the other not, or either unresolved"),
    (("unresolved", "unresolved"), "one eastward and the other not, or either unresolved"),
])
def test_the_outcome_rows_of_section_5(classes, row):
    assert M.outcome(classes) == row


def test_the_outcome_needs_both_configurations():
    with pytest.raises(SystemExit, match="both primary"):
        M.outcome(("eastward",))


def fake_res(env850="standing", env700="westward"):
    def one(env, ph="westward"):
        return {"envelope_class": env, "phase_class": ph}
    res = {name: {"850": {"-15": one(env850)}, "700": {"-30": one(env700)}} for name, _, _ in M.READINGS}
    return res


def test_dependence_names_the_sensitivity_that_changes_a_class():
    res = fake_res()
    assert M.dependence(res) == {"850 hPa, -15 E": [], "700 hPa, -30 E": []}
    res["band sensitivity"]["850"]["-15"]["envelope_class"] = "eastward"
    res["season sensitivity"]["700"]["-30"]["phase_class"] = "unresolved"
    assert M.dependence(res) == {"850 hPa, -15 E": ["band sensitivity"], "700 hPa, -30 E": ["season sensitivity"]}


def test_the_summary_carries_the_outcome_and_the_published_speeds():
    s = M.summary(fake_res("eastward", "eastward"))
    assert s["outcome"] == "both eastward" and s["direction"]["850 hPa, -15 E"] == "agrees in direction"
    assert round(s["published_deg_per_day"]["700 hPa, -30 E"]["maxima"], 2) == 2.85


def test_the_generated_text_names_no_mechanism():
    seasons = seasons_on(M.MLONS, -15.0, 0.0, n=3)
    one = json.loads(json.dumps(M.results(seasons, M.TRANSECT, M.MLONS, n_boot=5)))
    res = {name: {"850": one, "700": one} for name, _, _ in M.READINGS}
    text = M.matched_text(res, M.summary(res)).lower()
    assert "outcome row of section 5" in text and "regressed wind maxima moving at +2.62" in text
    for word in ("group", "dispers", "packet", "propagat", "energy", "mechanism"):
        assert word not in text


def test_records_are_never_overwritten(tmp_path):
    (tmp_path / "eval60_phase_env_matched.json").write_text("{}")
    with pytest.raises(SystemExit, match="never overwritten"):
        M.main(["read", "--xband", str(tmp_path), "--stored", str(tmp_path / "x.json"), "--out-dir", str(tmp_path)])


def test_the_reading_needs_an_output_directory(tmp_path):
    with pytest.raises(SystemExit, match="out-dir is required"):
        M.main(["read", "--xband", str(tmp_path), "--stored", str(tmp_path / "x.json")])


# ------------------------------------------------------------------ the retained fields

ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "era5", "region6h", "era5_v850_1990_6h_region.nc")
XBAND = os.path.join(ROOT, "docs", "aewc_v2", "evidence", "validation", "eastern_pilot_2026-10-05", "reports", "extent", "xband")


@pytest.mark.skipif(not (os.path.exists(DATA) and os.path.isdir(XBAND)), reason="the retained fields are not on this machine")
def test_the_builder_on_the_retained_fields(monkeypatch):
    monkeypatch.chdir(ROOT)
    S = M.band_means(1990, 850, M.MATCHED_BAND)
    assert S.shape == (N_T, 161) and np.isfinite(S).all() and np.array_equal(S, np.round(S, 2))
    with pytest.raises(SystemExit, match="missing sample"):
        M.band_means(1990, 850, M.MATCHED_BAND, span=(-45.0, 40.0))
    with pytest.raises(SystemExit, match="half a degree"):
        M.band_means(1990, 700, M.MATCHED_BAND, span=(-45.25, 35.0))
    monkeypatch.setitem(M.BANDS, "short", (8.0, 15.0, 14))
    with pytest.raises(SystemExit, match="holds 15 rows, not 14"):
        M.band_means(1990, 700, "short")
    files, by = PE.load_xband(XBAND)
    assert len(M.check_builder(by, years=(1990,))) == 1
