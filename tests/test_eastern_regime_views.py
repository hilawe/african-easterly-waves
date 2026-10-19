"""The regime characterization's derived numbers must behave as section 3 of
`EASTERN_REGIME_CHARACTERIZATION_2026-10-18.md` declares: ten-row bands or refusal, runs
counted with censoring at the month's ends and broken by invalid steps, and a lag and
offset correlation whose ridge tilts with an imposed westward movement and stands for a
standing anomaly."""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("regime_views_under_test", os.path.join(ROOT, "scripts", "eastern_regime_views.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bands_hold_ten_rows_each_or_refuse():
    M = _load()
    lat = np.arange(35.0, 4.9, -0.5)
    rows = M.band_rows(lat)
    assert list(rows) == ["10 to 14.5 N", "15 to 19.5 N", "20 to 24.5 N", "25 to 29.5 N"] and all(r.size == 10 for r in rows.values())
    assert sorted(lat[rows["20 to 24.5 N"]]) == [20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5]
    try:
        M.band_rows(np.arange(35.0, 4.9, -1.0)); assert False, "integer rows do not make ten-row bands"
    except SystemExit as e:
        assert "not 10" in str(e)


def test_persistence_counts_runs_with_censoring_and_invalid_breaks():
    M = _load()
    nt = 20
    v = np.zeros((nt, 1, 3)); sp = np.full((nt, 1, 3), 101325.0)
    v[2:6, 0, 0] = -6.0; v[10:13, 0, 0] = -5.0          # two interior runs of 4 and 3 steps
    v[0:5, 0, 1] = -6.0; v[17:20, 0, 1] = -6.0           # runs touching both ends, censored
    v[4:12, 0, 2] = -6.0; sp[8, 0, 2] = 69000.0          # one run broken by a below-ground step
    p = M.persistence(v, sp)
    assert p["uncensored_runs"][0, 0] == 2 and abs(p["mean_run_days"][0, 0] - 3.5 * 0.25) < 1e-12 and abs(p["fraction"][0, 0] - 7 / 20) < 1e-12
    assert p["uncensored_runs"][0, 1] == 0 and p["censored_runs"][0, 1] == 2 and np.isnan(p["mean_run_days"][0, 1])
    assert p["uncensored_runs"][0, 2] == 2 and p["valid_steps"][0, 2] == 19 and p["invalid_at_any_step"] == 1
    assert abs(p["mean_run_days"][0, 2] - (4 + 3) / 2 * 0.25) < 1e-12, "the invalid step at index 8 splits 4 to 11 into 4 to 7 and 9 to 11"


def test_lag_offset_ridge_tilts_with_westward_movement_and_stands_otherwise():
    M = _load()
    lons = np.arange(20.0, 75.01, 0.5)
    t = np.arange(120) * 0.25
    X, T = np.meshgrid(lons, t)
    moving = -8.0 * np.exp(-0.5 * ((X - (62.0 - 2.0 * T)) / 3.0) ** 2)      # 2 degrees per day westward
    standing = -8.0 * np.exp(-0.5 * ((X - 52.0) / 3.0) ** 2) * (1.0 + 0.6 * np.sin(2 * np.pi * T / 5.0))
    cm = M.lag_offset_correlation(moving, lons); rm = M.ridge(cm)
    assert rm["speed_deg_per_day"] is not None and -2.6 < rm["speed_deg_per_day"] < -1.5, rm
    cs = M.lag_offset_correlation(standing, lons); rs = M.ridge(cs)
    assert rs["slope_days_per_degree"] is not None and abs(rs["slope_days_per_degree"]) < 0.05, ("a standing anomaly's ridge lies along zero lag", rs)
    assert np.isnan(cm).sum() < cm.size and abs(cm[M.OFFSET_CELLS, M.LAG_STEPS] - 1.0) < 1e-9, "zero offset at zero lag correlates perfectly"
    sparse = np.full((120, len(lons)), np.nan); sparse[:, 60] = np.sin(t)
    assert M.ridge(M.lag_offset_correlation(sparse, lons))["speed_deg_per_day"] is None, "fewer than three qualifying offsets give no speed"


def test_month_window_and_the_held_out_refusal():
    M = _load()
    s, e = M.month_window(1987, 9)
    assert s.month == 9 and s.day == 1 and s.hour == 0 and e.month == 9 and e.day == 30 and e.hour == 18
    try:
        M.load_month(2005, 7); assert False, "2005 is never opened"
    except SystemExit as ex:
        assert "held-out" in str(ex)


def test_the_ridge_uses_the_declared_correlation_floor():
    M = _load()
    assert M.RIDGE_MIN_CORR == 0.3
    corr = np.full((2 * M.OFFSET_CELLS + 1, 2 * M.LAG_STEPS + 1), 0.0)
    for di in range(corr.shape[0]):
        d = (di - M.OFFSET_CELLS) * 0.5
        lag_index = M.LAG_STEPS + int(round(-d / 0.5)) if abs(d) <= 3.0 else M.LAG_STEPS   # a ridge at two degrees per day inside the lag range, flat beyond it
        corr[di, lag_index] = 0.5 if abs(d) <= 1.0 else 0.2                                # strong only within one degree
    r = M.ridge(corr)
    assert r["points"] == 5, ("only the offsets whose maximum exceeds 0.3 enter the fit", r)
    assert abs(r["slope_days_per_degree"] + 0.5) < 1e-9 and abs(r["speed_deg_per_day"] + 2.0) < 1e-9, r
    assert M.ridge(corr, min_corr=0.1)["points"] == 41, "a lower floor admits every offset"


def test_render_without_figures_writes_a_compressed_record_and_no_figure(tmp_path):
    """The standing climatology's archive run (2026-10-18): figures=False and gzip_out=True."""
    M = _load()
    import datetime as dt, gzip, json, os
    lat = np.arange(35.0, 4.9, -0.5); lon = np.arange(20.0, 75.01, 0.5)
    nt = 40      # ten days, more than twice the lag window, as every real month is
    X, Y = np.meshgrid(lon, lat)
    def synthetic(year, month, data_dir=None):
        v700 = np.stack([-10.0 * np.exp(-0.5 * ((X - (56.0 - 0.5 * k)) / 3.0) ** 2 - 0.5 * ((Y - 20.0) / 3.0) ** 2) for k in range(nt)])
        sp = np.full(v700.shape, 101325.0)
        times = [dt.datetime(2000, 7, 1) + dt.timedelta(hours=6 * k) for k in range(nt)]
        return {"year": year, "month": month, "times": times, "lat": lat, "lon": lon, "v700": v700, "v850": v700, "v600": v700, "u700": v700 * 0, "u925": v700 * 0, "v925": v700 * 0, "sp": sp,
                "files": {"v700": {"path": "synthetic", "sha256": "0" * 64}}}
    M.load_month = synthetic
    rec = M.render_month(2000, 7, str(tmp_path), figures=False, gzip_out=True)
    assert rec["figures"] is None and sorted(os.listdir(tmp_path)) == ["eval60_regime_2000_07.json.gz"]
    with gzip.open(tmp_path / "eval60_regime_2000_07.json.gz", "rt") as fh:
        back = json.load(fh)
    assert back["steps"] == nt and set(back["band_means_700"]) == set(M.BANDS) and back["figures"] is None
    try:
        M.render_month(2000, 7, str(tmp_path), figures=False, gzip_out=True); assert False
    except SystemExit as ex:
        assert "never overwritten" in str(ex)
    cli = tmp_path / "cli"; cli.mkdir()
    M.main(["render", "--year", "2000", "--month", "7", "--out-dir", str(cli), "--no-figures", "--gzip"])
    assert sorted(os.listdir(cli)) == ["eval60_regime_2000_07.json.gz"], "the command-line flags reach render_month"

