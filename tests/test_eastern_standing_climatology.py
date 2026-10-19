"""The standing climatology must aggregate month records by calendar month as section 2
of `EASTERN_STANDING_CLIMATOLOGY_2026-10-18.md` declares: means, standard deviations and
counts over seasons with NaN cells contributing nothing, band-bin means over seasons, ridge
readings counted by the characterization's convention, candidate recurrence per band-day
at the declared floor, and the reference check against committed records."""
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("standing_climatology_under_test", os.path.join(ROOT, "scripts", "eastern_standing_climatology.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _record(year, month, frac, runs, ridge_slopes, lat, lon, band_means=None):
    bands = {"10 to 14.5 N": [10.0, 14.5], "15 to 19.5 N": [15.0, 19.5]}
    times = [f"{year}-{month:02d}-{d:02d}T{h:02d}" for d in (1, 2) for h in (0, 6, 12, 18)]
    bm = band_means if band_means is not None else {b: np.zeros((len(times), len(lon))).tolist() for b in bands}
    return {"year": year, "month": month, "label": "t", "lons": lon.tolist(), "times": times, "bands": bands,
            "persistence": {"lat": lat.tolist(), "fraction": frac.tolist(), "mean_run_days": runs.tolist()},
            "lag_correlation": {"ridges": {b: {"slope_days_per_degree": s, "speed_deg_per_day": (None if s is None or abs(s) < 1e-12 else 1.0 / s)} for b, s in zip(bands, ridge_slopes)}},
            "band_means_700": bm}


def test_persistence_climatology_averages_over_seasons_and_ignores_nan_cells():
    M = _load()
    lat = np.array([19.5, 19.0, 14.5, 14.0]); lon = np.arange(20.0, 75.01, 0.5)
    f1 = np.full((4, len(lon)), 0.2); f1[0, 0] = 0.8; f2 = np.full((4, len(lon)), 0.6); f2[0, 0] = np.nan; f3 = np.full((4, len(lon)), 0.4)
    r = [_record(1990, 6, f1, np.full((4, len(lon)), 1.0), [-0.3, 0.0], lat, lon), _record(1991, 6, f2, np.full((4, len(lon)), 3.0), [0.0, 0.5], lat, lon), _record(1990, 7, f3, np.full((4, len(lon)), 2.0), [None, -0.05], lat, lon)]
    pc = M.persistence_climatology(r)
    assert pc["6"]["seasons"] == 2 and abs(pc["6"]["mean"][1, 1] - 0.4) < 1e-12 and abs(pc["6"]["std"][1, 1] - np.std([0.2, 0.6], ddof=1)) < 1e-12 and pc["6"]["count"][1, 1] == 2
    assert pc["6"]["count"][0, 0] == 1 and abs(pc["6"]["mean"][0, 0] - 0.8) < 1e-12 and np.isnan(pc["6"]["std"][0, 0]), "a NaN cell in one season contributes nothing, so the mean is the other season's and the std undefined"
    assert pc["7"]["seasons"] == 1 and pc["all"]["seasons"] == 3 and abs(pc["all"]["mean"][1, 1] - 0.4) < 1e-12 and abs(pc["all"]["run_mean"][1, 1] - 2.0) < 1e-12
    assert pc["6"]["half_mask"].shape == (4, len(lon)) and pc["6"]["half_mask"][0, 0] and pc["6"]["half_mask"].sum() == 1, "the 0.8 cell whose other season is NaN is at least one half, nothing else is"
    boundary = np.array([[[0.7]], [[0.6]], [[0.2]]])                     # thousandths 1500 of 1500, a floating mean just below 0.5
    assert np.mean(boundary[:, 0, 0]) < 0.5 and M.at_least_half(boundary)[0, 0], "the exact 0.500 mean is decided on the stored thousandths, not on the floating mean (review of 2026-10-18, F1)"
    below = np.array([[[0.5]]] * 23 + [[[0.499]]])                      # thousandths 11999 of 12000, mean 0.4999583, displayed 0.5000 at four decimals
    assert not M.at_least_half(below)[0, 0] and round(float(np.mean(below)), 4) == 0.5
    with_nan = np.array([[[0.5]], [[np.nan]], [[0.5]]]); assert M.at_least_half(with_nan)[0, 0] and not M.at_least_half(np.full((2, 1, 1), np.nan))[0, 0]
    bt = M.band_bin_table(r, lat, lon)
    v = bt["6"]["15 to 19.5 N"]["20 to 24.5 E"]
    assert v["seasons"] == 2 and abs(v["mean"] - np.mean([np.nanmean(f1[:2, lon <= 24.5]), np.nanmean(f2[:2, lon <= 24.5])])) < 1e-12
    rc = M.ridge_climatology(r)
    assert rc["6"]["10 to 14.5 N"] == {"westward": 1, "standing": 1, "eastward": 0, "none": 0, "median_westward_speed": 1.0 / -0.3}
    assert rc["6"]["15 to 19.5 N"]["standing"] == 1 and rc["6"]["15 to 19.5 N"]["eastward"] == 1
    assert rc["7"]["10 to 14.5 N"]["none"] == 1 and rc["7"]["15 to 19.5 N"]["standing"] == 1, "a slope of magnitude 0.05 reads as standing"


def test_candidate_recurrence_counts_prominent_minima_per_band_day():
    M = _load()
    lat = np.array([19.5, 19.0, 14.5, 14.0]); lon = np.arange(20.0, 75.01, 0.5)
    rows = []
    for d in (1, 2):
        for h in (0, 6, 12, 18):
            v = np.zeros(len(lon)); v[lon == 50.0] = -10.0; v[lon == 49.5] = -7.0; v[lon == 50.5] = -7.0     # a deep candidate: ridges 0 beyond the run and -4.5 toward 51.5 E, prominence 5.5
            v[lon == 51.0] = -4.5; v[lon == 51.5] = -5.0; v[lon == 52.0] = -4.5                                  # a shallow one in the same region: the -4.5 ridge toward 50 E bounds it, prominence 0.5
            rows.append(v)
    bm = {"10 to 14.5 N": np.array(rows).tolist(), "15 to 19.5 N": np.zeros((8, len(lon))).tolist()}
    rec = _record(1990, 6, np.zeros((4, len(lon))), np.zeros((4, len(lon))), [None, None], lat, lon, band_means=bm)
    cr = M.candidate_recurrence([rec, dict(rec, year=1991)])
    e = cr["by_month_band"]["6"]["10 to 14.5 N"]
    assert e["band_days"] == 4 and abs(e["2"]["mean_crowdedness"] - 1.0) < 1e-12 and abs(e["4"]["mean_crowdedness"] - 1.0) < 1e-12, "only the deep candidate passes either floor, once per band-day"
    edges = np.array(cr["bin_edges_deg"]); i = int(np.searchsorted(edges, 50.0, side="right") - 1)
    assert abs(e["2"]["recurrence_per_band_day"][i] - 1.0) < 1e-12 and sum(e["2"]["recurrence_per_band_day"]) == 1.0 and e["2"]["top_bins"][0]["lon_lo"] == edges[i]
    assert cr["by_month_band"]["6"]["15 to 19.5 N"]["2"]["mean_crowdedness"] == 0.0


def test_reference_check_compares_the_archive_record_with_the_committed_one(tmp_path):
    M = _load()
    import gzip
    lat = np.array([19.5, 19.0, 14.5, 14.0]); lon = np.arange(20.0, 75.01, 0.5)
    rec = _record(1987, 9, np.full((4, len(lon)), 0.3), np.zeros((4, len(lon))), [None, None], lat, lon)
    (tmp_path / "ref").mkdir(); (tmp_path / "arc").mkdir()
    json.dump(rec, open(tmp_path / "ref" / "eval60_regime_1987_09.json", "w"))
    with gzip.open(tmp_path / "arc" / "eval60_regime_1987_09.json.gz", "wt") as fh:
        json.dump(rec, fh)
    assert M.check_reference(str(tmp_path / "arc"), str(tmp_path / "ref"), [(1987, 9)]) == {"1987-09": {"band_means_equal": True, "persistence_equal": True}}
    other = dict(rec, persistence={**rec["persistence"], "fraction": np.full((4, len(lon)), 0.31).tolist()})
    with gzip.open(tmp_path / "arc" / "eval60_regime_1987_09.json.gz", "wt") as fh:
        json.dump(other, fh)
    assert M.check_reference(str(tmp_path / "arc"), str(tmp_path / "ref"), [(1987, 9)])["1987-09"]["persistence_equal"] is False
    bm = {b: (np.array(v) + 0.5).tolist() for b, v in rec["band_means_700"].items()}
    with gzip.open(tmp_path / "arc" / "eval60_regime_1987_09.json.gz", "wt") as fh:
        json.dump(dict(rec, band_means_700=bm), fh)
    assert M.check_reference(str(tmp_path / "arc"), str(tmp_path / "ref"), [(1987, 9)])["1987-09"] == {"band_means_equal": False, "persistence_equal": True}
    assert M.YEARS == [y for y in range(1983, 2008) if y != 2005] and len(M.YEARS) == 24


def test_render_all_skips_present_records_and_main_refuses_an_incomplete_archive(tmp_path):
    M = _load()
    import gzip
    calls = []
    def stub(year, month, out_dir, figures=True, gzip_out=False):
        assert figures is False and gzip_out is True, "the archive run writes compressed records without figures"
        calls.append((year, month))
        with gzip.open(os.path.join(out_dir, f"eval60_regime_{year}_{month:02d}.json.gz"), "wt") as fh:
            json.dump({"year": year, "month": month}, fh)
    M.V.render_month = stub
    with gzip.open(tmp_path / "eval60_regime_1990_06.json.gz", "wt") as fh:
        json.dump({"year": 1990, "month": 6}, fh)
    res = M.render_all(str(tmp_path), years=[1990], months=(6, 7))
    assert res == {"already_present": [(1990, 6)], "rendered": [(1990, 7)]} and calls == [(1990, 7)], "a record already present is never rendered again"
    try:
        M.main(["aggregate", "--records", str(tmp_path), "--reference", str(tmp_path), "--out-dir", str(tmp_path / "out")]); assert False
    except SystemExit as ex:
        assert "2 month records, not 96" in str(ex)


def test_archive_identity_refuses_a_duplicated_season_and_a_misnamed_file_and_the_reference_set_is_the_declared_ten(tmp_path):
    """Review of 2026-10-18, F2: 96 records with one season twice and one missing were accepted by the count."""
    M = _load()
    lat = np.array([19.5, 19.0, 14.5, 14.0]); lon = np.arange(20.0, 75.01, 0.5)
    def rec(y, m):
        return _record(y, m, np.zeros((4, len(lon))), np.zeros((4, len(lon))), [None, None], lat, lon)
    files = [f"eval60_regime_{y}_{m:02d}.json.gz" for y in M.YEARS for m in M.MONTHS]
    records = [rec(y, m) for y in M.YEARS for m in M.MONTHS]
    M.check_archive_identity(files, records)
    bad = list(records); bad[1] = rec(1984, 7)      # the 1983 July file holds 1984 July
    try:
        M.check_archive_identity(files, bad); assert False
    except SystemExit as ex:
        assert "missing [(1983, 7)]" in str(ex) and "duplicated [(1984, 7)]" in str(ex), str(ex)
    misnamed = list(files); misnamed[0], misnamed[1] = misnamed[1], misnamed[0]
    try:
        M.check_archive_identity(misnamed, records); assert False
    except SystemExit as ex:
        assert "holds the record of 1983-06" in str(ex), str(ex)
    other_grid = list(records); other_grid[5] = dict(records[5], lons=(lon + 0.5).tolist())
    try:
        M.check_archive_identity(files, other_grid); assert False
    except SystemExit as ex:
        assert "not on the first record's grid" in str(ex)
    try:
        M.reference_months(str(tmp_path)); assert False
    except SystemExit as ex:
        assert "reference months are not declared" in str(ex)
    draw = {"draw": [[1994, 9], [1988, 7], [1999, 8], [1985, 9], [1995, 9], [1999, 7]], "development_months": [[1983, 6], [1987, 9], [1990, 8], [2006, 6]]}
    json.dump(draw, open(tmp_path / "eval60_regime_month_draw.json", "w"))
    for y, m in draw["draw"] + draw["development_months"][:-1]:
        json.dump(rec(y, m), open(tmp_path / f"eval60_regime_{y}_{m:02d}.json", "w"))
    try:
        M.reference_months(str(tmp_path)); assert False
    except SystemExit as ex:
        assert "missing [(2006, 6)]" in str(ex), str(ex)
    json.dump(rec(2006, 6), open(tmp_path / "eval60_regime_2006_06.json", "w"))
    assert M.reference_months(str(tmp_path)) == [(1994, 9), (1988, 7), (1999, 8), (1985, 9), (1995, 9), (1999, 7), (1983, 6), (1987, 9), (1990, 8), (2006, 6)]


def test_peaks_are_window_maxima_at_or_above_the_floor_with_a_plateau_kept_once():
    M = _load()
    lat = np.arange(34.5, 4.9, -0.5)[:20]; lon = np.arange(20.0, 75.01, 0.5)[:30]
    mean = np.full((20, 30), 0.1); std = np.full((20, 30), 0.05); count = np.full((20, 30), 24); runs = np.full((20, 30), 1.5)
    mean[5, 10] = 0.8; mean[12, 25] = 0.6; mean[15, 5] = 0.4; mean[2, 2] = 0.7; mean[2, 3] = 0.7; mean[6, 11] = 0.75
    std[5, 10] = np.nan; runs[12, 25] = 4.25
    pk = M.peaks(mean, std, count, runs, lat, lon)
    assert [(q["lon"], q["lat"], q["mean"]) for q in pk] == [(lon[10], lat[5], 0.8), (lon[2], lat[2], 0.7), (lon[25], lat[12], 0.6)], pk
    assert pk[0]["std"] is None and pk[0]["seasons"] == 24 and pk[2]["run_mean_days"] == 4.25, "the 0.75 cell beside the 0.8 peak is not a window maximum, the 0.4 cell is below the floor, the plateau appears once at its first cell"
    assert M.peaks(mean, std, count, runs, lat, lon, limit=1) == pk[:1]
    u = np.full((20, 30), 0.1); u[2:15, 2] = 0.9; u[2:15, 22] = 0.9; u[14, 2:23] = 0.9     # one connected plateau whose arms are 20 cells apart, wider than the window
    two = M.peaks(u, std, count, runs, lat, lon, floor=0.5, window=5)
    assert [(q["lon"], q["lat"]) for q in two] == [(lon[2], lat[2]), (lon[22], lat[2])], "the local rule lists a connected plateau wider than the window at more than one cell, as its docstring now says (review of 2026-10-18, F9)"


def test_composite_comparison_subtracts_the_climatology_bin_mean_from_the_composite():
    M = _load()
    lat = np.arange(35.0, 4.9, -0.5); lon = np.arange(20.0, 75.01, 0.5)
    allmean = np.full((len(lat), len(lon)), 0.3); allmean[(lat >= 15) & (lat <= 19.5)][:, :] = 0.3
    rows = (lat >= 15.0) & (lat <= 19.5); cols = (lon >= 45.0) & (lon <= 49.5)
    allmean[np.ix_(rows, cols)] = 0.55    # a clear 0.15 below the composite, so the at-least-0.10 count is not decided by float representation
    composite = {"mean_fraction_by_band_and_bin": {b: {f"{x0:g} to {x1:g} E": 0.4 for (x0, x1) in M.LON_BINS} for b in M.V.BANDS}}
    cc = M.composite_comparison(allmean, composite, lat, lon)
    r = next(x for x in cc["rows"] if x["band"] == "15 to 19.5 N" and x["bin"] == "45 to 49.5 E")
    assert abs(r["difference"] - (0.4 - 0.55)) < 1e-12 and abs(r["climatology"] - 0.55) < 1e-12
    other = next(x for x in cc["rows"] if x["band"] == "10 to 14.5 N" and x["bin"] == "45 to 49.5 E")
    assert abs(other["difference"] - 0.1) < 1e-12
    assert cc["summary"]["bins"] == 44 and cc["summary"]["composite_lower_by_0.10"] == 1 and cc["summary"]["composite_higher_by_0.10"] == 43 and abs(cc["summary"]["min"] + 0.15) < 1e-12
