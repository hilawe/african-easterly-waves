"""The standing-element flag must implement section 2 of `EASTERN_STANDING_FLAG_2026-10-18.md`
and pass the adequacy constructions of its section 3 before any real position is flagged:
leaving out the own season changes the decision where it should, the flag is co-location
and not identity, decisions are made in stored precision, and the flagged sets nest by
threshold."""
import csv
import importlib.util
import json
import os
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
LONS = np.arange(20.0, 75.01, 0.5)


def _load():
    spec = importlib.util.spec_from_file_location("standing_flag_under_test", os.path.join(ROOT, "scripts", "eastern_standing_flag.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _record(year, month, daily_rows, band="b", lat=(20.0, 19.5)):
    """A month record whose band means repeat each daily row at the four six-hourly hours."""
    times, series = [], []
    for d, row in enumerate(daily_rows, 1):
        for h in ("00", "06", "12", "18"):
            times.append(f"{year}-{month:02d}-{d:02d}T{h}"); series.append(np.round(row, 2).tolist())
    return {"year": year, "month": month, "lons": LONS.tolist(), "times": times, "bands": {band: [15.0, 19.5]},
            "band_means_700": {band: series}, "persistence": {"lat": list(lat), "fraction": np.zeros((len(lat), len(LONS))).tolist()}}


def test_leaving_out_the_own_season_changes_the_decision_where_it_should():
    M = _load()
    north = set(M.YEARS[:12])
    fr = {(y, 6, "b"): (np.full(3, 30 if y in north else 0), np.full(3, 30)) for y in M.YEARS}
    assert M.band_mask(fr, M.YEARS[0], 6, "b", Fraction(1, 2), leave_out=False).all(), "12 of 24 seasons fully northerly is a mean of exactly one half, in the all-season mask"
    for y in M.YEARS:
        got = M.band_mask(fr, y, 6, "b", Fraction(1, 2), leave_out=True)
        assert (not got.any()) if y in north else got.all(), (y, got)
    th = {y: np.full((2, 3), 1000.0 if y in north else 0.0) for y in M.YEARS}
    assert M.cell_mask(th, M.YEARS[0], Fraction(1, 2), leave_out=False).all()
    for y in M.YEARS:
        got = M.cell_mask(th, y, Fraction(1, 2), leave_out=True)
        assert (not got.any()) if y in north else got.all(), (y, got)


def test_the_flag_is_colocation_and_not_identity():
    """A standing minimum at 50 E in every season and, in 1990 only, a transient minimum
    moving west from 60 E at 2 degrees per day."""
    M = _load()
    def gauss(c):
        return -10.0 * np.exp(-0.5 * ((LONS - c) / 1.0) ** 2)
    records = []
    for y in M.YEARS:
        rows = []
        for d in range(30):
            v = gauss(50.0)
            if y == 1990 and d <= 10:
                v = v + gauss(60.0 - 2.0 * d)
            rows.append(v)
        rec = _record(y, 6, rows)
        rec["bands"]["c"] = [20.0, 24.5]                                  # a second band with its own standing minimum at 30 E
        rec["band_means_700"]["c"] = [np.round(gauss(30.0), 2).tolist() for _ in rec["times"]]
        records.append(rec)
    masks = M.Masks(records)
    cands = M.flag_candidates([r for r in records if r["year"] == 1990], masks)
    in_c = [c for c in cands if c["band"] == "c"]
    assert len(in_c) == 30 and all(c["lon"] == 30.0 and c["flags"]["primary"] for c in in_c), "each band's candidates are tested against their own band's mask"
    cands = [c for c in cands if c["band"] == "b"]
    by_day = {}
    for c in cands:
        by_day.setdefault(c["date"], {})[c["lon"]] = c
    for d in (0, 1, 2, 3, 7, 8, 9, 10):
        day = f"1990-06-{d + 1:02d}"; x = 60.0 - 2.0 * d
        assert x in by_day[day] and not by_day[day][x]["flags"]["primary"], (day, x, sorted(by_day[day]))
        assert by_day[day][50.0]["flags"]["primary"], "the standing candidate is on the mask"
        assert not by_day[day][x]["flags"]["all_seasons"], "the transient's own season does not put its path on the all-season mask either"
        assert by_day[day][x]["loo_fraction"] == 0.0 and by_day[day][50.0]["loo_fraction"] == 1.0, "the graded value leaves the own season out, so the transient's path holds none of its own northerly days"
    assert by_day["1990-06-06"][50.0]["flags"]["primary"] and sorted(by_day["1990-06-06"]) == [50.0], "the transient's candidate on the day it lies at 50 E is flagged, so a flag is co-location and never identity"
    assert sorted(by_day["1990-06-05"]) == [51.0] and sorted(by_day["1990-06-07"]) == [49.0] and all(by_day[d][x]["flags"][f] for d, x in (("1990-06-05", 51.0), ("1990-06-07", 49.0)) for f in M.FORMS), "on the days beside the overlap the two Gaussian minima merge into one candidate on the mask (review of 2026-10-18, F2)"
    assert all(c["flags"]["primary"] for c in cands if c["date"] >= "1990-06-12"), "after the transient leaves only the standing candidate remains"


def test_decisions_are_made_in_stored_precision():
    M = _load()
    rec = _record(2000, 6, [np.full(len(LONS), 0.0)])
    S = np.zeros((4, len(LONS)))
    S[:, 0] = [-4.1, -3.9, -4.3, -3.7]          # sums to -1600 hundredths exactly
    S[:, 1] = [-3.99, -4.0, -4.0, -4.0]         # -1599, not northerly
    S[:, 2] = [-4.0, -4.0, -4.0, -4.0]
    S[:, 3] = [-9.0, np.nan, -9.0, -9.0]        # unavailable
    rec["band_means_700"]["b"] = S.tolist()
    north, avail = M.band_day_counts(rec, "b")
    assert north[:4].tolist() == [1, 0, 1, 0] and avail[:4].tolist() == [1, 1, 1, 0]
    th = {y: np.full((1, 2), 400.0) for y in M.YEARS}
    th[M.YEARS[1]][0, 1] = np.nan                # a season that stays in the sum when the first is left out
    assert M.cell_mask(th, M.YEARS[0], Fraction(2, 5)).all() and not M.cell_mask(th, M.YEARS[0], Fraction(1, 2)).any(), "a sum of thousandths exactly at the threshold is in, and a NaN season contributes nothing, neither to the sum nor to the count"
    th2 = {y: np.full((1, 1), 500.0) for y in M.YEARS}; th2[M.YEARS[3]][0, 0] = 499.0
    assert not M.cell_mask(th2, M.YEARS[0], Fraction(1, 2))[0, 0] and M.cell_mask(th2, M.YEARS[3], Fraction(1, 2))[0, 0], "one thousandth below is out, and leaving that season out restores the cell"
    fr = {(y, 6, "b"): (np.array([1]), np.array([3])) for y in M.YEARS}
    assert not M.band_mask(fr, M.YEARS[0], 6, "b", Fraction(2, 5))[0] and M.band_mask(fr, M.YEARS[0], 6, "b", Fraction(1, 3))[0], "one third against two fifths and against one third, exactly"
    fr0 = {(y, 6, "b"): (np.array([0]), np.array([0])) for y in M.YEARS}
    assert not M.band_mask(fr0, M.YEARS[0], 6, "b", Fraction(0))[0], "a longitude with no available band-day in any season is out"


def test_flagged_sets_nest_by_threshold():
    M = _load()
    rng = np.random.default_rng(7)
    fr = {(y, 6, "b"): (rng.integers(0, 31, 40), np.full(40, 30)) for y in M.YEARS}
    th = {y: np.rint(rng.uniform(0, 1, (5, 8)) * 1000) for y in M.YEARS}
    for y in M.YEARS[:5]:
        b4, b5, b6 = (M.band_mask(fr, y, 6, "b", t) for t in (Fraction(2, 5), Fraction(1, 2), Fraction(3, 5)))
        c4, c5, c6 = (M.cell_mask(th, y, t) for t in (Fraction(2, 5), Fraction(1, 2), Fraction(3, 5)))
        assert not (b6 & ~b5).any() and not (b5 & ~b4).any() and not (c6 & ~c5).any() and not (c5 & ~c4).any()
    assert set(M.FORMS) == {"primary", "t0.4", "t0.6", "all_seasons"} and M.FORMS["primary"] == (Fraction(1, 2), True) and M.FORMS["all_seasons"] == (Fraction(1, 2), False)


class _Stub:
    def __init__(self, cell=None, band=None, value=None):
        self._c, self._b, self._v = cell, band, value

    def cell_value(self, y, m):
        return self._v

    def cell(self, y, m, f):
        return self._c[f]

    def band(self, y, m, b, f):
        return self._b[f]


def test_selected_positions_are_flagged_by_their_cell_with_off_mask_alternatives_counted(tmp_path):
    M = _load()
    lat = np.array([21.0, 20.0, 19.0]); lon = np.array([44.0, 45.0, 50.0, 55.0, 70.0])
    on = np.zeros((3, 5), bool); on[1, 2] = True                 # only 20 N 50 E on the primary mask
    wide = np.ones((3, 5), bool)
    val = np.full((3, 5), 0.2); val[1, 2] = 0.7
    stub = _Stub(cell={"primary": on, "t0.4": wide, "t0.6": np.zeros((3, 5), bool), "all_seasons": on}, value=val)
    cols = ["interval", "time", "row_lat", "status", "window_lon_lo", "window_lon_hi", "selected_lon", "selected_v_ms", "all_minima", "motion_label"]
    rows = [{"interval": "x", "time": "1990-08-10T00", "row_lat": "20", "status": "assigned", "window_lon_lo": "44", "window_lon_hi": "63", "selected_lon": "50.0", "selected_v_ms": "-9.5",
             "all_minima": "45:-6.0:I1;50:-9.5:I1;55:-3.0:I1;70:-8.0:other", "motion_label": "west"},
            {"interval": "x", "time": "1990-08-11T00", "row_lat": "20", "status": "assigned", "window_lon_lo": "44", "window_lon_hi": "63", "selected_lon": "50.0", "selected_v_ms": "-7.0",
             "all_minima": "44:-7.0:I1;45:-6.5:I1;50:-7.0:I1", "motion_label": "west"},
            {"interval": "x", "time": "1990-08-10T06", "row_lat": "20", "status": "assigned", "window_lon_lo": "44", "window_lon_hi": "63", "selected_lon": "45.0", "selected_v_ms": "-6.0",
             "all_minima": "45:-6.0:I1;50:-9.5:I1;55:-5.0:I1", "motion_label": ""},
            {"interval": "x", "time": "1990-08-10T12", "row_lat": "19", "status": "abstained", "window_lon_lo": "44", "window_lon_hi": "63", "selected_lon": "", "selected_v_ms": "",
             "all_minima": "", "motion_label": ""},
            {"interval": "x", "time": "1990-08-10T18", "row_lat": "20", "status": "abstained", "window_lon_lo": "44", "window_lon_hi": "63", "selected_lon": "50.0", "selected_v_ms": "-9.0",
             "all_minima": "50:-9.0:I1", "motion_label": ""}]
    with open(tmp_path / "x.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    sel = M.flag_selected(str(tmp_path), ["x"], stub, lat, lon)
    assert len(sel) == 3, "abstained row-steps are not flagged, even one that carries a longitude"
    a, c3, b = sel
    assert a["other_window_minima_off_mask_at_least_as_deep"] == 0 and c3["other_window_minima_off_mask_at_least_as_deep"] == 1 and c3["other_window_minima_off_mask"] == 2, "an off-mask minimum as deep as the selection (-7.0 at 44 E) counts, a shallower one (-6.5) does not"
    assert a["loo_cell_fraction"] == 0.7 and b["loo_cell_fraction"] == 0.2
    assert b["other_window_minima_off_mask_at_least_as_deep"] == 0 and b["other_window_minima_off_mask"] == 1, "a deeper alternative on the mask (-9.5 at 50 E) is not an off-mask alternative, and a shallower one off it (-5.0 at 55 E) is not as deep"
    assert a["flags"] == {"primary": True, "t0.4": True, "t0.6": False, "all_seasons": True} and a["other_window_minima"] == 1 and a["other_window_minima_off_mask"] == 1, a
    assert b["flags"]["primary"] is False and b["motion_label"] == "unlabeled" and b["other_window_minima"] == 2 and b["other_window_minima_off_mask"] == 1, b
    st = M.selected_tables(sel)
    assert st["alternatives"] == {"flagged": 2, "flagged_with_off_mask_alternative": 2, "flagged_with_off_mask_alternative_at_least_as_deep": 1, "unflagged": 1}
    assert st["by_interval"]["x"] == {"assigned": 3, "flagged": {"primary": 2, "t0.4": 3, "t0.6": 0, "all_seasons": 2}, "flagged_with_off_mask_alternative": 2} and st["by_latitude"]["20 to 24 N"]["assigned"] == 3
    assert st["by_latitude"]["20 to 24 N"]["flagged_with_off_mask_alternative"] == 2 and st["by_motion_label"]["unlabeled"]["flagged_with_off_mask_alternative"] == 0, "an unflagged row with an off-mask alternative is not counted"
    assert st["by_motion_label"]["west"]["flagged"]["primary"] == 2 and st["by_motion_label"]["unlabeled"]["flagged"]["primary"] == 0
    assert st["graded"]["histogram_tenths"] == [0, 0, 1, 0, 0, 0, 0, 2, 0, 0] and abs(st["graded"]["quantiles"][50] - 0.7) < 1e-12


def test_ledger_anchors_are_flagged_by_the_band_mask_and_counted_by_class():
    M = _load()
    m = np.zeros(len(LONS), bool); m[LONS == 50.0] = True
    stub = _Stub(band={f: m for f in M.FORMS})
    led = [{"id": "E1", "year": 1990, "month": 8, "band": "b", "speeds": {"direction": "westward"},
            "days": [{"date": "1990-08-01", "anchored": True, "anchor_lon": 50.0, "edge": False}, {"date": "1990-08-02", "anchored": True, "anchor_lon": 48.0, "edge": True},
                     {"date": "1990-08-03", "anchored": False, "anchor_lon": None, "edge": False}]},
           {"id": "E2", "year": 1990, "month": 8, "band": "b", "speeds": {"direction": None}, "days": [{"date": "1990-08-05", "anchored": True, "anchor_lon": 50.0, "edge": False}]}]
    rows = M.flag_ledger(led, stub, LONS)
    assert [r["flags"]["primary"] for r in rows] == [True, False, True]
    t = M.ledger_table(rows)
    assert t["westward"] == {"entries": 1, "anchored_days": 2, "flagged": {f: 1 for f in M.FORMS}} and t["no fit"]["anchored_days"] == 1


def test_candidate_tables_and_the_layer_reproduction():
    M = _load()
    recs = [_record(1990, 6, [np.zeros(len(LONS))] * 30), _record(1991, 7, [np.zeros(len(LONS))] * 31)]
    def c(y, m, lon, p, flag, frac=0.6, dist=0.0):
        return {"year": y, "month": m, "band": "b", "date": f"{y}-{m:02d}-01", "lon": lon, "depth": -6.0, "prominence": p, "flags": {f: flag for f in M.FORMS}, "loo_fraction": frac, "mask_distance_deg": dist}
    cands = [c(1990, 6, 50.0, 3.0, True), c(1990, 6, 36.0, 5.0, False, 0.05, 1.5), c(1991, 7, 41.0, 1.0, True), c(1991, 7, 75.0 - 0.5, 2.0, False, 0.35, None),
             c(1990, 6, 30.0, 2.5, False, 0.0, 2.0)]      # off the mask by exactly 2 degrees, in the first distance class only
    cands[-1]["date"] = "1990-06-02"
    ct = M.candidate_tables(cands, recs)
    p2 = ct["pooled"]["p2"]["b"]
    assert p2["count"] == 4 and p2["flagged"]["primary"] == 1 and p2["east_count"] == 2 and p2["east_flagged"]["primary"] == 1
    assert ct["pooled"]["all"]["b"]["count"] == 5 and ct["pooled"]["p4"]["b"]["count"] == 1 and ct["pooled"]["p4"]["b"]["east_count"] == 0
    assert ct["by_month_band"]["p2"]["6"]["b"]["count"] == 3 and ct["by_month_band"]["p2"]["7"]["b"]["count"] == 1
    assert ct["band_days"]["b"] == {"all": 61, "6": 30, "7": 31, "8": 0, "9": 0}
    edges = np.array(ct["bin_edges_deg"]); k = int(np.searchsorted(edges, 50.0, side="right") - 1)
    assert ct["by_bin"]["p2"]["b"]["all"][k] == {"count": 1, "flagged": {f: 1 for f in M.FORMS}} and ct["by_bin"]["p2"]["b"]["6"][k] == {"count": 1, "flagged": {f: 1 for f in M.FORMS}}
    k36 = int(np.searchsorted(edges, 36.0, side="right") - 1)
    assert ct["by_bin"]["p2"]["b"]["all"][k36] == {"count": 1, "flagged": {f: 0 for f in M.FORMS}}
    assert ct["by_bin"]["p2"]["b"]["all"][-1]["count"] == 1, "a candidate at 74.5 E falls in the last bin"
    g = ct["graded"]["b"]
    assert g["all"]["count"] == 4 and g["all"]["histogram_tenths"] == [2, 0, 0, 1, 0, 0, 1, 0, 0, 0] and g["all"]["off_mask"] == 3
    assert g["all"]["off_mask_by_distance"] == {"0.5 to 2 degrees": 2, "2.5 to 5 degrees": 0, "more than 5 degrees": 0, "no mask in the band-month": 1}
    assert g["east"]["count"] == 2 and g["east"]["off_mask_by_distance"]["0.5 to 2 degrees"] == 0 and g["east"]["off_mask_by_distance"]["no mask in the band-month"] == 1
    committed = {"layer": {"1990-06": {"b": {"1990-06-01": [{"lon": 50.0, "depth": -6.0, "prominence": 3.0}, {"lon": 36.0, "depth": -6.0, "prominence": 5.0}]}}}}
    assert M.reproduce_layer(cands, committed) == {"1990-06": {"equal": True, "candidates": 2}}
    committed["layer"]["1990-06"]["b"]["1990-06-01"][0]["lon"] = 50.5
    assert M.reproduce_layer(cands, committed)["1990-06"]["equal"] is False, "a moved longitude with the same depth and prominence is not equal"
    committed["layer"]["1990-06"]["b"]["1990-06-01"][0]["lon"] = 50.0
    committed["layer"]["1990-06"]["b"]["1990-06-01"][0]["prominence"] = 3.5
    assert M.reproduce_layer(cands, committed)["1990-06"]["equal"] is False


def test_main_refuses_to_overwrite(tmp_path):
    M = _load()
    (tmp_path / "eval60_flag_tables.md").write_text("x")
    try:
        M.main(["--records", str(tmp_path), "--layer-dir", str(tmp_path), "--ledger", "x", "--candidates", "x", "--out-dir", str(tmp_path)]); assert False
    except SystemExit as ex:
        assert "never overwritten" in str(ex)


def test_the_graded_values_and_the_distance_to_the_mask():
    """Added after the declaration: the threshold-free fraction behind each mask and the
    distance from an off-mask position to the nearest mask longitude."""
    M = _load()
    fr = {(y, 6, "b"): (np.array([10, 0, 0]), np.array([30, 30, 0])) for y in M.YEARS}
    fr[(M.YEARS[0], 6, "b")] = (np.array([30, 0, 0]), np.array([30, 30, 0]))
    v = M.band_fraction(fr, M.YEARS[0], 6, "b")
    assert abs(v[0] - 1.0 / 3.0) < 1e-12 and v[1] == 0.0 and np.isnan(v[2]), "the own season (30 of 30) is left out, and a longitude with no available band-day is NaN"
    assert abs(M.band_fraction(fr, M.YEARS[0], 6, "b", leave_out=False)[0] - (23 / 3 + 1) / 24) < 1e-12
    m = np.zeros(len(LONS), bool); m[(LONS >= 46.0) & (LONS <= 50.0)] = True
    assert M.mask_distance(m, LONS, 48.0) == 0.0 and M.mask_distance(m, LONS, 51.5) == 1.5 and M.mask_distance(m, LONS, 40.0) == 6.0 and M.mask_distance(np.zeros(len(LONS), bool), LONS, 40.0) is None
    records = [_record(y, 6, [np.zeros(len(LONS))]) for y in M.YEARS]
    for k, r in enumerate(records):
        r["persistence"]["fraction"] = np.full((2, len(LONS)), 0.3 if k else 0.9).tolist()
    r0 = records[0]; r0["persistence"]["fraction"][0][0] = None
    masks = M.Masks(records)
    cv = masks.cell_value(M.YEARS[0], 6)
    assert abs(cv[1, 1] - 0.3) < 1e-12 and abs(masks.cell_value(M.YEARS[1], 6)[1, 1] - (0.9 + 0.3 * 22) / 23) < 1e-12, "the own season is left out of the cell value"
    assert M.tenths_histogram([0.6, float(Fraction(3, 5)), 0.5999, 0.0, 1.0, 0.95]) == [1, 0, 0, 0, 0, 1, 2, 0, 0, 2], "exactly 0.6 lands in 0.6 to 0.7, and 1 in the last tenth"


def test_the_discrete_transient_stays_separate_on_every_day_but_the_overlap():
    """Review of 2026-10-18, F2: a standing single-cell minimum at 50 E in every season and,
    in 1990 only, a single-cell transient moving west from 60 E at 2 degrees per day, so the
    two stay separate on every day but the one at 50 E. The complete candidate list and its
    flags are asserted for every day."""
    M = _load()
    def cell(c):
        v = np.zeros(len(LONS)); v[LONS == c] = -10.0; return v
    records = []
    for y in M.YEARS:
        rows = []
        for d in range(30):
            v = cell(50.0)
            if y == 1990 and d <= 10:
                v = v + cell(60.0 - 2.0 * d)
            rows.append(v)
        records.append(_record(y, 6, rows))
    masks = M.Masks(records)
    assert np.flatnonzero(masks.band(1990, 6, "b", "primary")).tolist() == [int(np.flatnonzero(LONS == 50.0)[0])], "the standing mask is the single cell at 50 E"
    by_day = {}
    for c in M.flag_candidates([r for r in records if r["year"] == 1990], masks):
        by_day.setdefault(c["date"], {})[c["lon"]] = c["flags"]
    for d in range(30):
        day = f"1990-06-{d + 1:02d}"; x = 60.0 - 2.0 * d
        if d <= 10 and d != 5:
            want = {50.0: True, x: False}
        else:
            want = {50.0: True}
        assert {lon: fl["primary"] for lon, fl in by_day[day].items()} == want, (day, by_day[day])
        if d <= 10 and d != 5:
            assert by_day[day][x] == {f: False for f in M.FORMS}, "the transient's own season does not put its path on any form's mask"


def test_the_forms_differ_through_the_public_path_and_the_distance_uses_the_primary_mask():
    """Review of 2026-10-18, F3: northerly band-days on 13, 17 and 20 of 30 days at 30, 40
    and 50 E in every season, so the forms must disagree, and a stub with different primary
    and all-season masks, so the distance must come from the primary one."""
    M = _load()
    assert M.FORMS == {"primary": (Fraction(1, 2), True), "t0.4": (Fraction(2, 5), True), "t0.6": (Fraction(3, 5), True), "all_seasons": (Fraction(1, 2), False)}
    def row(d):
        v = np.zeros(len(LONS))
        for lon, days in ((30.0, 13), (40.0, 17), (50.0, 20)):
            if d < days:
                v[LONS == lon] = -10.0
        return v
    records = [_record(y, 6, [row(d) for d in range(30)]) for y in M.YEARS]
    masks = M.Masks(records)
    first = {c["lon"]: c for c in M.flag_candidates([records[0]], masks) if c["date"].endswith("-01")}
    assert first[30.0]["flags"] == {"primary": False, "t0.4": True, "t0.6": False, "all_seasons": False}, first[30.0]
    assert first[40.0]["flags"] == {"primary": True, "t0.4": True, "t0.6": False, "all_seasons": True}, first[40.0]
    assert first[50.0]["flags"] == {f: True for f in M.FORMS} and abs(first[30.0]["loo_fraction"] - 13 / 30) < 1e-12
    assert first[30.0]["mask_distance_deg"] == 10.0, "the nearest primary-mask longitude to 30 E is 40 E"
    prim = np.zeros(len(LONS), bool); prim[LONS == 40.0] = True
    alls = np.zeros(len(LONS), bool); alls[LONS == 30.0] = True
    class StubMasks:
        def band(self, y, m, b, f):
            return prim if f in ("primary", "t0.4", "t0.6") else alls
        def band_value(self, y, m, b):
            return np.zeros(len(LONS))
    rec = _record(1990, 6, [np.where(LONS == 31.0, -10.0, 0.0)])
    c = M.flag_candidates([rec], StubMasks())
    assert [(x["lon"], x["mask_distance_deg"]) for x in c] == [(31.0, 9.0)], "9 degrees to the primary mask at 40 E, not 1 degree to the all-season mask at 30 E"


def test_the_season_effect_is_reported_by_scale_and_the_ledger_coordinates_are_counted():
    M = _load()
    def fl(p, a):
        return {"primary": p, "t0.4": p, "t0.6": p, "all_seasons": a}
    bins = [{"count": 0, "flagged": fl(0, 0)} for _ in range(28)]
    bins[0] = {"count": 28, "flagged": fl(9, 10)}            # 20 to 22 E, 0.036, met first
    bins[5] = {"count": 28, "flagged": fl(9, 13)}            # 30 to 32 E, 0.143, the largest, met later
    bins[21] = {"count": 53, "flagged": fl(4, 10)}           # 62 to 64 E, 0.113, the largest at or east of 40 E
    ct = {"bin_edges_deg": list(np.arange(20.0, 76.0 + 1e-9, 2.0)),
          "pooled": {cls: {"b": {"count": 100, "flagged": fl(50, 52), "east_count": 50, "east_flagged": fl(25, 26)}} for cls in M.PROMINENCE_CLASSES},
          "by_month_band": {cls: {"6": {"b": {"count": 10, "flagged": fl(5, 5), "east_count": 0, "east_flagged": fl(0, 0)}}} for cls in M.PROMINENCE_CLASSES},
          "by_bin": {cls: {"b": {"all": bins}} for cls in M.PROMINENCE_CLASSES}}
    st = {"by_interval": {"x": {"assigned": 200, "flagged": fl(100, 103)}}, "by_latitude": {}, "by_motion_label": {}}
    lt = {"westward": {"anchored_days": 50, "flagged": fl(25, 25)}}
    se = M.season_effect(ct, st, lt)
    assert abs(se["two-degree bins"]["difference"] - 4 / 28) < 1e-12 and se["two-degree bins"]["count"] == 28 and "30 to 32 E" in se["two-degree bins"]["where"], "the largest, not the first"
    assert abs(se["two-degree bins at or east of 40 E"]["difference"] - 6 / 53) < 1e-12 and "62 to 64 E" in se["two-degree bins at or east of 40 E"]["where"]
    assert abs(se["pooled by band"]["difference"] - 0.02) < 1e-12 and abs(se["selected positions"]["difference"] - 0.015) < 1e-12 and se["ledger classes"]["difference"] == 0.0
    rows = [{"id": "E26", "class": "standing", "band": "b", "date": "1994-09-06", "anchor_lon": 50.5, "flags": {"primary": False}},
            {"id": "E27", "class": "westward", "band": "b", "date": "1994-09-06", "anchor_lon": 50.5, "flags": {"primary": False}},
            {"id": "E31", "class": "standing", "band": "c", "date": "1988-07-14", "anchor_lon": 52.5, "flags": {"primary": True}},
            {"id": "E32", "class": "standing", "band": "c", "date": "1988-07-14", "anchor_lon": 52.5, "flags": {"primary": True}},
            {"id": "E01", "class": "westward", "band": "b", "date": "1988-07-14", "anchor_lon": 52.5, "flags": {"primary": True}}]
    lc = M.ledger_coordinates(rows)
    assert lc["entry_days"] == 5 and lc["distinct_coordinates"] == 3 and len(lc["shared"]) == 2 and lc["shared_across_classes"] == 1, "the same date and longitude in another band is a different coordinate"


def test_every_bin_carries_every_form():
    """Review of 2026-10-18, F1: the declared bin tables carry all four forms."""
    M = _load()
    recs = [_record(1990, 6, [np.zeros(len(LONS))] * 30)]
    cand = {"year": 1990, "month": 6, "band": "b", "date": "1990-06-01", "lon": 60.0, "depth": -6.0, "prominence": 3.0,
            "flags": {"primary": False, "t0.4": True, "t0.6": False, "all_seasons": True}, "loo_fraction": 0.45, "mask_distance_deg": 1.0}
    ct = M.candidate_tables([cand], recs)
    edges = np.array(ct["bin_edges_deg"]); k = int(np.searchsorted(edges, 60.0, side="right") - 1)
    for cls in ("all", "p2"):
        assert ct["by_bin"][cls]["b"]["all"][k] == {"count": 1, "flagged": {"primary": 0, "t0.4": 1, "t0.6": 0, "all_seasons": 1}}
        assert ct["by_bin"][cls]["b"]["6"][k]["flagged"]["t0.4"] == 1
    assert ct["by_bin"]["p4"]["b"]["all"][k]["count"] == 0

