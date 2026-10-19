"""The anchored ledger must follow section 2 of `EASTERN_T1_LEDGER_2026-10-18.md`: intervals
interpolated between the stated ends, a day anchored only at or below the floor, the edge
flag when the anchor sits on the interval's end cell, speeds over anchored days only, the
companion check east of the anchor, and censoring read from the numbers."""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("t1_ledger_under_test", os.path.join(ROOT, "scripts", "eastern_t1_ledger.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _record(fn, ndays=8, year=2000, month=7):
    """A month record with four six-hourly samples per day of a band mean fn(lon, day)."""
    lons = np.arange(20.0, 75.01, 0.5)
    times, rows = [], []
    for d in range(ndays):
        for h in (0, 6, 12, 18):
            times.append(f"{year}-{month:02d}-{d + 1:02d}T{h:02d}"); rows.append(fn(lons, d + h / 24.0))
    return {"label": "test", "lons": lons.tolist(), "times": times, "bands": {"15 to 19.5 N": [15.0, 19.5]}, "band_means_700": {"15 to 19.5 N": np.array(rows).tolist()}, "generated_by": {"sha256": "0" * 64}}


def _entry(first, c0, last, c1, **flags):
    f = {"companion": False, "coexisting": False, "wobble_alternative": False, "censored_start": False, "censored_end": False, "censored_east": False}
    f.update(flags)
    return {"id": "T", "year": 2000, "month": 7, "band": "15 to 19.5 N", "first": {"date": first, "center": c0}, "last": {"date": last, "center": c1}, "flags": f, "note": ""}


def test_intervals_interpolate_between_the_stated_ends():
    M = _load()
    import datetime as dt
    e = _entry("2000-07-02", 60.0, "2000-07-06", 50.0)
    assert M.interval_for(e, dt.date(2000, 7, 2)) == (55.0, 65.0) and M.interval_for(e, dt.date(2000, 7, 6)) == (45.0, 55.0)
    assert M.interval_for(e, dt.date(2000, 7, 4)) == (50.0, 60.0), "the midpoint in time is the midpoint in longitude"
    try:
        M.interval_for(e, dt.date(2000, 7, 7)); assert False
    except SystemExit as ex:
        assert "outside" in str(ex)


def test_anchors_follow_the_daily_minimum_with_the_floor_and_the_edge_flag():
    M = _load()
    rec = _record(lambda lons, t: -12.0 * np.exp(-0.5 * ((lons - (60.0 - 2.0 * t)) / 2.0) ** 2) + 5.0 * np.exp(-0.5 * ((lons - 70.0) / 2.0) ** 2))
    dm = M.daily_means(rec, "15 to 19.5 N"); lons = np.array(rec["lons"])
    e = _entry("2000-07-01", 60.0, "2000-07-08", 46.0)
    days = M.anchor_entry(e, dm, lons)
    assert len(days) == 8 and all(r["anchored"] for r in days)
    anchors = [r["anchor_lon"] for r in days]
    assert abs(anchors[0] - 59.25) <= 0.25 and abs(anchors[-1] - 45.25) <= 0.25 and not any(r["edge"] for r in days), anchors   # the daily mean of the four samples centers 0.375 day past midnight, so the first day's trough sits near 59.25 E
    sp = M.speeds(days)
    assert abs(sp["fitted"] + 2.0) < 0.05 and sp["direction"] == "westward" and sp["anchored_days"] == 8 and abs(sp["net"] + 2.0) < 0.05 and len(sp["segments"]) == 7
    assert days[0]["companion"] and days[0]["companion_max"] > 4.0 and not days[-1]["companion"], "the +5 m/s region at 70 E lies 2 to 12 degrees east of the first anchor only"
    near = _entry("2000-07-01", 52.0, "2000-07-08", 52.0)
    nd = M.anchor_entry(near, dm, lons)
    assert nd[0]["anchored"] and nd[0]["edge"] and nd[0]["anchor_lon"] == 57.0, "day one's trough near 59 E lies outside the interval 47 to 57, so the deepest cell inside is its last cell"
    assert nd[-1]["anchored"] and nd[-1]["edge"] and nd[-1]["anchor_lon"] == 47.0, "the last day's trough near 45 E lies outside, so the deepest cell inside is the first cell"
    far = _entry("2000-07-01", 30.0, "2000-07-08", 30.0)
    fd = M.anchor_entry(far, dm, lons)
    assert not any(r["anchored"] for r in fd) and all(r["v"] > M.FLOOR_MS for r in fd), "a day whose minimum inside the interval is above the floor is not anchored"
    assert M.speeds(fd)["fitted"] is None and M.speeds(fd)["net"] is None and M.speeds(fd)["anchored_days"] == 0


def test_censoring_and_summary_are_read_from_the_numbers():
    M = _load()
    rec = _record(lambda lons, t: -12.0 * np.exp(-0.5 * ((lons - 71.0) / 2.0) ** 2))
    dm = M.daily_means(rec, "15 to 19.5 N"); lons = np.array(rec["lons"])
    e = _entry("2000-07-01", 71.0, "2000-07-08", 71.0)
    days = M.anchor_entry(e, dm, lons)
    c = M.censoring(days, rec)
    assert c == {"first_day_is_month_start": True, "last_day_is_month_end": True, "first_anchor_at_east_edge": True, "last_anchor_at_west_edge": False}
    ledger = M.build_ledger({"entries": [e, _entry("2000-07-02", 71.0, "2000-07-05", 71.0)]}, {(2000, 7): rec})
    s = M.summarize(ledger)
    assert s["entries"] == 2 and s["with_fit"] == 2 and s["by_direction"]["standing"]["n"] == 2 and s["first_anchor_east_of_70"] == 2 and s["last_anchor_west_of_42"] == 0
    assert s["by_month"]["2000-07"]["anchored_days"] == 12 and s["by_month"]["2000-07"]["days"] == 12
    assert s["by_band"]["15 to 19.5 N"]["entries"] == 2 and s["by_band"]["15 to 19.5 N"]["standing"] == 2 and s["by_band"]["15 to 19.5 N"]["median_fitted_westward"] is None
    short = M.build_ledger({"entries": [_entry("2000-07-03", 71.0, "2000-07-04", 71.0)]}, {(2000, 7): rec})
    assert short[0]["speeds"]["fitted"] is None and short[0]["speeds"]["net"] is not None, "two anchored days give a net speed and no fit"


def test_the_floor_is_exactly_minus_four_and_an_incomplete_day_is_refused_or_unavailable():
    """Review of 2026-10-18, F7 and F8."""
    M = _load()
    import datetime as dt
    lons = np.arange(20.0, 75.01, 0.5)
    def rec_with(value_at_50):
        rows = []
        for h in (0, 6, 12, 18):
            v = np.zeros(len(lons)); v[np.argmin(abs(lons - 50.0))] = value_at_50; rows.append(v)
        return {"label": "t", "lons": lons.tolist(), "times": [f"2000-07-01T{h:02d}" for h in (0, 6, 12, 18)], "bands": {"b": [15.0, 19.5]}, "band_means_700": {"b": np.array(rows).tolist()}, "generated_by": {"sha256": "0" * 64}}
    e = _entry("2000-07-01", 50.0, "2000-07-01", 50.0); e["band"] = "b"
    for value, anchored in ((-4.0, True), (-3.9, False), (-4.5, True), (-5.1, True)):
        dm = M.daily_means(rec_with(value), "b")
        assert M.anchor_entry(e, dm, lons)[0]["anchored"] is anchored, (value, anchored)
    three = rec_with(-8.0); three["times"] = three["times"][:3]; three["band_means_700"]["b"] = three["band_means_700"]["b"][:3]
    try:
        M.daily_means(three, "b"); assert False, "a date with three samples is refused"
    except SystemExit as ex:
        assert "four six-hourly" in str(ex)
    dup = rec_with(-8.0); dup["times"][1] = "2000-07-01T00"
    try:
        M.daily_means(dup, "b"); assert False, "a duplicated hour is refused"
    except SystemExit as ex:
        assert "four six-hourly" in str(ex)
    partial = rec_with(-8.0); partial["band_means_700"]["b"][2][np.argmin(abs(lons - 50.0))] = None
    dm = M.daily_means(partial, "b")
    assert np.isnan(dm["2000-07-01"][np.argmin(abs(lons - 50.0))]) and dm["2000-07-01"][0] == 0.0, "a cell missing one sample is unavailable that day, the others keep their four-sample mean"


def test_speeds_use_elapsed_days_through_internal_gaps_and_the_boundary_is_strict():
    """Review of 2026-10-18, F6 and F7."""
    M = _load()
    def day(i, lon, anchored=True):
        return {"date": f"2000-07-{i + 1:02d}", "anchored": anchored, "anchor_lon": lon if anchored else None}
    days = [day(0, 60.0), day(1, None, False), day(2, None, False), day(3, 54.0), day(4, 52.0)]
    sp = M.speeds(days)
    assert sp["anchored_days"] == 3 and [s["from"] + ">" + s["to"] for s in sp["segments"]] == ["2000-07-01>2000-07-04", "2000-07-04>2000-07-05"]
    assert sp["segments"][0]["deg_per_day"] == -2.0 and sp["segments"][1]["deg_per_day"] == -2.0, "a segment across a two-day gap divides by the three elapsed days"
    assert abs(sp["fitted"] + 2.0) < 1e-9 and abs(sp["net"] + 2.0) < 1e-9 and sp["fit_rms"] < 1e-9 and sp["direction"] == "westward"
    assert M.speeds([day(0, 50.0), day(1, 49.0), day(2, 48.0)])["direction"] == "westward", "a slope of exactly -1 is not under 1"
    assert M.speeds([day(0, 50.0), day(1, 51.0), day(2, 52.0)])["direction"] == "eastward"
    assert M.speeds([day(0, 50.0), day(1, 49.5), day(2, 48.5), day(3, 47.5)])["direction"] == "standing", "a four-day fit of about minus 0.85 is under 1"
    assert M.speeds([day(0, 50.0), day(1, 49.0), day(2, 47.5), day(3, 46.5)])["direction"] == "westward", "a four-day fit of about minus 1.2 is not"

