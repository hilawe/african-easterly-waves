"""The history overlay: the cohort rule (first stored time in the window, first stored
longitude strictly east of 40 E), the visible portions breaking at omissions, step gaps
and band exits with the start marker only at a displayed first observation, the two
attributes, and the record's counts and bindings."""
import datetime as dt
import gzip
import importlib.util
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("history_overlay_under_test", os.path.join(ROOT, "scripts", "pilot_season_history_overlay.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _days(y, m, d, h=0):
    return (dt.datetime(y, m, d, h) - dt.datetime(1900, 1, 1)).total_seconds() / 86400.0


def _hist(y, m, d, lats, lons, steps=None, h=0):
    n = len(lats)
    t0 = _days(y, m, d, h)
    steps = steps or list(range(n))
    return {"birth": f"{steps[0]}:0", "lat": lats, "lon": lons, "raw_lat": lats, "raw_lon": lons, "steps": steps,
            "time": [t0 + (s - steps[0]) * 0.25 for s in steps], "n_points": n}


def test_the_cohort_rule_is_first_time_in_window_and_first_longitude_strictly_east_of_40():
    O = _load()
    fin = [_hist(2001, 6, 1, [10.0, 10.0], [41.0, 40.0]),          # selected
           _hist(2001, 7, 1, [10.0, 10.0], [40.0, 39.0]),          # first longitude exactly 40: not selected
           _hist(2001, 5, 31, [10.0, 10.0], [50.0, 49.0], h=18),   # May: not selected
           _hist(2001, 10, 1, [10.0, 10.0], [50.0, 49.0]),         # October: not selected
           _hist(2001, 9, 30, [20.0, 21.0], [50.0, 39.9], h=18),   # selected, first latitude 20.0 inclusive, ends west of 40
           _hist(2001, 8, 1, [20.5, 20.0], [50.0, 40.0])]          # selected, first latitude 20.5 is outside 5 to 20, ends at 40.0 which is not west
    sel = O.select_cohort(fin, 2001)
    assert [i for i, _ in sel] == [0, 4, 5]
    t_first, t_last = dt.datetime(2001, 6, 1), dt.datetime(2001, 9, 30, 18)
    r4 = O.history_record(4, fin[4], t_first, t_last); r5 = O.history_record(5, fin[5], t_first, t_last)
    assert r4["starts_within_5_to_20_N"] and r4["ends_west_of_40_E"]
    assert not r5["starts_within_5_to_20_N"] and not r5["ends_west_of_40_E"]


def test_portions_break_at_omissions_step_gaps_and_band_exits_and_the_start_marker_needs_a_displayed_first_observation():
    O = _load()
    lats = [10.0, 10.0, 14.7, 16.0, 16.0, 16.0, 30.0, 16.0, 16.0]
    lons = [50.0, 49.0, 48.0, 47.0, 46.0, 45.0, 44.0, 43.0, 80.0]
    steps = [600, 601, 602, 603, 605, 606, 607, 608, 609]            # a gap between 603 and 605
    h = _hist(2001, 6, 1, lats, lons, steps)
    h["time"] = [_days(2001, 6, 1) + (s - 600) * 0.25 for s in steps]
    r = O.history_record(0, h, dt.datetime(2001, 6, 1), dt.datetime(2001, 9, 30, 18))
    south, north = r["portions"]["southern"], r["portions"]["northern"]
    assert [p["obs"] for p in south] == [[0, 1]], "the southern portion ends where the latitude enters the gap between the bands"
    assert [p["obs"] for p in north] == [[3], [4, 5], [7]], "northern portions break at the step gap, at the 30 N omission, and the 80 E omission closes the last"
    assert r["omitted"] == {"outside_window": 0, "outside_longitudes": 1, "between_bands": 1, "outside_bands": 1}
    assert r["displayed"] == {"southern": 2, "northern": 4} and r["band_sequence"] == ["southern", "northern"] and r["band_changes"] == 1
    assert r["start_displayed"]
    h2 = _hist(2001, 6, 1, [30.0, 16.0, 16.0], [50.0, 49.0, 48.0])   # first observation outside both bands: no start marker
    r2 = O.history_record(1, h2, dt.datetime(2001, 6, 1), dt.datetime(2001, 9, 30, 18))
    assert not r2["start_displayed"] and [p["obs"] for p in r2["portions"]["northern"]] == [[1, 2]]
    h3 = _hist(2001, 9, 30, [10.0, 10.0, 10.0], [50.0, 49.0, 48.0], h=12)   # the last observation falls after the window
    r3 = O.history_record(2, h3, dt.datetime(2001, 6, 1), dt.datetime(2001, 9, 30, 18))
    assert r3["omitted"]["outside_window"] == 1 and r3["displayed"]["southern"] == 2


def test_main_binds_both_sources_counts_the_cohort_and_draws_the_declared_figures(tmp_path):
    pytest.importorskip("matplotlib")
    O = _load()
    year = 2001
    times = [dt.datetime(year, 6, 1) + dt.timedelta(hours=6 * k) for k in range(488)]
    lons = np.arange(20.0, 75.5, 0.5)
    rng = np.random.default_rng(2)
    overview = {"year": year, "lons": lons.tolist(), "times": [t.strftime("%Y-%m-%dT%H") for t in times],
                "bands": {b: {"series": rng.normal(0, 3, (488, lons.size)).tolist()} for b in ("southern", "northern")}}
    op = tmp_path / "overview.json"; op.write_text(json.dumps(overview))
    k0 = int((dt.datetime(year, 7, 10) - dt.datetime(year, 1, 1)).total_seconds() // 21600)
    fin = [_hist(year, 7, 10, [10.0, 10.5, 15.5], [50.0, 48.0, 46.0], [k0, k0 + 1, k0 + 2]),     # displayed on both panels
           _hist(year, 7, 10, [30.0, 31.0], [50.0, 48.0], [k0, k0 + 1]),                          # selected, nothing displayed
           _hist(year, 3, 1, [10.0, 10.0], [50.0, 48.0], [0, 1])]                                 # not in the window
    rp = tmp_path / "replay.json.gz"
    with gzip.open(rp, "wb") as fh:
        fh.write(json.dumps({"finished": fin, "thresholds": {"x": 1}, "configuration": {"y": 2}}).encode())
    out = tmp_path / "o.json"
    assert O.main(["--overview", str(op), "--replay", str(rp), "--year", str(year), "--out", str(out), "--png-prefix", str(tmp_path / "ov")]) == 0
    rec = json.loads(out.read_text())
    c = rec["counts"]
    assert c["cohort"] == 2 and c["finished_in_replay"] == 3 and c["with_any_displayed_observation"] == 1 and c["with_none_displayed"] == 1
    assert c["histories_displayed_per_band"] == {"southern": 1, "northern": 1} and c["histories_on_both_panels"] == 1
    assert c["observations"]["omitted_by_reason"]["outside_bands"] == 2 and c["starts_within_5_to_20_N"] == 1
    assert rec["sources"]["replay"]["thresholds"] == {"x": 1} and len(rec["sources"]["replay"]["sha256"]) == 64 and len(rec["sources"]["overview"]["sha256"]) == 64
    assert [f["name"] for f in rec["figures"]] == ["season", "June", "July", "August", "September"] and all(os.path.exists(f["path"]) for f in rec["figures"])
    assert rec["figures"][2]["visible_portions"] == {"southern": 1, "northern": 1} and rec["figures"][1]["visible_portions"] == {"southern": 0, "northern": 0}
    with pytest.raises(SystemExit, match="not the declared year"):
        O.main(["--overview", str(op), "--replay", str(rp), "--year", "2002", "--out", str(tmp_path / "o2.json"), "--png-prefix", str(tmp_path / "ov2")])
