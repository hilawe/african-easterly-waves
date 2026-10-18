"""The stage D selection rule (`EASTERN_STAGE_D_EVALUATION_2026-10-17.md`, section 2): the
season by the largest absolute northern-band filtered value within 40 to 60 E, the streak
by maximal runs of a sign's extremum that stay within 40 to 60 E, keep at least 1 m/s and
move at most 2.5 degrees per step, qualifying at 48 elapsed hours or more with a westward
least-squares slope of at least 1 degree per day, the largest amplitude among them, and
the window one day either side capped at eight days."""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("stage_d_select_under_test", os.path.join(ROOT, "scripts", "pilot_stage_d_select.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _record(series, lons, start="2000-07-01T00"):
    import datetime as dt
    t0 = dt.datetime.strptime(start, "%Y-%m-%dT%H")
    times = [(t0 + dt.timedelta(hours=6 * k)).strftime("%Y-%m-%dT%H") for k in range(len(series))]
    return {"year": 2000, "lons": list(lons), "times": times, "bands": {"northern": {"series": series}}}


def test_streaks_follow_the_declared_rule_and_the_largest_qualifying_one_is_selected():
    S = _load()
    lons = np.arange(30.0, 70.5, 0.5)
    n = 40  # 10 days
    series = []
    for k in range(n):
        row = np.zeros(lons.shape)
        # a westward-moving negative streak, 2 deg/day, amplitude 3 m/s, from day 1 to day 5 (steps 4 to 20), 60 to 52 E,
        # kept away from the standing blob so the two do not cancel (a first construction put the blob on its path)
        if 4 <= k <= 20:
            c = 60.0 - 2.0 * (k - 4) / 4.0
            row -= 3.0 * np.exp(-0.5 * ((lons - c) / 2.0) ** 2)
        # a standing positive blob at 44 E, amplitude 4 m/s, the whole time (larger but not moving)
        row += 4.0 * np.exp(-0.5 * ((lons - 44.0) / 2.0) ** 2)
        series.append([float(x) for x in row])
    rec = _record(series, lons)
    assert abs(S.season_amplitude(rec) - 4.0) < 0.2, "the season amplitude is the largest absolute value within 40 to 60 E, here the standing blob"
    streaks = S.streaks(rec)
    qual = [s for s in streaks if s["qualifies"]]
    assert len(qual) == 1 and qual[0]["sign"] == -1, [(s["sign"], s["hours"], s["slope_deg_per_day"], s["qualifies"]) for s in streaks]
    sel = S.select_streak(rec)
    assert sel["sign"] == -1 and 84 <= sel["hours"] <= 102 and sel["slope_deg_per_day"] < -1.0, sel
    assert sel["amplitude"] > 2.5
    # the positive standing streak spans the whole period but has no westward slope, so it does not qualify
    pos = [s for s in streaks if s["sign"] == 1]
    assert pos and not any(s["qualifies"] for s in pos)
    w = S.window_for(sel, rec)
    assert w["start"] <= sel["start"] and w["end"] >= sel["end"]
    assert S.hours_between(w["start"], w["end"]) <= 8 * 24


def test_a_jump_breaks_a_streak_and_a_short_one_does_not_qualify():
    S = _load()
    lons = np.arange(30.0, 70.5, 0.5)
    series = []
    n = 24
    for k in range(n):
        row = np.zeros(lons.shape)
        c = 55.0 if k < 6 else 45.0            # a 10-degree jump at step 6
        if k < 12:
            row -= 3.0 * np.exp(-0.5 * ((lons - c) / 2.0) ** 2)
        series.append([float(x) for x in row])
    rec = _record(series, lons)
    neg = [s for s in S.streaks(rec) if s["sign"] == -1]
    assert len(neg) == 2 and all(s["hours"] < 48 for s in neg) and not any(s["qualifies"] for s in neg), neg
    assert S.select_streak(rec) is None


# ------------------------------------------------------------------ stage E corrections (2026-10-17)
# The stage D record (section 6, corrected at section 15) recorded two latent defects of the
# selector: it assumed a six-hour cadence without validating it (the slope's time axis was a
# fixed 0.25 day per sample), and the eight-day cap centered on a streak could put a window
# bound between the record's time steps. Both are corrected before the stage E selection.

def test_the_cadence_is_read_from_the_record_and_a_non_uniform_one_is_refused():
    import datetime as dt
    import pytest
    S = _load()
    lons = np.arange(30.0, 70.5, 0.5)
    series = []
    for k in range(20):   # a 12-hourly record: 2 degrees per day is 1 degree per sample
        row = np.zeros(lons.shape)
        c = 60.0 - 1.0 * k
        row -= 3.0 * np.exp(-0.5 * ((lons - c) / 2.0) ** 2)
        series.append([float(x) for x in row])
    t0 = dt.datetime(2000, 7, 1)
    rec = {"year": 2000, "lons": list(lons), "times": [(t0 + dt.timedelta(hours=12 * k)).strftime("%Y-%m-%dT%H") for k in range(20)], "bands": {"northern": {"series": series}}}
    assert S.cadence_hours(rec) == 12.0
    sel = S.select_streak(rec)
    assert sel is not None and abs(sel["slope_deg_per_day"] + 2.0) < 0.15, "the slope is per real day, not per assumed quarter day"
    rec["times"][5] = "2000-07-03T09"   # one off-cadence stamp
    with pytest.raises(SystemExit):
        S.cadence_hours(rec)


def test_the_capped_window_lands_on_the_record_grid_and_covers_the_streak_center():
    S = _load()
    lons = np.arange(30.0, 70.5, 0.5)
    n = 60   # 15 days at six hours
    series = []
    for k in range(n):
        row = np.zeros(lons.shape)
        if 4 <= k <= 45:   # a 42-sample streak (41 steps, an odd span whose midpoint falls between steps), longer than the cap
            c = 59.0 - 1.5 * (k - 4) / 4.0
            row -= 3.0 * np.exp(-0.5 * ((lons - c) / 2.0) ** 2)
        series.append([float(x) for x in row])
    rec = _record(series, lons)
    sel = S.select_streak(rec)
    assert sel is not None and sel["samples"] == 42
    w = S.window_for(sel, rec)
    assert w["start"] in rec["times"] and w["end"] in rec["times"], "every window bound is a record time"
    assert w["hours"] <= 8 * 24 and w["hours"] >= 8 * 24 - 6
    mid = rec["times"][(sel["i_start"] + sel["i_end"]) // 2]
    assert w["start"] <= mid <= w["end"]
    assert w["i_start"] == rec["times"].index(w["start"]) and w["i_end"] == rec["times"].index(w["end"])
