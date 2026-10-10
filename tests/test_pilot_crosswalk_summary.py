"""The crosswalk summary derives each assessment count from the report alone: item 2 as
the brief writes it with its eastern and same-end subsets, the eastern starts, the
half-open no-identical-partner bands, the shifts by pair and by control track, the
comparison pool, and with a trace the Africa-origin edge-band subset."""
import importlib.util
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("pilot_crosswalk_summary_under_test", os.path.join(ROOT, "scripts", "pilot_crosswalk_summary.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _attr(lon, t, month=8, lat=10.0, end_lon=None, obs=10, region="AFR", entry=0, censored=False):
    return {"start_lon": lon, "start_lat": lat, "start_time": t, "start_month": month, "end_lon": lon - 10 if end_lon is None else end_lon,
            "observations": obs, "first_detection_region": region, "first_entry_into_africa_index": entry, "potentially_censored_start": censored}


def _pair(a, b, cls, shift, steps=0):
    return {"a": a, "b": b, "class": cls, "start_shift_days": shift, "candidate_prefix_steps": steps, "candidate_prefix_degrees": float(steps)}


def _report():
    A = [_attr(35.0, 100.0), _attr(30.0, 200.0, month=10), _attr(-30.0, 300.0), _attr(38.0, 400.0, lat=7.0), _attr(40.0, 500.0), _attr(39.0, 600.0),
         _attr(33.0, 800.0, month=10)]                                  # October on the control side only
    B = [_attr(45.0, 99.0, region="OTH", entry=3),                     # earlier and farther east, east of 40 E, same end
         _attr(50.0, 198.0, month=10, region="OTH", entry=None),       # earlier and farther east, east of 40 E, endpoint changed
         _attr(-20.0, 299.0),                                           # earlier and farther east, in the Atlantic
         _attr(38.0, 400.0, lat=7.0),                                   # same start: the comparison pool
         _attr(40.0, 500.0),
         _attr(55.0, 700.0, month=7, lat=25.0, end_lon=36.0, censored=True),   # unpaired eastern start, in season, reaches west of 40
         _attr(39.0, 601.0),                                            # starts later than its control
         _attr(48.0, 799.0, month=9)]                                   # earlier and farther east, in season on the treatment side only
    pairs = [_pair(0, 0, "otherwise changed", -1.0, 4), _pair(1, 1, "endpoint changed", -2.0, 8), _pair(2, 2, "otherwise changed", -1.0, 4),
             _pair(3, 3, "otherwise changed", 0.0), _pair(4, 4, "identical", 0.0), _pair(5, 6, "otherwise changed", 1.0), _pair(0, 4, "otherwise changed", 0.0),
             _pair(6, 7, "otherwise changed", -1.0, 2)]
    return {"year": 1990, "runs": {"a": {"tracks_sha256": "s" * 64}, "b": {"tracks_sha256": "t" * 64}}, "attributes_a": A, "attributes_b": B, "pairs": pairs,
            "unpaired_b": [5], "boundary_deg": 2.0}


def test_item_2_as_written_and_its_subsets():
    S = _load()
    i2 = S.summarize(_report())["item2"]
    assert i2["pairs"] == 4 and i2["control_tracks"] == 4 and i2["by_class"] == {"otherwise changed": 3, "endpoint changed": 1}
    assert i2["east_of_40"]["pairs"] == 3 and [r["b"] for r in i2["east_of_40"]["rows"]] == [1, 0, 7]       # longest first
    assert i2["east_of_40_same_end"]["pairs"] == 2 and i2["in_season_pairs"] == 3
    assert i2["east_of_40"]["in_season_pairs"] == 2                                          # pair (6, 7) is in season by its treatment side alone


def test_eastern_starts_bands_shifts_pool_and_flag():
    S = _load()
    s = S.summarize(_report())
    e = s["eastern_starts"]
    assert e["all_months"]["tracks"] == 4 and e["june_to_september"]["tracks"] == 3 and e["june_to_september"]["unpaired"] == 1
    assert e["all_months"]["ending_west_of_40"] == 3 and e["june_to_september"]["ending_west_of_30"] == 0   # B1 ends at exactly 40 E, not west of it
    assert e["all_months"]["later_entry_into_africa"] == 1 and e["all_months"]["first_detection_region"] == {"OTH": 2, "AFR": 2}
    assert e["by_month"]["7"] == 1 and e["all_months"]["start_lat_5_to_20_N"] == 3
    rows = s["no_identical_partner"]["rows"]
    assert rows["30 E to 40 E"] == {"changed": 5, "all": 5, "season_changed": 3, "season_all": 3}             # 30 and 33 (October), 35, 38, 39 E
    assert rows["exactly 40 E"] == {"changed": 0, "all": 1, "season_changed": 0, "season_all": 1}
    assert rows["20 E to 30 E"]["all"] == 0 and rows["west of 20 W"]["changed"] == 1
    sh = s["start_shifts"]
    assert sh["pairs"] == {"negative": 4, "zero": 3, "positive": 1} and sh["control_tracks_with_a_partner"] == 7
    assert sh["control_tracks"] == {"every_partner_same_start": 2, "some_partner_earlier": 4, "some_partner_later": 1}
    assert sh["month_transitions_of_pairs"] == {"10->9": 1}
    assert s["comparison_pool"]["pairs"] == 1 and s["comparison_pool"]["rows"][0]["a"] == 3
    assert s["potentially_censored"]["count"] == 1


def test_the_africa_origin_subset_joins_a_trace_from_the_same_run_only():
    S = _load()
    r = _report()
    trace = {"runs": {"a": {"tracks_sha256": "s" * 64}, "b": {"tracks_sha256": "t" * 64}},
             "rows": [{"a": 0, "life": 10, "partner": {"b": 0, "steps_within_close": 10}}, {"a": 3, "life": 10, "partner": {"b": 3, "steps_within_close": 5}},
                      {"a": 1, "life": 10, "partner": {"b": 1, "steps_within_close": 10}}]}                    # a1 is October, out of season
    sub = S.summarize(r, trace)["africa_origin_edge_band"]
    assert sub["controls"] == 2 and sub["covered_80_percent_within_3_deg"] == 1 and sub["covered_over_whole_life"] == [{"a": 0, "b": 0}]
    trace["runs"]["b"]["tracks_sha256"] = "u" * 64
    with pytest.raises(SystemExit, match="not made from the crosswalk's treatment run"):
        S.summarize(r, trace)
    trace["runs"]["b"]["tracks_sha256"] = "t" * 64
    trace["runs"]["a"]["tracks_sha256"] = "v" * 64                                          # a reordered control run
    with pytest.raises(SystemExit, match="not made from the crosswalk's control run"):
        S.summarize(r, trace)
    del trace["runs"]["a"]["tracks_sha256"], r["runs"]["a"]["tracks_sha256"]                 # missing on both sides never matches
    with pytest.raises(SystemExit, match="not made from the crosswalk's control run"):
        S.summarize(r, trace)
