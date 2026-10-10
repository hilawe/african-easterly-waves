"""The eastern-start screen selects finished tracks whose first observation is in June
through September, between 5 N and 20 N inclusive and strictly east of the declared
longitude, and reports how many reach the declared western mark, as a description and
never as an acceptance test."""
import importlib.util
import os
from datetime import date

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load(name):
    spec = importlib.util.spec_from_file_location(name + "_under_test", os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _days(y, m, d, hour=0.0):
    return (date(y, m, d) - date(1900, 1, 1)).days + hour / 24.0


def _track(birth, t0, lat0, lons):
    n = len(lons)
    return {"birth": birth, "time": [t0 + 0.25 * k for k in range(n)], "lat": [lat0] * n, "lon": list(lons), "steps": list(range(n)), "raw_lat": [lat0] * n, "raw_lon": list(lons), "n_points": [5] * n, "region_sha256": ["r"] * n}


def test_the_screen_selects_summer_eastern_starts_in_the_latitude_band_and_counts_those_reaching_the_mark():
    E = _load("pilot_eastern_start_screen")
    finished = [_track("0:0", _days(1990, 6, 15, 6), 10.0, [45.0, 40.0, 30.0, 15.0]),      # selected, reaches 20 E
                _track("0:1", _days(1990, 5, 31, 18), 10.0, [45.0, 35.0]),                # May, excluded
                _track("0:2", _days(1990, 7, 1), 4.0, [45.0, 35.0]),                       # south of 5 N, excluded
                _track("0:3", _days(1990, 8, 1), 20.0, [40.0, 35.0]),                      # at 40 E, not strictly east, excluded
                _track("0:4", _days(1990, 9, 30, 18), 20.0, [50.0, 30.0]),                 # selected, does not reach
                _track("0:5", _days(1990, 10, 1), 10.0, [50.0, 10.0])]                     # October, excluded
    out = E.screen(finished, start_east_of=40.0, reach=20.0, lat_range=(5.0, 20.0))
    assert out["selected"] == 2 and out["indices"] == [0, 4] and out["reaching"] == 1 and out["westernmost_lon_among_selected"] == 15.0
    assert out["rows"][1]["westernmost_lon"] == 30.0 and out["rows"][1]["reaches"] is False and out["rows"][0]["first"]["date"] == "1990-06-15 06:00:00"
    assert E.screen([], start_east_of=40.0, reach=20.0, lat_range=(5.0, 20.0))["westernmost_lon_among_selected"] is None
    assert "first_entry_into_africa" not in out["rows"][0]                                           # no region function, no attribute
    with_regions = E.screen(finished, start_east_of=40.0, reach=20.0, lat_range=(5.0, 20.0), region_of=lambda lon, lat: "AFR" if lon <= 40.0 else "OTH")
    assert with_regions["rows"][0]["first_entry_into_africa"]["index"] == 1 and with_regions["rows"][0]["first_entry_into_africa"]["lon"] == 40.0   # first detection at 45 E, first entry at 40 E
    assert with_regions["rows"][1]["first_entry_into_africa"]["index"] == 1 and with_regions["rows"][0]["first_detection"] == "0:0"
    assert E.screen([_track("1:1", _days(1990, 7, 1), 10.0, [50.0, 45.0])], start_east_of=40.0, reach=20.0, lat_range=(5.0, 20.0), region_of=lambda lon, lat: "OTH")["rows"][0]["first_entry_into_africa"] is None
    with pytest.raises(SystemExit, match="never overwritten"):
        E.main(["--replay", "x", "--out", __file__])
