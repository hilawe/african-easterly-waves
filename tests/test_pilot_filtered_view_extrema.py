"""The extrema listings read positions and values from a filtered view record: rank
separates events by the declared days and respects the longitude range, track reports
the maximum and minimum per step and marks unsupported steps."""
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("extrema_under_test", os.path.join(ROOT, "scripts", "pilot_filtered_view_extrema.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _record(tmp_path):
    times = [f"2001-06-{1 + k // 4:02d}T{6 * (k % 4):02d}" for k in range(24)]      # six days, six-hourly
    lons = [40.0, 45.0, 50.0, 55.0, 60.0]
    series = [[0.0] * 5 for _ in times]
    series[4][2] = -5.0                       # 2 June 00Z, 50 E, the largest magnitude
    series[6][3] = 4.0                        # 2 June 12Z, 55 E, inside three days of the first: suppressed by rank
    series[20][1] = 3.0                       # 6 June 00Z, 45 E, beyond three days: listed second
    series[8] = [None] * 5                    # 3 June 00Z unsupported
    r = {"year": 2001, "level_hpa": 700, "role": "primary", "lons": lons, "times": times,
         "declared": {"time": {"hours": 6}}, "bands": {"northern": {"series": series}}}
    p = tmp_path / "view.json"
    p.write_text(json.dumps(r))
    return str(p)


def test_rank_separates_events_by_the_declared_days_and_keeps_to_the_range(tmp_path):
    E = _load()
    _, out = E.rank(_record(tmp_path), (40.0, 60.0), 3.0, 3)
    events = out[0][1]
    assert [(e["time"], e["lon"], e["value"]) for e in events[:2]] == [("2001-06-02T00", 50.0, -5.0), ("2001-06-06T00", 45.0, 3.0)]
    assert all(abs(e["value"]) != 4.0 for e in events), "the 55 E event within three days of a larger one is suppressed"
    _, out = E.rank(_record(tmp_path), (52.0, 60.0), 3.0, 3)
    assert out[0][1][0]["lon"] == 55.0, "a range that excludes 50 E lists the 55 E event first"


def test_track_reports_max_and_min_per_step_and_marks_unsupported(tmp_path):
    E = _load()
    _, rows = E.track(_record(tmp_path), "northern", "2001-06-02T00", "2001-06-03T00", (40.0, 60.0), 12)
    assert rows[0]["min"] == -5.0 and rows[0]["min_lon"] == 50.0 and rows[0]["max"] == 0.0
    assert rows[1]["max"] == 4.0 and rows[1]["max_lon"] == 55.0
    assert rows[2]["unsupported"] is True
