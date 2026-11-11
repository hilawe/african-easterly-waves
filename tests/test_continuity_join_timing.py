"""Tests for scripts/continuity_join_timing.py (`CONTINUITY_ACROSS_AFRICA_2026-11-09.md`, section 16)."""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import continuity_join_timing as T  # noqa: E402

D0 = 7000.0


def tr(tid, lats, lons, start=D0):
    n = len(lats)
    return {"id": tid, "time": [start + 0.25 * k for k in range(n)], "lat": list(map(float, lats)), "lon": list(map(float, lons))}


def test_gap_and_distance_for_a_switch_after_a_day_alone():
    track = tr(0, [10.0] * 6, [0.0, -2.0, -4.0, -6.0, -8.0, -10.0])
    a = tr(1, [10.0] * 6, [0.0, 6.0, 8.0, 10.0, 12.0, 14.0])
    b = tr(2, [10.0], [-10.0], start=D0 + 1.25)
    q = [a, b]
    idx = T.C.index_by_time([{**t, "time": np.asarray(t["time"]), "lat": np.asarray(t["lat"]), "lon": np.asarray(t["lon"])} for t in q])
    row = {"id": 0, "events": [{"time": D0 + 1.25, "kind": "switch", "new": [2], "lost": [1]}, {"time": D0 + 1.25, "kind": "addition", "new": [], "lost": []}]}
    out = T.event_timing(row, track, idx, {t["id"]: t for t in q})
    assert len(out) == 1 and out[0][0] == "switch" and out[0][1] == 30, "five steps after the first accompanied observation"
    assert out[0][2] == pytest.approx(float(T.C.km(10.0, -10.0, 10.0, 14.0)), abs=0.1)


def test_a_switch_with_no_position_left_behind_is_refused():
    track = tr(0, [10.0] * 2, [0.0, -2.0])
    a = tr(1, [10.0], [0.0]); b = tr(2, [10.0], [-2.0], start=D0 + 0.25)
    q = [a, b]
    idx = T.C.index_by_time([{**t, "time": np.asarray(t["time"]), "lat": np.asarray(t["lat"]), "lon": np.asarray(t["lon"])} for t in q])
    row = {"id": 0, "events": [{"time": D0 + 0.25, "kind": "switch", "new": [2], "lost": [1]}]}
    with pytest.raises(SystemExit, match="no system left behind"):
        T.event_timing(row, track, idx, {t["id"]: t for t in q})


def test_the_summary_counts_as_described():
    s = T.summary([("switch", 54, 1700.0), ("switch", 12, 1100.0), ("switch", 30, 2000.0), ("relay", 6, None)])
    assert s["switch"] == {"n": 3, "gap_hours_min": 12, "gap_hours_median": 30.0, "at_6_hours": 0, "over_24_hours": 2, "left_behind_km_median": 1700.0}
    assert s["relay"]["at_6_hours"] == 1 and "left_behind_km_median" not in s["relay"]


def test_records_are_never_overwritten(tmp_path):
    (tmp_path / "continuity_join_timing.json").write_text("x")
    with pytest.raises(SystemExit, match="never overwritten"):
        T.main(["--out-dir", str(tmp_path)])
