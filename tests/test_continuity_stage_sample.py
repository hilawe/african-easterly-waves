"""Tests for scripts/continuity_stage_sample.py (`CONTINUITY_ACROSS_AFRICA_2026-11-09.md`, section 8)."""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import continuity_stage_sample as S  # noqa: E402


def test_the_stage_classes_follow_the_declared_order():
    assert S.stage_class(100.0, 0.0, 0.0, False, True) == "tracked"
    assert S.stage_class(600.0, 500.0, 0.0, False, True) == "candidate_not_tracked", "500 km itself counts as within"
    assert S.stage_class(600.0, 501.0, 200.0, True, False) == "placed_away", "an axis near the position outranks the masks"
    assert S.stage_class(600.0, 900.0, 800.0, True, True) == "masked_by_westerlies"
    assert S.stage_class(600.0, 900.0, 800.0, False, True) == "signal_without_axis"
    assert S.stage_class(float("inf"), float("inf"), float("inf"), False, False) == "below_threshold"


def test_the_sample_takes_the_two_earliest_per_class_with_ties_by_identifier():
    census = [{"id": i, "outcome": "atlantic_side", "class": c} for i, c in ((1, "carried"), (2, "carried"), (3, "carried"), (4, "dropped"), (5, "dropped"))]
    census.append({"id": 6, "outcome": "none", "class": None})
    art = [{"id": i, "first": {"time": t}} for i, t in ((1, "1990-08-03 00:00:00"), (2, "1990-07-01 00:00:00"), (3, "1990-07-01 00:00:00"),
                                                        (4, "1990-09-01 00:00:00"), (5, "1990-06-01 00:00:00"), (6, "1990-06-01 00:00:00"))]
    assert S.select(census, art) == [("carried", 2), ("carried", 3), ("dropped", 5), ("dropped", 4)]


def _stages():
    lat_c = np.array([20.0, 18.0, 16.0, 14.0, 12.0, 10.0]); lon_c = np.arange(0.0, 21.0, 2.0)
    shape = (lat_c.size, lon_c.size)
    region = np.zeros(shape, bool); region[4, 5] = True                     # the cell at 12 N, 10 E
    st = {"westerly": np.zeros(shape, bool), "curvature_unmasked": np.full(shape, 0.5),
          "axes": [(np.array([12.0, 13.0]), np.array([11.0, 11.0]))],
          "coarse": [{"lat_mean": 20.0, "lon_mean": 10.0, "region": region}], "final": [{"lat_mean": 20.0, "lon_mean": 10.0}]}
    st["curvature_unmasked"][4, 5] = 2.0
    return st, {"lat_c": lat_c, "lon_c": lon_c}


def test_reading_a_position_finds_the_holder_region_and_classes_placement():
    st, grids = _stages()
    r = S.read_position(st, grids, 1.0, 12.2, 10.3, float("inf"))
    assert r["cell"] == [12.0, 10.0] and r["above"] and r["curvature_over_threshold"] == 2.0
    assert r["axis_km"] < 200 and r["final_km"] > 800 and r["class"] == "placed_away"
    assert r["coarse_holder_center_km"] == pytest.approx(S.nearest_km(12.2, 10.3, [20.0], [10.0]), abs=0.1)
    st["westerly"][4, 5] = True; st["axes"] = []
    r = S.read_position(st, grids, 1.0, 12.2, 10.3, float("inf"))
    assert r["class"] == "masked_by_westerlies" and r["westerly"] and not r["above"], "a masked cell is not above threshold whatever its value"
    st["westerly"][4, 5] = False; st["curvature_unmasked"][4, 5] = 0.5
    r = S.read_position(st, grids, 1.0, 12.2, 10.3, float("inf"))
    assert r["class"] == "below_threshold" and r["axis_km"] == float("inf")
    assert S.read_position(st, grids, 1.0, 12.2, 10.3, 300.0)["class"] == "tracked"


def test_only_development_seasons_are_sampled():
    for y in (1984, 2005, 2002):
        with pytest.raises(SystemExit, match="not a development season"):
            S.season_sample(y, {"seasons": []})


def test_a_staged_detection_that_does_not_reproduce_the_detector_is_refused(monkeypatch):
    import pilot_alledge_replay as A
    import pilot_event_continuity as EC
    import aew.v1port.detection as DET
    case = {"lat_c": np.zeros(2), "lon_c": np.zeros(2), "latgrid": np.zeros((2, 2)), "longrid": np.zeros((2, 2)), "time": np.array([1.0]),
            "u_c": [0], "currv_anom_c": [0], "advcurrv_anom_c": [0], "currv_anom": [0]}
    monkeypatch.setattr(A, "prepared_fields", lambda *a, **k: {})
    monkeypatch.setattr(EC, "stages_from_prepared", lambda *a, **k: {"final": [{"lat_mean": 1.0, "lon_mean": 2.0, "region": np.ones((2, 2), bool)}]})
    monkeypatch.setattr(DET, "detect_troughs", lambda *a, **k: [])
    with pytest.raises(SystemExit, match="does not reproduce detect_troughs"):
        S.staged(case, 0, 1.0, 1.0, {"latgrid_c": 0, "longrid_c": 0, "latgrid_f": 0, "longrid_f": 0}, False)


def test_records_are_never_overwritten(tmp_path):
    (tmp_path / "continuity_stage_sample.json").write_text("x")
    with pytest.raises(SystemExit, match="never overwritten"):
        S.main(["--out-dir", str(tmp_path)])
