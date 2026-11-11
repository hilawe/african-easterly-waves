"""Tests for scripts/continuity_merge_variants.py (`CONTINUITY_ACROSS_AFRICA_2026-11-09.md`, section 12)."""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import continuity_merge_variants as V  # noqa: E402


def w(lat, lon):
    return {"lat_mean": lat, "lon_mean": lon}


def test_the_sector_and_close_pairs():
    waves = [w(10.0, 0.0), w(10.0, 3.0), w(10.0, 10.0), w(10.0, -60.0), w(-5.0, 0.0)]
    assert len(V.in_sector(waves)) == 3
    assert V.close_pairs(V.in_sector(waves)) == 1, "only the pair 3 degrees apart lies within 500 km"
    assert V.C.km(10.0, 0.0, 10.0, 4.6) > 500.0 and V.close_pairs([w(10.0, 0.0), w(10.0, 4.6)]) == 0


def test_recovery_and_regression_are_counted_as_declared():
    rows = [{"stage_class": "placed_away", "released": 900.0, "VIN": 300.0},
            {"stage_class": "placed_away", "released": 900.0, "VIN": 700.0},
            {"stage_class": "tracked", "released": 100.0, "VIN": 800.0},
            {"stage_class": "candidate_not_tracked", "released": 500.0, "VIN": 500.0},
            {"stage_class": "candidate_not_tracked", "released": 200.0, "VIN": 600.0},
            {"stage_class": "placed_away", "released": 900.0, "VIN": 500.0}]
    t = V.tally(rows, "VIN")
    assert t["recovery"] == 2 and t["regression"] == 2, "a recovered position at exactly 500 km counts"
    assert t["regression_by_stage"] == {"tracked": 1, "candidate_not_tracked": 1, "placed_away": 0}


def test_the_variants_are_the_two_declared_options():
    assert V.VARIANTS == {"VIN": {"coarse_position_from_input": True}, "VCLIP": {"coarse_clip_radius_deg": 5.0}}


def test_only_development_seasons_are_read():
    with pytest.raises(SystemExit, match="not a development season"):
        V.season_variants(1997, {"seasons": []})


def test_records_are_never_overwritten(tmp_path):
    (tmp_path / "continuity_merge_variants.json").write_text("x")
    with pytest.raises(SystemExit, match="never overwritten"):
        V.main(["--out-dir", str(tmp_path)])
