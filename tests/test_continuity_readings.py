"""Tests for scripts/continuity_readings.py (`CONTINUITY_ACROSS_AFRICA_2026-11-09.md`, section 10)."""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import continuity_readings as R  # noqa: E402

LONS = np.arange(0.0, 21.0, 1.0)
LATS = np.arange(20.0, 0.0 - 0.5, -1.0)
LATGRID, LONGRID = np.meshgrid(LATS, LONS, indexing="ij")


def test_a_crossing_needs_the_trough_order_within_four_degrees():
    v = np.where(LONS <= 10.0, -1.0, 1.0)                      # northerly to 10 E, southerly from 11 E
    assert R.crossing_on_row(v, LONS, 12.0) == pytest.approx(1.5)
    assert R.crossing_on_row(-v, LONS, 12.0) is None, "southerly west of northerly is a ridge"
    assert R.crossing_on_row(v, LONS, 15.0) is None, "4.5 degrees away is beyond the limit"
    z = np.where(LONS <= 10.0, 0.0, 2.0)
    assert R.crossing_on_row(z, LONS, 10.5) == pytest.approx(0.5), "zero counts on the northerly side"


def test_the_crossing_must_appear_on_two_of_three_rows():
    v = np.ones(LATGRID.shape)
    i10 = int(np.flatnonzero(LATS == 10.0)[0])
    v[i10, :11] = -1.0
    assert not R.trough_crossing(v, LATGRID, LONGRID, 10.0, 11.0)["present"]
    v[i10 + 1, :11] = -1.0
    tc = R.trough_crossing(v, LATGRID, LONGRID, 10.0, 11.0)
    assert tc["present"] and tc["rows_with_crossing"] == 2 and tc["rows_read"] == 3


def test_a_local_maximum_is_strict_at_or_above_the_threshold_and_within_300_km():
    c = np.zeros(LATGRID.shape)
    i, j = int(np.flatnonzero(LATS == 10.0)[0]), int(np.flatnonzero(LONS == 11.0)[0])
    c[i, j] = 2.0
    m = R.local_maximum(c, LATGRID, LONGRID, 1.0, 10.0, 12.0)
    assert m["lat"] == 10.0 and m["lon"] == 11.0 and m["km"] == pytest.approx(R.C.km(10.0, 12.0, 10.0, 11.0), abs=0.1)
    assert R.local_maximum(c, LATGRID, LONGRID, 2.5, 10.0, 12.0) is None, "below the threshold"
    assert R.local_maximum(c, LATGRID, LONGRID, 1.0, 10.0, 15.0) is None, "about 440 km away"
    c[i, j + 1] = 2.0
    assert R.local_maximum(c, LATGRID, LONGRID, 1.0, 10.0, 12.0) is None, "a tie with a neighbor is not a strict maximum"


def test_support_counts_the_two_properties():
    assert R.support(True, {"km": 1}) == "supported" and R.support(False, {"km": 1}) == "partial"
    assert R.support(True, None) == "partial" and R.support(False, None) == "unsupported"


def _entry():
    return {"step": 7, "candidates": [{"index": 0, "lat_mean": 10.0, "lon_mean": 12.0}, {"index": 1, "lat_mean": 10.0, "lon_mean": 30.0}],
            "claims": [{"birth": "3:0", "candidate": 0}], "seeds": [{"birth": "7:1", "candidate": 1}]}


def test_the_association_attribution_names_the_holder_and_its_fate():
    e = _entry()
    res = {"fates": {"3:0": "pruned_in_loop", "7:1": "finished"}, "pruned_at": {"3:0": {"n_obs": 4}}, "finished_index": {"7:1": 0}}
    a = R.attribute(e, 10.0, 11.0, res, [])
    assert a["holder"] == "3:0" and a["fate"] == "pruned_in_loop" and a["n_obs"] == 4
    res["fates"]["3:0"] = "finished"; res["finished_index"]["3:0"] = 0
    fin = [{"steps": [5, 6, 7], "lat": [10.0, 10.0, 14.0], "lon": [12.0, 12.0, 11.0]}]
    a = R.attribute(e, 10.0, 11.0, res, fin)
    assert a["fate"] == "finished" and a["stored_km"] == pytest.approx(R.C.km(10.0, 11.0, 14.0, 11.0), abs=0.1) and a["n_obs"] == 3
    e["claims"] = []
    assert R.attribute(e, 10.0, 11.0, res, fin)["fate"] == "not_held"
    assert R.attribute(e, 10.0, 21.0, res, fin)["fate"] == "no_candidate", "both candidates lie beyond 500 km"
    e["claims"] = []; e["seeds"] = [{"birth": "7:0", "candidate": 0}]; res["fates"]["7:0"] = "removed_as_single_observation"
    assert R.attribute(e, 10.0, 11.0, res, fin)["fate"] == "removed_as_single_observation"
    del res["fates"]["7:0"]
    with pytest.raises(SystemExit, match="no recorded fate"):
        R.attribute(e, 10.0, 11.0, res, fin)


def test_only_development_seasons_are_read():
    with pytest.raises(SystemExit, match="not a development season"):
        R.season_readings(2002, {"seasons": []})


def test_a_replay_that_does_not_reproduce_the_stored_tracks_is_refused(tmp_path, monkeypatch):
    import pilot_alledge_replay as A
    import pilot_margin_replay as MR
    import pilot_stage_diagnostic as SD
    import pilot_track_crosswalk as P
    ev = tmp_path / "ev" / "era5_1990"; ev.mkdir(parents=True)
    (ev / "tracking_era5_1990.json").write_text(json.dumps({"dataset_specific": {"tracks_sha256": "t", "coarse_threshold": 1.0, "fine_threshold": 1.0,
                                                                                  "tracker_flags": {"absorb": False, "exclusive": False}}}))
    case = {"sha256": "x", "lat_c": LATS, "lon_c": LONS, "latgrid": LATGRID, "longrid": LONGRID, "time": np.array([0.0] * 8)}
    monkeypatch.setattr(P, "load_run", lambda d, year=None: {"tracks_sha256": "t"})
    monkeypatch.setattr(SD, "load_case", lambda p, r: case)
    monkeypatch.setattr(A, "prepared_fields", lambda *a, **k: {"fine_meridional_wind_smoothed": np.ones(LATGRID.shape), "fine_curvature": np.zeros(LATGRID.shape)})
    monkeypatch.setattr(R.S, "staged", lambda *a, **k: {"coarse": []})

    def run_season(case_b, grids, ct, ft, flags, trace=None, **k):
        trace["result"] = {"steps": [dict(_entry(), step=0)], "fates": {"3:0": "pruned_in_loop"}, "pruned_at": {"3:0": {"n_obs": 2}}, "finished_index": {}}
        return [], {}, 0, None
    monkeypatch.setattr(A, "run_season", run_season)
    monkeypatch.setattr(MR, "archive_gate", lambda finished, run: {"passed": False})
    sample = {"seasons": [{"year": 1990, "case_sha256": "x", "tracks": [{"qtrack_id": 1, "steps": [{"step": 0, "class": "candidate_not_tracked", "lat": 10.0, "lon": 10.0}]}]}]}
    with pytest.raises(SystemExit, match="not the stored campaign tracks"):
        R.season_readings(1990, sample, campaign=str(tmp_path), evidence=str(tmp_path / "ev"))
    monkeypatch.setattr(MR, "archive_gate", lambda finished, run: {"passed": True})
    out = R.season_readings(1990, sample, campaign=str(tmp_path), evidence=str(tmp_path / "ev"))
    assert out["field"][0]["support"] == "unsupported" and out["association"][0]["holder"] == "3:0"


def test_records_are_never_overwritten(tmp_path):
    (tmp_path / "continuity_readings.md").write_text("x")
    with pytest.raises(SystemExit, match="never overwritten"):
        R.main(["--out-dir", str(tmp_path)])
