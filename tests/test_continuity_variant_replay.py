"""Tests for scripts/continuity_variant_replay.py (`CONTINUITY_ACROSS_AFRICA_2026-11-09.md`, section 13)."""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import continuity_variant_replay as R  # noqa: E402

D0 = 7000.0
Q5_LONS = [20.0 - 2.0 * k for k in range(12)]        # three days, 20 E to 2 W


def tr(tid, lats, lons, start=D0):
    n = len(lats)
    return {"id": tid, "time": start + 0.25 * np.arange(n), "lat": np.asarray(lats, float), "lon": np.asarray(lons, float), "n_valid": n}


def test_multi_system_needs_two_systems_at_different_times():
    t = tr(0, [10.0, 10.0], [0.0, -5.0])
    q_same = R.C.index_by_time([tr(1, [10.0], [0.5]), tr(2, [10.0], [-0.5])])
    assert not R.multi_system(t, q_same), "two systems at one time only"
    q_diff = R.C.index_by_time([tr(1, [10.0], [0.5]), tr(2, [10.0], [-5.5], start=D0 + 0.25)])
    assert R.multi_system(t, q_diff)
    q_one = R.C.index_by_time([tr(1, [10.0, 10.0], [0.5, -5.5])])
    assert not R.multi_system(t, q_one), "one system at two times"


def test_stationary_is_three_days_with_ends_within_500_km():
    near = [10.0] * 13
    assert R.stationary(tr(0, near, [0.0] * 12 + [4.0])), "exactly three days, ends about 440 km apart"
    assert not R.stationary(tr(0, near[:12], [0.0] * 11 + [4.0])), "2.75 days"
    assert not R.stationary(tr(0, near, [0.0] * 12 + [5.0])), "ends about 550 km apart"


def test_side_effects_count_in_the_sector_only():
    inside = tr(0, [10.0] * 13, [0.0] * 13)
    outside = tr(1, [10.0] * 13, [-60.0] * 13)
    partner = tr(2, [10.0] * 3, [0.2] * 3)
    far_group = tr(3, [10.0] * 3, [-60.2] * 3)
    q = R.C.index_by_time([tr(9, [10.0], [0.5]), tr(8, [10.0], [0.5], start=D0 + 1.0), tr(7, [10.0], [-60.5]), tr(6, [10.0], [-60.5], start=D0 + 1.0)])
    out = R.side_effects([inside, outside, partner, far_group], q, {0: 0, 2: 0, 1: 1, 3: 1})
    assert out == {"tracks_in_sector": 2, "multi_system_tracks": 1, "stationary_tracks": 1, "duplicate_groups": 1}


def _q_rows():
    return [{"id": 5, "first": {"lon": 20.0}, "outcome": "atlantic_side", "first_atlantic_side": {"index": 9}, "last_used": {"index": 11}},
            {"id": 6, "first": {"lon": 20.0}, "outcome": "none", "first_atlantic_side": None, "last_used": {"index": 1}},
            {"id": 7, "first": {"lon": 35.0}, "outcome": "atlantic_side", "first_atlantic_side": {"index": 0}, "last_used": {"index": 0}}]


def _q():
    return {5: tr(5, [12.0] * 12, Q5_LONS), 6: tr(6, [12.0] * 2, [20.0, 25.0]), 7: tr(7, [12.0], [35.0])}


def test_correspondence_applies_the_census():
    ours = [tr(0, [12.0] * 12, Q5_LONS)]
    rows = [{"id": 0, "first": {"lon": 20.0}, "outcome": "atlantic_side"}, {"id": 1, "first": {"lon": 9.0}, "outcome": "none"}]
    c = R.correspondence(ours, rows, _q(), _q_rows())
    assert c["qtrack_classes"]["carried"] == 1 and sum(c["qtrack_classes"].values()) == 1, "the 35 E start is outside the band"
    assert (c["accompanied_steps"], c["qtrack_steps"]) == (11, 12), "ten of track 5, to its first Atlantic-side step, and one of track 6"
    assert (c["aewc_cohort"], c["aewc_atlantic_side"]) == (1, 1)
    late = [tr(0, [12.0] * 3, Q5_LONS[9:], start=D0 + 2.25)]
    c = R.correspondence(late, rows, _q(), _q_rows())
    assert c["qtrack_classes"]["picked_up_late"] == 1 and c["accompanied_steps"] == 1, "first met 54 hours after the start"


def _season(name_values):
    base = {"qtrack_classes": {k: 0 for k in R.C.Q_CLASSES}, "accompanied_steps": 0, "qtrack_steps": 100, "aewc_cohort": 0, "aewc_atlantic_side": 0,
            "tracks_in_sector": 50, "multi_system_tracks": 8, "stationary_tracks": 4, "duplicate_groups": 4}
    out = {}
    for n, (acc, carried, fj) in name_values.items():
        r = json.loads(json.dumps(base)); r["accompanied_steps"] = acc; r["qtrack_classes"]["carried"] = carried; r.update(fj)
        out[n] = r
    return {"configs": out}


def test_the_provisional_rule():
    s = _season({"released": (20, 1, {}), "VIN": (25, 2, {"multi_system_tracks": 10}), "VCLIP": (24, 2, {})})
    v = R.decide([s])["verdicts"]
    assert v["VIN"]["carried_to_freezing"], "a 5-point rise and a 25 % false-join rise pass"
    assert not v["VCLIP"]["carried_to_freezing"], "a 4-point rise fails"
    s = _season({"released": (20, 1, {}), "VIN": (30, 2, {"stationary_tracks": 6}), "VCLIP": (30, 1, {})})
    v = R.decide([s])["verdicts"]
    assert not v["VIN"]["carried_to_freezing"], "a 50 % false-join rise fails"
    assert not v["VCLIP"]["carried_to_freezing"], "no more carried tracks fails"
    s = _season({"released": (20, 1, {"duplicate_groups": 0}), "VIN": (30, 2, {"duplicate_groups": 1}), "VCLIP": (30, 2, {"duplicate_groups": 0})})
    v = R.decide([s])["verdicts"]
    assert not v["VIN"]["carried_to_freezing"] and v["VCLIP"]["carried_to_freezing"], "any rise from zero fails"
    two = [_season({"released": (20, 1, {}), "VIN": (30, 1, {}), "VCLIP": (20, 1, {})}),
           _season({"released": (20, 1, {}), "VIN": (20, 2, {}), "VCLIP": (20, 1, {})})]
    v = R.decide(two)["verdicts"]
    assert v["VIN"]["carried_to_freezing"] and v["VIN"]["share_rise"] == pytest.approx(0.05), "the seasons are pooled"


def _fixture(tmp_path, monkeypatch, gate=True, census_carried=1, census_sha=None, stored_outcome="atlantic_side"):
    import coast_crossing_measurement as CC
    import pilot_margin_replay as MR
    import pilot_stage_diagnostic as SD
    import pilot_track_crosswalk as P
    import qtrack_pairing_pilot as M
    import season_metrics as SM
    f1, f2 = tmp_path / "ours.mat", tmp_path / "q.nc"
    f1.write_text("ours"); f2.write_text("q")
    art = {"inputs": {"this_record": {"path": str(f1), "sha256": R.C.sha256_of(f1)}, "qtrack": {"path": str(f2), "sha256": R.C.sha256_of(f2)},
                      "region_polygons": {"dir": "x"}},
           "sides": {"qtrack": {"tracks": _q_rows()}, "this_record": {"tracks": [{"id": 0, "outcome": stored_outcome}]}}}
    ad = tmp_path / "art"; ad.mkdir()
    (ad / R.C.ARTIFACT.format(year=1990)).write_text(json.dumps(art))
    ev = tmp_path / "ev" / "era5_1990"; ev.mkdir(parents=True)
    (ev / "tracking_era5_1990.json").write_text(json.dumps({"dataset_specific": {"tracks_sha256": R.C.sha256_of(f1), "coarse_threshold": 1.0, "fine_threshold": 1.0,
                                                                                  "tracker_flags": {"absorb": False, "exclusive": False}}}))
    census = {"generated_by": {"sha256": census_sha or R.C.SCRIPT_SHA256_AT_LOAD},
              "seasons": [{"year": 1990, "qtrack": [{"class": "carried" if census_carried else "dropped", "accompanied_steps": 11, "steps": 12}],
                           "this_record": [{"outcome": "atlantic_side"}]}]}
    cp = tmp_path / "census.json"; cp.write_text(json.dumps(census))
    monkeypatch.setattr(P, "load_run", lambda d, year=None: {"tracks_sha256": R.C.sha256_of(f1)})
    monkeypatch.setattr(SD, "load_case", lambda p, r: {"sha256": "c"})
    monkeypatch.setattr(MR, "control_grids", lambda case: {})
    monkeypatch.setattr(MR, "archive_gate", lambda finished, run: {"passed": gate})
    monkeypatch.setattr(SM, "load_regions", lambda d: ([], {}))
    monkeypatch.setattr(M, "load_qtrack", lambda p: (list(_q().values()), {}))
    monkeypatch.setattr(CC, "groups_ours", lambda tracks: {})
    monkeypatch.setattr(CC, "measure", lambda tracks, *a: {"tracks": [{"id": t["id"], "first": {"lon": float(t["lon"][0])}, "outcome": "atlantic_side"} for t in tracks]})
    seen = []

    def loop(case, grids, ct, ft, flags, options):
        seen.append(dict(options))
        return [{"time": D0 + 0.25 * np.arange(12), "lat": np.full(12, 12.0), "lon": np.array(Q5_LONS)}]
    kw = dict(campaign=str(tmp_path), evidence=str(tmp_path / "ev"), artifact_dir=str(ad), census_path=str(cp), loop=loop)
    return kw, seen


def test_a_season_runs_every_configuration_once_bound_to_production(tmp_path, monkeypatch):
    kw, seen = _fixture(tmp_path, monkeypatch)
    out = R.run_season(1990, **kw)
    assert seen == [{}, {"coarse_position_from_input": True}, {"coarse_clip_radius_deg": 5.0}]
    assert set(out["configs"]) == {"released", "VIN", "VCLIP"} and out["configs"]["VIN"]["qtrack_classes"]["carried"] == 1


def test_a_released_run_that_is_not_production_is_refused(tmp_path, monkeypatch):
    kw, _ = _fixture(tmp_path, monkeypatch, gate=False)
    with pytest.raises(SystemExit, match="does not reproduce the stored campaign tracks"):
        R.run_season(1990, **kw)


def test_a_released_run_that_differs_from_the_census_is_refused(tmp_path, monkeypatch):
    kw, _ = _fixture(tmp_path, monkeypatch, census_carried=0)
    with pytest.raises(SystemExit, match="differs from the census"):
        R.run_season(1990, **kw)


def test_a_census_from_another_script_or_differing_coast_rows_are_refused(tmp_path, monkeypatch):
    kw, _ = _fixture(tmp_path, monkeypatch, census_sha="0" * 64)
    with pytest.raises(SystemExit, match="not written by the current census script"):
        R.run_season(1990, **kw)
    sub = tmp_path / "second"; sub.mkdir()
    kw, _ = _fixture(sub, monkeypatch, stored_outcome="none")
    with pytest.raises(SystemExit, match="coast-crossing rows differ"):
        R.run_season(1990, **kw)


def test_configurations_scored_over_different_qtrack_observations_are_refused():
    s = _season({"released": (20, 1, {}), "VIN": (30, 2, {}), "VCLIP": (30, 2, {})})
    s["configs"]["VIN"]["qtrack_steps"] = 101
    with pytest.raises(SystemExit, match="different QTrack observations"):
        R.decide([s])


def test_only_development_seasons_are_read():
    with pytest.raises(SystemExit, match="not a development season"):
        R.run_season(1992)


def test_records_are_never_overwritten(tmp_path):
    (tmp_path / "continuity_variant_replay_1990.json").write_text("x")
    with pytest.raises(SystemExit, match="never overwritten"):
        R.main(["season", "--year", "1990", "--out-dir", str(tmp_path)])
    (tmp_path / "continuity_variant_replay.md").write_text("x")
    with pytest.raises(SystemExit, match="never overwritten"):
        R.main(["combine", "--out-dir", str(tmp_path)])
