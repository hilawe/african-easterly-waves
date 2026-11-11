"""Tests for scripts/continuity_join_test.py (`CONTINUITY_ACROSS_AFRICA_2026-11-09.md`, section 15)."""

import gzip
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import continuity_join_test as J  # noqa: E402

D0 = 7000.0


def tr(tid, lats, lons, start=D0):
    n = len(lats)
    return {"id": tid, "time": start + 0.25 * np.arange(n), "lat": np.asarray(lats, float), "lon": np.asarray(lons, float), "n_valid": n}


def kinds(track, q_tracks):
    return [e["kind"] for e in J.transitions(track, J.C.index_by_time(q_tracks), J.spans(q_tracks))]


T = tr(0, [10.0] * 4, [0.0, -2.0, -4.0, -6.0])                    # one AEWC v2 track moving west


def test_a_switch_leaves_a_system_that_still_has_a_position():
    a = tr(1, [10.0] * 4, [0.0, -2.0, 4.0, 8.0])                    # with the track, then turns away, still alive
    b = tr(2, [10.0] * 2, [-4.0, -6.0], start=D0 + 0.5)
    assert kinds(T, [a, b]) == ["switch"]


def test_a_relay_follows_a_system_that_has_ended():
    a = tr(1, [10.0] * 2, [0.0, -2.0])                               # ends at the second step
    b = tr(2, [10.0] * 2, [-4.0, -6.0], start=D0 + 0.5)
    assert kinds(T, [a, b]) == ["relay"]


def test_a_gap_in_the_missing_system_is_undetermined():
    a = tr(1, [10.0, 10.0, np.nan, 10.0], [0.0, -2.0, np.nan, 8.0])   # alive, but no position at the third step
    b = tr(2, [10.0] * 2, [-4.0, -6.0], start=D0 + 0.5)
    assert kinds(T, [a, b]) == ["undetermined"]


def test_an_addition_loses_nothing_and_a_loss_alone_is_no_transition():
    a = tr(1, [10.0] * 4, [0.0, -2.0, -4.0, -6.0])
    b = tr(2, [10.0] * 2, [-4.0, -6.0], start=D0 + 0.5)
    assert kinds(T, [a, b]) == ["addition"]
    both = tr(2, [10.0] * 2, [0.3, -2.3])                            # with the track from the start, then ends
    assert kinds(T, [a, both]) == [], "only a system lost, none new"


def test_a_switch_across_an_unaccompanied_gap():
    a = tr(1, [10.0] * 4, [0.0, 6.0, 8.0, 10.0])                     # leaves after the first step, still alive
    b = tr(2, [10.0], [-6.0], start=D0 + 0.75)
    assert kinds(T, [a, b]) == ["switch"], "the previous accompanied set is the first step's"


def test_a_system_ending_at_the_transition_with_a_position_is_a_switch():
    a = tr(1, [10.0] * 3, [0.0, -2.0, 6.0])                          # last position at the third step, far away
    b = tr(2, [10.0] * 2, [-4.0, -6.0], start=D0 + 0.5)
    assert kinds(T, [a, b]) == ["switch"]


def test_swapping_back_and_forth_is_two_switches():
    a = tr(1, [10.0] * 3, [0.0, 6.0, -4.0])
    b = tr(2, [10.0] * 3, [-8.0, -2.0, 4.0])
    assert kinds(tr(0, [10.0] * 3, [0.0, -2.0, -4.0]), [a, b]) == ["switch", "switch"]


def test_only_multi_system_tracks_in_the_sector_are_classed():
    inside = tr(0, [10.0] * 2, [0.0, -2.0]); outside = tr(1, [10.0] * 2, [-60.0, -62.0])
    q = [tr(5, [10.0], [0.3]), tr(6, [10.0], [-2.3], start=D0 + 0.25), tr(7, [10.0], [-60.3]), tr(8, [10.0], [-62.3], start=D0 + 0.25)]
    rows = J.classify([inside, outside], J.C.index_by_time(q), J.spans(q))
    assert [r["id"] for r in rows] == [0] and rows[0]["class"] == "relay"


def test_track_classes_take_the_declared_order():
    assert J.track_class([{"kind": "relay"}, {"kind": "switch"}]) == "switch"
    assert J.track_class([{"kind": "undetermined"}, {"kind": "relay"}]) == "relay"
    assert J.track_class([{"kind": "addition"}, {"kind": "undetermined"}]) == "undetermined"
    assert J.track_class([{"kind": "addition"}]) == "addition_only" and J.track_class([]) == "no_transition"


def _rec(sector, switch, undetermined=0, stationary=4, duplicates=10, acc=20, carried=1):
    classes = {k: 0 for k in J.CLASS_ORDER}; classes["switch"] = switch; classes["undetermined"] = undetermined
    counts = {"tracks": 2 * sector, "tracks_in_sector": sector, "accompanied_steps": acc, "qtrack_steps": 100,
              "qtrack_classes": {"carried": carried}, "multi_system_tracks": switch + undetermined, "stationary_tracks": stationary, "duplicate_groups": duplicates}
    return {"counts": counts, "classes": classes}


def test_the_rule_uses_rates_and_counts_undetermined_against_the_variant():
    ok = {"configs": {"released": _rec(100, 4), "VCLIP": _rec(200, 10, acc=30, carried=2)}}
    assert J.decide([ok])["carried_to_freezing"], "a switch rate of exactly 1.25 times passes"
    edge = {"configs": {"released": _rec(100, 9), "VCLIP": _rec(400, 45, acc=30, carried=2)}}
    assert J.decide([edge])["checks"]["flagged_rate"], "0.1125 against 1.25 times 0.09, equal in counts though not in floating point"
    bad = {"configs": {"released": _rec(100, 4), "VCLIP": _rec(200, 9, undetermined=2, acc=30, carried=2)}}
    d = J.decide([bad])
    assert not d["carried_to_freezing"] and not d["checks"]["flagged_rate"], "undetermined tracks count with switches"
    zero = {"configs": {"released": _rec(100, 0), "VCLIP": _rec(200, 1, acc=30, carried=2)}}
    assert not J.decide([zero])["carried_to_freezing"], "any flagged track over none fails"
    flat = {"configs": {"released": _rec(100, 4), "VCLIP": _rec(200, 4, acc=24, carried=2)}}
    assert not J.decide([flat])["checks"]["share_rise"], "a four-point rise fails"
    stat = {"configs": {"released": _rec(100, 4), "VCLIP": _rec(200, 4, stationary=11, acc=30, carried=2)}}
    assert not J.decide([stat])["checks"]["stationary_rate"]
    dup = {"configs": {"released": _rec(100, 4), "VCLIP": _rec(200, 4, duplicates=26, acc=30, carried=2)}}
    assert not J.decide([dup])["checks"]["duplicate_rate"]
    same = {"configs": {"released": _rec(100, 4), "VCLIP": _rec(200, 4, acc=30, carried=1)}}
    assert not J.decide([same])["checks"]["carried_rise"]


def test_the_seasons_are_pooled():
    s1 = {"configs": {"released": _rec(100, 4), "VCLIP": _rec(100, 8, acc=30, carried=2)}}
    s2 = {"configs": {"released": _rec(100, 4), "VCLIP": _rec(300, 2, acc=30, carried=2)}}
    d = J.decide([s1, s2])
    assert d["pooled"]["VCLIP"]["flagged_rate"] == pytest.approx(0.025) and d["carried_to_freezing"]


Q5_LONS = [20.0 - 2.0 * k for k in range(12)]


def _fixture(tmp_path, monkeypatch, gate=True, stage_tracks=1, stage_sha=None, case_sha="c"):
    import coast_crossing_measurement as CC
    import pilot_margin_replay as MR
    import pilot_stage_diagnostic as SD
    import pilot_track_crosswalk as P
    import qtrack_pairing_pilot as M
    import season_metrics as SM
    f1, f2 = tmp_path / "ours.mat", tmp_path / "q.nc"
    f1.write_text("ours"); f2.write_text("q")
    qt = tr(5, [12.0] * 12, Q5_LONS)
    rows = [{"id": 5, "first": {"lon": 20.0}, "outcome": "atlantic_side", "first_atlantic_side": {"index": 9}, "last_used": {"index": 11}}]
    art = {"inputs": {"this_record": {"path": str(f1), "sha256": J.C.sha256_of(f1)}, "qtrack": {"path": str(f2), "sha256": J.C.sha256_of(f2)},
                      "region_polygons": {"dir": "x"}},
           "sides": {"qtrack": {"tracks": rows}, "this_record": {"tracks": [{"id": 0, "outcome": "atlantic_side"}]}}}
    ad = tmp_path / "art"; ad.mkdir()
    (ad / J.C.ARTIFACT.format(year=1990)).write_text(json.dumps(art))
    ev = tmp_path / "ev" / "era5_1990"; ev.mkdir(parents=True)
    (ev / "tracking_era5_1990.json").write_text(json.dumps({"dataset_specific": {"tracks_sha256": J.C.sha256_of(f1), "coarse_threshold": 1.0, "fine_threshold": 1.0,
                                                                                  "tracker_flags": {"absorb": False, "exclusive": False}}}))
    census = {"generated_by": {"sha256": J.C.SCRIPT_SHA256_AT_LOAD},
              "seasons": [{"year": 1990, "qtrack": [{"class": "carried", "accompanied_steps": 10, "steps": 10}], "this_record": [{"outcome": "atlantic_side"}]}]}
    cp = tmp_path / "census.json"; cp.write_text(json.dumps(census))
    counts = {"tracks": stage_tracks, "qtrack_classes": {"carried": 1, "relayed": 0, "picked_up_late": 0, "dropped": 0, "unaccompanied_at_both": 0},
              "accompanied_steps": 10, "qtrack_steps": 10, "aewc_cohort": 1, "aewc_atlantic_side": 1,
              "tracks_in_sector": 1, "multi_system_tracks": 0, "stationary_tracks": 0, "duplicate_groups": 0}
    sd = tmp_path / "stage"; sd.mkdir()
    (sd / "continuity_variant_replay_1990.json").write_text(json.dumps({"generated_by": {"sha256": stage_sha or J.R.SCRIPT_SHA256_AT_LOAD}, "case_sha256": "c",
                                                                        "configs": {"released": counts, "VCLIP": counts}}))
    monkeypatch.setattr(P, "load_run", lambda d, year=None: {"tracks_sha256": J.C.sha256_of(f1)})
    monkeypatch.setattr(SD, "load_case", lambda p, r: {"sha256": case_sha})
    monkeypatch.setattr(MR, "control_grids", lambda case: {})
    monkeypatch.setattr(MR, "archive_gate", lambda finished, run: {"passed": gate})
    monkeypatch.setattr(SM, "load_regions", lambda d: ([], {}))
    monkeypatch.setattr(M, "load_qtrack", lambda p: ([qt], {}))
    monkeypatch.setattr(CC, "groups_ours", lambda tracks: {})
    monkeypatch.setattr(CC, "measure", lambda tracks, *a: {"tracks": [{"id": t["id"], "first": {"lon": float(t["lon"][0])}, "outcome": "atlantic_side"} for t in tracks]})
    seen = []

    def loop(case, grids, ct, ft, flags, options):
        seen.append(dict(options))
        return [{"time": D0 + 0.25 * np.arange(10), "lat": np.full(10, 12.0), "lon": np.array(Q5_LONS[:10])}]
    kw = dict(campaign=str(tmp_path), evidence=str(tmp_path / "ev"), artifact_dir=str(ad), census_path=str(cp), stage_dir=str(sd), loop=loop)
    return kw, seen


def test_a_season_runs_both_configurations_bound_to_the_tracking_stage(tmp_path, monkeypatch):
    kw, seen = _fixture(tmp_path, monkeypatch)
    out = J.run_season(1990, **kw)
    assert seen == [{}, {"coarse_clip_radius_deg": 5.0}]
    assert out["configs"]["VCLIP"]["classes"] == {k: 0 for k in J.CLASS_ORDER} and len(out["configs"]["released"]["tracks"]) == 1


def test_a_loop_differing_from_the_tracking_stage_or_production_is_refused(tmp_path, monkeypatch):
    kw, _ = _fixture(tmp_path, monkeypatch, stage_tracks=2)
    with pytest.raises(SystemExit, match="differs from the tracking stage's record"):
        J.run_season(1990, **kw)
    sub = tmp_path / "second"; sub.mkdir()
    kw, _ = _fixture(sub, monkeypatch, gate=False)
    with pytest.raises(SystemExit, match="does not reproduce the stored campaign tracks"):
        J.run_season(1990, **kw)


def test_a_stage_record_or_case_from_elsewhere_and_a_broken_class_sum_are_refused(tmp_path, monkeypatch):
    kw, _ = _fixture(tmp_path, monkeypatch, stage_sha="0" * 64)
    with pytest.raises(SystemExit, match="not written by the current variant-replay script"):
        J.run_season(1990, **kw)
    sub = tmp_path / "second"; sub.mkdir()
    kw, _ = _fixture(sub, monkeypatch, case_sha="d")
    with pytest.raises(SystemExit, match="not the one the tracking stage read"):
        J.run_season(1990, **kw)
    third = tmp_path / "third"; third.mkdir()
    kw, _ = _fixture(third, monkeypatch)
    monkeypatch.setattr(J, "classify", lambda *a: [{"id": 0, "class": "switch", "events": []}])
    with pytest.raises(SystemExit, match="do not sum to the multi-system count"):
        J.run_season(1990, **kw)


def test_only_development_seasons_are_read():
    with pytest.raises(SystemExit, match="not a development season"):
        J.run_season(2003)


def test_records_are_never_overwritten(tmp_path):
    with gzip.open(tmp_path / "continuity_join_test_1990.json.gz", "wt") as fh:
        fh.write("x")
    with pytest.raises(SystemExit, match="never overwritten"):
        J.main(["season", "--year", "1990", "--out-dir", str(tmp_path)])
    (tmp_path / "continuity_join_test.json").write_text("x")
    with pytest.raises(SystemExit, match="never overwritten"):
        J.main(["combine", "--out-dir", str(tmp_path)])
