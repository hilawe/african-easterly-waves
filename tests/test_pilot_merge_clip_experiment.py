"""The merge-clip experiment's contracts. Close pairs use the merge's own great-circle
distance, preexisting pairs match in either order, matching radii are inclusive, the
evaluation holds controls at 2 degrees and recovers at 5, the no-addition restriction
fires only on more candidates than the baseline, the invariance check skips exhausted
inputs, the downstream flag marks non-exhausted origins, and the merge's lineage follows
each input through the three passes without changing the output."""
import importlib.util
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load(name, folder="scripts"):
    spec = importlib.util.spec_from_file_location(name + "_under_test", os.path.join(ROOT, folder, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _f(lat, lon, n=10, sha="a"):
    return [float(lat), float(lon), int(n), sha * 64]


def test_close_pairs_use_the_great_circle_distance_and_preexisting_pairs_match_in_either_order():
    M = _load("pilot_merge_clip_experiment")
    finals = [_f(60.0, 0.0), _f(60.0, 9.0), _f(0.0, 20.0)]
    pairs = M.close_pairs(finals)
    assert [(p["i"], p["j"]) for p in pairs] == [(0, 1)] and 4.0 < pairs[0]["distance_deg"] < 5.0
    base = [_f(60.5, 9.0), _f(59.5, 0.0)]                                  # the same pair, listed the other way round
    assert M.pair_preexisting(pairs[0], finals, M.close_pairs(base), base) is True
    assert M.pair_preexisting(pairs[0], finals, [], base) is False


def test_matching_is_inclusive_and_added_and_removed_are_both_reported():
    M = _load("pilot_merge_clip_experiment")
    finals = [_f(10.0, 20.0), _f(12.0, 20.0), _f(10.0, 30.0)]
    assert [w["index"] for w in M.within(finals, 10.0, 20.0, 2.0)] == [0, 1]
    base = [_f(10.0, 20.0), _f(10.0, 32.5), _f(11.5, 20.0)]
    added, removed = M.added_and_removed(finals, base)
    assert added == [2] and removed == [1]


def _records():
    """A baseline and a variant with one step, synthetic lineage, for `evaluate`."""
    def step(finals, coarse, fine, date="2006-09-05 00Z"):
        return {"date": date, "final": finals, "coarse_lineage": coarse, "fine_lineage": fine, "axes": len(coarse), "coarse_inputs": len(coarse), "coarse_exhausted_inputs": 0}
    bfin = [_f(14.5, 27.0, sha="b"), _f(30.0, 10.0, sha="c")]
    bcoarse = [{"input": 0, "merged": 0, "exhausted": True, "selected": {"sha256": "x" * 64}, "base_sha256": "p" * 64, "output": 0},
               {"input": 1, "merged": 1, "exhausted": False, "selected": {"sha256": "y" * 64}, "base_sha256": "q" * 64, "output": 1}]
    bfine = [{"input": 0, "merged": 0, "output": 0, "exhausted": False}, {"input": 1, "merged": 1, "output": 1, "exhausted": False}]
    vfin = [_f(14.5, 24.0, sha="d"), _f(30.0, 10.5, sha="e"), _f(13.0, 22.0, sha="g")]
    vcoarse = [{"input": 0, "merged": 0, "exhausted": True, "selected": {"sha256": "z" * 64}, "base_sha256": "p" * 64, "output": 0, "clip": {"kept_cells": 9, "dropped_pieces": 0}},
               {"input": 1, "merged": 1, "exhausted": False, "selected": {"sha256": "w" * 64}, "base_sha256": "q" * 64, "output": 1},
               {"input": 2, "merged": 2, "exhausted": True, "selected": {"sha256": "v" * 64}, "base_sha256": "p" * 64, "output": 2, "clip": {"kept_cells": 4, "dropped_pieces": 1}}]
    vfine = [{"input": 0, "merged": 0, "output": 0, "exhausted": False}, {"input": 1, "merged": 1, "output": 1, "exhausted": False}, {"input": 2, "merged": 2, "output": 2, "exhausted": False}]
    base = {"steps": {"988": step(bfin, bcoarse, bfine)}}
    var = {"steps": {"988": step(vfin, vcoarse, vfine)}}
    declared = {"missing": [(988, 14.5, 26.0)], "controls": [(988, 30.0, 13.5)], "pre_arrival": [(988, 14.5, 22.0)],
                "stationary": {"S": {"steps": [988], "lat": [30.0], "lon": [10.0]}}, "windows": {"W": {"first": 988, "last": 988, "box": [5.0, 25.0, 10.0, 35.0]}}}
    return base, var, declared


def test_evaluate_recovers_at_five_degrees_holds_controls_at_two_and_separates_the_restrictions_and_flags():
    M = _load("pilot_merge_clip_experiment")
    base, var, declared = _records()
    out = M.evaluate(base, var, declared)
    assert out["A"]["missing"][0]["recovered"] is True and out["A"]["missing"][0]["nearest"]["index"] == 0
    assert out["A"]["controls"][0]["held"] is False                                 # 3 degrees away, held needs 2
    assert out["C"][0]["within_5"] == 2 and out["C"][0]["unique"] is False          # the 13 N 22 E candidate is 4.5 away too
    assert out["B"]["variant_pairs"] == 1 and len(out["B"]["new_pairs"]) == 1       # 14.5 N 24 E and 13 N 22 E are 2.5 apart
    pair = out["B"]["new_pairs"][0]
    assert pair["either_clipped"] is True and pair["share_base_component"] is True and pair["members"][0]["chain"]["axis"] == 0
    assert out["D1"]["S"]["steps_with_addition"] == 0                               # one within 2 degrees in both, equal is not an addition
    assert out["D2"][0]["added"] is True and out["D2"][0]["variant_within_2"] == 2   # 13 N 22 E at 1.5 and 14.5 N 24 E at exactly 2
    assert out["F1"]["violations"] == [{"step": 988, "input": 1}] and out["F1"]["non_exhausted_inputs_checked"] == 1
    flags = {tuple(d["final"][:2]): d["downstream_interaction"] for d in out["F3"]["differing_finals"]}
    assert flags[(14.5, 24.0)] is False and flags[(30.0, 10.5)] is True and flags[(13.0, 22.0)] is False
    vf = var["steps"]["988"]["final"]
    assert out["changes"]["988"]["added"] == [vf[0], vf[2]] and out["changes"]["988"]["removed"] == [base["steps"]["988"]["final"][0]]
    assert out["E"]["W"]["unmatched_total"] == 1 and out["D3"]["W"]["added_in_box_total"] == 2 and out["D3"]["W"]["added_persistent_total"] == 0
    sides = {(d["side"], tuple(d["final"][:2])): d["downstream_interaction"] for d in out["F3"]["differing_finals"]}
    assert sides[("baseline", (14.5, 27.0))] is False and sides[("baseline", (30.0, 10.0))] is True      # the lost non-exhausted baseline final is a downstream loss
    assert sum(1 for d in out["F3"]["differing_finals"] if d["side"] == "baseline") == 2 and sum(1 for d in out["F3"]["differing_finals"] if d["side"] == "variant") == 3
    assert out["G"]["coarse_inputs"] == 3 and out["G"]["exhausted_selections"] == 2 and out["G"]["clipped_selections"] == 2 and out["G"]["clipped_dropped_pieces_total"] == 1 and out["G"]["clipped_kept_cells"] == {"min": 4, "max": 9}


def _blob(field, latgrid, longrid, lat0, lat1, lon0, lon1, value):
    field[(latgrid >= lat0) & (latgrid <= lat1) & (longrid >= lon0) & (longrid <= lon1)] = value


def test_the_merge_lineage_follows_each_input_through_the_passes_and_records_exhaustion():
    pytest.importorskip("scipy")
    from aew.v1port.contours import _binary_masks, _select_region, merge_contours
    lat = np.arange(0.0, 41.0, 1.0); lon = np.arange(0.0, 61.0, 1.0)
    latgrid, longrid = np.meshgrid(lat, lon, indexing="ij")
    ct = 1.0e-7
    field = np.full(latgrid.shape, np.nan)
    _blob(field, latgrid, longrid, 10, 14, 5, 9, 4 * ct)                     # A, a 5 by 5 blob
    _blob(field, latgrid, longrid, 20, 20, 30, 36, 4 * ct)                   # B, one row, too short north to south
    _blob(field, latgrid, longrid, 5, 9, 50, 54, 4 * ct)                     # C
    cands = [{"time": 1.0, "lat_mean": 12.0, "lon_mean": 7.0}, {"time": 1.0, "lat_mean": 20.0, "lon_mean": 33.0}, {"time": 1.0, "lat_mean": 7.0, "lon_mean": 52.0}]
    lineage = []
    out = merge_contours(cands, latgrid, longrid, field, ct, lineage=lineage)
    assert len(out) == 2 and [e["output"] for e in lineage] == [0, None, 1]
    assert lineage[1]["kept_after_pass_two"] is True and lineage[1]["rejected_min_extent"] is True
    assert lineage[0]["seed"] == {"row": 12, "col": 7, "lat": 12.0, "lon": 7.0} and lineage[0]["seed_distance_deg"] == 0.0
    assert lineage[0]["select"]["exhausted"] is False and lineage[0]["select"]["level_landed"] == 0 and lineage[0]["select"]["selected_cells"] == 25
    plain = merge_contours(cands, latgrid, longrid, field, ct)
    assert [(w["lat_mean"], w["lon_mean"], w["lat_wave"].size) for w in plain] == [(w["lat_mean"], w["lon_mean"], w["lat_wave"].size) for w in out]
    # two inputs seeded in one blob: the second is dropped by the duplicate pass
    dup = [{"time": 1.0, "lat_mean": 12.0, "lon_mean": 7.0}, {"time": 1.0, "lat_mean": 11.0, "lon_mean": 8.0}, {"time": 1.0, "lat_mean": 7.0, "lon_mean": 52.0}]
    lineage = []
    merge_contours(dup, latgrid, longrid, field, ct, lineage=lineage)
    assert lineage[1]["kept_after_pass_two"] is False and lineage[1]["output"] is None and lineage[1].get("rejected_min_extent") is False
    # a lone wave skips the later passes and is its own output
    lineage = []
    merge_contours(dup[:1], latgrid, longrid, field, ct, lineage=lineage)
    assert lineage[0]["lone_wave"] is True and lineage[0]["output"] == 0
    # exhaustion: inner level under the trigger, last level at the trigger, last level under it
    rows = (latgrid >= 10.0) & (latgrid <= 20.0)
    f1 = np.full(latgrid.shape, np.nan); f1[rows & (longrid <= 24.0)] = 4 * ct; f1[rows & (longrid == 25.0)] = 2.2 * ct; f1[rows & (longrid == 26.0)] = 1.7 * ct; f1[rows & (longrid >= 27.0) & (longrid <= 30.0)] = 1.2 * ct
    f2 = np.full(latgrid.shape, np.nan); f2[rows & (longrid <= 25.0)] = 4 * ct
    f3 = np.full(latgrid.shape, np.nan); f3[rows & (longrid <= 20.0)] = 4 * ct; f3[rows & (longrid >= 21.0) & (longrid <= 25.0)] = 3.2 * ct
    for f, landed, exhausted in ((f1, 3, False), (f2, 5, True), (f3, 5, False)):
        t = {}
        _select_region(_binary_masks(f, ct), (15, 12), latgrid, longrid, trace=t)
        assert (t["level_landed"], t["exhausted"]) == (landed, exhausted), (landed, exhausted, t["level_landed"], t["exhausted"], [l["lon_extent"] for l in t["levels"]])


def test_recording_lineage_does_not_change_the_detection():
    pytest.importorskip("scipy")
    T = _load("test_pilot_alledge_replay", "tests")
    A = _load("pilot_alledge_replay")
    case_b, wide = T.analytic_case(35.0, -140.0, 39.0), T.analytic_case(39.0, -144.0, 43.0)
    grids, sel = A.MR.control_grids(case_b), A.selection(case_b, wide)
    fields = A.prepared_fields(wide, 3, *sel)
    t = float(np.asarray(case_b["time"], float).ravel()[3])
    plain = A.detect_from_prepared(fields, t, 4.0e-7, 2.5e-6, grids)
    lin = {}
    traced = A.detect_from_prepared(fields, t, 4.0e-7, 2.5e-6, grids, lineage=lin)
    assert A.MR.signature(plain) == A.MR.signature(traced) and plain
    assert lin["axes"] and len(lin["coarse"]) == len(lin["axes"]) and len(lin["fine"]) == sum(1 for e in lin["coarse"] if e["output"] is not None)


def test_the_clip_applies_only_at_exhaustion_with_the_declared_geometry_and_counts():
    pytest.importorskip("scipy")
    from aew.v1port.contours import _binary_masks, _select_region
    lat = np.arange(0.0, 41.0, 1.0); lon = np.arange(0.0, 61.0, 1.0)
    latgrid, longrid = np.meshgrid(lat, lon, indexing="ij")
    ct = 1.0e-7
    rows = (latgrid >= 10.0) & (latgrid <= 20.0)
    # C1: a cascade that lands short of the last level is untouched by the clip
    f1 = np.full(latgrid.shape, np.nan); f1[rows & (longrid <= 24.0)] = 4 * ct; f1[rows & (longrid == 25.0)] = 2.2 * ct; f1[rows & (longrid == 26.0)] = 1.7 * ct; f1[rows & (longrid >= 27.0) & (longrid <= 30.0)] = 1.2 * ct
    t_on, t_off = {}, {}
    on = _select_region(_binary_masks(f1, ct), (15, 12), latgrid, longrid, trace=t_on, clip_radius_deg=5.0)
    off = _select_region(_binary_masks(f1, ct), (15, 12), latgrid, longrid, trace=t_off)
    assert np.array_equal(on, off) and "clip" not in t_on and t_on["exhausted"] is False
    # C4, C5, C7, C8: an exhausted band 30 degrees wide with a gap column at 16 that the top
    # row bridges outside the disk, so the cell at (15, 17) enters the disk on its own, and two
    # cells near the seed between 1 and 3.5 times the threshold that the base level keeps
    f2 = np.full(latgrid.shape, np.nan); f2[rows & (longrid <= 30.0)] = 4 * ct
    f2[(latgrid >= 10.0) & (latgrid <= 19.0) & (longrid == 16.0)] = np.nan
    f2[(latgrid == 15.0) & ((longrid == 10.0) | (longrid == 11.0))] = 2.0 * ct
    t = {}
    kept = _select_region(_binary_masks(f2, ct), (15, 12), latgrid, longrid, trace=t, clip_radius_deg=5.0)
    assert t["exhausted"] is True and t["level_landed"] == 5
    c = t["clip"]
    assert c == {"radius_deg": 5.0, "base_cells_in_disk": 74, "last_level_cells_in_disk": 72, "kept_cells": 73, "dropped_pieces": 1, "dropped_cells": 1,
                 "kept_lat_extent": 10.0, "kept_lon_extent": 8.0}
    assert kept[15, 12] and kept[19, 15] and kept[15, 10] and not kept[15, 17] and not kept[15, 16] and int(kept.sum()) == 73
    assert t["selected_cells"] == 73 and np.array_equal(t["region"], kept) and t["base_region"].sum() == 11 * 31 - 10
    # C3: coordinate degrees, not great-circle: at 60 N a cell 3 degrees of longitude east is
    # outside a 2.5-degree disk although it is about 1.5 great-circle degrees away
    lat = np.arange(50.0, 71.0, 1.0); lon = np.arange(0.0, 41.0, 1.0)
    latgrid, longrid = np.meshgrid(lat, lon, indexing="ij")
    f3 = np.full(latgrid.shape, np.nan); f3[(latgrid >= 55.0) & (latgrid <= 65.0) & (longrid <= 30.0)] = 4 * ct
    t = {}
    kept = _select_region(_binary_masks(f3, ct), (10, 10), latgrid, longrid, trace=t, clip_radius_deg=2.5)
    assert t["exhausted"] is True and t["clip"]["kept_cells"] == 21 and kept[10, 12] and not kept[10, 13]


def test_the_clip_reaches_the_coarse_merge_only_in_both_detection_entry_points(monkeypatch):
    pytest.importorskip("scipy")
    import aew.v1port.contours as K
    import aew.v1port.detection as D
    T = _load("test_pilot_alledge_replay", "tests")
    A = _load("pilot_alledge_replay")
    case_b, wide = T.analytic_case(35.0, -140.0, 39.0), T.analytic_case(39.0, -144.0, 43.0)
    grids, sel = A.MR.control_grids(case_b), A.selection(case_b, wide)
    fields = A.prepared_fields(wide, 3, *sel)
    t = float(np.asarray(case_b["time"], float).ravel()[3])
    real = K.merge_contours
    calls = []

    def recorder(*args, **kwargs):
        calls.append(kwargs.get("clip_radius_deg"))
        return real(*args, **kwargs)
    monkeypatch.setattr(K, "merge_contours", recorder)
    monkeypatch.setattr(D, "merge_contours", recorder)
    A.detect_from_prepared(fields, t, 4.0e-7, 2.5e-6, grids, coarse_clip_radius_deg=5.0)
    assert calls == [5.0, None]
    calls.clear()
    D.detect_troughs(t, grids["latgrid_c"], grids["longrid_c"], np.asarray(case_b["u_c"][3], float), np.asarray(case_b["currv_anom_c"][3], float),
                     np.asarray(case_b["advcurrv_anom_c"][3], float), grids["latgrid_f"], grids["longrid_f"], np.asarray(case_b["currv_anom"][3], float),
                     4.0e-7, 2.5e-6, coarse_clip_radius_deg=5.0)
    assert calls == [5.0, None]
    calls.clear()
    A.detect_from_prepared(fields, t, 4.0e-7, 2.5e-6, grids)
    assert calls == [None, None]


def test_a_changed_non_exhausted_selection_refuses_the_variant_step_before_it_is_stored():
    M = _load("pilot_merge_clip_experiment")
    base = {"coarse_lineage": [{"input": 0, "merged": 0, "exhausted": False, "selected": {"sha256": "a" * 64}}, {"input": 1, "merged": 1, "exhausted": True, "selected": {"sha256": "b" * 64}}]}
    same = {"coarse_lineage": [{"input": 0, "merged": 0, "exhausted": False, "selected": {"sha256": "a" * 64}}, {"input": 1, "merged": 1, "exhausted": True, "selected": {"sha256": "z" * 64}}], "final": []}
    assert M.selection_invariance(base, same, 5) == (1, [])                                        # the exhausted input may change, the other may not
    rec = M.variant_step(base, 5, "d", {"final": [[1.0, 2.0, 3, "f" * 64]]}, lambda: dict(same))
    assert rec["step"] == 5 and rec["flag_off_final"] == [[1.0, 2.0, 3, "f" * 64]]
    changed = {"coarse_lineage": [{"input": 0, "merged": 0, "exhausted": False, "selected": {"sha256": "c" * 64}}], "final": []}
    assert M.selection_invariance(base, changed, 5) == (1, [{"step": 5, "input": 0}])
    with pytest.raises(SystemExit, match="F1"):
        M.variant_step(base, 5, "d", {"final": []}, lambda: dict(changed))


def test_band_windows_span_each_stationary_history_s_whole_lifetime_including_its_gaps():
    M = _load("pilot_merge_clip_experiment")
    w = M.band_windows({"H": {"steps": [10, 11, 13, 16]}}, [5, 25, 45, 60])
    assert w == {"H": {"first": 10, "last": 16, "box": [5.0, 25.0, 45.0, 60.0]}}


def test_the_clip_keeps_diagonally_connected_cells():
    pytest.importorskip("scipy")
    from aew.v1port.contours import _clip_to_seed
    latgrid, longrid = np.meshgrid(np.arange(3.0), np.arange(3.0), indexing="ij")
    base = np.eye(3, dtype=bool)
    kept, detail = _clip_to_seed(base, base, (0, 0), latgrid, longrid, 5.0)
    assert np.array_equal(kept, base) and detail["kept_cells"] == 3 and detail["dropped_pieces"] == 0 and detail["dropped_cells"] == 0


def test_the_c3_regression_at_a_stored_position_is_assessable_only_with_a_baseline_candidate_and_remains_only_with_a_variant_one():
    M = _load("pilot_merge_clip_c3_regression")
    fb = [_f(12.0, 40.0), _f(30.0, 10.0)]
    assert M.remained_at(12.0, 42.0, fb, [_f(12.0, 44.0)]) == {"baseline_within_2": 1, "variant_within_2": 1, "assessable": True, "remained": True}   # both exactly 2 away
    assert M.remained_at(12.0, 42.0, fb, [_f(12.0, 44.5)]) == {"baseline_within_2": 1, "variant_within_2": 0, "assessable": True, "remained": False}
    assert M.remained_at(0.0, 0.0, fb, [_f(0.0, 1.0)]) == {"baseline_within_2": 0, "variant_within_2": 1, "assessable": False, "remained": False}    # no baseline candidate: not a pass
    track = {"time": [100.0, 100.25, 100.75], "lat": [1.0, 2.0, 3.0], "lon": [10.0, 11.0, 12.0]}
    assert M.observation_at(track, 100.25) == (2.0, 11.0) and M.observation_at(track, 100.5) is None


def test_the_input_position_variant_changes_only_the_pass_one_position():
    pytest.importorskip("scipy")
    from aew.v1port.contours import merge_contours
    lat = np.arange(0.0, 41.0, 1.0); lon = np.arange(0.0, 61.0, 1.0)
    latgrid, longrid = np.meshgrid(lat, lon, indexing="ij")
    ct = 1.0e-7
    field = np.full(latgrid.shape, np.nan)
    _blob(field, latgrid, longrid, 10, 14, 5, 9, 4 * ct)                     # A, median and peak mean 12 N 7 E
    _blob(field, latgrid, longrid, 25, 35, 20, 50, 4 * ct)                   # a wide region shared by two inputs
    cands = [{"time": 1.0, "lat_mean": 11.0, "lon_mean": 6.0}, {"time": 1.0, "lat_mean": 30.0, "lon_mean": 22.0}, {"time": 1.0, "lat_mean": 30.0, "lon_mean": 47.0}]
    lo, lv = [], []
    off = merge_contours(cands, latgrid, longrid, field, ct, lineage=lo)
    on = merge_contours(cands, latgrid, longrid, field, ct, lineage=lv, position_from_input=True)
    assert [e["select"]["selected_cells"] for e in lo] == [e["select"]["selected_cells"] for e in lv]          # region selection unchanged
    assert all(np.array_equal(a["select"]["region"], b["select"]["region"]) for a, b in zip(lo, lv))
    assert (off[0]["lat_mean"], off[0]["lon_mean"]) == (12.0, 7.0)                                           # median, then halfway to the peak
    assert (on[0]["lat_mean"], on[0]["lon_mean"]) == (11.5, 6.5)                                             # the input, then halfway to the peak
    assert len(off) == 2 and lo[2]["kept_after_pass_two"] is False                                           # one region, one median, a duplicate
    assert len(on) == 3 and lv[2]["kept_after_pass_two"] is True                                             # two input positions 25 degrees apart
    assert (on[1]["lon_mean"], on[2]["lon_mean"]) == (28.5, 41.0)                                            # each refined halfway to the region's peak mean, 35 E


def test_the_input_position_reaches_the_coarse_merge_only(monkeypatch):
    pytest.importorskip("scipy")
    import aew.v1port.contours as K
    T = _load("test_pilot_alledge_replay", "tests")
    A = _load("pilot_alledge_replay")
    case_b, wide = T.analytic_case(35.0, -140.0, 39.0), T.analytic_case(39.0, -144.0, 43.0)
    grids, sel = A.MR.control_grids(case_b), A.selection(case_b, wide)
    fields = A.prepared_fields(wide, 3, *sel)
    t = float(np.asarray(case_b["time"], float).ravel()[3])
    real, calls = K.merge_contours, []

    def recorder(*args, **kwargs):
        calls.append(kwargs.get("position_from_input", False))
        return real(*args, **kwargs)
    monkeypatch.setattr(K, "merge_contours", recorder)
    A.detect_from_prepared(fields, t, 4.0e-7, 2.5e-6, grids, coarse_position_from_input=True)
    assert calls == [True, False]
    calls.clear()
    A.detect_from_prepared(fields, t, 4.0e-7, 2.5e-6, grids)
    assert calls == [False, False]


def test_input_fates_segment_correspondence_and_the_strict_selection_check():
    M = _load("pilot_merge_clip_experiment")
    def step(finals, coarse, fine):
        return {"final": finals, "coarse_lineage": coarse, "fine_lineage": fine}
    coarse_b = [{"input": 0, "merged": 0, "lat_mean": 12.0, "lon_mean": 35.0, "kept_after_pass_two": True, "output": 0, "exhausted": True, "selected": {"sha256": "a" * 64}},
                {"input": 1, "merged": 1, "lat_mean": 13.0, "lon_mean": 37.0, "kept_after_pass_two": False, "output": None, "exhausted": False, "selected": {"sha256": "b" * 64}}]
    base = step([_f(19.0, 39.0)], coarse_b, [{"input": 0, "merged": 0, "output": 0}])
    coarse_v = [dict(coarse_b[0]), dict(coarse_b[1], kept_after_pass_two=True, output=1)]
    var = step([_f(12.5, 35.5), _f(13.0, 30.0)], coarse_v, [{"input": 0, "merged": 0, "output": 0}, {"input": 1, "merged": 1, "output": None}])
    fb, fv = M.input_fate(base, 1), M.input_fate(var, 1)
    assert fb["removed_at"] == "duplicate_pass" and fb["final"] is None
    assert fv["removed_at"] == "fine_merge" and fv["coarse_output"] == 1 and fv["final"] is None                # kept by the coarse merge, lost in the fine one
    f0 = M.input_fate(var, 0)
    assert f0["final"] == 0 and f0["final_position"] == [12.5, 35.5] and f0["relocation_deg"] == round(float(np.hypot(0.5, 0.5)), 3)
    fates = M.input_fates(base, var)
    assert fates["counts"] == {"inputs": 2, "changed": 2, "removed_baseline": 1, "removed_variant": 1, "newly_removed": 0, "newly_surviving": 0}
    seg = {"step": 1006, "axis_lon": 35.0, "lat_range": [9.0, 15.0]}
    sb, sv = M.segment_correspondence(base, seg), M.segment_correspondence(var, seg)
    assert sb["nearest"]["distance_deg"] == round(float(np.hypot(4.0, 4.0)), 3) and sb["within_tol"] == 0       # 19 N is 4 degrees north of the segment's end
    assert sv["nearest"]["index"] == 0 and sv["nearest"]["distance_deg"] == 0.5 and sv["within_tol"] == 1 and sv["within_alt"] == 2
    changed_exhausted = {"coarse_lineage": [dict(coarse_b[0], selected={"sha256": "z" * 64}), dict(coarse_b[1])]}
    assert M.selection_invariance(base, changed_exhausted, 7) == (1, [])
    assert M.selection_invariance(base, changed_exhausted, 7, include_exhausted=True) == (2, [{"step": 7, "input": 0}])
    with pytest.raises(SystemExit, match="F1"):
        M.variant_step(base, 7, "d", {"final": []}, lambda: dict(changed_exhausted, final=[]), strict=True)
