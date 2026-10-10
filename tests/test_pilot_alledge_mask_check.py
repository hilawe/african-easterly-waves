"""The mask check compares two threshold pairs by the decisions the detector and the merge
make, on both grids, at the base level and every ladder level, and refuses an artifact
that is not a smooth-then-crop calibration."""
import importlib.util
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load(name):
    spec = importlib.util.spec_from_file_location(name + "_under_test", os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_decisions_are_the_detectors_and_the_merges_and_differ_only_where_a_cell_crosses_a_level():
    M = _load("pilot_alledge_mask_check")
    from aew.v1port.contours import THRESHOLD_LADDER
    coarse = np.array([[1.0e-7, 4.5e-7, np.nan], [6.5e-7, 1.0e-6, 1.6e-6]])
    fine = np.array([[2.0e-6, 2.6e-6], [np.nan, 9.0e-6]])
    fields = {"coarse_curvature": coarse, "fine_curvature": fine}
    same = M.compare_pairs(fields, (4.0e-7, 2.5e-6), (4.0e-7, 2.5e-6))
    assert same["coarse"]["base"] == 0 and all(v == 0 for v in same["coarse"]["ladder"].values()) and same["fine"]["base"] == 0
    assert list(same["coarse"]["ladder"]) == [f"{lv:g}" for lv in THRESHOLD_LADDER] and same["coarse"]["cells"] == 6
    moved = M.compare_pairs(fields, (4.0e-7, 2.5e-6), (4.8e-7, 2.7e-6))
    assert moved["coarse"]["base"] == 1                                                         # 4.5e-7 is below 4.8e-7 and not below 4.0e-7
    assert moved["coarse"]["ladder"]["1"] == 1 and moved["coarse"]["ladder"]["1.5"] == 1       # 4.5e-7 at level 1, 6.5e-7 at level 1.5 (6.0e-7 against 7.2e-7)
    assert moved["fine"]["base"] == 1 and moved["fine"]["ladder"]["1"] == 1                       # 2.6e-6 crosses the fine base and level 1
    base, ladder, levels = M.mask_decisions(coarse, 4.0e-7)
    assert base.tolist() == [[True, False, False], [False, False, False]]                         # NaN is not below the threshold, as in detect_troughs
    assert ladder[0].tolist() == [[False, True, False], [True, True, True]]                       # NaN is not at or above a level, as in merge_contours


def test_a_non_smooth_then_crop_artifact_and_an_existing_output_are_refused(tmp_path, monkeypatch):
    M = _load("pilot_alledge_mask_check")
    out = tmp_path / "o.json"
    out.write_text("{}")
    with pytest.raises(SystemExit, match="never overwritten"):
        M.main(["--control-run", "a", "--control-case", "b", "--wide-dir", "c", "--recomputed", "d", "--year", "1990", "--out", str(out)])


def test_the_recomputed_pair_is_read_only_from_a_bound_audit_record_for_this_runs_pair():
    M = _load("pilot_alledge_mask_check")
    rec = {"generated_by": "scripts/pilot_alledge_calibration.py",
           "orders": {"crop_then_smooth": {"coarse": {"threshold": 1.0}, "fine": {"threshold": 2.0}}, "smooth_then_crop": {"coarse": {"threshold": 1.1}, "fine": {"threshold": 2.2}}},
           "bind": {"archived_order_reproduces_released_pair": True, "got": [1.0, 2.0], "released": [1.0, 2.0]}, "released_artifact": {"coarse": 1.0, "fine": 2.0}}
    assert M.recomputed_pair(rec, (1.0, 2.0)) == (1.1, 2.2)
    with pytest.raises(SystemExit, match="not this control run's pair"):
        M.recomputed_pair(rec, (1.0, 2.5))
    with pytest.raises(SystemExit, match="did not reproduce"):
        M.recomputed_pair({**rec, "bind": {"archived_order_reproduces_released_pair": False}}, (1.0, 2.0))
    with pytest.raises(SystemExit, match="not the calibration audit's record"):
        M.recomputed_pair({**rec, "orders": {"crop_then_smooth": rec["orders"]["crop_then_smooth"]}}, (1.0, 2.0))
    with pytest.raises(SystemExit, match="not the calibration audit's record"):
        M.recomputed_pair({**rec, "generated_by": "scripts/compute_thresholds.py"}, (1.0, 2.0))


def test_an_inconsistent_audit_record_is_refused_whatever_its_flag_says():
    M = _load("pilot_alledge_mask_check")
    good = {"generated_by": "scripts/pilot_alledge_calibration.py",
            "orders": {"crop_then_smooth": {"coarse": {"threshold": 1.0}, "fine": {"threshold": 2.0}}, "smooth_then_crop": {"coarse": {"threshold": 1.1}, "fine": {"threshold": 2.2}}},
            "bind": {"archived_order_reproduces_released_pair": True, "got": [1.0, 2.0], "released": [1.0, 2.0]}, "released_artifact": {"coarse": 1.0, "fine": 2.0}}
    assert M.recomputed_pair(good, (1.0, 2.0)) == (1.1, 2.2)
    with pytest.raises(SystemExit, match="did not reproduce"):                                        # a truthy string is not the flag
        M.recomputed_pair({**good, "bind": {**good["bind"], "archived_order_reproduces_released_pair": "false"}}, (1.0, 2.0))
    with pytest.raises(SystemExit, match="not the calibration audit's record"):                       # the archived order must be present
        M.recomputed_pair({**good, "orders": {"smooth_then_crop": good["orders"]["smooth_then_crop"]}}, (1.0, 2.0))
    with pytest.raises(SystemExit, match="not this control run's pair"):                              # the archived pair and bind.got say 1.0 is the released coarse value
        M.recomputed_pair({**good, "orders": {**good["orders"], "crop_then_smooth": {"coarse": {"threshold": 1.5}, "fine": {"threshold": 2.0}}},
                           "bind": {**good["bind"], "got": [1.5, 2.0]}}, (1.0, 2.0))
    with pytest.raises(SystemExit, match="not this control run's pair"):                              # bind.got disagrees with bind.released
        M.recomputed_pair({**good, "bind": {**good["bind"], "got": [1.5, 2.0]}}, (1.0, 2.0))
    with pytest.raises(SystemExit, match="not this control run's pair"):                              # bind.released disagrees with the audited artifact
        M.recomputed_pair({**good, "bind": {**good["bind"], "released": [1.0, 2.5]}}, (1.0, 2.0))
    with pytest.raises(SystemExit, match="archived order gives"):                                     # the archived pair disagrees with the bind it claims
        M.recomputed_pair({**good, "orders": {**good["orders"], "crop_then_smooth": {"coarse": {"threshold": 1.0}, "fine": {"threshold": 2.5}}}}, (1.0, 2.0))
    with pytest.raises(SystemExit, match="audited released artifact gives"):                          # only the audited artifact's pair disagrees
        M.recomputed_pair({**good, "released_artifact": {"coarse": 1.0, "fine": 2.5}}, (1.0, 2.0))
    with pytest.raises(SystemExit, match="got value gives"):                                          # only bind.got disagrees
        M.recomputed_pair({**good, "bind": {**good["bind"], "got": [1.0, 2.5]}}, (1.0, 2.0))
    with pytest.raises(SystemExit, match="released value gives"):                                     # only bind.released disagrees
        M.recomputed_pair({**good, "bind": {**good["bind"], "released": [1.0, 2.5]}}, (1.0, 2.0))


def test_a_cell_exactly_at_a_threshold_decides_as_the_detector_and_the_merge_do():
    M = _load("pilot_alledge_mask_check")
    t = 4.0e-7
    below, above = np.nextafter(t, 0.0), np.nextafter(t, 1.0)
    coarse = np.array([[below, t, above], [np.nan, 1.5 * t, np.nextafter(1.5 * t, 0.0)]])
    base, ladder, levels = M.mask_decisions(coarse, t)
    assert base.tolist() == [[True, False, False], [False, False, False]]                            # strictly below passes the base mask
    assert ladder[0].tolist() == [[False, True, True], [False, True, True]]                          # at or above the level passes the ladder
    assert ladder[1].tolist() == [[False, False, False], [False, True, False]]                       # level 1.5 exactly at 1.5 t, not one float below
