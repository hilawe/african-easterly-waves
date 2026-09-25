"""The pair vertex substitution's frozen procedure, with the tracker faked.

What is tested is the procedure: the replacement lands at the paired position and leaves
every other axis untouched, and refuses a population it was not chosen against; a
replacement that changes nothing at detection makes no replay; items that share a step and
an observation are judged from one replay; the negative step lies outside the union of the
items' neighborhoods; the computed pair must reproduce the recorded comparison; the merge
must receive the supplied center exactly; the replay ceiling is enforced before each
replay; and both controls run after a failed intervention.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import validation_phase_b as B  # noqa: E402
import substitution_pilot as S  # noqa: E402

pytest.importorskip("scipy")

STEP = 1.25
OBS = (11.5, 11.5)
TIMES = np.arange(1.0, 4.0, 0.25)
REF7 = (np.array([1.0, 1.25, 1.5]), np.array([10.0, 11.5, 12.0]), np.array([10.0, 11.5, 12.0]))
REF8 = (np.array([1.25, 1.5]), np.array([11.5, 13.0]), np.array([11.5, 13.0]))


def _load():
    path = os.environ.get("PAIR_SCRIPT", os.path.join(ROOT, "scripts", "pair_vertex_substitution.py"))
    spec = importlib.util.spec_from_file_location("pair_vertex_substitution", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _track(t, la, lo):
    return {"time": np.array(t), "meanlat": np.array(la), "meanlon": np.array(lo)}


BASE = [_track([1.25, 1.5], [11.6, 12.0], [11.6, 12.0])]
OTHER = [_track([1.25, 1.5], [11.7, 12.0], [11.7, 12.0])]


def _port_axes(case, when):
    return [(np.array([0.0]), np.array([0.0])),
            (np.array([10.0, 12.0]), np.array([10.0, 12.0])),
            (np.array([20.0]), np.array([20.0]))]


def _v1(vertices):
    return [(np.array([0.0]), np.array([0.0])), vertices, (np.array([30.0]), np.array([30.0]))]


def _record(v1_centroid, v1_points):
    return {"nearest_axis_any_side": {"port": [1.0, [11.0, 11.0], 2],
                                      "v1": [1.0, list(v1_centroid), v1_points]}}


def _centers_of(axes):
    return [(float(np.mean(a)), float(np.mean(b))) for a, b in axes]


ACTING = (np.array([11.4, 11.4, 11.4]), np.array([11.4, 11.4, 11.4]))


def _stage(monkeypatch, v1_vertices=ACTING, elsewhere=None, reproduce=(7, 8), discrepancy=None,
           merge_center_override=None, exact_center=None):
    P = _load()
    reference = {}
    monkeypatch.setattr(B, "reference_track", lambda ref, index: REF7 if index == 7 else REF8)
    monkeypatch.setattr(B, "axes_at", _port_axes)
    discrepancy = discrepancy or {7: {"v1_extra": ["1.2500"], "port_extra": [], "displaced": []},
                                  8: {"v1_extra": ["1.2500"], "port_extra": [], "displaced": []}}
    monkeypatch.setattr(S, "item_discrepancy", lambda ref, port, item: discrepancy[item["index"]])
    monkeypatch.setattr(B.D, "merge_contours", lambda candidates, *a, **k: [])
    elsewhere = v1_vertices if elsewhere is None else elsewhere
    capture = {f"{STEP:.4f}": _v1(v1_vertices), "2.5000": _v1(elsewhere), "3.0000": _v1(elsewhere),
               "3.5000": _v1(elsewhere)}
    centers = {k: _centers_of(v) for k, v in capture.items()}
    if exact_center is not None:
        # the recorded full-precision center, supplied APART from the vertex list's mean
        centers[f"{STEP:.4f}"][1] = exact_center
    log = {"replays": [], "screens": 0}

    def fake_detect_at(case, when, edit=None):
        B.COST["screen_calls"] += 1
        log["screens"] += 1
        axes = _port_axes(case, when)
        if edit is not None:
            axes = edit(axes)
        return tuple((float(when),) + c for c in _centers_of(axes))

    def fake_replay(case, hook=None):
        B.COST["replay_calls"] += 1
        if hook is None:
            log["replays"].append(("baseline", None))
            return BASE
        now, fired = [None], []
        spy = hook(lambda *a, **k: _port_axes(None, STEP), now)
        for t in case["time"]:
            now[0] = float(t)
            out = spy(case, float(t), None)
            if _centers_of(out) != _centers_of(_port_axes(case, t)):
                fired.append((float(t), _centers_of(out)))
                cands = [{"time": float(t), "lat_mean": la, "lon_mean": lo} for la, lo in _centers_of(out)]
                if merge_center_override is not None:
                    cands[1]["lat_mean"], cands[1]["lon_mean"] = merge_center_override
                B.D.merge_contours(cands, None, None, None, None)
        log["replays"].append(("hooked", fired))
        if not fired:
            return BASE
        step, centers_now = fired[0]
        if step == STEP and centers_now[1] != (11.0, 11.0) and centers_now[1] != (11.6, 11.6):
            tracks = []
            if 7 in reproduce:
                tracks.append(_track(*REF7))
            if 8 in reproduce:
                tracks.append(_track(*REF8))
            return tracks or OTHER
        return OTHER
    monkeypatch.setattr(B, "replay", fake_replay)
    monkeypatch.setattr(B, "detect_at", fake_detect_at)
    items = [{"index": 7, "kind": "extra_v1_only", "port_index": 0},
             {"index": 8, "kind": "unmatched_v1", "port_index": None}]
    return P, reference, items, capture, centers, log


def _run(P, reference, items, capture, centers, record, ceiling=10):
    case = {"time": TIMES}
    budget = P.Budget(ceiling)
    baseline = budget.replay(case)
    out, runs = P.investigate(case, reference, {}, items, STEP, capture, centers, TIMES, record,
                              baseline, budget)
    return out, runs, budget


def test_replacement_edit_lands_at_the_position_and_refuses_a_wrong_population():
    P = _load()
    port = _port_axes(None, STEP)
    edit, hits = P.replacement_edit(1, 3, port[1], ([11.4, 11.4], [11.4, 11.4]))
    out = edit(port)
    assert hits[0] == 1
    assert out[0] is port[0] and out[2] is port[2]
    assert list(out[1][0]) == [11.4, 11.4] and list(out[1][1]) == [11.4, 11.4]
    with pytest.raises(P.EditTargetMismatch):
        edit(port[:2])
    with pytest.raises(P.EditTargetMismatch):
        edit([port[1], port[0], port[2]])
    # a different axis with the same centroid is not the axis the pair was chosen against
    same_centroid = (np.array([9.0, 13.0]), np.array([9.0, 13.0]))
    with pytest.raises(P.EditTargetMismatch):
        edit([port[0], same_centroid, port[2]])


def test_a_replacement_that_changes_nothing_at_detection_makes_no_replay(monkeypatch):
    # version 1's line has the same mean as the port's, with a different extent
    P, ref, items, capture, centers, log = _stage(
        monkeypatch, v1_vertices=(np.array([10.0, 12.0, 10.0, 12.0]), np.array([10.0, 12.0, 10.0, 12.0])),
        elsewhere=ACTING)
    out, runs, budget = _run(P, ref, items, capture, centers, _record((11.0, 11.0), 4))
    assert out["outcomes"] == {"7": "UNEXPLAINED", "8": "UNEXPLAINED"}
    assert "screened out at detection" in out["why"]
    assert out["detection_stage"]["vertex_list"]["acts"] is False
    assert out["replays_after_baseline"] == 0 and budget.made == 1
    # the detection reading comes first, so no control was selected for a screened-out step
    assert out["controls"]["negative"]["selected_step"] is None
    assert out["controls"]["negative"]["eligible_steps"] == []


def test_items_sharing_the_step_are_judged_from_one_replay(monkeypatch):
    P, ref, items, capture, centers, log = _stage(monkeypatch, reproduce=(7,))
    out, runs, budget = _run(P, ref, items, capture, centers, _record((11.4, 11.4), 3))
    assert out["outcomes"] == {"7": "EXPLAINED", "8": "UNEXPLAINED"}
    assert out["reproduced"] == {"7": {"intervention": True}, "8": {"intervention": False}}
    # one intervention replay plus two controls, and the exact center was not replayed
    assert out["replays_after_baseline"] == 3
    assert set(runs) == {"baseline", "intervention", "representation_control", "negative_control"}
    assert out["detection_stage"]["exact_potwv_center_as_single_point"]["same_signature_as_vertex_list"]
    assert out["controls"]["representation"]["passed"] and out["controls"]["negative"]["passed"]
    assert out["merge_received"]["intervention"]["equals_supplied_exactly"] is True
    assert out["merge_received"]["intervention"]["received_center_at_position"] == [11.4, 11.4]


def test_negative_step_lies_outside_the_union_of_the_items_neighborhoods(monkeypatch):
    disc = {7: {"v1_extra": ["1.2500"], "port_extra": [], "displaced": []},
            8: {"v1_extra": ["1.2500"], "port_extra": [], "displaced": ["2.5000"]}}
    P, ref, items, capture, centers, log = _stage(monkeypatch, discrepancy=disc)
    out, runs, budget = _run(P, ref, items, capture, centers, _record((11.4, 11.4), 3))
    # 2.5 and its two neighbors either side are excluded by item 8, so 3.0 is out as well
    assert out["controls"]["negative"]["selected_step"] == 3.5
    assert [e["step"] for e in out["controls"]["negative"]["eligible_steps"]] == [3.5]


def test_the_pair_must_reproduce_the_recorded_comparison(monkeypatch):
    P, ref, items, capture, centers, log = _stage(monkeypatch)
    out, runs, budget = _run(P, ref, items, capture, centers, _record((11.4, 11.5), 3))
    assert out["outcomes"] == {"7": "UNSUPPORTED", "8": "UNSUPPORTED"}
    assert "does not reproduce the recorded comparison" in out["why"]
    assert out["replays_after_baseline"] == 0


def test_the_merge_must_receive_the_supplied_center_exactly(monkeypatch):
    P, ref, items, capture, centers, log = _stage(monkeypatch, merge_center_override=(11.4, 11.40001))
    out, runs, budget = _run(P, ref, items, capture, centers, _record((11.4, 11.4), 3))
    assert out["outcomes"] == {"7": "ANOMALOUS", "8": "ANOMALOUS"}
    assert "did not receive the supplied centers exactly" in out["why"]


def test_the_replay_ceiling_is_enforced_before_each_replay(monkeypatch):
    P, ref, items, capture, centers, log = _stage(monkeypatch)
    out, runs, budget = _run(P, ref, items, capture, centers, _record((11.4, 11.4), 3), ceiling=2)
    assert out["outcomes"] == {"7": "NOT_RUN", "8": "NOT_RUN"}
    assert budget.made == 2 and "ceiling" in out["why"]
    assert not out["controls"]["representation"]["executed"]


def test_both_controls_run_after_a_failed_intervention(monkeypatch):
    P, ref, items, capture, centers, log = _stage(monkeypatch, reproduce=())
    out, runs, budget = _run(P, ref, items, capture, centers, _record((11.4, 11.4), 3))
    assert out["outcomes"] == {"7": "UNEXPLAINED", "8": "UNEXPLAINED"}
    assert out["controls"]["representation"]["executed"] and out["controls"]["negative"]["executed"]
    assert out["replays_after_baseline"] == 3
    assert "negative_control" in runs and "representation_control" in runs


def test_the_exact_center_is_screened_apart_from_the_vertex_list(monkeypatch):
    # the recorded center differs from the vertex list's mean and changes detection on its own
    P, ref, items, capture, centers, log = _stage(monkeypatch, reproduce=(7,), exact_center=(11.6, 11.6))
    out, runs, budget = _run(P, ref, items, capture, centers, _record((11.4, 11.4), 3))
    stage = out["detection_stage"]["exact_potwv_center_as_single_point"]
    assert stage["acts"] is True and stage["same_signature_as_vertex_list"] is False
    assert set(runs) == {"baseline", "intervention", "intervention_exact_center",
                         "representation_control", "negative_control"}
    assert out["reproduced"] == {"7": {"intervention": True, "intervention_exact_center": False},
                                 "8": {"intervention": False, "intervention_exact_center": False}}
    assert out["merge_received"]["intervention_exact_center"]["received_center_at_position"] == [11.6, 11.6]
    assert out["replays_after_baseline"] == 4


def test_a_vertex_list_that_changes_nothing_still_yields_to_an_acting_exact_center(monkeypatch):
    P, ref, items, capture, centers, log = _stage(
        monkeypatch, v1_vertices=(np.array([10.0, 12.0, 10.0, 12.0]), np.array([10.0, 12.0, 10.0, 12.0])),
        elsewhere=ACTING, reproduce=(), exact_center=(11.6, 11.6))
    out, runs, budget = _run(P, ref, items, capture, centers, _record((11.0, 11.0), 4))
    assert out["detection_stage"]["vertex_list"]["acts"] is False
    assert "only the exact center is replayed" in out["note"]
    assert set(runs) == {"baseline", "intervention_exact_center", "representation_control", "negative_control"}
