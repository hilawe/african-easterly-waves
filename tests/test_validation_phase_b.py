"""The phase B driver's control and screening paths.

THESE BIND THREE DEFECTS FOUND BY READING THE DRIVER, not by running it. Each would have
made the automated search report confidently wrong things, and none would have shown up as
an error.

1. THE DETECTION SCREEN decides whether an edit can affect anything downstream, and its
   first signature was `sorted((lat_mean, lon_mean))`. Sorting hides a pure REORDERING, and
   association iterates candidates in order. Dropping the footprints hides a change to
   `lat_wave` and `lon_wave`, from which association builds its hull polygons. Either makes
   the screen say "cannot change anything" about an edit that changes plenty.
2. THE INJECTION NEGATIVE CONTROL must carry as many injections as the experiment it
   controls. Its first version capped the control at three steps and paired the axis groups
   with control steps by enumeration order, so a four-step intervention was controlled by a
   three-step control carrying the wrong groups.
3. NO CREDIT WITHOUT A NEGATIVE CONTROL. Its first version fell through to EXPLAINED when
   no control could be built.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import validation_phase_b as B  # noqa: E402


def wave(time, lat, lon, foot_lat=(1.0, 2.0), foot_lon=(3.0, 4.0)):
    return {"time": time, "lat_mean": lat, "lon_mean": lon,
            "lat_wave": np.array(foot_lat), "lon_wave": np.array(foot_lon)}


def test_the_screen_sees_a_pure_reordering():
    """Association iterates candidates in order, so order is a downstream-visible change."""
    a = [wave(1.0, 10.0, 20.0), wave(1.0, 30.0, 40.0)]
    b = [a[1], a[0]]
    assert B.candidate_signature(a) != B.candidate_signature(b)


def test_the_screen_sees_a_changed_footprint_with_the_same_center():
    """Association builds hull polygons from lat_wave and lon_wave, not from the center."""
    a = [wave(1.0, 10.0, 20.0, foot_lat=(1.0, 2.0))]
    b = [wave(1.0, 10.0, 20.0, foot_lat=(1.0, 9.0))]
    assert B.candidate_signature(a) != B.candidate_signature(b)
    a2 = [wave(1.0, 10.0, 20.0, foot_lon=(3.0, 4.0))]
    b2 = [wave(1.0, 10.0, 20.0, foot_lon=(3.0, 9.0))]
    assert B.candidate_signature(a2) != B.candidate_signature(b2)


def test_the_screen_sees_a_changed_footprint_length():
    a = [wave(1.0, 10.0, 20.0, foot_lat=(1.0, 2.0), foot_lon=(3.0, 4.0))]
    b = [wave(1.0, 10.0, 20.0, foot_lat=(1.0, 2.0, 5.0), foot_lon=(3.0, 4.0, 6.0))]
    assert B.candidate_signature(a) != B.candidate_signature(b)


def test_the_screen_calls_identical_candidates_identical():
    """It must not be so strict that every edit survives the screen and nothing is saved."""
    a = [wave(1.0, 10.0, 20.0), wave(2.0, 30.0, 40.0)]
    b = [wave(1.0, 10.0, 20.0), wave(2.0, 30.0, 40.0)]
    assert B.candidate_signature(a) == B.candidate_signature(b)


def test_the_screen_sees_a_changed_time_or_center():
    base = [wave(1.0, 10.0, 20.0)]
    for other in ([wave(2.0, 10.0, 20.0)], [wave(1.0, 11.0, 20.0)],
                  [wave(1.0, 10.0, 21.0)]):
        assert B.candidate_signature(base) != B.candidate_signature(other)


def _item(discrepancy_all):
    return {"discrepancy_all": list(discrepancy_all)}


def _target(times):
    times = np.array(times, dtype=float)
    return times, np.zeros_like(times), np.zeros_like(times)


def test_the_negative_control_carries_as_many_injections_as_the_experiment():
    """A four-step intervention controlled by a three-step control is not a comparison."""
    target = _target([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    item = _item(["1.0000", "2.0000"])
    assert B.negative_steps(target, item, None, 4) == [3.0, 4.0, 5.0, 6.0]
    assert len(B.negative_steps(target, item, None, 6)) == 6


def test_it_refuses_rather_than_truncating_when_too_few_agreed_steps_exist():
    target = _target([1.0, 2.0, 3.0, 4.0])
    item = _item(["1.0000", "2.0000"])
    assert B.negative_steps(target, item, None, 2) == [3.0, 4.0]
    # three injections, only two agreed steps: None, so the caller refuses the credit
    assert B.negative_steps(target, item, None, 3) is None


def test_a_step_in_the_discrepancy_is_never_used_as_a_negative_control():
    """The control has to sit where the two sides ALREADY agree, or it controls nothing."""
    target = _target([1.0, 2.0, 3.0])
    item = _item(["2.0000"])
    steps = B.negative_steps(target, item, None, 2)
    assert steps == [1.0, 3.0]
    assert 2.0 not in steps


@pytest.mark.parametrize("wanted", [1, 2, 3])
def test_the_control_schedule_preserves_the_injected_groups(wanted):
    """Each control step carries the group its own intervention step carried, not whichever
    one enumeration paired with it."""
    picked = {f"{float(k):.4f}": [f"axes-{k}"] for k in range(1, wanted + 1)}
    negative = [10.0 + k for k in range(wanted)]
    ordered = sorted(picked)
    moved = {f"{float(negative[k]):.4f}": picked[key] for k, key in enumerate(ordered)}
    assert len(moved) == len(picked)
    assert sum(len(v) for v in moved.values()) == sum(len(v) for v in picked.values())
    assert [v for _, v in sorted(moved.items())] == [picked[k] for k in ordered]


def test_the_negative_control_is_sized_from_what_was_injected_not_from_the_candidates():
    """The candidate axis set is filtered before injection, and the control must follow the
    filtered set. Sizing it on the unfiltered one makes the control inject MORE groups at
    MORE steps than the experiment, which is the truncation defect in the other direction
    and is what the first fix missed."""
    picked = {"1.0000": ["a"], "2.0000": ["b"], "3.0000": ["c"]}
    injected = {"2.0000": ["b"]}                      # attempt_injection filtered two away
    negative = [10.0, 11.0, 12.0]
    ordered = sorted(injected)
    moved = {f"{float(negative[k]):.4f}": injected[key] for k, key in enumerate(ordered)}
    assert len(moved) == len(injected) == 1
    assert sum(len(v) for v in moved.values()) == sum(len(v) for v in injected.values())
    # sized on `picked` instead, the control would carry three groups against one
    wrong = {f"{float(negative[k]):.4f}": picked[key]
             for k, key in enumerate(sorted(picked))}
    assert len(wrong) != len(injected)


def test_a_null_says_whether_the_search_finished_or_ran_out():
    """Reporting an exhausted search and an empty one the same way hides which happened."""
    import inspect
    source = inspect.getsource(B.investigate)
    assert "SEARCH BUDGET EXHAUSTED" in source
    assert "No conclusion about whether a reordering" in source
    assert '== "cap"' in source
    cap_source = inspect.getsource(B.attempt_reordering)
    assert 'return "cap"' in cap_source


# ---------------------------------------------------------------------------------------
# F2 and F6 from the independent review. F2: removal and reordering could be credited with
# NO EFFECTIVE negative control (a one-axis removal dropped a nonexistent index so its output
# was the baseline; a two-axis reorder ran its control loop zero times). F6: reordering
# discarded its negative trajectories, so the verdict could not be recomputed from what was
# kept. The expensive search and replay are replaced by test doubles, as the review did.
# ---------------------------------------------------------------------------------------


def _fake_case(n_steps=3):
    return {"time": np.array([1.0, 2.0, 3.0][:n_steps])}


def _fake_reference(index, times=(1.0, 2.0)):
    t = np.array(times)
    return {f"time{index}": t, f"lat{index}": np.zeros_like(t), f"lon{index}": np.zeros_like(t)}


def _fake_item(index, kind="extra_port_only"):
    """Distinct from `_item(discrepancy_all)` above; a first version reused that name and
    silently broke two earlier tests by shadowing it."""
    return {"index": index, "kind": kind, "discrepancy_counts": {},
            "discrepancy": {"v1_extra": [], "port_extra": ["2.0000"], "displaced": []},
            "discrepancy_all": ["2.0000"]}


def test_a_one_axis_removal_is_unsupported_not_credited(monkeypatch):
    """The review's counterexample: spare=1 does not exist, the negative IS the baseline."""
    monkeypatch.setattr(B, "replay", lambda case, hook=None: ["run"])
    monkeypatch.setattr(B, "axes_at", lambda case, when: [("only", "axis")])
    monkeypatch.setattr(B, "attempt_removal",
                        lambda case, target, steps, baseline:
                        {"step": 2.0, "index": 0, "final": ["run"], "centroid": [0, 0]})
    monkeypatch.setattr(B, "attempt_reordering", lambda *a, **k: None)
    monkeypatch.setattr(B.X, "holds_exactly", lambda final, t, la, lo: False)
    monkeypatch.setattr(B.X, "require_identical", lambda *a, **k: True)
    out = B.investigate(_fake_case(), {}, _fake_reference(7), _fake_item(7), ["run"])
    assert out["outcome"] == "UNSUPPORTED"
    assert "only one axis" in out["why"]
    assert "operation" not in out


def test_a_two_axis_reordering_is_unsupported_not_credited(monkeypatch):
    """The review's counterexample: excluding source and destination leaves no alternative,
    the loop ran zero times, and the item was credited with negative_controls_tried=0."""
    monkeypatch.setattr(B, "replay", lambda case, hook=None: ["run"])
    monkeypatch.setattr(B, "axes_at", lambda case, when: [("a",), ("b",)])
    monkeypatch.setattr(B, "attempt_removal", lambda *a, **k: None)
    monkeypatch.setattr(B, "attempt_reordering",
                        lambda case, target, steps:
                        {"step": 2.0, "moved_from": 1, "moved_to": 0, "final": ["run"]})
    monkeypatch.setattr(B.X, "holds_exactly", lambda final, t, la, lo: False)
    monkeypatch.setattr(B.X, "require_identical", lambda *a, **k: True)
    out = B.investigate(_fake_case(), {}, _fake_reference(7), _fake_item(7), ["run"])
    assert out["outcome"] == "UNSUPPORTED"
    assert "no third axis" in out["why"]
    assert out.get("negative_controls_tried", 0) == 0


def test_a_reordering_credit_retains_every_negative_trajectory_with_its_move(monkeypatch):
    """F6: the negative verdict must be recomputable from what was kept."""
    seen = []
    def fake_replay(case, hook=None):
        seen.append(hook)
        return [f"run{len(seen)}"]
    monkeypatch.setattr(B, "replay", fake_replay)
    monkeypatch.setattr(B, "axes_at", lambda case, when: [("a",), ("b",), ("c",), ("d",)])
    monkeypatch.setattr(B, "attempt_removal", lambda *a, **k: None)
    monkeypatch.setattr(B, "attempt_reordering",
                        lambda case, target, steps:
                        {"step": 2.0, "moved_from": 3, "moved_to": 0, "final": ["hit"]})
    monkeypatch.setattr(B.X, "holds_exactly", lambda final, t, la, lo: False)
    monkeypatch.setattr(B.X, "require_identical", lambda *a, **k: True)
    out = B.investigate(_fake_case(), {}, _fake_reference(7), _fake_item(7), ["base"])
    assert out["outcome"] == "EXPLAINED" and out["operation"] == "reordering"
    # alternatives are the axes other than source 3 and destination 0: indices 1 and 2
    assert out["negative_controls_tried"] == 2
    retained = sorted(k for k in out["runs"] if k.startswith("negative_control_"))
    assert retained == ["negative_control_1_to_0", "negative_control_2_to_0"]
    assert all(out["runs"][k] for k in retained)
    receipts = out["receipts"]["negative_controls"]
    assert [(r["moved_from"], r["moved_to"]) for r in receipts] == [(1, 0), (2, 0)]
    assert out["receipts"]["representation_control"]["identity"] is True


def test_a_removal_credit_carries_receipts_naming_a_different_negative_target(monkeypatch):
    monkeypatch.setattr(B, "replay", lambda case, hook=None: ["run"])
    monkeypatch.setattr(B, "axes_at", lambda case, when: [("a",), ("b",), ("c",)])
    monkeypatch.setattr(B, "attempt_removal",
                        lambda case, target, steps, baseline:
                        {"step": 2.0, "index": 0, "final": ["run"], "centroid": [0, 0]})
    monkeypatch.setattr(B, "attempt_reordering", lambda *a, **k: None)
    monkeypatch.setattr(B.X, "holds_exactly", lambda final, t, la, lo: False)
    monkeypatch.setattr(B.X, "require_identical", lambda *a, **k: True)
    out = B.investigate(_fake_case(), {}, _fake_reference(7), _fake_item(7), ["run"])
    assert out["outcome"] == "EXPLAINED"
    r = out["receipts"]
    assert r["intervention"]["removed_index"] == 0
    assert r["negative_control"]["removed_index"] != r["intervention"]["removed_index"]
    assert r["negative_control"]["population_after"] == r["negative_control"]["population_before"] - 1
    assert r["representation_control"]["identity"] is True


@pytest.mark.parametrize("operation", ["removal", "reordering"])
def test_an_anomalous_verdict_still_retains_its_control_trajectories(monkeypatch, operation):
    """The confirmation review found the anomalous return preceded the retention, so the
    trajectories were discarded on exactly the path where they matter most."""
    monkeypatch.setattr(B, "replay", lambda case, hook=None: ["run"])
    monkeypatch.setattr(B, "axes_at", lambda case, when: [("a",), ("b",), ("c",), ("d",)])
    monkeypatch.setattr(B.X, "require_identical", lambda *a, **k: True)
    monkeypatch.setattr(B.X, "holds_exactly", lambda final, t, la, lo: final == ["run"])
    # the baseline must NOT reproduce, or the item is anomalous before any search runs
    baseline = ["base"]
    if operation == "removal":
        monkeypatch.setattr(B, "attempt_removal", lambda case, target, steps, b:
                            {"step": 2.0, "index": 0, "final": ["run"], "centroid": [0, 0]})
        monkeypatch.setattr(B, "attempt_reordering", lambda *a, **k: None)
    else:
        monkeypatch.setattr(B, "attempt_removal", lambda *a, **k: None)
        monkeypatch.setattr(B, "attempt_reordering", lambda case, target, steps:
                            {"step": 2.0, "moved_from": 3, "moved_to": 0, "final": ["run"]})
    out = B.investigate(_fake_case(), {}, _fake_reference(7), _fake_item(7), baseline)
    assert out["outcome"] == "ANOMALOUS", out
    runs = out.get("runs") or {}
    assert "representation_control" in runs
    negatives = [k for k in runs if k.startswith("negative_control")]
    assert negatives, "the anomalous path must retain the negative trajectories"
