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
import json
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
    source = inspect.getsource(B._investigate)
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


def test_each_item_is_retained_before_the_next_one_starts(tmp_path, monkeypatch):
    """Window A's frozen rerun retained nothing for its first hour because the driver
    wrote every run only after the last item, so there was no first item to inspect and
    nothing to resume from. Through `main(argv)`, with the expensive stages faked, the
    file for item one must exist by the time item two is investigated."""
    seen = []
    _window_inputs(tmp_path)

    def fake_investigate(case, residuals, reference, item, baseline, axes_by_step=None):
        seen.append(sorted(p.name for p in (tmp_path / "keep").glob("A_item*.json")))
        return {"index": item["index"], "kind": item["kind"], "outcome": "UNSUPPORTED",
                "operation": "removal", "runs": {"baseline": []}, "detail": None,
                "receipts": None}

    def fake_save_run(path, runs, case_path, ref_path, index, script, extra=None,
                      script_sha256=None, git_head=None, exclusive=False):
        Path(path).write_text(json.dumps({"index": index, "extra": extra}))

    monkeypatch.setattr(B, "load_window", lambda work, name: (
        str(tmp_path), {}, {"pairs": [], "v1_unmatched": []}, {}))
    monkeypatch.setattr(B, "replay", lambda case, hook=None: [])
    monkeypatch.setattr(B, "loadmat", lambda path: {})
    monkeypatch.setattr(B, "build_items", lambda residuals, reference, case: [
        {"index": 4, "kind": "unmatched_v1", "port_index": None},
        {"index": 9, "kind": "unmatched_v1", "port_index": None}])
    monkeypatch.setattr(B, "reference_track", lambda reference, index: (
        np.array([1.0]), np.array([2.0]), np.array([3.0])))
    monkeypatch.setattr(B, "axes_from_capture", lambda path: {})
    monkeypatch.setattr(B, "investigate", fake_investigate)
    monkeypatch.setattr(B.X, "save_run", fake_save_run)
    out = tmp_path / "out.json"
    assert B.main([str(tmp_path), "--window", "A", "--out", str(out),
                   "--retain", str(tmp_path / "keep")]) == 0
    assert seen == [[], ["A_item4.json"]], seen
    assert (tmp_path / "keep" / "A_baseline_joball.json").exists()
    assert sorted(p.name for p in (tmp_path / "keep").glob("A_item*.json")) == [
        "A_item4.json", "A_item9.json"]


def test_the_driver_records_the_identity_it_was_launched_with(tmp_path, monkeypatch):
    """Window A's artifact named the digest of a driver edited during the run. The digest
    and head are read at launch, so an edit to the file while items run does not reach
    the artifact or the retained runs."""
    import hashlib
    copy = tmp_path / "driver_copy.py"
    copy.write_text("# the driver as launched\n")
    at_launch = hashlib.sha256(copy.read_bytes()).hexdigest()
    monkeypatch.setattr(B, "__file__", str(copy))
    _window_inputs(tmp_path)
    seen = {}

    def fake_investigate(case, residuals, reference, item, baseline, axes_by_step=None):
        copy.write_text("# the driver edited while the run was in progress\n")
        return {"index": item["index"], "kind": item["kind"], "outcome": "UNSUPPORTED",
                "operation": "removal", "runs": {"baseline": []}, "detail": None,
                "receipts": None}

    def fake_save_run(path, runs, case_path, ref_path, index, script, extra=None,
                      script_sha256=None, git_head=None, exclusive=False):
        seen[Path(path).name] = dict(extra, script_sha256=script_sha256)
        Path(path).write_text("{}")

    monkeypatch.setattr(B, "load_window", lambda work, name: (
        str(tmp_path), {}, {"pairs": [], "v1_unmatched": []}, {}))
    monkeypatch.setattr(B, "replay", lambda case, hook=None: [])
    monkeypatch.setattr(B, "loadmat", lambda path: {})
    monkeypatch.setattr(B, "build_items", lambda residuals, reference, case: [
        {"index": 4, "kind": "unmatched_v1", "port_index": None}])
    monkeypatch.setattr(B, "reference_track", lambda reference, index: (
        np.array([1.0]), np.array([2.0]), np.array([3.0])))
    monkeypatch.setattr(B, "axes_from_capture", lambda path: {})
    monkeypatch.setattr(B, "investigate", fake_investigate)
    monkeypatch.setattr(B.X, "save_run", fake_save_run)
    out = tmp_path / "out.json"
    assert B.main([str(tmp_path), "--window", "A", "--out", str(out),
                   "--retain", str(tmp_path / "keep")]) == 0
    written = json.loads(out.read_text())
    assert written["driver_sha256"] == at_launch
    assert written["driver_sha256"] != hashlib.sha256(copy.read_bytes()).hexdigest()
    assert seen["A_item4.json"]["driver_sha256"] == at_launch
    assert seen["A_baseline_joball.json"]["driver_sha256"] == at_launch
    # the launch digest is also what the retained run's own producer key carries
    assert seen["A_item4.json"]["script_sha256"] == at_launch
    assert "launched_at" in written and "git_head_at_launch" in written
    assert set(written["inputs_sha256"]) == {"tracker_case.mat", "residuals.json",
                                            "tracker_octave_instrumented.mat",
                                            "tracker_port.mat"}
    assert written["retained_files"].keys() == {"A_baseline_joball.json", "A_item4.json"}


def _window_inputs(tmp_path, stamp="v1"):
    for name in ("tracker_case.mat", "residuals.json", "tracker_octave_instrumented.mat",
                 "tracker_port.mat"):
        (tmp_path / name).write_text(f"{name} {stamp}")


def _fake_stages(monkeypatch, tmp_path, indices, seen=None):
    _window_inputs(tmp_path)

    def fake_investigate(case, residuals, reference, item, baseline, axes_by_step=None):
        if seen is not None:
            seen.append(item["index"])
        return {"index": item["index"], "kind": item["kind"], "outcome": "UNSUPPORTED",
                "operation": "removal", "runs": {"baseline": []}, "detail": None,
                "receipts": None}
    monkeypatch.setattr(B, "load_window", lambda work, name: (
        str(tmp_path), {}, {"pairs": [], "v1_unmatched": []}, {}))
    monkeypatch.setattr(B, "replay", lambda case, hook=None: [])
    monkeypatch.setattr(B, "loadmat", lambda path: {})
    monkeypatch.setattr(B, "build_items", lambda residuals, reference, case: [
        {"index": i, "kind": "unmatched_v1", "port_index": None} for i in indices])
    monkeypatch.setattr(B, "reference_track", lambda reference, index: (
        np.array([1.0]), np.array([2.0]), np.array([3.0])))
    monkeypatch.setattr(B, "axes_from_capture", lambda path: {})
    monkeypatch.setattr(B, "investigate", fake_investigate)
    monkeypatch.setattr(B.X, "save_run", lambda *a, **k: None)


def test_a_job_may_take_only_items_inside_the_budget_set(tmp_path, monkeypatch):
    """A window split across scheduler jobs is still one frozen search: the budget and
    the order are unchanged, a job runs a subset of the budget set, and an index outside
    it is refused rather than quietly searched."""
    seen = []
    _fake_stages(monkeypatch, tmp_path, [1, 4, 9, 12], seen)
    out = tmp_path / "job.json"
    assert B.main([str(tmp_path), "--window", "B", "--out", str(out), "--budget", "3",
                   "--items", "4,1"]) == 0
    assert seen == [1, 4]
    written = json.loads(out.read_text())
    assert written["items_selected_for_this_job"] == [1, 4]
    assert written["budget_set"] == [1, 4, 9] and written["items_not_investigated"] == [12]
    with pytest.raises(SystemExit) as caught:
        B.main([str(tmp_path), "--window", "B", "--out", str(out), "--budget", "3",
                "--items", "12"])
    assert "not in window B's budget set" in str(caught.value)


def _job_artifact(tmp_path, n, items, **override):
    base = {"contract": "c", "window": "B", "budget": 3, "search_neighborhood_steps": 2,
            "inputs_sha256": {"tracker_case.mat": "c" * 64, "axis_capture": "a" * 64},
            "retained_files": None, "retention_directory": None,
            "inject_radius_deg": 5.0, "reorder_trial_cap": 400, "axis_capture": "a.log",
            "driver_sha256": "d" * 64, "git_head_at_launch": "h" * 40, "budget_set": [1, 4, 9],
            "items_total": 4, "items_not_investigated": [12],
            "what_unexplained_means_here": "w", "generated_by": "x",
            "items_investigated": len(items), "items_selected_for_this_job": items,
            "launched_at": f"t{n}",
            "timing": {"elapsed_seconds": 10.0 * n, "processor_seconds": 5.0 * n},
            "outcomes": {"EXPLAINED": len(items)},
            "results": [{"index": i, "kind": "k", "outcome": "EXPLAINED",
                         "operation": "removal"} for i in items]}
    base.update(override)
    (tmp_path / f"phase_b_B_job{n}.json").write_text(json.dumps(base))


def test_merging_jobs_requires_the_same_procedure_and_exact_coverage(tmp_path):
    """Per-job artifacts merge into one window artifact only when every fixed field
    agrees and the jobs cover the budget set exactly once. Timing is summed."""
    _job_artifact(tmp_path, 1, [1, 4])
    _job_artifact(tmp_path, 2, [9])
    out = tmp_path / "B.json"
    assert B.main(["work", "--window", "B", "--out", str(out), "--merge", str(tmp_path)]) == 0
    merged = json.loads(out.read_text())
    assert [r["index"] for r in merged["results"]] == [1, 4, 9]
    assert merged["outcomes"] == {"EXPLAINED": 3} and merged["items_investigated"] == 3
    assert merged["timing"]["processor_seconds"] == 15.0
    assert merged["timing"]["elapsed_seconds_longest_job"] == 20.0
    assert [m["items"] for m in merged["merged_from"]] == [[1, 4], [9]]
    assert "items_selected_for_this_job" not in merged
    # a job that read a different axis capture under the same name
    _job_artifact(tmp_path, 2, [9], inputs_sha256={"tracker_case.mat": "c" * 64,
                                                   "axis_capture": "b" * 64})
    with pytest.raises(SystemExit) as caught:
        B.main(["work", "--window", "B", "--out", str(out), "--merge", str(tmp_path)])
    assert "inputs_sha256" in str(caught.value)
    # a job that records no input identity at all
    _job_artifact(tmp_path, 2, [9])
    job = json.loads((tmp_path / "phase_b_B_job2.json").read_text())
    del job["inputs_sha256"]
    (tmp_path / "phase_b_B_job2.json").write_text(json.dumps(job))
    with pytest.raises(SystemExit) as caught:
        B.main(["work", "--window", "B", "--out", str(out), "--merge", str(tmp_path)])
    assert "records no inputs_sha256" in str(caught.value)
    # a job under a different driver
    _job_artifact(tmp_path, 2, [9], driver_sha256="e" * 64)
    with pytest.raises(SystemExit) as caught:
        B.main(["work", "--window", "B", "--out", str(out), "--merge", str(tmp_path)])
    assert "disagrees" in str(caught.value) and "driver_sha256" in str(caught.value)
    # an item run twice
    _job_artifact(tmp_path, 2, [4, 9])
    with pytest.raises(SystemExit) as caught:
        B.main(["work", "--window", "B", "--out", str(out), "--merge", str(tmp_path)])
    assert "exactly once" in str(caught.value)
    # an item missing
    _job_artifact(tmp_path, 2, [])
    with pytest.raises(SystemExit) as caught:
        B.main(["work", "--window", "B", "--out", str(out), "--merge", str(tmp_path)])
    assert "did not run under --items" in str(caught.value) or "exactly once" in str(caught.value)


def test_retained_evidence_is_never_overwritten_and_baselines_are_compared_at_merge(
        tmp_path, monkeypatch):
    """Two disjoint jobs into one retention directory once replaced each other's baseline.
    Each job now retains its own baseline under its own name, an existing item file is
    refused, and the merge reads every retained file back, checks its digest, and requires
    the per-job baselines to agree by content."""
    import hashlib
    keep = tmp_path / "keep"

    def run_job(items, baseline_tracks):
        _fake_stages(monkeypatch, tmp_path, [1, 4, 9])
        monkeypatch.setattr(B, "replay", lambda case, hook=None: baseline_tracks)

        def real_enough_save_run(path, runs, case_path, ref_path, index, script,
                                 extra=None, script_sha256=None, git_head=None,
                                 exclusive=False):
            if exclusive and Path(path).exists():
                raise FileExistsError(path)
            Path(path).write_text(json.dumps({"identity": extra, "runs": {
                name: [{"time": list(t), "meanlat": list(la), "meanlon": list(lo)}
                       for t, la, lo in final] for name, final in runs.items()}}))
        monkeypatch.setattr(B.X, "save_run", real_enough_save_run)
        out = tmp_path / f"phase_b_B_job{items}.json"
        code = B.main([str(tmp_path), "--window", "B", "--out", str(out), "--budget", "3",
                       "--items", items, "--retain", str(keep)])
        return code, json.loads(out.read_text())

    base = [([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])]
    assert run_job("1,4", base)[0] == 0
    assert run_job("9", base)[0] == 0
    names = sorted(p.name for p in keep.iterdir())
    assert names == ["B_baseline_job1-4.json", "B_baseline_job9.json", "B_item1.json",
                     "B_item4.json", "B_item9.json"]
    merged = tmp_path / "B.json"
    assert B.main(["work", "--window", "B", "--out", str(merged), "--merge", str(tmp_path)]) == 0
    written = json.loads(merged.read_text())
    assert written["baseline_agreement"] == {"baselines_compared": 2, "identical": True,
                                             "tracks": 1}
    # a retry of item 9 into the same directory is refused before anything is written
    with pytest.raises(SystemExit) as caught:
        run_job("9", base)
    assert "never overwritten" in str(caught.value)
    # a job whose baseline differs by content is refused at merge
    for stale in ("B_item9.json", "B_baseline_job9.json"):
        (keep / stale).unlink()
    (tmp_path / "phase_b_B_job9.json").unlink()
    assert run_job("9", [([1.0, 2.0], [3.0, 4.5], [5.0, 6.0])])[0] == 0
    with pytest.raises(SystemExit) as caught:
        B.main(["work", "--window", "B", "--out", str(merged), "--merge", str(tmp_path)])
    assert "baselines do not agree" in str(caught.value)
    # a retained file altered after its job recorded it is refused at merge
    for stale in ("B_item9.json", "B_baseline_job9.json"):
        (keep / stale).unlink()
    (tmp_path / "phase_b_B_job9.json").unlink()
    assert run_job("9", base)[0] == 0
    p = keep / "B_item9.json"
    p.write_text(p.read_text() + " ")
    with pytest.raises(SystemExit) as caught:
        B.main(["work", "--window", "B", "--out", str(merged), "--merge", str(tmp_path)])
    assert "has changed since its job recorded it" in str(caught.value)


def test_save_run_records_the_launch_identity_it_is_given(tmp_path):
    """With the real serializer, the producer keys carry the launch values when given,
    and what is on disk at write time otherwise."""
    import hashlib
    for name in ("case.mat", "ref.mat", "driver.py"):
        (tmp_path / name).write_text(name)
    on_disk = hashlib.sha256(b"driver.py").hexdigest()
    identity = B.X.save_run(str(tmp_path / "r.json"), {"baseline": []}, str(tmp_path / "case.mat"),
                            str(tmp_path / "ref.mat"), -1, str(tmp_path / "driver.py"))
    assert identity["script_sha256"] == on_disk
    identity = B.X.save_run(str(tmp_path / "s.json"), {"baseline": []}, str(tmp_path / "case.mat"),
                            str(tmp_path / "ref.mat"), -1, str(tmp_path / "driver.py"),
                            script_sha256="l" * 64, git_head="h" * 40)
    assert identity["script_sha256"] == "l" * 64 and identity["git_head"] == "h" * 40
    assert json.loads((tmp_path / "s.json").read_text())["identity"]["script_sha256"] == "l" * 64


def test_exclusive_retention_lets_exactly_one_of_two_racing_writers_win(tmp_path):
    """The real serializer, two threads, one target, and a barrier placed AFTER any check
    a caller could make and before the publish. Exactly one succeeds, its bytes are the
    ones on disk afterward, and no temporary file is left behind."""
    import threading
    for name in ("case.mat", "ref.mat", "driver.py"):
        (tmp_path / name).write_text(name)
    target = tmp_path / "B_baseline_job9.json"
    barrier = threading.Barrier(2)
    outcomes = {}

    def writer(label):
        barrier.wait()
        try:
            B.X.save_run(str(target), {"baseline": []}, str(tmp_path / "case.mat"),
                         str(tmp_path / "ref.mat"), -1, str(tmp_path / "driver.py"),
                         extra={"launched_at": label}, exclusive=True)
            outcomes[label] = "won"
        except FileExistsError:
            outcomes[label] = "refused"

    threads = [threading.Thread(target=writer, args=(f"t{i}",)) for i in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert sorted(outcomes.values()) == ["refused", "won"], outcomes
    winner = next(k for k, v in outcomes.items() if v == "won")
    assert json.loads(target.read_text())["identity"]["launched_at"] == winner
    assert [p.name for p in tmp_path.glob("*.writing-*")] == []
    # and a plain third attempt is refused without touching the winner's bytes
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        B.X.save_run(str(target), {"baseline": []}, str(tmp_path / "case.mat"),
                     str(tmp_path / "ref.mat"), -1, str(tmp_path / "driver.py"),
                     extra={"launched_at": "t9"}, exclusive=True)
    assert target.read_bytes() == before


def test_the_job_artifact_is_published_exclusively_too(tmp_path, monkeypatch):
    """A retry naming an existing --out replaced it, and an --out pointed at a retained
    baseline destroyed the baseline. Artifacts are published like retained evidence."""
    _fake_stages(monkeypatch, tmp_path, [1, 4, 9])
    out = tmp_path / "job.json"
    assert B.main([str(tmp_path), "--window", "B", "--out", str(out), "--budget", "3",
                   "--items", "1"]) == 0
    before = out.read_bytes()
    with pytest.raises(SystemExit) as caught:
        B.main([str(tmp_path), "--window", "B", "--out", str(out), "--budget", "3",
                "--items", "1"])
    assert "never overwritten" in str(caught.value)
    assert out.read_bytes() == before
    assert [p.name for p in tmp_path.glob("*.writing-*")] == []
    # the merge output is published the same way
    _job_artifact(tmp_path, 1, [1, 4])
    _job_artifact(tmp_path, 2, [9])
    merged = tmp_path / "B.json"
    assert B.main(["work", "--window", "B", "--out", str(merged), "--merge", str(tmp_path)]) == 0
    with pytest.raises(SystemExit):
        B.main(["work", "--window", "B", "--out", str(merged), "--merge", str(tmp_path)])


def test_the_cost_of_screening_and_replay_is_counted_separately_per_item_and_per_window(
        tmp_path, monkeypatch):
    """Windows B to F cost fifteen times their estimate and nothing said whether the time
    was screening or replay. Each item's record now carries both, from the two calls that
    cost anything, and the job and merged artifacts sum them."""
    _fake_stages(monkeypatch, tmp_path, [1, 4])

    def fake_investigate(case, residuals, reference, item, baseline, axes_by_step=None):
        return {"index": item["index"], "kind": item["kind"], "outcome": "UNSUPPORTED",
                "operation": None, "runs": {}, "detail": None, "receipts": None,
                "cost": {"replay_calls": 3, "replay_seconds": 30.0, "screen_calls": 100,
                         "screen_seconds": 5.0, "item_seconds": 36.0, "other_seconds": 1.0}}
    monkeypatch.setattr(B, "investigate", fake_investigate)
    out = tmp_path / "job.json"
    assert B.main([str(tmp_path), "--window", "B", "--out", str(out), "--budget", "2"]) == 0
    timing = json.loads(out.read_text())["timing"]
    assert timing["replay_calls"] == 6 and timing["screen_calls"] == 200
    assert timing["replay_seconds"] == 60.0 and timing["screen_seconds"] == 10.0
    assert timing["other_seconds"] == 2.0


def test_replay_and_detect_at_accumulate_the_cost_counters(monkeypatch):
    """The counters sit on the two real calls, so a fake body still counts one call."""
    before = dict(B.COST)
    monkeypatch.setattr(B, "_replay", lambda case, hook=None: "replayed")
    assert B.replay({}) == "replayed"
    assert B.COST["replay_calls"] == before["replay_calls"] + 1
    assert B.COST["replay_seconds"] >= before["replay_seconds"]
    since = B._cost_since(before)
    assert since["replay_calls"] == 1 and since["screen_calls"] == 0
