"""The substitution pilot's frozen operation, through its command path with the replay faked.

What is tested is the procedure, not the tracker: the capture's validation and order
check, timestep resolution, that a step whose substitution leaves detection unchanged is
not replayed, that the negative control is chosen before the search from steps that act
outside the neighborhood and is counted as having fired exactly once, the outcomes and
control states, that retention carries the digests of the bytes consumed, and that the
aggregate replay count includes the baselines.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import validation_phase_b as B  # noqa: E402

pytest.importorskip("scipy")


def _load():
    spec = importlib.util.spec_from_file_location(
        "substitution_pilot", os.path.join(ROOT, "scripts", "substitution_pilot.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _track(t, la, lo):
    return {"time": np.array(t), "meanlat": np.array(la), "meanlon": np.array(lo)}


REF = (np.array([1.0, 1.25, 1.5]), np.array([10.0, 10.5, 11.0]), np.array([-20.0, -20.5, -21.0]))
BASE = [_track([1.25, 1.5], [10.6, 11.0], [-20.5, -21.0])]     # lacks the first observation, displaced at 1.25
EXACT = [_track([1.0, 1.25, 1.5], [10.0, 10.5, 11.0], [-20.0, -20.5, -21.0])]
OTHER = [_track([1.25, 1.5], [10.6, 11.0], [-20.5, -21.0])]


def _capture_text(steps, reversed_at=()):
    """A capture with one two-point axis per step, POTWV in the same order."""
    lines = []
    for s in steps:
        pts = [(1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)]
        for la1, lo1, la2, lo2 in pts:
            lines.append(f"AXISPTS {s:.4f} 2 {la1} {lo1} {la2} {lo2}")
        centers = [(2.0, 3.0), (6.0, 7.0)]
        if s in reversed_at:
            centers = centers[::-1]
        for k, (la, lo) in enumerate(centers, 1):
            lines.append(f"POTWV {s:.17g} {k} {la} {lo} {s:.17g}")
    return "\n".join(lines) + "\n"


def _stage(monkeypatch, tmp_path, capture_steps, reproduce_at=(), acting=(), fire_twice_at=()):
    """Fake the window and the tracker. `acting` are steps where substitution changes the
    detected signature, and `reproduce_at` are steps whose replay is exact."""
    S = _load()
    times = np.arange(1.0, 4.0, 0.25)
    case = {"time": times}
    port = {"time0": np.array([1.25, 1.5]), "lat0": np.array([10.6, 11.0]),
            "lon0": np.array([-20.5, -21.0])}
    digests = {name: f"{name}-digest" for name in S.INPUT_NAMES}
    monkeypatch.setattr(S, "load_window_snapshot", lambda work, name: (
        str(tmp_path), case, {"pairs": [{"v1_index": 7, "port_index": 0, "extra_kind": "extra_v1_only"}],
                              "v1_unmatched": []}, {}, port, _capture_text(capture_steps), digests))
    monkeypatch.setattr(B, "reference_track", lambda reference, index: REF)
    monkeypatch.setattr(B, "build_items", lambda residuals, reference, case: [
        {"index": 7, "kind": "extra_v1_only", "port_index": 0}])
    monkeypatch.setattr(B, "axes_at", lambda case, when: [("a", "b")] * 3)
    replays = []

    def fake_replay(case, hook=None):
        B.COST["replay_calls"] += 1
        if hook is None:
            return BASE
        # drive the real edit_hook the way the pipeline would, once per timestep
        now = [None]
        spy = hook(lambda *a, **k: ["port"], now)
        fired_at, substituted = [], False
        for t in case["time"]:
            now[0] = float(t)
            out = spy(None, None, None)
            repeats = 2 if float(t) in fire_twice_at else 1
            for _ in range(repeats - 1):
                spy(None, None, None)
            if out != ["port"]:
                fired_at.append(float(t))
                substituted = True
        replays.append((fired_at, substituted))
        if not substituted:
            return BASE
        return EXACT if fired_at and fired_at[0] in reproduce_at else OTHER

    def fake_detect_at(case, when, edit=None):
        B.COST["screen_calls"] += 1
        if edit is None or edit(["port"]) == ["port"]:
            return ("plain",)
        return ("changed",) if when in acting else ("plain",)
    monkeypatch.setattr(B, "replay", fake_replay)
    monkeypatch.setattr(B, "detect_at", fake_detect_at)
    saved = {}

    def fake_save_run(path, runs, case_path, ref_path, index, script, extra=None, **kw):
        saved[os.path.basename(path)] = {"extra": extra, **kw}
        open(path, "w").write("{}")
    monkeypatch.setattr(B.X, "save_run", fake_save_run)
    return S, replays, saved


def _run(S, tmp_path, name="pilot"):
    out = tmp_path / f"{name}.json"
    code = S.main([str(tmp_path), "--items", "B:7", "--out", str(out),
                   "--retain", str(tmp_path / f"keep_{name}")])
    return code, json.loads(out.read_text())


def test_an_explained_item_needs_an_acting_negative_that_fired_once(tmp_path, monkeypatch):
    S, replays, saved = _stage(monkeypatch, tmp_path, [1.0, 1.25, 3.0], reproduce_at=(1.0,),
                               acting=(1.0, 3.0))
    code, written = _run(S, tmp_path)
    assert code == 0
    r = written["results"][0]
    assert r["outcome"] == "EXPLAINED", r
    assert r["candidate_steps"] == [1.0, 1.25] and r["steps_tried"] == [1.0]
    c = r["controls"]
    assert c["negative"]["selected_step"] == 3.0 and c["negative"]["executed"] is True
    assert c["negative"]["passed"] is True and c["negative"]["edit_fired"] == 1
    assert c["representation"]["passed"] is True and c["representation"]["edit_fired"] == 1
    assert [e["step"] for e in c["negative"]["eligible_steps"]] == [3.0]
    assert r["achieved_grade"].startswith("two-control")
    assert r["replays_after_baseline"] == 3
    # the aggregate counts the baseline the window replayed
    assert written["timing"]["replays_including_baselines"] == 4
    assert written["timing"]["replays_after_baselines"] == 3
    assert written["capture"]["B"]["usable_steps"] == ["1.0000", "1.2500", "3.0000"]
    # retention carries the digests of the bytes consumed, as the producer's own keys
    item = saved["B_item7_pilot.json"]
    assert item["case_sha256"] == "tracker_case.mat-digest"
    assert item["reference_output_sha256"] == "tracker_octave_instrumented.mat-digest"
    assert item["extra"]["inputs_sha256"]["axes_capture.log"] == "axes_capture.log-digest"


def test_a_capture_between_timesteps_is_never_a_control_and_never_a_candidate(
        tmp_path, monkeypatch):
    """A review put a capture at 3.1, between two timesteps: the screen read the nearest
    field and passed it, the replay's edit never fired, and the item was credited."""
    S, replays, _ = _stage(monkeypatch, tmp_path, [1.0, 3.1], reproduce_at=(1.0,),
                           acting=(1.0, 3.1))
    code, written = _run(S, tmp_path)
    r = written["results"][0]
    assert r["outcome"] == "UNSUPPORTED" and "no negative control" in r["why"]
    assert r["controls"]["negative"]["eligible_steps"] == []


def test_an_edit_that_fires_other_than_once_is_anomalous(tmp_path, monkeypatch):
    S, _, _ = _stage(monkeypatch, tmp_path, [1.0, 3.0], reproduce_at=(1.0,),
                     acting=(1.0, 3.0), fire_twice_at=(3.0,))
    code, written = _run(S, tmp_path)
    r = written["results"][0]
    assert r["outcome"] == "ANOMALOUS" and "fired 2 times" in r["why"]
    assert r["achieved_grade"] == "none"


def test_a_step_whose_substitution_leaves_detection_unchanged_is_not_replayed(
        tmp_path, monkeypatch):
    S, replays, _ = _stage(monkeypatch, tmp_path, [1.0, 1.25, 3.0], reproduce_at=(1.0,),
                           acting=(3.0,))
    code, written = _run(S, tmp_path)
    r = written["results"][0]
    assert r["outcome"] == "UNEXPLAINED"
    assert r["screened_out"] == [1.0, 1.25] and r["steps_tried"] == []
    assert not any(sub for _, sub in replays)


def test_without_a_capture_at_any_candidate_step_or_without_a_negative_step_it_is_unsupported(
        tmp_path, monkeypatch):
    S, _, _ = _stage(monkeypatch, tmp_path, [3.0], acting=(3.0,))
    code, written = _run(S, tmp_path, "a")
    assert written["results"][0]["outcome"] == "UNSUPPORTED"
    assert "no candidate step" in written["results"][0]["why"]
    S, _, _ = _stage(monkeypatch, tmp_path, [1.0, 1.25], reproduce_at=(1.0,), acting=(1.0,))
    code, written = _run(S, tmp_path, "b")
    assert written["results"][0]["outcome"] == "UNSUPPORTED"
    assert "no negative control" in written["results"][0]["why"]
    S, _, _ = _stage(monkeypatch, tmp_path, [1.0, 1.75], reproduce_at=(1.0,), acting=(1.0, 1.75))
    code, written = _run(S, tmp_path, "c")
    assert written["results"][0]["outcome"] == "UNSUPPORTED"


def test_a_negative_control_that_also_reproduces_the_reference_withholds_credit(
        tmp_path, monkeypatch):
    S, _, _ = _stage(monkeypatch, tmp_path, [1.0, 3.0], reproduce_at=(1.0, 3.0), acting=(1.0, 3.0))
    code, written = _run(S, tmp_path)
    r = written["results"][0]
    assert r["outcome"] == "UNEXPLAINED" and "not specific" in r["why"]
    assert r["controls"]["negative"]["reproduces_reference"] is True
    assert r["achieved_grade"] == "none"


def test_a_raw_capture_time_off_the_grid_is_refused_before_it_can_become_a_control(
        tmp_path, monkeypatch):
    """A confirmation round put a POTWV record at 3.00004: rounding to four decimals made
    it "3.0000", it resolved to the grid step 3.0, and it became the negative control."""
    S = _load()
    times = np.arange(1.0, 4.0, 0.25)
    text = _capture_text([1.0, 3.0]).replace("POTWV 3 ", "POTWV 3.00004 ")
    usable, refused, _ = S.ordered_capture(text, times)
    assert "1.0000" in usable and "3.0000" not in usable
    assert any("no case timestep equals" in v for v in refused.values() if isinstance(v, str))
    # two raw times landing on one step, and a malformed record at a step, refuse that step
    text = _capture_text([1.0, 3.0]) + "AXISPTS 3.0000 2 9 20\n"
    usable, refused, _ = S.ordered_capture(text, times)
    assert "3.0000" not in usable and "unreadable AXISPTS" in refused["3.0000"]
    # a record whose time cannot be read refuses the whole capture
    usable, refused, _ = S.ordered_capture(_capture_text([1.0]) + "POTWV x 1 1 2 3\n", times)
    assert usable == {} and "_capture" in refused
    # and through the command path such a capture yields no control and no credit
    S2, _, _ = _stage(monkeypatch, tmp_path, [1.0, 3.0], reproduce_at=(1.0,), acting=(1.0, 3.0))
    monkeypatch.setattr(S2, "load_window_snapshot", lambda work, name: (
        str(tmp_path), {"time": times}, {"pairs": [{"v1_index": 7, "port_index": 0,
                                                    "extra_kind": "extra_v1_only"}], "v1_unmatched": []},
        {}, {"time0": np.array([1.25, 1.5]), "lat0": np.array([10.6, 11.0]), "lon0": np.array([-20.5, -21.0])},
        _capture_text([1.0, 3.0]).replace("POTWV 3 ", "POTWV 3.00004 ") + "AXISPTS 1.0000 2 9 20\n",
        {name: "d" for name in S2.INPUT_NAMES}))
    code, written = _run(S2, tmp_path, "offgrid")
    assert written["results"][0]["outcome"] == "UNSUPPORTED"
    assert written["capture"]["B"]["usable_steps"] == []


def test_the_capture_is_validated_and_its_order_checked_at_a_tight_tolerance():
    S = _load()
    usable, refused, deviation = S.ordered_capture(_capture_text([5.0, 6.0], reversed_at=(6.0,)))
    assert sorted(usable) == ["5.0000"] and deviation["5.0000"] == 0.0
    assert "6.0000" in refused and "differs from its POTWV center" in refused["6.0000"]
    # a near-swap that an absolute tolerance of 1e-3 accepted is refused at 1e-4
    text = ("AXISPTS 7.0000 1 10.0004 10.0\nAXISPTS 7.0000 1 10.0 10.0004\n"
            "POTWV 7 1 10.0 10.0 7\nPOTWV 7 2 10.0004 10.0004 7\n")
    usable, refused, _ = S.ordered_capture(text)
    assert usable == {} and "7.0000" in refused
    # a not-a-number center, a gapped index sequence, and a count mismatch are refused
    for text, why in (
            ("AXISPTS 8.0000 1 1.0 2.0\nPOTWV 8 1 nan 2.0 8\n", "not finite"),
            ("AXISPTS 9.0000 1 1.0 2.0\nPOTWV 9 2 1.0 2.0 9\n", "not 1.."),
            ("AXISPTS 10.0000 1 1.0 2.0\nAXISPTS 10.0000 1 3.0 4.0\nPOTWV 10 1 1.0 2.0 10\n",
             "AXISPTS axes and 1 POTWV")):
        usable, refused, _ = S.ordered_capture(text)
        assert usable == {}, text
        assert any(why in v for v in refused.values() if isinstance(v, str)), (text, refused)
    # a step within the rounding bound is usable and its deviation is reported
    text = "AXISPTS 11.0000 1 1.00004 2.0\nPOTWV 11 1 1.0 2.0 11\n"
    usable, refused, deviation = S.ordered_capture(text)
    assert "11.0000" in usable and 0 < deviation["11.0000"] <= 1e-4


def test_save_run_records_supplied_input_digests(tmp_path):
    for name in ("case.mat", "ref.mat", "driver.py"):
        (tmp_path / name).write_text(name)
    identity = B.X.save_run(str(tmp_path / "r.json"), {"baseline": []}, str(tmp_path / "case.mat"),
                            str(tmp_path / "ref.mat"), -1, str(tmp_path / "driver.py"),
                            case_sha256="c" * 64, reference_output_sha256="r" * 64)
    assert identity["case_sha256"] == "c" * 64 and identity["reference_output_sha256"] == "r" * 64


def test_the_merge_spy_verifies_the_first_pass_input_exactly(monkeypatch):
    S = _load()
    calls = []
    monkeypatch.setattr(B.D, "merge_contours", lambda c, *a, **k: calls.append(len(c)) or ["merged"])
    spy, seen = S.merge_spy(3.0, [(10.123456789012345, -20.5), (11.0, -21.0)])
    first = [{"time": 3.0, "lat_mean": 10.123456789012345, "lon_mean": -20.5},
             {"time": 3.0, "lat_mean": 11.0, "lon_mean": -21.0}]
    assert spy(first, "grid") == ["merged"]
    # the second (fine) pass carries regions and is not the boundary
    spy([{"time": 3.0, "lat_mean": 10.1, "lon_mean": -20.5, "region": None}], "grid")
    spy([{"time": 3.25, "lat_mean": 1.0, "lon_mean": 2.0}], "grid")
    assert seen == {"calls_at_step": 1, "received": 2, "exact": True}
    spy2, seen2 = S.merge_spy(3.0, [(10.1234, -20.5), (11.0, -21.0)])
    spy2(first, "grid")
    assert seen2["exact"] is False


def test_ordered_centers_are_the_full_precision_potwv_values_in_order():
    S = _load()
    text = ("AXISPTS 5.0000 1 1.0 2.0\nAXISPTS 5.0000 1 3.0 4.0\n"
            "POTWV 5 1 1.00001234567891 2.0 5\nPOTWV 5 2 3.0 4.00009 5\n")
    centers, refused = S.ordered_centers(text, np.array([4.75, 5.0, 5.25]))
    assert centers == {"5.0000": [(1.00001234567891, 2.0), (3.0, 4.00009)]}
    axes = S.as_single_point_axes(centers["5.0000"])
    assert float(np.mean(axes[0][0])) == 1.00001234567891 and axes[1][1][0] == 4.00009


def test_the_centers_operation_uses_the_ports_own_centers_as_its_representation_control(
        tmp_path, monkeypatch):
    """Through the command path with the tracker faked: the intervention's edit reaches a
    stubbed merge exactly, the representation control is the port's own centers as
    single-point axes (must equal the baseline), and the record names the operation."""
    S, replays, saved = _stage(monkeypatch, tmp_path, [1.0, 3.0], reproduce_at=(1.0,),
                               acting=(1.0, 3.0))
    monkeypatch.setattr(B.D, "merge_contours", lambda c, *a, **k: c)
    monkeypatch.setattr(B, "axes_at", lambda case, when: [(np.array([9.0, 11.0]), np.array([20.0, 22.0]))])
    port_axes = [(np.array([9.0, 11.0]), np.array([20.0, 22.0]))]

    def replay_calling_merge(case, hook=None):
        # drives the hook ONCE per timestep, as the pipeline would, feeds whatever the edit
        # produced to the merge (so the spy sees it), and answers by what fired where
        B.COST["replay_calls"] += 1
        if hook is None:
            return BASE
        now, fired = [None], []
        spy = hook(lambda *a, **k: list(port_axes), now)
        for t in case["time"]:
            now[0] = float(t)
            out = spy(None, None, None)
            B.D.merge_contours([{"time": float(t), "lat_mean": float(np.mean(a)),
                                 "lon_mean": float(np.mean(b))} for a, b in out], "grid")
            if [(list(a), list(b)) for a, b in out] != [(list(a), list(b)) for a, b in port_axes]:
                fired.append((float(t), out))
        if not fired:
            return BASE
        step, out = fired[0]
        # the port's own centers as single points at the step is the representation control
        if [(float(a[0]), float(b[0])) for a, b in out] == [(10.0, 21.0)]:
            return BASE
        return EXACT if step == 1.0 else OTHER
    monkeypatch.setattr(B, "replay", replay_calling_merge)
    out = tmp_path / "centers.json"
    code = S.main([str(tmp_path), "--items", "B:7", "--out", str(out), "--retain",
                   str(tmp_path / "keep_c"), "--operation", "centers"])
    assert code == 0
    written = json.loads(out.read_text())
    r = written["results"][0]
    assert r["mode"] == "centers" and r["operation"].startswith("substitute version 1's captured merge-input centers")
    assert r["outcome"] == "EXPLAINED", r
    assert r["merge_boundary"]["1.0000"] == {"calls_at_step": 1, "received": 2, "exact": True}
    assert r["controls"]["representation"]["operation"].startswith("the port's own centers")
    assert r["controls"]["representation"]["passed"] is True
    assert "single-point axes" in written["what_is_substituted"]
    assert saved["B_item7_pilot.json"]["extra"]["merge_boundary"]["1.0000"]["exact"] is True
