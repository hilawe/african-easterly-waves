"""The Sahara case's mechanism, bound on the synthetic field that isolates it, and the
trace's evidence contract, bound on fake outputs and logs.

MUTATION LIST, written before the assertions:
  T1 missing cells are filled with zero before contouring, so the lone column yields
     axes (and every masked boundary becomes a zero contour);
  T2 the two-column control yields no axes (the finder is broken outright, and the
     lone-column result would be a false negative);
  T3 the trace's box test admits a candidate outside the box.
Added after a review ran the trace with an EMPTY reference log and watched it write
its conclusion anyway:
  T4 an empty reference log is accepted;
  T5 a log whose dumped timesteps differ from the producer record's dump list is
     accepted;
  T6 a log whose returned track count differs from the output's is accepted;
  T7 the conclusion states the divergence without version 1's axes at that step;
  T8 the conclusion says every western fragment was pruned when one was not.
Added after a review fed the trace outputs holding only a case id, a count and a bare
producer object, with the genuine field and log, and watched it conclude:
  T11 the producer contract is not applied (a record without source hashes verifies);
  T12 finished-track arrays are not required behind the declared count;
  T13 the log is not bound to the output (an unrelated log with the same schedule and
      count is accepted);
  T14 the replayed port modules are not fingerprinted into the artifact.
Added for the third case, where both sides draw axes and partition the same vertices
differently:
  T9 the vertex comparison ignores the box, so vertices outside it count;
  T10 the "same vertices, different partition" statement is made when a vertex is
      one side's only;
  T15 the statement asserts that the candidates differ by partition (a review found
      identical candidates over differently partitioned vertices, and the South
      Atlantic cause in a later stage).
Added for the fourth case, the first walked under the rule that a stage difference is
not a mechanism until an intervention at that stage changes the output and one
elsewhere does not:
  T16 the injected axes are added at every timestep rather than at the named one, so
      the control reproduces the case as well as the intervention does;
  T17 the injection is dropped and the intervened replay is the untouched one;
  T18 reference-track equality is loosened, so a track that differs reads as an exact
      reproduction;
  T19 the intervention statement is made when a replay's comparison is absent;
  T20 the intervention statement is made when no intervention ran;
  T21 the zero crossing is put at the midpoint of the edge instead of interpolated;
  T22 the coincidence tolerance is widened past the log's four-decimal printing, so a
      vertex that is not on a crossing counts as on one;
  T23 the "same field, different tracer" statement is made when a vertex lies on no
      crossing of the port's own field;
  T24 the interior-change row yields a port axis (the shape the fourth case rests on is
      not the shape that was isolated).
Added after a review defeated the first version of the fourth case, building a second
masked field on which Octave draws the same vertices while the port's tracer draws a
line, passing a control step outside the window and watching an injection that never
happened be reported as one with no effect, and finding two mutations the catalogue had
not named:
  T25 the same-field verdict is made without comparing the two fields (the vertices are
      taken as the evidence again);
  T26 a cell one side masked and the other did not still counts as one field;
  T27 the field comparison is read from the wrong timestep's records;
  T28 the per-coordinate printing bound is widened back to a Euclidean 5e-4;
  T29 every vertical zero crossing is dropped (both the fixture and the fourth case's
      own field change sign along a row, so only a column fixture binds it);
  T30 the injection is dropped at `intervene`'s call site, which destroys both
      experiments while leaving the hook itself intact;
  T31 a control step outside the window, or equal to the divergence step, is accepted.
"""
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import pytest
from scipy.io import savemat

HERE = os.path.dirname(os.path.abspath(__file__))


def _load():
    sys.path.insert(0, os.path.join(HERE, "..", "src"))
    spec = importlib.util.spec_from_file_location(
        "trace_sahara_case", os.path.join(HERE, "..", "scripts", "trace_sahara_case.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["trace_sahara_case"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_a_lone_masked_bounded_column_yields_no_port_axis_and_two_columns_do():
    M = _load()
    r = M.port_synthetic()
    assert r["lone_column_axes"] == []
    assert len(r["two_column_axes"]) == 2
    assert all(a["n"] >= 2 for a in r["two_column_axes"])
    # the eastern Pacific shape, one finite row with the sign change on an edge whose
    # neighbors above and below are masked, draws nothing either
    assert r["lone_row_axes"] == []
    rlat, rlon, rf = M.lone_row_field()
    assert np.isfinite(rf).sum() == 3 and np.isfinite(rf[1, :]).sum() == 3
    assert np.sign(rf[1, 1]) > 0 and np.sign(rf[1, 2]) > 0 and np.sign(rf[1, 3]) < 0
    lat, lon, f, g = M.lone_column_field()
    assert np.isfinite(f).sum() == 3 and np.isfinite(f[:, 2]).sum() == 3
    assert np.sign(f[1, 2]) > 0 and np.sign(f[2, 2]) < 0 and np.sign(f[3, 2]) > 0
    assert np.isfinite(g).sum() == 6


def test_box_membership_is_the_declared_box():
    M = _load()
    assert M.in_box(18.0, -9.0) is True
    assert M.in_box(18.0, -2.0) is False and M.in_box(25.0, -9.0) is False


DUMPS = [33024.0, 33024.5, 33024.75, 33025.0, 33025.5]
RUNNER = os.path.join(HERE, "..", "scripts", "octave", "run_tracker_instrumented.m")
PIPELINE = os.path.join(HERE, "..", "src", "aew", "v1port", "pipeline.py")
EXPORTER = os.path.join(HERE, "..", "scripts", "export_tracker_case.py")
TRACKS = [([0, 1, 2], [10, 10, 10], [-20, -21, -22]), ([3, 4], [5, 5], [30, 31])]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def write_tracks(path, tracks, case_id, extra=None):
    payload = {"n": float(len(tracks)), "case_id": case_id}
    for i, (t, la, lo) in enumerate(tracks):
        payload[f"time{i}"] = np.asarray(t, dtype=float)
        payload[f"lat{i}"] = np.asarray(la, dtype=float)
        payload[f"lon{i}"] = np.asarray(lo, dtype=float)
    payload.update(extra or {})
    savemat(path, payload)


def fake_log(path, producer_text, times=DUMPS, returned=2, records=True, run_digest=None):
    digest = run_digest or hashlib.sha256(producer_text.encode()).hexdigest()
    lines = ["dumping intermediates at %d timesteps" % len(times), f"RUN {digest}"]
    if records:
        for t in times:
            lines.append(f"COARSE {t:.4f} 18.0000 -9.0000")
            lines.append(f"AXIS {t:.4f} 3 18.4236 -9.0000 0.0000 0.0000")
    if returned is not None:
        lines.append(f"find_ews_f returned {returned} tracks in 449 s")
    lines.append("saved")
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture()
def exchange(tmp_path):
    """Complete evidence: both outputs carry finished-track arrays and producer records
    that verify under the comparison's full contract (genuine digests of this
    checkout's files), and the log carries the RUN digest of the oracle's record."""
    d = tmp_path / "oracle"
    (d / "v1_instrumented").mkdir(parents=True)
    src = d / "v1_instrumented" / "find_ews_f.m"
    src.write_text("function ews = find_ews_f()\n% repaired: convhull(x,y)\n")
    savemat(str(d / "tracker_case.mat"), {"case_id": "abc123"})
    oracle_rec = {"producer": "scripts/octave/run_tracker_instrumented.m",
                  "case_id": "abc123", "instrumented": True, "repaired_convhull": True,
                  "unrepaired_convhull_sites": 0, "octave_version": "11.3.0",
                  "dump_times": DUMPS, "runner_sha256": sha(RUNNER),
                  "executed_source_sha256": {"v1_instrumented__find_ews_f_m": sha(str(src))}}
    oracle_text = json.dumps(oracle_rec)
    write_tracks(str(d / "tracker_octave_instrumented.mat"), TRACKS, "abc123",
                 {"producer_json": oracle_text})
    port_rec = {"producer": "scripts/export_tracker_case.py", "case_id": "abc123",
                "git_head": "f" * 40, "git_dirty": False,
                "source_sha256": {"src/aew/v1port/pipeline.py": sha(PIPELINE),
                                  "scripts/export_tracker_case.py": sha(EXPORTER)},
                "settings": {"exclusive": False, "absorb": False}}
    write_tracks(str(d / "tracker_port.mat"), TRACKS, "abc123",
                 {"exclusive": 0.0, "absorb": 0.0, "producer_json": json.dumps(port_rec)})
    fake_log(d / "octave3.log", oracle_text)
    return d


def oracle_text_of(d):
    from scipy.io import loadmat
    return str(np.asarray(loadmat(str(d / "tracker_octave_instrumented.mat"),
                                  variable_names=["producer_json"])["producer_json"]).ravel()[0])


def test_complete_evidence_validates(exchange):
    M = _load()
    problems, prov = M.validate_evidence(str(exchange), "abc123", str(exchange / "octave3.log"))
    assert problems == []
    assert prov["oracle"]["status"] == "verified" and prov["oracle"]["faithful"] is True
    assert prov["port"]["status"] == "verified"


def _strip_oracle_sources(d):
    rec = json.loads(oracle_text_of(d))
    rec["executed_source_sha256"] = {}
    text = json.dumps(rec)
    write_tracks(str(d / "tracker_octave_instrumented.mat"), TRACKS, "abc123",
                 {"producer_json": text})
    fake_log(d / "octave3.log", text)


def _strip_port_record(d):
    write_tracks(str(d / "tracker_port.mat"), TRACKS, "abc123",
                 {"exclusive": 0.0, "absorb": 0.0,
                  "producer_json": json.dumps({"case_id": "abc123"})})


def _change_source(d):
    (d / "v1_instrumented" / "find_ews_f.m").write_text("changed since the run\n")


def _no_arrays(d):
    savemat(str(d / "tracker_octave_instrumented.mat"),
            {"case_id": "abc123", "n": 2.0, "producer_json": oracle_text_of(d)})


def _drop_run_line(d):
    fake_log(d / "octave3.log", oracle_text_of(d))
    lines = [l for l in (d / "octave3.log").read_text().splitlines() if not l.startswith("RUN ")]
    (d / "octave3.log").write_text("\n".join(lines) + "\n")


def _unrelated_log(d):
    fake_log(d / "octave3.log", "another run's record")     # same schedule and count


@pytest.mark.parametrize("label, mutate, reason", [
    ("empty log", lambda d: fake_log(d / "octave3.log", oracle_text_of(d), records=False),
     "holds no dumped records"),
    ("dump times differ", lambda d: fake_log(d / "octave3.log", oracle_text_of(d),
                                             times=DUMPS[:-1]),
     "not the producer record's dump list"),
    ("returned count differs", lambda d: fake_log(d / "octave3.log", oracle_text_of(d),
                                                  returned=99),
     "reports 99 tracks but the output holds 2"),
    ("another case", lambda d: savemat(str(d / "tracker_case.mat"), {"case_id": "other"}),
     "do not carry the exported case id"),
    ("oracle record without source hashes", _strip_oracle_sources,
     "oracle provenance invalid"),
    ("port record with the case id alone", _strip_port_record, "port provenance invalid"),
    ("source changed beside the output", _change_source, "oracle provenance inconsistent"),
    ("declared count without arrays", _no_arrays, "does not hold complete finished-track"),
    ("unrelated log with matching schedule and count", _unrelated_log,
     "RUN digest is not the digest of the oracle output's producer record"),
    ("log without a RUN line", lambda d: _drop_run_line(d), "carries no RUN digest"),
])
def test_incomplete_or_unrelated_evidence_is_refused_before_any_replay(
        exchange, tmp_path, capsys, label, mutate, reason):
    M = _load()
    mutate(exchange)
    out = tmp_path / "trace.json"
    # the case file carries no fields, so a replay would crash: the refusal must come first
    assert M.main(["--oracle-dir", str(exchange), "--out", str(out)]) == 2, label
    assert not out.exists(), label
    text = capsys.readouterr().out
    assert "REFUSED" in text and reason in text, (label, text)


def test_conclusion_is_derived_from_the_recorded_observations():
    M = _load()
    v1 = ([{"kind": "COARSE", "time": t, "lat_mean": 18.0, "lon_mean": -9.0}
           for t in M.AGREEING_STEPS]
          + [{"kind": "AXIS", "time": M.DIVERGENCE_STEP, "n_points": 3, "lat_mean": 18.4,
              "lon_mean": -9.0, "lat_range": 0.0, "lon_range": 0.0},
             {"kind": "COARSE", "time": M.DIVERGENCE_STEP, "lat_mean": 18.0, "lon_mean": -9.0}])
    log = [{"time": t, "lat_mean": 18.5, "lon_mean": -8.5, "n_points": 14,
            "taken_by_tracks_of_length": [k + 1]} for k, t in enumerate(M.AGREEING_STEPS)]
    v1_at = [r for r in v1 if r["time"] == M.DIVERGENCE_STEP]
    divergence = {"finite_cells": 4, "port_axes_in_box": [], "port_waves_in_box": []}
    western = [{"observations": [[33024.0, 19.0, -5.0], [33024.5, 18.75, -8.75],
                                 [33024.75, 18.5, -8.5]], "pruned_at": 33026.75},
               {"observations": [[33025.5, 18.0, -11.5]] + [[33026.0 + 0.25 * k, 21.0, -2.0]
                                                            for k in range(5)],
                "pruned_at": 33028.75}]
    statements, missing = M.derive_conclusion(log, v1, v1_at, [], divergence, [], western)
    assert missing == []
    assert any("both sides detect a candidate" in s for s in statements)
    assert any("version 1 dumps 1 axes and 1 coarse candidate" in s and "port has no "
               "candidate and no axis" in s for s in statements)
    assert any("3 observations from 33024.0 to 33024.75, pruned at 33026.75" in s
               for s in statements)
    assert any("every port track that entered the western box was removed" in s
               for s in statements)
    # without version 1's axes at the divergence step there is no divergence statement
    statements2, missing2 = M.derive_conclusion(log, [r for r in v1 if r["kind"] != "AXIS"],
                                                [], [], divergence, [], western)
    assert any("version 1 axes and coarse candidate" in m for m in missing2)
    assert not any("port has no candidate and no axis" in s for s in statements2)
    # a western fragment that survives forbids the "every ... removed" statement
    kept = [dict(western[0]), dict(western[1], pruned_at=None)]
    statements3, _ = M.derive_conclusion(log, v1, v1_at, [], divergence, [], kept)
    assert not any("every port track that entered the western box was removed" in s
                   for s in statements3)
    assert any("never pruned" in s for s in statements3)


def test_vertex_comparison_states_shared_vertices_and_different_partition():
    M = _load()
    # the same six vertices: version 1 joins A+B (4) and leaves C (2); the port joins
    # A+C (4) and leaves B (2); one vertex far outside the box on each side must not count
    A = [(-24.0, -18.0), (-22.0, -21.0)]
    B = [(-18.0, -19.4), (-16.0, -14.4)]
    C = [(-16.0, -31.3), (-14.2, -29.0)]
    far = (10.0, 60.0)
    v1 = [{"kind": "AXISPTS", "time": M.DIVERGENCE_STEP, "n_points": 5,
           "lat": [p[0] for p in A + B] + [far[0]], "lon": [p[1] for p in A + B] + [far[1]]},
          {"kind": "AXISPTS", "time": M.DIVERGENCE_STEP, "n_points": 2,
           "lat": [p[0] for p in C], "lon": [p[1] for p in C]}]
    port = [{"n": 5, "lat_mean": 0.0, "lon_mean": 0.0,
             "lat": [p[0] for p in A + C] + [far[0]], "lon": [p[1] for p in A + C] + [far[1]]},
            {"n": 2, "lat_mean": 0.0, "lon_mean": 0.0,
             "lat": [p[0] for p in B], "lon": [p[1] for p in B]}]
    M.BOX = {"lat": (-30.0, -14.0), "lon": (-36.0, -13.0)}
    cmp = M.compare_vertices(v1, port)
    assert cmp["v1_vertices"] == 6 and cmp["port_vertices"] == 6
    assert cmp["shared_vertices"] == 6 and cmp["v1_only_vertices"] == 0
    assert cmp["port_only_vertices"] == 0
    assert cmp["v1_lines"] == 2 and cmp["port_lines"] == 2
    # the conclusion path: agreeing steps satisfied, port axes present, no port wave
    v1_dumps = ([{"kind": "COARSE", "time": t, "lat_mean": -22.0, "lon_mean": -23.0}
                 for t in M.AGREEING_STEPS]
                + [{"kind": "AXIS", "time": M.DIVERGENCE_STEP, "n_points": 5, "lat_mean": -20.0,
                    "lon_mean": -18.0, "lat_range": 8.0, "lon_range": 7.0},
                   {"kind": "COARSE", "time": M.DIVERGENCE_STEP, "lat_mean": -20.0,
                    "lon_mean": -18.0}] + v1)
    log = [{"time": t, "lat_mean": -26.0, "lon_mean": -17.0, "n_points": 100,
            "taken_by_tracks_of_length": [9]} for t in M.AGREEING_STEPS]
    v1_at = [r for r in v1_dumps if r["time"] == M.DIVERGENCE_STEP]
    divergence = {"finite_cells": 30, "port_axes_in_box": port, "port_waves_in_box": []}
    western = [{"observations": [[33035.0, -24.9, -21.7]], "pruned_at": None}]
    statements, missing = M.derive_conclusion(log, v1_dumps, v1_at, [], divergence,
                                              [{"steps": 1}], western)
    assert missing == []
    assert any("both sides draw axes in the box" in s and "6 distinct vertices are shared"
               in s for s in statements)
    assert any("join them into different lines" in s for s in statements)
    # the statement is descriptive: it never says the candidates differ because of it
    assert not any("partition alone" in s or "by partition" in s for s in statements)
    assert any("not established by this comparison" in s for s in statements)
    # a vertex one side alone holds forbids the shared-vertex statement
    port2 = [dict(port[0]), dict(port[1], lat=port[1]["lat"] + [-15.0],
                                lon=port[1]["lon"] + [-20.0])]
    divergence2 = dict(divergence, port_axes_in_box=port2)
    statements2, _ = M.derive_conclusion(log, v1_dumps, v1_at, [], divergence2,
                                         [{"steps": 1}], western)
    assert not any("join them into different lines" in s for s in statements2)
    assert any("1 the port's only" in s for s in statements2)


# ---------------------------------------------------------------------------
# The intervention, its control, and the field the two sides share (fourth case)


def test_the_injected_axes_are_added_at_the_named_step_and_nowhere_else():
    M = _load()
    drawn = [(np.array([1.0, 2.0]), np.array([10.0, 11.0]))]
    inject = {"time": 33035.25, "axes": [{"lat": [-32.0, -32.0], "lon": [-60.3167, -60.3167]}]}
    now, captured = [None], {}
    spy = M.axis_spy(lambda *a, **k: list(drawn), inject, now, captured)
    now[0] = 33035.25
    at_step = spy(None, None, np.zeros((2, 2)))
    assert len(at_step) == 2
    assert list(at_step[1][0]) == [-32.0, -32.0] and list(at_step[1][1]) == [-60.3167] * 2
    now[0] = 33035.5
    assert len(spy(None, None, np.zeros((2, 2)))) == 1
    # and with nothing to inject the hook is the port's own axis finder
    plain = M.axis_spy(lambda *a, **k: list(drawn), None, [33035.25], {})
    assert len(plain(None, None, np.zeros((2, 2)))) == 1


def _track(times, lats, lons):
    return {"time": list(times), "meanlat": list(lats), "meanlon": list(lons)}


def _reference(times, lats, lons):
    return {"time": np.asarray(times, float), "lat": np.asarray(lats, float),
            "lon": np.asarray(lons, float)}


def test_a_reference_track_counts_as_reproduced_only_when_it_is_equal_step_for_step():
    M = _load()
    ref = _reference([33035.0, 33035.25], [-29.3, -28.95], [-61.4, -60.3])
    exact = _track([33035.0, 33035.25], [-29.3, -28.95], [-61.4, -60.3])
    near = _track([33035.0, 33035.25], [-29.3, -28.95], [-61.4, -60.31])
    short = _track([33035.0], [-29.3], [-61.4])
    got = M.reproduction([exact, near], [(81, ref)])
    assert got[0]["reproduced_exactly"] is True and got[0]["reference_steps"] == 2
    assert got[0]["nearest_worst_step_deg"] == 0.0
    missed = M.reproduction([near, short], [(81, ref)])
    assert missed[0]["reproduced_exactly"] is False
    # the nearest is the one differing by a hundredth of a degree in longitude, and the
    # one-step fragment, which coincides where it does overlap, is not counted at all
    assert abs(missed[0]["nearest_worst_step_deg"] - 0.01) < 1e-9
    assert missed[0]["port_tracks_covering_its_steps"] == 1
    fragment_only = M.reproduction([short], [(81, ref)])
    assert fragment_only[0]["nearest_worst_step_deg"] is None
    assert fragment_only[0]["port_tracks_covering_its_steps"] == 0
    assert fragment_only[0]["nearest_port_track_steps"] is None
    # a LONGER track can sit at zero separation on the reference's own steps while
    # holding observations the reference does not, so its own length is recorded beside
    # the distance and the exact-equality verdict stays false
    longer = _track([33035.0, 33035.25, 33035.5], [-29.3, -28.95, -29.2],
                    [-61.4, -60.3, -59.0])
    over = M.reproduction([longer], [(81, ref)])
    assert over[0]["nearest_worst_step_deg"] == 0.0
    assert over[0]["nearest_port_track_steps"] == 3 and over[0]["reference_steps"] == 2
    assert over[0]["reproduced_exactly"] is False
    # a port track that shares no timestep with the reference is no reproduction at all
    elsewhere = M.reproduction([_track([33040.0], [0.0], [0.0])], [(81, ref)])
    assert elsewhere[0]["reproduced_exactly"] is False
    assert elsewhere[0]["nearest_worst_step_deg"] is None


def test_observations_at_reads_one_timestep_inside_the_box():
    M = _load()
    M.BOX = {"lat": (-35.0, -26.0), "lon": (-67.0, -57.0)}
    tracks = [_track([33035.0, 33035.25], [-29.3, -28.95], [-61.4, -60.3]),
              _track([33035.25], [-28.95], [-60.3]),          # the same position, one copy
              _track([33035.25], [-10.0], [-60.3]),           # outside the box
              _track([33035.5], [-29.2], [-59.0])]            # another timestep
    assert M.observations_at(tracks, 33035.25) == [(-28.95, -60.3)]


def _runs(baseline=0, intervention=2, control=0, n=2, control_at=33035.5,
          applied=(1, 1), control_problems=(), shape=0):
    def refs(hits):
        return [{"reference_index": i, "reference_steps": 8, "reproduced_exactly": i < hits,
                 "nearest_worst_step_deg": 0.0 if i < hits else 0.78} for i in range(n)]
    return {"injected_axes": [{"lat": [-32.0], "lon": [-60.3167]}],
            "control_step_problems": list(control_problems),
            "baseline": {"injected_at": None, "injections_applied": 0,
                         "reference_tracks": refs(baseline),
                         "observations_at_divergence_in_box": [],
                         "finished_tracks_in_the_western_box": []},
            "intervention": {"injected_at": 33035.25, "injections_applied": applied[0],
                             "applied_at": [33035.25] * applied[0],
                             "reference_tracks": refs(intervention),
                             "observations_at_divergence_in_box": [(-28.95, -60.3)],
                             "finished_tracks_in_the_western_box": [{"steps": 13},
                                                                    {"steps": 8}]},
            "control": {"injected_at": control_at, "injections_applied": applied[1],
                        "applied_at": [control_at] * applied[1],
                        "reference_tracks": refs(control),
                        "observations_at_divergence_in_box": [],
                        "finished_tracks_in_the_western_box": []},
            "shape_control": {"injected_at": 33035.25, "injections_applied": 1,
                              "applied_at": [33035.25], "longitude_offset_deg": 4.0,
                              "axes": [{"lat": [-32.0], "lon": [-64.3167]}],
                              "reference_tracks": refs(shape),
                              "observations_at_divergence_in_box": [],
                              "finished_tracks_in_the_western_box": []}}


def test_the_intervention_statement_reports_all_three_replays():
    M = _load()
    M.DIVERGENCE_STEP = 33035.25
    statements, missing = M.intervention_statements(_runs())
    assert missing == []
    assert any("at 33035.25 alone reproduces 2 of 2 reference tracks exactly, against 0 "
               "of 2 with the port untouched" in s for s in statements)
    assert any("injected at 33035.5 instead reproduce 0 of 2" in s
               and "moved 4.0 degrees west of the crossing at 33035.25 reproduces 0 of 2" in s
               for s in statements)
    # A CONTROL IN TIME IS NOT A CONTROL IN SHAPE. The vertices moved off the crossing at
    # the SAME step are what answer "these vertices" rather than "an axis here".
    shaped, _ = M.intervention_statements(_runs(shape=2))
    assert any("moved 4.0 degrees west of the crossing at 33035.25 reproduces 2 of 2" in s
               for s in shaped)
    same_axes = _runs()
    same_axes["shape_control"]["axes"] = same_axes["injected_axes"]
    said6, why9 = M.intervention_statements(same_axes)
    assert said6 == [] and any("vertices are the intervention's own" in m for m in why9)
    moved_step = _runs()
    moved_step["shape_control"]["injected_at"] = 33035.5
    moved_step["shape_control"]["applied_at"] = [33035.5]
    said7, why10 = M.intervention_statements(moved_step)
    assert said7 == [] and any("shape_control replay injected at 33035.5" in m
                               for m in why10)
    assert any("untouched []" in s and "with the injection at 33035.25 [(-28.95, -60.3)]"
               in s for s in statements)
    # the TRACK-level effect beside the observation-level one, which the Sahara case
    # separates: there the injection restores the observation and no finished track
    assert any("finished port tracks in the western box: untouched 0, with the injection "
               "at 33035.25 2, with the time control 0, with the shape control 0" in s
               for s in statements)
    # a control that reproduces the case as well as the intervention is reported as such
    both, _ = M.intervention_statements(_runs(control=2))
    assert any("injected at 33035.5 instead reproduce 2 of 2" in s for s in both)
    # an intervention that was not run says nothing at all
    assert M.intervention_statements(None) == ([], [])
    # and a replay whose comparison is absent is a missing observation, not a silent pass
    incomplete = _runs()
    incomplete["control"]["reference_tracks"] = []
    statements2, missing2 = M.intervention_statements(incomplete)
    assert statements2 == [] and missing2 == ["the control replay's reference-track comparison"]
    # A REPLAY THAT INJECTED NOTHING IS NOT A NEGATIVE RESULT. A review passed a control
    # time outside the window, nothing was injected, and the untouched replay was
    # reported as an injection that reproduced nothing.
    never, why = M.intervention_statements(_runs(applied=(1, 0)))
    assert never == []
    assert any("control replay applied 0 of the 1 injection" in m for m in why)
    unapplied, why2 = M.intervention_statements(_runs(applied=(0, 1)))
    assert unapplied == []
    assert any("intervention replay applied 0 of the 1 injection" in m for m in why2)
    # and a control step the replay should never have been given stops the statement too
    bad, why3 = M.intervention_statements(_runs(control_problems=["the control step 99999.0 "
                                                                 "is not a timestep of the "
                                                                 "exported window"]))
    assert bad == [] and any("99999.0" in m for m in why3)
    # A RECEIPT THAT CONTRADICTS ITS REQUEST is not a negative control. A review moved the
    # control's recorded application to the divergence step and watched the statement
    # still call it a control at 33035.5.
    lying = _runs()
    lying["control"]["applied_at"] = [33035.25]
    said, why4 = M.intervention_statements(lying)
    assert said == [] and any("recorded an application at [33035.25] for an injection "
                              "requested at 33035.5" in m for m in why4)
    # and an intervention that injected somewhere other than the divergence step is not
    # this case's intervention
    elsewhere = _runs()
    elsewhere["intervention"]["injected_at"] = 33036.0
    elsewhere["intervention"]["applied_at"] = [33036.0]
    said2, why5 = M.intervention_statements(elsewhere)
    assert said2 == [] and any("injected at 33036.0 and the case's divergence step is "
                               "33035.25" in m for m in why5)
    # A NOT-A-NUMBER APPLICATION TIME passes every comparison, so it is refused outright
    nan_receipt = _runs()
    nan_receipt["control"]["applied_at"] = [float("nan")]
    said3, why6 = M.intervention_statements(nan_receipt)
    assert said3 == [] and any("recorded an application at [nan]" in m for m in why6)
    # A CONTROL ON THE DIVERGENCE STEP is the intervention, whatever the problem list
    # computed when the replays ran says, and a review left that list empty.
    not_a_control = _runs(control_at=33035.25)
    not_a_control["control"]["applied_at"] = [33035.25]
    said4, why7 = M.intervention_statements(not_a_control)
    assert said4 == [] and any("time control injected at the divergence step" in m
                               for m in why7)
    # AN ABSENT WESTERN MEASUREMENT IS NOT A ZERO, which is the comparison the Sahara
    # case's claim rests on
    unmeasured = _runs()
    del unmeasured["intervention"]["finished_tracks_in_the_western_box"]
    said5, why8 = M.intervention_statements(unmeasured)
    assert said5 == []
    assert any("intervention replay's finished tracks in the western box" in m
               for m in why8)


def test_the_conclusion_carries_the_intervention_only_when_one_ran():
    M = _load()
    M.DIVERGENCE_STEP = 33035.25
    v1 = ([{"kind": "COARSE", "time": t, "lat_mean": 18.0, "lon_mean": -9.0}
           for t in M.AGREEING_STEPS]
          + [{"kind": "AXIS", "time": M.DIVERGENCE_STEP, "n_points": 3, "lat_mean": 18.4,
              "lon_mean": -9.0, "lat_range": 0.0, "lon_range": 0.0},
             {"kind": "COARSE", "time": M.DIVERGENCE_STEP, "lat_mean": 18.0, "lon_mean": -9.0}])
    log = [{"time": t, "lat_mean": 18.5, "lon_mean": -8.5, "n_points": 14,
            "taken_by_tracks_of_length": [1]} for t in M.AGREEING_STEPS]
    v1_at = [r for r in v1 if r["time"] == M.DIVERGENCE_STEP]
    divergence = {"finite_cells": 4, "port_axes_in_box": [], "port_waves_in_box": [],
                  "rows_lat": [20.0, 18.0], "cols_lon": [-9.0, -7.0],
                  "masked_smoothed_advection": [[None, None], [None, None]]}
    western = [{"observations": [[33024.0, 19.0, -5.0]], "pruned_at": 33026.75}]
    plain, _ = M.derive_conclusion(log, v1, v1_at, [], divergence, [], western)
    assert not any("injecting version 1" in s for s in plain)
    with_it, missing = M.derive_conclusion(log, v1, v1_at, [], divergence, [], western,
                                           _runs())
    assert missing == [] and any("injecting version 1" in s for s in with_it)


ROW_FIELD = {"rows_lat": [-30.0, -32.0, -34.0], "cols_lon": [-63.0, -61.0, -59.0, -57.0],
             "masked_smoothed_advection": [[None, None, None, None],
                                           [3.615261478456151e-10, 1.202971274689068e-10,
                                            -2.3182610628854524e-10, -1.3041326376378817e-10],
                                           [None, None, None, None]]}


def test_the_zero_crossing_is_interpolated_on_the_edge_it_names():
    M = _load()
    crossings = M.zero_crossings(ROW_FIELD)
    assert len(crossings) == 1
    c = crossings[0]
    # written out by hand: the fraction 1.202971274689068 / 3.5212323375745204 is
    # 0.3416337, so the crossing is -61 + 2 * 0.3416337 = -60.3167326
    assert abs(c["lon"] - (-60.3167326)) < 1e-5
    assert c["lat"] == -32.0
    assert c["between"] == [[-32.0, -61.0], [-32.0, -59.0]]
    # the midpoint of that edge is 60W, which is not the crossing
    assert abs(c["lon"] - (-60.0)) > 0.3


def test_a_dumped_vertex_counts_as_on_the_crossing_only_within_the_logs_precision():
    M = _load()
    crossings = M.zero_crossings(ROW_FIELD)
    on = M.vertices_on_crossings([{"lat": [-32.0], "lon": [-60.3167]}], crossings)
    assert on[0]["on_a_crossing"] is True and on[0]["distance_deg"] < 5e-5
    off = M.vertices_on_crossings([{"lat": [-32.0], "lon": [-60.0]}], crossings)
    assert off[0]["on_a_crossing"] is False and off[0]["distance_deg"] > 0.3
    # FOUR-DECIMAL PRINTING BOUNDS EACH COORDINATE BY HALF THE LAST DIGIT, which is
    # 5e-5, not the 5e-4 a first version allowed as a Euclidean distance. A review
    # displaced one coordinate by 4e-4, which no rounding explains, and watched it pass.
    assert M.VERTEX_PRINT_TOLERANCE == 5e-5
    crossing_lon = crossings[0]["lon"]
    inside = M.vertices_on_crossings([{"lat": [-32.0], "lon": [crossing_lon + 4.9e-5]}],
                                     crossings)
    outside = M.vertices_on_crossings([{"lat": [-32.0], "lon": [crossing_lon + 5.1e-5]}],
                                      crossings)
    assert inside[0]["on_a_crossing"] is True and outside[0]["on_a_crossing"] is False
    # and the bound is PER COORDINATE, so a latitude displacement of the same size fails
    # even though the Euclidean distance would have passed the old bound
    lat_off = M.vertices_on_crossings([{"lat": [-32.0 + 4e-4], "lon": [crossing_lon]}],
                                      crossings)
    assert lat_off[0]["on_a_crossing"] is False
    assert lat_off[0]["distance_deg"] < 5e-4


def test_the_same_field_statement_rests_on_the_two_fields_and_not_on_the_vertices():
    M = _load()
    M.BOX = {"lat": (-35.0, -26.0), "lon": (-67.0, -57.0)}
    M.AXIS_BOX = None
    M.AGREEING_STEPS = (33035.0,)
    M.DIVERGENCE_STEP = 33035.25
    base = ([{"kind": "COARSE", "time": 33035.0, "lat_mean": -29.0, "lon_mean": -60.5}]
            + [{"kind": "AXIS", "time": 33035.25, "n_points": 3, "lat_mean": -32.0,
                "lon_mean": -60.3167, "lat_range": 0.0, "lon_range": 0.0},
               {"kind": "COARSE", "time": 33035.25, "lat_mean": -32.0, "lon_mean": -59.0}])
    log = [{"time": 33035.0, "lat_mean": -29.0, "lon_mean": -60.5, "n_points": 145,
            "taken_by_tracks_of_length": [8]}]
    western = [{"observations": [[33035.0, -29.0, -60.5]], "pruned_at": None}]
    on_it = base + [{"kind": "AXISPTS", "time": 33035.25, "n_points": 3,
                     "lat": [-32.0] * 3, "lon": [-60.3167] * 3}]
    at = [r for r in on_it if r["time"] == 33035.25]

    def field(cmp_result, full=True):
        # ROW_FIELD carries one sign change whose quads are masked above and below, so
        # the generated geometry statement has something to report
        record = dict(ROW_FIELD, finite_cells=4, port_axes_in_box=[],
                      port_waves_in_box=[], field_comparison=cmp_result)
        if full:
            record["full_field"] = {"rows_lat": ROW_FIELD["rows_lat"],
                                    "cols_lon": ROW_FIELD["cols_lon"],
                                    "grid": ROW_FIELD["masked_smoothed_advection"]}
            crossings = M.zero_crossings(ROW_FIELD)
            record["zero_crossings"] = crossings
            record["crossing_neighborhoods"] = M.crossing_neighborhoods(
                ROW_FIELD["rows_lat"], ROW_FIELD["cols_lon"],
                ROW_FIELD["masked_smoothed_advection"], crossings, domain_complete=True)
        return record

    same = {"available": True, "cells_unmasked_on_both_sides": 1874,
            "cells_unmasked_only_in_the_port": 0, "cells_unmasked_only_in_version_1": 0,
            "worst_relative_difference": 3e-14, "same_field": True}
    statements, missing = M.derive_conclusion(log, on_it, at, [], field(same), [], western)
    assert missing == []
    assert any("unmasked in exactly the same 1874 cells" in s and "one field" in s
               for s in statements)
    # THE VERTEX CHECK IS A SECOND OBSERVATION AND NOT THE CLAIM. A review built a
    # different masked field on which the same vertices are drawn, so vertices alone
    # cannot say the fields agree.
    assert any("locates where version 1 drew and does not by itself say the two fields "
               "are the same" in s for s in statements)
    differs = dict(same, cells_unmasked_only_in_version_1=1, same_field=False)
    statements2, _ = M.derive_conclusion(log, on_it, at, [], field(differs), [], western)
    assert not any("the two sides hold one field" in s for s in statements2)
    assert any("is not established as contouring alone" in s for s in statements2)
    # a field comparison that could not be made says so, and claims nothing
    absent = {"available": False, "reason": "no FIELD records"}
    statements3, _ = M.derive_conclusion(log, on_it, at, [], field(absent), [], western)
    assert any("was not dumped, so whether the port's axes and the port's candidates "
               "differ from version 1's by contouring or by masking is not established"
               in s for s in statements3)
    assert not any("one field" in s for s in statements3)
    # AND SO IS THE CROSSING GEOMETRY, which the prose got wrong twice and is therefore
    # generated. A review deleted the statement and the suite stayed green.
    assert any("zero crossings on the port's field in the printed box" in s
               and "two or more masked corners" in s
               and "taken over the WHOLE coarse grid" in s for s in statements)
    # THE FIELD QUESTION IS REPORTED WHATEVER THE PORT HELD THERE. A review skipped the
    # statement whenever the port had a candidate at the divergence step, which is the
    # third case's situation exactly.
    with_candidate, _ = M.derive_conclusion(
        log, on_it, at,
        [{"time": 33035.25, "lat_mean": -29.0, "lon_mean": -60.0, "n_points": 9,
          "taken_by_tracks_of_length": [4]}],
        field(same), [{"steps": 3}], western)
    assert any("unmasked in exactly the same 1874 cells" in s for s in with_candidate)
    # and an off-crossing vertex is reported as such without touching the field claim
    off_it = base + [{"kind": "AXISPTS", "time": 33035.25, "n_points": 3,
                      "lat": [-32.0] * 3, "lon": [-60.3167, -60.3167, -58.0]}]
    statements4, _ = M.derive_conclusion(log, off_it,
                                         [r for r in off_it if r["time"] == 33035.25],
                                         [], field(same), [], western)
    assert any("do not all sit on zero crossings" in s for s in statements4)


def test_the_interior_change_row_draws_no_port_axis():
    M = _load()
    assert M.port_synthetic()["interior_change_row_axes"] == []
    lat, lon, f = M.interior_change_row_field()
    assert np.isfinite(f).sum() == 7
    assert np.isfinite(f[2, :]).sum() == 5 and np.isfinite(f[3, :]).sum() == 2
    # the sign change is in the interior of the row, not at its end
    signs = np.sign(f[2, :5])
    assert list(signs) == [1.0, 1.0, 1.0, -1.0, -1.0]
    assert lat[2] == -32.0 and lon[2] == -61.0 and lon[3] == -59.0


def test_a_vertical_crossing_is_found_as_well_as_a_horizontal_one():
    """A review deleted every vertical crossing and the suite stayed green, because both
    the fixture above and the fourth case's own field change sign along a row. The Sahara
    column changes sign down a COLUMN, so the vertical branch is load-bearing."""
    M = _load()
    column = {"rows_lat": [20.0, 18.0, 16.0], "cols_lon": [-11.0, -9.0, -7.0],
              "masked_smoothed_advection": [[None, 1.0e-10, None],
                                            [None, -3.0e-10, None],
                                            [None, None, None]]}
    crossings = M.zero_crossings(column)
    assert len(crossings) == 1
    c = crossings[0]
    # by hand: 20 + (18 - 20) * 1.0 / (1.0 + 3.0) = 19.5
    assert abs(c["lat"] - 19.5) < 1e-12 and c["lon"] == -9.0
    assert c["between"] == [[20.0, -9.0], [18.0, -9.0]]


def test_intervene_requests_an_injection_at_each_step_and_reports_what_was_applied():
    """The wiring, bound at the call site. A review dropped the injection argument inside
    `intervene`, which destroys both experiments, and every test still passed because
    nothing called `intervene` itself."""
    M = _load()
    M.DIVERGENCE_STEP = 33035.25
    M.BOX = {"lat": (-35.0, -26.0), "lon": (-67.0, -57.0)}
    case = {"time": np.array([33035.0, 33035.25, 33035.5])}
    calls = []
    ref = _reference([33035.25], [-28.95], [-60.3])
    hit = _track([33035.25], [-28.95], [-60.3])

    def fake_replay(c, inject=None, reference_field=None):
        calls.append(inject)
        applied = [inject["time"]] if inject else []
        at_divergence = bool(inject) and abs(inject["time"] - 33035.25) < 1e-6
        final = [hit] if at_divergence else []
        west = [{"steps": 13, "first": 33033.0, "last": 33036.25}] if at_divergence else []
        return [], None, west, [], final, applied

    M.replay_port = fake_replay
    runs = M.intervene(case, [{"lat": [-32.0] * 3, "lon": [-60.3167] * 3}],
                       [(81, ref)], 33035.5, [], [{"steps": 9}])
    assert [c["time"] for c in calls] == [33035.25, 33035.5, 33035.25]
    assert [c["axes"][0]["lon"] for c in calls[:2]] == [[-60.3167] * 3] * 2
    # the third replay is the shape control: same step, vertices moved off the crossing
    assert calls[2]["axes"][0]["lon"] == [-64.3167] * 3
    assert runs["shape_control"]["longitude_offset_deg"] == 4.0
    assert runs["intervention"]["injections_applied"] == 1
    assert runs["control"]["injections_applied"] == 1
    assert runs["control_step_problems"] == []
    # each replay's own western result travels with it, and so does the baseline's
    assert runs["intervention"]["finished_tracks_in_the_western_box"] == [
        {"steps": 13, "first": 33033.0, "last": 33036.25}]
    assert runs["control"]["finished_tracks_in_the_western_box"] == []
    assert runs["baseline"]["finished_tracks_in_the_western_box"] == [{"steps": 9}]
    assert runs["intervention"]["reference_tracks"][0]["reproduced_exactly"] is True
    assert runs["control"]["reference_tracks"][0]["reproduced_exactly"] is False
    assert runs["baseline"]["reference_tracks"][0]["reproduced_exactly"] is False
    statements, missing = M.intervention_statements(runs)
    assert missing == [] and any("reproduces 1 of 1" in s for s in statements)

    # AND THE COUNT IS THE REPLAY'S OWN. A review hard-coded it to one, which turns a
    # replay that injected nothing into a negative result.
    def injects_nothing(c, inject=None, reference_field=None):
        return [], None, [], [], [], []

    M.replay_port = injects_nothing
    silent = M.intervene(case, [{"lat": [-32.0] * 3, "lon": [-60.3167] * 3}],
                         [(81, ref)], 33035.5, [], [])
    assert silent["intervention"]["injections_applied"] == 0
    assert silent["control"]["injections_applied"] == 0
    assert silent["intervention"]["applied_at"] == []
    said, why = M.intervention_statements(silent)
    assert said == [] and len(why) >= 2


def test_a_control_step_outside_the_window_or_on_the_divergence_is_refused():
    M = _load()
    M.DIVERGENCE_STEP = 33035.25
    case = {"time": np.array([33035.0, 33035.25, 33035.5])}
    assert M.control_step_problems(case, 33035.5) == []
    assert any("not a timestep of the exported window" in w
               for w in M.control_step_problems(case, 99999.0))
    assert any("is the divergence step" in w
               for w in M.control_step_problems(case, 33035.25))
    assert any("not a finite time" in w for w in M.control_step_problems(case, float("nan")))
    assert any("not a finite time" in w for w in M.control_step_problems(case, None))


REFERENCE_CELLS = {(-32.0, -61.0): 1.202971274689068e-10,
                   (-32.0, -59.0): -2.3182610628854524e-10}
PORT_ROWS = [-32.0]
PORT_COLS = [-61.0, -59.0]


def test_the_two_masked_fields_are_compared_cell_by_cell():
    M = _load()
    port = [[1.202971274689068e-10, -2.3182610628854524e-10]]
    same = M.compare_fields(PORT_ROWS, PORT_COLS, port, REFERENCE_CELLS)
    assert same["same_field"] is True and same["cells_unmasked_on_both_sides"] == 2
    assert same["cells_unmasked_only_in_the_port"] == 0
    assert same["cells_unmasked_only_in_version_1"] == 0
    # A CELL ONE SIDE MASKED AND THE OTHER DID NOT is the masking difference a vertex
    # cannot show, and it must stop the same-field verdict.
    port_extra = [[1.202971274689068e-10, -2.3182610628854524e-10, 5.0e-10]]
    extra = M.compare_fields(PORT_ROWS, PORT_COLS + [-57.0], port_extra, REFERENCE_CELLS)
    assert extra["same_field"] is False and extra["cells_unmasked_only_in_the_port"] == 1
    assert extra["example_port_only_cells"] == [[-32.0, -57.0]]
    v1_extra = M.compare_fields(PORT_ROWS, PORT_COLS + [-57.0],
                                [[1.202971274689068e-10, -2.3182610628854524e-10, None]],
                                {**REFERENCE_CELLS, (-32.0, -57.0): 5.0e-10})
    assert v1_extra["same_field"] is False
    assert v1_extra["cells_unmasked_only_in_version_1"] == 1
    # and a value that differs by more than the tolerance is not the same field either
    moved = M.compare_fields(PORT_ROWS, PORT_COLS,
                             [[1.202971274689068e-10 * 1.001, -2.3182610628854524e-10]],
                             REFERENCE_CELLS)
    assert moved["same_field"] is False
    assert abs(moved["worst_relative_difference"] - 0.000999) < 1e-5
    assert moved["worst_cell"] == [-32.0, -61.0]
    assert M.FIELD_RELATIVE_TOLERANCE == 1e-9
    # a comparison over no shared cell is not agreement
    assert M.compare_fields(PORT_ROWS, PORT_COLS, [[None, None]], {})["same_field"] is False


def test_the_reference_field_is_read_from_the_logs_own_records(tmp_path):
    M = _load()
    log = tmp_path / "octave.log"
    log.write_text("FIELD 33035.2500 -32.0000 -61.0000 1.202971274689e-10\n"
                   "FIELD 33035.2500 -32.0000 -59.0000 -2.318261062885e-10\n"
                   "FIELD 33035.5000 -32.0000 -61.0000 9.000000000000e-10\n"
                   "AXIS 33035.2500 3 -32.0000 -60.3167 0.0000 0.0000\n")
    at = M.reference_field_at(str(log), 33035.25)
    assert sorted(at["cells"]) == [(-32.0, -61.0), (-32.0, -59.0)]
    assert abs(at["cells"][(-32.0, -61.0)] - 1.202971274689e-10) < 1e-22
    assert at["records"] == 2 and at["problems"] == []
    # the other timestep's cells belong to that timestep and not to this one
    assert M.reference_field_at(str(log), 33035.5)["cells"] == {(-32.0, -61.0): 9.0e-10}
    assert M.reference_field_at(str(log), 33036.0)["cells"] == {}


def test_contradictory_or_malformed_field_records_are_refused(tmp_path):
    """A review put two values for one cell and a not-a-number timestamp through the
    reader: the duplicate silently became whichever came last, and the nonfinite time
    passed the step filter, since every comparison against it is false."""
    M = _load()
    good = "FIELD 33035.2500 -32.0000 -61.0000 1.202971274689e-10\n"
    duplicate = tmp_path / "dup.log"
    duplicate.write_text(good + "FIELD 33035.2500 -32.0000 -61.0000 7.0e-10\n")
    read = M.reference_field_at(str(duplicate), 33035.25)
    assert any("two different values" in w for w in read["problems"])
    assert read["records"] == 2 and len(read["cells"]) == 1
    nonfinite = tmp_path / "nan.log"
    nonfinite.write_text(good + "FIELD nan -32.0000 -59.0000 5.0e-10\n")
    read2 = M.reference_field_at(str(nonfinite), 33035.25)
    assert any("not finite" in w for w in read2["problems"])
    assert sorted(read2["cells"]) == [(-32.0, -61.0)]
    malformed = tmp_path / "short.log"
    malformed.write_text(good + "FIELD 33035.2500 -32.0000\n")
    assert any("fields" in w for w in M.reference_field_at(str(malformed), 33035.25)["problems"])
    # and a record set carrying problems is never compared
    port = [[1.202971274689e-10]]
    refused = M.field_comparison([-32.0], [-61.0], port, read)
    assert refused["available"] is False and "two different values" in refused["reason"]
    clean = M.field_comparison([-32.0], [-61.0], port,
                               M.reference_field_at(str(tmp_path / "dup.log"), 33036.0))
    assert clean["available"] is False and "no FIELD records" in clean["reason"]


def test_the_field_comparison_covers_every_row_and_every_reference_cell():
    """A review ran three mutations past the single-row fixture: comparing only the first
    latitude row, comparing magnitudes without their sign, and walking the port's cells
    alone so a reference cell off the port's grid went unseen."""
    M = _load()
    rows, cols = [-30.0, -32.0], [-61.0, -59.0]
    port = [[4.0e-10, -2.0e-10], [1.2e-10, -2.3e-10]]
    reference = {(-30.0, -61.0): 4.0e-10, (-30.0, -59.0): -2.0e-10,
                 (-32.0, -61.0): 1.2e-10, (-32.0, -59.0): -2.3e-10}
    same = M.compare_fields(rows, cols, port, reference)
    assert same["same_field"] is True and same["cells_unmasked_on_both_sides"] == 4
    # a difference in the SECOND row must be found
    second_row = M.compare_fields(rows, cols, port,
                                  {**reference, (-32.0, -59.0): -3.0e-10})
    assert second_row["same_field"] is False
    assert second_row["worst_cell"] == [-32.0, -59.0]
    # a value of the same magnitude and the opposite SIGN is a different field
    flipped = M.compare_fields(rows, cols, port,
                               {**reference, (-32.0, -61.0): -1.2e-10})
    assert flipped["same_field"] is False
    assert abs(flipped["worst_relative_difference"] - 2.0) < 1e-12
    # a reference cell the port's grid does not carry is version 1's own, not invisible
    off_grid = M.compare_fields(rows, cols, port, {**reference, (-34.0, -59.0): 5.0e-10})
    assert off_grid["same_field"] is False
    assert off_grid["cells_unmasked_only_in_version_1"] == 1
    assert off_grid["reference_cells_off_the_port_grid"] == 1


def test_a_reference_value_that_is_not_finite_is_refused_and_two_zeros_agree():
    """A review fed an infinite reference value through the comparison: the division gave
    not-a-number, the worst difference stayed at zero, and the verdict was agreement. Two
    equal zeros, a legitimate value of the field, raised instead."""
    M = _load()
    rows, cols = [-32.0], [-61.0]
    infinite = M.compare_fields(rows, cols, [[1.2e-10]], {(-32.0, -61.0): float("inf")})
    assert infinite["same_field"] is False
    assert infinite["reference_cells_not_finite"] == 1
    assert infinite["cells_unmasked_on_both_sides"] == 0
    nan = M.compare_fields(rows, cols, [[1.2e-10]], {(-32.0, -61.0): float("nan")})
    assert nan["same_field"] is False and nan["reference_cells_not_finite"] == 1
    zeros = M.compare_fields(rows, cols, [[0.0]], {(-32.0, -61.0): 0.0})
    assert zeros["same_field"] is True and zeros["worst_relative_difference"] == 0.0
    assert M.relative_difference(0.0, 0.0) == 0.0
    assert M.relative_difference(1e-10, -1e-10) == 2.0
    assert M.relative_difference(0.0, 1e-10) == 1.0


def test_the_printing_bound_holds_in_both_coordinates_at_once():
    """A Euclidean bound of the same size survived the earlier tests, which displaced one
    coordinate at a time. Two displacements of 4e-5 are inside the per-coordinate bound
    and outside a Euclidean one."""
    M = _load()
    crossings = M.zero_crossings(ROW_FIELD)
    lat, lon = crossings[0]["lat"], crossings[0]["lon"]
    both = M.vertices_on_crossings([{"lat": [lat + 4e-5], "lon": [lon + 4e-5]}], crossings)
    assert both[0]["on_a_crossing"] is True
    assert both[0]["distance_deg"] > 5e-5          # a Euclidean 5e-5 bound would refuse it
    assert both[0]["latitude_difference"] <= 5e-5
    assert both[0]["longitude_difference"] <= 5e-5


def _tiny_case(times):
    lat, lon = np.array([4.0, 2.0, 0.0, -2.0, -4.0]), np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    grid_lat, grid_lon = np.meshgrid(lat, lon, indexing="ij")
    zeros = np.zeros((len(times), lat.size, lon.size))
    return {"time": np.asarray(times, dtype=float), "lat_c": lat, "lon_c": lon,
            "latgrid": grid_lat, "longrid": grid_lon,
            "u_c": zeros, "v_c": zeros, "currv_anom_c": zeros, "advcurrv_anom_c": zeros,
            "u": zeros, "v": zeros, "currv_anom": zeros}


def test_the_real_replay_installs_the_injection_and_records_where_it_applied():
    """Through `replay_port` itself rather than a stand-in, because a review installed the
    hook with nothing to inject and hard-coded the applied count, and both survived tests
    that never ran the replay."""
    M = _load()
    M.DIVERGENCE_STEP = 2.0
    M.BOX = {"lat": (-10.0, 10.0), "lon": (0.0, 10.0)}
    M.WEST, M.FEATURE_LIFE, M.WINDOW = M.BOX, (1.0, 3.0), (1.0, 3.0)
    M.FIELD_ROWS, M.FIELD_COLS = (-4.0, 4.0), (0.0, 10.0)
    case = _tiny_case([1.0, 2.0, 3.0])
    axes = [{"lat": [1.0, 1.0, 1.0], "lon": [4.0, 4.0, 4.0]}]
    *_rest, applied = M.replay_port(case, inject={"time": 2.0, "axes": axes})
    assert applied == [2.0]
    *_rest2, none_applied = M.replay_port(case)
    assert none_applied == []
    # the injection lands at ITS step and no other
    *_rest3, elsewhere = M.replay_port(case, inject={"time": 3.0, "axes": axes})
    assert elsewhere == [3.0]


def test_the_main_replay_is_given_the_reference_field_the_log_carries(exchange, tmp_path):
    """A review dropped `reference_field` at the main call site, so the measurement was
    never made while every helper test stayed green."""
    M = _load()
    seen = {}

    def recorder(case, inject=None, reference_field=None):
        seen["reference_field"] = reference_field
        return [], None, [], [], [], []

    log = exchange / "octave3.log"
    log.write_text(log.read_text()
                   + "FIELD %.4f -32.0000 -61.0000 1.202971274689e-10\n" % DUMPS[3]
                   + "FIELD %.4f -32.0000 -59.0000 -2.318261062885e-10\n" % DUMPS[3])
    M.replay_port = recorder
    M.DIVERGENCE_STEP = DUMPS[3]
    M.main(["--oracle-dir", str(exchange), "--octave-log", str(log)])
    assert seen["reference_field"]["cells"] == {(-32.0, -61.0): 1.202971274689e-10,
                                               (-32.0, -59.0): -2.318261062885e-10}


def _nonuniform_case(times):
    """A tiny case whose coarse curvature clears the threshold in a patch, so the replay's
    captured field is NOT all missing and a reference that differs from it can be told
    apart. A review compared a uniform zero field with a reference built from itself and
    watched both mutations pass."""
    lat = np.array([4.0, 2.0, 0.0, -2.0, -4.0])
    lon = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    grid_lat, grid_lon = np.meshgrid(lat, lon, indexing="ij")
    n = len(times)
    zeros = np.zeros((n, lat.size, lon.size))
    curvature = np.zeros((n, lat.size, lon.size)) + 5.0e-6      # above both thresholds
    advection = np.zeros((n, lat.size, lon.size))
    advection[:, 1, :] = 3.0e-10
    advection[:, 2, :] = -7.0e-10
    return {"time": np.asarray(times, dtype=float), "lat_c": lat, "lon_c": lon,
            "latgrid": grid_lat, "longrid": grid_lon,
            "u_c": zeros, "v_c": zeros, "currv_anom_c": curvature,
            "advcurrv_anom_c": advection, "u": zeros, "v": zeros, "currv_anom": curvature}


def test_the_real_replay_compares_the_field_it_captured_against_the_reference_given():
    M = _load()
    M.DIVERGENCE_STEP = 2.0
    M.BOX = {"lat": (-10.0, 10.0), "lon": (0.0, 10.0)}
    M.WEST, M.FEATURE_LIFE, M.WINDOW = M.BOX, (1.0, 3.0), (1.0, 3.0)
    M.FIELD_ROWS, M.FIELD_COLS = (-4.0, 4.0), (0.0, 10.0)
    case = _nonuniform_case([1.0, 2.0, 3.0])
    # what the port actually holds there, read from a first replay
    *_r, div, _w, _h, _f, _a = (None,) + tuple(M.replay_port(case))
    finite = {(round(la, 4), round(lo, 4)): v
              for la, row in zip(div["rows_lat"], div["masked_smoothed_advection"])
              for lo, v in zip(div["cols_lon"], row) if v is not None}
    assert len(finite) >= 4, "the fixture must leave a field to compare"
    # THE CAPTURED FIELD IS THE PORT'S OWN, SIGNS AND ALL. A review captured the negation
    # and every comparison still agreed, because the reference in this test is built from
    # the captured field itself. The fixture's advection is positive along 2N and negative
    # along the equator, so the captured field has to be too.
    rows = {round(la, 4): [v for v in row if v is not None]
            for la, row in zip(div["rows_lat"], div["masked_smoothed_advection"])}
    assert rows.get(0.0) and all(v < 0 for v in rows[0.0])
    assert sum(sum(r) for r in rows.values()) < 0
    agreeing = {"cells": dict(finite), "records": len(finite), "problems": []}
    _l, div_same, *_ = M.replay_port(case, reference_field=agreeing)
    cmp_same = div_same["field_comparison"]
    assert cmp_same["available"] is True and cmp_same["same_field"] is True
    assert cmp_same["cells_unmasked_on_both_sides"] == len(finite)
    # A REFERENCE THAT DIFFERS MUST BE SEEN TO DIFFER, in value and in sign
    key = sorted(finite)[0]
    differing = {"cells": {**finite, key: finite[key] * 1.5}, "records": len(finite),
                 "problems": []}
    _l2, div_diff, *_ = M.replay_port(case, reference_field=differing)
    assert div_diff["field_comparison"]["same_field"] is False
    flipped = {"cells": {**finite, key: -finite[key]}, "records": len(finite),
               "problems": []}
    _l3, div_flip, *_ = M.replay_port(case, reference_field=flipped)
    assert div_flip["field_comparison"]["same_field"] is False
    # and with no reference the comparison is recorded as unavailable, never as agreement
    _l4, div_none, *_ = M.replay_port(case)
    assert div_none["field_comparison"]["available"] is False
    # THE REPLAY READS THE CROSSING GEOMETRY FROM THE WHOLE GRID, not from the printed
    # crop. Narrow the crop so a quad completing a crossing falls outside it: read from
    # the crop that crossing is undetermined, and read from the grid it is open, which is
    # what the field actually is.
    M.FIELD_COLS = (0.0, 0.0)
    _l5, div_cropped, *_ = M.replay_port(case)
    hoods = div_cropped["crossing_neighborhoods"]
    assert hoods, "the fixture must put a crossing inside the crop"
    assert div_cropped["cols_lon"] == [0.0]
    assert div_cropped["full_field"]["cols_lon"] == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    # the quad completing this crossing lies one column outside the crop and inside the
    # grid: read from the crop it would be unobserved, and the grid says it is open
    assert all(h["closed_off"] is False for h in hoods)
    assert not any(q["unobserved"] for h in hoods for q in h["quads"])
    assert any(q["beyond_the_domain"] for h in hoods for q in h["quads"])
    M.FIELD_COLS = (0.0, 10.0)


def test_the_western_box_count_is_the_box_and_the_feature_life():
    """A review replaced the western results with empty lists and removed the box and
    lifetime filtering, and both survived, because nothing asserted which finished tracks
    the count includes."""
    M = _load()
    M.DIVERGENCE_STEP = 2.0
    M.BOX = {"lat": (-10.0, 10.0), "lon": (0.0, 10.0)}
    M.WEST = {"lat": (0.0, 4.0), "lon": (0.0, 4.0)}
    M.FEATURE_LIFE, M.WINDOW = (1.0, 2.0), (1.0, 3.0)
    M.FIELD_ROWS, M.FIELD_COLS = (-4.0, 4.0), (0.0, 10.0)
    case = _nonuniform_case([1.0, 2.0, 3.0])
    _log, _div, west, _hist, final, _applied = M.replay_port(case)
    # every counted track has an observation inside the west box during the feature's life
    for tr in final:
        inside = any(M.in_box(la, lo, M.WEST) and 1.0 <= t <= 2.0
                     for t, la, lo in zip(tr["time"], tr["meanlat"], tr["meanlon"]))
        assert inside == any(w["first"] == float(min(tr["time"]))
                             and w["steps"] == len(tr["time"]) for w in west) or not inside
    assert len(west) <= len(final)
    # a box the feature never enters counts nothing, while the finished tracks remain
    M.WEST = {"lat": (80.0, 89.0), "lon": (170.0, 179.0)}
    _l, _d, none_west, _h, final2, _a = M.replay_port(case)
    assert none_west == [] and len(final2) == len(final)


def test_a_tab_delimited_field_record_is_read_and_its_contradiction_seen(tmp_path):
    """A review sent a tab-delimited record for a cell already read: the literal prefix
    test skipped it, so the contradiction went unseen and the comparison agreed."""
    M = _load()
    log = tmp_path / "octave.log"
    log.write_text("FIELD 33035.2500 -32.0000 -61.0000 1.0e-10\n"
                   "FIELD\t33035.2500\t-32.0000\t-61.0000\t9.9e-09\n"
                   "FIELDWORK 33035.2500 -32.0000 -59.0000 5.0e-10\n")
    read = M.reference_field_at(str(log), 33035.25)
    assert read["records"] == 2
    assert any("two different values" in w for w in read["problems"])
    # and a line that merely starts with the letters is not a record
    assert sorted(read["cells"]) == [(-32.0, -61.0)]


def test_an_output_made_under_other_settings_is_refused(exchange):
    """The replay hard-codes exclusive=False and absorb=False, so an output produced under
    other settings would be diagnosed by a computation that is not the one it records."""
    M = _load()
    from scipy.io import loadmat
    raw = loadmat(str(exchange / "tracker_port.mat"))
    payload = {k: v for k, v in raw.items() if not k.startswith("__")}
    payload["exclusive"] = 1.0
    savemat(str(exchange / "tracker_port.mat"), payload)
    problems, _prov = M.validate_evidence(str(exchange), "abc123",
                                          str(exchange / "octave3.log"))
    assert any("exclusive=True" in w and "only reproduces exclusive=False" in w
               for w in problems)


def test_an_unparseable_producer_record_is_refused_rather_than_skipped(exchange):
    """A reader outside this work found the reader returning a placeholder on a parse
    failure while the dump-schedule check was guarded by the presence of its key, so an
    uninterpretable record SKIPPED that check instead of failing it."""
    M = _load()
    write_tracks(str(exchange / "tracker_octave_instrumented.mat"), TRACKS, "abc123",
                 {"producer_json": "{this is not json"})
    fake_log(exchange / "octave3.log", "{this is not json")
    problems, _prov = M.validate_evidence(str(exchange), "abc123",
                                          str(exchange / "octave3.log"))
    assert any("producer record does not parse" in w for w in problems)
    # and a record that parses but names no dump list is refused for that, not skipped
    rec = {"producer": "scripts/octave/run_tracker_instrumented.m", "case_id": "abc123",
           "instrumented": True, "repaired_convhull": True, "unrepaired_convhull_sites": 0,
           "runner_sha256": sha(RUNNER), "executed_source_sha256": {"x": "y"}}
    text = json.dumps(rec)
    write_tracks(str(exchange / "tracker_octave_instrumented.mat"), TRACKS, "abc123",
                 {"producer_json": text})
    fake_log(exchange / "octave3.log", text)
    problems2, _p2 = M.validate_evidence(str(exchange), "abc123",
                                         str(exchange / "octave3.log"))
    assert any("names no dump list" in w for w in problems2)


def test_the_declared_claims_default_to_nothing(exchange, tmp_path):
    """A reader outside this work found `--explains-unmatched` defaulting to [19], a
    Sahara fact wearing a generic flag: any other case that forgot to override it claimed
    the Sahara track. Read through the real entry point, from the artifact it writes."""
    M = _load()

    def quiet_replay(case, inject=None, reference_field=None):
        return [], {"finite_cells": 0, "port_axes_in_box": [], "port_waves_in_box": [],
                    "field_comparison": {"available": False, "reason": "no records"}}, \
            [], [], [], []

    M.replay_port = quiet_replay
    M.derive_conclusion = lambda *a, **k: (["nothing happened"], [])
    out = tmp_path / "case.json"
    assert M.main(["--oracle-dir", str(exchange), "--out", str(out)]) == 0
    written = json.loads(out.read_text())
    assert written["explains"] == {"unmatched_v1_tracks": [], "v1_extra_pairs": []}
    assert written["parameters"]["reference_tracks"] == []
    # and a case that declares one gets exactly that one
    assert M.main(["--oracle-dir", str(exchange), "--out", str(out),
                   "--explains-unmatched", "19"]) == 0
    assert json.loads(out.read_text())["explains"]["unmatched_v1_tracks"] == [19]


def test_a_crossing_is_closed_off_only_when_both_quads_lose_two_corners():
    """A reviewer traced the port's own axis finder over a two by two field with ONE
    masked corner and got a line, because the tracer runs with corner_mask=True and traces
    the unmasked triangle. So one masked corner is not the obstruction, and the earlier
    prose said it was."""
    M = _load()
    # the real shape: a finite row with everything above and below it masked
    row = {"rows_lat": [-30.0, -32.0, -34.0], "cols_lon": [-63.0, -61.0, -59.0, -57.0],
           "masked_smoothed_advection": [[None, None, None, None],
                                         [3.6e-10, 1.2e-10, -2.3e-10, -1.3e-10],
                                         [None, None, None, None]]}
    hoods = M.crossing_neighborhoods(row["rows_lat"], row["cols_lon"],
                                     row["masked_smoothed_advection"],
                                     M.zero_crossings(row), domain_complete=True)
    assert len(hoods) == 1 and hoods[0]["closed_off"] is True
    assert [q["masked_corners"] for q in hoods[0]["quads"]] == [2, 2]
    # ONE masked corner leaves a triangle, and the port really does trace it
    import numpy as np
    from aew.v1port import detection as D
    lat, lon = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    LG, NG = np.meshgrid(lat, lon, indexing="ij")
    traced = D.trough_axes(LG, NG, np.array([[1.0, -1.0], [np.nan, 1.0]]))
    assert len(traced) == 1, "the tracer traces the unmasked triangle of a quad"
    open_hood = {"rows_lat": [0.0, 1.0], "cols_lon": [0.0, 1.0],
                 "masked_smoothed_advection": [[1.0, -1.0], [None, 1.0]]}
    hoods2 = M.crossing_neighborhoods(open_hood["rows_lat"], open_hood["cols_lon"],
                                      open_hood["masked_smoothed_advection"],
                                      M.zero_crossings(open_hood), domain_complete=True)
    assert hoods2 and any(h["closed_off"] is False for h in hoods2)
    assert min(q["masked_corners"] for h in hoods2 for q in h["quads"]) <= 1


def test_a_crop_cannot_close_a_crossing_off_and_says_so():
    """A CROP SAYS NOTHING ABOUT THE CELLS BEYOND IT. A reviewer cropped a field so that
    the quad completing a crossing fell outside, watched the first generated version call
    that crossing closed, and then traced the very same crossing on the full field."""
    M = _load()
    import numpy as np
    from aew.v1port import detection as D
    full_rows, full_cols = [0.0, 1.0, 2.0], [0.0, 1.0]
    full_grid = [[1.0, -1.0], [1.0, -1.0], [None, None]]
    crop = {"rows_lat": full_rows[1:], "cols_lon": full_cols,
            "masked_smoothed_advection": full_grid[1:]}
    crossings = M.zero_crossings(crop)
    assert len(crossings) == 1
    # read off the CROP, with the field declared incomplete, the crossing is undetermined
    cropped = M.crossing_neighborhoods(crop["rows_lat"], crop["cols_lon"],
                                       crop["masked_smoothed_advection"], crossings,
                                       domain_complete=False)
    assert cropped[0]["closed_off"] is None
    assert any(q["unobserved"] for q in cropped[0]["quads"])
    # read off the WHOLE field it came from, it is open, and the port really traces it
    whole = M.crossing_neighborhoods(full_rows, full_cols, full_grid, crossings,
                                     domain_complete=True)
    assert whole[0]["closed_off"] is False
    LG, NG = np.meshgrid(np.array(full_rows), np.array(full_cols), indexing="ij")
    field = np.array([[1.0, -1.0], [1.0, -1.0], [np.nan, np.nan]])
    assert len(D.trough_axes(LG, NG, field)) == 1
    # and a quad beyond a REAL domain edge is an obstruction, not an unobserved cell
    edge = M.crossing_neighborhoods(crop["rows_lat"], crop["cols_lon"],
                                    crop["masked_smoothed_advection"], crossings,
                                    domain_complete=True)
    assert edge[0]["closed_off"] is True
    assert any(q["beyond_the_domain"] for q in edge[0]["quads"])


def test_every_module_that_decides_the_artifact_is_fingerprinted():
    """A review found the shared experiment contract had become a dependency of this trace
    without joining the record of what produced a case artifact. Anything the trace
    imports and relies on to decide what it writes belongs in REPLAY_SOURCES."""
    M = _load()
    import os as _os
    assert "scripts/residue_membership.py" in M.REPLAY_SOURCES
    assert "scripts/compare_tracker_oracle.py" in M.REPLAY_SOURCES
    # the port modules whose behaviour the replay exercises
    for module in ("detection", "contours", "association", "pipeline", "climatology"):
        assert f"src/aew/v1port/{module}.py" in M.REPLAY_SOURCES
    # and every one of them is a file that exists, so a fingerprint can be taken
    repo = _os.path.join(HERE, "..")
    for rel in M.REPLAY_SOURCES:
        assert _os.path.exists(_os.path.join(repo, rel)), rel
    # the modules this script imports from scripts/ are exactly the ones it records
    source = open(_os.path.join(repo, "scripts", "trace_sahara_case.py")).read()
    imported = {line.split()[1] for line in source.splitlines()
                if line.startswith("import ") and "as " in line and "#" in line}
    for name in imported:
        assert f"scripts/{name}.py" in M.REPLAY_SOURCES, name
