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
    # neighbours above and below are masked, draws nothing either
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
