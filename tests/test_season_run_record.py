"""The run record: the runner's own line is the source of the timing and count, a log
with no or several such lines is refused, a count that disagrees with the output is
refused, and two runs are compared under the exact comparison."""
import importlib.util
import io
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
pytest.importorskip("scipy")


def _load():
    spec = importlib.util.spec_from_file_location(
        "season_run_record", os.path.join(ROOT, "scripts", "season_run_record.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mat(tracks, case="c"):
    from scipy.io import savemat
    payload = {"n": float(len(tracks)), "case_id": case}
    for i, (t, la, lo) in enumerate(tracks):
        payload[f"time{i}"], payload[f"lat{i}"], payload[f"lon{i}"] = np.array(t), np.array(la), np.array(lo)
    buf = io.BytesIO()
    savemat(buf, payload)
    return buf.getvalue()


def _run_dir(tmp_path, name, tracks, log_text, case_bytes=b"case"):
    d = tmp_path / name
    d.mkdir()
    (d / "tracker_case.mat").write_bytes(case_bytes)
    (d / "tracker_octave_instrumented.mat").write_bytes(_mat(tracks))
    log = tmp_path / f"{name}.log"
    log.write_text(log_text)
    return d, log


TRACKS = [([1.0, 1.25], [10.0, 10.5], [-20.0, -20.5]), ([2.0, 2.25, 2.5], [5.0, 5.5, 6.0], [0.0, 0.5, 1.0])]


def test_two_identical_runs_are_recorded_as_identical(tmp_path):
    R = _load()
    a, la = _run_dir(tmp_path, "a", TRACKS, "noise\nfind_ews_f returned 2 tracks in 4084 s\nsaved\n")
    b, lb = _run_dir(tmp_path, "b", TRACKS, "find_ews_f returned 2 tracks in 3997 s\n")
    out = tmp_path / "rec.json"
    R.main(["--run", f"{a}:{la}", "--run", f"{b}:{lb}", "--out", str(out)])
    art = json.loads(out.read_text())
    assert [r["runner_seconds"] for r in art["runs"]] == [4084, 3997]
    assert art["same_case_bytes"] and art["tracks_identical_under_exact_comparison"]
    assert art["outputs_byte_identical"] is True     # same bytes here, though not required


def test_a_moved_observation_or_another_case_is_not_identical(tmp_path):
    R = _load()
    moved = [TRACKS[0], (TRACKS[1][0], [5.0, 5.5, 6.000001], TRACKS[1][2])]
    a, la = _run_dir(tmp_path, "a", TRACKS, "find_ews_f returned 2 tracks in 10 s\n")
    b, lb = _run_dir(tmp_path, "b", moved, "find_ews_f returned 2 tracks in 11 s\n", case_bytes=b"other")
    out = tmp_path / "rec.json"
    R.main(["--run", f"{a}:{la}", "--run", f"{b}:{lb}", "--out", str(out)])
    art = json.loads(out.read_text())
    assert art["same_case_bytes"] is False and art["tracks_identical_under_exact_comparison"] is False


def test_a_log_without_one_runner_line_or_a_disagreeing_count_is_refused(tmp_path):
    R = _load()
    a, la = _run_dir(tmp_path, "a", TRACKS, "nothing here\n")
    with pytest.raises(SystemExit):
        R.read_run(str(a), str(la))
    b, lb = _run_dir(tmp_path, "b", TRACKS, "find_ews_f returned 3 tracks in 10 s\n")
    with pytest.raises(SystemExit):
        R.read_run(str(b), str(lb))
    c, lc = _run_dir(tmp_path, "c", TRACKS, "find_ews_f returned 2 tracks in 10 s\nfind_ews_f returned 2 tracks in 12 s\n")
    with pytest.raises(SystemExit):
        R.read_run(str(c), str(lc))
