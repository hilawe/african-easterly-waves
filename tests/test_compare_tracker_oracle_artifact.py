"""The whole-tracker comparison's recorded artifact, bound on synthetic exchange files
that carry one case id and producer records written the way the two harness sides
write them, with well-formed digests.

MUTATION LIST, written before the assertions:
  C1 the artifact omits the executed-source digests the oracle's producer record
     carries (a repaired-hull run and a degraded run then look alike on record);
  C2 faithfulness is read from the file name instead of the producer record's
     detected repair mode;
  C3 the identical-pair count is not recorded (a pair equal on shared steps but longer
     on one side would pass as identical if the count came from the shared-step test);
  C4 a case-id mismatch among the three files still writes an artifact.
Added after a review found provenance promoted by file name:
  C5 an output with NO producer record is reported as verified or faithful;
  C6 a producer record whose executed-source digests differ from the tree beside the
     output is reported as verified;
  C7 the comparison checkout's head is recorded under the port's producing head.
Added after a second review fed the validator four contradictions it accepted:
  C8 a producer record naming ANOTHER case is verified;
  C9 port settings that contradict the output's own flags are verified;
  C10 an EMPTY executed-source mapping is verified (no mismatch to find);
  C11 a digest that is not a sha256 hexdigest is verified;
  C12 a repair flag that disagrees with the recorded count of unrepaired sites is
      verified, or the runner's own digest is not checked against the runner file.
Added after a third review showed the port side checked only digest syntax:
  C13 a well-formed port digest that matches no file is verified;
  C14 a port source path that does not exist is verified;
  C15 a port record naming any one path under the package, but not the pipeline
      module or the exporter, is valid.
"""
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import pytest
from scipy.io import loadmat, savemat

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "..", "scripts", "octave", "run_tracker_instrumented.m")


def _load():
    spec = importlib.util.spec_from_file_location(
        "compare_tracker_oracle",
        os.path.join(HERE, "..", "scripts", "compare_tracker_oracle.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compare_tracker_oracle"] = mod
    spec.loader.exec_module(mod)
    return mod


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def digest_of(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write_tracks(path, tracks, case_id, extra=None):
    payload = {"n": float(len(tracks)), "case_id": case_id}
    for i, (t, la, lo) in enumerate(tracks):
        payload[f"time{i}"] = np.asarray(t, dtype=float)
        payload[f"lat{i}"] = np.asarray(la, dtype=float)
        payload[f"lon{i}"] = np.asarray(lo, dtype=float)
    payload.update(extra or {})
    savemat(path, payload)


def read_record(path):
    return json.loads(str(np.asarray(loadmat(path, variable_names=["producer_json"])
                                     ["producer_json"]).ravel()[0]))


V1 = [([0, 1, 2], [10, 10, 10], [-20, -21, -22]),
      ([0, 1, 2, 3], [5, 5, 5, 5], [-30, -31, -32, -33]),
      ([3, 4, 5], [-9, -9, -9], [35, 36, 37])]
PORT = [([0, 1, 2], [10, 10, 10], [-20, -21, -22]),
        ([0, 1, 2], [5, 5, 5], [-30, -31, -32]),
        ([2, 3, 4], [20, 20, 20], [-60, -61, -62])]
HEAD = "feedface" * 5
REPO = os.path.join(HERE, "..")
PIPELINE = os.path.join(REPO, "src", "aew", "v1port", "pipeline.py")
EXPORTER = os.path.join(REPO, "scripts", "export_tracker_case.py")


def oracle_record(exchange, **overrides):
    rec = {"producer": "scripts/octave/run_tracker_instrumented.m",
           "case_id": "abc123", "instrumented": True, "repaired_convhull": True,
           "unrepaired_convhull_sites": 0, "octave_version": "11.3.0",
           "dump_times": [1.0], "runner_sha256": sha(RUNNER),
           "executed_source_sha256": {
               "v1_instrumented__find_ews_f_m": sha(str(exchange / "v1_instrumented"
                                                        / "find_ews_f.m"))}}
    rec.update(overrides)
    return rec


def port_record(**overrides):
    rec = {"producer": "scripts/export_tracker_case.py", "case_id": "abc123",
           "git_head": HEAD, "git_dirty": False,
           # the GENUINE digests of this checkout's files, so the positive control is a
           # record that really identifies the producing code
           "source_sha256": {"src/aew/v1port/pipeline.py": sha(PIPELINE),
                             "scripts/export_tracker_case.py": sha(EXPORTER)},
           "settings": {"exclusive": False, "absorb": False, "start": 600, "steps": 60},
           "climo_cache_sha256": digest_of("cache")}
    rec.update(overrides)
    return rec


def write_oracle(exchange, rec):
    write_tracks(str(exchange / "tracker_octave_instrumented.mat"), V1, "abc123",
                 {"producer_json": json.dumps(rec)})


def write_port(exchange, rec, exclusive=False):
    write_tracks(str(exchange / "tracker_port.mat"), PORT, "abc123",
                 {"exclusive": float(exclusive), "absorb": 0.0,
                  "producer_json": json.dumps(rec)})


@pytest.fixture()
def exchange(tmp_path):
    """Three files for one case with valid producer records on both outputs."""
    d = tmp_path / "oracle"
    (d / "v1_instrumented").mkdir(parents=True)
    (d / "v1_instrumented" / "find_ews_f.m").write_text(
        "function ews = find_ews_f()\n% repaired: convhull(x,y)\n")
    savemat(str(d / "tracker_case.mat"), {"case_id": "abc123", "time": np.arange(6.0)})
    write_oracle(d, oracle_record(d))
    write_port(d, port_record())
    return d


def run(M, exchange, out, oracle="tracker_octave_instrumented.mat"):
    return M.main(["--oracle-dir", str(exchange), "--oracle", oracle, "--out", str(out)])


def test_valid_records_verify_and_the_matching_is_recorded(exchange, tmp_path, capsys):
    M = _load()
    out = tmp_path / "cmp.json"
    assert run(M, exchange, out) == 0
    v = json.load(open(out))
    assert v["case_id"] == "abc123"
    assert v["oracle_provenance"] == "verified" and v["oracle_faithful"] is True
    assert v["oracle_problems"] == [] and v["oracle_tree_mismatches"] == []
    assert v["oracle_producer"]["executed_source_sha256"] == {
        "v1_instrumented__find_ews_f_m": sha(str(exchange / "v1_instrumented" / "find_ews_f.m"))}
    assert v["port_provenance"] == "verified" and v["port_problems"] == []
    assert v["port_tree_mismatches"] == []
    assert v["port_producer"]["git_head"] == HEAD
    assert v["comparison_git_head"] != HEAD
    assert v["v1_tracks"] == 3 and v["port_tracks"] == 3
    assert v["v1_matched"] == 2 and v["port_matched"] == 2
    assert v["identical_pairs"] == 1 and v["nonidentical_pairs"] == 1
    for name in ("tracker_case.mat", "tracker_port.mat", "tracker_octave_instrumented.mat",
                 "v1_instrumented/find_ews_f.m"):
        assert v["input_sha256"][name] == sha(str(exchange / name))
    text = capsys.readouterr().out
    assert "PROVENANCE" not in text and "NOT repaired" not in text


def test_no_producer_record_is_unverified_and_faithfulness_unknown(exchange, tmp_path,
                                                                    capsys):
    M = _load()
    write_tracks(str(exchange / "tracker_octave_instrumented.mat"), V1, "abc123")
    write_tracks(str(exchange / "tracker_port.mat"), PORT, "abc123",
                 {"exclusive": 0.0, "absorb": 0.0})
    out = tmp_path / "cmp.json"
    assert run(M, exchange, out) == 0
    v = json.load(open(out))
    assert v["oracle_provenance"] == "unverified" and v["oracle_faithful"] is None
    assert v["port_provenance"] == "unverified"
    assert "PROVENANCE UNVERIFIED" in capsys.readouterr().out


def test_changed_source_beside_old_output_is_inconsistent_but_the_record_kept(
        exchange, tmp_path, capsys):
    M = _load()
    (exchange / "v1_instrumented" / "find_ews_f.m").write_text("this source never produced "
                                                                "those tracks\n")
    out = tmp_path / "cmp.json"
    assert run(M, exchange, out) == 0
    v = json.load(open(out))
    assert v["oracle_provenance"] == "inconsistent"
    assert v["oracle_tree_mismatches"] == ["v1_instrumented/find_ews_f.m"]
    assert v["oracle_faithful"] is True                 # the record of that run stands
    assert v["oracle_producer"]["case_id"] == "abc123"
    assert "PROVENANCE INCONSISTENT" in capsys.readouterr().out


def test_unrepaired_record_under_the_instrumented_name_is_not_faithful(exchange, tmp_path,
                                                                        capsys):
    M = _load()
    write_oracle(exchange, oracle_record(exchange, repaired_convhull=False,
                                         unrepaired_convhull_sites=3))
    out = tmp_path / "cmp.json"
    assert run(M, exchange, out) == 0
    v = json.load(open(out))
    assert v["oracle_provenance"] == "verified" and v["oracle_faithful"] is False
    assert "NOT repaired" in capsys.readouterr().out


@pytest.mark.parametrize("label, overrides, problem", [
    ("another case", {"case_id": "DIFFERENT-CASE"}, "record case 'DIFFERENT-CASE'"),
    ("empty sources", {"executed_source_sha256": {}}, "executed_source_sha256 is empty"),
    ("bad digest", {"executed_source_sha256": {"v1_instrumented__find_ews_f_m": "not-a-digest"}},
     "is not a sha256 digest"),
    ("no find_ews_f", {"executed_source_sha256": {"shims__smooth_m": "0" * 64}},
     "does not name find_ews_f.m"),
    ("repair flag contradicts count", {"repaired_convhull": True,
                                       "unrepaired_convhull_sites": 3},
     "disagrees with unrepaired_convhull_sites"),
    ("runner digest malformed", {"runner_sha256": "r"}, "runner_sha256 is not a sha256"),
    ("missing field", {"instrumented": None}, "instrumented is not bool"),
])
def test_oracle_record_contradictions_are_invalid_with_a_reason(exchange, tmp_path, capsys,
                                                                 label, overrides, problem):
    M = _load()
    write_oracle(exchange, oracle_record(exchange, **overrides))
    out = tmp_path / "cmp.json"
    assert run(M, exchange, out) == 0
    v = json.load(open(out))
    assert v["oracle_provenance"] == "invalid", label
    assert v["oracle_faithful"] is None, label
    assert any(problem in p for p in v["oracle_problems"]), (label, v["oracle_problems"])
    assert "PROVENANCE INVALID" in capsys.readouterr().out


def test_runner_digest_is_checked_against_the_runner_file(exchange, tmp_path):
    M = _load()
    write_oracle(exchange, oracle_record(exchange, runner_sha256="0" * 64))
    out = tmp_path / "cmp.json"
    assert run(M, exchange, out) == 0
    v = json.load(open(out))
    assert v["oracle_provenance"] == "inconsistent"
    assert v["oracle_tree_mismatches"] == ["scripts/octave/run_tracker_instrumented.m"]


@pytest.mark.parametrize("label, overrides, exclusive, problem", [
    ("another case", {"case_id": "DIFFERENT-CASE"}, False, "record case 'DIFFERENT-CASE'"),
    ("settings contradict output", {"settings": {"exclusive": True, "absorb": True}}, False,
     "settings.exclusive True disagrees with the output's"),
    ("bad digest", {"source_sha256": {"src/aew/v1port/pipeline.py": "not-a-digest",
                                      "scripts/export_tracker_case.py": "0" * 64}}, False,
     "is not a sha256 digest"),
    ("empty sources", {"source_sha256": {}}, False, "source_sha256 is empty"),
    ("no pipeline entry", {"source_sha256": {"src/aew/v1port/not_the_tracker.py": "0" * 64,
                                             "scripts/export_tracker_case.py": "0" * 64}},
     False, "does not name src/aew/v1port/pipeline.py"),
    ("no exporter entry", {"source_sha256": {"src/aew/v1port/pipeline.py": "0" * 64}}, False,
     "does not name scripts/export_tracker_case.py"),
    ("head malformed", {"git_head": "feedface"}, False, "git_head is not a commit"),
])
def test_port_record_contradictions_are_invalid_with_a_reason(exchange, tmp_path, label,
                                                               overrides, exclusive, problem):
    M = _load()
    write_port(exchange, port_record(**overrides), exclusive=exclusive)
    out = tmp_path / "cmp.json"
    assert run(M, exchange, out) == 0
    v = json.load(open(out))
    assert v["port_provenance"] == "invalid", label
    assert any(problem in p for p in v["port_problems"]), (label, v["port_problems"])


def test_port_settings_that_agree_with_an_exclusive_output_verify(exchange, tmp_path):
    M = _load()
    write_port(exchange, port_record(settings={"exclusive": True, "absorb": False}),
               exclusive=True)
    out = tmp_path / "cmp.json"
    assert run(M, exchange, out) == 0
    assert json.load(open(out))["port_provenance"] == "verified"


def test_case_mismatch_writes_nothing(exchange, tmp_path):
    M = _load()
    write_tracks(str(exchange / "tracker_port.mat"), [([0, 1], [1, 1], [1, 1])], "other",
                 {"exclusive": 0.0, "absorb": 0.0})
    out = tmp_path / "cmp.json"
    with pytest.raises(SystemExit, match="not one case"):
        run(M, exchange, out)
    assert not out.exists()


@pytest.mark.parametrize("label, sources, expected_mismatch", [
    ("well-formed digest matching no file",
     {"src/aew/v1port/pipeline.py": "0" * 64, "scripts/export_tracker_case.py": None},
     "src/aew/v1port/pipeline.py"),
    ("path that does not exist",
     {"src/aew/v1port/pipeline.py": None, "scripts/export_tracker_case.py": None,
      "src/aew/v1port/not_the_tracker.py": "0" * 64},
     "src/aew/v1port/not_the_tracker.py"),
])
def test_port_source_that_does_not_match_the_tree_is_inconsistent_with_the_record_kept(
        exchange, tmp_path, capsys, label, sources, expected_mismatch):
    M = _load()
    genuine = {"src/aew/v1port/pipeline.py": sha(PIPELINE),
               "scripts/export_tracker_case.py": sha(EXPORTER)}
    mapping = {k: (v if v is not None else genuine[k]) for k, v in sources.items()}
    write_port(exchange, port_record(source_sha256=mapping))
    out = tmp_path / "cmp.json"
    assert run(M, exchange, out) == 0
    v = json.load(open(out))
    assert v["port_provenance"] == "inconsistent", label
    assert v["port_tree_mismatches"] == [expected_mismatch], label
    assert v["port_producer"]["git_head"] == HEAD           # the historical record stands
    assert "PORT PROVENANCE INCONSISTENT" in capsys.readouterr().out


def test_a_declared_count_that_is_not_a_whole_number_is_refused(tmp_path):
    """A review set both outputs' counts to -1 and watched the comparison succeed on zero
    tracks a side, with both provenance records still verified."""
    import importlib.util
    import sys as _sys
    import numpy as _np
    import pytest as _pytest
    from scipy.io import savemat as _savemat
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "compare_tracker_oracle", os.path.join(here, "..", "scripts",
                                               "compare_tracker_oracle.py"))
    C = importlib.util.module_from_spec(spec)
    _sys.modules["compare_tracker_oracle"] = C
    spec.loader.exec_module(C)
    for bad in (-1.0, 2.5, float("nan"), float("inf")):
        path = tmp_path / f"tracks_{str(bad).replace('.', '_')}.mat"
        _savemat(str(path), {"n": bad, "case_id": "abc123",
                             "lat0": _np.array([1.0]), "lon0": _np.array([2.0]),
                             "time0": _np.array([3.0])})
        with _pytest.raises(SystemExit) as e:
            C.read_tracks(str(path))
        assert "not a whole count" in str(e.value), bad
    ok = tmp_path / "good.mat"
    _savemat(str(ok), {"n": 1.0, "case_id": "abc123", "lat0": _np.array([1.0]),
                       "lon0": _np.array([2.0]), "time0": _np.array([3.0])})
    tracks, case = C.read_tracks(str(ok))
    assert len(tracks) == 1 and case == "abc123"
