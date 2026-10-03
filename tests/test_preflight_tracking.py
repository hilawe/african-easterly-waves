"""The tracking preflight: the window is cut from the year as asked, and the run passes only
when its tracks, bindings and provenance are the reference's. The entry point's stages are
stood in for here, since the real window needs the retained inputs."""
import importlib.util
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    path = os.environ.get("PREFLIGHT_SCRIPT", os.path.join(ROOT, "scripts", "preflight_tracking.py"))
    spec = importlib.util.spec_from_file_location("preflight_tracking_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TRACK = {"time": np.array([40390.0, 40390.25, 40390.5]), "lat": np.array([10.0, 10.25, 10.5]), "lon": np.array([5.0, 4.5, 4.0])}
BIND = {"manifest_sha256": "m" * 64, "calibration_sha256": "c" * 64,
        "climatology_inputs_sha256": {"era5_u700_1979_6h_region.nc": "1" * 64}, "inputs_sha256": {"era5_u700_2010_6h_region.nc": "2" * 64}}


SOURCE = {"src/aew/v1port/detection.py": "5" * 64, "scripts/export_protocol_case.py": "6" * 64}


def _record(head="a" * 40, dirty=False, source=None):
    return {"git_head": head, "git_dirty": dirty, "protocol_settings": {"manifest_sha256": BIND["manifest_sha256"]},
            "source_sha256": dict(source or SOURCE),
            "dataset_specific": {k: v for k, v in BIND.items() if k != "manifest_sha256"}}


def _stand_in(P, monkeypatch, seen, track=TRACK, record=None):
    """The entry point's stages, recording the window handed to the tracking stage."""
    from scipy.io import savemat
    times = np.arange(1460, dtype=float)
    arrays = {"times": times, "u": times[:, None, None] * np.ones((1, 2, 2)), "v": np.zeros((1460, 2, 2)),
              "curvature": np.zeros((1460, 2, 2)), "coarse": {"u": times[:, None, None] * np.ones((1, 1, 1))},
              "latgrid": np.zeros((2, 2))}
    monkeypatch.setattr(P.E, "load_manifest", lambda p: ({"datasets": {"era5": {"prefix": "era5"}}}, "m" * 64))
    monkeypatch.setattr(P.E, "validate_inputs", lambda m, d, years: ({}, BIND["inputs_sha256"]))
    monkeypatch.setattr(P.E, "inputs_stage", lambda m, d, y, pf: ({"timesteps": 1460}, arrays))

    def tracking_stage(manifest, sha, name, dataset, year, a, cache, cal, run_dir, inputs, readiness):
        seen["times"], seen["coarse_u"], seen["latgrid"] = a["times"], a["coarse"]["u"], a["latgrid"]
        savemat(os.path.join(run_dir, "tracker_port.mat"), {"n": 1.0, "case_id": "f" * 32, "lat0": track["lat"],
                                                            "lon0": track["lon"], "time0": track["time"]})
        return (record or _record()), 1
    monkeypatch.setattr(P.E, "tracking_stage", tracking_stage)


def _reference(tmp_path, P, monkeypatch, **window):
    path = tmp_path / "reference.json"
    _stand_in(P, monkeypatch, {})
    argv = ["--climo-cache", "c.npz", "--write-reference", str(path)] + sum(([f"--{k}", str(v)] for k, v in window.items()), [])
    assert P.main(argv) == 0
    return path


def test_the_window_is_cut_from_the_year_as_asked_and_nothing_else_is_cut(monkeypatch):
    P = _load()
    seen = {}
    _stand_in(P, monkeypatch, seen)
    tracks, _ = P.run_window("m.json", "era5", 2010, P.window_start(2010, 8, 1), 24, "c.npz", "cal.json")
    assert P.window_start(2010, 8, 1) == 848
    assert list(seen["times"]) == list(np.arange(848, 872, dtype=float))
    assert list(seen["coarse_u"][:, 0, 0]) == list(np.arange(848, 872, dtype=float))
    assert seen["latgrid"].shape == (2, 2)                                   # the grids are not time-major and are not cut
    assert tracks == [[list(TRACK["time"]), list(TRACK["lat"]), list(TRACK["lon"])]]
    with pytest.raises(SystemExit, match="not inside the year"):
        P.run_window("m.json", "era5", 2010, 1450, 24, "c.npz", "cal.json")


def test_a_run_equal_to_the_reference_passes(tmp_path, monkeypatch):
    P = _load()
    ref = _reference(tmp_path, P, monkeypatch)
    assert P.main(["--climo-cache", "c.npz", "--reference", str(ref)]) == 0
    with pytest.raises(SystemExit, match="exists and a reference is never overwritten"):
        P.main(["--climo-cache", "c.npz", "--write-reference", str(ref)])


@pytest.mark.parametrize("track, record, expected", [
    (dict(TRACK, lat=TRACK["lat"] + np.array([0.0, 0.0, 1e-12])), None, "1 of 1 tracks differ from the reference"),
    (TRACK, _record(dirty=None), "git cannot record provenance here (head 'aaaaaaaaaa"),
    (TRACK, _record(head=None), "git cannot record provenance here (head None"),
    (TRACK, _record(dirty=True), "has uncommitted tracking source"),
    (TRACK, _record(source=dict(SOURCE, **{"src/aew/v1port/detection.py": "7" * 64})),
     "source_sha256 is not the reference's, so this is not the reference run. A changed tracker needs a new reference"),
    (TRACK, dict(_record(), dataset_specific=dict(_record()["dataset_specific"], calibration_sha256="d" * 64)),
     "calibration_sha256 is not the reference's"),
    (TRACK, dict(_record(), dataset_specific=dict(_record()["dataset_specific"],
                                                  climatology_inputs_sha256={"era5_u700_1979_6h_region.nc": "3" * 64})),
     "climatology_inputs_sha256 is not the reference's"),
])
def test_a_run_that_is_not_the_references_is_refused_and_says_why(tmp_path, monkeypatch, capsys, track, record, expected):
    P = _load()
    ref = _reference(tmp_path, P, monkeypatch)
    _stand_in(P, monkeypatch, {}, track=track, record=record)
    assert P.main(["--climo-cache", "c.npz", "--reference", str(ref)]) == 1
    out = capsys.readouterr().out
    assert expected in out and "PREFLIGHT REFUSED" in out, out


def test_another_window_is_not_compared_with_the_reference(tmp_path, monkeypatch, capsys):
    P = _load()
    ref = _reference(tmp_path, P, monkeypatch)
    assert P.main(["--climo-cache", "c.npz", "--reference", str(ref), "--start", "900"]) == 1
    assert "is not the reference's" in capsys.readouterr().out


def test_a_reference_is_written_only_from_a_named_commit_with_a_clean_tree(tmp_path, monkeypatch):
    P = _load()
    _stand_in(P, monkeypatch, {}, record=_record(dirty=True))
    with pytest.raises(SystemExit, match="only from a named commit with a checked clean tree"):
        P.main(["--climo-cache", "c.npz", "--write-reference", str(tmp_path / "r.json")])
    assert not (tmp_path / "r.json").exists()


def test_a_missing_tracker_library_is_refused_by_name(monkeypatch):
    P = _load()
    monkeypatch.setattr(P, "MODULES", ("numpy", "no_such_contouring_library"))
    with pytest.raises(SystemExit, match="no_such_contouring_library does not import here"):
        P.environment()
    assert "contourpy" in _load().MODULES                                  # the library whose absence stopped a batch


def test_another_commit_with_the_same_tracking_source_passes(tmp_path, monkeypatch):
    """A commit that changes no tracking source, documentation for instance, changes nothing
    the preflight compares, so the commit itself is not required to be the reference's."""
    P = _load()
    ref = _reference(tmp_path, P, monkeypatch)
    _stand_in(P, monkeypatch, {}, record=_record(head="b" * 40))
    assert P.main(["--climo-cache", "c.npz", "--reference", str(ref)]) == 0
