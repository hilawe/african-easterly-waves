"""The campaign binding check: every link of the retained chain is recorded with its
verdict, a planted break in any link is named as the failed link, and a run whose raw
inputs the validator refuses is not bound. The producer check is the comparison gate's
own and is stubbed here, since it has its own tests."""
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _load():
    path = os.environ.get("BINDINGS_SCRIPT", os.path.join(ROOT, "scripts", "check_campaign_bindings.py"))
    spec = importlib.util.spec_from_file_location("check_campaign_bindings", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


DAY_1994 = 34333.0     # January 1 1994 in days since 1900-01-01


def _world(tmp_path, dataset="era5", year=1994, *, record_year=None, times=None, case_in_mat=None,
           declared_n=None, raw_digest_ok=True, embedded_year=None, record_fine=None, extra_keys=(), drop_keys=(),
           input_names=("u700", "v700"), calibration_digest=None):
    """A fake campaign world for one run: raw inputs, a calibration artifact, a tracks
    file, a record, a paired artifact and a summary row, all consistent unless a keyword
    plants a break."""
    from scipy.io import savemat
    raw_dir = tmp_path / "raw" / dataset
    raw_dir.mkdir(parents=True)
    inputs = {}
    for var in input_names:
        p = raw_dir / f"{dataset}_{var}_{year}_6h_region.nc"
        p.write_bytes(f"raw {var}".encode())
        inputs[p.name] = _sha(p) if raw_digest_ok else "0" * 64
    cal = tmp_path / f"thresholds_protocol_{dataset}_1979_2010.json"
    cal.write_text(json.dumps({"coarse": {"threshold": 4e-7}, "fine": {"threshold": 2e-6}}))
    run = tmp_path / "evidence" / f"{dataset}_{year}"
    run.mkdir(parents=True)
    case = f"case-{dataset}".ljust(32, "0")
    embedded = {"dataset_specific": {"dataset": dataset, "year": embedded_year if embedded_year is not None else year, "inputs_sha256": inputs}}
    t = times if times is not None else np.array([DAY_1994 + 10.0, DAY_1994 + 10.25, DAY_1994 + 10.5])
    mat = {"n": np.array([[declared_n if declared_n is not None else 1]]), "case_id": np.array([case_in_mat or case]),
           "producer_json": np.array([json.dumps(embedded)]),
           "time0": t.reshape(1, -1), "lat0": np.zeros((1, t.size)), "lon0": np.zeros((1, t.size))}
    for k in extra_keys:
        mat[k] = np.array([[DAY_1994 + 400.0]])          # an observation outside the year, under a stray index
    for k in drop_keys:
        mat.pop(k)
    savemat(run / "tracker_port.mat", mat)
    record = {"stage": "tracking", "dataset_specific": {
        "dataset": dataset, "year": record_year if record_year is not None else year, "case_id": case,
        "tracks_sha256": _sha(run / "tracker_port.mat"), "inputs_sha256": inputs,
        "calibration_sha256": calibration_digest or _sha(cal),
        "coarse_threshold": 4e-7, "fine_threshold": record_fine if record_fine is not None else 2e-6}}
    (run / f"tracking_{dataset}_{year}.json").write_text(json.dumps(record))
    manifest = {"datasets": {dataset: {"directory": str(raw_dir), "prefix": dataset}}}
    return manifest, str(tmp_path / "evidence"), str(cal), record


def _stub_producer(monkeypatch):
    import season_metrics
    monkeypatch.setattr(season_metrics, "producer_problems", lambda *a, **k: [])


def _failed(links):
    return [l["link"] for l in links if not l["passed"]]


def test_a_consistent_run_is_bound_and_every_link_is_recorded(tmp_path, monkeypatch):
    C = _load()
    _stub_producer(monkeypatch)
    manifest, evidence, cal, _ = _world(tmp_path)
    links = C.check_run(evidence, "era5", 1994, manifest, "m" * 64, lambda p, y, v: None, lambda ds: cal)
    assert _failed(links) == [] and len(links) == 14
    assert all("link" in l and "passed" in l for l in links)


@pytest.mark.parametrize("plant, expected", [
    ({"record_year": 1995}, "record filed under its dataset and year"),
    ({"times": np.array([DAY_1994 + 364.75, DAY_1994 + 365.0])}, "every observation falls inside the calendar year"),
    ({"times": np.array([DAY_1994 - 0.25, DAY_1994])}, "every observation falls inside the calendar year"),
    ({"case_in_mat": "d" * 32}, "tracks file carries the record's case id"),
    ({"declared_n": 2}, "tracks file holds the number of tracks it declares, contiguously indexed"),
    ({"extra_keys": ("time2",)}, "tracks file holds the number of tracks it declares, contiguously indexed"),
    ({"declared_n": 3, "extra_keys": ("time2",)}, "tracks file holds the number of tracks it declares, contiguously indexed"),
    ({"raw_digest_ok": False}, "raw input era5_u700_1994_6h_region.nc digest equals the record's"),
    ({"embedded_year": 1993}, "embedded producer record names the same dataset, year and raw inputs"),
    ({"record_fine": 3e-6}, "record thresholds equal the calibration artifact's"),
    ({"input_names": ("u700", "u700x")}, "record names exactly the year's two canonical input files"),
    ({"input_names": ("u700",)}, "record names exactly the year's two canonical input files"),
    ({"calibration_digest": "0" * 64}, "calibration digest equals the record's"),
])


def test_each_planted_break_is_named_as_the_failed_link(tmp_path, monkeypatch, plant, expected):
    C = _load()
    _stub_producer(monkeypatch)
    manifest, evidence, cal, _ = _world(tmp_path, **plant)
    links = C.check_run(evidence, "era5", 1994, manifest, "m" * 64, lambda p, y, v: None, lambda ds: cal)
    assert expected in _failed(links)


def test_two_copies_of_one_component_never_count_as_both_validated(tmp_path, monkeypatch):
    C = _load()
    _stub_producer(monkeypatch)
    manifest, evidence, cal, _ = _world(tmp_path, input_names=("u700", "u700x"))     # no v700 file at all
    calls = []
    links = C.check_run(evidence, "era5", 1994, manifest, "m" * 64, lambda p, y, v: calls.append((os.path.basename(p), v)), lambda ds: cal)
    assert "record names exactly the year's two canonical input files" in _failed(links)
    assert "raw input era5_v700_1994_6h_region.nc present" in _failed(links)
    assert calls == [("era5_u700_1994_6h_region.nc", "u700")]                        # the validator is called per canonical path, by its variable
    assert sum(1 for l in links if l.get("raw_validator_called")) == 1


def test_a_producer_the_comparison_gate_rejects_is_named(tmp_path, monkeypatch):
    C = _load()
    import season_metrics
    seen = []

    def rejecting(record, manifest, manifest_sha256, year, side):
        seen.append((year, side))
        return ["stage is not tracking", "manifest digest differs"]
    monkeypatch.setattr(season_metrics, "producer_problems", rejecting)
    manifest, evidence, cal, _ = _world(tmp_path)
    links = C.check_run(evidence, "era5", 1994, manifest, "m" * 64, lambda p, y, v: None, lambda ds: cal)
    assert _failed(links) == ["record passes the comparison gate's producer check"]
    assert [l for l in links if not l["passed"]][0]["detail"] == "stage is not tracking; manifest digest differs"
    assert seen == [(1994, "port")]                                    # the gate is asked about this year and this side


def test_both_ends_of_the_year_and_a_leap_year_are_accepted(tmp_path, monkeypatch):
    import datetime as dt
    C = _load()
    _stub_producer(monkeypatch)
    day = lambda y, m, d: float((dt.date(y, m, d) - dt.date(1900, 1, 1)).days)
    manifest, evidence, cal, _ = _world(tmp_path, year=1994, times=np.array([DAY_1994, day(1994, 12, 31) + 0.75]))
    assert _failed(C.check_run(evidence, "era5", 1994, manifest, "m" * 64, lambda p, y, v: None, lambda ds: cal)) == []
    manifest, evidence, cal, _ = _world(tmp_path / "leap", year=1992, times=np.array([day(1992, 2, 29), day(1992, 12, 31) + 0.75]))
    assert _failed(C.check_run(evidence, "era5", 1992, manifest, "m" * 64, lambda p, y, v: None, lambda ds: cal)) == []


def test_a_stray_index_never_crashes_the_check_and_is_named(tmp_path, monkeypatch):
    C = _load()
    _stub_producer(monkeypatch)
    manifest, evidence, cal, _ = _world(tmp_path, declared_n=3, extra_keys=("time2",))
    links = C.check_run(evidence, "era5", 1994, manifest, "m" * 64, lambda p, y, v: None, lambda ds: cal)
    assert "tracks file holds the number of tracks it declares, contiguously indexed" in _failed(links)
    assert "every observation falls inside the calendar year" in _failed(links)     # the stray array is read, not skipped


def test_a_refused_raw_input_or_a_changed_tracks_file_is_not_bound(tmp_path, monkeypatch):
    C = _load()
    _stub_producer(monkeypatch)
    manifest, evidence, cal, _ = _world(tmp_path)
    links = C.check_run(evidence, "era5", 1994, manifest, "m" * 64,
                        lambda p, y, v: "time vector does not cover the year" if v == "v700" else None, lambda ds: cal)
    assert _failed(links) == ["raw input era5_v700_1994_6h_region.nc passes the retrieval validator for 1994 as v700"]
    with open(os.path.join(evidence, "era5_1994", "tracker_port.mat"), "ab") as fh:
        fh.write(b"\0")
    links = C.check_run(evidence, "era5", 1994, manifest, "m" * 64, lambda p, y, v: None, lambda ds: cal)
    assert "tracks digest equals the record's" in _failed(links)


def _year_world(tmp_path, monkeypatch, *, swap_sides=False, wrong_row_digest=False):
    _stub_producer(monkeypatch)
    manifest = {"datasets": {}}
    for ds in ("eraint", "era5"):
        m, evidence, cal, rec = _world(tmp_path, ds)
        manifest["datasets"].update(m["datasets"])
    ids = {ds: json.load(open(os.path.join(evidence, f"{ds}_1994", f"tracking_{ds}_1994.json")))["dataset_specific"]["case_id"]
           for ds in ("eraint", "era5")}
    tracks = {ds: _sha(os.path.join(evidence, f"{ds}_1994", "tracker_port.mat")) for ds in ("eraint", "era5")}
    v1, port = ("era5", "eraint") if swap_sides else ("eraint", "era5")
    art = {"year": 1994, "sides": {"v1": {"case_id": ids[v1], "dataset": "eraint", "manifest_sha256": "m" * 64},
                                   "port": {"case_id": ids[port], "dataset": "era5", "manifest_sha256": "m" * 64}},
           "inputs_sha256": {"v1": {"sha256": tracks["eraint"]}, "port": {"sha256": tracks["era5"]}},
           "comparison": {"season": {"v1": 135, "port": 125}}}
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    p = artifacts / "paired_1994_eraint_era5.json"
    p.write_text(json.dumps(art))
    rows = {"1994": {"artifact_sha256": "0" * 64 if wrong_row_digest else _sha(p), "africa_origin": {"v1": 135, "port": 125}}}
    return evidence, str(artifacts), rows


def test_a_year_is_bound_when_sides_rows_and_digests_agree(tmp_path, monkeypatch):
    C = _load()
    evidence, artifacts, rows = _year_world(tmp_path, monkeypatch)
    assert _failed(C.check_year(evidence, artifacts, rows, 1994, "m" * 64)) == []


def test_swapped_sides_a_wrong_row_digest_and_a_wrong_row_count_are_named(tmp_path, monkeypatch):
    C = _load()
    evidence, artifacts, rows = _year_world(tmp_path, monkeypatch, swap_sides=True)
    failed = _failed(C.check_year(evidence, artifacts, rows, 1994, "m" * 64))
    assert "side port carries the era5 record's case id" in failed and "side v1 carries the eraint record's case id" in failed
    evidence, artifacts, rows = _year_world(tmp_path / "b", monkeypatch, wrong_row_digest=True)
    assert _failed(C.check_year(evidence, artifacts, rows, 1994, "m" * 64)) == ["summary row names the artifact's digest"]
    evidence, artifacts, rows = _year_world(tmp_path / "c", monkeypatch)
    rows["1994"]["africa_origin"]["port"] = 124
    assert _failed(C.check_year(evidence, artifacts, rows, 1994, "m" * 64)) == ["summary row copies the artifact's season counts"]
    evidence, artifacts, rows = _year_world(tmp_path / "d", monkeypatch)
    p = os.path.join(artifacts, "paired_1994_eraint_era5.json")
    art = json.load(open(p))
    art["year"] = 1993
    open(p, "w").write(json.dumps(art))
    rows["1994"]["artifact_sha256"] = _sha(p)
    assert _failed(C.check_year(evidence, artifacts, rows, 1994, "m" * 64)) == ["artifact names the year"]


def test_run_check_refuses_an_empty_range_and_never_reports_unperformed_validation_as_done(tmp_path, monkeypatch):
    C = _load()
    evidence, artifacts, rows = _year_world(tmp_path, monkeypatch)
    manifest = {"datasets": {ds: {"directory": str(tmp_path / "raw" / ds), "prefix": ds} for ds in ("eraint", "era5")}}
    cal = lambda ds: str(tmp_path / f"thresholds_protocol_{ds}_1979_2010.json")
    with pytest.raises(SystemExit):
        C.run_check([], manifest, "m" * 64, evidence, artifacts, rows, lambda p, y, v: None, cal)
    calls = []
    ok = C.run_check([1994], manifest, "m" * 64, evidence, artifacts, rows, lambda p, y, v: calls.append(p), cal)
    assert ok["verdict"] == "bound" and ok["raw_validator_calls"] == 4 == ok["raw_validations_needed"] == len(calls)
    assert ok["raw_inputs_validated"] is True and ok["links_failed"] == 0
    # a year whose records are absent: nothing validated, nothing bound, and the flag says so
    calls.clear()
    no = C.run_check([1995], manifest, "m" * 64, evidence, artifacts, rows, lambda p, y, v: calls.append(p), cal)
    assert no["verdict"] == "NOT BOUND" and no["raw_inputs_validated"] is False and no["raw_validator_calls"] == 0 == len(calls)
    assert no["links_failed"] >= 3 and all(f["where"] for f in no["failed"])


def test_the_real_validator_is_called_with_the_manifest_grid_the_year_and_the_variable(monkeypatch):
    C = _load()
    import download_eraint_v1port as D
    import export_protocol_case as E
    seen = []
    monkeypatch.setattr(E, "area_and_grid_text", lambda m: ("AREA", "GRID"))
    monkeypatch.setattr(D, "validate_file", lambda path, year, var, area, grid: seen.append((path, year, var, area, grid)) or "why")
    v = C.real_raw_validator({"any": "manifest"})
    assert v("/p/era5_u700_1994_6h_region.nc", 1994, "u700") == "why"
    assert seen == [("/p/era5_u700_1994_6h_region.nc", 1994, "u700", "AREA", "GRID")]
