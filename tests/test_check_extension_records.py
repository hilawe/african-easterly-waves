"""The extension campaign checker, on a synthetic world: the real base manifest extended by
two tracking years, a retained original campaign of 32 records, and two runs."""
import hashlib
import importlib.util
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
BASE = os.path.join(ROOT, "docs", "aewc_v2", "protocol", "manifest_2026-09-25.json")
TRACKER = ("src/aew/v1port/pipeline.py", "src/aew/v1port/detection.py")
THRESHOLDS = (4.8e-07, 2.6e-06)


def _load():
    path = os.environ.get("EXTCHECK_SCRIPT", os.path.join(ROOT, "scripts", "check_extension_records.py"))
    spec = importlib.util.spec_from_file_location("check_extension_records_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _world(tmp_path, C, monkeypatch):
    """Returns (paths, mapping) for a world that passes, with E.check_calibration stood in."""
    from scipy.io import savemat
    base = json.load(open(BASE))
    ext = dict(base, tracking_years=[2011, 2012], extends={"manifest": BASE, "sha256": _sha(open(BASE, "rb").read())},
               extension_note="test")
    man = tmp_path / "extension.json"
    man.write_text(json.dumps(ext))
    man_sha = _sha(man.read_bytes())
    mapping = {f"era5_{v}_{y}_6h_region.nc": _sha(f"{v}{y}".encode()) for y in range(1979, 2011) for v in ("u700", "v700")}
    monkeypatch.setattr(C.E, "check_calibration", lambda path, m, d: (THRESHOLDS[0], THRESHOLDS[1], "c" * 64, dict(mapping)))
    tracker = {k: _sha(k.encode()) for k in TRACKER}
    sources = {k: "a" * 64 for k in C.S.expected_tracking_sources()}
    sources.update({k: v for k, v in tracker.items() if k in sources})
    tracker_in_inventory = {k: v for k, v in sources.items() if k.startswith("src/aew/v1port/")}
    retained = tmp_path / "retained"
    for y in range(1979, 2011):
        (retained / f"era5_{y}").mkdir(parents=True)
        (retained / f"era5_{y}" / f"tracking_era5_{y}.json").write_text(json.dumps(
            {"dataset_specific": {"climatology_inputs_sha256": mapping}, "source_sha256": tracker_in_inventory}))
    inputs, runs = tmp_path / "inputs", tmp_path / "runs"
    inputs.mkdir()
    for year in (2011, 2012):
        names = {}
        for v in ("u700", "v700"):
            f = inputs / f"era5_{v}_{year}_6h_region.nc"
            f.write_bytes(f"input {v} {year}".encode())
            names[f.name] = _sha(f.read_bytes())
        run = runs / f"era5_{year}"
        run.mkdir(parents=True)
        day0 = float((np.datetime64(f"{year}-08-01") - np.datetime64("1900-01-01")).astype(int))
        savemat(str(run / "tracker_port.mat"), {"n": 1.0, "case_id": "f" * 32, "lat0": np.array([10.0, 10.5]),
                                                "lon0": np.array([5.0, 4.0]), "time0": np.array([day0, day0 + 0.25])})
        record = {"stage": "tracking", "git_head": "b" * 40, "git_dirty": False, "source_sha256": sources,
                  "protocol_settings": {**{k: ext[k] for k in C.S.PROTOCOL_KEYS}, "manifest_sha256": man_sha},
                  "dataset_specific": {"dataset": "era5", "label": ext["datasets"]["era5"]["label"],
                                       "prefix": ext["datasets"]["era5"]["prefix"], "year": year,
                                       "inputs_sha256": names, "climatology_inputs_sha256": dict(mapping),
                                       "coarse_threshold": THRESHOLDS[0], "fine_threshold": THRESHOLDS[1],
                                       "calibration_sha256": "c" * 64,
                                       "tracks_sha256": _sha((run / "tracker_port.mat").read_bytes()),
                                       "manifest_extension": {"tracking_years": [2011, 2012], "extends": ext["extends"],
                                                              "climatology_and_calibration_years": [1979, 2010]}}}
        (run / f"tracking_era5_{year}.json").write_text(json.dumps(record))
    return {"man": str(man), "retained": str(retained), "inputs": str(inputs), "runs": str(runs)}, mapping


def _rewrite(runs, year, change):
    path = os.path.join(runs, f"era5_{year}", f"tracking_era5_{year}.json")
    record = json.load(open(path))
    change(record)
    open(path, "w").write(json.dumps(record))


def _check(C, w):
    return C.check(w["runs"], w["man"], w["retained"], w["inputs"])


def test_a_consistent_extension_campaign_passes(tmp_path, monkeypatch):
    C = _load()
    w, _ = _world(tmp_path, C, monkeypatch)
    report = _check(C, w)
    assert report["passed"] and report["years"] == {"2011": [], "2012": []}, report


def test_one_changed_digest_under_an_unchanged_file_name_is_refused(tmp_path, monkeypatch):
    """The strengthened climatology check: the whole name-to-digest mapping is compared with
    the retained campaign's, so a record that keeps every file name but names one other
    digest fails, and names the file."""
    C = _load()
    w, mapping = _world(tmp_path, C, monkeypatch)
    name = "era5_v700_1994_6h_region.nc"
    _rewrite(w["runs"], 2012, lambda r: r["dataset_specific"]["climatology_inputs_sha256"].__setitem__(name, "0" * 64))
    after = json.load(open(os.path.join(w["runs"], "era5_2012", "tracking_era5_2012.json")))["dataset_specific"]["climatology_inputs_sha256"]
    assert set(after) == set(mapping) and after[name] != mapping[name]          # the same names, one digest moved
    report = _check(C, w)
    assert not report["passed"] and report["years"]["2011"] == []
    assert report["years"]["2012"] == [f"the climatology inputs names a different digest for {name}"]


def test_an_added_or_a_missing_climatology_file_is_refused(tmp_path, monkeypatch):
    C = _load()
    w, _ = _world(tmp_path, C, monkeypatch)
    _rewrite(w["runs"], 2011, lambda r: r["dataset_specific"]["climatology_inputs_sha256"].pop("era5_u700_1979_6h_region.nc"))
    _rewrite(w["runs"], 2012, lambda r: r["dataset_specific"]["climatology_inputs_sha256"].__setitem__("era5_u700_2011_6h_region.nc", "1" * 64))
    report = _check(C, w)
    assert report["years"]["2011"] == ["the climatology inputs lacks era5_u700_1979_6h_region.nc"]
    assert report["years"]["2012"] == ["the climatology inputs adds era5_u700_2011_6h_region.nc"]


def test_retained_records_that_disagree_are_refused_before_any_record_is_read(tmp_path, monkeypatch):
    C = _load()
    w, _ = _world(tmp_path, C, monkeypatch)
    path = os.path.join(w["retained"], "era5_1990", "tracking_era5_1990.json")
    r = json.load(open(path))
    r["dataset_specific"]["climatology_inputs_sha256"]["era5_u700_1990_6h_region.nc"] = "2" * 64
    open(path, "w").write(json.dumps(r))
    with pytest.raises(SystemExit, match="retained records disagree on the climatology mapping"):
        _check(C, w)


def test_the_calibration_inputs_must_be_the_retained_mapping(tmp_path, monkeypatch):
    C = _load()
    w, mapping = _world(tmp_path, C, monkeypatch)
    moved = dict(mapping, **{"era5_u700_2000_6h_region.nc": "3" * 64})
    monkeypatch.setattr(C.E, "check_calibration", lambda path, m, d: (THRESHOLDS[0], THRESHOLDS[1], "c" * 64, moved))
    report = _check(C, w)
    assert not report["passed"] and report["calibration_problems"] == [
        "the calibration's inputs names a different digest for era5_u700_2000_6h_region.nc"]


@pytest.mark.parametrize("change, expected", [
    (lambda r: r.__setitem__("git_dirty", None), "the record's tree state is None, not a checked clean tree"),
    (lambda r: r.__setitem__("git_head", None), "the record names no commit (None)"),
    (lambda r: r["dataset_specific"].__setitem__("fine_threshold", 3e-06), "the thresholds are not the calibration artifact's"),
    (lambda r: r["dataset_specific"].__setitem__("calibration_sha256", "d" * 64), "the calibration digest is not the artifact's"),
    (lambda r: r["dataset_specific"]["manifest_extension"].__setitem__("climatology_and_calibration_years", [1979, 2011]),
     "the extension block is not the manifest's tracking, climatology and calibration years and base"),
    (lambda r: r["source_sha256"].__setitem__("src/aew/v1port/detection.py", "e" * 64), "the tracker source is not the retained campaign's"),
    (lambda r: r["dataset_specific"]["inputs_sha256"].__setitem__("era5_u700_2011_6h_region.nc", "9" * 64),
     "input era5_u700_2011_6h_region.nc does not have the digest the record names"),
    # reached only through the reused producer check
    (lambda r: r["protocol_settings"].__setitem__("smoothing", {"passes": 2}), "record: protocol setting smoothing is not the manifest's"),
    (lambda r: r.__setitem__("stage", "inputs"), "record: stage 'inputs' is not tracking"),
])
def test_each_planted_break_is_named(tmp_path, monkeypatch, change, expected):
    C = _load()
    w, _ = _world(tmp_path, C, monkeypatch)
    _rewrite(w["runs"], 2011, change)
    report = _check(C, w)
    assert not report["passed"] and expected in report["years"]["2011"], report["years"]["2011"]


def test_an_absent_input_a_missing_year_and_another_manifest_are_problems_never_passes(tmp_path, monkeypatch):
    C = _load()
    w, _ = _world(tmp_path, C, monkeypatch)
    os.remove(os.path.join(w["inputs"], "era5_v700_2011_6h_region.nc"))
    report = _check(C, w)
    assert "input era5_v700_2011_6h_region.nc is absent from " + w["inputs"] + ", so its digest is unchecked" in report["years"]["2011"]
    os.rename(os.path.join(w["runs"], "era5_2012"), os.path.join(w["runs"], "moved_aside"))
    report = _check(C, w)
    assert report["years"]["2012"] == [f"no record at {os.path.join(w['runs'], 'era5_2012', 'tracking_era5_2012.json')}"]
    _rewrite(w["runs"], 2011, lambda r: r["protocol_settings"].__setitem__("manifest_sha256", "4" * 64))
    report = _check(C, w)
    assert any("the record was made under manifest 444444444444" in p for p in report["years"]["2011"])


def test_observations_outside_the_year_and_a_run_outside_the_tracking_years_are_refused(tmp_path, monkeypatch):
    from scipy.io import savemat
    C = _load()
    w, _ = _world(tmp_path, C, monkeypatch)
    run = os.path.join(w["runs"], "era5_2011")
    day0 = float((np.datetime64("2011-12-31") - np.datetime64("1900-01-01")).astype(int))
    os.remove(os.path.join(run, "tracker_port.mat"))
    savemat(os.path.join(run, "tracker_port.mat"), {"n": 1.0, "case_id": "f" * 32, "lat0": np.array([10.0, 10.0]),
                                                    "lon0": np.array([5.0, 4.0]), "time0": np.array([day0, day0 + 1.0])})
    _rewrite(w["runs"], 2011, lambda r: r["dataset_specific"].__setitem__(
        "tracks_sha256", _sha(open(os.path.join(run, "tracker_port.mat"), "rb").read())))
    os.makedirs(os.path.join(w["runs"], "era5_2010"))
    report = _check(C, w)
    assert report["years"]["2011"] == ["observations lie outside 2011"]
    assert report["run_directories_not_a_tracking_year"] == ["era5_2010"] and not report["passed"]


def test_a_tracking_record_outside_the_canonical_paths_fails_the_campaign(tmp_path, monkeypatch):
    """A review planted a parseable record under era5_2011_duplicate, which an exact name
    pattern ignored. Every tracking record under the root is inventoried now, so a duplicate
    run, a kept attempt, or a record nested anywhere else is named."""
    import shutil
    C = _load()
    w, _ = _world(tmp_path, C, monkeypatch)
    assert _check(C, w)["passed"]
    shutil.copytree(os.path.join(w["runs"], "era5_2011"), os.path.join(w["runs"], "era5_2011_duplicate"))
    report = _check(C, w)
    assert not report["passed"] and report["years"]["2011"] == []
    assert report["run_directories_not_a_tracking_year"] == ["era5_2011_duplicate"]
    assert report["tracking_records_outside_the_canonical_paths"] == [os.path.join("era5_2011_duplicate", "tracking_era5_2011.json")]
    shutil.rmtree(os.path.join(w["runs"], "era5_2011_duplicate"))
    nested = os.path.join(w["runs"], "era5_2012", "kept")
    os.makedirs(nested)
    shutil.copy(os.path.join(w["runs"], "era5_2012", "tracking_era5_2012.json"), os.path.join(nested, "tracking_era5_2012.json"))
    report = _check(C, w)
    assert not report["passed"] and report["run_directories_not_a_tracking_year"] == []
    assert report["tracking_records_outside_the_canonical_paths"] == [os.path.join("era5_2012", "kept", "tracking_era5_2012.json")]


def test_a_manifest_that_is_not_an_extension_is_refused(tmp_path, monkeypatch):
    C = _load()
    w, _ = _world(tmp_path, C, monkeypatch)
    with pytest.raises(SystemExit, match="not an extension"):
        C.check(w["runs"], BASE, w["retained"], w["inputs"])


def test_a_symbolic_link_under_the_root_fails_the_campaign(tmp_path, monkeypatch):
    """A review reached a duplicate record through runs/kept, a link to a directory outside
    the root, which the inventory did not follow. Any link under the root now fails."""
    import shutil
    C = _load()
    w, _ = _world(tmp_path, C, monkeypatch)
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(os.path.join(w["runs"], "era5_2011"), elsewhere / "era5_2011")
    os.symlink(elsewhere, os.path.join(w["runs"], "kept"))
    report = _check(C, w)
    assert not report["passed"] and report["symbolic_links_under_the_root"] == ["kept"]
    assert report["tracking_records_outside_the_canonical_paths"] == []          # not reached through the link
    os.remove(os.path.join(w["runs"], "kept"))
    os.symlink(os.path.join(w["runs"], "era5_2011", "tracking_era5_2011.json"), os.path.join(w["runs"], "era5_2012", "spare.json"))
    report = _check(C, w)
    assert not report["passed"] and report["symbolic_links_under_the_root"] == [os.path.join("era5_2012", "spare.json")]
