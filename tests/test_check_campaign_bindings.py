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
           input_names=("u700", "v700"), calibration_digest=None, case_bytes=b"the retained case", case_present=True):
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
    campaign = tmp_path / "campaign" / f"{dataset}_{year}"
    campaign.mkdir(parents=True, exist_ok=True)
    if case_present:
        (campaign / "tracker_case.mat").write_bytes(case_bytes)
    record = {"stage": "tracking", "dataset_specific": {
        "dataset": dataset, "year": record_year if record_year is not None else year, "case_id": case,
        "case_sha256": hashlib.sha256(b"the retained case").hexdigest(),
        "tracks_sha256": _sha(run / "tracker_port.mat"), "inputs_sha256": inputs,
        "calibration_sha256": calibration_digest or _sha(cal),
        "coarse_threshold": 4e-7, "fine_threshold": record_fine if record_fine is not None else 2e-6}}
    (run / f"tracking_{dataset}_{year}.json").write_text(json.dumps(record))
    manifest = {"datasets": {dataset: {"directory": str(raw_dir), "prefix": dataset}}}
    return manifest, str(tmp_path / "evidence"), str(cal), record


def _run(C, tmp_path, evidence, manifest, cal, dataset="era5", year=1994, validator=lambda p, y, v: None):
    return C.check_run(evidence, dataset, year, manifest, "m" * 64, validator, lambda ds: cal, str(tmp_path / "campaign"))


def _stub_producer(monkeypatch):
    import season_metrics
    monkeypatch.setattr(season_metrics, "producer_problems", lambda *a, **k: [])


def _failed(links):
    return [l["link"] for l in links if not l["passed"]]


def test_a_consistent_run_is_bound_and_every_link_is_recorded(tmp_path, monkeypatch):
    C = _load()
    _stub_producer(monkeypatch)
    manifest, evidence, cal, _ = _world(tmp_path)
    links = _run(C, tmp_path, evidence, manifest, cal)
    assert _failed(links) == [] and len(links) == 17
    assert all("link" in l and "passed" in l for l in links)
    assert sum(1 for l in links if l.get("raw_validation_expected")) == 2 == sum(1 for l in links if l.get("raw_validator_called"))
    # without a campaign root the case is not read, and that is a failed link, never a pass
    links = C.check_run(evidence, "era5", 1994, manifest, "m" * 64, lambda p, y, v: None, lambda ds: cal)
    assert _failed(links) == ["retained case present with the record's digest"]


@pytest.mark.parametrize("plant, expected", [
    ({"record_year": 1995}, "record filed under its dataset and year (the driver's own predicate)"),
    ({"record_year": "1994"}, "record filed under its dataset and year (the driver's own predicate)"),
    ({"times": np.array([DAY_1994 + 364.75, DAY_1994 + 365.0])}, "every observation falls inside the calendar year"),
    ({"times": np.array([DAY_1994 - 0.25, DAY_1994])}, "every observation falls inside the calendar year"),
    ({"case_in_mat": "d" * 32}, "tracks file carries the record's case id"),
    ({"declared_n": 2}, "tracks file holds the number of tracks it declares, contiguously indexed, each with time, lat and lon of one length"),
    ({"extra_keys": ("time2",)}, "tracks file holds the number of tracks it declares, contiguously indexed, each with time, lat and lon of one length"),
    ({"declared_n": 3, "extra_keys": ("time2",)}, "tracks file holds the number of tracks it declares, contiguously indexed, each with time, lat and lon of one length"),
    ({"raw_digest_ok": False}, "raw input era5_u700_1994_6h_region.nc digest equals the record's"),
    ({"embedded_year": 1993}, "embedded producer record names the same dataset, year and raw inputs"),
    ({"record_fine": 3e-6}, "record thresholds equal the calibration artifact's"),
    ({"input_names": ("u700", "u700x")}, "record names exactly the year's two canonical input files"),
    ({"input_names": ("u700",)}, "record names exactly the year's two canonical input files"),
    ({"calibration_digest": "0" * 64}, "calibration digest equals the record's"),
    ({"case_bytes": b"another case"}, "retained case present with the record's digest"),
    ({"case_present": False}, "retained case present with the record's digest"),
    ({"drop_keys": ("lat0",)}, "tracks file holds the number of tracks it declares, contiguously indexed, each with time, lat and lon of one length"),
    ({"declared_n": 0, "drop_keys": ("time0", "lat0", "lon0")}, "tracks file holds at least one track"),
])


def test_each_planted_break_is_named_as_the_failed_link(tmp_path, monkeypatch, plant, expected):
    C = _load()
    _stub_producer(monkeypatch)
    manifest, evidence, cal, _ = _world(tmp_path, **plant)
    links = _run(C, tmp_path, evidence, manifest, cal)
    assert expected in _failed(links)


def test_two_copies_of_one_component_never_count_as_both_validated(tmp_path, monkeypatch):
    C = _load()
    _stub_producer(monkeypatch)
    manifest, evidence, cal, _ = _world(tmp_path, input_names=("u700", "u700x"))     # no v700 file at all
    calls = []
    links = _run(C, tmp_path, evidence, manifest, cal, validator=lambda p, y, v: calls.append((os.path.basename(p), v)))
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
    links = _run(C, tmp_path, evidence, manifest, cal)
    assert _failed(links) == ["record passes the comparison gate's producer check"]
    assert [l for l in links if not l["passed"]][0]["detail"] == "stage is not tracking; manifest digest differs"
    assert seen == [(1994, "port")]                                    # the gate is asked about this year and this side


def test_both_ends_of_the_year_and_a_leap_year_are_accepted(tmp_path, monkeypatch):
    import datetime as dt
    C = _load()
    _stub_producer(monkeypatch)
    day = lambda y, m, d: float((dt.date(y, m, d) - dt.date(1900, 1, 1)).days)
    manifest, evidence, cal, _ = _world(tmp_path, year=1994, times=np.array([DAY_1994, day(1994, 12, 31) + 0.75]))
    assert _failed(_run(C, tmp_path, evidence, manifest, cal)) == []
    manifest, evidence, cal, _ = _world(tmp_path / "leap", year=1992, times=np.array([day(1992, 2, 29), day(1992, 12, 31) + 0.75]))
    assert _failed(_run(C, tmp_path / "leap", evidence, manifest, cal, year=1992)) == []


def test_a_stray_index_never_crashes_the_check_and_is_named(tmp_path, monkeypatch):
    C = _load()
    _stub_producer(monkeypatch)
    manifest, evidence, cal, _ = _world(tmp_path, declared_n=3, extra_keys=("time2",))
    links = _run(C, tmp_path, evidence, manifest, cal)
    assert "tracks file holds the number of tracks it declares, contiguously indexed, each with time, lat and lon of one length" in _failed(links)
    assert "every observation falls inside the calendar year" in _failed(links)     # the stray array is read, not skipped


def test_a_refused_raw_input_or_a_changed_tracks_file_is_not_bound(tmp_path, monkeypatch):
    C = _load()
    _stub_producer(monkeypatch)
    manifest, evidence, cal, _ = _world(tmp_path)
    links = _run(C, tmp_path, evidence, manifest, cal, validator=lambda p, y, v: "time vector does not cover the year" if v == "v700" else None)
    assert _failed(links) == ["raw input era5_v700_1994_6h_region.nc passes the retrieval validator for 1994 as v700"]
    with open(os.path.join(evidence, "era5_1994", "tracker_port.mat"), "ab") as fh:
        fh.write(b"\0")
    links = _run(C, tmp_path, evidence, manifest, cal)
    assert "tracks digest equals the record's" in _failed(links)


def _year_world(tmp_path, monkeypatch):
    """Two consistent runs, their auxiliary inputs as small real files, a per-year artifact
    bound to all of them and to the current instrument, and the summary row the collector
    would build from it. Returns what check_year takes plus the artifact path."""
    import collect_protocol_campaign as C
    import season_metrics as S
    import exact_tracks as X
    from scipy.io import savemat
    _stub_producer(monkeypatch)
    manifest = {"datasets": {}}
    for ds in ("eraint", "era5"):
        m, evidence, cal, rec = _world(tmp_path, ds)
        manifest["datasets"].update(m["datasets"])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    regions = tmp_path / "regions"
    regions.mkdir()
    for code, name in S.REGION_FILES:
        savemat(regions / f"{name}.mat", {name: np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])})
    record = tmp_path / "record"
    record.mkdir()
    for y in (1983, 1994):
        (record / f"ERA-Int_ew_700hPa_{y}_AFR.nc").write_bytes(f"record {y}".encode())
    rdig, pdig, published = C.auxiliary_digests(str(regions), str(record), str(record), 1994)
    ids = {ds: json.load(open(os.path.join(evidence, f"{ds}_1994", f"tracking_{ds}_1994.json")))["dataset_specific"]["case_id"]
           for ds in ("eraint", "era5")}
    tracks = {ds: _sha(os.path.join(evidence, f"{ds}_1994", "tracker_port.mat")) for ds in ("eraint", "era5")}
    art = {"year": 1994, "generated_by": "scripts/season_metrics.py", "script_sha256": X.digest(S.__file__),
           "sides": {"v1": {"case_id": ids["eraint"], "dataset": "eraint", "manifest_sha256": _sha(manifest_path)},
                     "port": {"case_id": ids["era5"], "dataset": "era5", "manifest_sha256": _sha(manifest_path)}},
           "inputs_sha256": {"v1": {"sha256": tracks["eraint"]}, "port": {"sha256": tracks["era5"]},
                             "published_year_file": {"sha256": published}},
           "region_polygons_sha256": rdig, "published_record_sha256": pdig,
           "comparison": {"season": {"v1": 135, "port": 125, "port_minus_v1": -10, "difference_over_published_sd": -0.57,
                                     "distinct_waves": {"v1": 93, "port": 84, "port_minus_v1": -9},
                                     "duplication": {"v1_fraction": 0.33, "port_fraction": 0.32},
                                     "distributions": {"lifetime": {"percentiles": {"50": {"v1": 13, "port": 12}}}}},
                          **{f"month {m}": {"v1": 30, "port": 29} for m in (6, 7, 8, 9)},
                          "band +0..+10": {"v1": 19, "port": 10}},
           "columns": {k: {"tracks_all": 1600, "tracks_in_season_whole_domain": 540,
                           "boundaries": {"crossing_in": {"tracks": 26}, "crossing_out": {"tracks": 14}, "year_end_potentially_censored": {"tracks": 44}}}
                       for k in ("v1", "port")},
           "published_context_column": {"tracks_in_season": 201}}
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    path = artifacts / "paired_1994_eraint_era5.json"
    path.write_text(json.dumps(art))
    rows = {"1994": C.row_from_artifact(str(path))}
    args = (evidence, str(artifacts), rows, 1994, str(manifest_path), str(regions), str(record), str(record))
    return args, path, art


def test_a_year_is_bound_when_the_artifact_and_the_row_agree_with_everything(tmp_path, monkeypatch):
    C = _load()
    args, path, art = _year_world(tmp_path, monkeypatch)
    assert _failed(C.check_year(*args)) == []


def test_every_copied_row_field_is_bound_not_only_the_two_counts(tmp_path, monkeypatch):
    C = _load()
    args, path, art = _year_world(tmp_path, monkeypatch)
    rows = args[2]
    good = json.loads(json.dumps(rows["1994"]))
    for field, value in (("months", {"6": {"v1": 30, "port": 23}, "7": {"v1": 30, "port": 29}, "8": {"v1": 30, "port": 29}, "9": {"v1": 30, "port": 29}}),
                         ("distinct_waves", {"v1": 93, "port": 85, "port_minus_v1": -8}),
                         ("boundaries", {"v1": {"crossing_in": 0, "crossing_out": 14, "year_end_potentially_censored": 44}, "port": good["boundaries"]["port"]}),
                         ("published_context_tracks", 200),
                         ("artifact_sha256", "0" * 64)):
        rows["1994"] = json.loads(json.dumps(good))
        rows["1994"][field] = value
        failed = [l for l in C.check_year(*args) if not l["passed"]]
        assert [l["link"] for l in failed] == ["summary row equals the row rebuilt from the artifact, every copied field"], field
        assert field in failed[0]["detail"]
    rows["1994"] = good
    assert _failed(C.check_year(*args)) == []
    # a missing key is not a present null: with no archive count the field is null in the
    # artifact's row, and a row that lacks the key entirely is still named (the 1979 case)
    import collect_protocol_campaign as C0
    art2 = json.loads(json.dumps(art))
    art2["published_context_column"] = {}
    path.write_text(json.dumps(art2))
    null_row = C0.row_from_artifact(str(path))
    assert null_row["published_context_tracks"] is None
    rows["1994"] = json.loads(json.dumps(null_row))
    assert _failed(C.check_year(*args)) == []
    del rows["1994"]["published_context_tracks"]
    failed = [l for l in C.check_year(*args) if not l["passed"]]
    assert [l["link"] for l in failed] == ["summary row equals the row rebuilt from the artifact, every copied field"]
    assert "published_context_tracks" in failed[0]["detail"]
    path.write_text(json.dumps(art))
    rows["1994"] = good


def test_a_stale_instrument_a_swapped_side_and_a_wrong_year_are_named_through_the_collectors_check(tmp_path, monkeypatch):
    import collect_protocol_campaign as C0
    C = _load()
    args, path, art = _year_world(tmp_path, monkeypatch)
    for plant, expect_in_detail in (({"script_sha256": "0" * 64}, "another revision of the instrument"),
                                    ({"sides": {**art["sides"], "port": {**art["sides"]["port"], "case_id": "case-eraint".ljust(32, "0")}}}, "side port case id"),
                                    ({"year": 1993}, "artifact year 1993")):
        a = json.loads(json.dumps(art))
        a.update(plant)
        path.write_text(json.dumps(a))
        args[2]["1994"] = C0.row_from_artifact(str(path))
        failed = [l for l in C.check_year(*args) if not l["passed"]]
        names = [l["link"] for l in failed]
        assert any(l["link"].startswith("artifact is bound to the collected runs") and expect_in_detail in l["detail"] for l in failed), (plant, names)
        if "year" in plant:
            assert "artifact names the year" in names
    path.write_bytes(b"{not json")
    assert _failed(C.check_year(*args)) == ["paired artifact readable"]
    for blob in (b"null", b"[]", b'{"comparison": null, "sides": {}}'):
        path.write_bytes(blob)
        assert _failed(C.check_year(*args)) == ["paired artifact is an artifact object with comparison and sides"], blob
    a = json.loads(json.dumps(art))
    a["sides"]["port"] = ["bad"]                                            # corruption below the top level
    path.write_text(json.dumps(a))
    failed = [l for l in C.check_year(*args) if not l["passed"]]
    assert any(l["link"].startswith("artifact is bound to the collected runs") and "structure defeated" in l["detail"] for l in failed)


def test_run_check_refuses_an_empty_range_and_never_reports_unperformed_validation_as_done(tmp_path, monkeypatch):
    C = _load()
    args, path, art = _year_world(tmp_path, monkeypatch)
    evidence, artifacts, rows, _, manifest_path, regions, record, published = args
    manifest = json.load(open(manifest_path))
    cal = lambda ds: str(tmp_path / f"thresholds_protocol_{ds}_1979_2010.json")
    common = dict(manifest_path=manifest_path, regions_dir=regions, record_dir=record, published_dir=published, campaign=str(tmp_path / "campaign"))
    with pytest.raises(SystemExit):
        C.run_check([], manifest, "m" * 64, evidence, artifacts, rows, lambda p, y, v: None, cal, **common)
    calls = []
    ok = C.run_check([1994], manifest, "m" * 64, evidence, artifacts, rows, lambda p, y, v: calls.append((p, y, v)), cal, **common)
    assert ok["verdict"] == "bound" and ok["raw_validator_calls"] == 4 == ok["raw_validations_needed"] == len(calls)
    assert ok["raw_inputs_validated"] is True and ok["links_failed"] == 0
    assert sorted(v for _, _, v in calls) == ["u700", "u700", "v700", "v700"]
    # a year whose records are absent: nothing validated, nothing bound, and the flag says so
    calls.clear()
    no = C.run_check([1995], manifest, "m" * 64, evidence, artifacts, rows, lambda p, y, v: calls.append(p), cal, **common)
    assert no["verdict"] == "NOT BOUND" and no["raw_inputs_validated"] is False and no["raw_validator_calls"] == 0 == len(calls)
    assert no["raw_validations_needed"] == 4                                 # owed for the run that was never reached
    assert no["links_failed"] >= 3 and all(f["where"] for f in no["failed"])
    # one complete year beside one missing year: the need counts both, so the flag cannot say validated
    calls.clear()
    mixed = C.run_check([1994, 1995], manifest, "m" * 64, evidence, artifacts, rows, lambda p, y, v: calls.append(p), cal, **common)
    assert mixed["raw_validations_needed"] == 8 and mixed["raw_validator_calls"] == 4 == len(calls) and mixed["raw_inputs_validated"] is False
    assert mixed["verdict"] == "NOT BOUND" and mixed["years_checked"] == 2
    # the summary's aggregates are bound to its rows when given
    import collect_protocol_campaign as C0
    agg = C0.aggregates(rows)
    ok2 = C.run_check([1994], manifest, "m" * 64, evidence, artifacts, rows, lambda p, y, v: None, cal, summary_aggregates=agg, **common)
    assert ok2["verdict"] == "bound" and ok2["years_detail"]["summary"][0]["passed"]
    agg["africa_origin"]["mean"]["port"] += 1
    bad = C.run_check([1994], manifest, "m" * 64, evidence, artifacts, rows, lambda p, y, v: None, cal, summary_aggregates=agg, **common)
    assert bad["verdict"] == "NOT BOUND" and bad["failed"][0]["link"] == "summary aggregates equal those recomputed from its rows"
    # as the command runs it: a summary without an aggregates object is a failed link, never a skip
    none = C.run_check([1994], manifest, "m" * 64, evidence, artifacts, rows, lambda p, y, v: None, cal, summary_aggregates=None, bind_aggregates=True, **common)
    assert none["verdict"] == "NOT BOUND" and none["failed"][0]["link"] == "summary carries an aggregates object"
    # a row the aggregation cannot read is a failed link and the report is still published
    broken = json.loads(json.dumps(rows))
    del broken["1994"]["published_context_tracks"]
    rep = C.run_check([1994], manifest, "m" * 64, evidence, artifacts, broken, lambda p, y, v: None, cal, summary_aggregates=C0.aggregates(rows), bind_aggregates=True, **common)
    names = [f["link"] for f in rep["failed"]]
    assert rep["verdict"] == "NOT BOUND" and "summary row equals the row rebuilt from the artifact, every copied field" in names
    assert any(n == "summary aggregates equal those recomputed from its rows" and "could not be aggregated" in f["detail"] for n, f in zip(names, rep["failed"]))
    # malformed values of several kinds reach the report as failed links and never as a lost report
    for field, value in (("africa_origin", {"v1": 10 ** 400, "port": 125, "port_minus_v1": -10, "over_published_sd": None}),
                         ("months", "not a mapping"), ("distinct_waves", None), ("boundaries", [])):
        broken = json.loads(json.dumps(rows))
        broken["1994"][field] = value
        rep = C.run_check([1994], manifest, "m" * 64, evidence, artifacts, broken, lambda p, y, v: None, cal,
                          summary_aggregates=C0.aggregates(rows), bind_aggregates=True, **common)
        names = [f["link"] for f in rep["failed"]]
        assert rep["verdict"] == "NOT BOUND", field
        assert "summary row equals the row rebuilt from the artifact, every copied field" in names, field
        assert "summary aggregates equal those recomputed from its rows" in names, field
    # a summary whose rows are not objects still yields a published report with named links
    for bad_rows in ({"1994": 42}, {"1994": ["artifact"]}, [], None):
        rep = C.run_check([1994], manifest, "m" * 64, evidence, artifacts, bad_rows, lambda p, y, v: None, cal,
                          summary_aggregates=C0.aggregates(rows), bind_aggregates=True, **common)
        names = [f["link"] for f in rep["failed"]]
        assert rep["verdict"] == "NOT BOUND" and "summary row present and an object" in names, bad_rows


@pytest.mark.parametrize("blob", [b"{}", b"[]", b"null", b"{not json", b'{"years": 7, "aggregates": []}', b"<directory>"])
def test_the_command_publishes_a_not_bound_report_for_a_malformed_summary(tmp_path, monkeypatch, blob):
    """Through main, with the prerequisites stubbed: whatever the summary file holds, the
    report is published, says NOT BOUND, names the summary links, and the command exits 1."""
    C = _load()
    import export_protocol_case as E
    args, path, art = _year_world(tmp_path, monkeypatch)
    evidence, artifacts, rows, _, manifest_path, regions, record, published = args
    monkeypatch.setattr(E, "load_manifest", lambda p: (json.load(open(p)), "m" * 64))
    monkeypatch.setattr(C, "real_raw_validator", lambda m: (lambda p, y, v: None))
    summary = tmp_path / "summary.json"
    if blob == b"<directory>":
        summary.mkdir()
    else:
        summary.write_bytes(blob)
    out = tmp_path / "bindings.json"
    rc = C.main(["--manifest", str(manifest_path), "--evidence", evidence, "--artifacts", artifacts, "--summary", str(summary),
                 "--out", str(out), "--years", "1994-1994", "--calibration-dir", str(tmp_path), "--campaign", str(tmp_path / "campaign"),
                 "--regions-dir", regions, "--record-dir", record, "--published-dir", published])
    assert rc == 1 and out.exists()
    report = json.load(open(out))
    names = [f["link"] for f in report["failed"]]
    assert report["verdict"] == "NOT BOUND" and "summary row present and an object" in names
    assert report["summary_sha256"] is None if blob in (b"<directory>",) else len(report["summary_sha256"]) == 64
    assert "summary carries an aggregates object" in names or "summary aggregates equal those recomputed from its rows" in names


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
