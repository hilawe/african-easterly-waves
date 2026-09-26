"""The campaign collector: a run's small files are copied into evidence with a pointer to
the case left outside git and never overwritten with different bytes, a run without its
record is reported missing, a refused comparison is listed and not dropped, and the
summary copies its values from the per-year artifacts and records each artifact's digest."""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _load():
    path = os.environ.get("COLLECT_SCRIPT", os.path.join(ROOT, "scripts", "collect_protocol_campaign.py"))
    spec = importlib.util.spec_from_file_location("collect_protocol_campaign", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_dir(campaign, dataset, year, with_record=True):
    d = campaign / f"{dataset}_{year}"
    d.mkdir(parents=True)
    (d / "tracker_port.mat").write_bytes(b"tracks-" + dataset.encode())
    (d / "run.log").write_text("log\n")
    (d / "tracker_case.mat").write_bytes(b"big")
    if with_record:
        (d / f"tracking_{dataset}_{year}.json").write_text(json.dumps({"dataset_specific": {"case_sha256": "x", "case_id": f"case-{dataset}-{year}"}}))
    return d


def _artifact(v1, port, months):
    return {"comparison": {"season": {"v1": v1, "port": port, "port_minus_v1": port - v1, "difference_over_published_sd": -0.5,
                                      "distinct_waves": {"v1": 9, "port": 8, "port_minus_v1": -1},
                                      "duplication": {"v1_fraction": 0.3, "port_fraction": 0.31},
                                      "distributions": {"lifetime": {"percentiles": {"50": {"v1": 13, "port": 12}}}}},
                           **{f"month {m}": {"v1": a, "port": b} for m, (a, b) in months.items()},
                           "band -10..+0": {"v1": v1 // 2, "port": port // 2}, "band +0..+10": {"v1": 1, "port": 2}},
            "sides": {"v1": {"case_id": "a"}, "port": {"case_id": "b"}},
            "columns": {k: {"tracks_all": 100, "tracks_in_season_whole_domain": 40,
                            "boundaries": {"crossing_in": {"tracks": 2}, "crossing_out": {"tracks": 1}, "year_end_potentially_censored": {"tracks": 3}}}
                        for k in ("v1", "port")},
            "published_context_column": {"tracks_in_season": 200}}


def test_collect_run_copies_small_files_with_a_pointer_and_never_overwrites_different_bytes(tmp_path):
    C = _load()
    campaign, evidence = tmp_path / "campaign", tmp_path / "evidence"
    _run_dir(campaign, "era5", 1990)
    dst = C.collect_run(str(campaign), str(evidence), "era5", 1990)
    assert sorted(os.listdir(dst)) == ["CASE_LOCATION.txt", "run.log", "tracker_port.mat", "tracking_era5_1990.json"]
    assert "tracker_case.mat" in open(os.path.join(dst, "CASE_LOCATION.txt")).read()
    assert C.collect_run(str(campaign), str(evidence), "era5", 1990) == dst        # a second pass is a no-op
    (campaign / "era5_1990" / "tracker_port.mat").write_bytes(b"changed")
    with pytest.raises(SystemExit):
        C.collect_run(str(campaign), str(evidence), "era5", 1990)
    _run_dir(campaign, "eraint", 1990, with_record=False)
    assert C.collect_run(str(campaign), str(evidence), "eraint", 1990) is None


def test_summary_copies_values_records_digests_and_lists_years_without_a_comparison(tmp_path):
    C = _load()
    a = tmp_path / "paired_1990.json"
    a.write_text(json.dumps(_artifact(135, 125, {6: (31, 23), 7: (29, 31), 8: (40, 38), 9: (35, 33)})))
    summary = C.summarize({1990: (str(a), "published"), 1991: (None, "refused: the gate said no"), 1992: (None, "a run is missing: era5")})
    y = summary["years"]["1990"]
    assert y["africa_origin"] == {"v1": 135, "port": 125, "port_minus_v1": -10, "over_published_sd": -0.5}
    assert y["months"]["6"] == {"v1": 31, "port": 23} and y["distinct_waves"]["port_minus_v1"] == -1
    assert y["boundaries"]["v1"]["year_end_potentially_censored"] == 3 and y["published_context_tracks"] == 200
    import hashlib
    assert y["artifact_sha256"] == hashlib.sha256(a.read_bytes()).hexdigest()
    assert summary["years_without_a_comparison"] == {"1991": "refused: the gate said no", "1992": "a run is missing: era5"}
    # the aggregates are computed from the rows: with one year, means equal the year and spreads are undefined
    ag = summary["aggregates"]
    assert ag["n_years"] == 1 and ag["africa_origin"]["mean"] == {"v1": 135.0, "port": 125.0}
    assert ag["africa_origin"]["interannual_sd"] == {"v1": None, "port": None} and ag["africa_origin"]["pearson_correlation_v1_port"] is None
    assert ag["africa_origin"]["difference_port_minus_v1"]["years_port_lower"] == 1
    assert ag["archive"] == {"years_with_archive": 1, "mean": 200.0, "interannual_sd": None,
                             "pearson_correlation_v1_archive": None, "pearson_correlation_port_archive": None}


def test_aggregates_over_several_years_are_the_plain_statistics(tmp_path):
    C = _load()
    paths = {}
    series = {1990: (100, 90, 150), 1991: (110, 120, 160), 1992: (120, 100, 170), 1993: (130, 140, None)}
    for y, (v1, port, arch) in series.items():
        art = _artifact(v1, port, {6: (1, 1), 7: (1, 1), 8: (1, 1), 9: (1, 1)})
        art["published_context_column"] = {"tracks_in_season": arch} if arch is not None else {}
        p = tmp_path / f"paired_{y}.json"
        p.write_text(json.dumps(art))
        paths[y] = (str(p), "published")
    import numpy as np
    ag = C.summarize(paths)["aggregates"]
    v1, port = np.array([100, 110, 120, 130.0]), np.array([90, 120, 100, 140.0])
    assert ag["africa_origin"]["mean"] == {"v1": 115.0, "port": 112.5}
    assert ag["africa_origin"]["interannual_sd"]["v1"] == pytest.approx(np.std(v1, ddof=1))
    assert ag["africa_origin"]["difference_port_minus_v1"]["mean"] == pytest.approx(-2.5)
    assert ag["africa_origin"]["difference_port_minus_v1"]["sd"] == pytest.approx(np.std(port - v1, ddof=1))
    assert ag["africa_origin"]["difference_port_minus_v1"]["years_port_lower"] == 2
    assert ag["africa_origin"]["pearson_correlation_v1_port"] == pytest.approx(np.corrcoef(v1, port)[0, 1])
    assert ag["archive"]["years_with_archive"] == 3
    assert ag["archive"]["pearson_correlation_v1_archive"] == pytest.approx(np.corrcoef(v1[:3], [150, 160, 170])[0, 1])
    assert ag["archive"]["pearson_correlation_port_archive"] == pytest.approx(np.corrcoef(port[:3], [150, 160, 170])[0, 1])
    assert ag["bands_mean_starts"] == {"-10..+0": {"v1": 57.5, "port": 56.25}, "+0..+10": {"v1": 1.0, "port": 2.0}}
    assert ag["boundaries"]["mean"]["v1"] == {"crossing_in": 2.0, "crossing_out": 1.0, "year_end_potentially_censored": 3.0}
    assert ag["duplication_fraction"]["mean"] == {"v1": pytest.approx(0.3), "port": pytest.approx(0.31)}
    assert ag["season_whole_domain"]["mean"] == {"v1": 40.0, "port": 40.0}


def test_aggregates_with_a_constant_series_and_no_archive_report_no_correlation(tmp_path):
    C = _load()
    paths = {}
    for y in (1990, 1991, 1992, 1993):
        art = _artifact(100, 90 + y - 1990, {6: (1, 1), 7: (1, 1), 8: (1, 1), 9: (1, 1)})
        art["published_context_column"] = {}
        p = tmp_path / f"paired_{y}.json"
        p.write_text(json.dumps(art))
        paths[y] = (str(p), "published")
    ag = C.summarize(paths)["aggregates"]
    assert ag["africa_origin"]["interannual_sd"]["v1"] == 0.0 and ag["africa_origin"]["pearson_correlation_v1_port"] is None
    assert ag["archive"] == {"years_with_archive": 0, "mean": None, "interannual_sd": None,
                             "pearson_correlation_v1_archive": None, "pearson_correlation_port_archive": None}


def _aux_dirs(tmp_path):
    """Region polygons, a published record and a published year file, as small real files,
    with their digests computed here independently of the collector."""
    import hashlib
    import numpy as np
    from scipy.io import savemat
    import season_metrics as S
    regions = tmp_path / "regions"
    regions.mkdir()
    rdig = {}
    for code, name in S.REGION_FILES:
        p = regions / f"{name}.mat"
        savemat(p, {name: np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])})
        rdig[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    record = tmp_path / "record"
    record.mkdir()
    pdig = {}
    for y in (1983, 1984, 2001):
        p = record / f"ERA-Int_ew_700hPa_{y}_AFR.nc"
        p.write_bytes(f"record {y}".encode())
        pdig[str(y)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return str(regions), rdig, str(record), pdig


def _bound_artifact(evidence, manifest, year, rdig=None, pdig=None, published=None):
    import hashlib
    art = _artifact(10, 9, {6: (1, 1), 7: (2, 2), 8: (3, 3), 9: (4, 3)})
    art["year"] = year
    art["region_polygons_sha256"] = rdig or {}
    art["published_record_sha256"] = pdig or {}
    msha = hashlib.sha256(open(manifest, "rb").read()).hexdigest()
    art["sides"] = {side: {"case_id": f"case-{ds}-{year}", "manifest_sha256": msha} for ds, side in (("eraint", "v1"), ("era5", "port"))}
    art["inputs_sha256"] = {side: {"sha256": hashlib.sha256(open(os.path.join(evidence, f"{ds}_{year}", "tracker_port.mat"), "rb").read()).hexdigest()}
                            for ds, side in (("eraint", "v1"), ("era5", "port"))}
    if published is not None:
        art["inputs_sha256"]["published_year_file"] = {"sha256": hashlib.sha256(open(published, "rb").read()).hexdigest()}
    return art


def test_an_existing_artifact_is_reused_only_when_bound_to_the_collected_runs(tmp_path):
    C = _load()
    campaign, evidence, artifacts = tmp_path / "campaign", tmp_path / "evidence", tmp_path / "artifacts"
    artifacts.mkdir()
    for ds in ("eraint", "era5"):
        _run_dir(campaign, ds, 2001)
        C.collect_run(str(campaign), str(evidence), ds, 2001)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    regions, rdig, record, pdig = _aux_dirs(tmp_path)
    published = os.path.join(record, "ERA-Int_ew_700hPa_2001_AFR.nc")
    bound = lambda: _bound_artifact(str(evidence), str(manifest), 2001, rdig, pdig, published)
    out = artifacts / "paired_2001_eraint_era5.json"
    calls = []
    runner = lambda argv: calls.append(argv)
    args = (str(evidence), str(artifacts), str(manifest), 2001, regions, record, record, runner)
    out.write_text(json.dumps(bound()))
    assert C.compare_year(*args) == (str(out), "exists") and calls == []
    # changed polygon bytes, a changed record file, or a published year file that is now absent: refused
    keep = open(os.path.join(regions, "africa.mat"), "rb").read()
    with open(os.path.join(regions, "africa.mat"), "ab") as fh:
        fh.write(b"\0")
    assert "region polygon digests" in C.compare_year(*args)[1]
    open(os.path.join(regions, "africa.mat"), "wb").write(keep)
    assert C.compare_year(*args)[1] == "exists"
    with open(os.path.join(record, "ERA-Int_ew_700hPa_1984_AFR.nc"), "ab") as fh:
        fh.write(b"\0")
    assert "published record digests" in C.compare_year(*args)[1]
    open(os.path.join(record, "ERA-Int_ew_700hPa_1984_AFR.nc"), "wb").write(b"record 1984")
    os.rename(published, published + ".away")
    assert "published year file" in C.compare_year(*args)[1]
    os.rename(published + ".away", published)
    assert C.compare_year(*args)[1] == "exists" and calls == []
    # the same artifact with one side's case id from another run is refused, not reused, not overwritten
    stale = bound()
    stale["sides"]["port"]["case_id"] = "case-era5-1990"
    out.write_text(json.dumps(stale))
    path, status = C.compare_year(*args)
    assert path is None and status.startswith("refused:") and "side port case id" in status and calls == []
    assert json.loads(out.read_text()) == stale
    # a year filed under another year, a foreign manifest, or changed tracks are each refused
    for plant in ({"year": 2002}, {"sides": {**stale["sides"], "v1": {**stale["sides"]["v1"], "manifest_sha256": "0" * 64}}}):
        art = bound()
        art.update(plant)
        out.write_text(json.dumps(art))
        assert C.compare_year(*args)[0] is None
    out.write_text(json.dumps(bound()))
    (evidence / "era5_2001" / "tracker_port.mat").write_bytes(b"other tracks")
    assert "side port input digest" in C.compare_year(*args)[1]


def test_main_runs_the_comparison_per_complete_year_through_the_runner(tmp_path, monkeypatch):
    C = _load()
    campaign, evidence, artifacts = tmp_path / "campaign", tmp_path / "evidence", tmp_path / "artifacts"
    artifacts.mkdir()
    for ds in ("eraint", "era5"):
        _run_dir(campaign, ds, 2001)
    _run_dir(campaign, "eraint", 2002)                       # 2002 lacks its era5 run
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    calls = []

    def fake_runner(argv):
        calls.append(argv)
        out = argv[argv.index("--out") + 1]
        year = int(argv[argv.index("--year") + 1])
        if year == 2001:
            with open(out, "w") as fh:
                json.dump(_artifact(10, 9, {6: (1, 1), 7: (2, 2), 8: (3, 3), 9: (4, 3)}), fh)
            return 0
        raise SystemExit("REFUSED: no")
    import season_metrics
    monkeypatch.setattr(season_metrics, "main", fake_runner)
    summary_path = tmp_path / "summary.json"
    C.main(["--campaign", str(campaign), "--evidence", str(evidence), "--manifest", str(manifest),
            "--artifacts", str(artifacts), "--summary", str(summary_path), "--years", "2001-2002",
            "--regions-dir", str(tmp_path), "--record-dir", str(tmp_path), "--published-dir", str(tmp_path)])
    summary = json.loads(summary_path.read_text())
    assert len(calls) == 1 and "--mode" in calls[0] and calls[0][calls[0].index("--year") + 1] == "2001"
    assert summary["years"]["2001"]["africa_origin"]["port_minus_v1"] == -1
    assert summary["years_without_a_comparison"] == {"2002": "a run is missing: era5"}
    assert summary["runs_collected"] == {"2001": {"eraint": True, "era5": True}, "2002": {"eraint": True, "era5": False}}
    with pytest.raises(SystemExit):                         # the summary is published exclusively
        C.main(["--campaign", str(campaign), "--evidence", str(evidence), "--manifest", str(manifest),
                "--artifacts", str(artifacts), "--summary", str(summary_path), "--years", "2001-2002",
                "--regions-dir", str(tmp_path), "--record-dir", str(tmp_path), "--published-dir", str(tmp_path)])
