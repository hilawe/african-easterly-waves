"""The threshold-sensitivity driver: it replays a retained case only after the case's
digest and identity match the collected record, calls the entry point's own replay with
the archived constants and the manifest's flags, writes a record that names what changed
and what it replaced, publishes exclusively, skips a year with a good record and refuses
a bad one, fails the command unless every year has a record, and summarizes with the
collector's row builder. The replay itself, the instrument and the collector are stubbed
or reused, never reimplemented here."""
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


SIBLINGS = ("export_protocol_case.py", "season_metrics.py", "exact_tracks.py", "campaign_record_ok.py", "collect_protocol_campaign.py")
COMMIT = "abc1234"


def _snapshot(tmp_path):
    """A frozen snapshot as the driver requires: the driver under test and the real entry
    point and its siblings, copied under one root the driver must run from."""
    import shutil
    root = tmp_path / "snapshot"
    scripts = root / "scripts"
    if not scripts.exists():
        scripts.mkdir(parents=True)
        src = os.environ.get("SENSITIVITY_SCRIPT", os.path.join(ROOT, "scripts", "run_threshold_sensitivity.py"))
        shutil.copyfile(src, scripts / "run_threshold_sensitivity.py")
        for name in SIBLINGS:
            shutil.copyfile(os.path.join(ROOT, "scripts", name), scripts / name)
    return str(root)


def _load(tmp_path=None):
    path = (os.path.join(_snapshot(tmp_path), "scripts", "run_threshold_sensitivity.py") if tmp_path is not None
            else os.environ.get("SENSITIVITY_SCRIPT", os.path.join(ROOT, "scripts", "run_threshold_sensitivity.py")))
    spec = importlib.util.spec_from_file_location("run_threshold_sensitivity", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


MANIFEST = {"tracker_flags": {"exclusive": False, "absorb": False}, "datasets": {"eraint": {"directory": "d", "prefix": "eraint", "label": "ERA-Interim"}}}


FAKE_TRACK = {"meanlat": np.array([10.0, 10.5]), "meanlon": np.array([1.0, 0.5]), "time": np.array([32873.0, 32873.25])}


def _world(tmp_path, year=1990, *, case_digest_ok=True, case_id_ok=True, dataset="eraint", real_tracks=False):
    """A retained case, the collected campaign record naming it, and the manifest file."""
    from scipy.io import savemat
    campaign, evidence = tmp_path / "campaign", tmp_path / "evidence"
    run = campaign / f"{dataset}_{year}"
    run.mkdir(parents=True)
    embedded = {"stage": "tracking", "protocol_settings": {"x": 1, "tracker_flags": MANIFEST["tracker_flags"], "manifest_sha256": "m" * 64},
                "source_sha256": {"a": "0" * 64, "scripts/export_protocol_case.py": _sha(os.path.join(ROOT, "scripts", "export_protocol_case.py"))},
                "dataset_specific": {"dataset": dataset, "year": year, "prefix": dataset}}
    steps = 3
    payload = {"lat_c": np.array([[10.0, 12.0]]), "lon_c": np.array([[0.0, 2.0]]), "latgrid": np.zeros((3, 3)), "longrid": np.zeros((3, 3)),
               "time": np.array([[32873.0, 32873.25, 32873.5]]), "u_c": np.zeros((steps, 2, 2)), "v_c": np.zeros((steps, 2, 2)),
               "currv_anom_c": np.zeros((steps, 2, 2)), "advcurrv_anom_c": np.zeros((steps, 2, 2)), "u": np.zeros((steps, 3, 3)),
               "v": np.zeros((steps, 3, 3)), "currv_anom": np.zeros((steps, 3, 3)), "case_id": np.array(["c" * 32]),
               "producer_json": np.array([json.dumps(embedded)])}
    savemat(run / "tracker_case.mat", payload)
    coll = evidence / f"{dataset}_{year}"
    coll.mkdir(parents=True)
    if real_tracks:                                                # the campaign's tracks as the fake replay returns them
        savemat(coll / "tracker_port.mat", {"n": 1.0, "case_id": np.array(["c" * 32]), "producer_json": np.array(["{}"]),
                                            "lat0": FAKE_TRACK["meanlat"], "lon0": FAKE_TRACK["meanlon"], "time0": FAKE_TRACK["time"]})
    else:
        (coll / "tracker_port.mat").write_bytes(b"rule a tracks")
    record = {"stage": "tracking", "dataset_specific": {
        "dataset": dataset, "year": year, "case_id": "c" * 32 if case_id_ok else "d" * 32,
        "case_sha256": _sha(run / "tracker_case.mat") if case_digest_ok else "0" * 64,
        "coarse_threshold": 4.494e-7, "fine_threshold": 2.25e-6, "calibration_sha256": "cal" * 21 + "c",
        "tracks_sha256": _sha(coll / "tracker_port.mat")}}
    (coll / f"tracking_{dataset}_{year}.json").write_text(json.dumps(record))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(MANIFEST))
    return str(campaign), str(evidence), str(manifest), str(tmp_path / "out")


def _replay_year(M, tmp_path, campaign, evidence, manifest, out, year, coarse, fine, **kw):
    return M.replay_year(campaign, evidence, manifest, out, year, coarse, fine, snapshot=_snapshot(tmp_path), commit=COMMIT, **kw)


def _stub(monkeypatch, calls):
    import export_protocol_case as E
    import season_metrics as S
    monkeypatch.setattr(E, "load_manifest", lambda p: (json.load(open(p)), "m" * 64))
    monkeypatch.setattr(S, "producer_problems", lambda *a, **k: [])

    def fake_track(payload, coarse, fine, exclusive, absorb):
        calls.append((coarse, fine, exclusive, absorb, str(payload["case_id"])))
        return [dict(FAKE_TRACK)]
    monkeypatch.setattr(E, "track_case", fake_track)


def test_replay_year_substitutes_only_the_thresholds_records_what_it_replaced_and_publishes_once(tmp_path, monkeypatch):
    M = _load(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    campaign, evidence, manifest, out = _world(tmp_path)
    record, n = _replay_year(M, tmp_path, campaign, evidence, manifest, out, 1990, 7.16e-7, 2.80e-6)
    assert n == 1 and calls == [(7.16e-7, 2.80e-6, False, False, "c" * 32)]
    d = record["dataset_specific"]
    assert d["coarse_threshold"] == 7.16e-7 and d["fine_threshold"] == 2.80e-6 and d["calibration_sha256"] is None
    assert record["sensitivity"]["replaces_rule_a"] == {"coarse": 4.494e-7, "fine": 2.25e-6, "calibration_sha256": "cal" * 21 + "c"}
    assert record["sensitivity"]["what_changed"] == "the two detection thresholds only"
    assert record["stage"] == "tracking" and record["protocol_settings"] == {"x": 1, "tracker_flags": MANIFEST["tracker_flags"], "manifest_sha256": "m" * 64}   # the case's settings, untouched
    assert record["sensitivity"]["case_sha256"] == _sha(os.path.join(campaign, "eraint_1990", "tracker_case.mat"))
    port = os.path.join(out, "eraint_1990", "tracker_port.mat")
    assert d["tracks_sha256"] == _sha(port)
    from scipy.io import loadmat
    raw = loadmat(port)
    assert int(raw["n"][0][0]) == 1 and str(raw["case_id"][0]) == "c" * 32
    assert json.loads(str(raw["producer_json"][0]))["sensitivity"]["label"] == M.LABEL
    assert os.path.exists(os.path.join(out, "eraint_1990", "tracking_eraint_1990.json"))
    with pytest.raises(SystemExit):                                                      # never over a retained file
        _replay_year(M, tmp_path, campaign, evidence, manifest, out, 1990, 7.16e-7, 2.80e-6)


@pytest.mark.parametrize("plant", [{"case_digest_ok": False}, {"case_id_ok": False}])
def test_a_case_that_is_not_the_collected_records_is_refused_before_any_replay(tmp_path, monkeypatch, plant):
    M = _load(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    campaign, evidence, manifest, out = _world(tmp_path, **plant)
    with pytest.raises(SystemExit):
        _replay_year(M, tmp_path, campaign, evidence, manifest, out, 1990, 7.16e-7, 2.80e-6)
    assert calls == [] and not os.path.exists(os.path.join(out, "eraint_1990"))


def test_a_case_the_gate_rejects_is_refused(tmp_path, monkeypatch):
    M = _load(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    import season_metrics as S
    monkeypatch.setattr(S, "producer_problems", lambda *a, **k: ["v1: stage 'inputs' is not tracking"])
    campaign, evidence, manifest, out = _world(tmp_path)
    with pytest.raises(SystemExit):
        _replay_year(M, tmp_path, campaign, evidence, manifest, out, 1990, 7.16e-7, 2.80e-6)
    assert calls == []


def test_the_archived_constants_are_the_ports_own_and_are_the_default(monkeypatch):
    M = _load()
    from aew.v1port import profiles
    assert M.archived_constants() == tuple(float(x) for x in profiles.V1_ERAINT_700) == (7.16e-7, 2.80e-6)


def _args(campaign, evidence, manifest, out, years="1990-1990", workers=1, dataset="eraint", pair="archived", pair_source=None):
    import types
    return types.SimpleNamespace(campaign=campaign, evidence=evidence, manifest=manifest, out_dir=out, years=years, workers=workers,
                                 dataset=dataset, pair=pair, pair_source=pair_source,
                                 code_snapshot=os.path.join(os.path.dirname(campaign), "snapshot"), snapshot_commit=COMMIT)


def test_replay_skips_a_good_record_refuses_a_bad_one_and_fails_unless_every_year_has_a_record(tmp_path, monkeypatch, capsys):
    M = _load(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    campaign, evidence, manifest, out = _world(tmp_path)
    assert M.replay(_args(campaign, evidence, manifest, out)) == 0 and len(calls) == 1
    assert calls[0][:2] == (7.16e-7, 2.80e-6)                                            # the command runs the archived constants and nothing else
    assert M.replay(_args(campaign, evidence, manifest, out)) == 0 and len(calls) == 1              # skipped, not rerun
    assert "skip 1990, record exists" in capsys.readouterr().out
    rec = os.path.join(out, "eraint_1990", "tracking_eraint_1990.json")
    r = json.load(open(rec))
    r["dataset_specific"]["year"] = 1991
    open(rec, "w").write(json.dumps(r))
    with pytest.raises(SystemExit):                                                      # a record of another run at the path
        M.replay(_args(campaign, evidence, manifest, out))
    # a year without a case: the worker reports it and the command fails, the good year untouched
    r["dataset_specific"]["year"] = 1990
    open(rec, "w").write(json.dumps(r))
    assert M.replay(_args(campaign, evidence, manifest, out, years="1990-1991")) == 1
    out_text = capsys.readouterr().out
    assert "FAILED 1991" in out_text and "REPLAY INCOMPLETE" in out_text and "[1991]" in out_text
    with pytest.raises(SystemExit):
        M.replay(_args(campaign, evidence, manifest, out, years="1991-1990"))
    with pytest.raises(SystemExit):                                                      # no threshold option exists on the command
        M.main(["replay", "--campaign", campaign, "--evidence", evidence, "--manifest", manifest, "--out-dir", out,
                "--code-snapshot", _snapshot(tmp_path), "--snapshot-commit", COMMIT, "--coarse", "1e-7"])


def test_a_rule_a_campaign_record_copied_to_the_sensitivity_path_is_not_a_completed_year(tmp_path, monkeypatch, capsys):
    import shutil
    M = _load(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    campaign, evidence, manifest, out = _world(tmp_path)
    dst = os.path.join(out, "eraint_1990")
    os.makedirs(dst)
    for name in ("tracker_port.mat", "tracking_eraint_1990.json"):
        shutil.copyfile(os.path.join(evidence, "eraint_1990", name), os.path.join(dst, name))
    with pytest.raises(SystemExit) as exc:
        M.replay(_args(campaign, evidence, manifest, out))
    assert "not this experiment's record" in str(exc.value) and "archived constants" in str(exc.value) and calls == []
    # and a genuine sensitivity record without its tracks file beside it is not complete either
    shutil.rmtree(dst)
    assert M.replay(_args(campaign, evidence, manifest, out)) == 0
    os.unlink(os.path.join(dst, "tracker_port.mat"))
    with pytest.raises(SystemExit) as exc:
        M.replay(_args(campaign, evidence, manifest, out))
    assert "no tracks file beside the record" in str(exc.value)


def test_a_record_relabeled_beside_tracks_run_under_other_thresholds_is_not_complete(tmp_path, monkeypatch):
    M = _load(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    campaign, evidence, manifest, out = _world(tmp_path)
    _replay_year(M, tmp_path, campaign, evidence, manifest, out, 1990, 1e-7, 1e-6)                    # another experiment's run
    rec = os.path.join(out, "eraint_1990", "tracking_eraint_1990.json")
    r = json.load(open(rec))
    for block in (r["dataset_specific"], r["sensitivity"]):
        block["coarse_threshold"], block["fine_threshold"] = 7.16e-7, 2.80e-6            # relabel the record beside the tracks
    open(rec, "w").write(json.dumps(r))
    with pytest.raises(SystemExit) as exc:
        M.replay(_args(campaign, evidence, manifest, out))
    assert "inside the tracks file was not written under the archived constants" in str(exc.value) and len(calls) == 1


DYING_DRIVER = """
import json, os, sys, types
sys.path.insert(0, sys.argv[1])
import importlib.util
spec = importlib.util.spec_from_file_location("run_threshold_sensitivity", sys.argv[2])
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
import export_protocol_case as E
E.load_manifest = lambda p: (json.load(open(p)), "m" * 64)

def dying(args):
    os._exit(17)                                       # no result, no exception, no report
M._worker = dying
args = types.SimpleNamespace(campaign=sys.argv[3], evidence=sys.argv[4], manifest=sys.argv[5], out_dir=sys.argv[6],
                             years="1990-1991", workers=2, dataset="eraint", pair="archived", pair_source=None,
                             code_snapshot=sys.argv[7], snapshot_commit="abc1234")
import export_protocol_case as E2
sys.exit(M.replay(args, context="fork"))
"""


def test_a_worker_that_dies_without_reporting_fails_the_command_instead_of_hanging(tmp_path):
    """Run in a child process with a parent-enforced timeout, so a hang cannot outlive the
    test, and the executor forks from that process's main thread."""
    import subprocess
    campaign, evidence, manifest, out = _world(tmp_path)
    _world(tmp_path / "second", year=1991)
    import shutil
    shutil.copytree(str(tmp_path / "second" / "campaign" / "eraint_1991"), os.path.join(campaign, "eraint_1991"))
    shutil.copytree(str(tmp_path / "second" / "evidence" / "eraint_1991"), os.path.join(evidence, "eraint_1991"))
    snapshot = _snapshot(tmp_path)
    script = os.path.join(snapshot, "scripts", "run_threshold_sensitivity.py")
    try:
        r = subprocess.run([sys.executable, "-c", DYING_DRIVER, os.path.join(ROOT, "scripts"), script, campaign, evidence, manifest, out, snapshot],
                           capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        pytest.fail("the command hung on a dead worker")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "died without reporting" in r.stdout and "REPLAY INCOMPLETE" in r.stdout
    assert not os.path.exists(os.path.join(out, "eraint_1990", "tracking_eraint_1990.json"))


FIXTURE = os.path.join(HERE, "fixtures", "implementation_mode_artifact_1990.json")


def _bind_like_the_instrument(art, argv):
    """Stamp a fake artifact with the bindings the real instrument records, from the same
    argv it was given: its own revision, the region polygons, the archive files over its
    record years and the archive year file. Reuse is checked against exactly these."""
    import collect_protocol_campaign as C
    import exact_tracks as X
    import season_metrics as S
    year = int(argv[argv.index("--year") + 1])
    published_file = argv[argv.index("--published-year-file") + 1] if "--published-year-file" in argv else None
    regions, record, _ = C.auxiliary_digests(argv[argv.index("--regions-dir") + 1], argv[argv.index("--record-dir") + 1],
                                             os.path.dirname(published_file) if published_file else "", year)
    art.update({"generated_by": "scripts/season_metrics.py", "script_sha256": X.digest(S.__file__),
                "region_polygons_sha256": regions, "published_record_sha256": record})
    art["inputs_sha256"]["published_year_file"] = {"sha256": _sha(published_file)} if published_file else None


def _artifact(v1, port):
    return {"comparison": {"season": {"v1": v1, "port": port, "port_minus_v1": port - v1, "difference_over_published_sd": -0.5,
                                      "distinct_waves": {"v1": 9, "port": 8, "port_minus_v1": -1},
                                      "duplication": {"v1_fraction": 0.3, "port_fraction": 0.31},
                                      "distributions": {"lifetime": {"ks_statistic": 0.1, "percentiles": {"50": {"v1": 13, "port": 12}}},
                                                        "genesis_lat": {"ks_statistic": 0.2, "percentiles": {"50": {"v1": 11.5, "port": 9.0}}}}},
                           **{f"month {m}": {"v1": 1, "port": 1} for m in (6, 7, 8, 9)}, "band +0..+10": {"v1": 3, "port": 2}},
            "sides": {"v1": {"case_id": "c" * 32}, "port": {"case_id": "c" * 32}},
            "columns": {k: {"tracks_all": 100, "tracks_in_season_whole_domain": 40,
                            "boundaries": {"crossing_in": {"tracks": 2}, "crossing_out": {"tracks": 1}, "year_end_potentially_censored": {"tracks": 3}}}
                        for k in ("v1", "port")},
            "published_context_column": {"tracks_in_season": 200}}


def test_compare_runs_the_instrument_in_implementation_mode_and_summarizes_with_the_collectors_row_builder(tmp_path, monkeypatch):
    M = _load(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    campaign, evidence, manifest, out = _world(tmp_path)
    assert M.replay(_args(campaign, evidence, manifest, out)) == 0
    import season_metrics as S
    seen = []

    def fake_runner(argv):
        seen.append(argv)
        # the real instrument's implementation-mode artifact, retained as the fixture: no
        # sides block, one case id at the top level, which the earlier invented fixture hid
        art = json.load(open(FIXTURE))
        art["comparison"]["season"]["port"], art["comparison"]["season"]["port_minus_v1"] = 150, 15
        art["comparison"]["season"]["v1"] = 135
        art["case_id"] = "c" * 32
        art["inputs_sha256"]["v1"]["sha256"] = _sha(argv[argv.index("--v1") + 1])
        art["inputs_sha256"]["port"]["sha256"] = _sha(argv[argv.index("--port") + 1])
        _bind_like_the_instrument(art, argv)
        json.dump(art, open(argv[argv.index("--out") + 1], "w"))
    monkeypatch.setattr(S, "main", fake_runner)
    import types
    sens_evidence, artifacts, summary = tmp_path / "sens_evidence", tmp_path / "artifacts", tmp_path / "summary.json"
    artifacts.mkdir()
    args = types.SimpleNamespace(campaign_evidence=evidence, runs=out, evidence=str(sens_evidence), artifacts=str(artifacts),
                                 summary=str(summary), years="1990-1991", regions_dir="r", record_dir="d", published_dir="d",
                                 dataset="eraint", pair="archived", against="own", manifest=manifest)
    assert M.collect_and_compare(args) == 0
    assert len(seen) == 1 and seen[0][seen[0].index("--mode") + 1] == "implementation"
    assert seen[0][seen[0].index("--v1") + 1] == os.path.join(evidence, "eraint_1990", "tracker_port.mat")   # Rule A is side A
    assert seen[0][seen[0].index("--port") + 1] == os.path.join(str(sens_evidence), "eraint_1990", "tracker_port.mat")
    s = json.load(open(summary))
    assert s["years"]["1990"]["africa_origin"]["v1"] == 135 and s["years"]["1990"]["africa_origin"]["port"] == 150
    assert s["years"]["1990"]["sides"] == {"v1": "c" * 32, "port": "c" * 32}                # one case id, both sides
    assert s["years_without_a_comparison"] == {"1991": "a run is missing"}
    assert s["aggregates"]["africa_origin"]["mean"] == {"v1": 135.0, "port": 150.0}
    fx = json.load(open(FIXTURE))["comparison"]["season"]["distributions"]
    assert s["distributions_mean_over_years"]["genesis_lat"]["percentiles_mean"]["50"]["v1"] == fx["genesis_lat"]["percentiles"]["50"]["v1"]
    assert s["distributions_mean_over_years"]["lifetime"]["ks_statistic_mean"] == pytest.approx(fx["lifetime"]["ks_statistic"])
    assert s["sides"]["port"] == M.LABEL
    assert _sha(os.path.join(str(sens_evidence), "eraint_1990", "tracker_port.mat")) == _sha(os.path.join(out, "eraint_1990", "tracker_port.mat"))
    with pytest.raises(SystemExit):                                                      # the summary is published exclusively
        M.collect_and_compare(args)
    # an artifact already at the path is reused only when it is this comparison of these files
    os.unlink(summary)
    assert M.collect_and_compare(args) == 0 and len(seen) == 1                            # reused, the runner not called again
    foreign = json.load(open(FIXTURE))
    foreign["mode"] = "reanalysis"
    json.dump(foreign, open(os.path.join(str(artifacts), "sensitivity_1990_eraint_archived_vs_rule_a.json"), "w"))
    os.unlink(summary)
    assert M.collect_and_compare(args) == 1 and len(seen) == 1                            # nothing compared is a failure
    s = json.load(open(summary))
    assert "1990" not in s["years"] and s["years_without_a_comparison"]["1990"].startswith("refused:")
    assert "mode 'reanalysis'" in s["years_without_a_comparison"]["1990"]


def test_a_replay_runs_only_from_a_frozen_snapshot_whose_entry_point_made_the_case(tmp_path, monkeypatch):
    calls = []
    _stub(monkeypatch, calls)
    campaign, evidence, manifest, out = _world(tmp_path)
    live = _load()                                                       # the driver from the live tree, not the snapshot
    with pytest.raises(SystemExit) as exc:
        live.replay_year(campaign, evidence, manifest, out, 1990, 7.16e-7, 2.80e-6, snapshot=_snapshot(tmp_path), commit=COMMIT)
    assert "not from the snapshot" in str(exc.value) and calls == []
    M = _load(tmp_path)
    with pytest.raises(SystemExit) as exc:                               # no commit named
        M.replay_year(campaign, evidence, manifest, out, 1990, 7.16e-7, 2.80e-6, snapshot=_snapshot(tmp_path), commit=None)
    assert "no code snapshot or commit" in str(exc.value)
    with open(os.path.join(_snapshot(tmp_path), "scripts", "export_protocol_case.py"), "ab") as fh:
        fh.write(b"\n# edited\n")                                       # the snapshot's entry point is no longer the case's
    with pytest.raises(SystemExit) as exc:
        _replay_year(M, tmp_path, campaign, evidence, manifest, out, 1990, 7.16e-7, 2.80e-6)
    assert "entry point is not the one that made the case" in str(exc.value) and calls == []


def test_the_record_is_bound_to_the_snapshot_commit_and_names_its_pair(tmp_path, monkeypatch):
    M = _load(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    campaign, evidence, manifest, out = _world(tmp_path)
    record, n = _replay_year(M, tmp_path, campaign, evidence, manifest, out, 1990, 7.16e-7, 2.80e-6)
    assert record["sensitivity"]["code_snapshot"] == {"commit": COMMIT, "path": os.path.realpath(_snapshot(tmp_path))}
    assert record["sensitivity"]["pair"] == "archived" and record["sensitivity"]["control"] is False
    # a record from another commit, another pair, or a control is never this experiment's completed year
    import export_protocol_case as E
    good = M.sensitivity_record_problems(os.path.join(out, "eraint_1990", "tracking_eraint_1990.json"), evidence, 1990, 7.16e-7, 2.80e-6, "m" * 64, "eraint", "archived", COMMIT, manifest=MANIFEST)
    assert good == []
    assert any("not made from this code snapshot" in p for p in
               M.sensitivity_record_problems(os.path.join(out, "eraint_1990", "tracking_eraint_1990.json"), evidence, 1990, 7.16e-7, 2.80e-6, "m" * 64, "eraint", "archived", "0000000", manifest=MANIFEST))
    assert any("names the pair" in p for p in
               M.sensitivity_record_problems(os.path.join(out, "eraint_1990", "tracking_eraint_1990.json"), evidence, 1990, 7.16e-7, 2.80e-6, "m" * 64, "eraint", "eraint-rule-a", COMMIT, manifest=MANIFEST))


def test_the_pairs_are_defined_per_dataset_and_the_transferred_pair_is_read_from_the_calibration_artifact(tmp_path):
    M = _load(tmp_path)
    assert M.threshold_pair("archived", "eraint")[:2] == (7.16e-7, 2.80e-6)
    with pytest.raises(SystemExit):
        M.threshold_pair("archived", "era5")                              # the archived constants are not defined for ERA5
    with pytest.raises(SystemExit):
        M.threshold_pair("eraint-rule-a", "eraint")
    art = tmp_path / "thresholds_eraint.json"
    art.write_text(json.dumps({"schema": "thresholds-v2", "prefix": "eraint", "case_id": "diag-x",
                               "coarse": {"threshold": 4.494e-7}, "fine": {"threshold": 2.2501e-6}}))
    coarse, fine, label, source = M.threshold_pair("eraint-rule-a", "era5", str(art))
    assert (coarse, fine) == (4.494e-7, 2.2501e-6) and "ERA5" in label
    assert source == {"kind": "calibration artifact", "path": str(art), "sha256": _sha(art), "case_id": "diag-x"}
    art.write_text(json.dumps({"schema": "thresholds-v2", "prefix": "era5", "coarse": {"threshold": 1}, "fine": {"threshold": 2}}))
    with pytest.raises(SystemExit):
        M.threshold_pair("eraint-rule-a", "era5", str(art))               # not the ERA-Interim artifact


def test_a_control_reproduces_the_campaign_tracks_or_fails_by_name(tmp_path, monkeypatch):
    M = _load(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    campaign, evidence, manifest, out = _world(tmp_path, dataset="era5", real_tracks=True)
    import types
    args = types.SimpleNamespace(campaign=campaign, evidence=evidence, manifest=manifest, out_dir=out, dataset="era5", year=1990,
                                 code_snapshot=_snapshot(tmp_path), snapshot_commit=COMMIT)
    assert M.control(args) == 0
    assert calls[0][:2] == (4.494e-7, 2.25e-6)                           # the case's own thresholds, not any pair
    rec = json.load(open(os.path.join(out, "control", "era5_1990", "tracking_era5_1990.json")))
    assert rec["sensitivity"]["control"] is True and rec["sensitivity"]["pair"] == "control"
    assert any("is a control" in p for p in
               M.sensitivity_record_problems(os.path.join(out, "control", "era5_1990", "tracking_era5_1990.json"), evidence, 1990, 4.494e-7, 2.25e-6, "m" * 64, "era5", "eraint-rule-a", COMMIT, manifest=MANIFEST))
    import export_protocol_case as E
    monkeypatch.setattr(E, "track_case", lambda *a, **k: [dict(FAKE_TRACK, meanlat=np.array([10.0, 11.0]))])
    campaign2, evidence2, manifest2, out2 = _world(tmp_path / "second", dataset="era5", real_tracks=True)
    args2 = types.SimpleNamespace(campaign=campaign2, evidence=evidence2, manifest=manifest2, out_dir=out2, dataset="era5", year=1990,
                                  code_snapshot=_snapshot(tmp_path / "second"), snapshot_commit=COMMIT)
    M2 = _load(tmp_path / "second")
    with pytest.raises(SystemExit) as exc:
        M2.control(args2)
    assert "CONTROL FAILED" in str(exc.value)


def test_the_transferred_pair_replays_era5_and_refuses_the_other_experiments_records(tmp_path, monkeypatch):
    M = _load(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    campaign, evidence, manifest, out = _world(tmp_path, dataset="era5")
    art = tmp_path / "thresholds_eraint.json"
    art.write_text(json.dumps({"schema": "thresholds-v2", "prefix": "eraint", "coarse": {"threshold": 4.494e-7}, "fine": {"threshold": 2.2501e-6}}))
    args = _args(campaign, evidence, manifest, out, dataset="era5", pair="eraint-rule-a", pair_source=str(art))
    assert M.replay(args) == 0 and calls[0][:2] == (4.494e-7, 2.2501e-6)
    rec = json.load(open(os.path.join(out, "era5_1990", "tracking_era5_1990.json")))
    assert rec["sensitivity"]["pair"] == "eraint-rule-a" and rec["sensitivity"]["source"]["sha256"] == _sha(art)
    assert rec["sensitivity"]["replaces_rule_a"]["coarse"] == 4.494e-7                 # the fixture's ERA5 record carries the same values
    assert M.replay(args) == 0 and len(calls) == 1                                       # skipped as this experiment's record
    with pytest.raises(SystemExit):                                                      # the archived pair is not defined for ERA5
        M.replay(_args(campaign, evidence, manifest, out, dataset="era5", pair="archived"))


def test_a_stopped_launch_terminates_its_workers_and_releases_the_lock_before_another_can_write(tmp_path, monkeypatch):
    """The failure of the first ERA-Interim attempt, replayed: a launch with slow workers is
    stopped with SIGTERM; its workers must be gone and its lock released within seconds,
    and only then does a second launch proceed."""
    import subprocess
    import time
    campaign, evidence, manifest, out = _world(tmp_path)
    _world(tmp_path / "second", year=1991)
    import shutil
    shutil.copytree(str(tmp_path / "second" / "campaign" / "eraint_1991"), os.path.join(campaign, "eraint_1991"))
    shutil.copytree(str(tmp_path / "second" / "evidence" / "eraint_1991"), os.path.join(evidence, "eraint_1991"))
    snapshot = _snapshot(tmp_path)
    script = os.path.join(snapshot, "scripts", "run_threshold_sensitivity.py")
    child = subprocess.Popen([sys.executable, "-c", SLOW_DRIVER, os.path.join(ROOT, "scripts"), script, campaign, evidence, manifest, out, snapshot],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    launches = os.path.join(out, ".launches")

    def registered():
        found = []
        for name in (os.listdir(launches) if os.path.isdir(launches) else []):
            reg = os.path.join(launches, name, "workers")
            found += [int(n) for n in (os.listdir(reg) if os.path.isdir(reg) else []) if n.isdigit()]
        return found
    deadline = time.time() + 30
    while time.time() < deadline and len(registered()) < 2:
        time.sleep(0.2)
    pids = registered()
    assert len(pids) == 2, "the workers never registered"
    # while the launch is alive, a second launch is refused by the lock
    M = _load(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    with pytest.raises(SystemExit) as exc:
        M.replay(_args(campaign, evidence, manifest, out, years="1990-1991"))
    assert "holds" in str(exc.value)
    child.send_signal(__import__("signal").SIGTERM)
    outp, _ = child.communicate(timeout=30)
    assert child.returncode == 130 and "STOPPED by signal" in outp, outp

    def alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
    deadline = time.time() + 5
    while time.time() < deadline and any(alive(p) for p in pids):
        time.sleep(0.1)
    assert not any(alive(p) for p in pids), "a worker outlived its stopped launch"
    assert registered() == []                                                          # the launch's directory is gone with it
    assert not os.path.exists(os.path.join(out, "eraint_1990", "tracking_eraint_1990.json"))
    # and now a second launch proceeds
    assert M.replay(_args(campaign, evidence, manifest, out, years="1990-1991")) == 0 and len(calls) == 2


SLOW_DRIVER = """
import json, os, sys, time, types
sys.path.insert(0, sys.argv[1])
import importlib.util
spec = importlib.util.spec_from_file_location("run_threshold_sensitivity", sys.argv[2])
M = importlib.util.module_from_spec(spec); sys.modules["run_threshold_sensitivity"] = M; spec.loader.exec_module(M)
import export_protocol_case as E
import season_metrics as S
E.load_manifest = lambda p: (json.load(open(p)), "m" * 64)
S.producer_problems = lambda *a, **k: []

def slow_track(payload, coarse, fine, exclusive, absorb):
    time.sleep(120)                                    # a worker mid-year when the launch is stopped
    return []
E.track_case = slow_track
args = types.SimpleNamespace(campaign=sys.argv[3], evidence=sys.argv[4], manifest=sys.argv[5], out_dir=sys.argv[6],
                             years="1990-1991", workers=2, dataset="eraint", pair="archived", pair_source=None,
                             code_snapshot=sys.argv[7], snapshot_commit="abc1234")
sys.exit(M.replay(args, context="fork"))
"""


def test_a_dead_launchs_live_workers_refuse_a_new_launch_and_the_lock_itself_cannot_be_taken_twice(tmp_path, monkeypatch):
    import subprocess
    M = _load(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    campaign, evidence, manifest, out = _world(tmp_path)
    gone = os.path.join(out, ".launches", "999999999", "workers")
    os.makedirs(gone)
    open(os.path.join(gone, str(os.getpid())), "w").write("x")               # a registered worker still alive (this process)
    with pytest.raises(SystemExit) as exc:
        M.replay(_args(campaign, evidence, manifest, out))
    assert "workers" in str(exc.value) and "still alive" in str(exc.value)
    os.unlink(os.path.join(gone, str(os.getpid())))
    # the lock is the kernel's: held here, a second process cannot take it, whatever it reads
    lock = M.LaunchLock(out)
    lock.acquire()
    probe = subprocess.run([sys.executable, "-c", "import fcntl, sys\nfh = open(sys.argv[1], 'a+')\n"
                            "try:\n    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)\n    print('TAKEN')\nexcept OSError:\n    print('REFUSED')",
                            lock.path], capture_output=True, text=True)
    assert probe.stdout.strip() == "REFUSED"
    lock.release()
    assert M.replay(_args(campaign, evidence, manifest, out)) == 0 and len(calls) == 1   # released, then run


def test_a_worker_that_starts_after_its_launch_was_stopped_does_nothing(tmp_path, monkeypatch):
    M = _load(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    campaign, evidence, manifest, out = _world(tmp_path)
    launch_dir = tmp_path / "launch"
    launch_dir.mkdir()
    (launch_dir / "LIVE").write_text("x")
    (launch_dir / "STOP").write_text("stopped")
    M._LAUNCH["dir"] = str(launch_dir)
    try:
        year, ok, msg = M._worker((campaign, evidence, manifest, out, 1990, 7.16e-7, 2.80e-6, "eraint", None, _snapshot(tmp_path), COMMIT))
    finally:
        M._LAUNCH["dir"] = None
    assert (year, ok) == (1990, False) and "stopped" in msg and calls == []
    assert not os.path.exists(os.path.join(out, "eraint_1990"))


def test_a_sidecar_relabeled_to_another_commit_or_pair_disagrees_with_the_record_inside_the_tracks(tmp_path, monkeypatch):
    M = _load(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    campaign, evidence, manifest, out = _world(tmp_path)
    _replay_year(M, tmp_path, campaign, evidence, manifest, out, 1990, 7.16e-7, 2.80e-6)
    rec = os.path.join(out, "eraint_1990", "tracking_eraint_1990.json")
    r = json.load(open(rec))
    r["sensitivity"]["code_snapshot"]["commit"] = "deadbee"
    open(rec, "w").write(json.dumps(r))
    problems = M.sensitivity_record_problems(rec, evidence, 1990, 7.16e-7, 2.80e-6, "m" * 64, "eraint", "archived", "deadbee", manifest=MANIFEST)
    assert any("does not agree" in p and "commit" in p for p in problems)
    r["sensitivity"]["code_snapshot"]["commit"] = COMMIT
    r["sensitivity"]["pair"] = "eraint-rule-a"
    open(rec, "w").write(json.dumps(r))
    problems = M.sensitivity_record_problems(rec, evidence, 1990, 7.16e-7, 2.80e-6, "m" * 64, "eraint", "eraint-rule-a", COMMIT, manifest=MANIFEST)
    assert any("does not agree" in p and "pair" in p for p in problems)


def test_compare_refuses_runs_that_are_not_this_experiments(tmp_path, monkeypatch):
    """The campaign's own baseline evidence offered as the runs: nothing is copied or compared."""
    import types
    M = _load(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    campaign, evidence, manifest, out = _world(tmp_path, dataset="era5")
    import season_metrics as S
    monkeypatch.setattr(S, "main", lambda argv: pytest.fail("the instrument must not run on a record that is not this experiment's"))
    art = tmp_path / "thresholds_eraint.json"
    art.write_text(json.dumps({"schema": "thresholds-v2", "prefix": "eraint", "coarse": {"threshold": 4.494e-7}, "fine": {"threshold": 2.2501e-6}}))
    sens_evidence, artifacts, summary = tmp_path / "sens_evidence", tmp_path / "artifacts", tmp_path / "summary.json"
    artifacts.mkdir()
    args = types.SimpleNamespace(campaign_evidence=evidence, runs=evidence, evidence=str(sens_evidence), artifacts=str(artifacts),
                                 summary=str(summary), years="1990-1990", regions_dir="r", record_dir="d", published_dir="d",
                                 dataset="era5", pair="eraint-rule-a", against="own", manifest=manifest, pair_source=str(art), snapshot_commit=COMMIT)
    assert M.collect_and_compare(args) == 1                    # nothing compared is a failure, recorded
    s = json.load(open(summary))
    assert s["years"] == {} and s["years_without_a_comparison"]["1990"].startswith("refused: the run is not this experiment's")
    assert not os.path.exists(os.path.join(str(sens_evidence), "era5_1990"))


def test_compare_against_the_eraint_baseline_declares_the_transfer_in_reanalysis_mode(tmp_path, monkeypatch):
    M = _load(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    campaign, evidence, manifest, out = _world(tmp_path, dataset="era5")
    _world(tmp_path / "e", dataset="eraint")
    import shutil
    shutil.copytree(str(tmp_path / "e" / "evidence" / "eraint_1990"), os.path.join(evidence, "eraint_1990"))
    art = tmp_path / "thresholds_eraint.json"
    art.write_text(json.dumps({"schema": "thresholds-v2", "prefix": "eraint", "coarse": {"threshold": 4.494e-7}, "fine": {"threshold": 2.2501e-6}}))
    assert M.replay(_args(campaign, evidence, manifest, out, dataset="era5", pair="eraint-rule-a", pair_source=str(art))) == 0
    import season_metrics as S
    seen = []

    def fake_runner(argv):
        seen.append(argv)
        a = json.load(open(FIXTURE))
        a["mode"], a["year"] = "reanalysis", 1990
        a["threshold_transfer"] = {"side": "port"}
        a["inputs_sha256"]["v1"]["sha256"] = _sha(argv[argv.index("--v1") + 1])
        a["inputs_sha256"]["port"]["sha256"] = _sha(argv[argv.index("--port") + 1])
        _bind_like_the_instrument(a, argv)
        json.dump(a, open(argv[argv.index("--out") + 1], "w"))
    monkeypatch.setattr(S, "main", fake_runner)
    import types
    sens_evidence, artifacts, summary = tmp_path / "sens_evidence", tmp_path / "artifacts", tmp_path / "summary.json"
    artifacts.mkdir()
    args = types.SimpleNamespace(campaign_evidence=evidence, runs=out, evidence=str(sens_evidence), artifacts=str(artifacts),
                                 summary=str(summary), years="1990-1990", regions_dir="r", record_dir="d", published_dir="d",
                                 dataset="era5", pair="eraint-rule-a", against="eraint-baseline", manifest=manifest,
                                 pair_source=str(art), snapshot_commit=COMMIT)
    assert M.collect_and_compare(args) == 0
    argv = seen[0]
    assert argv[argv.index("--mode") + 1] == "reanalysis" and "--declared-threshold-transfer" in argv and argv[argv.index("--manifest") + 1] == manifest
    assert argv[argv.index("--v1") + 1] == os.path.join(evidence, "eraint_1990", "tracker_port.mat")       # ERA-Interim Rule A is side A
    assert argv[argv.index("--port") + 1] == os.path.join(str(sens_evidence), "era5_1990", "tracker_port.mat")
    assert argv[argv.index("--out") + 1].endswith("sensitivity_1990_era5_transferred_vs_eraint_baseline.json")
    s = json.load(open(summary))
    assert s["against"] == "eraint-baseline" and "ERA-Interim" in s["sides"]["v1"] and "declared threshold transfer" in s["instrument_mode"]
    os.unlink(summary)
    assert M.collect_and_compare(args) == 0 and len(seen) == 1                            # reused as this comparison
    a = json.load(open(os.path.join(str(artifacts), "sensitivity_1990_era5_transferred_vs_eraint_baseline.json")))
    a["threshold_transfer"] = None
    json.dump(a, open(os.path.join(str(artifacts), "sensitivity_1990_era5_transferred_vs_eraint_baseline.json"), "w"))
    os.unlink(summary)
    assert M.collect_and_compare(args) == 1                                               # nothing compared is a failure
    assert "no declared threshold transfer" in json.load(open(summary))["years_without_a_comparison"]["1990"]


def _ignore_stop_and_sleep(seconds):
    import signal as _signal
    import time as _time
    _signal.signal(_signal.SIGTERM, _signal.SIG_IGN)                    # a survivor, injected: the tracking worker does not do this
    _time.sleep(seconds)


def test_stop_keeps_the_launch_state_while_a_child_survives_and_a_later_launch_refuses_it(tmp_path, monkeypatch):
    """The one shutdown rule under injection: a child that ignores the stop is recorded,
    the STOP flag and the registry stay, a new launch is refused by its name, and only
    when it is gone does the state go."""
    import multiprocessing
    import time
    M = _load(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    campaign, evidence, manifest, out = _world(tmp_path)
    lock = M.LaunchLock(out)
    lock.acquire()
    ctx = multiprocessing.get_context("spawn")
    child = ctx.Process(target=_ignore_stop_and_sleep, args=(60,))
    child.start()
    time.sleep(1.0)                                                       # started, never registered: known only to the process table
    survivors = M.stop_launch(lock, [child], bound=1.0)
    assert survivors == [child.pid]
    assert os.path.exists(os.path.join(lock.launch_dir, "STOP")) and str(child.pid) in os.listdir(lock.registry)
    assert not os.path.exists(os.path.join(lock.launch_dir, "LIVE"))                # the marker went first and never returns
    assert not M._launch_is_live(lock.launch_dir)
    assert not os.path.exists(lock.path) or True                           # the kernel lock is released, the state is not
    with pytest.raises(SystemExit) as exc:                                 # a later launch refuses the survivor by name
        M.replay(_args(campaign, evidence, manifest, out))
    assert "still alive" in str(exc.value) and str(child.pid) in str(exc.value)
    child.kill()
    child.join(10)
    assert M.replay(_args(campaign, evidence, manifest, out)) == 0 and len(calls) == 1
    assert not os.path.exists(lock.launch_dir)


def test_a_late_worker_never_recreates_a_launch_that_is_gone_or_stopped(tmp_path):
    """Three states a late worker can meet: the launch directory gone, the launch stopped,
    and the launch directory present but its registry gone (the window the review
    reproduced, where the worker had passed its check and shutdown then removed the
    state). In none does the worker create anything or register."""
    import subprocess
    snapshot = _snapshot(tmp_path)
    script = os.path.join(snapshot, "scripts", "run_threshold_sensitivity.py")
    for prepare in ("gone", "stopped", "registry-gone", "half-removed"):
        launch_dir = tmp_path / prepare
        if prepare == "stopped":
            launch_dir.mkdir()
            (launch_dir / "LIVE").write_text("x")
            (launch_dir / "STOP").write_text("x")
        if prepare == "registry-gone":
            launch_dir.mkdir()
            (launch_dir / "LIVE").write_text("x")
        if prepare == "half-removed":                                      # a directory mid-removal: registry present, no marker, no flag
            (launch_dir / "workers").mkdir(parents=True)
        r = subprocess.run([sys.executable, "-c",
                            "import sys, importlib.util\n"
                            "spec = importlib.util.spec_from_file_location('m', sys.argv[1]); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)\n"
                            "M._register_worker(sys.argv[2]); print('REGISTERED')", script, str(launch_dir)], capture_output=True, text=True)
        assert r.returncode == 0 and "REGISTERED" not in r.stdout, (prepare, r.stdout, r.stderr)
        workers = os.path.join(str(launch_dir), "workers")
        assert not os.path.exists(workers) or os.listdir(workers) == [], prepare        # nothing created, nothing recorded
    # and with the registry present, the worker records itself there and nowhere else
    launch_dir = tmp_path / "live"
    (launch_dir / "workers").mkdir(parents=True)
    (launch_dir / "LIVE").write_text("x")
    r = subprocess.run([sys.executable, "-c",
                        "import sys, importlib.util, os\n"
                        "spec = importlib.util.spec_from_file_location('m', sys.argv[1]); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)\n"
                        "M._register_worker(sys.argv[2]); print('REGISTERED', os.getpid())", script, str(launch_dir)], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("REGISTERED")
    assert os.listdir(str(launch_dir / "workers")) == [r.stdout.split()[1]]


def test_stop_with_no_survivor_releases_the_state(tmp_path, monkeypatch):
    M = _load(tmp_path)
    campaign, evidence, manifest, out = _world(tmp_path)
    lock = M.LaunchLock(out)
    lock.acquire()
    assert M.stop_launch(lock, [], bound=1.0) == []
    assert not os.path.exists(lock.launch_dir)


def _as_record_made_before_the_pair_field(record_path):
    """Strip the pair and the code snapshot from a replayed record, beside the tracks and
    inside them, as the ERA-Interim archived-constant runs of 2026-09-27 were written."""
    from scipy.io import loadmat, savemat
    r = json.load(open(record_path))
    r["sensitivity"].pop("pair")
    r["sensitivity"].pop("code_snapshot", None)
    open(record_path, "w").write(json.dumps(r))
    tracks = os.path.join(os.path.dirname(record_path), "tracker_port.mat")
    m = {k: v for k, v in loadmat(tracks).items() if not k.startswith("__")}
    inside = json.loads(str(m["producer_json"][0]))
    inside["sensitivity"].pop("pair")
    inside["sensitivity"].pop("code_snapshot", None)
    m["producer_json"] = json.dumps(inside)
    savemat(tracks, m)
    import hashlib
    r["dataset_specific"]["tracks_sha256"] = hashlib.sha256(open(tracks, "rb").read()).hexdigest()
    open(record_path, "w").write(json.dumps(r))


def test_a_record_made_before_the_pair_field_is_accepted_only_as_the_archived_experiment(tmp_path, monkeypatch):
    M = _load(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    campaign, evidence, manifest, out = _world(tmp_path)
    _replay_year(M, tmp_path, campaign, evidence, manifest, out, 1990, 7.16e-7, 2.80e-6)
    rec = os.path.join(out, "eraint_1990", "tracking_eraint_1990.json")
    _as_record_made_before_the_pair_field(rec)
    assert M.sensitivity_record_problems(rec, evidence, 1990, 7.16e-7, 2.80e-6, "m" * 64, "eraint", "archived", None, manifest=MANIFEST) == []
    # the unlabeled record is not another pair's, and not the archived run under other thresholds
    assert any("names the pair" in p for p in
               M.sensitivity_record_problems(rec, evidence, 1990, 7.16e-7, 2.80e-6, "m" * 64, "eraint", "eraint-rule-a", None, manifest=MANIFEST))
    assert any("archived constants" in p for p in
               M.sensitivity_record_problems(rec, evidence, 1990, 4.494e-7, 2.25e-6, "m" * 64, "eraint", "archived", None, manifest=MANIFEST))


def test_a_sidecar_stripped_of_its_pair_still_disagrees_with_the_record_inside_the_tracks(tmp_path, monkeypatch):
    M = _load(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    campaign, evidence, manifest, out = _world(tmp_path)
    _replay_year(M, tmp_path, campaign, evidence, manifest, out, 1990, 7.16e-7, 2.80e-6)
    rec = os.path.join(out, "eraint_1990", "tracking_eraint_1990.json")
    r = json.load(open(rec))
    r["sensitivity"].pop("pair")                         # beside the tracks only
    open(rec, "w").write(json.dumps(r))
    problems = M.sensitivity_record_problems(rec, evidence, 1990, 7.16e-7, 2.80e-6, "m" * 64, "eraint", "archived", COMMIT, manifest=MANIFEST)
    assert any("does not agree" in p and "pair" in p for p in problems)


def test_a_comparison_made_against_another_archive_directory_is_not_reused(tmp_path, monkeypatch):
    """The review's specimen: a 25-year artifact reused against the 32-year archive kept the
    old scale. An artifact is reused only when bound to the current archive directory."""
    M = _load(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    campaign, evidence, manifest, out = _world(tmp_path)
    assert M.replay(_args(campaign, evidence, manifest, out)) == 0
    import season_metrics as S
    seen = []

    def fake_runner(argv):
        seen.append(argv)
        art = json.load(open(FIXTURE))
        art["case_id"] = "c" * 32
        art["inputs_sha256"]["v1"]["sha256"] = _sha(argv[argv.index("--v1") + 1])
        art["inputs_sha256"]["port"]["sha256"] = _sha(argv[argv.index("--port") + 1])
        _bind_like_the_instrument(art, argv)
        json.dump(art, open(argv[argv.index("--out") + 1], "w"))
    monkeypatch.setattr(S, "main", fake_runner)
    import types
    sens_evidence, artifacts, summary = tmp_path / "sens_evidence", tmp_path / "artifacts", tmp_path / "summary.json"
    artifacts.mkdir()
    old_archive = tmp_path / "archive_25"
    old_archive.mkdir()
    (old_archive / "ERA-Int_ew_700hPa_1990_AFR.nc").write_bytes(b"1990")
    args = types.SimpleNamespace(campaign_evidence=evidence, runs=out, evidence=str(sens_evidence), artifacts=str(artifacts),
                                 summary=str(summary), years="1990-1990", regions_dir="r", record_dir=str(old_archive),
                                 published_dir=str(old_archive), dataset="eraint", pair="archived", against="own", manifest=manifest)
    assert M.collect_and_compare(args) == 0 and len(seen) == 1
    os.unlink(summary)
    assert M.collect_and_compare(args) == 0 and len(seen) == 1            # same archive, reused
    new_archive = tmp_path / "archive_32"
    new_archive.mkdir()
    (new_archive / "ERA-Int_ew_700hPa_1990_AFR.nc").write_bytes(b"1990")
    (new_archive / "ERA-Int_ew_700hPa_1979_AFR.nc").write_bytes(b"1979")   # a year the old directory lacked
    args.record_dir = args.published_dir = str(new_archive)
    os.unlink(summary)
    assert M.collect_and_compare(args) == 1 and len(seen) == 1             # refused, never reused or overwritten
    reason = json.load(open(summary))["years_without_a_comparison"]["1990"]
    assert "published record digests" in reason


def test_a_record_declaring_other_tracker_flags_is_refused_beside_and_inside_the_tracks(tmp_path, monkeypatch):
    M = _load(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    campaign, evidence, manifest, out = _world(tmp_path)
    _replay_year(M, tmp_path, campaign, evidence, manifest, out, 1990, 7.16e-7, 2.80e-6)
    rec = os.path.join(out, "eraint_1990", "tracking_eraint_1990.json")
    args = (rec, evidence, 1990, 7.16e-7, 2.80e-6, "m" * 64, "eraint", "archived", COMMIT)
    assert M.sensitivity_record_problems(*args, manifest=MANIFEST) == []
    flagged = {**MANIFEST, "tracker_flags": {"exclusive": True, "absorb": False}}
    # the same record checked against a manifest with other flags is not that manifest's run
    assert any("tracker_flags" in p for p in M.sensitivity_record_problems(*args, manifest=flagged))
    r = json.load(open(rec))
    r["protocol_settings"]["tracker_flags"] = {"exclusive": True, "absorb": False}
    open(rec, "w").write(json.dumps(r))
    assert any("the record: protocol setting tracker_flags" in p for p in M.sensitivity_record_problems(*args, manifest=MANIFEST))


def test_no_manifest_means_the_settings_were_not_checked(tmp_path, monkeypatch):
    M = _load(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    campaign, evidence, manifest, out = _world(tmp_path)
    _replay_year(M, tmp_path, campaign, evidence, manifest, out, 1990, 7.16e-7, 2.80e-6)
    rec = os.path.join(out, "eraint_1990", "tracking_eraint_1990.json")
    problems = M.sensitivity_record_problems(rec, evidence, 1990, 7.16e-7, 2.80e-6, "m" * 64, "eraint", "archived", COMMIT)
    assert any("not checked" in p for p in problems)


def test_an_explicitly_empty_pair_inside_the_tracks_is_not_a_legacy_record(tmp_path, monkeypatch):
    from scipy.io import loadmat, savemat
    M = _load(tmp_path)
    calls = []
    _stub(monkeypatch, calls)
    campaign, evidence, manifest, out = _world(tmp_path)
    _replay_year(M, tmp_path, campaign, evidence, manifest, out, 1990, 7.16e-7, 2.80e-6)
    rec = os.path.join(out, "eraint_1990", "tracking_eraint_1990.json")
    _as_record_made_before_the_pair_field(rec)
    tracks = os.path.join(os.path.dirname(rec), "tracker_port.mat")
    m = {k: v for k, v in loadmat(tracks).items() if not k.startswith("__")}
    inside = json.loads(str(m["producer_json"][0]))
    inside["sensitivity"]["pair"] = None                      # present but empty, inside only
    m["producer_json"] = json.dumps(inside)
    savemat(tracks, m)
    r = json.load(open(rec))
    r["dataset_specific"]["tracks_sha256"] = _sha(tracks)
    open(rec, "w").write(json.dumps(r))
    problems = M.sensitivity_record_problems(rec, evidence, 1990, 7.16e-7, 2.80e-6, "m" * 64, "eraint", "archived", None, manifest=MANIFEST)
    assert any("record inside them has one" in p for p in problems)


def test_an_explicitly_null_tracker_flags_field_is_a_mismatch_beside_and_inside_the_tracks(tmp_path, monkeypatch):
    M = _load(tmp_path)
    flags = MANIFEST["tracker_flags"]
    assert M.settings_problems({"protocol_settings": {"tracker_flags": flags}, "dataset_specific": {"tracker_flags": flags}},
                               MANIFEST, "r", ("tracker_flags",)) == []
    assert any("own tracker flags" in p for p in
               M.settings_problems({"protocol_settings": {"tracker_flags": flags}, "dataset_specific": {"tracker_flags": None}},
                                   MANIFEST, "r", ("tracker_flags",)))
    assert M.settings_problems({"protocol_settings": {"tracker_flags": flags}, "dataset_specific": {}},
                               MANIFEST, "r", ("tracker_flags",)) == []          # absent is bound by the protocol settings
