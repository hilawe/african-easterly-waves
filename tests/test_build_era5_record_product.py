"""The ERA5 1979 to 2025 product builder: its season row is the retained summary's ERA5
column by the instrument's own definitions, the original years must reproduce the retained
summary, and the original files are copied unchanged."""
import collections
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _load():
    path = os.environ.get("PRODUCT_SCRIPT", os.path.join(ROOT, "scripts", "build_era5_record_product.py"))
    spec = importlib.util.spec_from_file_location("build_era5_record_product_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _regions():
    from matplotlib.path import Path
    return [("NAL", Path([(-60, 10), (-20, 10), (-20, 30), (-60, 30)])), ("AFR", Path([(-20, 0), (40, 0), (40, 25), (-20, 25)]))]


def _day(y, m, d):
    import datetime as dt
    return float((dt.date(y, m, d) - dt.date(1900, 1, 1)).days)


def _tracks(year, seed):
    """A season's worth of synthetic tracks in this script's layout, some in Africa, some
    elsewhere, some before June, one crossing October 1, two near copies."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(40):
        start = _day(year, 5, 20) + float(rng.integers(0, 140)) + 0.25 * int(rng.integers(0, 4))
        n = int(rng.integers(6, 30))
        lon0 = float(rng.uniform(-50, 35))
        out.append({"time": start + 0.25 * np.arange(n), "lat": np.full(n, float(rng.uniform(5, 20))),
                    "lon": lon0 - 0.5 * np.arange(n)})
    out.append({"time": out[0]["time"].copy(), "lat": out[0]["lat"] + 0.1, "lon": out[0]["lon"] + 0.1})
    out.append({"time": _day(year, 9, 29) + 0.25 * np.arange(20), "lat": np.full(20, 12.0), "lon": 10 - 0.5 * np.arange(20)})
    # an Africa-origin track and its exact copy, so the duplication fraction is not zero
    africa = {"time": _day(year, 7, 10) + 0.25 * np.arange(12), "lat": np.full(12, 11.0), "lon": 15 - 0.5 * np.arange(12)}
    out += [africa, {k: v.copy() for k, v in africa.items()}]
    return out


def _artifact_row(side_a, side_b, year, regions, spread):
    """The retained summary's row for one year, made the way the campaign made it: the
    instrument's compare under a published spread, then the collector's one row definition."""
    import collect_protocol_campaign as C0
    import season_metrics as S
    import tempfile
    whole = {k: [t for t in v if S.in_season(t, year)] for k, v in (("v1", side_a), ("port", side_b))}
    dom = {k: [t for t in v if S.in_record_domain(t, regions)] for k, v in whole.items()}
    comparison, _ = S.compare(dom["v1"], dom["port"], spread)
    art = {"comparison": comparison, "columns": {k: {"tracks_all": len(v), "tracks_in_season_whole_domain": len(whole[k]),
                                                     "boundaries": S.boundary_lines(v, year)}
                                                 for k, v in (("v1", side_a), ("port", side_b))}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(art, fh)
    try:
        return C0.row_from_artifact(fh.name), whole, dom
    finally:
        os.unlink(fh.name)


def _spread(regions):
    """A published spread from synthetic record years, so the sparse rule decides as it did
    in the campaign rather than by hand."""
    import season_metrics as S
    return S.published_spread({y: [t for t in _tracks(y, 100 + y) if S.in_record_domain(t, regions)] for y in range(1990, 1996)})


def test_the_season_row_is_the_retained_rows_era5_column_by_the_instruments_own_definitions():
    """The contract the product rests on: where the campaign's reporting rule gives a year
    distributions, season_row equals the side B column of the row
    collect_protocol_campaign.row_from_artifact copies from a season_metrics comparison."""
    import season_metrics as S
    B = _load()
    regions, year = _regions(), 1995
    side_a, side_b = _tracks(year, 1), _tracks(year, 2)
    row, whole, dom = _artifact_row(side_a, side_b, year, regions, _spread(regions))
    assert dom["port"] and len(dom["port"]) < len(whole["port"]) < len(side_b)       # the filters each removed something
    assert 0 < S.duplication(dom["port"])["fraction"] != S.duplication(whole["port"])["fraction"]
    assert row["lifetime_percentiles"] is not None                                    # the reporting rule gave distributions
    assert B.season_row(side_b, year, regions) == B.retained_row(row)


def test_in_a_sparse_year_the_only_difference_is_the_stated_one():
    """Below five Africa-origin tracks the comparison reports counts only, so the retained
    row has no lifetime percentiles, while the product row reports them. That is the one
    difference the summary states, and the original-years gate would refuse such a year."""
    import season_metrics as S
    B = _load()
    regions, year = _regions(), 1995
    side_a = _tracks(year, 1)
    africa_b = [t for t in _tracks(year, 2) if S.in_season(t, year) and S.in_record_domain(t, regions)][:3]
    side_b = [t for t in _tracks(year, 2) if not (S.in_season(t, year) and S.in_record_domain(t, regions))] + africa_b
    row, _, dom = _artifact_row(side_a, side_b, year, regions, _spread(regions))
    assert len(dom["port"]) == 3 and row["lifetime_percentiles"] is None
    assert B.row_differences(B.season_row(side_b, year, regions), B.retained_row(row)) == ["lifetime_percentiles"]


def _summary_world(tmp_path, B, monkeypatch):
    """A retained campaign of two years under a tiny base manifest, its summary, an
    extension campaign of one year, and the protocol files; S.load_regions and C.check are
    stood in for."""
    from scipy.io import savemat
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"years": [2001, 2002]}))
    base_sha = hashlib.sha256(base.read_bytes()).hexdigest()
    monkeypatch.setattr(B.S, "load_regions", lambda d: (_regions(), {"africa": "a" * 64}))

    def campaign(root, years, seed0):
        for y in years:
            run = root / f"era5_{y}"
            run.mkdir(parents=True)
            tracks = _tracks(y, seed0 + y)
            mat = {"n": float(len(tracks)), "case_id": "c" * 32}
            for i, t in enumerate(tracks):
                mat[f"lat{i}"], mat[f"lon{i}"], mat[f"time{i}"] = t["lat"], t["lon"], t["time"]
            savemat(str(run / "tracker_port.mat"), mat)
            digest = hashlib.sha256((run / "tracker_port.mat").read_bytes()).hexdigest()
            (run / f"tracking_era5_{y}.json").write_text(json.dumps(
                {"dataset_specific": {"dataset": "era5", "year": y, "tracks_sha256": digest},
                 "protocol_settings": {"manifest_sha256": base_sha}}))
    retained, ext = tmp_path / "retained", tmp_path / "ext"
    campaign(retained, (2001, 2002), 10)
    campaign(ext, (2003,), 20)
    rows = {}
    for y in (2001, 2002):
        r = B.season_row(B._tracks(str(retained / f"era5_{y}" / "tracker_port.mat")), y, _regions())
        rows[str(y)] = {k: {"port": r[k], "v1": None} for k in ("tracks_in_year", "season_whole_domain", "africa_origin",
                                                                 "distinct_waves", "duplication_fraction")}
        rows[str(y)].update({"lifetime_percentiles": {p: [None, v] for p, v in r["lifetime_percentiles"].items()},
                             "months": {m: {"port": v} for m, v in r["months"].items()},
                             "bands": {b: {"port": v} for b, v in r["bands"].items()},
                             "boundaries": {"port": r["boundaries"]}})
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"manifest_sha256": base_sha, "years": rows}))
    for name in ("extension.json", "calibration.json"):
        (tmp_path / name).write_text(json.dumps({"name": name}))
    report = {"passed": True, "tracking_years": [2003, 2003], "manifest_sha256": "e" * 64, "calibration_sha256": "f" * 64,
              "script_sha256": "1" * 64, "retained_climatology_mapping_sha256": "2" * 64}
    monkeypatch.setattr(B.C, "check", lambda *a, **k: dict(report))
    kw = dict(retained=str(retained), summary_path=str(summary), base_manifest=str(base),
              extension_manifest=str(tmp_path / "extension.json"), calibration=str(tmp_path / "calibration.json"))
    return kw, retained, ext, summary, report


def _digests(root):
    return {os.path.relpath(os.path.join(d, f), root): hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest()
            for d, _, fs in os.walk(root) for f in fs}


def test_the_product_copies_the_original_years_unchanged_and_lists_every_file(tmp_path, monkeypatch):
    B = _load()
    kw, retained, ext, _, _ = _summary_world(tmp_path, B, monkeypatch)
    before = (_digests(retained), _digests(ext))
    out = B.build(str(tmp_path / "product"), str(ext), "unused", "unused", **kw)
    names = B.export_artifacts(out, str(tmp_path / "artifacts_dir"), campaigns=(str(retained), str(ext)))
    assert (_digests(retained), _digests(ext)) == before                                  # neither campaign written to
    assert B.verify_product(out) == []
    man = json.load(open(os.path.join(out, "MANIFEST.json")))
    for y, src in ((2001, retained), (2002, retained), (2003, ext)):
        for rel, name in ((f"tracks/era5_{y}_tracks.mat", "tracker_port.mat"), (f"records/tracking_era5_{y}.json", f"tracking_era5_{y}.json")):
            original = (src / f"era5_{y}" / name).read_bytes()
            assert open(os.path.join(out, rel), "rb").read() == original
            assert man["files"][rel]["sha256"] == hashlib.sha256(original).hexdigest()
    assert man["coverage"]["years"] == [2001, 2003] and man["coverage"]["extension_years"] == [2003, 2003]
    season = json.load(open(os.path.join(out, "season_summary_africa_jjas.json")))
    assert sorted(season["years"]) == ["2001", "2002", "2003"] and season["years"]["2003"]["campaign"] == "extension campaign"
    assert open(names["manifest"], "rb").read() == open(os.path.join(out, "MANIFEST.json"), "rb").read()
    assert B.export_artifacts(out, str(tmp_path / "artifacts_dir"), campaigns=(str(retained), str(ext))) == names   # repeatable
    with pytest.raises(SystemExit, match="exists and a product is never written into"):
        B.build(out, str(ext), "unused", "unused", **kw)


def test_a_failed_artifact_export_is_completed_by_repeating_it(tmp_path, monkeypatch):
    """A review planted a failure publishing the second artifact after the product went
    final, and the retry was refused. The product is the release now, and the export can
    be repeated: what is present must be identical, what is missing is made."""
    B = _load()
    kw, retained, ext, _, _ = _summary_world(tmp_path, B, monkeypatch)
    out = B.build(str(tmp_path / "product"), str(ext), "unused", "unused", **kw)
    real, calls = B.publish_bytes, []

    def fail_second(dir_fd, name, blob):
        calls.append(name)
        if len(calls) == 2:
            raise OSError("planted failure on the second artifact")
        real(dir_fd, name, blob)
    monkeypatch.setattr(B, "publish_bytes", fail_second)
    with pytest.raises(OSError, match="planted failure"):
        B.export_artifacts(out, str(tmp_path / "a"), campaigns=(str(retained), str(ext)))
    assert len(os.listdir(tmp_path / "a")) == 1
    monkeypatch.setattr(B, "publish_bytes", real)
    names = B.export_artifacts(out, str(tmp_path / "a"), campaigns=(str(retained), str(ext)))
    for key, inside in B.ARTIFACT_FILES:
        assert open(names[key], "rb").read() == open(os.path.join(out, inside), "rb").read()
    open(names["manifest"], "ab").write(b" ")
    with pytest.raises(SystemExit, match="exists and is not the product's MANIFEST.json"):
        B.export_artifacts(out, str(tmp_path / "a"), campaigns=(str(retained), str(ext)))


def test_an_incomplete_or_altered_product_exports_nothing(tmp_path, monkeypatch):
    B = _load()
    kw, retained, ext, _, _ = _summary_world(tmp_path, B, monkeypatch)
    out = B.build(str(tmp_path / "product"), str(ext), "unused", "unused", **kw)
    with open(os.path.join(out, "tracks", "era5_2001_tracks.mat"), "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(SystemExit, match="tracks/era5_2001_tracks.mat does not have the digest the manifest lists"):
        B.export_artifacts(out, str(tmp_path / "a"), campaigns=(str(retained), str(ext)))
    os.remove(os.path.join(out, B.MARKER))
    with pytest.raises(SystemExit, match="has no completion marker"):
        B.export_artifacts(out, str(tmp_path / "a"), campaigns=(str(retained), str(ext)))
    assert not (tmp_path / "a").exists()


def test_no_destination_inside_either_campaign_is_accepted(tmp_path, monkeypatch):
    """A review published the artifacts into the retained campaign. Destinations are
    resolved through symbolic links and refused inside either campaign before anything is
    created."""
    B = _load()
    kw, retained, ext, _, _ = _summary_world(tmp_path, B, monkeypatch)
    before = (_digests(retained), _digests(ext))
    with pytest.raises(SystemExit, match="lies inside the campaign"):
        B.build(str(retained / "product"), str(ext), "unused", "unused", **kw)
    with pytest.raises(SystemExit, match="lies inside the campaign"):
        B.build(str(ext / "era5_2003" / "product"), str(ext), "unused", "unused", **kw)
    os.symlink(retained, tmp_path / "looks_elsewhere")
    with pytest.raises(SystemExit, match="lies inside the campaign"):
        B.build(str(tmp_path / "looks_elsewhere" / "product"), str(ext), "unused", "unused", **kw)
    out = B.build(str(tmp_path / "product"), str(ext), "unused", "unused", **kw)
    for artifacts in (str(retained), str(ext / "artifacts"), str(tmp_path / "looks_elsewhere" / "x")):
        with pytest.raises(SystemExit, match="lies inside the campaign"):
            B.export_artifacts(out, artifacts, campaigns=(str(retained), str(ext)))
    assert (_digests(retained), _digests(ext)) == before


def test_an_original_year_that_does_not_reproduce_the_retained_summary_refuses_the_build(tmp_path, monkeypatch):
    B = _load()
    kw, retained, ext, summary, _ = _summary_world(tmp_path, B, monkeypatch)
    s = json.load(open(summary))
    s["years"]["2002"]["months"]["8"]["port"] += 1
    summary.write_text(json.dumps(s))
    with pytest.raises(SystemExit, match=r"2002: the recomputed row differs from the retained summary in \['months'\]"):
        B.build(str(tmp_path / "product"), str(ext), "unused", "unused", **kw)
    assert not (tmp_path / "product").exists() and not list(tmp_path.glob(".product-*"))


def test_a_failing_extension_check_or_a_changed_original_file_refuses_the_build(tmp_path, monkeypatch):
    B = _load()
    kw, retained, ext, _, report = _summary_world(tmp_path, B, monkeypatch)
    monkeypatch.setattr(B.C, "check", lambda *a, **k: dict(report, passed=False))
    with pytest.raises(SystemExit, match="does not pass check_extension_records"):
        B.build(str(tmp_path / "product"), str(ext), "unused", "unused", **kw)
    monkeypatch.setattr(B.C, "check", lambda *a, **k: dict(report))
    rec = retained / "era5_2001" / "tracking_era5_2001.json"
    r = json.load(open(rec))
    r["dataset_specific"]["tracks_sha256"] = "0" * 64                                    # the record no longer names these tracks
    rec.write_text(json.dumps(r))
    with pytest.raises(SystemExit, match="2001: the tracks file beside the record does not have the digest"):
        B.build(str(tmp_path / "product"), str(ext), "unused", "unused", **kw)


def test_campaigns_that_overlap_or_leave_a_gap_are_refused(tmp_path, monkeypatch):
    B = _load()
    kw, retained, ext, _, report = _summary_world(tmp_path, B, monkeypatch)
    os.rename(ext / "era5_2003", ext / "era5_2004")
    p = ext / "era5_2004" / "tracking_era5_2003.json"
    os.rename(p, ext / "era5_2004" / "tracking_era5_2004.json")
    monkeypatch.setattr(B.C, "check", lambda *a, **k: dict(report, tracking_years=[2004, 2004]))
    with pytest.raises(SystemExit, match="do not tile the years once each"):
        B.build(str(tmp_path / "product"), str(ext), "unused", "unused", **kw)


@pytest.mark.parametrize("marker", [{}, "drop the season summary", [], "not json"])
def test_a_completion_marker_that_does_not_name_exactly_both_files_certifies_nothing(tmp_path, monkeypatch, marker):
    """A review exported altered statistics past a marker that no longer named the season
    summary, and a partial release past an empty one."""
    B = _load()
    kw, retained, ext, _, _ = _summary_world(tmp_path, B, monkeypatch)
    out = B.build(str(tmp_path / "product"), str(ext), "unused", "unused", **kw)
    path = os.path.join(out, B.MARKER)
    named = json.load(open(path))
    if marker == "drop the season summary":
        named.pop("season_summary_africa_jjas.json")
        open(path, "w").write(json.dumps(named))
        with open(os.path.join(out, "season_summary_africa_jjas.json"), "a") as fh:
            fh.write(" ")                                                       # the statistics altered
    elif marker == "not json":
        open(path, "w").write("{")
    else:
        open(path, "w").write(json.dumps(marker))
    problems = B.verify_product(out)
    assert problems and ("completion marker" in problems[0]), problems
    with pytest.raises(SystemExit, match="the product is not complete"):
        B.export_artifacts(out, str(tmp_path / "a"), campaigns=(str(retained), str(ext)))


def test_an_artifacts_directory_replaced_after_the_check_cannot_redirect_the_export(tmp_path, monkeypatch):
    """A review replaced the artifacts directory with a link into the retained campaign
    after the path check, and both artifacts landed there. The export now writes through a
    descriptor whose own ancestry is checked, so the replacement is refused."""
    B = _load()
    kw, retained, ext, _, _ = _summary_world(tmp_path, B, monkeypatch)
    out = B.build(str(tmp_path / "product"), str(ext), "unused", "unused", **kw)
    before = _digests(retained)
    artifacts = tmp_path / "artifacts"
    real_check = B.refuse_destinations_inside

    def check_then_replace(destinations, campaigns):
        real_check(destinations, campaigns)
        if artifacts.is_dir() and not artifacts.is_symlink():
            artifacts.rmdir()
        if not artifacts.exists():
            os.symlink(retained, artifacts)                                   # replaced after the check
    artifacts.mkdir()
    monkeypatch.setattr(B, "refuse_destinations_inside", check_then_replace)
    with pytest.raises(SystemExit, match="lies inside a campaign"):
        B.export_artifacts(out, str(artifacts), campaigns=(str(retained), str(ext)))
    assert _digests(retained) == before


def test_an_older_product_exported_by_a_newer_builder_keeps_its_own_version(tmp_path, monkeypatch):
    """A review exported the retained rc2 product with the rc3 builder and got files named
    rc3 holding rc2. Copies are named by the version the verified product declares."""
    B = _load()
    kw, retained, ext, _, _ = _summary_world(tmp_path, B, monkeypatch)
    current = B.VERSION
    monkeypatch.setattr(B, "VERSION", "0.9.0-older")
    older = B.build(str(tmp_path / "older"), str(ext), "unused", "unused", **kw)
    monkeypatch.setattr(B, "VERSION", current)
    newer = B.build(str(tmp_path / "newer"), str(ext), "unused", "unused", **kw)
    camps = (str(retained), str(ext))
    a = tmp_path / "artifacts"

    def named_by_their_own_version(names, version, product):
        for key, inside in B.ARTIFACT_FILES:
            assert os.path.basename(names[key]) == f"era5_record_1979_2025_{version}_{key}.json"
            assert json.load(open(names[key]))["version"] == version
            assert open(names[key], "rb").read() == open(os.path.join(product, inside), "rb").read()
    old_names = B.export_artifacts(older, str(a), campaigns=camps)
    named_by_their_own_version(old_names, "0.9.0-older", older)
    new_names = B.export_artifacts(newer, str(a), campaigns=camps)
    named_by_their_own_version(new_names, current, newer)
    listing = sorted(os.listdir(a))
    assert len(listing) == 4
    assert B.export_artifacts(older, str(a), campaigns=camps) == old_names and sorted(os.listdir(a)) == listing   # repeatable
    with open(old_names["season_summary"], "a") as fh:
        fh.write(" ")
    with pytest.raises(SystemExit, match="exists and is not the product's season_summary_africa_jjas.json"):
        B.export_artifacts(older, str(a), campaigns=camps)                                                         # never overwritten


def test_a_product_whose_files_disagree_on_its_version_exports_nothing(tmp_path, monkeypatch):
    B = _load()
    kw, retained, ext, _, _ = _summary_world(tmp_path, B, monkeypatch)
    out = B.build(str(tmp_path / "product"), str(ext), "unused", "unused", **kw)
    season = os.path.join(out, "season_summary_africa_jjas.json")
    doc = json.load(open(season))
    doc["version"] = "0.0.1-other"
    open(season, "w").write(json.dumps(doc))
    marker = os.path.join(out, B.MARKER)
    named = json.load(open(marker))
    named["season_summary_africa_jjas.json"] = hashlib.sha256(open(season, "rb").read()).hexdigest()
    open(marker, "w").write(json.dumps(named))                          # complete by its marker, inconsistent inside
    assert B.verify_product(out) == []
    with pytest.raises(SystemExit, match="disagree on its identity"):
        B.export_artifacts(out, str(tmp_path / "a"), campaigns=(str(retained), str(ext)))
    assert not (tmp_path / "a").exists() or os.listdir(tmp_path / "a") == []


def test_a_file_replaced_after_its_version_is_read_cannot_be_exported_under_that_version(tmp_path, monkeypatch):
    """A confirmation replaced the season summary between the read that gave the version and
    the read that was published, and exported a file named by one version holding another.
    The version is now read from the very bytes that are published."""
    B = _load()
    kw, retained, ext, _, _ = _summary_world(tmp_path, B, monkeypatch)
    out = B.build(str(tmp_path / "product"), str(ext), "unused", "unused", **kw)
    season = os.path.join(out, "season_summary_africa_jjas.json")
    real = B.product_identity

    def identity_then_replace(blobs):
        found = real(blobs)
        doc = json.loads(open(season).read())
        doc["version"] = "0.0.1-other"
        open(season, "w").write(json.dumps(doc))                         # replaced after the version was read
        return found
    monkeypatch.setattr(B, "product_identity", identity_then_replace)
    names = B.export_artifacts(out, str(tmp_path / "a"), campaigns=(str(retained), str(ext)))
    for key, _ in B.ARTIFACT_FILES:
        assert json.load(open(names[key]))["version"] == B.VERSION
        assert os.path.basename(names[key]) == f"era5_record_1979_2025_{B.VERSION}_{key}.json"


def test_a_file_changed_after_verification_is_refused_even_at_the_same_version(tmp_path, monkeypatch):
    """The bytes read for export must be the ones the completion marker names, so a season
    summary altered after the product was verified, keeping its version, exports nothing."""
    B = _load()
    kw, retained, ext, _, _ = _summary_world(tmp_path, B, monkeypatch)
    out = B.build(str(tmp_path / "product"), str(ext), "unused", "unused", **kw)
    season = os.path.join(out, "season_summary_africa_jjas.json")
    real = B.verify_product

    def verify_then_alter(path):
        found = real(path)
        doc = json.loads(open(season).read())
        doc["years"]["2001"]["africa_origin"] += 1                         # same version, other statistics
        open(season, "w").write(json.dumps(doc))
        return found
    monkeypatch.setattr(B, "verify_product", verify_then_alter)
    with pytest.raises(SystemExit, match="season_summary_africa_jjas.json changed after the product was verified"):
        B.export_artifacts(out, str(tmp_path / "a"), campaigns=(str(retained), str(ext)))
    assert not (tmp_path / "a").exists() or os.listdir(tmp_path / "a") == []
