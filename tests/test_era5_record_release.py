"""Every committed version of the ERA5 1979 to 2025 product's artifacts, with no data
directory: the season rows for the original years must be the retained summary's ERA5 column
in every field, the manifest must list each original track and record file with the digest
and size of the retained evidence copy, every version's inventory must be complete, and
every version must carry the same data files and season rows as every other, since the
versions differ only in the code that built them. Skipped only where the retained evidence
is absent, as in the public export, which does not carry it."""
import glob
import hashlib
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
EVIDENCE = os.path.join(ROOT, "docs", "aewc_v2", "evidence", "validation", "protocol_campaign")
ARTIFACTS = os.path.join(ROOT, "docs", "aewc_v2", "artifacts")
SUMMARY = os.path.join(ARTIFACTS, "protocol_campaign_summary_v5_2026-10-02.json")
sys.path.insert(0, os.path.join(ROOT, "scripts"))

pytestmark = pytest.mark.skipif(not os.path.isdir(EVIDENCE), reason="the retained campaign evidence is not in this tree")


def _builder():
    spec = importlib.util.spec_from_file_location("build_era5_record_product_audited", os.path.join(ROOT, "scripts", "build_era5_record_product.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _versions():
    found = sorted(glob.glob(os.path.join(ARTIFACTS, "era5_record_1979_2025_*_season_summary.json")))
    return [os.path.basename(p)[len("era5_record_1979_2025_"):-len("_season_summary.json")] for p in found]


def test_at_least_one_version_is_committed():
    assert _versions(), "no committed ERA5 1979 to 2025 product artifacts"


@pytest.mark.parametrize("version", _versions())
def test_the_original_years_rows_are_the_retained_summarys_era5_column(version):
    B = _builder()
    rows = json.load(open(os.path.join(ARTIFACTS, f"era5_record_1979_2025_{version}_season_summary.json")))["years"]
    retained = json.load(open(SUMMARY))["years"]
    years = [y for y in rows if rows[y]["campaign"].startswith("original campaign")]
    assert years == [str(y) for y in range(1979, 2011)]
    for y in years:
        assert B.row_differences(rows[y], B.retained_row(retained[y])) == [], y


@pytest.mark.parametrize("version", _versions())
def test_the_manifest_lists_every_original_file_with_the_retained_copys_digest(version):
    files = json.load(open(os.path.join(ARTIFACTS, f"era5_record_1979_2025_{version}_manifest.json")))["files"]
    for y in range(1979, 2011):
        for rel, name in ((f"tracks/era5_{y}_tracks.mat", "tracker_port.mat"), (f"records/tracking_era5_{y}.json", f"tracking_era5_{y}.json")):
            path = os.path.join(EVIDENCE, f"era5_{y}", name)
            with open(path, "rb") as fh:
                blob = fh.read()
            assert files[rel]["sha256"] == hashlib.sha256(blob).hexdigest() and files[rel]["bytes"] == len(blob), rel
            assert files[rel]["origin"] == "original campaign, retained evidence"
    assert sorted(r for r in files if r.startswith("tracks/")) == sorted(f"tracks/era5_{y}_tracks.mat" for y in range(1979, 2026))


EXPECTED_FILES = ({f"tracks/era5_{y}_tracks.mat" for y in range(1979, 2026)}
                  | {f"records/tracking_era5_{y}.json" for y in range(1979, 2026)}
                  | {"protocol/manifest_2026-09-25.json", "protocol/manifest_extension_era5_2011_2025_2026-10-03.json",
                     "protocol/thresholds_protocol_era5_1979_2010.json"})


def _payload(version):
    files = json.load(open(os.path.join(ARTIFACTS, f"era5_record_1979_2025_{version}_manifest.json")))["files"]
    rows = json.load(open(os.path.join(ARTIFACTS, f"era5_record_1979_2025_{version}_season_summary.json")))["years"]
    return files, rows


@pytest.mark.parametrize("version", _versions())
def test_every_versions_inventory_is_complete(version):
    files, rows = _payload(version)
    assert set(files) == EXPECTED_FILES
    assert sorted(rows) == [str(y) for y in range(1979, 2026)]
    assert all(rows[str(y)]["campaign"] == "extension campaign" for y in range(2011, 2026))
    assert all(files[f"tracks/era5_{y}_tracks.mat"]["origin"] == "extension campaign" for y in range(2011, 2026))


def test_every_version_carries_the_same_data_and_rows():
    """A review changed a 2025 row, a later year's track digest, and dropped a record entry,
    and the per-version checks accepted all three. The later years have no retained
    authority in git, so every version is held to every other instead."""
    versions = _versions()
    first_files, first_rows = _payload(versions[0])
    for version in versions[1:]:
        files, rows = _payload(version)
        assert {k: (v["sha256"], v["bytes"]) for k, v in files.items()} == \
               {k: (v["sha256"], v["bytes"]) for k, v in first_files.items()}, version
        assert rows == first_rows, version
