"""The matrix report reads twelve artifacts and refuses to look complete when it is not.

MUTATION LIST, written before the assertions: drop the missing-cell refusal; drop the
case_id check; drop the mixed-source-version refusal; drop the expected-block check;
render a missing row without the INCOMPLETE title; format the difference unsigned;
drop the current-source-hash comparison; drop the registry settings check; drop the
transformation-object comparison; drop the expected-block recomputation; drop the
coarse-scale consistency check.

A review found the first version passed a synthetic set with arbitrary hashes and
settings, so the fixtures now carry the CURRENT tree's fingerprint and the registry's
settings, and the two new refusals are watched firing on a stale hash and a settings
mismatch.
"""
import importlib.util
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "report_threshold_matrix", os.path.join(_HERE, "..", "scripts",
                                            "report_threshold_matrix.py"))
rep = importlib.util.module_from_spec(_spec)
sys.modules["report_threshold_matrix"] = rep
_spec.loader.exec_module(rep)


CURRENT_HASHES = rep._cli_module().source_hashes()


def synthetic(cid, coarse=5.0e-7, fine=2.5e-6, version=None):
    """A producible artifact: the transformation object is the frozen table's entry,
    the coarse threshold is the unscaled value scaled in that direction, and the
    expected block is the CLI's own judgement of the thresholds."""
    case = rep.T.matrix_case(cid)
    passes, scale = rep.T.TRANSFORMATIONS[case["transformation"]]
    hashes = dict(CURRENT_HASHES) if version is None else {
        k: version * 64 for k in CURRENT_HASHES}
    unscaled = coarse
    coarse = rep.T.apply_coarse_scale(unscaled, scale)
    return {"case_id": cid, "estimator": "exact",
            "source_sha256": hashes,
            "climatology_years": list(case["climatology_years"]),
            "population_years": list(case["population_years"]),
            "transformation": {"id": case["transformation"], "smoothing_passes": passes,
                               "coarse_scale": scale},
            "domain": {"lat_range": list(case["lat_range"]),
                       "lon_range": list(case["lon_range"])},
            "coarse_resolution": case["coarse_resolution"],
            "subsample_stride": case["subsample"], "prefix": case["prefix"],
            "grids": {"timesteps": 46752},
            "coarse": {"threshold": coarse, "threshold_unscaled": unscaled},
            "fine": {"threshold": fine},
            "expected": {"coarse": 7.16e-7, "fine": 2.8e-6,
                         "coarse_relative_difference": coarse / 7.16e-7 - 1.0,
                         "fine_relative_difference": fine / 2.8e-6 - 1.0,
                         "criterion": 0.1,
                         "reproduces": max(abs(coarse / 7.16e-7 - 1.0),
                                           abs(fine / 2.8e-6 - 1.0)) <= 0.1}}


@pytest.fixture
def full(tmp_path):
    for cid in rep.cell_ids():
        with open(tmp_path / f"thresholds_{cid}.json", "w") as fh:
            json.dump(synthetic(cid), fh)
    return str(tmp_path)


def test_a_complete_matrix_renders_twelve_rows_with_signed_differences(full, tmp_path):
    out = str(tmp_path / "report.md")
    assert rep.main(["--artifact-dir", full, "--out", out]) == 0
    text = open(out).read()
    assert "INCOMPLETE" not in text and "MISSING" not in text
    rows = [ln for ln in text.splitlines() if ln.startswith("| P")]
    assert len(rows) == 12 and [r.split(" | ")[0].strip("| ") for r in rows] == rep.cell_ids()
    # 5.0e-7 / 7.16e-7 - 1 = -30.2 percent, written with its sign, and the verdict
    # (P1-T0 is unscaled; the T2 row two below shows the same value scaled by 1/0.9)
    assert "| -30.2% |" in rows[0] and "| -10.7% |" in rows[0] and "| FAILS |" in rows[0]
    assert "| -22.4% |" in rows[4] and rows[4].startswith("| P1-T2 |")
    assert "| 46,752 |" in rows[0] and "thresholds_P1-T0.json" in rows[0]


def test_a_missing_cell_is_refused_unless_allowed_and_then_labelled(full, tmp_path):
    os.remove(os.path.join(full, "thresholds_P2-T3.json"))
    assert rep.main(["--artifact-dir", full, "--out", str(tmp_path / "r.md")]) == 2
    out = str(tmp_path / "partial.md")
    assert rep.main(["--artifact-dir", full, "--allow-missing", "--out", out]) == 0
    text = open(out).read()
    assert text.splitlines()[0].startswith("# The threshold run matrix")
    assert "INCOMPLETE, 1 missing: P2-T3" in text.splitlines()[0]
    assert "| P2-T3 | " in text and "| MISSING |" in text
    assert sum(ln.startswith("| P") for ln in text.splitlines()) == 12


def test_a_mislabelled_artifact_is_refused(full, tmp_path):
    with open(os.path.join(full, "thresholds_P1-T4.json"), "w") as fh:
        json.dump(synthetic("P1-T5"), fh)
    assert rep.main(["--artifact-dir", full, "--out", str(tmp_path / "r.md")]) == 2


def test_mixed_source_versions_are_refused(full, tmp_path):
    with open(os.path.join(full, "thresholds_P2-T5.json"), "w") as fh:
        json.dump(synthetic("P2-T5", version="b"), fh)
    assert rep.main(["--artifact-dir", full, "--out", str(tmp_path / "r.md")]) == 2


def test_an_artifact_without_the_expected_block_is_refused(full, tmp_path):
    a = synthetic("P1-T1")
    del a["expected"]
    with open(os.path.join(full, "thresholds_P1-T1.json"), "w") as fh:
        json.dump(a, fh)
    assert rep.main(["--artifact-dir", full, "--out", str(tmp_path / "r.md")]) == 2


def test_a_consistent_set_from_old_code_is_refused(tmp_path):
    """Twelve artifacts that AGREE with each other but fingerprint code no longer in
    the tree: the mixed-version refusal cannot see this, only the comparison against
    the current files can. A first version of this test staled one artifact and was
    caught by the mixed-version check instead, so it bound nothing."""
    for cid in rep.cell_ids():
        with open(tmp_path / f"thresholds_{cid}.json", "w") as fh:
            json.dump(synthetic(cid, version="0"), fh)
    assert rep.main(["--artifact-dir", str(tmp_path),
                     "--out", str(tmp_path / "r.md")]) == 2


def test_settings_that_contradict_the_case_id_are_refused(full, tmp_path):
    """A P2-T3 artifact recording P1's population years, or the wrong transformation,
    is a mislabeled run whatever its filename says."""
    for field, value in (("population_years", [1979, 2010]),
                         ("transformation", {"id": "T2", "smoothing_passes": 1,
                                             "coarse_scale": "up"})):
        a = synthetic("P2-T3")
        a[field] = value
        with open(os.path.join(full, "thresholds_P2-T3.json"), "w") as fh:
            json.dump(a, fh)
        assert rep.main(["--artifact-dir", full, "--out", str(tmp_path / "r.md")]) == 2, field


def _write(full, cid, a):
    with open(os.path.join(full, f"thresholds_{cid}.json"), "w") as fh:
        json.dump(a, fh)


def _refused(full, tmp_path):
    return rep.main(["--artifact-dir", full, "--out", str(tmp_path / "r.md")]) == 2


def test_the_reviewers_forgery_is_refused(full, tmp_path):
    """Thresholds set to zero and the metadata to 99 passes with coarse scaling on,
    hashes and case id kept: the first version of this generator tabled it."""
    a = synthetic("P1-T0")
    a["coarse"]["threshold"] = 0.0
    a["fine"]["threshold"] = 0.0
    a["transformation"] = {"id": "T0", "smoothing_passes": 99, "coarse_scale": "up"}
    _write(full, "P1-T0", a)
    assert _refused(full, tmp_path)


def test_a_transformation_object_that_contradicts_the_table_is_refused(full, tmp_path):
    a = synthetic("P1-T1")
    a["transformation"]["smoothing_passes"] = 1           # id says T1, two passes
    _write(full, "P1-T1", a)
    assert _refused(full, tmp_path)


def test_a_displayed_difference_or_verdict_that_disagrees_with_the_thresholds_is_refused(
        full, tmp_path):
    a = synthetic("P2-T2")
    a["expected"]["fine_relative_difference"] = -0.01
    _write(full, "P2-T2", a)
    assert _refused(full, tmp_path)
    a = synthetic("P2-T2")
    a["expected"]["reproduces"] = True
    _write(full, "P2-T2", a)
    assert _refused(full, tmp_path)


def test_a_coarse_threshold_not_equal_to_its_scaled_unscaled_value_is_refused(full,
                                                                              tmp_path):
    a = synthetic("P1-T3")
    a["coarse"]["threshold_unscaled"] = a["coarse"]["threshold"]  # T3 must be 0.9 x
    _write(full, "P1-T3", a)
    assert _refused(full, tmp_path)
