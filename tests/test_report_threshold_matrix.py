"""The matrix report reads twelve artifacts and refuses to look complete when it is not.

MUTATION LIST, written before the assertions: drop the missing-cell refusal; drop the
case_id check; drop the mixed-source-version refusal; drop the expected-block check;
render a missing row without the INCOMPLETE title; format the difference unsigned.
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


def synthetic(cid, coarse=5.0e-7, fine=2.5e-6, version="a"):
    return {"case_id": cid, "estimator": "exact",
            "source_sha256": {"m.py": version * 64},
            "grids": {"timesteps": 46752},
            "coarse": {"threshold": coarse}, "fine": {"threshold": fine},
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
    assert "| -30.2% |" in rows[0] and "| -10.7% |" in rows[0] and "| FAILS |" in rows[0]
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
