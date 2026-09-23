"""The committed accounting artifact must be what the checker on disk produces from the
committed evidence, today.

THE DEFECT THIS GATES. The residue closure was committed at fifteen of fifteen; the
validator was changed two commits later; nothing re-ran it; and the checker as it then
stood refused all fifteen while the artifact still said fifteen. The project's own rule
says a validator change requires revalidation, and a rule nobody runs is a wish.

TWO IDENTITIES, CHECKED SEPARATELY. The first version compared only the validator digest
the artifact records against the file on disk, and a review showed it passing in a tree
with every producer, residual, case, log and pinned output ABSENT. So the second test
reruns the accounting itself, through `main(argv)`, over the committed input set, and
requires the complete result to be identical. It costs about two seconds. The third test
regenerates the two operation cases through their producer and requires them to be
byte-identical to the committed artifacts, so a producer change that would change a case
is caught too. UNCHECKED NEVER MEANS PASSED: an artifact recording no digest fails, it is
not skipped.
"""
import glob
import hashlib
import importlib.util
import json
import os
import shutil
import sys

import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ARTIFACTS = os.path.join(ROOT, "docs", "aewc_v2", "artifacts")
EVIDENCE = os.path.join(ROOT, "docs", "aewc_v2", "evidence")
ARTIFACT = os.path.join(ARTIFACTS, "residue_membership_2026-09-22.json")
VALIDATOR = os.path.join(ROOT, "scripts", "residue_membership.py")
OPERATION_CASES = {"pair63_reordering": "pair63_reordering_case_2026-09-22.json",
                   "pair30_removal": "pair30_removal_case_2026-09-22.json"}
OPERATION_INPUTS = ["pair63_reorder_runs_2026-09-22.json",
                    "south_atlantic_start_case_2026-09-21.json",
                    "pair30_removal_runs_2026-09-22.json", "peru_case_2026-09-20.json",
                    "tracker_oracle_residuals_2026-09-18.json"]

# THE ACCOUNTING AND ITS EVIDENCE ARE NOT PUBLISHED, so in the public tree this gate has
# nothing to check. The skip is explicit, and the private suite runs it every time.
if not os.path.exists(ARTIFACT) or not os.path.isdir(EVIDENCE):
    pytest.skip("the committed accounting and its evidence are not present in this tree",
                allow_module_level=True)


def _digest(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _load(name):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def committed_accounting_command(out):
    """The exact argument list the committed artifact was produced with."""
    return (["--residuals", os.path.join(ARTIFACTS, "tracker_oracle_residuals_2026-09-18.json"),
             "--cases", *sorted(glob.glob(os.path.join(ARTIFACTS, "*_case_*.json"))),
             "--reference-log", *sorted(glob.glob(os.path.join(EVIDENCE, "reference_log_*.log.gz"))),
             "--reference-output", *sorted(glob.glob(os.path.join(EVIDENCE, "reference_output_*.mat"))),
             os.path.join(EVIDENCE, "port_output_9a72b9272ad9.mat"),
             os.path.join(EVIDENCE, "tracker_port.mat"),
             "--manifest", os.path.join(ARTIFACTS, "reference_logs.json"),
             "--out", str(out)])


def test_every_case_artifact_on_disk_is_an_input_of_the_committed_accounting():
    """A no-folder review noted the gate's case glob would miss a case file named with a
    hyphen, and the rerun would then reproduce the old artifact and stay green. So every
    JSON in the artifacts directory that has the shape of a case is required to be among
    the inputs the accounting records, whatever it is called."""
    artifact = json.load(open(ARTIFACT))
    recorded = set(artifact["input_sha256"])
    for path in sorted(glob.glob(os.path.join(ARTIFACTS, "*.json"))):
        try:
            body = json.load(open(path))
        except ValueError:
            continue
        if isinstance(body, dict) and "explains" in body and "intervention" in body:
            assert os.path.basename(path) in recorded, (
                f"{os.path.basename(path)} looks like a case and the accounting did not "
                f"read it. Regenerate over every case")


def test_the_manifest_the_accounting_read_is_the_one_on_disk():
    artifact = json.load(open(ARTIFACT))
    recorded = artifact.get("manifest_sha256")
    assert recorded, "the accounting records no manifest digest. Regenerate it"
    assert recorded == _digest(os.path.join(ARTIFACTS, "reference_logs.json"))


def test_the_committed_accounting_was_produced_by_the_current_validator():
    artifact = json.load(open(ARTIFACT))
    recorded = artifact.get("validator_sha256")
    assert recorded, ("the accounting artifact records no validator digest, so nothing "
                      "says which validator produced it; regenerate it")
    assert recorded == _digest(VALIDATOR), (
        "the accounting artifact was produced by a different validator than the one on "
        "disk; regenerate it, and read the count that follows rather than the old one")


def test_rerunning_the_accounting_over_the_committed_evidence_reproduces_the_artifact(
        tmp_path, monkeypatch):
    pytest.importorskip("scipy")
    monkeypatch.chdir(ROOT)
    M = _load("residue_membership")
    out = tmp_path / "rerun.json"
    code = M.main(committed_accounting_command(out))
    assert code == 0, "the checker on disk refuses the committed evidence"
    rerun = json.load(open(out))
    committed = json.load(open(ARTIFACT))
    assert rerun == committed, (
        "the checker on disk produces a different accounting from the committed evidence "
        "than the committed artifact. Regenerate it and read the count that follows")


@pytest.mark.parametrize("key", sorted(OPERATION_CASES))
def test_regenerating_an_operation_case_reproduces_the_committed_artifact(
        tmp_path, monkeypatch, key):
    pytest.importorskip("scipy")
    for name in OPERATION_INPUTS:
        shutil.copy(os.path.join(ARTIFACTS, name), tmp_path / name)
    monkeypatch.chdir(ROOT)
    W = _load("write_operation_cases")
    monkeypatch.setattr(W, "ARTIFACTS", str(tmp_path))
    assert W.main(["--only", key]) == 0
    regenerated = (tmp_path / OPERATION_CASES[key]).read_bytes()
    committed = open(os.path.join(ARTIFACTS, OPERATION_CASES[key]), "rb").read()
    assert regenerated == committed, (
        f"{OPERATION_CASES[key]} is not what its producer writes today. Regenerate it")
