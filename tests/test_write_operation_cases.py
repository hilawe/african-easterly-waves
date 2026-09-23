"""The operation-case producer refuses retained runs that are not the experiment it transcribes.

THE DEFECT THIS GATES. A review mutated a retained identity block, the case digest to
sixty-four zeroes, the reference digest to ones, the reference index to another pair and the
diagnostic digest to twos, left every trajectory untouched, and both the producer's
`main(argv)` and the accounting's credited pair 63. The producer had checked only that the
named runs existed, then copied the contradiction into `evidence_binding`, which nothing
reads. Every test here drives `main(argv)`, the command path, over copies of the real
retained runs and source cases, and reads the real evidence directory, which is not written.
"""
import importlib.util
import json
import os
import shutil
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
ARTIFACTS = os.path.join(ROOT, "docs", "aewc_v2", "artifacts")
NEEDED = ["pair63_reorder_runs_2026-09-22.json", "south_atlantic_start_case_2026-09-21.json",
          "pair30_removal_runs_2026-09-22.json", "peru_case_2026-09-20.json",
          "tracker_oracle_residuals_2026-09-18.json"]

pytest.importorskip("scipy")
# THE RETAINED EVIDENCE IS NOT PUBLISHED. The public export carries the code and the tests
# but not the retained runs, cases or pinned outputs, so in that tree these tests cannot
# run. A skip is explicit; it is not a pass, and the private suite runs them every time.
if not os.path.exists(os.path.join(ARTIFACTS, NEEDED[0])):
    pytest.skip("the retained runs and cases are not present in this tree",
                allow_module_level=True)


def _load():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    spec = importlib.util.spec_from_file_location(
        "write_operation_cases", os.path.join(ROOT, "scripts", "write_operation_cases.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def staged(tmp_path, monkeypatch):
    """Copies of the retained runs and source cases in a scratch artifacts directory."""
    for name in NEEDED:
        shutil.copy(os.path.join(ARTIFACTS, name), tmp_path / name)
    monkeypatch.chdir(ROOT)
    W = _load()
    monkeypatch.setattr(W, "ARTIFACTS", str(tmp_path))
    return W, tmp_path


def _mutate(path, **identity):
    retained = json.loads(path.read_text())
    retained["identity"].update(identity)
    path.write_text(json.dumps(retained))


def test_the_real_retained_runs_produce_a_case_whose_binding_was_checked(staged):
    W, art = staged
    assert W.main(["--only", "pair63_reordering"]) == 0
    case = json.loads((art / "pair63_reordering_case_2026-09-22.json").read_text())
    binding = case["evidence_binding"]
    assert binding["identity_checked_against"] == "south_atlantic_start_case_2026-09-21.json"
    assert binding["diagnostic_script"] == "pair63_finished_track.py"
    assert binding["diagnostic_script_sha256_now"]
    assert "bound to independently checked execution evidence" in \
        case["intervention"]["receipts_basis"]


@pytest.mark.parametrize("field, value, reason", [
    ("case_sha256", "0" * 64, "produced from case input"),
    ("reference_output_sha256", "1" * 64, "no retained evidence file has the digest"),
    ("reference_track_index", 30, "judged against reference track 30"),
    ("step", 33026.5, "intervene at 33026.5"),
    ("reference_output", "tracker_port.mat", "name reference output tracker_port.mat"),
    ("comparison", "track length and first time only", "exact whole-array comparison"),
    ("case_file", "tracker_port.mat", "rather than tracker_case.mat"),
    ("script", "pair30_merge_input.py", "this case transcribes pair63_finished_track.py"),
    ("exact_reproduction_by_intervention_alone", False, "as true"),
])
def test_a_contradictory_retained_identity_is_refused_through_the_command(
        staged, field, value, reason):
    W, art = staged
    _mutate(art / "pair63_reorder_runs_2026-09-22.json", **{field: value})
    with pytest.raises(SystemExit) as caught:
        W.main(["--only", "pair63_reordering"])
    assert reason in str(caught.value), str(caught.value)
    assert not (art / "pair63_reordering_case_2026-09-22.json").exists()


def test_the_reviewers_exact_mutation_is_refused_and_names_every_contradiction(staged):
    W, art = staged
    _mutate(art / "pair63_reorder_runs_2026-09-22.json", case_sha256="0" * 64,
            reference_output_sha256="1" * 64, reference_track_index=30,
            script_sha256="2" * 64)
    with pytest.raises(SystemExit) as caught:
        W.main(["--only", "pair63_reordering"])
    text = str(caught.value)
    assert text.startswith("REFUSED")
    for fragment in ("case input", "reference track 30", "no retained evidence file"):
        assert fragment in text, text


@pytest.mark.parametrize("field", ["case_sha256", "reference_output_sha256",
                                   "reference_output", "reference_track_index", "step",
                                   "case_file", "comparison", "script", "script_sha256",
                                   "exact_reproduction_by_intervention_alone"])
def test_every_binding_field_of_the_retained_identity_is_read(staged, field):
    """THE CLASS, not an instance. Three rounds running found a recorded field the
    producer copied without reading, so this deletes each field the binding rests on and
    requires a refusal, which is what a field being READ means."""
    W, art = staged
    path = art / "pair63_reorder_runs_2026-09-22.json"
    retained = json.loads(path.read_text())
    del retained["identity"][field]
    path.write_text(json.dumps(retained))
    with pytest.raises(SystemExit) as caught:
        W.main(["--only", "pair63_reordering"])
    assert str(caught.value).startswith("REFUSED"), str(caught.value)


def test_a_retained_identity_without_a_step_is_refused(staged):
    """Deleting the step skipped the check and the producer supplied its own."""
    W, art = staged
    path = art / "pair63_reorder_runs_2026-09-22.json"
    retained = json.loads(path.read_text())
    del retained["identity"]["step"]
    path.write_text(json.dumps(retained))
    with pytest.raises(SystemExit) as caught:
        W.main(["--only", "pair63_reordering"])
    assert "no numeric intervention step" in str(caught.value)
    _mutate(path, step="soon")
    with pytest.raises(SystemExit) as caught:
        W.main(["--only", "pair63_reordering"])
    assert "no numeric intervention step" in str(caught.value)


def test_a_judged_reference_with_swapped_track_indices_is_refused(staged, tmp_path, monkeypatch):
    """The judged reference must hold the same tracks AT THE SAME INDICES as the pinned
    one. A sorted-multiset comparison accepted a file with tracks 30 and 63 swapped, and
    index 63 then named a different track than the one the runs were judged against."""
    import hashlib
    from scipy.io import loadmat, savemat
    W, art = staged
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    real = os.path.join(ROOT, "docs", "aewc_v2", "evidence")
    for name in os.listdir(real):
        if name.endswith(".mat"):
            os.symlink(os.path.join(real, name), evidence / name)
    runs = json.loads((art / "pair63_reorder_runs_2026-09-22.json").read_text())
    judged = W.retained_output(runs["identity"]["reference_output_sha256"])
    m = {k: v for k, v in loadmat(judged).items() if not k.startswith("__")}
    for k in ("time", "lat", "lon"):
        m[f"{k}30"], m[f"{k}63"] = m[f"{k}63"], m[f"{k}30"]
    swapped = evidence / "reference_output_swapped.mat"
    savemat(str(swapped), m)
    digest = hashlib.sha256(swapped.read_bytes()).hexdigest()
    monkeypatch.setattr(W, "EVIDENCE", str(evidence))
    _mutate(art / "pair63_reorder_runs_2026-09-22.json", reference_output_sha256=digest,
            reference_output="reference_output_swapped.mat")
    with pytest.raises(SystemExit) as caught:
        W.main(["--only", "pair63_reordering"])
    assert "AT THE SAME INDICES" in str(caught.value), str(caught.value)


def test_a_different_reference_file_with_identical_tracks_is_accepted_by_content(staged):
    """Four retained reference outputs hold identical finished tracks under different
    digests. The runs name one and the source case pins another, and that is accepted
    because the CONTENT is compared, which is also what the removal case relies on."""
    W, art = staged
    source = json.loads((art / "peru_case_2026-09-20.json").read_text())
    runs = json.loads((art / "pair30_removal_runs_2026-09-22.json").read_text())
    assert (source["input_sha256"]["tracker_octave_instrumented.mat"]
            != runs["identity"]["reference_output_sha256"])
    assert W.main(["--only", "pair30_removal"]) == 0
    case = json.loads((art / "pair30_removal_case_2026-09-22.json").read_text())
    # pair 30's recorded diagnostic is not the source on disk today, and the artifact says so
    assert case["evidence_binding"]["diagnostic_script_matches_current_source"] is False
    assert "does NOT match the current source" in case["intervention"]["receipts_basis"]
