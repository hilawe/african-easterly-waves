"""The committed threshold artifact must match the source that claims to have produced it.

WHY THIS EXISTS. The artifact carries per-file source hashes so a published number can be
traced to the code behind it. That guarantee is worth nothing if nobody checks it, and it
was already broken once: the first committed artifact was generated before the script
reached its committed form, so its fingerprint named source that no longer existed while its
numbers happened to still be right. A review caught it by regenerating. Nothing in the suite
would have.

THIS RUNS IN MILLISECONDS AND DOES NOT REDO THE CALCULATION, which is the point. Rerunning
the diagnostic takes minutes and needs the ERA-Interim tree, so it cannot be a suite gate.
Comparing hashes can be, and it catches the failure that actually happened: an artifact left
behind by an edit to its own producer.

WHAT IT DOES NOT ESTABLISH. That the numbers are right, or that the inputs still exist. It
establishes only that the artifact was produced by the source now in the tree. A number can
be faithfully produced by wrong code.
"""
import hashlib
import importlib.util
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SCRIPT = os.path.join(_ROOT, "scripts", "diagnose_threshold_population.py")
_ARTIFACT = os.path.join(_ROOT, "docs", "aewc_v2", "artifacts",
                         "threshold_population_diagnostic.json")

_spec = importlib.util.spec_from_file_location("diagnose_threshold_population", _SCRIPT)
diag = importlib.util.module_from_spec(_spec)
sys.modules["diagnose_threshold_population"] = diag
_spec.loader.exec_module(diag)


@pytest.fixture
def artifact():
    if not os.path.exists(_ARTIFACT):
        pytest.skip("no committed threshold artifact")
    with open(_ARTIFACT) as fh:
        return json.load(fh)


def test_the_artifact_was_produced_by_the_source_now_in_the_tree(artifact):
    """The failure that actually happened, and it names WHICH file moved.

    Per-file hashes rather than one combined digest exist for this message. A single
    fingerprint tells you something changed; this tells you it was `load.py`, which is the
    difference between a puzzle and a fix.
    """
    recorded = artifact.get("source_sha256")
    assert recorded, "the artifact must carry per-file source hashes"
    current = diag._source_hashes()

    assert set(recorded) == set(current), (
        f"the artifact and the script disagree about WHICH files matter. "
        f"only in artifact: {sorted(set(recorded) - set(current))}, "
        f"only in script: {sorted(set(current) - set(recorded))}")

    moved = [f for f in current if recorded[f] != current[f]]
    assert not moved, (
        f"the committed artifact was produced by different source than the tree now holds. "
        f"Files that moved: {moved}. Regenerate it with the command in the script's "
        f"docstring, and do so only once the script is final.")


def test_every_module_the_numbers_pass_through_is_fingerprinted():
    """The second half of the same failure, and the one hashes cannot catch by themselves.

    The first version of this list omitted `vorticity.py`, which supplies the curvature the
    whole diagnostic is about. Every recorded hash matched and the fingerprint still
    certified a provenance it had not checked, because the file it needed to watch was not
    in the list. Comparing hashes cannot detect an absent entry, so the list itself is
    asserted here against the modules the computation actually imports.
    """
    required = {
        "src/aew/v1port/load.py",        # the sampler and the climatology
        "src/aew/v1port/climatology.py",  # smoothing, decimation, the anomaly
        "src/aew/v1port/pipeline.py",     # curvature_from_winds
        "src/aew/v1port/vorticity.py",    # component_vorticity, the curvature itself
        "scripts/diagnose_threshold_population.py",
    }
    listed = set(diag._SOURCE_FILES)
    assert required <= listed, (
        f"these modules produce the numbers and are not fingerprinted: "
        f"{sorted(required - listed)}")
    root = diag._repo_root()
    for rel in listed:
        assert os.path.exists(os.path.join(root, rel)), \
            f"{rel} is fingerprinted but does not exist, so the hash is of nothing"


def test_the_artifact_records_the_environment_and_full_length_hashes(artifact):
    """A published artifact records what it ran on, and does not truncate its hashes."""
    env = artifact.get("environment")
    assert env, "the artifact must record the library versions the arithmetic ran on"
    for key in ("python", "numpy"):
        assert env.get(key), f"the artifact must record the {key} version"

    for name, digest in artifact["source_sha256"].items():
        assert len(digest) == 64, (
            f"{name} carries a {len(digest)}-character hash, and a published artifact "
            f"uses a full SHA-256")
    for name, digest in artifact.get("input_file_sha256", {}).items():
        assert len(digest) == 64, f"{name} carries a truncated input hash"


def test_the_artifact_states_what_it_cannot_settle(artifact):
    """The claim-width note travels with the numbers, not only in the prose that cites them.

    The conclusion this artifact supports is narrow: it rules out raw curvature carried
    through the port's own chain, not every reading of the archive's wording. That sentence
    belongs in the artifact, because an artifact outlives the document quoting it.
    """
    note = artifact.get("what_this_can_settle", "")
    assert "tracker-equivalent" in note, \
        "the artifact must record that it bears on the tracker-equivalent recipe only"


def test_the_recorded_input_hashes_are_of_files_that_were_actually_read(artifact):
    """An empty input-hash map would satisfy every other assertion here."""
    files = artifact.get("input_file_sha256", {})
    years = artifact.get("years", [])
    assert len(files) == 2 * len(years), (
        f"expected a u700 and a v700 file per year, {2 * len(years)} for {len(years)} "
        f"years, but the artifact records {len(files)}")
    for name in files:
        assert name.endswith(".nc"), f"{name} is not a retrieval file"


def test_the_hash_helper_reads_whole_files():
    """`_sha256` streams in chunks, so a mistake there would silently hash a prefix."""
    import tempfile

    payload = os.urandom(3 * (1 << 20) + 7)      # spans several read chunks
    with tempfile.NamedTemporaryFile(delete=False) as fh:
        fh.write(payload)
        path = fh.name
    try:
        assert diag._sha256(path) == hashlib.sha256(payload).hexdigest()
    finally:
        os.unlink(path)


# THE COMPLETE SET IS FROZEN HERE, independently of any artifact's own claims. A first
# version of the matrix gate looped only over hashes the artifact happened to record, so
# an artifact OMITTING vorticity.py, the exact historical failure this gate exists for,
# would have passed it. The set is stated here, not read from the artifact or the script.
MATRIX_REQUIRED_SOURCES = frozenset({
    "src/aew/v1port/load.py",
    "src/aew/v1port/climatology.py",
    "src/aew/v1port/pipeline.py",
    "src/aew/v1port/vorticity.py",
    "src/aew/v1port/geometry.py",
    "src/aew/v1port/percentile.py",
    "src/aew/v1port/thresholds.py",
    "scripts/compute_thresholds.py",
})


def matrix_artifact_problems(artifact):
    """Every way a matrix artifact fails source consistency, as strings. Pure, so the
    committed-artifact gate and a synthetic exercise share one implementation."""
    problems = []
    recorded = artifact.get("source_sha256") or {}
    missing = MATRIX_REQUIRED_SOURCES - set(recorded)
    extra = set(recorded) - MATRIX_REQUIRED_SOURCES
    if missing:
        problems.append(f"omits required modules: {sorted(missing)}")
    if extra:
        problems.append(f"fingerprints unexpected files: {sorted(extra)}")
    for rel in sorted(MATRIX_REQUIRED_SOURCES & set(recorded)):
        full = os.path.join(_ROOT, rel)
        if not os.path.exists(full):
            problems.append(f"fingerprints missing file {rel}")
            continue
        with open(full, "rb") as fh:
            current = hashlib.sha256(fh.read()).hexdigest()
        if current != recorded[rel]:
            problems.append(f"{rel} moved since the artifact was produced")
    return problems


def test_the_matrix_gate_logic_works_before_any_artifact_exists():
    """The gate is exercised synthetically NOW, because it will spend its life skipped
    until P1-T0 appears, and a gate first executed on the artifact it guards is a gate
    nobody has watched fail."""
    good = {"source_sha256": {}}
    for rel in MATRIX_REQUIRED_SOURCES:
        with open(os.path.join(_ROOT, rel), "rb") as fh:
            good["source_sha256"][rel] = hashlib.sha256(fh.read()).hexdigest()
    assert matrix_artifact_problems(good) == []

    omitting = {"source_sha256": {k: v for k, v in good["source_sha256"].items()
                                  if not k.endswith("vorticity.py")}}
    problems = matrix_artifact_problems(omitting)
    assert any("omits" in p and "vorticity" in p for p in problems),         "the historical failure, an omitted module, must be named"

    moved = {"source_sha256": dict(good["source_sha256"],
                                   **{"src/aew/v1port/load.py": "0" * 64})}
    assert any("moved" in p for p in matrix_artifact_problems(moved))

    empty = {"source_sha256": {}}
    assert matrix_artifact_problems(empty), "no hashes at all must fail loudly"


def test_every_committed_matrix_artifact_matches_current_source():
    """The generic gate: every committed thresholds_P*-T*.json must fingerprint the
    complete frozen module set AND match the tree. Skips only while none exists."""
    import glob as _glob

    pattern = os.path.join(_ROOT, "docs", "aewc_v2", "artifacts",
                           "thresholds_P[12]-T[0-5].json")
    paths = sorted(_glob.glob(pattern))
    if not paths:
        pytest.skip("no committed matrix artifacts yet")
    for path in paths:
        with open(path) as fh:
            artifact = json.load(fh)
        problems = matrix_artifact_problems(artifact)
        assert not problems, f"{os.path.basename(path)}: {problems}"


# --- one key, one meaning: the signed relative difference across every producer -------

# EVERY EXCLUSION IS A DELIBERATE ONE, with its reason. The provisional thirty-year
# artifact was produced by the retired version of compute_thresholds.py, which stored
# the ABSOLUTE relative difference under the same key. It is provisional and non-gating
# (HANDOFF, session 13), its producer no longer exists, and regenerating it would forge
# provenance, so it is kept as the record of what was believed and named here so a
# reader knows its two relative differences are unsigned magnitudes.
LEGACY_UNSIGNED_ARTIFACTS = {"thresholds_eraint_tracking_domain_1981_2010.json"}


def relative_difference_problems(artifact, name):
    """Every way an artifact's `expected` block departs from value / expected - 1.

    Three producers write `coarse_relative_difference` and `fine_relative_difference`
    (compute_thresholds.py, diagnose_threshold_population.py, and the retired script
    behind the legacy artifact above), and the defect this guards is one key carrying
    two meanings in one directory. Exact equality, not a tolerance, because the value is
    derived from two numbers the same artifact records.
    """
    problems = []
    exp = artifact.get("expected")
    if not isinstance(exp, dict):
        return problems
    blocks = artifact.get("populations")
    if isinstance(blocks, dict):
        pairs = [(f"populations.{label}", block) for label, block in blocks.items()]
    else:
        pairs = [("", artifact)]
    for prefix, block in pairs:
        for which in ("coarse", "fine"):
            value = block.get(which)
            if isinstance(value, dict):
                value = value.get("threshold")
            if value is None:
                value = block.get(f"{which}_threshold")
            key = f"{which}_relative_difference"
            holder = block if key in block else exp
            if key not in holder or value is None:
                continue
            want = value / exp[which] - 1.0
            if holder[key] != want:
                problems.append(f"{name}: {prefix}{key} is {holder[key]!r}, "
                                f"value / expected - 1 gives {want!r}")
    return problems


def test_every_committed_artifact_uses_the_signed_relative_difference():
    import glob as _glob

    paths = sorted(_glob.glob(os.path.join(_ROOT, "docs", "aewc_v2", "artifacts",
                                           "*.json")))
    if not paths:
        # the public export carries no artifacts; a skip is an explicit non-claim,
        # never a pass, and the source tree always has them
        pytest.skip("no committed threshold artifacts in this tree")
    checked, problems = 0, []
    for path in paths:
        name = os.path.basename(path)
        if name in LEGACY_UNSIGNED_ARTIFACTS:
            continue
        with open(path) as fh:
            artifact = json.load(fh)
        found = relative_difference_problems(artifact, name)
        if "expected" in artifact:
            checked += 1
        problems.extend(found)
    assert not problems, problems
    assert checked >= 3, "the gate must actually reach the artifacts it exists for"


def test_the_signed_difference_gate_fires_on_the_legacy_convention():
    """THE CANARY. The legacy artifact carries the unsigned convention, so the gate,
    pointed at it without the exclusion, must refuse. A gate that passes the file it
    excludes has not been watched failing."""
    path = os.path.join(_ROOT, "docs", "aewc_v2", "artifacts",
                        "thresholds_eraint_tracking_domain_1981_2010.json")
    if not os.path.exists(path):
        pytest.skip("legacy artifact not present")
    with open(path) as fh:
        legacy = json.load(fh)
    problems = relative_difference_problems(legacy, "legacy")
    assert len(problems) == 2, problems
    assert all("coarse" in p or "fine" in p for p in problems)

    # and a synthetic artifact in the current schema with one flipped sign
    forged = {"expected": {"coarse": 2.0, "fine": 4.0,
                           "coarse_relative_difference": -0.5,
                           "fine_relative_difference": 0.5},
              "coarse": {"threshold": 1.0}, "fine": {"threshold": 2.0}}
    problems = relative_difference_problems(forged, "forged")
    assert len(problems) == 1 and "fine_relative_difference" in problems[0]


def test_the_determinism_record_matches_the_committed_primary_artifacts():
    """The bit-identical claim in THRESHOLD_PRIMARY_CASES.md rests on this record, so the
    record's second-run values must be the committed artifacts' values, and the two runs
    it records must actually agree. Neither is trusted from the record's own flag."""
    path = os.path.join(_ROOT, "docs", "aewc_v2", "artifacts",
                        "determinism_primary_cells_2026-09-15.json")
    if not os.path.exists(path):
        pytest.skip("no determinism record in this tree (the public export carries none)")
    with open(path) as fh:
        rec = json.load(fh)
    for cell in ("P1-T0", "P2-T0"):
        runs = rec["cells"][cell]["runs"]
        assert len(runs) >= 2
        last = runs[-1]
        with open(os.path.join(_ROOT, "docs", "aewc_v2", "artifacts",
                               f"thresholds_{cell}.json")) as fh:
            committed = json.load(fh)
        assert last["coarse_threshold"] == committed["coarse"]["threshold"]
        assert last["fine_threshold"] == committed["fine"]["threshold"]
        assert last["compute_thresholds_py_sha256"] == \
            committed["source_sha256"]["scripts/compute_thresholds.py"]
        for r in runs:
            for key in ("coarse_threshold", "fine_threshold", "coarse_finite_count",
                        "fine_finite_count"):
                assert r[key] == runs[0][key], (cell, r["run"], key)
        # the runs used DIFFERENT versions of the producing script, which is the
        # whole reason more than one run exists
        hashes = [r["compute_thresholds_py_sha256"] for r in runs]
        assert len(set(hashes)) == len(hashes), cell
