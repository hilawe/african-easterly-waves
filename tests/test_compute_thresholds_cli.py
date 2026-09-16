"""End-to-end tests for the rewritten threshold program and its artifact schema.

WHAT THESE BIND, beyond the module tests in test_thresholds.py, is the artifact-consistency
rules the contracts attach to every new field. Exactness is recomputed from the recorded
counts, never trusted from a flag. The source hashes are full length and match the tree.
Exact mode carries no sampling fields at all, their absence being part of the schema. The
coarse scale reaches the coarse threshold only. And the periods travel separately from
argument to artifact, which is the defect the old program existed to be rewritten out of.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

nc = pytest.importorskip("netCDF4")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from test_thresholds import (COARSE_RES, LAT_RANGE, LON_RANGE,  # noqa: E402
                             brute_force_population, write_year)

_spec = importlib.util.spec_from_file_location(
    "compute_thresholds_v2", os.path.join(_HERE, "..", "scripts",
                                          "compute_thresholds.py"))
cli = importlib.util.module_from_spec(_spec)
sys.modules["compute_thresholds_v2"] = cli
_spec.loader.exec_module(cli)


@pytest.fixture
def tree(tmp_path):
    for year in (1981, 1982, 1983):
        write_year(str(tmp_path), year)
    return str(tmp_path)


def run(tree_dir, out_path, *extra):
    args = ["--directory", tree_dir, "--prefix", "era5",
            "--climatology-years", "1981", "1983",
            "--population-years", "1982", "1983",
            "--lat-range", str(LAT_RANGE[0]), str(LAT_RANGE[1]),
            "--lon-range", str(LON_RANGE[0]), str(LON_RANGE[1]),
            "--out", out_path,
            # the synthetic years are eight steps, deliberately cheap, so every test
            # opts out of the full-calendar preflight EXPLICITLY; the refusal it opts
            # out of has its own test below
            "--allow-partial-years", *extra]
    return cli.main(args)


def test_exact_mode_end_to_end_with_a_consistent_artifact(tree, tmp_path):
    out = str(tmp_path / "artifact.json")
    assert run(tree, out, "--transformation", "T0", "--case-id", "test-T0") == 0
    with open(out) as fh:
        a = json.load(fh)

    # the thresholds themselves are the brute-force oracle's, bit for bit
    fine_ref, coarse_ref = brute_force_population(tree, [1982, 1983],
                                                  [1981, 1982, 1983], passes=1)
    for which, ref, q in (("fine", fine_ref, 66.0), ("coarse", coarse_ref, 55.0)):
        finite = ref[np.isfinite(ref)]
        assert a[which]["threshold"] == float(
            np.percentile(finite, q, method="linear"))
        assert a[which]["finite_count"] == finite.size
        assert a[which]["finite_count"] <= a[which]["domain_upper_bound"]

    # the periods travel separately, which is what the rewrite exists for
    assert a["climatology_years"] == [1981, 1983]
    assert a["population_years"] == [1982, 1983]
    assert a["transformation"] == {"id": "T0", "smoothing_passes": 1,
                                   "coarse_scale": None}
    assert a["percentile_method"] == "linear" and a["case_id"] == "test-T0"
    # the population definition is stated in the artifact, not inferred from the id
    pop = a["population"]
    assert pop["smoothing_passes"] == 1 and pop["wind_mask"] == "none"
    assert pop["hemispheres"] == "both" and pop["values"] == "every finite cell"
    assert pop["order"].startswith("decimate the buffered anomaly, crop")

    # full-length source hashes over every module the numbers pass through,
    # matching the tree they claim to describe
    assert set(a["source_sha256"]) == set(cli.SOURCE_FILES)
    for rel, digest in a["source_sha256"].items():
        assert len(digest) == 64, f"{rel} carries a truncated hash"
    assert a["source_sha256"] == cli.source_hashes(), \
        "the recorded hashes must match the tree that produced them"
    for name, digest in a["input_file_sha256"].items():
        assert len(digest) == 64 and name.endswith(".nc")
    for key in ("python", "numpy", "platform"):
        assert a["environment"].get(key)

    # EXACT MODE CARRIES NO SAMPLING FIELDS AT ALL, per the contract; their absence is
    # the record that nothing was drawn
    text = json.dumps(a)
    for forbidden in ("between_seed", "half_sample", "sampling_spread", "seed",
                      "per_step"):
        assert forbidden not in text, \
            f"an exact artifact must not carry {forbidden!r}"


def test_the_coarse_scale_reaches_the_coarse_threshold_only(tree, tmp_path):
    out0 = str(tmp_path / "t0.json")
    out2 = str(tmp_path / "t2.json")
    out3 = str(tmp_path / "t3.json")
    assert run(tree, out0, "--transformation", "T0") == 0
    assert run(tree, out2, "--transformation", "T2") == 0
    assert run(tree, out3, "--transformation", "T3") == 0
    a0, a2, a3 = (json.load(open(p)) for p in (out0, out2, out3))
    assert a2["coarse"]["threshold"] == a0["coarse"]["threshold"] / 0.9
    assert a3["coarse"]["threshold"] == a0["coarse"]["threshold"] * 0.9
    assert a2["coarse"]["threshold_unscaled"] == a0["coarse"]["threshold"]
    assert a2["fine"]["threshold"] == a0["fine"]["threshold"], \
        "the 90 percent note is about the coarse threshold and must never touch the fine"
    assert a3["fine"]["threshold"] == a0["fine"]["threshold"]


def test_ladder_mode_records_the_contract_block_and_exactness_is_recomputable(tree,
                                                                              tmp_path):
    out = str(tmp_path / "ladder.json")
    assert run(tree, out, "--estimator", "ladder") == 0
    with open(out) as fh:
        a = json.load(fh)
    block = a["between_seed"]
    assert block["seeds"] == [0, 1, 2, 3, 4]
    assert block["generator"] == "PCG64"
    assert block["range_formula"] == "(max - min) / abs(median)"
    assert block["gate"] == "both < 0.01 strictly"
    assert a["outcome"] == {"per_step": 400}, \
        "a tiny grid is fully exact at the first level"
    for entry in block["per_step_ladder"]:
        for which in ("coarse", "fine"):
            b = entry[which]
            # EXACTNESS IS COMPUTED FROM THE TWO COUNTS, never read from a flag
            recomputed = [s == v for s, v in zip(b["selected_counts"],
                                                 b["available_finite_counts"])]
            assert all(recomputed), "this tiny grid must be fully exact"
            assert b["exact_timestep_count"] == b["total_timestep_count"]
            assert len(b["estimates"]) == 5
            assert b["passes"] == (b["relative_range"] < 0.01)


def test_the_expected_pair_gate_sets_the_exit_status(tree, tmp_path):
    out = str(tmp_path / "expect.json")
    status = run(tree, out, "--expect", "1.0", "2.0")     # absurd pair, cannot reproduce
    assert status == 1
    with open(out) as fh:
        a = json.load(fh)
    e = a["expected"]
    assert e["reproduces"] is False
    assert e["criterion"] == 0.10
    # THE DIFFERENCE IS SIGNED, value / expected - 1, and a run far BELOW its target
    # must say so with a negative number. The pair is asymmetric so a swapped
    # comparison cannot pass. Expected values written out from the definition, not
    # read back from the module.
    assert e["coarse_relative_difference"] == a["coarse"]["threshold"] / 1.0 - 1.0
    assert e["fine_relative_difference"] == a["fine"]["threshold"] / 2.0 - 1.0
    assert e["coarse_relative_difference"] < 0 and e["fine_relative_difference"] < 0
    assert e["sign_convention"].startswith("value / expected - 1")

    # ONE SIDE EXACT AND THE OTHER FAR LOW MUST STILL FAIL, end to end. The fixture's
    # coarse threshold is negative, so only the fine side can be made exact here; the
    # four sign patterns are covered by test_the_judgement_is_on_each_side_separately.
    out2 = str(tmp_path / "expect_fine_exact.json")
    status = run(tree, out2, "--expect", "1.0", repr(a["fine"]["threshold"]))
    with open(out2) as fh:
        e2 = json.load(fh)["expected"]
    assert status == 1 and e2["reproduces"] is False
    assert e2["fine_relative_difference"] == 0.0


def test_the_judgement_is_on_each_side_separately():
    """Hand-written sign patterns against judge_against_expected, the pure function
    behind the artifact's `expected` block. Found by the external pre-commit check: with
    both live cases negative on both sides, dropping abs() on ONE side survived the
    end-to-end test because the coarse magnitude masked it."""
    j = cli.judge_against_expected
    # both within, mixed signs: reproduces
    r = j(1.05, 1.9, 1.0, 2.0)
    assert r["reproduces"] is True
    assert r["coarse_relative_difference"] == 1.05 / 1.0 - 1.0
    assert r["fine_relative_difference"] == 1.9 / 2.0 - 1.0
    # coarse exact, fine 50 percent LOW: a signed maximum would pass this
    r = j(1.0, 1.0, 1.0, 2.0)
    assert r["reproduces"] is False and r["fine_relative_difference"] == -0.5
    # fine exact, coarse 50 percent LOW
    r = j(0.5, 2.0, 1.0, 2.0)
    assert r["reproduces"] is False and r["coarse_relative_difference"] == -0.5
    # fine exact, coarse 50 percent HIGH
    r = j(1.5, 2.0, 1.0, 2.0)
    assert r["reproduces"] is False and r["coarse_relative_difference"] == 0.5
    # just inside the criterion is within, just past it is not (1.1 / 1.0 - 1.0 lands
    # one bit above 0.1 in floating point, so the boundary itself is not asserted)
    assert j(1.09, 2.0, 1.0, 2.0)["reproduces"] is True
    assert j(1.11, 2.0, 1.0, 2.0)["reproduces"] is False
    assert r["sign_convention"].startswith("value / expected - 1")
    assert r["criterion"] == 0.10


def test_the_domain_bounds_and_periods_are_required(tree, tmp_path):
    """No defaults for what defines a named case: argparse must refuse, not assume."""
    with pytest.raises(SystemExit):
        cli.main(["--directory", tree, "--prefix", "era5",
                  "--climatology-years", "1981", "1983",
                  "--population-years", "1982", "1983"])     # no domain bounds
    with pytest.raises(SystemExit):
        cli.main(["--directory", tree, "--prefix", "era5",
                  "--lat-range", "5", "20", "--lon-range", "-10", "9"])  # no periods
    with pytest.raises(SystemExit, match="backwards"):
        cli.main(["--directory", tree, "--prefix", "era5",
                  "--climatology-years", "1983", "1981",
                  "--population-years", "1982", "1983",
                  "--lat-range", "5", "20", "--lon-range", "-10", "9"])


def test_a_truncated_year_is_refused_unless_explicitly_allowed(tree, tmp_path, capsys):
    """The preflight's reason to exist: a partial retrieval must not become an artifact
    claiming the full period. The synthetic years are eight steps, so without the opt-out
    the run must refuse before any computation."""
    out = str(tmp_path / "x.json")
    with pytest.raises(ValueError, match="complete calendar year"):
        cli.main(["--directory", tree, "--prefix", "era5",
                  "--climatology-years", "1981", "1983",
                  "--population-years", "1982", "1983",
                  "--lat-range", str(LAT_RANGE[0]), str(LAT_RANGE[1]),
                  "--lon-range", str(LON_RANGE[0]), str(LON_RANGE[1]),
                  "--out", out])
    assert not os.path.exists(out), "a refused run must leave no artifact"
    capsys.readouterr()


def test_a_forged_matrix_case_is_refused_before_any_computation(tree, tmp_path, capsys):
    """The review's demonstration, as a regression: a ladder run over the wrong periods
    labelled P1-T0 must be refused with the differences listed, and no artifact written.
    Every violated setting must be named, so the refusal is a diff, not a shrug."""
    out = str(tmp_path / "forged.json")
    status = run(tree, out, "--case-id", "P1-T0", "--transformation", "T5",
                 "--estimator", "ladder")
    assert status == 2
    assert not os.path.exists(out), "a forged case must leave no artifact"
    text = capsys.readouterr().out
    assert "REFUSED" in text and "not the P1-T0 it claims" in text
    for named in ("climatology_years", "population_years", "transformation",
                  "estimator", "prefix", "expect", "allow-partial-years"):
        assert named in text, f"the refusal must name the violated setting {named}"


def test_a_free_form_case_id_stays_free(tree, tmp_path):
    """Diagnostic labels remain labels. Only the matrix pattern is a claim."""
    out = str(tmp_path / "diag.json")
    assert run(tree, out, "--case-id", "smoke-not-a-cell",
               "--transformation", "T5", "--estimator", "ladder") == 0
    assert json.load(open(out))["case_id"] == "smoke-not-a-cell"


def test_files_changing_during_the_run_refuse_the_artifact(tree, tmp_path, monkeypatch,
                                                           capsys):
    """Provenance is captured before the computation and must still hold after it. A
    tree that moved mid-run gets no artifact, because the hashes would describe a later
    state than the numbers came from."""
    out = str(tmp_path / "moved.json")
    real = cli.input_hashes
    calls = {"n": 0}

    def unstable(directory, prefix, years):
        calls["n"] += 1
        hashes = dict(real(directory, prefix, years))
        if calls["n"] > 1:                       # the after-capture sees a moved file
            first = sorted(hashes)[0]
            hashes[first] = "0" * 64
        return hashes

    monkeypatch.setattr(cli, "input_hashes", unstable)
    status = run(tree, out)
    assert status == 2
    assert not os.path.exists(out), "a moved tree must leave no artifact"
    assert "changed during the run" in capsys.readouterr().out


def test_cheap_guards_fire_before_expensive_work(tree, tmp_path):
    """Reversed bounds and a non-positive expected pair refuse immediately."""
    out = str(tmp_path / "g.json")
    base = ["--directory", tree, "--prefix", "era5",
            "--climatology-years", "1981", "1983",
            "--population-years", "1982", "1983", "--allow-partial-years",
            "--out", out]
    with pytest.raises(SystemExit, match="reversed"):
        cli.main(base + ["--lat-range", "20", "5", "--lon-range", "-10", "9"])
    with pytest.raises(SystemExit, match="non-positive"):
        cli.main(base + ["--lat-range", "5", "20", "--lon-range", "-10", "9",
                         "--expect", "-1e-7", "2.8e-6"])


def test_the_artifact_records_the_grid_and_calendar_facts(tree, tmp_path):
    """Cell counts alone cannot detect a shifted grid, so the artifact carries the
    shapes, bounds, orientation and per-year step counts a reader can check."""
    out = str(tmp_path / "meta.json")
    assert run(tree, out) == 0
    a = json.load(open(out))
    assert a["grids"]["fine_shape"] == [16, 20]      # 5..20N of 22..3N, -10..9E of -12..11E
    assert a["grids"]["timesteps"] == 16
    pf = a["preflight"]
    assert pf["steps_per_year"] == {"1981": 8, "1982": 8, "1983": 8}
    assert pf["row_order"] == "descending"
    assert pf["native_resolution"] == 1.0
    assert pf["complete_calendar_required"] is False
    assert os.path.isabs(a["directory"]) and a["prefix"] == "era5"


def test_a_matrix_typo_is_refused_with_guidance(tree, tmp_path, capsys):
    """P1-T6 is not a cell and not a diagnostic; it must refuse, not run unprotected."""
    out = str(tmp_path / "typo.json")
    assert run(tree, out, "--case-id", "P1-T6") == 2
    assert not os.path.exists(out)
    text = capsys.readouterr().out
    assert "matrix-looking namespace" in text and "P2-T5" in text and "smoke-" in text


def test_an_unprefixed_free_label_is_refused_too(tree, tmp_path):
    """Diagnostic labels carry an explicit prefix, so nothing ambiguous slips through."""
    out = str(tmp_path / "x.json")
    assert run(tree, out, "--case-id", "my-experiment") == 2


def test_a_matrix_case_binds_its_canonical_artifact_path(tree, tmp_path, capsys):
    """An official-looking P1-T0 under /tmp is a provenance hole with a correct label,
    so the out path is part of the case and a wrong one is a named violation."""
    out = str(tmp_path / "thresholds_P1-T0.json")     # right name, wrong place
    status = run(tree, out, "--case-id", "P1-T0", "--expect", "7.16e-7", "2.80e-6")
    assert status == 2
    text = capsys.readouterr().out
    assert "docs" in text and "artifacts" in text and "belongs at" in text


def test_the_artifact_write_is_atomic(tree, tmp_path, monkeypatch):
    """A failure during serialization must not truncate an existing artifact into an
    empty official file. The old artifact survives a failed rewrite untouched."""
    out = str(tmp_path / "keep.json")
    assert run(tree, out) == 0
    before = open(out).read()
    assert json.loads(before)["schema"] == "thresholds-v2"

    real_dump = json.dump

    def exploding(obj, fh, **kw):
        fh.write("{ partial garbage")
        raise OSError("disk vanished mid-serialization")

    monkeypatch.setattr(cli.json, "dump", exploding)
    with pytest.raises(OSError, match="disk vanished"):
        run(tree, out)
    monkeypatch.setattr(cli.json, "dump", real_dump)
    assert open(out).read() == before, \
        "the existing artifact must be byte-identical after the failed rewrite"
    assert not os.path.exists(out + ".tmp"), \
        "a failed write cleans up its temp file; a first version asserted this with an "\
        "'or True' that asserted nothing, which a review read correctly as vacuous"
