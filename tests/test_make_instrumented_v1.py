"""The instrumented copy's patch, bound on a synthetic source that carries its anchors.

WHY THIS FILE EXISTS. A review mutated the builder so that it printed the NEGATION of the
field version 1 contours, and nothing caught it: the measurement the fourth case rests on
comes out of this patch, and no test referenced the builder at all. The archived source
itself is untracked data, so a test reading it would SKIP under the mutation checker,
which stages tracked files only. The anchors are therefore exercised on a synthetic
source small enough to live here.

MUTATION LIST, written before the assertions:
  B1 the FIELD block prints a different expression from the one the contour call
     contours (the review's `-acrvt`);
  B2 the FIELD block is inserted somewhere other than immediately before the contour
     call, so the field it prints is not the field that was contoured;
  B3 the guard is dropped, so the copy prints at every timestep and is no longer inert
     with an empty dump list;
  B4 a missing anchor is patched anyway rather than refused.
"""
import importlib.util
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _load():
    spec = importlib.util.spec_from_file_location(
        "make_instrumented_v1", os.path.join(HERE, "..", "scripts", "make_instrumented_v1.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_instrumented_v1"] = mod
    spec.loader.exec_module(mod)
    return mod


def synthetic_source(M):
    """A source carrying every anchor the patch needs, in the order find_ews_f.m has
    them, and nothing else."""
    return "\n".join([
        M.SIGNATURE,
        "  %Mask 2",
        "  acrvt(id) = nan;",
        "  %Identify easterly wave troughs",
        M.CONTOUR_CALL,
        "  id = find(ch(1,:) == tr_thr);",
        M.COARSE_MERGE,
        M.AFTER_COARSE,
        M.CLEAR_LINE,
        "  more code",
        M.PRUNE_END,
        "",
    ])


def emitted_field_printf(block):
    """The FIELD printf as (format, arguments), parsed from the patched text, so a test
    can say WHAT is printed and in WHICH order rather than that a substring occurs
    somewhere. A review printed the absolute value, twice the value, the longitude in the
    latitude column and one column fewer, and every one passed a substring test."""
    call = block[block.index("printf('FIELD"):]
    call = call[:call.index(");") + 1]
    call = call.replace("...", "").replace("\n", " ")
    inner = call[call.index("(") + 1:call.rindex(")")]
    fmt = inner[inner.index("'") + 1:inner.index("'", inner.index("'") + 1)]
    rest, args, depth, current = inner[inner.index(",") + 1:], [], 0, ""
    for ch in rest:                       # split on TOP-LEVEL commas, since every
        if ch == "(":                     # argument carries an (rr,cc) subscript
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(current.strip())
            current = ""
        else:
            current += ch
    args.append(current.strip())
    return fmt, [a for a in args if a]


def test_the_field_block_prints_the_field_the_contour_call_contours():
    M = _load()
    patched = M.patch(synthetic_source(M))
    where = patched.index(M.CONTOUR_CALL)
    before = patched[:where]
    block = before[before.rindex("if any(abs(time(t) - AEWDBG_TIMES)"):]
    assert "printf('FIELD" in block
    fmt, args = emitted_field_printf(block)
    # B1: the record is the time, the two coordinates and the contoured value, in that
    # order, each printed once
    assert fmt == "FIELD %.4f %.4f %.4f %.12e\\n"
    assert args == ["time(t)", "latgrid_c(rr,cc)", "longrid_c(rr,cc)", "acrvt(rr,cc)"]
    assert "acrvt" in M.CONTOUR_CALL
    # B2: nothing but the block's own closing lies between it and the contour call, so the
    # field printed is the field contoured and nothing reassigns it in between
    tail = block[block.index("acrvt(rr,cc)"):]
    assert tail.strip().endswith("end")
    assert "contours(" not in block
    between = before[before.index(block) + len(block):]
    assert "acrvt" not in between.replace("%Identify easterly wave troughs", "")
    # B3: and it is guarded by the dump-time test, so an empty list prints nothing
    assert "if any(abs(time(t) - AEWDBG_TIMES) < 1e-6);" in block
    # every cell it prints is one version 1 did not mask, which is what carries the mask,
    # and the loops cover the whole grid rather than part of it
    assert "~isnan(acrvt(rr,cc))" in patched
    assert "for rr = 1:size(acrvt,1);" in block and "for cc = 1:size(acrvt,2);" in block


def test_the_contour_anchor_is_the_call_the_archive_makes():
    """The anchor is what binds the dump to the field that gets contoured. A review
    changed it to contour twice the field, which passes a synthetic source built from the
    anchor itself and no longer matches the archive."""
    M = _load()
    assert M.CONTOUR_CALL == "  ch = contours(longrid_c,latgrid_c,acrvt,[tr_thr,tr_thr]);"
    archive = os.path.join(HERE, "..", "data", "aewc_v2_pilot", "v1_src", "find_ews_f.m")
    if not os.path.exists(archive):
        pytest.skip("the archived source is untracked data and is not in this tree")
    with open(archive) as fh:
        assert M.CONTOUR_CALL in fh.read()


def test_the_patch_refuses_a_source_whose_anchors_have_moved():
    M = _load()
    source = synthetic_source(M)
    for anchor, name in ((M.CONTOUR_CALL, "contour call"), (M.COARSE_MERGE, "coarse merge"),
                         (M.SIGNATURE, "signature")):
        with pytest.raises(SystemExit) as e:
            M.patch(source.replace(anchor, "% the archive moved this line"))
        assert name in str(e.value)


def test_the_copy_is_inert_until_a_dump_time_matches():
    M = _load()
    patched = M.patch(synthetic_source(M))
    # one global, five guarded blocks, and every guard is the same time test
    assert patched.count("if any(abs(time(t) - AEWDBG_TIMES) < 1e-6)") == 5
    assert "global AEWDBG_TIMES" in patched
    assert "if isempty(AEWDBG_TIMES); AEWDBG_TIMES = []; end" in patched
    for kind in ("FIELD", "AXIS", "AXISPTS", "COARSE", "FINE", "TRACK"):
        assert f"printf('{kind} " in patched


def test_the_output_path_refuses_the_repository_and_anything_holding_it(tmp_path):
    """A review found the first guard accepting the REPOSITORY ITSELF, which the next line
    deletes whole, archived source and all. It rejected only paths starting with the
    repository plus a separator."""
    M = _load()
    repo = str(tmp_path / "repo")
    archive = os.path.join(repo, "data", "v1_src")
    os.makedirs(archive)
    # EACH REFUSAL FOR ITS OWN REASON, since a later check refusing a path for an
    # unrelated reason would hide the missing one: a review's mutation that accepted the
    # repository itself passed a test that only asked whether something was refused.
    for bad, reason in ((repo, "the repository or a directory containing it"),
                        (os.path.dirname(repo), "the repository or a directory containing it"),
                        (archive, "the archived source itself"),
                        (os.path.join(archive, "deeper"), "the archived source itself"),
                        (os.path.join(repo, "inside"), "inside the repository")):
        problems = M.output_path_problems(bad, repo=repo, archive=archive)
        assert any(reason in w for w in problems), (bad, problems)
    assert M.output_path_problems(str(tmp_path / "elsewhere" / "v1_instrumented"),
                                  repo=repo, archive=archive) == []
    # a directory that is not a previous instrumented copy is not deleted either
    other = tmp_path / "somebody_elses_work"
    (other / "sub").mkdir(parents=True)
    (other / "sub" / "notes.txt").write_text("keep me\n")
    assert any("was not written by this script" in w
               for w in M.output_path_problems(str(other), repo=repo, archive=archive))
    # ANOTHER CHECKOUT'S ARCHIVE holds find_ews_f.m and is the record, so holding that
    # file cannot be what makes a directory disposable. A review handed the guard exactly
    # that path and watched it pass.
    other_archive = tmp_path / "another_checkout" / "v1_src"
    other_archive.mkdir(parents=True)
    (other_archive / "find_ews_f.m").write_text("function ews = find_ews_f()\n")
    assert any(M.MARKER in w for w in
               M.output_path_problems(str(other_archive), repo=repo, archive=archive))
    previous = tmp_path / "v1_instrumented"
    previous.mkdir()
    (previous / "find_ews_f.m").write_text("function ews = find_ews_f()\n")
    (previous / M.MARKER).write_text("written by the builder\n")
    assert M.output_path_problems(str(previous), repo=repo, archive=archive) == []


def test_main_writes_the_marker_and_can_replace_its_own_output(tmp_path, capsys):
    """B5, and the guard is only half a control without it: the marker is what makes a
    directory disposable, so a copy written without one could never be replaced, and a
    run that skipped writing it would leave the next run refusing its own output."""
    M = _load()
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "find_ews_f.m").write_text(synthetic_source(M))
    (archive / "other_file.m").write_text("% carried along\n")
    M.ARCHIVE, M.REPO = str(archive), str(tmp_path / "repo")
    out = tmp_path / "v1_instrumented"
    assert M.main(["--out", str(out)]) == 0
    assert (out / M.MARKER).exists()
    assert (out / "other_file.m").exists()
    assert "printf('FIELD" in (out / "find_ews_f.m").read_text()
    # the archived source is untouched, and a second run replaces its own output
    assert (archive / "find_ews_f.m").read_text() == synthetic_source(M)
    assert M.main(["--out", str(out)]) == 0
    # a directory this script did not write is refused instead
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "find_ews_f.m").write_text("the archive of another checkout\n")
    with pytest.raises(SystemExit) as e:
        M.main(["--out", str(foreign)])
    assert M.MARKER in str(e.value)
    assert (foreign / "find_ews_f.m").read_text() == "the archive of another checkout\n"
