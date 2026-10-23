"""The eastern descriptive product's inventory builder."""

import hashlib
import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location("product_inventory_under_test", os.path.join(ROOT, "scripts", "eastern_product_inventory.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_groups_digests_and_refusals(tmp_path):
    M = _load()
    d = tmp_path / "x"; d.mkdir()
    (d / "a_r1.json").write_bytes(b"one"); (d / "a.json").write_bytes(b"old"); (d / "b.md").write_bytes(b"two")
    groups = (("L9", "g", "R.md", ("x/a_r1.*", "x/b.md", "x/a_r1.json")),)
    rows = M.inventory(str(tmp_path), groups)
    r = rows[0]
    assert [f["path"] for f in r["contents"]] == ["x/a_r1.json", "x/b.md"], "the earlier version is left out and a file matched twice counts once"
    assert r["files"] == 2 and r["bytes"] == 6 and r["record"] == "/".join(("docs", "aewc_v2", "R.md")), "the record name is placed under the records directory"
    line = "".join(f"{p}\t{hashlib.sha256(c).hexdigest()}\n" for p, c in (("x/a_r1.json", b"one"), ("x/b.md", b"two")))
    assert r["sha256"] == hashlib.sha256(line.encode()).hexdigest()
    text = M.inventory_text(rows)
    assert f"| L9 | g | R.md | 2 | 6 | {r['sha256'][:16]} |" in text and "total 2 files, 6 bytes" in text
    try:
        M.inventory(str(tmp_path), (("L9", "g", "R.md", ("x/a_r1.*", "x/missing_*")),)); assert False
    except SystemExit as ex:
        assert "no file matches x/missing_*" in str(ex)
    (d / "sub").mkdir()
    assert [f["path"] for f in M.inventory(str(tmp_path), (("L9", "g", "R.md", ("x/*",)),))[0]["contents"]] == ["x/a.json", "x/a_r1.json", "x/b.md"], "directories are not files"
    assert len(M.GROUPS) == 21 and {g[0] for g in M.GROUPS} == {"L1", "L2", "L3", "L4"}


def test_the_real_groups_resolve_in_the_repository(tmp_path):
    M = _load()
    if not (os.path.isdir(os.path.join(ROOT, M.EXTENT, "xband")) and os.path.isdir(os.path.join(ROOT, "docs/aewc_v2/eastern_descriptive_layer_v1"))):
        pytest.skip("the eastern records are not in this copy of the repository")
    rows = M.inventory(ROOT)
    by = {r["group"]: r for r in rows}
    assert by["month records of the 700 hPa band means, 24 seasons"]["files"] == 96 and by["band means extended to 10 W"]["files"] == 96
    assert all(r["files"] > 0 for r in rows)
    names = [f["path"] for r in rows for f in r["contents"]]
    assert not any(n.endswith("eval60_hist_archive.json") or n.endswith("eval60_lookback_archive.json") or n.endswith("eval60_twosided_archive.json") or n.endswith("eval60_lookback_constructions_r1.json") or n.endswith("eval60_lookback_constructions.json") for n in names), "superseded builds are not listed"
    assert by["QTrack look-back"]["files"] == 7 and by["minima histories"]["files"] == 7 and by["two-sided look-back"]["files"] == 5
    assert by["westward-history subset, version 1"]["files"] == 3 and by["westward histories, lobe position"]["files"] == 4
    assert not any("eastern_westward_histories_v1/" in n or "eastern_westward_histories_v1_r1/" in n for n in names), "the current build is listed, not the earlier ones"
    assert not any("lobe_feasibility" in n or n.endswith(("eval60_lagreg_archive_r1.json", "eval60_shape_archive_r1.json", "eval60_inject_archive.json")) for n in names), "only current builds"
    assert json.loads(json.dumps(rows)) == rows
    oj, om = tmp_path / "i.json", tmp_path / "i.md"
    assert M.main(["--root", ROOT, "--out-json", str(oj), "--out-md", str(om)]) == 0
    assert json.loads(oj.read_text())["groups"] == rows and om.read_text() == M.inventory_text(rows)
    try:
        M.main(["--root", ROOT, "--out-json", str(oj), "--out-md", str(tmp_path / "other.md")]); assert False
    except SystemExit as ex:
        assert "never overwritten" in str(ex)
