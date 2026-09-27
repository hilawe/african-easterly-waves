"""The pilot figure: drawn from a retained artifact and the inputs it names, refusing an
input whose digest is not the artifact's, and never overwriting its output."""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _load():
    spec = importlib.util.spec_from_file_location("fig_qtrack_pairing_pilot", os.path.join(ROOT, "scripts", "fig_qtrack_pairing_pilot.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_figure_is_drawn_from_an_artifact_and_its_verified_inputs(tmp_path):
    import test_qtrack_pairing_pilot as T
    M = T._load()
    evidence, qdir, adir, regions = T._world(tmp_path)
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    out_art = art_dir / "pairing_1990_2026-09-27.json"
    assert M.run(1990, evidence, qdir, adir, regions, str(out_art)) == 0
    F = _load()
    png = tmp_path / "fig.png"
    assert F.main(["--artifacts", str(art_dir), "--years", "1990", "--out", str(png)]) == 0
    assert png.exists() and png.stat().st_size > 1000
    art = json.load(open(out_art))
    drawn = F.render([(1990, art)], str(tmp_path / "again.png"))
    assert drawn[1990]["pairs"] == 2 and drawn[1990]["a_none"] == drawn[1990]["drawn_a_none"] and drawn[1990]["b_none"] == drawn[1990]["drawn_b_none"]
    with pytest.raises(SystemExit):
        F.main(["--artifacts", str(art_dir), "--years", "1990", "--out", str(png)])
    with open(os.path.join(evidence, "era5_1990", "tracker_port.mat"), "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(SystemExit):                                     # the input is no longer the artifact's
        F.render([(1990, art)], str(tmp_path / "third.png"))
