"""The principal figure is drawn from the origin re-tabulation and the monthly
tabulation after they are verified against each other, and is never overwritten."""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_figure_is_drawn_from_verified_artifacts_and_never_overwritten(tmp_path):
    import test_coast_crossing_measurement as T
    C = _load("coast_crossing_measurement")
    O = _load("coast_crossing_origin")
    B = _load("coast_crossing_by_month")
    F = _load("fig_intercomparison_principal")
    evidence, qdir, regions = T._world(tmp_path)
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    season = art_dir / "coast_crossing_1990_2026-09-27.json"
    assert C.main(["--year", "1990", "--campaign-evidence", evidence, "--qtrack-dir", qdir, "--regions-dir", regions, "--out", str(season)]) == 0
    origin = art_dir / "origin.json"
    assert O.main(["--artifacts", str(art_dir), "--years", "1990", "--out", str(origin), "--pattern", "coast_crossing_{year}_2026-09-27.json"]) == 0
    by_month = art_dir / "by_month.json"
    assert B.main(["--artifacts", str(art_dir), "--origin", str(origin), "--years", "1990", "--out", str(by_month)]) == 0
    png = tmp_path / "principal.png"
    assert F.main(["--origin", str(origin), "--by-month", str(by_month), "--out", str(png)]) == 0
    assert png.exists() and png.stat().st_size > 1000
    with pytest.raises(SystemExit):                                              # never overwritten
        F.main(["--origin", str(origin), "--by-month", str(by_month), "--out", str(png)])
    o2 = art_dir / "origin2.json"
    o2.write_text(origin.read_text().replace('"pooled_three_seasons"', '"pooled_three_seasons_"', 1))   # a different origin file
    with pytest.raises(SystemExit, match="not made from this origin"):
        F.main(["--origin", str(o2), "--by-month", str(by_month), "--out", str(tmp_path / "p2.png")])
