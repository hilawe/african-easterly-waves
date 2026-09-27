"""The origin re-tabulation reproduces the season artifact's totals exactly, keeps every
longitude band present when empty, places a start on an edge in the band east of it,
keeps the latitude composition inside each band, and the map is drawn from verified
artifacts and never overwritten."""
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


def _season_artifact(tmp_path):
    import test_coast_crossing_measurement as T
    C = _load("coast_crossing_measurement")
    evidence, qdir, regions = T._world(tmp_path)
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    out = art_dir / "coast_crossing_1990_v3_2026-09-27.json"
    assert C.main(["--year", "1990", "--campaign-evidence", evidence, "--qtrack-dir", qdir, "--regions-dir", regions, "--out", str(out)]) == 0
    return art_dir, out


def test_bands_reproduce_totals_keep_empty_cells_and_place_edges_east(tmp_path):
    O = _load("coast_crossing_origin")
    art_dir, season_path = _season_artifact(tmp_path)
    out = tmp_path / "origin.json"
    assert O.main(["--artifacts", str(art_dir), "--years", "1990", "--out", str(out)]) == 0
    origin = json.load(open(out))
    season = json.load(open(season_path))
    assert origin["inputs"]["1990"]["sha256"] and origin["longitude_bands"]["edges"] == [-10.0, 10.0, 30.0]
    for name in ("this_record", "qtrack"):
        rows = origin["seasons"]["1990"][name]["by_longitude_band"]
        assert set(rows) == set(O.LON_BANDS) | {"total"}                          # every band present, empty or not
        tot = rows["total"]
        ref = season["sides"][name]["by_band"]["total"]
        assert all(tot[k] == ref[k] for k in ("cohort", "atlantic_side", "gulf_only", "none", "follow_up_incomplete", "single_observation_entrants"))
        assert sum(r["cohort"] for lab, r in rows.items() if lab != "total") == tot["cohort"]
        for r in rows.values():
            assert sum(v["cohort"] for v in r["latitude_composition"].values()) == r["cohort"]
            assert sum(v["atlantic_side"] for v in r["latitude_composition"].values()) == r["atlantic_side"]   # numerators too
            assert r["denominators"]["cohort"] == r["cohort"]
        g = origin["seasons"]["1990"][name]["grouping_sensitivity_by_longitude_band"]
        assert g["total"]["cohort"] == season["sides"][name]["grouping_sensitivity"]["units"]
    ours = origin["seasons"]["1990"]["this_record"]["by_longitude_band"]
    assert ours["east of 30 E"]["cohort"] == 0 and ours["east of 30 E"]["fraction_atlantic_side_over_cohort"] is None   # an empty cell, retained
    # the synthetic world's track 0 starts at exactly 10 E: on the edge, so in "10 E to 30 E", not "10 W to 10 E"
    first = {p["id"]: p for p in origin["seasons"]["1990"]["this_record"]["first_positions"]}
    assert first[0]["lon"] == 10.0 and O.lon_band(10.0) == "10 E to 30 E" and O.lon_band(9.999) == "10 W to 10 E"
    assert O.lon_band(-10.0) == "10 W to 10 E" and O.lon_band(-10.001) == "west of 10 W" and O.lon_band(30.0) == "east of 30 E"
    assert ours["10 E to 30 E"]["cohort"] >= 1
    with pytest.raises(SystemExit):                                              # never overwritten
        O.main(["--artifacts", str(art_dir), "--years", "1990", "--out", str(out)])


def test_a_season_artifact_that_disagrees_with_its_own_totals_is_refused(tmp_path):
    O = _load("coast_crossing_origin")
    art_dir, season_path = _season_artifact(tmp_path)
    season = json.load(open(season_path))
    season["sides"]["this_record"]["by_band"]["total"]["atlantic_side"] += 1          # the recorded total no longer matches the tracks
    tampered = art_dir / "coast_crossing_2002_v3_2026-09-27.json"
    season["year"] = 2002
    tampered.write_text(json.dumps(season))
    with pytest.raises(SystemExit, match="REFUSED: the tracks total"):
        O.main(["--artifacts", str(art_dir), "--years", "2002", "--out", str(tmp_path / "o.json")])
    season["sides"]["this_record"]["by_band"]["total"]["atlantic_side"] -= 1
    season["sides"]["qtrack"]["grouping_sensitivity"]["by_band"]["total"]["none"] += 1   # the recorded grouping total disagrees with the units
    tampered.write_text(json.dumps(season))
    with pytest.raises(SystemExit, match="REFUSED: the units total"):
        O.main(["--artifacts", str(art_dir), "--years", "2002", "--out", str(tmp_path / "o3.json")])
    season["sides"]["qtrack"]["grouping_sensitivity"]["by_band"]["total"]["none"] -= 1
    season["sides"]["qtrack"]["grouping_sensitivity"]["units"] += 1                       # the recorded unit count disagrees
    tampered.write_text(json.dumps(season))
    with pytest.raises(SystemExit, match="unit count"):
        O.main(["--artifacts", str(art_dir), "--years", "2002", "--out", str(tmp_path / "o4.json")])
    season["sides"]["qtrack"]["grouping_sensitivity"]["units"] -= 1
    season["year"] = 1990                                                             # a file named for 2002 holding 1990's artifact
    tampered.write_text(json.dumps(season))
    with pytest.raises(SystemExit, match="not the artifact for 2002"):
        O.main(["--artifacts", str(art_dir), "--years", "2002", "--out", str(tmp_path / "o2.json")])


def test_the_map_is_drawn_from_verified_artifacts_and_never_overwritten(tmp_path):
    O = _load("coast_crossing_origin")
    F = _load("fig_coast_crossing_origin")
    art_dir, season_path = _season_artifact(tmp_path)
    out = tmp_path / "origin.json"
    assert O.main(["--artifacts", str(art_dir), "--years", "1990", "--out", str(out)]) == 0
    png = tmp_path / "map.png"
    assert F.main(["--origin", str(out), "--artifacts", str(art_dir), "--out", str(png)]) == 0
    assert png.exists() and png.stat().st_size > 1000
    origin = json.load(open(out))
    n = len(origin["seasons"]["1990"]["this_record"]["first_positions"]) + len(origin["seasons"]["1990"]["qtrack"]["first_positions"])
    assert F.render(origin, F.verified_season_artifacts(origin, str(art_dir)), str(tmp_path / "again.png"))["1990"] == n
    with pytest.raises(SystemExit):
        F.main(["--origin", str(out), "--artifacts", str(art_dir), "--out", str(png)])
    edited = json.loads(json.dumps(origin))
    edited["seasons"]["1990"]["this_record"]["first_positions"][0]["status"] = "atlantic_side"   # an edited origin payload
    edited["seasons"]["1990"]["this_record"]["first_positions"][0]["lon"] += 5.0
    with pytest.raises(SystemExit, match="not the season artifact's"):
        F.render(edited, F.verified_season_artifacts(edited, str(art_dir)), str(tmp_path / "edited.png"))
    art = json.load(open(season_path))
    with open(os.path.join(art["inputs"]["region_polygons"]["dir"], "africa.mat"), "ab") as fh:
        fh.write(b"\0")                                                                        # a polygon file is no longer the artifact's
    with pytest.raises(SystemExit, match="region polygons"):
        F.polygons(art)
    with open(season_path, "ab") as fh:
        fh.write(b"\0")                                                              # the season artifact is no longer the one named
    with pytest.raises(SystemExit, match="not the season artifact"):
        F.verified_season_artifacts(origin, str(art_dir))
