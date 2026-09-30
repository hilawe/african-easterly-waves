"""The start-position map bins first positions on a fixed grid with stated edge rules,
keeps empty boxes apart from zero-fraction boxes, refuses any input that does not match its
recorded digest or its recorded totals, and never overwrites a figure."""
import hashlib
import importlib.util
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def _load():
    spec = importlib.util.spec_from_file_location("fig_start_position_map",
                                                  os.path.join(ROOT, "scripts", "fig_start_position_map.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _track(lon, lat, atlantic):
    return {"first": {"lon": lon, "lat": lat},
            "first_atlantic_side": ({"lon": -15.0, "lat": 10.0} if atlantic else None), "first_gulf": None}


def _lon_band(lon):
    return "west of 10 W" if lon < -10 else "10 W to 10 E" if lon < 10 else "10 E to 30 E" if lon < 30 else "east of 30 E"


def _lat_band(lat):
    k = min(int(lat // 5), 4)
    return ("0 to 5", "5 to 10", "10 to 15", "15 to 20", "20 to 25")[k]


def _world(tmp_path, sides=None):
    """One season with hand-placed tracks, its origin re-tabulation computed independently
    here, and an Africa outline whose digest the season records."""
    from scipy.io import savemat
    regions = tmp_path / "regions"
    regions.mkdir()
    savemat(str(regions / "africa.mat"), {"africa": np.array([[-17.0, 51.0, 51.0, -17.0], [0.0, 0.0, 30.0, 30.0]])})
    sides = sides or {
        "this_record": [_track(-12.0, 25.0, True), _track(12.0, 0.0, False), _track(12.5, 2.0, False),
                        _track(-10.0, 12.0, True)],
        "qtrack": [_track(20.0, 10.0, True), _track(20.0, 11.0, False)],
    }
    season = {"inputs": {"region_polygons": {"sha256": {"africa": _sha(regions / "africa.mat")}}}, "sides": {}}
    pooled = {}
    for side, tracks in sides.items():
        a = sum(t["first_atlantic_side"] is not None for t in tracks)
        season["sides"][side] = {"tracks": tracks, "by_band": {"total": {"cohort": len(tracks), "atlantic_side": a}}}
        bands = {b: {"atlantic_side": 0, "cohort": 0,
                     "latitude_composition": {l: {"atlantic_side": 0, "cohort": 0} for l in M.LAT_LABELS}}
                 for b, _, _ in M.LON_BANDS}
        for t in tracks:
            b = bands[_lon_band(t["first"]["lon"])]
            hit = int(t["first_atlantic_side"] is not None)
            b["atlantic_side"] += hit
            b["cohort"] += 1
            c = b["latitude_composition"][_lat_band(t["first"]["lat"])]
            c["atlantic_side"] += hit
            c["cohort"] += 1
        pooled[side] = {"by_longitude_band": bands}
    arts = tmp_path / "arts"
    arts.mkdir()
    spath = arts / "coast_crossing_1990.json"
    spath.write_text(json.dumps(season))
    origin = {"years": [1990], "inputs": {"1990": {"path": os.path.relpath(spath, tmp_path), "sha256": _sha(spath)}},
              "pooled_all_years": pooled}
    opath = tmp_path / "origin.json"
    opath.write_text(json.dumps(origin))
    return opath, spath, regions


def test_the_grid_edges_follow_the_stated_rules():
    assert M.box_index(-20.0, 0.0) == (0, 0)
    assert M.box_index(-10.0, 12.0) == (2, 2)            # a start on a meridian belongs to the box east of it
    assert M.box_index(-10.0001, 12.0) == (2, 1)
    assert M.box_index(12.0, 25.0) == (4, 6)             # the northern row is closed at 25 N
    assert M.box_index(12.0, 24.999) == (4, 6)
    for lon, lat in ((55.0, 10.0), (-20.1, 10.0), (10.0, 25.01), (10.0, -0.01)):
        with pytest.raises(SystemExit):
            M.box_index(lon, lat)


def test_empty_boxes_stay_apart_from_zero_fraction_boxes(tmp_path):
    opath, _, _ = _world(tmp_path)
    grids, totals, meta = M.tabulate(str(opath), str(tmp_path))
    g = grids["this_record"]
    assert (g["cohort"][0, 6], g["atlantic_side"][0, 6]) == (2, 0)   # two starts, none on the Atlantic side
    assert g["cohort"][0, 0] == 0                                    # no start at all
    assert (g["cohort"][4, 1], g["atlantic_side"][4, 1]) == (1, 1)
    assert totals == {"this_record": (2, 4), "qtrack": (1, 2)}
    assert meta["years"] == (1990, 1990)


def test_an_empty_box_is_not_drawn_and_a_zero_box_is_drawn_at_zero():
    assert M.box_style(0, 0) is None
    assert M.box_style(3, 0) == (0.0, True)
    assert M.box_style(M.SPARSE, 0) == (0.0, False)
    assert M.box_style(12, 6) == (50.0, False)
    assert M.box_style(M.SPARSE - 1, M.SPARSE - 1) == (100.0, True)


def test_a_season_that_does_not_match_its_recorded_digest_is_refused(tmp_path):
    opath, spath, _ = _world(tmp_path)
    spath.write_text(spath.read_text() + " ")
    with pytest.raises(SystemExit, match="digest"):
        M.tabulate(str(opath), str(tmp_path))


def test_a_season_whose_tracks_disagree_with_its_own_totals_is_refused(tmp_path):
    opath, spath, _ = _world(tmp_path)
    season = json.loads(spath.read_text())
    season["sides"]["qtrack"]["by_band"]["total"]["atlantic_side"] = 2
    spath.write_text(json.dumps(season))
    origin = json.loads(opath.read_text())
    origin["inputs"]["1990"]["sha256"] = _sha(spath)
    opath.write_text(json.dumps(origin))
    with pytest.raises(SystemExit, match="recomputes"):
        M.tabulate(str(opath), str(tmp_path))


def test_a_map_that_disagrees_with_the_origin_composition_is_refused(tmp_path):
    opath, _, _ = _world(tmp_path)
    origin = json.loads(opath.read_text())
    comp = origin["pooled_all_years"]["this_record"]["by_longitude_band"]["10 E to 30 E"]["latitude_composition"]
    comp["0 to 5"]["cohort"] -= 1
    comp["5 to 10"]["cohort"] += 1                      # same band total, wrong latitude row
    opath.write_text(json.dumps(origin))
    with pytest.raises(SystemExit, match="10 E to 30 E"):
        M.tabulate(str(opath), str(tmp_path))


def test_band_totals_that_err_in_compensating_directions_are_refused(tmp_path):
    opath, _, _ = _world(tmp_path)
    origin = json.loads(opath.read_text())
    bands = origin["pooled_all_years"]["this_record"]["by_longitude_band"]
    bands["west of 10 W"]["cohort"] += 1
    bands["10 W to 10 E"]["cohort"] -= 1                 # the grand total still agrees
    opath.write_text(json.dumps(origin))
    with pytest.raises(SystemExit, match="west of 10 W recomputes"):
        M.tabulate(str(opath), str(tmp_path))


def test_the_figure_is_written_once_in_vector_and_preview_form(tmp_path):
    opath, _, regions = _world(tmp_path)
    stem = tmp_path / "map"
    assert M.main(["--origin", str(opath), "--regions-dir", str(regions), "--out-stem", str(stem),
                   "--root", str(tmp_path)]) == 0
    assert (tmp_path / "map.pdf").stat().st_size > 1000 and (tmp_path / "map.png").stat().st_size > 1000
    with pytest.raises(SystemExit, match="never overwritten"):
        M.main(["--origin", str(opath), "--regions-dir", str(regions), "--out-stem", str(stem),
                "--root", str(tmp_path)])


def test_an_outline_that_does_not_match_the_recorded_digest_is_refused(tmp_path):
    opath, _, regions = _world(tmp_path)
    from scipy.io import savemat
    savemat(str(regions / "africa.mat"), {"africa": np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])})
    with pytest.raises(SystemExit, match="Africa outline"):
        M.main(["--origin", str(opath), "--regions-dir", str(regions), "--out-stem", str(tmp_path / "m"),
                "--root", str(tmp_path)])
