"""Tests for assigning each wave the basin it came from.

THE MUTATION LIST, WRITTEN BEFORE THE ASSERTIONS EXISTED, executable as
tests/mutations_v1port_regions.py.

    G1   the basin is taken from somewhere other than the track's first observation
    G2   the quarter-degree snap is skipped, moving the answer near a boundary
    G3   the polygons are tested in a different order, so an overlapping basin wins
    G4   the catch-all is tested first, so every wave lands in it
    G5   a wave matching no polygon is filed somewhere anyway
    G6   the polygon test excludes points on the boundary, where inpolygon includes them
    G7   the stored polygon order is not preserved when the file is read
    G8   the count of unassigned waves is not reported, so a run missing waves looks clean

THE REAL CHECK IS THE LAST TEST IN THIS FILE. Everything above binds the rule as read from
generate_ew_stats_f.m; that one runs it against 12,163 tracks version 1 itself filed, and
they must all come back with the basin version 1 gave them. It is the only place in this
project where a stage is checked against version 1's own answers at scale, which is worth
more than any amount of reading the MATLAB.
"""

import os

import numpy as np
import pytest

from aew.v1port import regions as R

pytest.importorskip("scipy")

REGIONS_FILE = R.DEFAULT_REGIONS_FILE
has_regions = pytest.mark.skipif(
    not os.path.exists(REGIONS_FILE),
    reason="version 1's archived region polygons are not in this clone")
has_record = pytest.mark.skipif(
    not os.path.isdir("data/aewc") or not os.listdir("data/aewc"),
    reason="the published C00784 files are not in this clone")


def box(name, west, east, south, north):
    """A rectangular stand-in polygon, so the rule can be tested without the real file."""
    return (name,
            np.array([west, east, east, west, west], dtype=float),
            np.array([south, south, north, north, south], dtype=float))


SYNTHETIC = (box("FIRST", -10.0, 10.0, -10.0, 10.0),
             box("SECOND", 0.0, 20.0, 0.0, 20.0),        # overlaps FIRST
             box("CATCHALL", -180.0, 180.0, -90.0, 90.0))


def track(lons, lats):
    return {"meanlon": list(lons), "meanlat": list(lats),
            "time": [38351.0 + 0.25 * i for i in range(len(lons))]}


# --- the rule ---------------------------------------------------------------------------

def test_a_point_inside_one_polygon_gets_that_basin():
    assert R.region_of_point(-5.0, -5.0, SYNTHETIC) == "FIRST"
    assert R.region_of_point(15.0, 15.0, SYNTHETIC) == "SECOND"


def test_the_first_matching_polygon_wins_not_the_best_one():
    """G3 and G4. The real polygons overlap and the last is a box covering the whole
    domain, so order is part of the definition rather than an implementation detail."""
    assert R.region_of_point(5.0, 5.0, SYNTHETIC) == "FIRST", (
        "this point is in FIRST and SECOND; the earlier one must win")
    assert R.region_of_point(100.0, 60.0, SYNTHETIC) == "CATCHALL", (
        "and the catch-all must only take what nothing else did")


def test_a_point_in_nothing_gets_no_basin():
    """G5. Version 1 leaves the field unset and the writer then files the wave nowhere.
    Inventing a basin would put a wave into a record version 1 never put it in."""
    only_small = SYNTHETIC[:2]
    assert R.region_of_point(100.0, 60.0, only_small) is None


def test_a_point_on_a_boundary_counts_as_inside():
    """G6. The original calls MATLAB inpolygon, which includes the boundary, and the
    polygons were drawn on a quarter-degree grid that track positions snap onto, so exact
    boundary hits are ordinary rather than pathological."""
    assert R.region_of_point(-10.0, 0.0, SYNTHETIC) == "FIRST"
    assert R.region_of_point(10.0, 10.0, SYNTHETIC) == "FIRST"


# --- the snap ------------------------------------------------------------------------------

def test_positions_snap_to_the_quarter_degree_grid():
    """G2. The original's own comment says the polygons were drawn on a 0.25 land-sea grid,
    "hence the rounding"."""
    assert R.snap(10.1) == pytest.approx(10.0)
    assert R.snap(10.2) == pytest.approx(10.25)
    assert R.snap(-10.2) == pytest.approx(-10.25)
    assert R.SNAP_DEGREES == 0.25


def test_the_snap_changes_the_answer_near_a_boundary():
    """The snap is only worth reproducing if it can move a wave between basins, so here is
    a case where it does. A point just outside the FIRST box snaps onto its edge, and the
    edge counts as inside."""
    just_outside = -10.1
    assert R.snap(just_outside) == pytest.approx(-10.0), "snaps onto the boundary"
    assert R.region_of_point(just_outside, 0.0, SYNTHETIC) == "FIRST"
    further_out = -10.2
    assert R.snap(further_out) == pytest.approx(-10.25), "stays outside"
    assert R.region_of_point(further_out, 0.0, SYNTHETIC) == "CATCHALL"


# --- assignment over a track ------------------------------------------------------------------

def test_a_track_is_filed_by_where_it_started():
    """G1. A wave that forms over one basin and dies in another is filed under the first.
    The fixture starts in FIRST and ends well inside SECOND, so reading any other
    observation gives a different answer."""
    crossing = track([-8.0, 2.0, 12.0, 16.0, 18.0, 19.0, 19.5],
                     [-8.0, 2.0, 12.0, 16.0, 18.0, 19.0, 19.5])
    assert R.assign_region(crossing, SYNTHETIC) == "FIRST"
    assert crossing["region_name"] == "FIRST"
    # The fixture has to separate the first observation from BOTH the last and the mean,
    # or a rule reading either of those passes and the test binds nothing. An earlier
    # version spent most of its life near the origin, so its mean was still in FIRST.
    assert R.region_of_point(crossing["meanlon"][-1], crossing["meanlat"][-1],
                             SYNTHETIC) == "SECOND", "must end in a different basin"
    mean_lon = float(np.mean(crossing["meanlon"]))
    mean_lat = float(np.mean(crossing["meanlat"]))
    assert R.region_of_point(mean_lon, mean_lat, SYNTHETIC) == "SECOND", (
        "and its mean position must also land in a different basin")


def test_a_track_with_no_observations_is_refused():
    with pytest.raises(ValueError, match="no observations"):
        R.assign_region(track([], []), SYNTHETIC)


def test_unassigned_tracks_are_counted_rather_than_hidden():
    """G8. A run where many waves match no basin is a run whose record is missing waves,
    which should reach the caller rather than being swallowed."""
    inside = track([-5.0], [-5.0])
    outside = track([100.0], [60.0])
    tracks, unassigned = R.assign_regions([inside, outside, outside], SYNTHETIC[:2])
    assert unassigned == 2
    assert inside["region_name"] == "FIRST"
    assert "region_name" not in outside


# --- version 1's own polygons -------------------------------------------------------------------

@has_regions
def test_the_archived_polygons_load_in_their_stored_order():
    """G7. The order is part of the definition, and the catch-all has to stay last."""
    loaded = R.load_regions()
    assert [name for name, _, _ in loaded] == [
        "NEP", "SEP", "CAM", "SAM", "NAL", "SAL", "AFR", "OTH"]
    for name, lon, lat in loaded:
        assert lon.size == lat.size and lon.size >= 5, name
    catchall = loaded[-1]
    assert catchall[0] == "OTH"
    assert catchall[1].min() <= -140.0 and catchall[1].max() >= 40.0, (
        "the last polygon must span the domain, or it is not a catch-all")


def test_a_missing_polygon_file_is_refused_rather_than_worked_around(tmp_path):
    """Not gated on the real file being present, because the refusal is exactly what
    matters when it is absent."""
    with pytest.raises(FileNotFoundError, match="not reconstructible"):
        R.load_regions(str(tmp_path / "no_such_regions.mat"))


def write_regions_file(path, entries):
    """A regions.mat shaped like version 1's, for testing the reader without the real one."""
    from scipy.io import savemat

    dtype = [("name", "O"), ("lat", "O"), ("lon", "O")]
    array = np.zeros((1, len(entries)), dtype=dtype)
    for i, (name, lon, lat) in enumerate(entries):
        array[0, i]["name"] = np.array([name])
        array[0, i]["lon"] = np.asarray(lon, dtype=float).reshape(-1, 1)
        array[0, i]["lat"] = np.asarray(lat, dtype=float).reshape(-1, 1)
    savemat(str(path), {"regions": array})
    return str(path)


def test_the_reader_preserves_the_stored_order(tmp_path):
    """G7, bound without needing version 1's own file. The names are deliberately stored in
    an order that is NOT alphabetical, because sorting them is the plausible mistake and it
    would put the catch-all somewhere in the middle."""
    path = write_regions_file(tmp_path / "regions.mat",
                              [box("ZEBRA", -5.0, 5.0, -5.0, 5.0),
                               box("ALPHA", -6.0, 6.0, -6.0, 6.0),
                               box("MIDDLE", -7.0, 7.0, -7.0, 7.0)])
    loaded = R.load_regions(path)
    assert [name for name, _, _ in loaded] == ["ZEBRA", "ALPHA", "MIDDLE"]
    # and the order is load-bearing: all three contain the origin, so the first wins
    assert R.region_of_point(0.0, 0.0, loaded) == "ZEBRA"


@has_regions
@has_record
def test_every_published_africa_track_is_assigned_to_africa():
    """THE CHECK THIS MODULE EXISTS FOR, and the only one in the project that tests a stage
    against version 1's own answers at scale.

    The files under data/aewc are the Africa region of C00784, so version 1 assigned every
    one of these tracks to AFR by running the rule this module ports. All 12,163 of them
    must come back AFR. Any other answer is this port disagreeing with the original about a
    case the original already decided, and there are twelve thousand of them.
    """
    from aew.v1port import validate as V

    tracks = V.read_record()
    assert len(tracks) > 10000, "the record did not load"
    loaded = R.load_regions()
    wrong = []
    for t in tracks:
        name = R.region_of_point(t["meanlon"][0], t["meanlat"][0], loaded)
        if name != "AFR":
            wrong.append((name, float(t["meanlon"][0]), float(t["meanlat"][0])))
    assert wrong == [], (
        f"{len(wrong)} of {len(tracks)} published Africa tracks were filed elsewhere; "
        f"first few: {wrong[:5]}")
