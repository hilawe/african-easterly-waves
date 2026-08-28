"""Tests for the version 1 track association port, and for the duplication defect.

THE MUTATION LIST, WRITTEN BEFORE THE ASSERTIONS EXISTED, executable as
tests/mutations_v1port_association.py.

    A1   the exclusivity flag does nothing, so the repair silently fails to repair
    A2   exclusivity is applied always, so version 1 can no longer be reproduced
    A3   a candidate outside the search polygon is still matched
    A4   the speed ceiling is not enforced, so an implausible jump joins a track
    A5   the match picks an arbitrary candidate rather than the nearest to the prediction
    A6   an unmatched track keeps its prediction instead of standing down
    A7   the twelve-hour gate's second condition is dropped, so a track stranded by a
         wave-free timestep is recovered when version 1 would leave it stranded
    A8   claimed candidates still seed new tracks, which is the one exclusion version 1
         does enforce
    A9   the predicted displacement ignores the wind
    A10  the polygon is not displaced by the predicted motion
    A11  the minimum-area inflation never fires
    A12  the inflation is symmetric, losing the original's along-track stretch
    A13  the six-hour pass inflates every extended track's polygon, not only the last
    A14  a zero-area polygon is guarded instead of dividing by zero
    A15  the hull is left open, so the inflation centers on the centroid rather than on
         the original's double-counted first vertex
    A16  a wave-free timestep clears predictions instead of being skipped whole
    A17  a candidate with no trough points seeds a track instead of being skipped
    A18  the in-loop prune tests span rather than observation count and recency
    A19  the in-loop prune runs before the eighth timestep
    A20  the final speed filter uses end-to-end displacement rather than the median
         segment speed
    A21  the segment durations come from an assumed six-hour spacing rather than the
         track's own times
    A22  the five-point smoothing of the retained track is dropped
    A23  the final filter runs on a run shorter than the minimum lifetime
    A24  the trough mask is not carried onto the track, so a later statistics stage has
         nothing to composite over
    A25  the point-in-polygon test is inverted or always answers one way
    A26  a point on an edge is excluded, where MATLAB's inpolygon includes it
    A27  a concave polygon is treated as its convex hull
    A28  the trough mask is stored under the key the record writer reads for the basin,
         which makes every track unfilable
    A29  the six-hour inflation acts on the last DUE track rather than on the last track
         in the list, which is where MATLAB's residual loop index actually points

HOW THIS LIST GOT TO TWENTY-EIGHT. The first version had fourteen entries and every one of
them was caught, which said more about the list than about the code. An independent review
with repository access then read the MATLAB line by line against the port and returned
fourteen findings, of which twelve were confirmed at source. Most were behaviors the port
had quietly cleaned up: version 1 skips a wave-free timestep instead of standing tracks
down, inflates only the last polygon in its six-hour pass, divides by zero on a degenerate
polygon, closes its convex hull, prunes by observation count inside the loop rather than by
span at the end, and measures speed as a median over segments using real elapsed times. The
entries below from A13 onward exist because of that pass. A28 came later still, from
checking the two stages against each other rather than each against the MATLAB.
"""

import numpy as np
import pytest

from aew.v1port.association import (
    LIFETIME_STEPS, MAX_SPEED_MS, MIN_SPEED_MS, NO_PREDICTION, POLY_AREA_6HR,
    RECENT_DAYS, STEP_HOURS, _hull_polygon, _inflate, _moving_average_5, _polygon_area,
    associate_step, duplicate_observations, finalize_tracks, predict_displacement,
    prune_stale_tracks,
)
from aew.v1port import record
from aew.v1port.geometry import great_circle_distance, points_in_polygon

DAY_PER_STEP = STEP_HOURS / 24.0


def wave(lat, lon, time=0.0, half=1.0, half_lon=None, n=5, region=None):
    """A rectangular trough centered on (lat, lon).

    `half` is the latitude half-width and `half_lon` the longitude one, defaulting to a
    square. They are separate because the polygon tests need a wave narrow in longitude:
    a square wave inflates to a search region wider than the speed ceiling can reach, so
    its polygon never rejects anything and cannot be tested through.
    """
    if half_lon is None:
        half_lon = half
    lats = np.repeat(np.linspace(-half, half, n), n) + lat
    lons = np.tile(np.linspace(-half_lon, half_lon, n), n) + lon
    out = {"time": time, "lat_mean": float(lat), "lon_mean": float(lon),
           "lat_wave": lats, "lon_wave": lons}
    if region is not None:
        out["region"] = region
    return out


def still(_):
    return 0.0


def westward(_):
    return -8.0          # meters per second, the sign an easterly wave travels


def implied_speed(lat1, lon1, lat2, lon2, hours):
    km = float(great_circle_distance(lat1, lon1, lat2, lon2, "km"))
    return km * 1000.0 / (hours * 3600.0)


def seed_one_track(start_lat=10.0, start_lon=0.0, u=westward, **wave_kwargs):
    tracks, states = associate_step(
        [], [], [wave(start_lat, start_lon, time=0.0, **wave_kwargs)],
        step=0, u_median=u, v_median=still)
    return tracks, states


# --- displacement and polygons -------------------------------------------------------

def test_displacement_follows_the_wind():
    """A9. A westward wind must move the prediction west."""
    dlat, dlon = predict_displacement(-10.0, 0.0, 10.0, 6.0)
    assert dlon < 0 and dlat == pytest.approx(0.0)


def test_no_wind_gives_no_displacement():
    dlat, dlon = predict_displacement(0.0, 0.0, 10.0, 6.0)
    assert (dlat, dlon) == (0.0, 0.0)


def test_a_small_polygon_is_inflated_to_the_minimum_area():
    """A11."""
    lons = np.array([0.0, 1.0, 1.0, 0.0])
    lats = np.array([0.0, 0.0, 1.0, 1.0])
    assert _polygon_area(lons, lats) == pytest.approx(1.0)
    out_lon, out_lat = _inflate(lons, lats, POLY_AREA_6HR)
    assert _polygon_area(out_lon, out_lat) > 1.0


def test_a_large_polygon_is_left_alone():
    lons = np.array([0.0, 20.0, 20.0, 0.0])
    lats = np.array([0.0, 0.0, 20.0, 20.0])
    out_lon, out_lat = _inflate(lons, lats, POLY_AREA_6HR)
    assert np.array_equal(out_lon, lons) and np.array_equal(out_lat, lats)


def test_the_inflation_is_asymmetric_as_the_original_writes_it():
    """A12. FAITHFUL: longitude is scaled by sqrt(4/3 * s) and latitude by sqrt(2/3 * s),
    stretching the search region more along the direction of travel than across it."""
    lons = np.array([0.0, 1.0, 1.0, 0.0])
    lats = np.array([0.0, 0.0, 1.0, 1.0])
    out_lon, out_lat = _inflate(lons, lats, POLY_AREA_6HR)
    assert np.ptp(out_lon) > np.ptp(out_lat)


def test_a_zero_area_polygon_is_inflated_into_non_finite_coordinates():
    """A14. FAITHFUL, and it looks like a bug because in any other codebase it would be.
    Version 1's guard is `polyarea(...) <= threshold`, which a degenerate polygon passes,
    so `scale` is a division by zero and every vertex becomes non-finite. Guarding it here
    would keep alive a track version 1 stranded, which is the opposite of faithful."""
    lons = np.array([1.0, 2.0, 3.0, 4.0])
    lats = np.array([1.0, 2.0, 3.0, 4.0])          # collinear, zero area
    assert _polygon_area(lons, lats) == pytest.approx(0.0)
    out_lon, out_lat = _inflate(lons, lats, POLY_AREA_6HR)
    assert not np.all(np.isfinite(out_lon)) or not np.all(np.isfinite(out_lat))


def test_the_inflated_degenerate_polygon_contains_nothing():
    """The consequence of the case above: the track stops matching anything."""
    lons = lats = np.array([1.0, 2.0, 3.0])
    poly_lon, poly_lat = _inflate(lons, lats, POLY_AREA_6HR)
    assert not np.any(np.isfinite(poly_lon))
    probes = np.linspace(-40.0, 40.0, 25)
    assert not points_in_polygon(poly_lon, poly_lat, probes, probes).any()


def test_a_partly_non_finite_polygon_contains_nothing():
    """A DEFENSIVE PROPERTY OF THE SHARED ROUTINE, not a reproduction of version 1, and
    the distinction is the point.

    `_inflate` multiplies every vertex by an infinite scale, so the degenerate polygons it
    produces are non-finite throughout, and a ray cast with no guard happens to reject
    those anyway. A polygon with SOME finite vertices is a different matter: measured over
    60,000 random points against 3,000 such polygons, an unguarded cast admitted points
    about seven percent of the time. Nothing in this port produces that shape today, so the
    guard is a contract on a routine two stages now share rather than a version 1
    behavior, and this test says so rather than inventing a reachability argument for it.

    The fixture is built by hand for exactly that reason. An earlier draft claimed the
    shape came out of `_inflate`, which is false.
    """
    poly_lon = np.array([0.0, 10.0, np.inf, 0.0])
    poly_lat = np.array([0.0, 0.0, np.nan, 10.0])
    probes = np.linspace(-20.0, 20.0, 21)
    grid_x, grid_y = np.meshgrid(probes, probes)
    assert not points_in_polygon(poly_lon, poly_lat,
                                 grid_x.ravel(), grid_y.ravel()).any()


def test_the_hull_is_closed_as_matlab_returns_it():
    """A15. MATLAB's `convhull` repeats the first vertex, and version 1 feeds that list
    straight into an inflation that centers on `mean(...)`, so the repeat is not cosmetic:
    it pulls the center off the centroid and moves the whole inflated polygon."""
    lons = np.array([0.0, 1.0, 1.0, 0.0, 0.5])
    lats = np.array([0.0, 0.0, 1.0, 1.0, 0.5])
    hull_lon, hull_lat = _hull_polygon(lons, lats)
    assert hull_lon[0] == hull_lat[0] * 0 + hull_lon[0]        # readability only
    assert hull_lon[0] == hull_lon[-1] and hull_lat[0] == hull_lat[-1]
    assert np.mean(hull_lon) != pytest.approx(np.mean(np.unique(hull_lon)))


def test_a_wave_too_small_for_a_hull_falls_back():
    assert _hull_polygon(np.array([0.0, 1.0]), np.array([0.0, 1.0])) is None


# --- point in polygon -------------------------------------------------------------------

SQUARE_LON = np.array([0.0, 4.0, 4.0, 0.0])
SQUARE_LAT = np.array([0.0, 0.0, 4.0, 4.0])


def test_an_interior_point_is_inside():
    """A25."""
    assert points_in_polygon(SQUARE_LON, SQUARE_LAT, [2.0], [2.0]).tolist() == [True]


def test_an_exterior_point_is_outside():
    assert points_in_polygon(SQUARE_LON, SQUARE_LAT, [9.0], [2.0]).tolist() == [False]
    assert points_in_polygon(SQUARE_LON, SQUARE_LAT, [-9.0], [2.0]).tolist() == [False]
    assert points_in_polygon(SQUARE_LON, SQUARE_LAT, [2.0], [9.0]).tolist() == [False]


def test_a_point_on_an_edge_counts_as_inside():
    """A26. MATLAB's `inpolygon` reports an edge point as inside in the first output,
    which is the one version 1 reads, in both this stage and the merge stage. The polygons
    are hulls of grid points and the points tested sit on the same regular grid, so an
    exact edge hit is ordinary."""
    on_edges = points_in_polygon(SQUARE_LON, SQUARE_LAT,
                                 [0.0, 4.0, 2.0, 2.0], [2.0, 2.0, 0.0, 4.0])
    assert on_edges.tolist() == [True, True, True, True]


def test_a_vertex_counts_as_inside():
    corners = points_in_polygon(SQUARE_LON, SQUARE_LAT,
                                [0.0, 4.0, 4.0, 0.0], [0.0, 0.0, 4.0, 4.0])
    assert corners.tolist() == [True, True, True, True]


def test_a_point_level_with_a_vertex_but_outside_is_still_outside():
    """The classic ray-cast failure. Measured over 24,000 random grid-aligned point and
    polygon pairs, the two conventions for the straddle test disagree only on points that
    lie ON the boundary, which the edge test resolves first. So this pins the outcome, not
    the convention."""
    assert points_in_polygon(SQUARE_LON, SQUARE_LAT, [-1.0], [0.0]).tolist() == [False]
    assert points_in_polygon(SQUARE_LON, SQUARE_LAT, [-1.0], [4.0]).tolist() == [False]


def test_a_concave_polygon_excludes_its_notch():
    """A27. The search polygons are convex hulls, but the fallback diamond is built from
    the wave's own extremes and need not be convex."""
    lons = np.array([0.0, 6.0, 6.0, 3.0, 0.0])
    lats = np.array([0.0, 0.0, 6.0, 3.0, 6.0])
    assert points_in_polygon(lons, lats, [3.0], [5.0]).tolist() == [False]
    assert points_in_polygon(lons, lats, [3.0], [1.0]).tolist() == [True]


def test_a_degenerate_polygon_contains_nothing():
    assert not points_in_polygon(np.array([0.0, 1.0]), np.array([0.0, 1.0]),
                                 [0.5], [0.5]).any()


# --- matching ------------------------------------------------------------------------

def test_a_candidate_near_the_prediction_extends_the_track():
    tracks, states = seed_one_track()
    predicted_lon = states[0]["est_lon_6hr"]
    tracks, states = associate_step(tracks, states,
                                    [wave(10.0, predicted_lon, time=DAY_PER_STEP)],
                                    step=1, u_median=westward, v_median=still)
    assert len(tracks) == 1, "a matching candidate should extend, not seed"
    assert len(tracks[0]["time"]) == 2


def test_a_candidate_outside_the_polygon_is_rejected_on_the_polygon_alone():
    """A3. The candidate sits just past the polygon's eastern edge and its implied speed
    is comfortably under the ceiling, so only the polygon test can reject it. The test
    asserts that speed itself, because an earlier version placed the candidate far enough
    away that the ceiling did the rejecting and the polygon test went unexercised."""
    tracks, states = seed_one_track()
    east_edge = float(np.max(states[0]["poly_lon_6hr"]))
    candidate = wave(10.0, east_edge + 0.4, time=DAY_PER_STEP, half=0.2)
    assert float(np.min(candidate["lon_wave"])) > east_edge, "candidate must be outside"
    assert implied_speed(10.0, 0.0, candidate["lat_mean"], candidate["lon_mean"],
                         STEP_HOURS) < MAX_SPEED_MS, "speed must not be what rejects it"
    tracks, states = associate_step(tracks, states, [candidate], step=1,
                                    u_median=westward, v_median=still)
    assert len(tracks) == 2
    assert len(tracks[0]["time"]) == 1


def test_a_candidate_inside_the_polygon_is_rejected_on_speed_alone():
    """A4. The mirror of the test above: this candidate is well inside the search polygon
    and only the twenty-five-metre-per-second ceiling stands between it and the track."""
    tracks, states = seed_one_track()
    west_edge = float(np.min(states[0]["poly_lon_6hr"]))
    candidate = wave(10.0, west_edge + 0.5, time=DAY_PER_STEP, half=0.2)
    assert points_in_polygon(states[0]["poly_lon_6hr"], states[0]["poly_lat_6hr"],
                             candidate["lon_wave"], candidate["lat_wave"]).all(), \
        "the fixture must sit inside the polygon"
    assert implied_speed(10.0, 0.0, candidate["lat_mean"], candidate["lon_mean"],
                         STEP_HOURS) > MAX_SPEED_MS, "the fixture must exceed the ceiling"
    tracks, states = associate_step(tracks, states, [candidate], step=1,
                                    u_median=westward, v_median=still)
    assert len(tracks) == 2
    assert len(tracks[0]["time"]) == 1


def test_the_polygon_is_carried_to_the_predicted_position():
    """A10. A wave narrow in longitude keeps its inflated polygon narrow too, so the
    downwind shift is the difference between matching and not."""
    tracks, states = seed_one_track(half=8.0, half_lon=0.5)
    lons = states[0]["poly_lon_6hr"]
    span = float(np.max(lons) - np.min(lons))
    candidate_lon = float(np.min(lons)) + 0.3
    assert candidate_lon < 0.0 - span / 2.0, "must be outside the undisplaced polygon"
    candidate = wave(10.0, candidate_lon, time=DAY_PER_STEP, half=0.2)
    assert implied_speed(10.0, 0.0, 10.0, candidate_lon, STEP_HOURS) < MAX_SPEED_MS
    tracks, states = associate_step(tracks, states, [candidate], step=1,
                                    u_median=westward, v_median=still)
    assert len(tracks) == 1, "the displaced polygon should have caught it"
    assert len(tracks[0]["time"]) == 2


def test_a_track_with_no_match_stands_down():
    """A6. A timestep that HAS candidates but none this track can take clears its six-hour
    prediction. Contrast the wave-free case below, which does not."""
    tracks, states = seed_one_track()
    far = wave(-30.0, 130.0, time=DAY_PER_STEP)
    tracks, states = associate_step(tracks, states, [far], step=1,
                                    u_median=westward, v_median=still)
    assert states[0]["est_step_6hr"] == NO_PREDICTION


def test_the_nearest_candidate_to_the_prediction_wins():
    """A5. Two candidates both inside the polygon: the closer one must be chosen."""
    tracks, states = seed_one_track()
    target_lon = states[0]["est_lon_6hr"]
    near = wave(10.0, target_lon, time=DAY_PER_STEP)
    far = wave(10.0, target_lon + 1.2, time=DAY_PER_STEP, half=0.4)
    tracks, states = associate_step(tracks, states, [far, near], step=1,
                                    u_median=westward, v_median=still)
    assert tracks[0]["meanlon"][-1] == pytest.approx(near["lon_mean"])


def test_a_claimed_candidate_does_not_also_seed_a_track():
    """A8. This is the one exclusion version 1 does enforce, through `excl`."""
    tracks, states = seed_one_track()
    tracks, states = associate_step(
        tracks, states,
        [wave(10.0, states[0]["est_lon_6hr"], time=DAY_PER_STEP)],
        step=1, u_median=westward, v_median=still)
    assert len(tracks) == 1


def test_a_candidate_with_no_trough_points_is_skipped(monkeypatch):
    """A17. Lines 374-377 skip such a candidate rather than seeding from it. Seeding would
    take the maximum of an empty array."""
    empty = {"time": 0.0, "lat_mean": 10.0, "lon_mean": 0.0,
             "lat_wave": np.array([]), "lon_wave": np.array([])}
    tracks, states = associate_step([], [], [empty], step=0,
                                    u_median=westward, v_median=still)
    assert tracks == [] and states == []


def test_a_missing_wind_ends_the_track_silently():
    """The failure mode the module docstring warns about, pinned so it is visible.

    Version 1 aggregates its winds with `nanmedian`. A caller that wires in a plain median
    gets NaN wherever a wave touches masked ground, and a NaN displacement puts both the
    prediction and its search polygon nowhere. Nothing raises; the track simply stops
    matching, which is the hardest kind of defect to notice in a record.
    """
    def nan_wind(_):
        return float("nan")

    tracks, states = associate_step([], [], [wave(10.0, 0.0)], step=0,
                                    u_median=nan_wind, v_median=still)
    assert np.isnan(states[0]["est_lon_6hr"])
    anywhere = [wave(10.0, lon, time=DAY_PER_STEP, half=0.2)
                for lon in (-4.0, -2.0, 0.0, 2.0)]
    tracks, states = associate_step(tracks, states, anywhere, step=1,
                                    u_median=nan_wind, v_median=still)
    assert len(tracks[0]["time"]) == 1, "the track should have matched nothing"
    assert len(tracks) == 1 + len(anywhere), "every candidate seeded instead"


# --- the timestep with no candidates ----------------------------------------------------

def test_a_wave_free_timestep_is_skipped_whole():
    """A16. Every no-wave path in the detection half is a `continue`, so association never
    runs and nothing is cleared. A track keeps the prediction it already had."""
    tracks, states = seed_one_track()
    before = dict(states[0])
    tracks, states = associate_step(tracks, states, [], step=1,
                                    u_median=westward, v_median=still)
    assert states[0]["est_step_6hr"] == before["est_step_6hr"] == 1
    assert states[0]["est_step_12hr"] == before["est_step_12hr"] == 2
    assert len(tracks) == 1 and len(tracks[0]["time"]) == 1


def test_a_track_stranded_by_a_wave_free_timestep_is_not_recovered():
    """A7, and the reason the twelve-hour gate's second condition is load-bearing.

    A track seeded at step 0 expects a match at step 1. Step 1 has no waves at all, so
    nothing runs and its six-hour prediction stays at 1 rather than standing down. At step
    2 the six-hour pass skips it (its prediction is not due) and the twelve-hour pass
    excludes it too, because its six-hour prediction is not NO_PREDICTION. So a wave-free
    timestep strands the track permanently, and a candidate sitting exactly on its
    twelve-hour prediction cannot pick it up.
    """
    tracks, states = seed_one_track()
    tracks, states = associate_step(tracks, states, [], step=1,
                                    u_median=westward, v_median=still)
    assert states[0]["est_step_12hr"] == 2
    assert states[0]["est_step_6hr"] != NO_PREDICTION, "nothing should have stood down"
    on_prediction = wave(states[0]["est_lat_12hr"], states[0]["est_lon_12hr"],
                         time=2 * DAY_PER_STEP, half=0.2)
    tracks, states = associate_step(tracks, states, [on_prediction], step=2,
                                    u_median=westward, v_median=still)
    assert len(tracks) == 2, "the stranded track must not be recovered"
    assert len(tracks[0]["time"]) == 1


def test_the_twelve_hour_branch_recovers_a_track_that_stood_down():
    """The case the twelve-hour branch is actually for: the timestep HAD candidates, this
    track matched none of them, so its six-hour prediction stood down and the twelve-hour
    one is still live a step later."""
    tracks, states = seed_one_track()
    far = wave(-30.0, 130.0, time=DAY_PER_STEP)
    tracks, states = associate_step(tracks, states, [far], step=1,
                                    u_median=westward, v_median=still)
    assert states[0]["est_step_6hr"] == NO_PREDICTION
    assert states[0]["est_step_12hr"] == 2
    resumed = wave(states[0]["est_lat_12hr"], states[0]["est_lon_12hr"],
                   time=2 * DAY_PER_STEP, half=0.2)
    tracks, states = associate_step(tracks, states, [resumed], step=2,
                                    u_median=westward, v_median=still)
    assert len(tracks) == 2, "the far candidate seeded a track of its own at step 1"
    assert tracks[0]["step"] == [0, 2], "the original track should have resumed"


# --- the six-hour pass's single inflation -----------------------------------------------

def test_only_the_last_track_touched_by_the_six_hour_pass_is_inflated():
    """A13. FAITHFUL, and the strangest behavior in this file. Version 1's `for st` loop
    ends at line 252 and the inflation block sits at 254-265 OUTSIDE it, so it acts on
    `poly(st)` with the residual loop index. Every other track extended in that pass keeps
    an un-inflated polygon and searches a much smaller region at the next step."""
    seed_a = wave(10.0, 0.0, time=0.0, half=0.5)
    seed_b = wave(-10.0, 60.0, time=0.0, half=0.5)
    tracks, states = associate_step([], [], [seed_a, seed_b], step=0,
                                    u_median=westward, v_median=still)
    # Seeding inflates inside its own loop, so both start inflated.
    areas_seeded = [_polygon_area(s["poly_lon_6hr"], s["poly_lat_6hr"]) for s in states]
    assert all(a > 1.0 for a in areas_seeded)

    matches = [wave(10.0, states[0]["est_lon_6hr"], time=DAY_PER_STEP, half=0.5),
               wave(-10.0, states[1]["est_lon_6hr"], time=DAY_PER_STEP, half=0.5)]
    tracks, states = associate_step(tracks, states, matches, step=1,
                                    u_median=westward, v_median=still)
    assert all(len(t["time"]) == 2 for t in tracks), "both should have extended"
    first = _polygon_area(states[0]["poly_lon_6hr"], states[0]["poly_lat_6hr"])
    last = _polygon_area(states[1]["poly_lon_6hr"], states[1]["poly_lat_6hr"])
    assert first == pytest.approx(1.0, abs=0.2), "the first track must not be inflated"
    assert last > 10.0, "the last track the pass touched must be inflated"


def test_the_inflation_acts_on_the_last_track_not_the_last_due_one():
    """A29. The residual index is `ew_num`, not the last track the loop did work for.

    MATLAB's `for st = 1:ew_num` advances `st` on every iteration including the ones that
    hit `continue`, so after the loop `st` names the LAST TRACK IN THE LIST whether or not
    it was due at this step, and the inflation block that follows acts on that track's
    stored polygon. A first draft read it as the last DUE track, which is a different track
    whenever the newest one is not due, and no test could tell the two apart because every
    fixture made all tracks due at once.

    The fixture below breaks that. At step two, track A is due and matches; track B has
    stood down and is not due, and is last in the list. Version 1 inflates B.

    The inflation is visible in the polygon's SHAPE rather than its area, because the
    asymmetric scaling has an area fixed point: it lands at sqrt(8/9) of the minimum, which
    is still below the minimum, so re-inflating grows the longitude span and shrinks the
    latitude span while leaving the area where it was.
    """
    def aspect(state):
        return float(np.ptp(state["poly_lon_6hr"]) / np.ptp(state["poly_lat_6hr"]))

    tracks, states = associate_step([], [], [wave(10.0, 0.0, half=0.5),
                                             wave(-25.0, 80.0, half=0.5)],
                                    step=0, u_median=westward, v_median=still)
    tracks, states = associate_step(
        tracks, states,
        [wave(10.0, states[0]["est_lon_6hr"], time=DAY_PER_STEP, half=0.5)],
        step=1, u_median=westward, v_median=still)
    assert states[0]["est_step_6hr"] == 2, "A must be due at step 2"
    assert states[1]["est_step_6hr"] == NO_PREDICTION, "B must NOT be due at step 2"
    before = aspect(states[1])

    tracks, states = associate_step(
        tracks, states,
        [wave(10.0, states[0]["est_lon_6hr"], time=2 * DAY_PER_STEP, half=0.5)],
        step=2, u_median=westward, v_median=still)
    assert len(tracks[0]["time"]) == 3, "A should have extended again"
    assert aspect(states[1]) > before, "the last track in the list must be inflated"
    assert aspect(states[0]) == pytest.approx(1.0), \
        "the track that matched rebuilds its polygon un-inflated"


# --- the duplication defect and its repair -------------------------------------------

def two_tracks_converging():
    """Two tracks whose predictions land close enough to see one shared candidate."""
    waves0 = [wave(10.0, 0.0, time=0.0), wave(10.4, 0.4, time=0.0)]
    tracks, states = associate_step([], [], waves0, step=0,
                                    u_median=westward, v_median=still)
    assert len(tracks) == 2
    return tracks, states


def test_version_one_lets_two_tracks_claim_the_same_trough():
    """THE DEFECT. With exclusivity off, as version 1 runs, one candidate can extend two
    tracks in a single timestep, and from then on both carry identical observations.

    `excl` is written by the matching loops but read only when seeding new tracks, so
    nothing stops a second track taking a trough the first already took.
    """
    tracks, states = two_tracks_converging()
    shared = wave(10.2, states[0]["est_lon_6hr"], time=DAY_PER_STEP, half=1.5)
    tracks, states = associate_step(tracks, states, [shared], step=1,
                                    u_median=westward, v_median=still,
                                    exclusive=False)
    extended = [t for t in tracks if len(t["time"]) > 1]
    assert len(extended) == 2, "the defect did not reproduce"
    assert duplicate_observations(tracks) >= 1


def test_exclusivity_repairs_it():
    """A1. One rule, and the same code path: a claimed candidate leaves the pool."""
    tracks, states = two_tracks_converging()
    shared = wave(10.2, states[0]["est_lon_6hr"], time=DAY_PER_STEP, half=1.5)
    tracks, states = associate_step(tracks, states, [shared], step=1,
                                    u_median=westward, v_median=still,
                                    exclusive=True)
    extended = [t for t in tracks if len(t["time"]) > 1]
    assert len(extended) == 1, "exclusivity should let only one track claim it"
    assert duplicate_observations(tracks) == 0


def test_the_default_reproduces_version_one():
    """A2. Faithful reproduction is what validates the port, so the default must be the
    original's behavior rather than the repaired one."""
    tracks, states = two_tracks_converging()
    shared = wave(10.2, states[0]["est_lon_6hr"], time=DAY_PER_STEP, half=1.5)
    tracks, _ = associate_step(tracks, states, [shared], step=1,
                               u_median=westward, v_median=still)
    assert len([t for t in tracks if len(t["time"]) > 1]) == 2


def test_duplicate_counter_finds_nothing_in_disjoint_tracks():
    tracks, _ = associate_step([], [], [wave(10.0, 0.0), wave(-10.0, 100.0)],
                               step=0, u_median=westward, v_median=still)
    assert duplicate_observations(tracks) == 0


# --- what the track carries ---------------------------------------------------------------

def test_the_trough_mask_is_carried_onto_the_track():
    """A24. Version 1 stores `wave_points`, the linear indices of the trough cells, and
    generate_ew_stats_f.m composites its satellite fields over exactly those cells. The
    equivalent here is the merge stage's boolean region mask, and dropping it would leave a
    port of that stage with nothing to work from and no way to recover it."""
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    tracks, states = associate_step([], [], [wave(10.0, 0.0, region=mask)], step=0,
                                    u_median=westward, v_median=still)
    assert np.array_equal(tracks[0]["wave_points"][0], mask)
    match = wave(10.0, states[0]["est_lon_6hr"], time=DAY_PER_STEP, region=~mask)
    tracks, states = associate_step(tracks, states, [match], step=1,
                                    u_median=westward, v_median=still)
    assert len(tracks[0]["wave_points"]) == 2
    assert np.array_equal(tracks[0]["wave_points"][1], ~mask)


def test_the_trough_mask_does_not_shadow_the_basin_key():
    """A28, and this one is a real repair rather than a hypothetical.

    Two different things want to be called the region. The merge stage returns `region`
    for the boolean trough mask, and version 1 puts `region` and `region_name` on a TRACK
    for the geographic basin the wave came from, which is what the record writer files each
    track under. The first draft stored the mask on the track under `region`, so
    `record.write_year` would have read a list of masks where it expected a basin code.
    """
    mask = np.zeros((3, 3), dtype=bool)
    tracks, _ = associate_step([], [], [wave(10.0, 0.0, region=mask)], step=0,
                               u_median=westward, v_median=still)
    track = tracks[0]
    assert "wave_points" in track
    assert "region" not in track and "region_name" not in track, \
        "the basin keys must be left free for the record writer"
    # The binding that matters: the record writer must be able to file this track.
    track["region_name"] = "AFR"
    track["wavelength"] = [2500.0] * len(track["time"])
    import tempfile, os
    with tempfile.TemporaryDirectory() as directory:
        paths = record.write_year(directory, [track], 2005, 700, "ERA-Int")
    assert len(paths) == len(record.REGIONS)
    assert os.path.basename(paths[record.REGIONS.index("AFR")]).endswith("_AFR.nc")


# --- the in-loop prune ---------------------------------------------------------------------

def synthetic_track(n_steps, lon_step=-2.0, first_step=0, times=None):
    steps = list(range(first_step, first_step + n_steps))
    if times is None:
        times = [s * DAY_PER_STEP for s in steps]
    return {"time": list(times), "step": steps,
            "meanlat": [10.0] * n_steps,
            "meanlon": [lon_step * i for i in range(n_steps)],
            "maxlat": [11.0] * n_steps, "minlat": [9.0] * n_steps,
            "meanlon_maxlat": [0.0] * n_steps, "meanlon_minlat": [0.0] * n_steps,
            "lat_wave": [], "lon_wave": [], "n_points": [], "region": []}


def test_the_prune_does_not_run_before_the_eighth_timestep():
    """A19. `ct_time >= tlm_thr*4` with tlm_thr two, so the prune starts at timestep eight
    and everything before it survives however stale."""
    stale = synthetic_track(2)
    tracks, states = prune_stale_tracks([stale], [{}], step=6, now=100.0)
    assert tracks == [stale]


def test_a_recently_seen_short_track_survives_the_prune():
    """A18. The test is recency OR count, not span. This track has two observations and
    would fail any lifetime test, and version 1 keeps it because it was seen just now."""
    fresh = synthetic_track(2, first_step=8)
    now = fresh["time"][-1] + RECENT_DAYS / 2.0
    tracks, _ = prune_stale_tracks([fresh], [{}], step=20, now=now)
    assert tracks == [fresh]


def test_a_stale_short_track_is_pruned():
    stale = synthetic_track(2)
    tracks, states = prune_stale_tracks([stale], [{}], step=20, now=50.0)
    assert tracks == [] and states == []


def test_a_stale_but_long_track_survives():
    """The other half of the disjunction: more than eight observations keeps it alive even
    when it has not been seen for a long time."""
    long_track = synthetic_track(LIFETIME_STEPS + 1)
    tracks, _ = prune_stale_tracks([long_track], [{}], step=20, now=50.0)
    assert tracks == [long_track]


def test_a_track_with_exactly_the_lifetime_count_is_not_long_enough():
    """The original writes `length(...) > tlm_thr*4`, a strict inequality, so eight
    observations is not enough and nine is."""
    exactly = synthetic_track(LIFETIME_STEPS)
    tracks, _ = prune_stale_tracks([exactly], [{}], step=20, now=50.0)
    assert tracks == []


def test_the_prune_drops_states_alongside_their_tracks():
    """Version 1 subsets ew_tracks, poly, and all six prediction arrays with the same
    index list. Letting them fall out of step would attach one track's polygon to another.
    """
    keep = synthetic_track(LIFETIME_STEPS + 1)
    drop = synthetic_track(2)
    tracks, states = prune_stale_tracks([drop, keep, drop], [{"n": 0}, {"n": 1}, {"n": 2}],
                                        step=20, now=50.0)
    assert tracks == [keep] and states == [{"n": 1}]


# --- the final speed filter and smoothing ---------------------------------------------------

def test_a_stationary_track_is_dropped_for_being_too_slow():
    assert finalize_tracks([synthetic_track(12, lon_step=0.0)]) == []
    assert MIN_SPEED_MS == 2.0


def test_a_moving_track_survives():
    assert len(finalize_tracks([synthetic_track(12)])) == 1


def test_the_speed_test_is_the_median_segment_speed_not_end_to_end():
    """A20. A track that loops back on itself covers ground on every segment and ends up
    near where it started. Version 1 keeps it, because it takes the median of the segment
    speeds; an end-to-end test would drop it."""
    n = 12
    lons = [(-3.0 * i if i < 6 else -3.0 * (11 - i)) for i in range(n)]
    looping = synthetic_track(n)
    looping["meanlon"] = lons
    end_to_end = implied_speed(10.0, lons[0], 10.0, lons[-1], (n - 1) * STEP_HOURS)
    assert end_to_end < MIN_SPEED_MS, "the fixture must fail an end-to-end test"
    assert len(finalize_tracks([looping])) == 1


def test_the_segment_durations_come_from_the_tracks_own_times():
    """A21. A track resumed by the twelve-hour branch has a twelve-hour gap in it, and
    treating every gap as six hours would report twice its real speed. This track's steps
    are twelve hours apart and its per-segment speed falls below the minimum only when the
    real elapsed time is used."""
    n = 12
    times = [2.0 * i * DAY_PER_STEP for i in range(n)]
    slow = synthetic_track(n, lon_step=-0.55, times=times)
    six_hour_speed = implied_speed(10.0, 0.0, 10.0, -0.55, STEP_HOURS)
    twelve_hour_speed = implied_speed(10.0, 0.0, 10.0, -0.55, 2 * STEP_HOURS)
    assert twelve_hour_speed < MIN_SPEED_MS < six_hour_speed, "fixture must separate them"
    assert finalize_tracks([slow]) == []


def test_the_surviving_track_is_smoothed():
    """A22. A five-point moving average over mean latitude and longitude, applied to the
    kept tracks. It changes the coordinates that reach the record, so it is part of the
    output rather than a display step."""
    noisy = synthetic_track(12)
    noisy["meanlat"] = [10.0 + (2.0 if i % 2 else -2.0) for i in range(12)]
    kept = finalize_tracks([noisy])
    assert len(kept) == 1
    # Compare the interior, where the window is the full five points. The endpoints are
    # returned unchanged by the shrinking window, so the overall range does not move.
    assert np.ptp(kept[0]["meanlat"][2:-2]) < np.ptp(noisy["meanlat"][2:-2])
    assert kept[0]["meanlat"][0] == pytest.approx(noisy["meanlat"][0]), \
        "the first value is returned unchanged by a shrinking end window"


def test_the_smoothing_can_be_turned_off():
    kept = finalize_tracks([synthetic_track(12)], smooth=False)
    assert kept[0]["meanlon"] == synthetic_track(12)["meanlon"]


def test_the_moving_average_shrinks_its_window_at_the_ends():
    out = _moving_average_5([0.0, 0.0, 10.0, 0.0, 0.0])
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(10.0 / 3.0)
    assert out[2] == pytest.approx(2.0)
    assert out[4] == pytest.approx(0.0)


def test_a_short_run_is_not_filtered_at_all():
    """A23. `ct_time >= tlm_thr*4` gates the whole filter, so a run of fewer than eight
    timesteps returns its tracks untouched however slow they are."""
    stationary = synthetic_track(3, lon_step=0.0)
    assert finalize_tracks([stationary], total_steps=4) == [stationary]
    assert finalize_tracks([stationary], total_steps=LIFETIME_STEPS) == []


def test_the_speed_ceiling_is_the_one_the_source_names():
    assert MAX_SPEED_MS == 25.0
