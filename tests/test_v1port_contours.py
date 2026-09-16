"""Tests for the version 1 contour merging port.

THE MUTATION LIST, WRITTEN BEFORE THE ASSERTIONS EXISTED, executable as
tests/mutations_v1port_contours.py.

    K1   the region search is 4-connected rather than 8-connected, so a diagonal
         neighbour breaks a trough in two
    K2   the region search returns the whole mask rather than the component containing
         the seed, merging unrelated troughs
    K3   SUPERSEDED BY K15 on 2026-08-30. It asserted that a seed on an empty cell
         returns an empty region, which is not what version 1 does.
    K4   the threshold ladder is not climbed, so an implausibly large region is kept
    K5   the ladder escalates on every region rather than only oversized ones
    K6   the convex hull absorbs candidates outside it
    K7   a degenerate region (two points or fewer) still absorbs
    K8   the near-duplicate pass keeps centres closer than the merge distance
    K9   the near-duplicate distance is compared in the wrong units
    K10  the minimum-extent pass keeps waves shorter than the threshold
    K11  the minimum-extent pass measures east to west instead of north to south
    K12  the refined centre replaces the median rather than averaging with it
    K13  NaN curvature is treated as above threshold
    K14  the early return for a lone wave is removed, so it gains checks version 1 never
         applied to it
    K15  a seed BELOW threshold returns an empty region rather than the seed cell plus
         the above-threshold components adjacent to it (added 2026-08-30 with the fix;
         the list below was written before the assertions, as this one was)
    K16  the seed cell itself is left out, so only the adjacent components come back
    K17  only ONE adjacent component is collected rather than all of them
    K18  the adjacent components are found 4-connected, so a diagonal one is missed
    K19  WITHDRAWN ON EVIDENCE, not carried as active. It asked whether the
         above-threshold branch could differ from the below-threshold path applied to
         the same seed. It cannot: with 8-connectivity any above-threshold cell adjacent
         to an above-threshold seed is already in the seed's own component. Measured
         over 4,000 random masks and exhaustively over every 3-by-3 mask and seed, the
         two paths agreed everywhere. The reproducer is
         tests/reproducers/k19_paths_agree.py.
    K20  an out-of-bounds seed returns the seed cell instead of nothing
    K21  the neighbour scan is skipped for a seed on a grid edge, so a boundary seed
         comes back as the seed alone (found by an independent check of the first
         version of these tests, which seeded only the interior)

    THE ABSORPTION FLAG, added 2026-08-30 after version 1's own code was run under
    Octave and its first pass was found never to absorb anything. Its inpolygon call
    passes two arguments the function does not take, the call raises, and an empty catch
    swallows it. `absorb=False` is therefore version 1 and is the default; `absorb=True`
    is what the code was written to do. This list was written before the assertions.

    K22  absorption happens by default, which is what version 1's broken call prevents
    K23  the flag is accepted and ignored, so absorb=True changes nothing
    K24  with absorption off the candidates are still consumed, so waves vanish anyway
    K25  with absorption off the wave still takes an absorbed candidate's time rather
         than its own
    K26  the flag reaches the coarse merge and not the fine one
"""

import numpy as np
import pytest

from aew.v1port.contours import (
    MAX_LAT_EXTENT_DEG, MERGE_DISTANCE_DEG, MIN_EXTENT_DEG, THRESHOLD_LADDER,
    connected_region, merge_contours,
)


def mesh(lat0=0.0, lat1=20.0, lon0=0.0, lon1=20.0, step=1.0):
    lat = np.arange(lat0, lat1 + step / 2, step)
    lon = np.arange(lon0, lon1 + step / 2, step)
    longrid, latgrid = np.meshgrid(lon, lat)
    return latgrid, longrid


def candidate(lat, lon, time=0):
    return {"lat_mean": float(lat), "lon_mean": float(lon), "time": time}


def two_blob_field(latgrid, longrid, extra_lat=32.0, extra_lon=32.0):
    """A tall, valid second wave far from the first.

    Passes two and three only run when pass one leaves more than one wave, so any test of
    them needs a companion. Placing it far away keeps the near-duplicate pass from
    collapsing the pair.
    """
    field = np.zeros(latgrid.shape)
    far = (np.abs(latgrid - extra_lat) <= 2) & (np.abs(longrid - extra_lon) <= 1)
    field[far] = 5e-6
    return field


# --- the connected region ------------------------------------------------------------

def test_region_is_eight_connected():
    """K1. The original tests all eight neighbours explicitly, so a diagonal link holds
    a region together. Four-connectivity would split this blob in two."""
    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True          # touches (1,1) only diagonally
    got = connected_region(mask, (1, 1))
    assert got[1, 1] and got[2, 2]
    assert got.sum() == 2


def test_region_returns_only_the_component_containing_the_seed():
    """K2. Two separate blobs must not be merged by the region search."""
    mask = np.zeros((7, 7), dtype=bool)
    mask[1, 1] = mask[1, 2] = True
    mask[5, 5] = True                       # a second, unconnected blob
    got = connected_region(mask, (1, 1))
    assert got[1, 1] and got[1, 2]
    assert not got[5, 5]
    assert got.sum() == 2


def test_seed_below_threshold_still_returns_the_seed_cell():
    """K15, and the port read this wrong until 2026-08-30.

    isolate_region_f.m marks its seed visited with no test that the seed qualifies:

        Zn(pos(1),pos(2)) = 2;
        output = pos;

    so a seed below threshold comes back as a region of exactly that cell when nothing
    above threshold touches it. The port returned nothing, which is not what version 1
    does and matters because `_select_region` seeds the stricter levels of the threshold
    ladder with a cell that may not survive them.
    """
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True                      # far from the seed, not adjacent to it
    got = connected_region(mask, (2, 2))
    assert got.sum() == 1, "version 1 marks the seed whether or not it qualifies"
    assert got[2, 2]


def test_seed_below_threshold_also_takes_every_adjacent_component():
    """K16, K17, K18. The seed is marked, THEN the neighbours are flood-filled.

    After marking the seed the original tests all eight neighbours against the ORIGINAL
    field and recurses into each one that is above threshold, so the region is the seed
    plus every above-threshold component touching it, not the seed alone. Two separate
    components touch the seed here, one orthogonally and one only diagonally, and the
    expected count is written out by hand: 1 seed + 3 in the left bar + 2 in the
    diagonal pair.
    """
    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 1] = True                    # a bar orthogonally adjacent to (3, 2)
    mask[2, 3] = True                      # diagonally adjacent to (3, 2)
    mask[1, 3] = True                      # and its own second cell
    got = connected_region(mask, (3, 2))   # (3, 2) is below threshold
    assert got[3, 2], "the seed itself must be in the region"
    assert got.sum() == 6, f"expected 1 + 3 + 2, got {int(got.sum())}"
    assert got[2, 1] and got[3, 1] and got[4, 1]
    assert got[2, 3] and got[1, 3]


def test_a_component_not_touching_the_seed_is_left_out():
    """K17 from the other side: only components ADJACENT to the seed are collected."""
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, 1] = True                      # touches the seed at (3, 2)
    mask[6, 6] = True                      # does not
    got = connected_region(mask, (3, 2))
    assert got.sum() == 2 and got[3, 2] and got[3, 1]
    assert not got[6, 6]


def test_a_below_threshold_seed_on_the_grid_EDGE_still_takes_its_neighbours():
    """K21, and the gap an independent check found in the first version of these tests.

    Every fixture above seeds the interior, so an implementation that marked the seed
    and then skipped the neighbour scan for any seed on an edge passed all of them. The
    original's eight neighbour tests are individually guarded against running off the
    grid, which is exactly a boundary behaviour, so the boundary needs its own case.
    Top edge, left edge and a corner, each with a component adjacent to the seed.
    """
    for seed, neighbour in (((0, 3), (1, 3)),       # top edge, neighbour below
                            ((3, 0), (3, 1)),       # left edge, neighbour right
                            ((0, 0), (1, 1)),       # corner, neighbour diagonally in
                            ((6, 6), (5, 5))):      # far corner, diagonally in
        mask = np.zeros((7, 7), dtype=bool)
        mask[neighbour] = True
        got = connected_region(mask, seed)
        assert got[seed], f"the seed at {seed} must be marked"
        assert got[neighbour], (f"the component at {neighbour} touches the seed at "
                                f"{seed} and must be collected")
        assert got.sum() == 2, f"expected the seed and its neighbour, got {int(got.sum())}"


def test_seed_outside_the_grid_gives_an_empty_region_still():
    """K20. The original returns BEFORE marking when the seed is out of bounds:

        if pos(1) < 1 | ... ; output = pos; return; end

    so nothing is marked and the region is empty. That is the one case where empty is
    right, and the K15 fix must not reach it.
    """
    mask = np.ones((3, 3), dtype=bool)
    assert connected_region(mask, (-1, 0)).sum() == 0
    assert connected_region(mask, (0, 9)).sum() == 0


def test_a_large_region_is_returned_whole():
    """The deliberate divergence. The original's recursive fill is wrapped in a catch
    that silently returns a partial region when MATLAB's recursion limit is reached, at a
    few hundred cells. This port returns the whole thing, which is why regions may
    legitimately differ from version 1 for large features."""
    mask = np.ones((40, 40), dtype=bool)      # 1600 cells, far past a 500-deep recursion
    assert connected_region(mask, (20, 20)).sum() == 1600


# --- the threshold ladder --------------------------------------------------------------

def test_an_oversized_region_escalates_to_a_stricter_threshold():
    """K4. A broad weak feature with a strong core: at the base threshold the region
    spans the whole domain, so the ladder must climb until it is smaller."""
    latgrid, longrid = mesh(lat0=0, lat1=40, lon0=0, lon1=40, step=1.0)
    curvature = np.full(latgrid.shape, 1.1e-6)          # everywhere just above base
    core = (np.abs(latgrid - 20) <= 2) & (np.abs(longrid - 20) <= 2)
    curvature[core] = 9e-6                              # a compact strong core
    out = merge_contours([candidate(20, 20)], latgrid, longrid, curvature, 1e-6)
    assert len(out) == 1
    span_lat = out[0]["lat_wave"].max() - out[0]["lat_wave"].min()
    assert span_lat < MAX_LAT_EXTENT_DEG, "the ladder did not climb"


def test_a_small_region_is_not_escalated():
    """K5. Escalating unconditionally would shrink a perfectly good region."""
    latgrid, longrid = mesh()
    curvature = np.zeros(latgrid.shape)
    blob = (np.abs(latgrid - 10) <= 2) & (np.abs(longrid - 10) <= 2)
    curvature[blob] = 2e-6                    # between the base and 1.5x the base
    out = merge_contours([candidate(10, 10)], latgrid, longrid, curvature, 1e-6)
    assert len(out) == 1
    assert out[0]["region"].sum() == blob.sum()


def test_the_ladder_is_the_one_the_source_names():
    assert THRESHOLD_LADDER == (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)


# --- merging by convex hull ------------------------------------------------------------

def test_a_candidate_inside_the_region_hull_is_absorbed():
    """K6, and it now needs `absorb=True` because absorption is no longer the default.

    Version 1 never reaches this code, so what it binds is the REPAIRED behaviour rather
    than version 1's. Left in place because the flag has to work when it is set, and the
    catalog caught these three tests going vacuous the moment the default changed.
    """
    latgrid, longrid = mesh()
    curvature = np.zeros(latgrid.shape)
    trough = (np.abs(latgrid - 10) <= 1) & (longrid >= 4) & (longrid <= 16)
    curvature[trough] = 5e-6
    out = merge_contours([candidate(10, 6), candidate(10, 14)],
                         latgrid, longrid, curvature, 1e-6, absorb=True)
    assert len(out) == 1


def test_a_candidate_outside_the_hull_survives_separately():
    """K6 the other way. Two genuinely separate troughs stay two waves."""
    latgrid, longrid = mesh()
    curvature = np.zeros(latgrid.shape)
    a = (np.abs(latgrid - 4) <= 1) & (np.abs(longrid - 4) <= 1)
    b = (np.abs(latgrid - 16) <= 1) & (np.abs(longrid - 16) <= 1)
    curvature[a | b] = 5e-6
    out = merge_contours([candidate(4, 4), candidate(16, 16)],
                         latgrid, longrid, curvature, 1e-6, absorb=True)
    assert len(out) == 2


def _absorption_fixture():
    """A region whose convex hull covers a SEPARATE region far from its own centre.

    The fixture has to be built this way and the reason is the finding itself. Two
    candidates on ONE connected trough grow the same region, so they produce identical
    centres that the five-degree pass collapses whether or not the first pass absorbed
    them, and the two settings agree. That is why the port matched version 1 on 171 of
    173 waves over six real timesteps despite performing a pass version 1 does not.

    So absorption is only observable when the absorbed candidate belongs to a DIFFERENT
    region that happens to fall inside the absorber's hull, and whose centre is further
    than the merge distance from the absorber's. An L of cells has a hull covering the
    triangle between its arms while its own centre stays in the corner.
    """
    latgrid, longrid = mesh(lat0=0, lat1=30, lon0=0, lon1=30, step=1.0)
    curvature = np.zeros(latgrid.shape)
    arm_a = (longrid == 5) & (latgrid >= 2) & (latgrid <= 28)
    arm_b = (latgrid == 2) & (longrid >= 6) & (longrid <= 25)
    separate = (longrid == 15) & (latgrid >= 4) & (latgrid <= 6)
    curvature[arm_a | arm_b | separate] = 5e-6
    return latgrid, longrid, curvature


def test_by_default_the_first_pass_absorbs_nothing():
    """K22, and the whole point of the flag.

    Version 1's first pass is written to absorb every candidate inside a region's convex
    hull and never does: the test calls `inpolygon` with two arguments it does not take,
    the call raises, and an empty catch swallows it. Established by running version 1's
    own merge_contours_f under Octave, where convhull succeeded 417 times over six real
    timesteps while the absorption loop completed 6 and the catch fired 411.
    """
    latgrid, longrid, curvature = _absorption_fixture()
    pair = [candidate(5, 5), candidate(5, 15)]
    faithful = merge_contours(pair, latgrid, longrid, curvature, 1e-6)
    repaired = merge_contours(pair, latgrid, longrid, curvature, 1e-6, absorb=True)
    assert len(faithful) == 2, ("version 1 absorbs nothing in the first pass, so the "
                                f"separate region survives it, got {len(faithful)}")
    assert len(repaired) == 1, ("with absorption enabled the separate region's centre "
                                "falls inside the L's hull and is taken into it")


def test_the_flag_is_not_merely_accepted():
    """K23. A flag that changed nothing could satisfy the test above by coincidence, so
    this pins that the settings differ where absorption is observable and agree where
    there is nothing to absorb."""
    latgrid, longrid, curvature = _absorption_fixture()
    pair = [candidate(5, 5), candidate(5, 15)]
    assert (len(merge_contours(pair, latgrid, longrid, curvature, 1e-6))
            != len(merge_contours(pair, latgrid, longrid, curvature, 1e-6, absorb=True)))
    lone = [candidate(5, 5)]
    assert (len(merge_contours(lone, latgrid, longrid, curvature, 1e-6))
            == len(merge_contours(lone, latgrid, longrid, curvature, 1e-6, absorb=True)))


def test_with_absorption_off_every_candidate_is_still_consumed_once():
    """K24. Absorbing nothing must not mean consuming nothing: the loop still has to
    retire the candidate it just turned into a wave, or it never terminates, and it must
    retire no other, or waves vanish for a different reason."""
    latgrid, longrid = mesh(lat0=0, lat1=30, lon0=0, lon1=30, step=1.0)
    curvature = np.zeros(latgrid.shape)
    for centre in (6, 15, 24):
        curvature[(np.abs(latgrid - 15) <= 2) & (longrid == centre)] = 5e-6
    three = [candidate(15, 6), candidate(15, 15), candidate(15, 24)]
    out = merge_contours(three, latgrid, longrid, curvature, 1e-6)
    assert len(out) == 3, f"three separate troughs, three waves, got {len(out)}"
    lons = sorted(round(w["lon_mean"], 3) for w in out)
    assert len(set(lons)) == 3, f"each wave must sit on its own trough, got {lons}"


def test_with_absorption_off_a_wave_keeps_its_own_time():
    """K25. Version 1 takes the time of the FIRST ABSORBED candidate when it absorbs,
    which the port reproduces under absorb=True. With nothing absorbed there is no such
    candidate, so each wave must carry the time of the candidate that made it."""
    latgrid, longrid, curvature = _absorption_fixture()
    first = dict(candidate(5, 5)); first["time"] = 100.0
    second = dict(candidate(5, 15)); second["time"] = 200.0
    out = merge_contours([first, second], latgrid, longrid, curvature, 1e-6)
    assert len(out) == 2
    assert {w["time"] for w in out} == {100.0, 200.0}, "each wave keeps its own time"
    absorbed = merge_contours([first, second], latgrid, longrid, curvature, 1e-6,
                              absorb=True)
    assert len(absorbed) == 1
    assert absorbed[0]["time"] == 200.0, ("FAITHFUL: when it absorbs, version 1 takes "
                                          "the absorbed candidate's time, not its own")


def test_near_duplicate_centres_collapse_to_one_wave():
    """K8, and it was NOT bound before 2026-08-30.

    Pass two drops any merged wave whose centre lies within five degrees of one already
    kept. The mutation that removes that test survived the suite, so this pins it: two
    separate troughs four degrees apart must come out as ONE wave. They survive pass one
    because each bar is a single column and therefore collinear, so no convex hull can
    be formed and neither can absorb the other, which is the degenerate-region behaviour
    pinned separately below.
    """
    latgrid, longrid = mesh(lat0=0, lat1=30, lon0=0, lon1=30, step=1.0)
    curvature = np.zeros(latgrid.shape)
    left = (np.abs(latgrid - 10) <= 2) & (longrid == 8)
    right = (np.abs(latgrid - 10) <= 2) & (longrid == 12)
    curvature[left | right] = 5e-6
    out = merge_contours([candidate(10, 8), candidate(10, 12)],
                         latgrid, longrid, curvature, 1e-6)
    assert len(out) == 1, ("centres four degrees apart are inside the five degree merge "
                           f"distance, got {[(w['lat_mean'], w['lon_mean']) for w in out]}")


def test_centres_beyond_the_merge_distance_stay_separate():
    """K8 the other way, so the test above cannot be satisfied by merging everything."""
    latgrid, longrid = mesh(lat0=0, lat1=30, lon0=0, lon1=30, step=1.0)
    curvature = np.zeros(latgrid.shape)
    left = (np.abs(latgrid - 10) <= 2) & (longrid == 5)
    right = (np.abs(latgrid - 10) <= 2) & (longrid == 20)
    curvature[left | right] = 5e-6
    out = merge_contours([candidate(10, 5), candidate(10, 20)],
                         latgrid, longrid, curvature, 1e-6)
    assert len(out) == 2, "fifteen degrees apart is well beyond the merge distance"


def test_the_merge_distance_is_compared_in_degrees_not_nautical_miles():
    """K9, also unbound before 2026-08-30.

    `great_circle_distance(..., "nm") / 60` converts to degrees, and dropping the
    division leaves a number about sixty times too large, so any separation the merge
    would otherwise collapse is left far above the five degree threshold. The expected
    separation is written out by hand rather than read back from the merge: two centres
    four degrees apart in longitude AT TEN DEGREES NORTH are 3.94 degrees apart, and only
    the degree figure is below the threshold.
    """
    from aew.v1port.geometry import great_circle_distance
    nautical = float(great_circle_distance(10.0, 8.0, 10.0, 12.0, "nm"))
    assert nautical / 60.0 == pytest.approx(3.94, abs=0.1), "four degrees of longitude at 10N"
    assert nautical / 60.0 < MERGE_DISTANCE_DEG, "in degrees it is inside the threshold"
    assert nautical > MERGE_DISTANCE_DEG, ("in nautical miles the same separation is far "
                                           "above the threshold, which is what the "
                                           "mutation exploits")


def test_a_degenerate_region_absorbs_nothing():
    """K7. The original wraps convhull in a try and leaves `check` unset on failure, so a
    region of two points or fewer cannot swallow anything. Both candidates therefore
    survive pass one as separate waves."""
    latgrid, longrid = mesh(lat0=0, lat1=40, lon0=0, lon1=40)
    curvature = np.zeros(latgrid.shape)
    curvature[10, 10] = 5e-6                  # a single cell: no hull exists
    tall = (np.abs(latgrid - 30) <= 2) & (np.abs(longrid - 30) <= 1)
    curvature[tall] = 5e-6
    out = merge_contours([candidate(10, 10), candidate(30, 30)],
                         latgrid, longrid, curvature, 1e-6, absorb=True)
    # the single cell is dropped by the minimum-extent pass, the tall one survives, and
    # the point is that the degenerate region absorbed nothing on the way
    assert len(out) == 1
    assert out[0]["lat_mean"] == pytest.approx(30.0, abs=1.0)


# --- the near-duplicate pass -----------------------------------------------------------

def test_centres_within_the_merge_distance_collapse():
    """K8 and K9. Two blobs whose centres sit inside the merge radius are one wave."""
    latgrid, longrid = mesh()
    curvature = np.zeros(latgrid.shape)
    a = (np.abs(latgrid - 10) <= 1) & (np.abs(longrid - 9) <= 1)
    b = (np.abs(latgrid - 10) <= 1) & (np.abs(longrid - 12) <= 1)
    curvature[a] = 5e-6
    curvature[b] = 5e-6
    out = merge_contours([candidate(10, 9), candidate(10, 12)],
                         latgrid, longrid, curvature, 1e-6)
    assert len(out) == 1
    assert MERGE_DISTANCE_DEG == 5.0


def test_centres_beyond_the_merge_distance_are_kept_apart():
    latgrid, longrid = mesh(lat0=0, lat1=40, lon0=0, lon1=40)
    curvature = np.zeros(latgrid.shape)
    a = (np.abs(latgrid - 8) <= 1) & (np.abs(longrid - 8) <= 1)
    b = (np.abs(latgrid - 30) <= 1) & (np.abs(longrid - 30) <= 1)
    curvature[a | b] = 5e-6
    out = merge_contours([candidate(8, 8), candidate(30, 30)],
                         latgrid, longrid, curvature, 1e-6)
    assert len(out) == 2


# --- the minimum-extent pass -----------------------------------------------------------

def test_a_wave_too_short_north_to_south_is_rejected():
    """K10 and K11. The extent test is meridional: a long thin east-west streak one row
    tall spans no latitude and is not a wave. A companion wave is present so that the
    pass actually runs, since a lone wave short-circuits out before it."""
    latgrid, longrid = mesh(lat0=0, lat1=40, lon0=0, lon1=40)
    curvature = two_blob_field(latgrid, longrid)
    curvature[10, 4:17] = 5e-6                # one row tall, thirteen columns wide
    out = merge_contours([candidate(10, 10), candidate(32, 32)],
                         latgrid, longrid, curvature, 1e-6)
    assert len(out) == 1, "the one-row streak should have been rejected"
    assert out[0]["lat_mean"] == pytest.approx(32.0, abs=1.0)
    assert MIN_EXTENT_DEG == 1.0


def test_a_lone_wave_skips_the_extent_check_entirely():
    """FAITHFUL. With only one wave the original returns before passes two and three, so
    the very streak rejected above is published when it is alone. Tidying this away is
    the obvious cleanup and would disagree with version 1."""
    latgrid, longrid = mesh(lat0=0, lat1=40, lon0=0, lon1=40)
    curvature = np.zeros(latgrid.shape)
    curvature[10, 4:17] = 5e-6
    out = merge_contours([candidate(10, 10)], latgrid, longrid, curvature, 1e-6)
    assert len(out) == 1, "a lone wave short-circuits past the minimum-extent pass"


def test_a_wave_tall_enough_is_kept():
    latgrid, longrid = mesh()
    curvature = np.zeros(latgrid.shape)
    curvature[8:13, 10] = 5e-6                # five rows tall
    out = merge_contours([candidate(10, 10)], latgrid, longrid, curvature, 1e-6)
    assert len(out) == 1


def test_the_refined_centre_is_the_average_of_the_peak_and_the_median():
    """K12. The original averages the peak-anomaly position with the region median rather
    than replacing one by the other. A companion wave is needed for the pass to run."""
    latgrid, longrid = mesh(lat0=0, lat1=40, lon0=0, lon1=40)
    curvature = two_blob_field(latgrid, longrid)
    region = (np.abs(latgrid - 10) <= 2) & (np.abs(longrid - 10) <= 2)
    curvature[region] = 2e-6
    curvature[12, 12] = 9e-6                  # the peak, offset from the region centre
    out = merge_contours([candidate(10, 10), candidate(32, 32)],
                         latgrid, longrid, curvature, 1e-6)
    first = [w for w in out if w["lat_mean"] < 20][0]
    # the region median is 10 and the peak sits at 12, so the answer is halfway
    assert first["lat_mean"] == pytest.approx(11.0)
    assert first["lon_mean"] == pytest.approx(11.0)


# --- input handling --------------------------------------------------------------------

def test_nan_curvature_is_not_above_threshold():
    """K13. A missing value must not seed or extend a region."""
    latgrid, longrid = mesh()
    curvature = np.full(latgrid.shape, np.nan)
    curvature[8:13, 10] = 5e-6
    out = merge_contours([candidate(10, 10)], latgrid, longrid, curvature, 1e-6)
    assert len(out) == 1
    assert out[0]["region"].sum() == 5


def test_no_candidates_gives_no_waves():
    latgrid, longrid = mesh()
    assert merge_contours([], latgrid, longrid, np.ones(latgrid.shape), 1e-6) == []


def test_nothing_above_threshold_gives_no_waves():
    latgrid, longrid = mesh()
    curvature = np.zeros(latgrid.shape)
    assert merge_contours([candidate(10, 10)], latgrid, longrid, curvature, 1e-6) == []


def test_mismatched_mesh_refuses():
    latgrid, longrid = mesh()
    with pytest.raises(ValueError, match="share a shape"):
        merge_contours([candidate(1, 1)], latgrid, longrid, np.zeros((3, 3)), 1e-6)
