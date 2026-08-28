"""Tests for the version 1 contour merging port.

THE MUTATION LIST, WRITTEN BEFORE THE ASSERTIONS EXISTED, executable as
tests/mutations_v1port_contours.py.

    K1   the region search is 4-connected rather than 8-connected, so a diagonal
         neighbour breaks a trough in two
    K2   the region search returns the whole mask rather than the component containing
         the seed, merging unrelated troughs
    K3   a seed on an empty cell returns a non-empty region
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


def test_seed_on_an_empty_cell_gives_an_empty_region():
    """K3."""
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    assert connected_region(mask, (2, 2)).sum() == 0


def test_seed_outside_the_grid_gives_an_empty_region():
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
    """K6. Two candidates on one elongated trough must come out as one wave."""
    latgrid, longrid = mesh()
    curvature = np.zeros(latgrid.shape)
    trough = (np.abs(latgrid - 10) <= 1) & (longrid >= 4) & (longrid <= 16)
    curvature[trough] = 5e-6
    out = merge_contours([candidate(10, 6), candidate(10, 14)],
                         latgrid, longrid, curvature, 1e-6)
    assert len(out) == 1


def test_a_candidate_outside_the_hull_survives_separately():
    """K6 the other way. Two genuinely separate troughs stay two waves."""
    latgrid, longrid = mesh()
    curvature = np.zeros(latgrid.shape)
    a = (np.abs(latgrid - 4) <= 1) & (np.abs(longrid - 4) <= 1)
    b = (np.abs(latgrid - 16) <= 1) & (np.abs(longrid - 16) <= 1)
    curvature[a | b] = 5e-6
    out = merge_contours([candidate(4, 4), candidate(16, 16)],
                         latgrid, longrid, curvature, 1e-6)
    assert len(out) == 2


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
                         latgrid, longrid, curvature, 1e-6)
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
