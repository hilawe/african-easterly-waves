"""Tests for the version 1 geometry helpers.

THE MUTATION LIST, WRITTEN BEFORE THE ASSERTIONS EXISTED, executable as
tests/mutations_v1port_geometry.py.

    G1   advection sign flipped, so the flow appears to carry values the wrong way
    G2   advection masks two rings instead of three, returning numbers where the
         vorticity it differentiates is itself undefined
    G3   advection symmetrizes the right-hand mask on a global grid
    G4   advection swaps the zonal and meridional derivative terms
    G5   metres to degrees uses sine where the original uses cosine
    G6   metres to degrees applies the latitude factor to the latitude spacing
    G7   great-circle distance uses a wrong unit factor
    G8   great-circle distance drops the cos(lat1)cos(lat2) term, so it is right on the
         equator and wrong everywhere else
    G9   wavelength omits the factor of four that converts a quarter to a whole
    G10  wavelength does not round the heading, so no grid point matches exactly
    G11  wavelength returns a value when the centre is below the trough threshold
    G12  wavelength interpolates the crossing from the wrong pair of points
"""

import numpy as np
import pytest

from aew.v1port.geometry import (
    ADVECTION_EDGE_WIDTH, advection_of_vorticity, great_circle_distance,
    meters_to_degrees, wavelength_along_ray,
)
from aew.v1port.vorticity import EARTH_RADIUS_M, MATLAB_FILL, grid_spacing_m


def mesh(lat0=-10.0, lat1=10.0, lon0=-10.0, lon1=10.0, step=1.0):
    lat = np.arange(lat0, lat1 + step / 2, step)
    lon = np.arange(lon0, lon1 + step / 2, step)
    longrid, latgrid = np.meshgrid(lon, lat)
    return latgrid, longrid


# --- metres and degrees are inverses -------------------------------------------------

@pytest.mark.parametrize("lat", [0.0, 15.0, 30.0, 45.0])
def test_degrees_and_metres_round_trip(lat):
    """G5 and G6. Converting one way and back must return the spacing we started from."""
    dx, dy = grid_spacing_m(lat, 1.0, 1.0)
    dlat, dlon = meters_to_degrees(dx, dy, lat)
    assert dlat == pytest.approx(1.0, rel=1e-12)
    assert dlon == pytest.approx(1.0, rel=1e-12)


def test_longitude_degrees_grow_toward_the_pole():
    """G5. A fixed distance in metres spans more degrees of longitude at high latitude."""
    _, dlon_equator = meters_to_degrees(111_000.0, 111_000.0, 0.0)
    _, dlon_sixty = meters_to_degrees(111_000.0, 111_000.0, 60.0)
    assert dlon_sixty == pytest.approx(2.0 * dlon_equator, rel=1e-9)


def test_latitude_degrees_do_not_depend_on_latitude():
    """G6."""
    a, _ = meters_to_degrees(1.0, 111_000.0, 0.0)
    b, _ = meters_to_degrees(1.0, 111_000.0, 55.0)
    assert a == pytest.approx(b)


# --- great-circle distance -----------------------------------------------------------

def test_distance_to_itself_is_zero():
    assert great_circle_distance(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0, abs=1e-9)


def test_one_degree_of_latitude_is_sixty_nautical_miles():
    """G7. The original routes every unit through nautical miles, where one degree of
    great circle is sixty by definition."""
    assert great_circle_distance(0.0, 0.0, 1.0, 0.0, "nm") == pytest.approx(60.0, rel=1e-9)
    assert great_circle_distance(0.0, 0.0, 1.0, 0.0, "km") == pytest.approx(111.12, rel=1e-4)
    assert great_circle_distance(0.0, 0.0, 1.0, 0.0, "degrees") == pytest.approx(1.0)


def test_a_degree_of_longitude_shrinks_with_latitude():
    """G8. Dropping the cosine term would make these two equal."""
    at_equator = great_circle_distance(0.0, 0.0, 0.0, 1.0, "km")
    at_sixty = great_circle_distance(60.0, 0.0, 60.0, 1.0, "km")
    assert at_sixty == pytest.approx(at_equator * 0.5, rel=1e-3)


def test_distance_is_symmetric():
    a = great_circle_distance(12.0, -30.0, -5.0, 40.0, "km")
    b = great_circle_distance(-5.0, 40.0, 12.0, -30.0, "km")
    assert a == pytest.approx(b)


def test_unknown_unit_refuses():
    with pytest.raises(ValueError, match="unknown unit"):
        great_circle_distance(0, 0, 1, 1, "furlongs")


# --- advection of vorticity ----------------------------------------------------------

def test_no_wind_gives_no_advection():
    latgrid, longrid = mesh()
    rng = np.random.default_rng(0)
    vort = rng.normal(size=latgrid.shape) * 1e-5
    zero = np.zeros_like(vort)
    adv = advection_of_vorticity(latgrid, longrid, zero, zero, vort)
    assert np.allclose(adv[np.isfinite(adv)], 0.0)


def test_uniform_vorticity_gives_no_advection():
    latgrid, longrid = mesh()
    u = np.full(latgrid.shape, 5.0)
    v = np.full(latgrid.shape, -3.0)
    vort = np.full(latgrid.shape, 2e-5)
    adv = advection_of_vorticity(latgrid, longrid, u, v, vort)
    assert np.allclose(adv[np.isfinite(adv)], 0.0, atol=1e-18)


def test_westerly_wind_down_an_eastward_gradient_gives_positive_advection():
    """G1 and G4. With vorticity increasing eastward and the wind blowing eastward, the
    flow carries lower values in, so the local tendency is negative. The sign convention
    is the meteorological one and a flip is the easiest error to make here."""
    latgrid, longrid = mesh()
    vort = 1e-6 * longrid                      # increases toward the east
    u = np.full(latgrid.shape, 10.0)           # blowing toward the east
    v = np.zeros_like(u)
    adv = advection_of_vorticity(latgrid, longrid, u, v, vort)
    inner = np.isfinite(adv)
    assert np.all(adv[inner] < 0)


def test_southerly_wind_up_a_northward_gradient_is_the_meridional_counterpart():
    """G4. Same statement with the axes exchanged, so a swap of the two terms shows."""
    latgrid, longrid = mesh()
    vort = 1e-6 * latgrid
    u = np.zeros(latgrid.shape)
    v = np.full(latgrid.shape, 10.0)
    adv = advection_of_vorticity(latgrid, longrid, u, v, vort)
    inner = np.isfinite(adv)
    assert np.all(adv[inner] < 0)


def test_advection_masks_three_rings_not_two():
    """G2. It differentiates the vorticity, which is itself undefined in the outer two
    rings, so calculate_advvort_f masks uncalc+1."""
    latgrid, longrid = mesh()
    rng = np.random.default_rng(1)
    f = rng.normal(size=latgrid.shape)
    adv = advection_of_vorticity(latgrid, longrid, f, f, f)
    assert ADVECTION_EDGE_WIDTH == 3
    assert np.all(np.isnan(adv[:ADVECTION_EDGE_WIDTH, :]))
    assert np.all(np.isnan(adv[-ADVECTION_EDGE_WIDTH:, :]))
    assert np.all(np.isnan(adv[:, :ADVECTION_EDGE_WIDTH]))
    assert np.all(np.isnan(adv[:, -ADVECTION_EDGE_WIDTH:]))
    assert np.any(np.isfinite(adv))


def test_global_grid_wraps_the_left_edge_but_still_masks_the_right():
    """G3. The same asymmetry the vorticity stage carries, for the same reason: the
    right-hand term sits outside the globflag guard in the MATLAB."""
    latgrid, longrid = mesh(lon0=0.0, lon1=359.0, step=1.0)
    rng = np.random.default_rng(2)
    f = rng.normal(size=latgrid.shape)
    adv = advection_of_vorticity(latgrid, longrid, f, f, f, global_grid=True)
    interior = slice(ADVECTION_EDGE_WIDTH, -ADVECTION_EDGE_WIDTH)
    assert np.all(np.isfinite(adv[interior, :ADVECTION_EDGE_WIDTH]))
    assert np.all(np.isnan(adv[:, -ADVECTION_EDGE_WIDTH:]))


def test_advection_preserves_a_time_axis_and_squeezes_a_flat_field():
    latgrid, longrid = mesh()
    rng = np.random.default_rng(3)
    f3 = rng.normal(size=(3,) + latgrid.shape)
    adv3 = advection_of_vorticity(latgrid, longrid, f3, f3, f3)
    assert adv3.shape == f3.shape
    adv2 = advection_of_vorticity(latgrid, longrid, f3[0], f3[0], f3[0])
    assert adv2.shape == latgrid.shape
    assert np.allclose(adv3[0][np.isfinite(adv3[0])], adv2[np.isfinite(adv2)])


def test_advection_refuses_mismatched_shapes_and_tiny_grids():
    latgrid, longrid = mesh()
    z = np.zeros(latgrid.shape)
    with pytest.raises(ValueError, match="same shape"):
        advection_of_vorticity(latgrid, longrid, z, z, np.zeros((2,) + latgrid.shape))
    small = np.zeros((5, 5))
    with pytest.raises(ValueError, match="too small"):
        advection_of_vorticity(small, small, small, small, small)


# --- wavelength ----------------------------------------------------------------------

def ray_case(profile_values, resolution=1.0):
    """A due-west-moving wave on a one-row mesh, with a chosen profile along the ray.

    The wave sits at longitude 0 and moves west, so the ray runs along negative
    longitudes and the profile is read outward from the centre.
    """
    n = len(profile_values)
    lon = np.arange(0, -n, -1, dtype=float) * resolution
    lat = np.zeros_like(lon)
    longrid = lon[np.newaxis, :]
    latgrid = lat[np.newaxis, :]
    return latgrid, longrid, np.asarray(profile_values, dtype=float)


def test_wavelength_is_four_times_the_quarter_it_measures():
    """G9. The zero crossing is a quarter wavelength, so the answer is four times it."""
    latgrid, longrid, crv = ray_case([4e-6, 2e-6, -2e-6, -4e-6])
    got = wavelength_along_ray(0.0, 5.0, 0.0, 0.0, latgrid, longrid, crv,
                               threshold=1e-6, resolution=1.0, max_threshold=1e9)
    # the crossing is midway between the points at 1 and 2 degrees from the centre
    midpoint_km = great_circle_distance(0.0, 0.0, 0.0, -1.5, "km")
    assert got == pytest.approx(4.0 * midpoint_km, rel=1e-6)


def test_wavelength_refuses_when_the_first_ray_point_is_below_threshold():
    """G11. If the ray does not start inside a trough it is not measuring a wave.

    Note which point that is. The wave centre is NOT on its own ray, because the
    direction from a point to itself is zero rather than the wave's heading, so the
    check falls on the first grid point one step out along the ray. An earlier draft of
    this test put the low value at the centre and the port correctly ignored it.
    """
    latgrid, longrid, crv = ray_case([4e-6, 1e-9, -2e-6, -4e-6])
    got = wavelength_along_ray(0.0, 5.0, 0.0, 0.0, latgrid, longrid, crv,
                               threshold=1e-6, resolution=1.0, max_threshold=1e9)
    assert got == MATLAB_FILL


def test_wavelength_refuses_with_too_few_points_on_the_ray():
    latgrid, longrid, crv = ray_case([4e-6, -1e-6])
    got = wavelength_along_ray(0.0, 5.0, 0.0, 0.0, latgrid, longrid, crv,
                               threshold=1e-6, resolution=1.0, max_threshold=1e9)
    assert got == MATLAB_FILL


def test_wavelength_refuses_when_the_profile_never_crosses_zero():
    latgrid, longrid, crv = ray_case([4e-6, 3e-6, 2e-6, 1e-6])
    got = wavelength_along_ray(0.0, 5.0, 0.0, 0.0, latgrid, longrid, crv,
                               threshold=1e-6, resolution=1.0, max_threshold=1e9)
    assert got == MATLAB_FILL


def test_wavelength_grows_when_the_crossing_is_further_out():
    """A longer quarter must give a longer wavelength, monotonically."""
    _, _, _ = ray_case([0])
    near = wavelength_along_ray(0.0, 5.0, 0.0, 0.0, *ray_case([4e-6, -4e-6, -5e-6, -6e-6]),
                               threshold=1e-6, resolution=1.0, max_threshold=1e9)
    far = wavelength_along_ray(0.0, 5.0, 0.0, 0.0,
                               *ray_case([4e-6, 3e-6, 2e-6, -2e-6, -3e-6]),
                               threshold=1e-6, resolution=1.0, max_threshold=1e9)
    assert far > near


def test_heading_rounding_is_what_lets_points_match_at_all():
    """G10. The original snaps both headings to the nearest degree precisely so that an
    exact equality test can find the points lying along the ray. Without the rounding
    the match set is empty and no wavelength is ever produced."""
    latgrid, longrid, crv = ray_case([4e-6, 2e-6, -2e-6, -4e-6])
    got = wavelength_along_ray(0.0, 5.0, 0.0, 0.0, latgrid, longrid, crv,
                               threshold=1e-6, resolution=1.0, max_threshold=1e9)
    assert got != MATLAB_FILL


def test_wavelength_refuses_a_mesh_that_does_not_match_the_field():
    latgrid, longrid, crv = ray_case([4e-6, 2e-6, -2e-6, -4e-6])
    with pytest.raises(ValueError, match="mesh"):
        wavelength_along_ray(0.0, 5.0, 0.0, 0.0, latgrid, longrid, crv[:2],
                             threshold=1e-6, resolution=1.0, max_threshold=1e9)
