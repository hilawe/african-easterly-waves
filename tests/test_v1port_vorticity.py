"""Tests for the version 1 vorticity decomposition port.

THE MUTATION LIST, WRITTEN BEFORE THE ASSERTIONS EXISTED. Each is a laziest plausible
implementation of this module, not a reversal of a known fix, and each is executable as
tests/mutations_v1port_vorticity.py.

    V1   refdir returns atan(u/v) with no quadrant wrapping, so two quadrants are wrong
    V2   refdir returns atan2(v, u), the mathematical convention, not atan2(u, v)
    V3   the zonal spacing uses sin(latitude) where the original means cos(latitude)
    V4   the zonal spacing drops the latitude factor, so every row is spaced as at the
         equator
    V5   the edge mask is not applied, so the border returns numbers instead of fill
    V6   curvature is computed as shear minus relative, with the sign flipped
    V7   the centered difference divides by one spacing instead of two
    V8   the zonal neighbour does not wrap, so a global grid indexes out of range or
         reflects
    V9   the two derivative terms of relative vorticity are swapped
    V10  a zero wind projects to NaN instead of the zero the original defines
    V11  the right-hand column mask is symmetrized on a global grid, which is the
         tempting cleanup and would silently disagree with version 1

No MATLAB is available here (the license server cannot be reached), so these tests bind
the port against ANALYTIC fields whose vorticity is known in closed form, plus the
structural properties the MATLAB source states. If a license appears, the same cases
become golden-value comparisons against the original.
"""

import numpy as np
import pytest

from aew.v1port.vorticity import (
    EARTH_RADIUS_M, EDGE_WIDTH, MATLAB_FILL, component_vorticity, grid_spacing_m,
    project_onto_direction, reference_direction,
)


def mesh(lat0=-10.0, lat1=10.0, lon0=-10.0, lon1=10.0, step=1.0):
    lat = np.arange(lat0, lat1 + step / 2, step)
    lon = np.arange(lon0, lon1 + step / 2, step)
    longrid, latgrid = np.meshgrid(lon, lat)
    return latgrid, longrid


# --- reference direction: the meteorological convention, verified per quadrant --------

@pytest.mark.parametrize("u,v,expected", [
    (0.0, 1.0, 0.0),                  # due north
    (1.0, 0.0, np.pi / 2),            # due east
    (0.0, -1.0, np.pi),               # due south
    (-1.0, 0.0, 3 * np.pi / 2),       # due west
    (1.0, 1.0, np.pi / 4),            # northeast
    (1.0, -1.0, 3 * np.pi / 4),       # southeast
    (-1.0, -1.0, 5 * np.pi / 4),      # southwest
    (-1.0, 1.0, 7 * np.pi / 4),       # northwest
    (0.0, 0.0, 0.0),                  # calm, defined as zero by the original
])
def test_reference_direction_matches_every_matlab_branch(u, v, expected):
    """V1 and V2. calculate_refdir_array_f.m enumerates exactly these cases."""
    assert reference_direction(u, v) == pytest.approx(expected, abs=1e-12)


def test_reference_direction_is_from_north_not_from_east():
    """V2 specifically. A due-east wind is pi/2 here and would be 0 under atan2(v, u)."""
    assert reference_direction(1.0, 0.0) == pytest.approx(np.pi / 2)
    assert reference_direction(0.0, 1.0) == pytest.approx(0.0)


def test_reference_direction_stays_in_zero_to_two_pi():
    rng = np.random.default_rng(0)
    u, v = rng.normal(size=500), rng.normal(size=500)
    d = reference_direction(u, v)
    assert np.all(d >= 0.0) and np.all(d < 2 * np.pi)


# --- projection ----------------------------------------------------------------------

def test_projection_onto_its_own_direction_is_the_full_magnitude():
    u, v = 3.0, 4.0
    d = reference_direction(u, v)
    assert project_onto_direction(u, v, d) == pytest.approx(5.0)


def test_projection_perpendicular_is_zero():
    d = reference_direction(0.0, 1.0)          # north
    assert project_onto_direction(1.0, 0.0, d) == pytest.approx(0.0, abs=1e-12)


def test_projection_opposite_is_negative_magnitude():
    d = reference_direction(0.0, 1.0)
    assert project_onto_direction(0.0, -2.0, d) == pytest.approx(-2.0)


def test_calm_wind_projects_to_zero_not_nan():
    """V10. The original assigns 0 rather than consulting an undefined angle."""
    out = project_onto_direction(np.array([0.0]), np.array([0.0]), np.array([1.234]))
    assert out[0] == 0.0 and np.isfinite(out[0])


# --- grid spacing --------------------------------------------------------------------

def test_zonal_spacing_uses_cosine_of_latitude():
    """V3 and V4. sin((lat+90) deg) in the original is cos(lat); check the value."""
    dx, dy = grid_spacing_m(0.0, 1.0, 1.0)
    assert dx == pytest.approx(np.deg2rad(1.0) * EARTH_RADIUS_M)
    dx60, _ = grid_spacing_m(60.0, 1.0, 1.0)
    assert dx60 == pytest.approx(dx * 0.5, rel=1e-9)      # cos 60 = 1/2, sin 60 would be 0.866


def test_zonal_spacing_shrinks_toward_the_pole_and_meridional_does_not():
    dx0, dy0 = grid_spacing_m(0.0, 1.0, 1.0)
    dx60, dy60 = grid_spacing_m(60.0, 1.0, 1.0)
    assert dx60 < dx0
    assert dy60 == pytest.approx(dy0)


def test_meridional_spacing_is_one_degree_of_arc():
    _, dy = grid_spacing_m(np.array([12.0]), 1.0, 1.0)
    assert dy[0] == pytest.approx(np.deg2rad(1.0) * EARTH_RADIUS_M)


# --- the decomposition, against fields with known vorticity --------------------------

def test_uniform_flow_has_zero_vorticity():
    latgrid, longrid = mesh()
    u = np.full(latgrid.shape, 7.0)
    v = np.zeros_like(u)
    rv, sv, cv = component_vorticity(latgrid, longrid, u, v)
    inner = np.isfinite(rv)
    assert np.allclose(rv[inner], 0.0, atol=1e-12)
    assert np.allclose(sv[inner], 0.0, atol=1e-12)
    assert np.allclose(cv[inner], 0.0, atol=1e-12)


def test_the_three_components_sum_exactly():
    """The original defines curvature as the residual, so this identity is exact."""
    rng = np.random.default_rng(3)
    latgrid, longrid = mesh()
    u = rng.normal(size=latgrid.shape) * 5
    v = rng.normal(size=latgrid.shape) * 5
    rv, sv, cv = component_vorticity(latgrid, longrid, u, v)
    inner = np.isfinite(rv)
    assert np.allclose(cv[inner], rv[inner] - sv[inner], atol=0)


def test_straight_flow_with_lateral_shear_is_all_shear_and_no_curvature():
    """A westerly jet varying only with latitude curves nowhere, so curvature is zero
    and the whole of the relative vorticity is shear. This is the physical statement the
    decomposition exists to make, and V6 and V9 both break it."""
    latgrid, longrid = mesh(step=0.5)
    u = 10.0 * np.exp(-((latgrid / 5.0) ** 2))
    v = np.zeros_like(u)
    rv, sv, cv = component_vorticity(latgrid, longrid, u, v)
    inner = np.isfinite(rv)
    assert np.max(np.abs(cv[inner])) < 1e-10
    assert np.allclose(sv[inner], rv[inner], atol=1e-12)
    assert np.max(np.abs(rv[inner])) > 1e-7      # the field really does have vorticity


def test_solid_body_rotation_splits_equally_between_curvature_and_shear():
    """Rigid rotation is the textbook equal split, not an all-curvature case.

    In natural coordinates the vorticity is V/R minus the shear across the flow. For
    solid-body rotation V = omega*r, so the curvature term is omega and the shear term is
    also omega, summing to the 2*omega a Cartesian calculation gives. Getting this exact
    ratio right is a much stronger check than an inequality, and an earlier draft of this
    test asserted all-curvature, which was wrong about the physics rather than about the
    port.
    """
    latgrid, longrid = mesh(lat0=-4, lat1=4, lon0=-4, lon1=4, step=0.25)
    u = -latgrid                      # metres per second per degree
    v = longrid
    rv, sv, cv = component_vorticity(latgrid, longrid, u, v)
    inner = np.isfinite(rv)

    # one degree of latitude in metres, so omega has the units the code works in
    omega = 1.0 / (np.deg2rad(1.0) * EARTH_RADIUS_M)
    assert np.allclose(cv[inner], omega, rtol=2e-3)
    assert np.allclose(sv[inner], omega, rtol=2e-3)
    assert np.allclose(rv[inner], 2 * omega, rtol=2e-3)
    assert np.all(cv[inner] > 0)                 # counterclockwise is positive


def test_a_constant_speed_vortex_is_almost_entirely_curvature():
    """The genuine all-curvature case. With the speed constant along and across circular
    streamlines the shear term vanishes and the vorticity is V/R alone, so this is what
    isolates the curvature branch."""
    latgrid, longrid = mesh(lat0=-6, lat1=6, lon0=-6, lon1=6, step=0.25)
    radius = np.hypot(longrid, latgrid)
    theta = np.arctan2(latgrid, longrid)
    speed = 10.0
    with np.errstate(invalid="ignore", divide="ignore"):
        u = -speed * np.sin(theta)
        v = speed * np.cos(theta)
    u[radius == 0] = 0.0
    v[radius == 0] = 0.0

    rv, sv, cv = component_vorticity(latgrid, longrid, u, v)
    # the centre is a genuine singularity of the streamline geometry, so exclude it
    away = np.isfinite(rv) & (radius > 2.0)
    assert np.max(np.abs(sv[away])) < 0.1 * np.max(np.abs(cv[away]))
    assert np.all(cv[away] > 0)


def test_counter_rotation_flips_the_sign_of_curvature():
    """V6. A sign-flipped residual would leave the magnitudes right and the sign wrong."""
    latgrid, longrid = mesh(lat0=-4, lat1=4, lon0=-4, lon1=4, step=0.25)
    _, _, cw = component_vorticity(latgrid, longrid, latgrid, -longrid)
    inner = np.isfinite(cw)
    assert np.all(cw[inner] < 0)


# --- the masking contract, including the asymmetry the original actually has ----------

def test_latitude_edges_are_not_computed():
    """V5."""
    latgrid, longrid = mesh()
    rng = np.random.default_rng(5)
    u = rng.normal(size=latgrid.shape)
    v = rng.normal(size=latgrid.shape)
    rv, _, _ = component_vorticity(latgrid, longrid, u, v)
    assert np.all(np.isnan(rv[:EDGE_WIDTH, :]))
    assert np.all(np.isnan(rv[-EDGE_WIDTH:, :]))


def test_regional_grid_masks_both_longitude_edges():
    latgrid, longrid = mesh()
    rng = np.random.default_rng(6)
    u = rng.normal(size=latgrid.shape)
    v = rng.normal(size=latgrid.shape)
    rv, _, _ = component_vorticity(latgrid, longrid, u, v, global_grid=False)
    assert np.all(np.isnan(rv[:, :EDGE_WIDTH]))
    assert np.all(np.isnan(rv[:, -EDGE_WIDTH:]))


def test_global_grid_wraps_the_left_edge_but_still_masks_the_right():
    """V11, and this is the finding worth protecting. In the MATLAB the right-hand mask
    sits OUTSIDE the globflag guard:

        elseif ( (globflag == 0) & (h <= uncalc) ) | (h > (nlon-uncalc));

    so a global grid computes its left edge and masks its right. Symmetrizing that is
    the tempting cleanup and would put this port quietly out of agreement with version 1.
    """
    latgrid, longrid = mesh(lon0=0.0, lon1=359.0, step=1.0)
    rng = np.random.default_rng(7)
    u = rng.normal(size=latgrid.shape)
    v = rng.normal(size=latgrid.shape)
    rv, _, _ = component_vorticity(latgrid, longrid, u, v, global_grid=True)
    interior = slice(EDGE_WIDTH, -EDGE_WIDTH)
    assert np.all(np.isfinite(rv[interior, :EDGE_WIDTH])), "the left edge should wrap"
    assert np.all(np.isnan(rv[:, -EDGE_WIDTH:])), "the right edge is masked even so"


def test_fill_value_can_reproduce_the_matlab_sentinel():
    latgrid, longrid = mesh()
    u = np.zeros(latgrid.shape)
    rv, _, _ = component_vorticity(latgrid, longrid, u, u, fill=MATLAB_FILL)
    assert np.all(rv[:EDGE_WIDTH, :] == MATLAB_FILL)
    assert not np.any(np.isnan(rv))


# --- shape and refusal contracts -----------------------------------------------------

def test_time_axis_is_preserved_and_two_dimensional_input_is_squeezed():
    latgrid, longrid = mesh()
    rng = np.random.default_rng(8)
    u3 = rng.normal(size=(4,) + latgrid.shape)
    v3 = rng.normal(size=(4,) + latgrid.shape)
    rv3, _, _ = component_vorticity(latgrid, longrid, u3, v3)
    assert rv3.shape == u3.shape
    rv2, _, _ = component_vorticity(latgrid, longrid, u3[0], v3[0])
    assert rv2.shape == latgrid.shape
    assert np.allclose(rv3[0][np.isfinite(rv3[0])], rv2[np.isfinite(rv2)])


def test_each_time_step_is_independent():
    """A time-coupled implementation would pass every single-step test."""
    latgrid, longrid = mesh()
    rng = np.random.default_rng(9)
    a_u, a_v = rng.normal(size=latgrid.shape), rng.normal(size=latgrid.shape)
    b_u, b_v = rng.normal(size=latgrid.shape), rng.normal(size=latgrid.shape)
    alone, _, _ = component_vorticity(latgrid, longrid, a_u, a_v)
    together, _, _ = component_vorticity(latgrid, longrid,
                                         np.stack([a_u, b_u]), np.stack([a_v, b_v]))
    ok = np.isfinite(alone)
    assert np.allclose(together[0][ok], alone[ok])


@pytest.mark.parametrize("shape", [(3, 3), (5, 4), (4, 5)])
def test_a_grid_too_small_to_compute_refuses(shape):
    latgrid = np.zeros(shape)
    longrid = np.zeros(shape)
    with pytest.raises(ValueError, match="too small"):
        component_vorticity(latgrid, longrid, np.zeros(shape), np.zeros(shape))


def test_mismatched_wind_and_mesh_refuses():
    latgrid, longrid = mesh()
    bad = np.zeros((latgrid.shape[0], latgrid.shape[1] + 1))
    with pytest.raises(ValueError, match="match the coordinate mesh"):
        component_vorticity(latgrid, longrid, bad, bad)


def test_mismatched_u_and_v_refuses():
    latgrid, longrid = mesh()
    with pytest.raises(ValueError, match="same shape"):
        component_vorticity(latgrid, longrid,
                            np.zeros(latgrid.shape), np.zeros((2,) + latgrid.shape))
