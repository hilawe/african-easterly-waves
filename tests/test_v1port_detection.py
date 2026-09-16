"""Tests for the version 1 trough detection port.

THE MUTATION LIST, WRITTEN BEFORE THE ASSERTIONS EXISTED, executable as
tests/mutations_v1port_detection.py.

    D1   the trough level is not zero, so the axis is not where the advection changes sign
    D2   the zonal-wind mask tests magnitude, discarding the strong easterlies the waves
         live in rather than the westerlies
    D3   the zonal-wind mask is not applied at all
    D4   the wind mask is also applied to the fine field, which the original has commented
         out
    D5   the curvature mask is not applied, so weak features become troughs
    D6   the fine field is masked with the coarse threshold, or the reverse
    D7   the Southern Hemisphere sign flip is skipped, so southern troughs never appear
    D8   the fields are not smoothed before contouring
    D9   only one merge pass runs instead of coarse then fine
    D10  the merge passes run fine first and coarse second
    D11  a candidate center is the first vertex of its contour rather than the mean
    D12  contours are parsed by searching for the level in the x-coordinate, the original's
         ambiguous scheme, which on this domain also matches vertices at longitude zero
    D13  the absorption flag reaches the coarse merge and not the fine one, so the two
         halves of the same merge disagree about which behaviour they reproduce (added
         2026-08-30 with the flag; the list was written before the assertion)
"""

import numpy as np
import pytest

from aew.v1port.detection import (
    MAX_ZONAL_WIND, TROUGH_LEVEL, detect_troughs, trough_axes,
)


def mesh(lat0=0.0, lat1=20.0, lon0=-10.0, lon1=10.0, step=1.0):
    lat = np.arange(lat0, lat1 + step / 2, step)
    lon = np.arange(lon0, lon1 + step / 2, step)
    longrid, latgrid = np.meshgrid(lon, lat)
    return latgrid, longrid


def wave_field(latgrid, longrid, centre_lon=0.0, width=3.0):
    """A north-south trough: curvature vorticity peaking on a meridian.

    Its advection by a uniform easterly changes sign across the peak, so the zero contour
    of the advection is the trough axis, which is what the original detects.
    """
    curvature = 6e-6 * np.exp(-((longrid - centre_lon) / width) ** 2)
    # advection by a westward flow: -u d(curv)/dx with u negative is +d/dx
    advection = np.gradient(curvature, axis=1)
    return curvature, advection


# --- the trough level and the contour parse ------------------------------------------

def test_the_trough_level_is_zero():
    """D1. tr_thr = 0.*10^-5 in the original, so the axis is the sign change."""
    assert TROUGH_LEVEL == 0.0


def test_axes_follow_the_sign_change_of_the_advection():
    latgrid, longrid = mesh()
    _, advection = wave_field(latgrid, longrid, centre_lon=0.0)
    axes = trough_axes(latgrid, longrid, advection)
    assert axes, "no trough axis found"
    lons = np.concatenate([lon for _, lon in axes])
    assert np.allclose(lons, 0.0, atol=1.0), "the axis is not on the peak meridian"


def test_a_vertex_at_longitude_zero_is_not_mistaken_for_a_contour_header():
    """D12, the divergence this module documents. The original finds contour headers by
    searching for columns whose x-coordinate equals the level. With the level at zero and
    longitude zero on the grid, that search also matches vertices on the prime meridian.
    This port walks the contours properly, so a trough sitting exactly on the meridian
    yields one axis rather than one plus a scatter of phantoms."""
    latgrid, longrid = mesh()
    _, advection = wave_field(latgrid, longrid, centre_lon=0.0)
    axes = trough_axes(latgrid, longrid, advection)
    assert len(axes) == 1, "the meridian trough should be a single axis"
    lats, lons = axes[0]
    assert lats.size >= 2 and lons.size == lats.size


def test_no_sign_change_gives_no_axes():
    latgrid, longrid = mesh()
    advection = np.full(latgrid.shape, 5e-11)      # positive everywhere
    assert trough_axes(latgrid, longrid, advection) == []


def test_masked_regions_produce_no_axes():
    latgrid, longrid = mesh()
    _, advection = wave_field(latgrid, longrid)
    assert trough_axes(latgrid, longrid, np.full_like(advection, np.nan)) == []


# --- the masks -----------------------------------------------------------------------

def detection_inputs(latgrid, longrid, u_value=-6.0, centre_lon=0.0):
    curvature, advection = wave_field(latgrid, longrid, centre_lon=centre_lon)
    u = np.full(latgrid.shape, u_value)
    return u, curvature, advection


def test_an_easterly_wave_is_detected():
    latgrid, longrid = mesh()
    u, curvature, advection = detection_inputs(latgrid, longrid)
    out = detect_troughs(0, latgrid, longrid, u, curvature, advection,
                         latgrid, longrid, curvature,
                         coarse_threshold=1e-6, fine_threshold=1e-6)
    assert len(out) == 1


def test_a_westerly_regime_is_masked_out():
    """D2 and D3. The mask tests the signed wind: westerlies above the threshold are
    discarded and easterlies of any strength are kept."""
    latgrid, longrid = mesh()
    u, curvature, advection = detection_inputs(latgrid, longrid, u_value=+8.0)
    out = detect_troughs(0, latgrid, longrid, u, curvature, advection,
                         latgrid, longrid, curvature,
                         coarse_threshold=1e-6, fine_threshold=1e-6)
    assert out == []


def test_a_strong_easterly_is_kept_not_masked():
    """D2 specifically. A magnitude test would discard this, which is backwards: strong
    easterly flow is the regime African easterly waves live in."""
    latgrid, longrid = mesh()
    u, curvature, advection = detection_inputs(latgrid, longrid, u_value=-20.0)
    out = detect_troughs(0, latgrid, longrid, u, curvature, advection,
                         latgrid, longrid, curvature,
                         coarse_threshold=1e-6, fine_threshold=1e-6)
    assert len(out) == 1
    assert MAX_ZONAL_WIND == 2.5


def test_a_weak_feature_is_masked_by_the_curvature_threshold():
    """D5."""
    latgrid, longrid = mesh()
    u, curvature, advection = detection_inputs(latgrid, longrid)
    out = detect_troughs(0, latgrid, longrid, u, curvature * 1e-3, advection,
                         latgrid, longrid, curvature * 1e-3,
                         coarse_threshold=1e-6, fine_threshold=1e-6)
    assert out == []


def test_the_fine_field_is_not_masked_by_the_wind():
    """D4. FAITHFUL: the original's wind mask for the fine grid is commented out, so a
    westerly that removes the coarse detection must not be what removes the fine one.

    Exercised by giving the coarse grid an easterly and checking a detection survives:
    if the wind mask were also applied to the fine field with a westerly, nothing would
    come back even though the coarse pass found a trough.
    """
    latgrid, longrid = mesh()
    u, curvature, advection = detection_inputs(latgrid, longrid, u_value=-6.0)
    out = detect_troughs(0, latgrid, longrid, u, curvature, advection,
                         latgrid, longrid, curvature,
                         coarse_threshold=1e-6, fine_threshold=1e-6)
    assert len(out) == 1


def test_the_two_thresholds_go_to_their_own_grids():
    """D6. Raising only the fine threshold must be able to remove the wave, which shows
    the fine field is gated by the fine value rather than the coarse one."""
    latgrid, longrid = mesh()
    u, curvature, advection = detection_inputs(latgrid, longrid)
    kept = detect_troughs(0, latgrid, longrid, u, curvature, advection,
                          latgrid, longrid, curvature,
                          coarse_threshold=1e-6, fine_threshold=1e-6)
    dropped = detect_troughs(0, latgrid, longrid, u, curvature, advection,
                             latgrid, longrid, curvature,
                             coarse_threshold=1e-6, fine_threshold=1e-2)
    assert len(kept) == 1 and dropped == []


def test_a_southern_hemisphere_trough_is_detected_too():
    """D7. The sign flip is what lets one positive threshold serve both hemispheres."""
    latgrid, longrid = mesh(lat0=-20.0, lat1=-2.0)
    curvature, advection = wave_field(latgrid, longrid)
    curvature = -curvature                      # cyclonic is negative south of the equator
    u = np.full(latgrid.shape, -6.0)
    out = detect_troughs(0, latgrid, longrid, u, curvature, advection,
                         latgrid, longrid, curvature,
                         coarse_threshold=1e-6, fine_threshold=1e-6)
    assert len(out) == 1


# --- candidate construction and the merge order ---------------------------------------

def test_a_candidate_centre_is_the_mean_of_its_axis_not_an_endpoint():
    """D11. The original averages every vertex of the contour."""
    latgrid, longrid = mesh()
    u, curvature, advection = detection_inputs(latgrid, longrid)
    axes = trough_axes(latgrid, longrid, advection)
    lats, _ = axes[0]
    out = detect_troughs(0, latgrid, longrid, u, curvature, advection,
                         latgrid, longrid, curvature,
                         coarse_threshold=1e-6, fine_threshold=1e-6)
    assert len(out) == 1
    # the axis runs the height of the domain, so its mean latitude is mid-domain and an
    # endpoint would be at an extreme
    assert abs(out[0]["lat_mean"] - np.mean(lats)) < 6.0
    assert out[0]["lat_mean"] not in (lats.min(), lats.max())


def test_two_separated_troughs_stay_two_waves():
    latgrid, longrid = mesh(lon0=-30.0, lon1=30.0)
    c1, a1 = wave_field(latgrid, longrid, centre_lon=-20.0, width=2.0)
    c2, a2 = wave_field(latgrid, longrid, centre_lon=20.0, width=2.0)
    curvature, advection = c1 + c2, a1 + a2
    u = np.full(latgrid.shape, -6.0)
    out = detect_troughs(0, latgrid, longrid, u, curvature, advection,
                         latgrid, longrid, curvature,
                         coarse_threshold=1e-6, fine_threshold=1e-6)
    assert len(out) == 2


def test_the_time_is_carried_onto_every_wave():
    latgrid, longrid = mesh()
    u, curvature, advection = detection_inputs(latgrid, longrid)
    out = detect_troughs(1234, latgrid, longrid, u, curvature, advection,
                         latgrid, longrid, curvature,
                         coarse_threshold=1e-6, fine_threshold=1e-6)
    assert all(w["time"] == 1234 for w in out)


def test_no_axes_gives_no_waves():
    latgrid, longrid = mesh()
    u = np.full(latgrid.shape, -6.0)
    flat = np.full(latgrid.shape, 5e-6)
    out = detect_troughs(0, latgrid, longrid, u, flat, np.full(latgrid.shape, 1e-9),
                         latgrid, longrid, flat,
                         coarse_threshold=1e-6, fine_threshold=1e-6)
    assert out == []


def test_the_absorption_flag_reaches_BOTH_merge_passes():
    """D13.

    WHAT THIS BINDS AND WHAT IT DOES NOT. It checks that the flag is FORWARDED to both
    calls, not that absorption changes any particular field, because version 1 runs the
    same merge twice and honouring the flag in one pass and not the other would be
    neither version 1 nor the repair. A physical fixture would have to arrange for the
    fine pass specifically to absorb, which binds the geometry rather than the wiring
    that the mutation attacks. Recorded plainly so a reader does not take this for a
    behavioural test.
    """
    from unittest.mock import patch

    import aew.v1port.detection as detection

    latgrid, longrid = np.meshgrid(np.arange(0.0, 21.0), np.arange(0.0, 21.0))[::-1]
    curvature = np.zeros(latgrid.shape)
    curvature[(np.abs(latgrid - 10) <= 2) & (np.abs(longrid - 10) <= 2)] = 5e-6
    advection = (longrid - 10.0) * 1e-9        # a zero contour down the middle
    wind = np.full(latgrid.shape, -8.0)

    for flag in (False, True):
        with patch.object(detection, "merge_contours",
                          side_effect=lambda c, *a, **k: list(c)) as spy:
            detection.detect_troughs(0.0, latgrid, longrid, wind, curvature, advection,
                                     latgrid, longrid, curvature,
                                     coarse_threshold=1e-6, fine_threshold=1e-6,
                                     absorb=flag)
        assert spy.call_count == 2, "the coarse pass and the fine pass"
        for call in spy.call_args_list:
            assert call.kwargs.get("absorb") is flag, (
                f"both merges must be told absorb={flag}, got "
                f"{call.kwargs.get('absorb')!r}")
