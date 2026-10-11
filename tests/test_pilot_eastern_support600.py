"""The 600 hPa support states on synthetic fields: a supported block, an unsupported block,
partially unavailable blocks, and the terrain, finite-wind and stencil rules that decide
which 600 hPa cells are valid."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import pilot_eastern_support600 as SU  # noqa: E402

THR = 1.0                                                            # 1.0e-5 per second, in units of 1e-5 per second


def blank(value=0.5):
    return np.full((5, 5), value), np.full((5, 5), "valid", dtype=object)


def test_one_valid_cell_above_the_threshold_supports_even_among_invalid_cells():
    z, cls = blank()
    cls[1:4, 1:4] = "terrain"
    cls[1, 3], z[1, 3] = "valid", 2.0
    s = SU.support_state(z, cls, 2, 2, THR)
    assert s["state"] == "supported" and s["valid_cells"] == 1 and s["cells_above"] == 1 and s["invalid_reasons"] == {"terrain": 8}


def test_nine_valid_cells_none_above_is_unsupported_and_equality_is_not_above():
    z, cls = blank()
    z[2, 2] = THR
    s = SU.support_state(z, cls, 2, 2, THR)
    assert s["state"] == "unsupported" and s["valid_cells"] == 9 and s["cells_above"] == 0


def test_a_partially_unavailable_block_without_support_is_unassessable_not_unsupported():
    z, cls = blank()
    cls[3, 1] = "neighbor_invalid"
    z[1, 1] = 7.0
    cls[1, 1] = "terrain"                                           # a large value in an invalid cell does not support
    s = SU.support_state(z, cls, 2, 2, THR)
    assert s["state"] == "unassessable" and s["valid_cells"] == 7 and s["invalid_reasons"] == {"neighbor_invalid": 1, "terrain": 1}
    s = SU.support_state(*blank(), 0, 0, THR)                         # a block reaching past the grid
    assert s["state"] == "unassessable" and s["invalid_reasons"] == {"outside_grid": 5}


def test_terrain_finite_winds_and_the_stencil_decide_validity_at_600_hpa():
    lat1, lon1 = np.arange(20.0, 9.5, -1.0), np.arange(30.0, 40.5, 1.0)
    LON = np.meshgrid(lat1, lon1, indexing="ij")[1]
    u, v = np.zeros(LON.shape), 2.0 * (LON - 35.0)                  # cyclonic shear, about 1.9e-5 per second
    sp = np.full(LON.shape, 65000.0)                                 # above 600 hPa, below 700 hPa
    sp[3, 3] = 59000.0
    u[7, 7] = np.nan
    z6, c6 = SU.vorticity(u, v, sp, 600.0, lat1, lon1)
    z7, c7 = SU.vorticity(u, v, sp, 700.0, lat1, lon1)
    assert np.all(np.isnan(z7)) and set(c7.ravel()) == {"terrain"}
    assert c6[3, 3] == "terrain" and c6[3, 4] == "neighbor_invalid" and c6[2, 3] == "neighbor_invalid"
    assert c6[7, 7] == "wind_missing" and c6[7, 6] == "neighbor_invalid" and np.isnan(z6[7, 6])
    assert c6[0, 5] == "edge" and np.isnan(z6[0, 5])
    expect = 2.0 / (6371000.0 * np.cos(np.deg2rad(15.0)) * np.deg2rad(1.0)) * 1e5
    assert abs(z6[5, 5] - expect) < 1e-3 * expect and c6[5, 5] == "valid"
    assert SU.support_state(z6, c6, 7, 7, THR)["state"] == "supported"   # its valid neighbors are above the threshold
    weak_z, weak_c = SU.vorticity(u, 0.25 * v, sp, 600.0, lat1, lon1)
    assert SU.support_state(weak_z, weak_c, 5, 5, THR)["state"] == "unsupported"
    assert SU.support_state(weak_z, weak_c, 7, 7, THR)["state"] == "unassessable"


def test_shares_leave_unassessable_out_and_the_controls_criterion_is_directional():
    t = {"supported": 2, "unsupported": 2, "unassessable": 6, "total": 10}
    assert SU.supported_share_of_assessable(t) == 0.5
    assert SU.supported_share_of_assessable({"supported": 0, "unsupported": 0, "unassessable": 3, "total": 3}) is None
    assert SU.controls_against(0.6, 0.6) is True and SU.controls_against(0.6, 0.5) is False and SU.controls_against(None, 0.5) is None
