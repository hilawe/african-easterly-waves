"""The extent verification: the wider wide case's grids contain the narrower one's in
order, the raw fields are equal on every shared cell, and the two margin preparations are
equal on every cell of the narrower control's grids, so that a difference between the two
track records can be read as an extent effect and nothing else."""
import importlib.util
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from test_pilot_alledge_replay import analytic_case  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name + "_under_test", os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_shared_raw_and_prepared_fields_are_equal_and_a_planted_difference_is_counted_only_inside_the_shared_cells():
    V = _load("pilot_extent_verification")
    wide_a, case_a = analytic_case(8.0, -10.0, 10.0), analytic_case(4.0, -6.0, 6.0)
    wide_b, case_b = analytic_case(8.0, -10.0, 20.0), analytic_case(4.0, -6.0, 16.0)
    out = V.verify(wide_a, case_a, wide_b, case_b)
    assert out["grids"]["wide_b_contains_wide_a_in_order"] and out["grids"]["control_b_contains_control_a_in_order"]
    assert all(v == 0 for v in out["raw_shared_cells_differing"].values()) and len(out["raw_shared_cells_differing"]) == 7
    assert all(v == 0 for v in out["prepared_control_a_cells_differing"].values()) and len(out["prepared_control_a_cells_differing"]) == 6
    assert out["steps"] == 12 and out["extent_is_the_only_difference"] is True
    inside = wide_b.copy()
    inside["u"] = inside["u"].copy()
    r, c = np.where(wide_b["latgrid"][:, 0] == 2.0)[0][0], np.where(wide_b["longrid"][0] == 3.0)[0][0]          # a cell inside control a
    inside["u"][3, r, c] += 1.0
    out_in = V.verify(wide_a, case_a, inside, case_b)
    assert out_in["raw_shared_cells_differing"]["u"] == 1 and out_in["prepared_control_a_cells_differing"]["fine_zonal_wind_smoothed"] > 0
    assert out_in["extent_is_the_only_difference"] is False
    outside = wide_b.copy()
    outside["u"] = outside["u"].copy()
    c_out = np.where(wide_b["longrid"][0] == 18.0)[0][0]                                                           # a cell beyond wide a
    outside["u"][3, r, c_out] += 1.0
    out_out = V.verify(wide_a, case_a, outside, case_b)
    assert out_out["raw_shared_cells_differing"]["u"] == 0 and out_out["prepared_control_a_cells_differing"]["fine_zonal_wind_smoothed"] == 0
    assert out_out["extent_is_the_only_difference"] is True
    with pytest.raises(SystemExit, match="not a subset"):
        V.verify(wide_a, case_a, analytic_case(8.0, -8.0, 20.0), case_b)                                           # wide b does not contain wide a


def test_the_comparison_tool_names_the_same_verification_fields_the_instrument_writes():
    V = _load("pilot_extent_verification")
    MC = _load("pilot_margin_compare")
    import pilot_alledge_replay as A
    assert tuple(MC.VERIFICATION_RAW_FIELDS) == tuple(V.RAW)
    assert tuple(MC.VERIFICATION_PREPARED_FIELDS) == tuple(name for name, _grid in A.PREPARED)
