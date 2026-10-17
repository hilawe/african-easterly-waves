"""The region geometry inspection's fixed readings: the transect points, the interior dip
with missing samples, the typed wind reversals and their orientation rule, and the
outcome classes."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import pilot_eastern_region_geometry as RG  # noqa: E402


def test_the_transect_runs_end_to_end_at_half_degree_steps():
    pts = RG.transect((10.0, 40.0), (10.0, 43.0))
    assert pts[0] == (10.0, 40.0) and pts[-1] == (10.0, 43.0) and len(pts) == 7


def test_an_interior_dip_must_lie_below_both_ends_and_missing_samples_leave_it_unknown():
    assert RG.interior_dip([3.0, 1.0, 2.0])["dip"] is True and RG.interior_dip([3.0, 1.0, 2.0])["depth"] == 1.0
    assert RG.interior_dip([3.0, 2.5, 2.0])["dip"] is False
    assert RG.interior_dip([3.0, None, 2.0]) is None
    assert RG.interior_dip([3.0, None, 1.0, 2.0]) is None                 # a gap anywhere leaves the dip unknown


def test_reversals_are_typed_going_east_and_interpreted_only_near_zonal():
    r = RG.reversals([-2.0, 1.0, -1.0], (10.0, 40.0), (10.0, 44.0))
    assert r == {"within_45_deg_of_zonal": True, "trough_type": 1, "ridge_type": 1, "interpreted": True}
    west = RG.reversals([-1.0, 1.0], (10.0, 44.0), (10.0, 40.0))         # traversed westward, so read reversed
    assert west["ridge_type"] == 1 and west["trough_type"] == 0
    assert RG.reversals([-1.0, 1.0], (10.0, 40.0), (16.0, 41.0))["interpreted"] is False


def test_the_outcome_classes():
    yes, no = {"dip": True, "depth": 1.0}, {"dip": False, "depth": 0.0}
    assert RG.outcome(yes, yes) == "distinguishable lobes" and RG.outcome(no, no) == "extended structure"
    assert RG.outcome(yes, no) == "unresolved geometry" and RG.outcome(None, yes) == "unresolved geometry"
