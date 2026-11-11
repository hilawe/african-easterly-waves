"""Tests for scripts/continuity_placement_check.py (`CONTINUITY_ACROSS_AFRICA_2026-11-09.md`, section 25).

The constructed cases of the review that led to it come first: carriage at the two ends only, a
late pickup inside the 48-hour start window, an internal gap, and copies of one history."""

import copy
import gzip
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import continuity_placement_check as P  # noqa: E402

E, L = P.E, P.L
D0 = 7000.0


def q(tid, lats, lons, start=D0):
    n = len(lats)
    return {"id": tid, "time": start + 0.25 * np.arange(n), "lat": np.asarray(lats, float), "lon": np.asarray(lons, float), "n_valid": n}


def hist(hid, lats, lons, start=D0, est=None):
    n = len(lats)
    if est is None:
        est = [[float(lats[j]), float(lons[j]), 6] for j in range(1, n)]
    return {"id": hid, "time": start + 0.25 * np.arange(n), "lat": np.asarray(lats, float), "lon": np.asarray(lons, float),
            "step": list(range(n)), "raw_lat": [float(x) for x in lats], "raw_lon": [float(x) for x in lons], "est": est, "fate": "finished"}


def ref_row(tid, first_lon, last, atlantic=True):
    row = {"id": tid, "first": {"lon": first_lon}, "outcome": "atlantic_side" if atlantic else "none", "last_used": {"index": last}}
    if atlantic:
        row["first_atlantic_side"] = {"index": last}
    return row


REF_LONS = [20.0 - 2.0 * k for k in range(12)]                       # about 219 km a step on 10 N


def reference(n=12, tid=9):
    lons = [20.0 - 2.0 * k for k in range(n)]
    return q(tid, [10.0] * n, lons), lons


def beside(hid, indices, lons, away=35.0):
    """A history at the reference's position at the given indices and far to the north elsewhere."""
    n = len(lons)
    return hist(hid, [10.0 if k in indices else away for k in range(n)], lons)


def rows_for(histories, refs, q_rows, extra=(), group_of=None):
    qs = list(refs) + list(extra)
    out, base = P.reference_rows(histories, group_of or {}, {x["id"]: x for x in qs}, E.C.index_by_time(qs), E.J.spans(qs), q_rows)
    return out, base


# --- the path --------------------------------------------------------------------------------------

def test_a_history_beside_the_whole_reference():
    ref, _ = reference()
    assert P.path(ref, 12, list(range(12))) == {"count": 12, "first": 0, "last": 11, "pickup_hours": 0, "pickup_km": 0.0, "start": True, "end": True,
                                               "majority": True, "longest_gap_hours": 6, "no_long_gap": True}


def test_the_two_ends_alone_are_not_a_majority_and_hold_a_long_gap():
    ref, _ = reference()
    p = P.path(ref, 12, [0, 11])
    assert (p["start"], p["end"], p["majority"], p["longest_gap_hours"], p["no_long_gap"]) == (True, True, False, 66, False)


def test_a_pickup_is_late_once_the_reference_has_traveled_more_than_500_km():
    ref, _ = reference()
    near = P.path(ref, 12, list(range(2, 12)))                        # 4 degrees, about 438 km
    late = P.path(ref, 12, list(range(3, 12)))                        # 6 degrees, about 657 km
    assert near["start"] is True and near["pickup_hours"] == 12 and near["pickup_km"] == pytest.approx(438.0, abs=1.0)
    assert late["start"] is False and late["pickup_hours"] == 18 and late["pickup_km"] == pytest.approx(657.0, abs=1.0)
    d = float(E.C.km(10.0, 16.0, 10.0, 20.0))
    assert P.path(ref, 12, [2, 3], radius_km=d)["start"] is True and P.path(ref, 12, [2, 3], radius_km=d - 1e-6)["start"] is False


def test_the_end_is_read_within_500_km_of_the_last_used_observation():
    ref, _ = reference()
    assert P.path(ref, 12, list(range(0, 10)))["end"] is True and P.path(ref, 12, list(range(0, 9)))["end"] is False
    assert P.path(ref, 6, list(range(0, 4)))["end"] is True, "read over six observations, the fourth is 4 degrees from the last"


def test_a_gap_is_long_past_24_hours_between_accompanied_observations():
    ref, _ = reference()
    at_bound = P.path(ref, 12, [0, 1, 2, 3, 7, 8, 9, 10, 11])
    past = P.path(ref, 12, [0, 1, 2, 3, 8, 9, 10, 11])
    assert (at_bound["longest_gap_hours"], at_bound["no_long_gap"]) == (24, True) and (past["longest_gap_hours"], past["no_long_gap"]) == (30, False)
    assert P.path(ref, 12, [0, 1, 2, 3, 8, 9, 10, 11], gap_hours=30)["no_long_gap"] is True


def test_half_the_observations_is_a_majority_and_one_fewer_is_not():
    ref, _ = reference()
    assert P.path(ref, 12, [0, 1, 2, 9, 10, 11])["majority"] is True and P.path(ref, 12, [0, 1, 9, 10, 11])["majority"] is False


def test_the_start_is_the_first_position_the_reference_has():
    ref, _ = reference()
    ref["lat"][0] = np.nan
    p = P.path(ref, 12, list(range(1, 12)))
    assert p["pickup_km"] == 0.0 and p["pickup_hours"] == 0 and p["start"] is True


# --- reference rows ---------------------------------------------------------------------------------

def test_one_history_from_start_to_coast_is_sustained_and_clean():
    ref, lons = reference()
    rows, base = rows_for([hist("h", [10.0] * 12, lons)], [ref], [ref_row(9, 20.0, 11)])
    r = rows[0]
    assert (r["endpoint_carried"], r["sustained"], r["unmixed_sustained"], r["endpoint_carriers"], r["unmixed_sustained_carrier_units"]) == (True, True, True, 1, 1)
    assert r["primary"]["history"] == "h" and r["primary"]["status"] == "clean" and r["best_count"] == 12 and base[0]["class"] == "carried"


def test_carriage_at_the_two_ends_only_is_endpoint_carried_and_not_sustained():
    ref, lons = reference()
    r = rows_for([beside("ends", {0, 11}, lons)], [ref], [ref_row(9, 20.0, 11)])[0][0]
    assert r["endpoint_class"] == "carried" and r["endpoint_carried"] is True and r["sustained"] is False and r["unmixed_sustained"] is False
    assert r["primary"]["count"] == 2 and r["primary"]["majority"] is False


def test_a_history_that_stops_short_of_the_coast_or_is_seldom_beside_the_reference_is_not_sustained():
    ref, lons = reference()
    short = rows_for([beside("short", set(range(0, 8)), lons)], [ref], [ref_row(9, 20.0, 11)])[0][0]
    p = short["primary"]
    assert (p["start"], p["end"], p["majority"], p["no_long_gap"]) == (True, False, True, True) and short["sustained"] is False
    seldom = rows_for([beside("seldom", {0, 4, 8, 11}, lons)], [ref], [ref_row(9, 20.0, 11)])[0][0]
    p = seldom["primary"]
    assert (p["start"], p["end"], p["majority"], p["no_long_gap"]) == (True, True, False, True) and seldom["sustained"] is False


def test_a_pickup_at_48_hours_qualifies_for_the_endpoint_class_and_is_late():
    ref, lons = reference(20)
    r = rows_for([beside("late", set(range(8, 20)), lons)], [ref], [ref_row(9, 20.0, 19)])[0][0]
    assert r["endpoint_carried"] is True, "the 48-hour start window of section 3 admits it"
    assert r["primary"]["start"] is False and r["primary"]["pickup_hours"] == 48 and r["sustained"] is False


def test_an_internal_gap_keeps_the_endpoint_class_and_removes_sustained_carriage():
    ref, lons = reference(20)
    gapped = beside("gap", set(range(0, 8)) | set(range(14, 20)), lons)
    r = rows_for([gapped], [ref], [ref_row(9, 20.0, 19)])[0][0]
    assert r["endpoint_carried"] is True and r["primary"]["count"] == 14 and r["primary"]["longest_gap_hours"] == 42 and r["sustained"] is False


def test_a_carrier_that_came_from_a_system_still_present_is_mixed():
    ref, lons = reference(12)
    other = q(1, [10.0] * 16, [30.0, 28.0, 26.0, 24.0] + [40.0] * 12, start=D0 - 1.0)     # beside the history first, then away and alive
    h = hist("m", [10.0] * 16, [30.0, 28.0, 26.0, 24.0] + lons, start=D0 - 1.0)
    rows, _ = rows_for([h], [ref], [ref_row(9, 20.0, 11)], extra=[other])
    r = rows[0]
    assert (r["sustained"], r["unmixed_sustained"], r["primary"]["status"]) == (True, False, "mixed"), "the path is sustained and the history is shared"
    assert r["primary"]["supported_false_link"] is False, "every link matched its prediction"
    assert (r["mixed_with_a_history"], r["open_with_a_history"], r["supported_false_link_with_a_history"], r["clean_sustained"]) == (True, False, False, False)
    assert r["sustained_carriers"] == 1 and r["unmixed_sustained_carriers"] == 0 and r["unmixed_sustained_carrier_units"] == 0


def test_a_carrier_that_followed_an_ended_record_is_open():
    ref, lons = reference(12)
    ended = q(1, [10.0] * 4, [30.0, 28.0, 26.0, 24.0], start=D0 - 1.0)
    h = hist("o", [10.0] * 16, [30.0, 28.0, 26.0, 24.0] + lons, start=D0 - 1.0)
    r = rows_for([h], [ref], [ref_row(9, 20.0, 11)], extra=[ended])[0][0]
    assert (r["unmixed_sustained"], r["primary"]["status"]) == (True, "open"), "a relay is unresolved, not adverse"
    assert (r["clean_sustained"], r["open_with_a_history"], r["mixed_with_a_history"]) == (False, True, False)


def test_a_reference_is_mixed_with_any_history_beside_it_not_only_its_primary():
    ref, lons = reference(12)
    other = q(1, [10.0] * 16, [30.0, 28.0, 26.0, 24.0] + [40.0] * 12, start=D0 - 1.0)
    visitor = hist("v", [10.0] * 7, [30.0, 28.0, 26.0, 24.0] + lons[:3], start=D0 - 1.0)     # came from the other system, beside the reference for three observations
    whole = hist("w", [10.0] * 12, lons)
    rows, _ = rows_for([visitor, whole], [ref], [ref_row(9, 20.0, 11)], extra=[other])
    r = rows[0]
    assert r["primary"]["history"] == "w" and r["primary"]["status"] == "clean" and r["clean_sustained"] is True
    assert r["mixed_with_a_history"] is True and r["histories"] == 2, "the visiting history is shared between two systems"
    assert P.tally(rows)["mixed_with_a_history"] == 1 and P.tally(rows)["primary_mixed"] == 0


def test_a_carrier_that_left_a_system_alive_without_a_position_is_mixed():
    ref, lons = reference(12)
    gone = q(1, [10.0] * 4 + [np.nan] + [10.0], [30.0, 28.0, 26.0, 24.0, np.nan, 60.0], start=D0 - 1.0)   # alive at the handover, no position there
    h = hist("u", [10.0] * 16, [30.0, 28.0, 26.0, 24.0] + lons, start=D0 - 1.0)
    qs = [ref, gone]
    assert [e["kind"] for e in E.events_of(h, E.C.index_by_time(qs), E.J.spans(qs))[0]] == ["undetermined"]
    r = rows_for([h], [ref], [ref_row(9, 20.0, 11)], extra=[gone])[0][0]
    assert (r["unmixed_sustained"], r["primary"]["status"]) == (False, "mixed"), "an unchecked handover counts against the history"


def test_a_supported_false_link_marks_the_primary():
    west = -15.7
    ref = q(9, [10.0] * 2, [20.0, west], start=D0)                                         # the reference the history jumps to
    old = q(1, [10.0] * 2, [0.0, -2.0])                                                     # goes on where the history was predicted
    h = hist("j", [10.0, 10.0], [0.0, west], est=[[10.0, -2.0, 6]])
    qs = [ref, old]
    inv = P.involvement(h, E.C.index_by_time(qs), E.J.spans(qs))
    assert inv == {"mixed": {1, 9}, "open": set(), "supported": {1, 9}}
    r = rows_for([h], [ref], [ref_row(9, 20.0, 1)], extra=[old])[0][0]
    assert r["primary"]["status"] == "mixed" and r["primary"]["supported_false_link"] is True and r["supported_false_link_with_a_history"] is True


def test_copies_of_one_history_change_the_carrier_counts_and_nothing_anchored_to_the_reference():
    ref, lons = reference()
    one = hist("h000", [10.0] * 12, lons)
    copies = [{**copy.deepcopy(one), "id": f"h{k:03d}"} for k in range(1, 101)]
    base_rows, _ = rows_for([one], [ref], [ref_row(9, 20.0, 11)])
    many, _ = rows_for([one] + copies, [ref], [ref_row(9, 20.0, 11)])
    grouped, _ = rows_for([one] + copies, [ref], [ref_row(9, 20.0, 11)], group_of={f"h{k:03d}": 7 for k in range(101)})
    keep = ("endpoint_carried", "sustained", "unmixed_sustained", "best_count", "accompanied", "primary")
    assert {k: many[0][k] for k in keep} == {k: base_rows[0][k] for k in keep}
    assert (many[0]["unmixed_sustained_carriers"], many[0]["unmixed_sustained_carrier_units"], many[0]["endpoint_carrier_units"]) == (101, 101, 101)
    assert (grouped[0]["unmixed_sustained_carriers"], grouped[0]["unmixed_sustained_carrier_units"], grouped[0]["endpoint_carrier_units"]) == (101, 1, 1)
    t = P.tally(many)
    assert (t["unmixed_sustained"], t["unmixed_sustained_with_one_carrier_unit"], t["with_a_primary_history"], t["primary_mixed"]) == (1, 0, 1, 0)
    assert P.tally(grouped)["unmixed_sustained_with_one_carrier_unit"] == 1 and t["histories_beside_references"] == 101


def test_the_review_construction_one_switch_in_6_pairs_becomes_one_in_306():
    t = hist("t", [10.0] * 4, [0.0, -2.0, -4.0, -6.0])
    a = q(1, [10.0] * 4, [0.0, -2.0, 4.0, 8.0])
    b = q(2, [10.0] * 2, [-4.0, -6.0], start=D0 + 0.5)
    ref = q(9, [10.0] * 4, [20.0, 18.0, 16.0, 14.0])
    safe = hist("h", [10.0] * 4, [20.0, 18.0, 16.0, 14.0])
    qs = [a, b, ref]

    def flags(histories):
        return E.flag_measures(histories, E.C.index_by_time(qs), E.J.spans(qs), [])[0]

    one, many = flags([t, safe]), flags([t, safe] + [{**copy.deepcopy(safe), "id": f"c{k}"} for k in range(100)])
    assert (one["events"]["switch"], one["exposure_pairs"]) == (1, 6) and (many["events"]["switch"], many["exposure_pairs"]) == (1, 306)


def test_the_primary_is_the_history_with_most_observations_then_the_earliest():
    ref, lons = reference()
    early = beside("zz_early", set(range(0, 6)), lons)
    late = beside("aa_late", set(range(6, 12)), lons)
    longer = beside("mm_longer", set(range(2, 9)), lons)
    r = rows_for([late, early], [ref], [ref_row(9, 20.0, 11)])[0][0]
    assert r["primary"]["history"] == "zz_early" and r["histories"] == 2
    assert rows_for([late, early, longer], [ref], [ref_row(9, 20.0, 11)])[0][0]["primary"]["history"] == "mm_longer"
    twin = {**copy.deepcopy(early), "id": "aa_twin"}
    assert rows_for([early, twin], [ref], [ref_row(9, 20.0, 11)])[0][0]["primary"]["history"] == "aa_twin", "an exact tie goes to the smaller identifier"


def test_a_reference_that_does_not_reach_the_coast_has_a_primary_and_no_carriage():
    ref, lons = reference()
    rows, _ = rows_for([hist("h", [10.0] * 12, lons)], [ref], [ref_row(9, 20.0, 11, atlantic=False)])
    r = rows[0]
    assert (r["coast_reaching"], r["endpoint_carried"], r["sustained"], r["unmixed_sustained"]) == (False, False, False, False) and r["primary"]["count"] == 12
    none, _ = rows_for([hist("far", [35.0] * 12, lons)], [ref], [ref_row(9, 20.0, 11)])
    assert none[0]["primary"] is None and none[0]["histories"] == 0 and none[0]["endpoint_carried"] is False


def test_endpoint_carriers_must_agree_with_the_endpoint_class(monkeypatch):
    ref, lons = reference()
    real = E.reference_measures

    def wrong(*args, **kwargs):
        rows = real(*args, **kwargs)
        rows[0]["class"] = "dropped"
        return rows

    monkeypatch.setattr(E, "reference_measures", wrong)
    with pytest.raises(SystemExit, match="disagree with its class"):
        rows_for([hist("h", [10.0] * 12, lons)], [ref], [ref_row(9, 20.0, 11)])


# --- exact tests and the three parts ---------------------------------------------------------------

def test_the_exact_tests():
    assert P.sign_test(5, 0) == (1, 32) and P.sign_test(3, 1) == (5, 16) and P.sign_test(0, 0) == (1, 1) and P.sign_test(0, 3) == (8, 8)
    assert P.fisher(2, 2, 0, 2) == (1, 6) and P.fisher(0, 5, 3, 5) == (252, 252) and P.fisher(1, 2, 1, 2) == (5, 6)
    assert P.at_most_alpha((1, 20)) and not P.at_most_alpha((1, 19)) and P.at_most_alpha((1, 32)) and not P.at_most_alpha((1, 16))


def test_the_benefit_part_at_its_edges():
    assert P.benefit_outcome(5, 0, 27) == "pass", "p is 1 in 32"
    assert P.benefit_outcome(4, 0, 27) == "insufficient evidence", "p is 1 in 16"
    assert P.benefit_outcome(9, 1, 27) == "pass" and P.benefit_outcome(8, 2, 27) == "insufficient evidence", "11 and 56 in 1,024"
    assert P.benefit_outcome(5, 0, 50) == "pass" and P.benefit_outcome(5, 0, 51) == "insufficient evidence", "one in ten of the coast-reaching references"
    assert P.benefit_outcome(1, 1, 27) == "fail" and P.benefit_outcome(0, 0, 27) == "fail" and P.benefit_outcome(2, 3, 27) == "fail"
    assert P.benefit_outcome(2, 0, 27) == "insufficient evidence" and P.benefit_outcome(0, 0, 0) == "insufficient evidence"


def test_the_adverse_part_at_its_edges():
    assert P.adverse_outcome(0, 19, 0, 50) == "insufficient evidence" and P.adverse_outcome(0, 50, 0, 19) == "insufficient evidence"
    assert P.adverse_outcome(5, 100, 0, 50) == "pass", "one in twenty passes whatever the released share"
    assert P.adverse_outcome(6, 100, 0, 50) == "insufficient evidence" and P.adverse_outcome(8, 100, 0, 50) == "fail"
    assert P.adverse_outcome(10, 100, 4, 50) == "pass", "exactly 1.25 times 4 in 50"
    assert P.adverse_outcome(11, 100, 4, 50) == "insufficient evidence", "above the margin and within chance"
    assert P.adverse_outcome(30, 100, 4, 50) == "fail"
    assert P.adverse_outcome(1, 20, 0, 20) == "pass" and P.adverse_outcome(2, 20, 0, 20) == "insufficient evidence"


def counts(tracks_in_sector=100, surplus=20, stationary=10):
    return {"tracks": 4 * tracks_in_sector, "tracks_in_sector": tracks_in_sector, "units_in_sector": tracks_in_sector - surplus, "surplus_tracks_in_sector": surplus,
            "stationary_tracks": stationary, "duplicate_groups": surplus // 2, "never_accompanied_in_sector": 0, "cohort_tracks": 0, "cohort_tracks_atlantic_side": 0,
            "cohort_units": 0, "cohort_units_atlantic_side": 0}


def test_the_product_part_at_its_edges():
    r = counts()
    assert P.product_outcome(9, 10, counts(200, 50, 25), r) == ({"unique_carrier_unit": "pass", "surplus_tracks_in_sector": "pass", "stationary_tracks": "pass"}, "pass")
    assert P.product_outcome(5, 5, counts(200, 50, 25), r)[0]["unique_carrier_unit"] == "pass" and P.product_outcome(4, 4, counts(200, 50, 25), r)[0]["unique_carrier_unit"] == "insufficient evidence"
    assert P.product_outcome(8, 10, counts(200, 50, 25), r)[0]["unique_carrier_unit"] == "insufficient evidence", "8 of 10 is 56 in 1,024, within chance"
    assert P.product_outcome(5, 10, counts(200, 50, 25), r)[0]["unique_carrier_unit"] == "insufficient evidence"
    assert P.product_outcome(1, 10, counts(200, 50, 25), r)[0]["unique_carrier_unit"] == "fail" and P.product_outcome(2, 10, counts(200, 50, 25), r)[0]["unique_carrier_unit"] == "insufficient evidence"
    assert P.product_outcome(0, 0, counts(200, 50, 25), r) == ({"unique_carrier_unit": "insufficient evidence", "surplus_tracks_in_sector": "pass", "stationary_tracks": "pass"},
                                                               "insufficient evidence")
    assert P.product_outcome(9, 10, counts(200, 51, 25), r) == ({"unique_carrier_unit": "pass", "surplus_tracks_in_sector": "fail", "stationary_tracks": "pass"}, "fail")
    assert P.product_outcome(9, 10, counts(200, 50, 26), r)[0]["stationary_tracks"] == "fail"
    assert P.product_outcome(9, 10, counts(0, 0, 0), r)[1] == "insufficient evidence"
    assert P.worst(["pass", "insufficient evidence", "fail"]) == "fail" and P.worst(["pass", "insufficient evidence"]) == "insufficient evidence" and P.worst(["pass"]) == "pass"


# --- units, the pool and consistency ------------------------------------------------------------------

def rrow(i, coast=True, sustained=False, mixed=False, primary=True, units=1, endpoint=False, late=False, gap=False, relay=False, supported=False):
    clean = sustained and not mixed
    return {"id": i, "outcome": "atlantic_side" if coast else "none", "coast_reaching": coast, "steps": 10, "accompanied": 5 if primary else 0,
            "best_count": 5 if primary else 0, "histories": 1 if primary else 0, "endpoint_class": "carried" if endpoint else None, "endpoint_carried": endpoint,
            "endpoint_carriers": int(endpoint), "endpoint_carrier_units": int(endpoint), "sustained": sustained, "sustained_carriers": int(sustained),
            "unmixed_sustained": clean, "unmixed_sustained_carriers": int(clean), "unmixed_sustained_carrier_units": units if clean else 0,
            "clean_sustained": clean and not relay, "mixed_with_a_history": mixed, "open_with_a_history": relay, "supported_false_link_with_a_history": supported,
            "primary": None if not primary else {"history": f"h{i}", "count": 5, "first": 0, "last": 4, "pickup_hours": 0, "pickup_km": 0.0, "start": not late, "end": True,
                                                 "majority": True, "longest_gap_hours": 30 if gap else 6, "no_long_gap": not gap,
                                                 "status": "mixed" if mixed else ("open" if relay else "clean"), "supported_false_link": supported, "sustained": sustained}}


def season(year, n=26, gained=(), lost=(), mixed_c=(), mixed_r=(), unit_counts=None, supported_c=(), bare_c=()):
    """Twenty-six coast-reaching references. Those in `gained` have unmixed sustained carriage under
    the candidate only, those in `lost` under the released configuration only, and those in
    `bare_c` have no history beside them under the candidate."""
    rel = [rrow(i, sustained=i in lost, mixed=i in mixed_r) for i in range(n)]
    cand = [rrow(i, sustained=i in gained, mixed=i in mixed_c, units=(unit_counts or {}).get(i, 1), supported=i in supported_c, primary=i not in bare_c) for i in range(n)]
    return {"year": year, "configs": {"released": {"rows": rel, "counts": counts()}, P.CANDIDATE: {"rows": cand, "counts": counts(200, 40, 20)}}}


def test_a_unit_is_evaluated_from_its_rows_and_counts():
    s = season(1990, gained=(0, 1, 2, 3, 4, 5), mixed_c=(20,), mixed_r=(21,))
    e = P.evaluate(s["configs"]["released"], s["configs"][P.CANDIDATE])
    assert e["paired_unmixed_sustained"] == {"gained": 6, "lost": 0, "both": 0, "neither": 20} and e["benefit"] == "pass" and e["sign_test_p"] == 0.015625
    assert e["adverse"] == "pass" and e["candidate"]["primary_mixed"] == 1 and e["released"]["with_a_primary_history"] == 26
    assert e["adverse_checks"] == {"mixed_with_a_history": "pass", "supported_false_link_with_a_history": "pass"} and e["fisher_p"]["mixed_with_a_history"] == 0.754902
    assert e["product_checks"]["unique_carrier_unit"] == "pass" and e["product"] == "pass" and e["paired_clean_sustained"]["gained"] == 6
    four = season(1990, gained=(0, 1, 2, 3))
    assert P.evaluate(four["configs"]["released"], four["configs"][P.CANDIDATE])["product"] == "insufficient evidence", "four carried references, all with one unit, are within chance"


def test_the_adverse_part_reads_both_measures_and_needs_most_references_to_have_a_history():
    s = season(1990, supported_c=tuple(range(8)))
    e = P.evaluate(s["configs"]["released"], s["configs"][P.CANDIDATE])
    assert e["adverse_checks"] == {"mixed_with_a_history": "pass", "supported_false_link_with_a_history": "fail"} and e["adverse"] == "fail", "8 of 26 against none of 26"
    s = season(1990, mixed_c=tuple(range(8)))
    assert P.evaluate(s["configs"]["released"], s["configs"][P.CANDIDATE])["adverse_checks"]["mixed_with_a_history"] == "fail"
    s = season(1990, n=50, bare_c=tuple(range(26)))                  # 24 of 50 references have a history under the candidate
    e = P.evaluate(s["configs"]["released"], s["configs"][P.CANDIDATE])
    assert e["candidate"]["with_a_primary_history"] == 24 and e["adverse"] == "insufficient evidence", "a clean minority is not a pass"
    s = season(1990, n=50, bare_c=tuple(range(25)))
    assert P.evaluate(s["configs"]["released"], s["configs"][P.CANDIDATE])["adverse"] == "pass", "half the references is enough"
    s = season(1990, n=50, bare_c=tuple(range(26)), mixed_c=tuple(range(26, 36)))
    e = P.evaluate(s["configs"]["released"], s["configs"][P.CANDIDATE])
    assert e["candidate"]["mixed_with_a_history"] == 10 and e["adverse_checks"]["mixed_with_a_history"] == "fail", "few references with a history do not turn a fail into less"


def test_pairs_are_of_coast_reaching_references_and_the_product_part_reads_the_candidate_against_released():
    rel = [rrow(0), rrow(1, coast=False)]
    cand = [rrow(0, sustained=True), {**rrow(1, coast=False), "unmixed_sustained": True}]
    assert P.paired(rel, cand, "unmixed_sustained") == {"gained": 1, "lost": 0, "both": 0, "neither": 0}
    s = season(1990, gained=tuple(range(12)))
    s["configs"][P.CANDIDATE]["counts"] = counts(200, 60, 20)                               # 0.30 surplus tracks a track against 0.20 released
    e = P.evaluate(s["configs"]["released"], s["configs"][P.CANDIDATE])
    assert e["product_checks"] == {"unique_carrier_unit": "pass", "surplus_tracks_in_sector": "fail", "stationary_tracks": "pass"} and e["product"] == "fail"


def test_units_read_against_different_references_are_refused():
    s = season(1990)
    other = copy.deepcopy(s["configs"][P.CANDIDATE]); other["rows"][0]["id"] = 99
    with pytest.raises(SystemExit, match="different reference tracks"):
        P.evaluate(s["configs"]["released"], other)
    other = copy.deepcopy(s["configs"][P.CANDIDATE]); other["rows"][0]["steps"] = 11
    with pytest.raises(SystemExit, match="different reference observations"):
        P.evaluate(s["configs"]["released"], other)


def test_the_pool_passes_only_with_gains_spread_over_seasons_and_no_net_loss():
    gains = [(0, 1, 2), (0, 1, 2), (0, 1, 2), (0, 1), (0, 1, 2)]
    d = P.decide([season(y, gained=g) for y, g in zip(P.RESERVED, gains)])
    assert d["pool"]["paired_unmixed_sustained"]["gained"] == 14 and d["pool"]["candidate"]["coast_reaching"] == 130
    assert d["pool"]["benefit"] == "pass" and d["consistency"] == {"seasons_with_a_net_gain": 5, "seasons_with_a_net_loss": 0, "seasons_failing_the_adverse_part": 0}
    assert d["pool"]["product_checks"]["unique_carrier_unit"] == "pass" and d["pool"]["adverse"] == "pass" and d["outcome"] == "pass"
    assert [d["seasons"][str(y)]["benefit"] for y in P.RESERVED] == ["insufficient evidence"] * 5, "three gains in one season are within chance"
    two = P.decide([season(y, gained=g) for y, g in zip(P.RESERVED, [tuple(range(8)), tuple(range(7)), (), (), ()])])
    assert two["pool"]["paired_unmixed_sustained"]["gained"] == 15 and two["pool"]["benefit"] == "insufficient evidence" and two["consistency"]["seasons_with_a_net_gain"] == 2
    loss = P.decide([season(y, gained=g, lost=l) for y, g, l in zip(P.RESERVED, gains + [()], [(), (), (), (), (9,)])][:5])
    assert loss["consistency"]["seasons_with_a_net_loss"] == 0
    lossy = [season(y, gained=g) for y, g in zip(P.RESERVED, [(0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3), ()])]
    lossy[4] = season(P.RESERVED[4], lost=(9,))
    d = P.decide(lossy)
    assert d["pool"]["paired_unmixed_sustained"] == {"gained": 16, "lost": 1, "both": 0, "neither": 113}
    assert d["consistency"]["seasons_with_a_net_loss"] == 1 and d["pool"]["benefit"] == "insufficient evidence" and d["outcome"] == "insufficient evidence"


def test_a_season_failing_the_adverse_part_holds_the_pool_back():
    seasons = [season(y, gained=(0, 1, 2)) for y in P.RESERVED]
    seasons[1] = season(P.RESERVED[1], gained=(0, 1, 2), mixed_c=tuple(range(3, 13)))
    d = P.decide(seasons)
    assert d["seasons"][str(P.RESERVED[1])]["adverse"] == "fail" and d["consistency"]["seasons_failing_the_adverse_part"] == 1
    assert d["pool"]["candidate"]["primary_mixed"] == 10 and d["pool"]["adverse"] in ("fail", "insufficient evidence")
    spread = [season(y, gained=(0, 1, 2), mixed_c=(5,)) for y in P.RESERVED]
    assert P.decide(spread)["pool"]["adverse"] == "pass", "5 mixed primaries of 130 is under one in twenty"
    one = [season(y, gained=(0, 1, 2)) for y in P.RESERVED]
    one[1] = season(P.RESERVED[1], gained=(0, 1, 2), mixed_c=tuple(range(3, 9)))
    d = P.decide(one)
    assert d["seasons"][str(P.RESERVED[1])]["adverse"] == "fail", "6 of 26 against none of 26"
    assert d["pool"]["candidate"]["primary_mixed"] == 6 and P.adverse_outcome(6, 130, 0, 130) == "pass", "6 of 130 is under one in twenty by itself"
    assert d["pool"]["adverse"] == "insufficient evidence", "the failing season holds the pool back"
    assert (d["pool"]["benefit"], d["pool"]["product"], d["outcome"]) == ("pass", "pass", "insufficient evidence")


def test_many_carrier_units_fail_the_product_part():
    seasons = [season(y, gained=(0, 1, 2), unit_counts={0: 3, 1: 2, 2: 2}) for y in P.RESERVED]
    d = P.decide(seasons)
    assert d["pool"]["candidate"]["unmixed_sustained"] == 15 and d["pool"]["candidate"]["unmixed_sustained_with_one_carrier_unit"] == 0
    assert d["pool"]["product_checks"]["unique_carrier_unit"] == "fail" and d["outcome"] == "fail"
    assert d["pool"]["candidate"]["most_unmixed_sustained_carrier_units_of_one_reference"] == 3
    some = P.decide([season(y, gained=(0, 1, 2), unit_counts={0: 3, 1: 2}) for y in P.RESERVED])
    assert some["pool"]["candidate"]["unmixed_sustained_with_one_carrier_unit"] == 5 and some["pool"]["product_checks"]["unique_carrier_unit"] == "insufficient evidence"
    assert some["outcome"] == "insufficient evidence", "10 of 15 with several units is within chance"


def test_the_tally_counts_late_pickups_gaps_and_references_without_a_history():
    rows = [rrow(0, late=True), rrow(1, gap=True), rrow(2, primary=False), rrow(3, coast=False, mixed=True), rrow(4, sustained=True, endpoint=True),
            rrow(5, sustained=True, relay=True), rrow(6, endpoint=True, supported=True)]
    t = P.tally(rows)
    assert (t["references"], t["coast_reaching"], t["with_a_primary_history"], t["without_any_history"]) == (7, 6, 6, 1)
    assert (t["primary_late_pickup"], t["primary_long_gap"], t["primary_mixed"], t["primary_clean"], t["primary_open"]) == (1, 1, 1, 4, 1)
    assert (t["endpoint_carried"], t["endpoint_carried_and_sustained"], t["sustained"], t["unmixed_sustained"], t["clean_sustained"]) == (2, 1, 2, 2, 1)
    assert (t["mixed_with_a_history"], t["open_with_a_history"], t["supported_false_link_with_a_history"], t["observations"], t["accompanied"]) == (1, 1, 1, 70, 30)


# --- the freeze ---------------------------------------------------------------------------------------

def dev_record(tmp_path, year, sha=None, modules=("scripts/continuity_census.py", "src/aew/v1port/association.py")):
    with gzip.open(tmp_path / f"{P.STEM['development']}_{year}.json.gz", "wt", encoding="utf-8") as fh:
        json.dump({"generated_by": {"sha256": sha or P.SCRIPT_SHA256_AT_LOAD}, "modules": list(modules)}, fh)


def test_the_freeze_records_every_loaded_file_and_is_written_once(tmp_path):
    dev_record(tmp_path, 1990); dev_record(tmp_path, 2006, modules=("scripts/continuity_census.py", "scripts/continuity_lifecycle.py"))
    path = str(tmp_path / "freeze.json")
    rec = P.write_freeze(str(tmp_path), path)
    assert sorted(rec["files"]) == ["scripts/continuity_census.py", "scripts/continuity_lifecycle.py", "scripts/continuity_placement_check.py", "src/aew/v1port/association.py"]
    assert rec["files"]["scripts/continuity_placement_check.py"] == P.SCRIPT_SHA256_AT_LOAD and rec["rule"] == P.rule() and sorted(rec["development_records"]) == ["1990", "2006"]
    with pytest.raises(SystemExit, match="written once"):
        P.write_freeze(str(tmp_path), path)
    sha, fr = P.require_freeze(path, is_committed=lambda paths: True)
    assert sha == E.C.sha256_of(path) and fr == rec


def test_a_freeze_needs_development_records_of_this_script(tmp_path):
    dev_record(tmp_path, 1990); dev_record(tmp_path, 2006, sha="0" * 64)
    with pytest.raises(SystemExit, match="2006 development record was not written by the current script"):
        P.write_freeze(str(tmp_path), str(tmp_path / "freeze.json"))


def test_no_reserved_season_without_a_matching_committed_freeze(tmp_path):
    path = str(tmp_path / "freeze.json")
    with pytest.raises(SystemExit, match="no freeze record"):
        P.require_freeze(path)
    dev_record(tmp_path, 1990); dev_record(tmp_path, 2006)
    P.write_freeze(str(tmp_path), path)
    seen = []
    with pytest.raises(SystemExit, match="not committed"):
        P.require_freeze(path, is_committed=lambda paths: seen.append(paths) or False)
    assert "scripts/continuity_placement_check.py" in seen[0] and any(p.endswith("freeze.json") for p in seen[0]), "the record itself and every frozen file"
    fr = json.load(open(path))
    changed = copy.deepcopy(fr); changed["files"]["scripts/continuity_census.py"] = "0" * 64
    json.dump(changed, open(path, "w"))
    with pytest.raises(SystemExit, match="files changed since the freeze: scripts/continuity_census.py"):
        P.require_freeze(path, is_committed=lambda paths: True)
    gone = copy.deepcopy(fr); gone["files"]["scripts/no_such_file.py"] = "0" * 64
    json.dump(gone, open(path, "w"))
    with pytest.raises(SystemExit, match="no_such_file.py"):
        P.require_freeze(path, is_committed=lambda paths: True)
    other = copy.deepcopy(fr); other["rule"]["gap_hours"] = 48
    json.dump(other, open(path, "w"))
    with pytest.raises(SystemExit, match="frozen rule is not the rule in this script"):
        P.require_freeze(path, is_committed=lambda paths: True)


def test_committed_means_tracked_and_unchanged(tmp_path, monkeypatch):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q"); git("config", "user.email", "t@example.org"); git("config", "user.name", "t")
    (tmp_path / "a.txt").write_text("a"); (tmp_path / "b.txt").write_text("b")
    git("add", "a.txt"); git("commit", "-q", "-m", "a")
    monkeypatch.setattr(P, "ROOT", str(tmp_path))
    assert P.committed(["a.txt"]) is True and P.committed(["a.txt", "b.txt"]) is False, "an untracked file is not committed"
    (tmp_path / "a.txt").write_text("changed")
    assert P.committed(["a.txt"]) is False
    git("add", "a.txt")
    assert P.committed(["a.txt"]) is False, "staged is not committed"
    git("commit", "-q", "-m", "a2")
    assert P.committed(["a.txt"]) is True
    (tmp_path / ".gitignore").write_text("c.txt\n"); (tmp_path / "c.txt").write_text("c")
    git("add", ".gitignore"); git("commit", "-q", "-m", "ignore")
    assert P.committed(["c.txt"]) is False, "an ignored file shows no change and is still not tracked"


def test_a_source_file_outside_the_freeze_is_refused():
    loaded = P.repo_modules()
    assert "scripts/continuity_placement_check.py" in loaded and "scripts/continuity_evaluation.py" in loaded and all(not f.startswith("tests") for f in loaded)
    P.check_modules({"files": {f: "x" for f in loaded}})
    with pytest.raises(SystemExit, match="outside the freeze were loaded: scripts/continuity_evaluation.py"):
        P.check_modules({"files": {f: "x" for f in loaded if f != "scripts/continuity_evaluation.py"}})


def test_a_reserved_season_is_refused_before_anything_is_loaded(tmp_path):
    def loader(year):
        raise AssertionError("a season was loaded")

    with pytest.raises(SystemExit, match="no freeze record"):
        P.run_season(1984, "reserved", loader=loader, freeze_path=str(tmp_path / "none.json"), artifact_dir=str(tmp_path))
    with pytest.raises(SystemExit, match="1984 is not a development season"):
        P.run_season(1984, "development", loader=loader)
    with pytest.raises(SystemExit, match="1990 is not a reserved season"):
        P.run_season(1990, "reserved", loader=loader)
    with pytest.raises(SystemExit, match="2005 is not a reserved season"):
        P.run_season(2005, "reserved", loader=loader)


# --- one season through the runner ---------------------------------------------------------------------

def runner(tmp_path, monkeypatch, gate=True, coast_rows=None, tracks_sha=None, groups=None):
    import coast_crossing_measurement as CC
    import pilot_margin_replay as MR
    import qtrack_pairing_pilot as M
    import season_metrics as SM
    f1, f2 = tmp_path / "ours.mat", tmp_path / "q.nc"
    f1.write_text("ours"); f2.write_text("q")
    ref, lons = reference()
    art = {"inputs": {"this_record": {"path": str(f1), "sha256": E.C.sha256_of(str(f1))}, "qtrack": {"path": str(f2), "sha256": E.C.sha256_of(str(f2))},
                      "region_polygons": {"dir": "x"}},
           "sides": {"qtrack": {"tracks": [ref_row(9, 20.0, 11)]}, "this_record": {"tracks": coast_rows or [{"id": 0, "outcome": "atlantic_side"}]}}}
    ad = tmp_path / "art"; ad.mkdir()
    for y in (1990, 1984):
        (ad / E.C.ARTIFACT.format(year=y)).write_text(json.dumps(art))
    monkeypatch.setattr(MR, "archive_gate", lambda finished, run: {"passed": gate})
    monkeypatch.setattr(SM, "load_regions", lambda d: ([], {}))
    monkeypatch.setattr(M, "load_qtrack", lambda path: ([ref], {}))
    monkeypatch.setattr(CC, "groups_ours", lambda tracks: dict(groups or {}))
    monkeypatch.setattr(CC, "measure", lambda tracks, *a: {"tracks": [side_row(t["id"], float(t["lon"][0]), "atlantic_side") for t in tracks]})
    raw = {"birth": "0:0", "fate": "finished", "finished_index": 0, "step": list(range(12)), "time": [D0 + 0.25 * k for k in range(12)], "lat": [10.0] * 12,
           "lon": lons, "est": [[10.0, x, 6] for x in lons[1:]]}
    pruned = {"birth": "3:1", "fate": "pruned_in_loop", "step": [3], "time": [D0 + 0.75], "lat": [10.0], "lon": [5.0], "est": []}
    stored = L.finalized([raw])[0]
    seen = []

    def loop(case, grids, ct, ft, flags, options):
        seen.append(dict(options))
        return [{"time": stored["time"], "lat": stored["lat"], "lon": stored["lon"]}], [copy.deepcopy(raw), copy.deepcopy(pruned)], {"pruned_with_both_attempts_spent": 1}

    def loader(year):
        return {"tracks_sha256": tracks_sha or E.C.sha256_of(str(f1))}, {"sha256": "c"}, 1.0, 1.0, {"absorb": False, "exclusive": False}, {}

    return dict(loader=loader, loop=loop, artifact_dir=str(ad)), seen


def test_a_development_season_runs_both_configurations_and_is_bound_to_the_census(tmp_path, monkeypatch):
    kw, seen = runner(tmp_path, monkeypatch)
    called = []
    monkeypatch.setattr(P, "check_development", lambda year, name, base, flags, costs: called.append((year, name, E.pool_reference(base)["accompanied"])))
    out = P.run_season(1990, "development", **kw)
    assert seen == [{}, {"coarse_clip_radius_deg": 5.0}] and called == [(1990, "released", 12), (1990, "VCLIP", 12)]
    c = out["configs"]["VCLIP"]
    assert out["mode"] == "development" and out["freeze_sha256"] is None and "scripts/continuity_placement_check.py" in out["modules"]
    assert c["rows"][0]["unmixed_sustained"] is True and c["counts"]["tracks"] == 1 and c["counts"]["cohort_units_atlantic_side"] == 1 and c["fates"]["pruned_in_loop"] == 1
    assert [h["birth"] for h in c["histories"]] == ["0:0"] and c["histories"][0]["unit"] is None, "the finished history beside the reference is retained"
    e = P.evaluate(out["configs"]["released"], c)
    assert e["paired_unmixed_sustained"] == {"gained": 0, "lost": 0, "both": 1, "neither": 0} and e["benefit"] == "fail"


def test_a_reserved_season_runs_only_under_the_freeze_and_checks_what_was_loaded(tmp_path, monkeypatch):
    kw, seen = runner(tmp_path, monkeypatch, groups={0: 4})
    monkeypatch.setattr(P, "check_development", lambda *a: (_ for _ in ()).throw(AssertionError("a reserved season has no census to match")))
    checked = []
    monkeypatch.setattr(P, "require_freeze", lambda path: ("f" * 64, {"files": {"all": "x"}}))
    monkeypatch.setattr(P, "check_modules", lambda frozen: checked.append(frozen))
    out = P.run_season(1984, "reserved", **kw)
    assert out["freeze_sha256"] == "f" * 64 and checked == [{"files": {"all": "x"}}] and out["configs"]["VCLIP"]["histories"][0]["unit"] == 4
    monkeypatch.setattr(P, "check_modules", lambda frozen: (_ for _ in ()).throw(SystemExit("REFUSED: source files outside the freeze were loaded")))
    with pytest.raises(SystemExit, match="outside the freeze"):
        P.run_season(1984, "reserved", **kw)


def test_the_runner_refuses_inputs_and_loops_that_do_not_bind(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "check_development", lambda *a: None)
    kw, _ = runner(tmp_path, monkeypatch, gate=False)
    with pytest.raises(SystemExit, match="does not reproduce the stored campaign tracks"):
        P.run_season(1990, "development", **kw)
    second = tmp_path / "second"; second.mkdir()
    kw, _ = runner(second, monkeypatch, coast_rows=[{"id": 0, "outcome": "none"}])
    with pytest.raises(SystemExit, match="coast-crossing rows differ from the artifact's"):
        P.run_season(1990, "development", **kw)
    third = tmp_path / "third"; third.mkdir()
    kw, _ = runner(third, monkeypatch, tracks_sha="0" * 64)
    with pytest.raises(SystemExit, match="do not name one set of tracks"):
        P.run_season(1990, "development", **kw)
    fourth = tmp_path / "fourth"; fourth.mkdir()
    kw, _ = runner(fourth, monkeypatch)
    (fourth / "q.nc").write_text("another file")
    with pytest.raises(SystemExit, match="is not the file the 1990 artifact names"):
        P.run_season(1990, "development", **kw)
    fifth = tmp_path / "fifth"; fifth.mkdir()
    kw, _ = runner(fifth, monkeypatch)
    real = P.R.correspondence
    monkeypatch.setattr(P.R, "correspondence", lambda *a: {**real(*a), "accompanied_steps": 11})
    with pytest.raises(SystemExit, match="and the correspondence of section 14 disagree"):
        P.run_season(1990, "development", **kw)
    monkeypatch.setattr(P.R, "correspondence", real)
    sixth = tmp_path / "sixth"; sixth.mkdir()
    kw, _ = runner(sixth, monkeypatch)
    monkeypatch.setattr(L, "finalized", lambda rows: [])
    with pytest.raises(SystemExit, match="does not pass the final filter a second time"):
        P.run_season(1990, "development", **kw)


# --- development binding, output counts and the records ---------------------------------------------------

def test_a_development_season_must_reproduce_the_lifecycle_census(tmp_path):
    ref, lons = reference()
    hs = [hist("h", [10.0] * 12, lons)]
    qs = [ref]
    base = E.reference_measures(hs, {9: ref}, [ref_row(9, 20.0, 11)])
    flags, costs = E.flag_measures(hs, E.C.index_by_time(qs), E.J.spans(qs), base)
    life = {"generated_by": {"sha256": L.SCRIPT_SHA256_AT_LOAD}, "configs": {"VCLIP": {"census": {"STORED": {"reference": E.pool_reference(base), "flags": flags, "costs": costs}}}}}

    def write(rec):
        with gzip.open(tmp_path / "continuity_lifecycle_1990.json.gz", "wt", encoding="utf-8") as fh:
            json.dump(rec, fh)

    write(life)
    P.check_development(1990, "VCLIP", base, flags, costs, artifacts=str(tmp_path))
    for key, field in (("reference", "accompanied"), ("flags", "exposure_pairs"), ("costs", "links")):
        bad = copy.deepcopy(life); bad["configs"]["VCLIP"]["census"]["STORED"][key][field] += 1
        write(bad)
        with pytest.raises(SystemExit, match="does not reproduce the lifecycle census"):
            P.check_development(1990, "VCLIP", base, flags, costs, artifacts=str(tmp_path))
    bad = copy.deepcopy(life); bad["generated_by"]["sha256"] = "0" * 64
    write(bad)
    with pytest.raises(SystemExit, match="not written by the current lifecycle script"):
        P.check_development(1990, "VCLIP", base, flags, costs, artifacts=str(tmp_path))


def side_row(i, lon, outcome):
    return {"id": i, "first": {"lon": lon, "lat": 10.0, "time": f"1990-07-0{i + 1}"}, "band": "5-10N", "outcome": outcome, "follow_up_incomplete": False,
            "single_observation_entrant": False}


def test_output_counts_by_track_and_by_unit():
    ref, lons = reference()
    stored = [hist("a", [10.0] * 12, lons), hist("b", [10.0] * 12, lons), hist("far", [30.0] * 13, [5.0] * 13), hist("out", [10.0] * 4, [100.0, 98.0, 96.0, 94.0])]
    tracks = [{"id": k, "time": h["time"], "lat": h["lat"], "lon": h["lon"], "n_valid": len(h["step"])} for k, h in enumerate(stored)]
    side = {"tracks": [side_row(0, 20.0, "atlantic_side"), side_row(1, 20.0, "atlantic_side"), side_row(2, 5.0, "none")]}
    got = P.output_counts(tracks, stored, {0: 0, 1: 0}, side, E.C.index_by_time([ref]))
    assert got == {"tracks": 4, "tracks_in_sector": 3, "units_in_sector": 2, "surplus_tracks_in_sector": 1, "stationary_tracks": 1, "duplicate_groups": 1,
                   "never_accompanied_in_sector": 1, "cohort_tracks": 2, "cohort_tracks_atlantic_side": 2, "cohort_units": 1, "cohort_units_atlantic_side": 1}
    apart = P.output_counts(tracks, stored, {0: 0, 3: 0}, side, E.C.index_by_time([ref]))
    assert (apart["units_in_sector"], apart["surplus_tracks_in_sector"], apart["duplicate_groups"]) == (3, 0, 1), "a group with one member in the sector is one unit there"
    with pytest.raises(SystemExit, match="disagree on the tracks in the sector"):
        P.output_counts(tracks, stored[:2], {0: 0, 1: 0}, side, E.C.index_by_time([ref]))


def season_file(tmp_path, year, mode, gained=(0, 1, 2), freeze_sha=None, stamp=None):
    s = season(year, gained=gained)
    flags = {"flagged_tracks": 1, "events": {"switch": 1, "relay": 2, "undetermined": 0, "addition": 0}, "exposure_pairs": 50, "systems_left_behind": 1,
             "reference_tracks_with_a_flagged_history": 1, "supported_false_link_candidates": 0}
    for c in s["configs"].values():
        c.update({"flags": flags, "links": {"links": 100, "jump_links": 10}, "fates": {k: 1 for k in L.FATES}})
    g = {"script": "x", "sha256": P.SCRIPT_SHA256_AT_LOAD, "lifecycle_sha256": L.SCRIPT_SHA256_AT_LOAD, "evaluation_sha256": E.SCRIPT_SHA256_AT_LOAD, "mode": mode, "rule": P.rule()}
    g.update(stamp or {})
    with gzip.open(tmp_path / f"{P.STEM[mode]}_{year}.json.gz", "wt", encoding="utf-8") as fh:
        json.dump({"generated_by": g, "mode": mode, "freeze_sha256": freeze_sha, **s}, fh)


def test_combine_reports_each_season_and_the_pool_and_never_overwrites(tmp_path):
    for y in P.DEVELOPMENT:
        season_file(tmp_path, y, "development", gained=(0, 1, 2, 3))
    assert P.main(["combine", "--out-dir", str(tmp_path)]) == 0
    rec = json.load(open(tmp_path / "continuity_placement_development.json", encoding="utf-8"))
    d = rec["decision"]
    assert sorted(d["seasons"]) == ["1990", "2006"] and d["pool"]["paired_unmixed_sustained"]["gained"] == 8 and d["pool"]["benefit"] == "pass"
    assert d["consistency"]["seasons_with_a_net_gain"] == 2 and d["outcome"] == "pass", "with two seasons, both must show a gain"
    text = (tmp_path / "continuity_placement_development.md").read_text()
    assert text.startswith("# Prospective evaluation of the unchanged VCLIP placement, development seasons (section 25)")
    assert "| pool | 52 | 8 | 0 | 0 | 44 | 0.003906 | 8 | 0 | 0 | 0 | 52 | 0 | 0 | 52 | 0 | 0 | 1.0 | pass | pass | pass |" in text
    assert "| 1990, VCLIP | 130 of 260 | 130 | 0 | 0 | 4 | 4 | 4 | 4 | 26 of 26 | 26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |" in text
    assert "- 1990:0, 10, 5, 0.0 km and 0 h, 6 h, clean; 0 in 0; 1 in 1; clean sustained; released primary 5" in text and "- 1990:9, 10, 5, 0.0 km and 0 h, 6 h, clean; 0 in 0; 0 in 0; not sustained; released primary 5" in text
    assert "| pool, VCLIP | 1600 | 400 | 320 | 80 | 0 | 40 | 40 |" in text and "Outcome: pass." in text and "| 2006, released | 1 | 1 | 2 | 0 | 50 | 1 | 1 | 0 | 100 | 10 |" in text
    assert "Adverse checks on the pool: mixed with a history pass, supported false link with a history pass." in text
    assert "Product checks on the pool: unique carrier unit pass, surplus tracks in sector pass, stationary tracks pass." in text
    with pytest.raises(SystemExit, match="never overwritten"):
        P.main(["combine", "--out-dir", str(tmp_path)])


@pytest.mark.parametrize("stamp", [{"sha256": "0" * 64}, {"evaluation_sha256": "0" * 64}, {"lifecycle_sha256": "0" * 64}, {"mode": "reserved"},
                                   {"rule": {**P.rule(), "gap_hours": 48}}])
def test_combine_refuses_season_records_of_other_scripts_rules_or_modes(tmp_path, stamp):
    season_file(tmp_path, 1990, "development"); season_file(tmp_path, 2006, "development", stamp=stamp)
    with pytest.raises(SystemExit, match="2006 season record was not written by the current scripts under this rule and mode"):
        P.main(["combine", "--out-dir", str(tmp_path)])


def test_reserved_records_are_combined_only_under_the_freeze_they_were_read_with(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "require_freeze", lambda *a, **k: ("f" * 64, {"files": {}}))
    for y in P.RESERVED:
        season_file(tmp_path, y, "reserved", freeze_sha="f" * 64)
    assert P.main(["combine", "--out-dir", str(tmp_path), "--reserved"]) == 0
    rec = json.load(open(tmp_path / "continuity_placement_reserved.json", encoding="utf-8"))
    assert rec["generated_by"]["freeze_sha256"] == "f" * 64 and sorted(rec["decision"]["seasons"]) == [str(y) for y in P.RESERVED] and rec["decision"]["outcome"] == "pass"
    os.remove(tmp_path / "continuity_placement_reserved.json"); os.remove(tmp_path / "continuity_placement_reserved.md")
    os.remove(tmp_path / f"{P.STEM['reserved']}_1997.json.gz")
    season_file(tmp_path, 1997, "reserved", freeze_sha="e" * 64)
    with pytest.raises(SystemExit, match="1997 season record was read under another freeze"):
        P.main(["combine", "--out-dir", str(tmp_path), "--reserved"])


def test_a_season_is_read_once(tmp_path):
    (tmp_path / "continuity_placement_reserved_1984.json.gz").write_text("x")
    with pytest.raises(SystemExit, match="read once and never overwritten"):
        P.main(["season", "--year", "1984", "--out-dir", str(tmp_path), "--reserved"])
    (tmp_path / "continuity_placement_development_1990.json.gz").write_text("x")
    with pytest.raises(SystemExit, match="read once and never overwritten"):
        P.main(["season", "--year", "1990", "--out-dir", str(tmp_path)])
