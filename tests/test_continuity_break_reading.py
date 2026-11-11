"""Tests for scripts/continuity_break_reading.py (`CONTINUITY_ACROSS_AFRICA_2026-11-09.md`, section 27).

The constructed cases section 27 names come first: a history beside a reference throughout, one
that ends, one that continues away, a 12-hour link that returns, a 12-hour link over the last
observation, copies of one history, and a stored position moved past 500 km by smoothing alone."""

import gzip
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import continuity_break_reading as B  # noqa: E402

C, E, L = B.C, B.E, B.L
D0 = 7000.0
STEP = 0.25
FAR = 35.0                                                            # a latitude about 2,780 km north of the reference


def ref(n=12, tid=9, extra=0, atlantic=True):
    """A reference moving 2 degrees west a step along 10 N from 20 E, read over `n` observations."""
    m = n + extra
    q = {"id": tid, "time": D0 + STEP * np.arange(m), "lat": np.full(m, 10.0), "lon": 20.0 - 2.0 * np.arange(m), "n_valid": m}
    row = {"id": tid, "first": {"lon": 20.0}, "outcome": "atlantic_side" if atlantic else "none", "last_used": {"index": n - 1}}
    if atlantic:
        row["first_atlantic_side"] = {"index": n - 1}
    return q, row


def hist(hid, steps, lats, lons=None, raw_lats=None, fate="finished"):
    steps = list(steps)
    lons = [20.0 - 2.0 * s for s in steps] if lons is None else lons
    raw = lats if raw_lats is None else raw_lats
    return {"id": hid, "time": D0 + STEP * np.asarray(steps, float), "lat": np.asarray(lats, float), "lon": np.asarray(lons, float), "step": steps,
            "raw_lat": [float(x) for x in raw], "raw_lon": [float(x) for x in lons], "est": [[float(lats[j]), float(lons[j]), 6] for j in range(1, len(steps))], "fate": fate}


def read(histories, n=12, extra=0, atlantic=True):
    q, row = ref(n, extra=extra, atlantic=atlantic)
    base = E.reference_measures(histories, {q["id"]: q}, [row])[0]
    return B.transitions(base, q, {h["id"]: h for h in histories}), base, q


def classes(ts):
    return [(t["k"], t["class"]) for t in ts]


# --- transitions -----------------------------------------------------------------------------------

def test_a_history_beside_the_reference_throughout_is_held_at_every_transition():
    ts, _, _ = read([hist("a", range(12), [10.0] * 12)])
    assert classes(ts) == [(k, "held") for k in range(11)], "the last observation read has no transition"


def test_a_history_that_ends_makes_one_break_and_no_transition_after_it():
    ts, _, _ = read([hist("a", range(5), [10.0] * 5)])
    assert classes(ts) == [(0, "held"), (1, "held"), (2, "held"), (3, "held"), (4, "break")]
    b = ts[-1]
    assert b["holders_did"] == "ended" and b["holders"]["a"] == {"step": 4, "outcome": "ended"} and "away_km" not in b
    assert b["main_holder"] == "a" and b["last_index"] == 11


def test_a_history_that_continues_away_makes_a_break_with_its_distance():
    ts, _, _ = read([hist("a", range(8), [10.0] * 5 + [FAR] * 3)])
    assert classes(ts) == [(0, "held"), (1, "held"), (2, "held"), (3, "held"), (4, "break")]
    b = ts[-1]
    assert b["holders_did"] == "continued_away" and b["marginal"] is False and b["by_smoothing_alone"] is False
    assert b["away_km"] == round(float(C.km(FAR, 10.0, 10.0, 10.0)), 1) and b["holders"]["a"]["next_step"] == 5


def test_a_12_hour_link_that_returns_beside_the_reference_is_held():
    ts, _, _ = read([hist("a", [0, 1, 3, 4], [10.0] * 4)], n=5)
    assert classes(ts) == [(0, "held"), (1, "held"), (3, "held")], "observation 2 is unaccompanied and has no transition"
    assert ts[1]["holders"]["a"] == {"step": 1, "to": 3, "next_step": 3, "outcome": "beside"}


def test_a_12_hour_link_over_the_last_observation_is_unresolved():
    ts, _, _ = read([hist("a", [0, 1, 2, 3, 5], [10.0] * 5)], n=5, extra=2)
    assert classes(ts)[-1] == (3, "unresolved") and ts[-1]["holders"]["a"]["outcome"] == "past_the_window"
    both, _, _ = read([hist("a", [0, 1, 2, 3, 5], [10.0] * 5), hist("b", [3, 4], [10.0] * 2)], n=5, extra=2)
    assert classes(both)[-1] == (3, "held"), "another holder beside settles it"
    ended, _, _ = read([hist("a", [0, 1, 2, 3, 5], [10.0] * 5), hist("b", [2, 3], [10.0] * 2)], n=5, extra=2)
    assert classes(ended)[-1] == (3, "unresolved"), "a holder that ended does not turn it into a break"


def test_copies_of_one_history_make_one_transition_each():
    one, _, _ = read([hist("a", range(5), [10.0] * 5)])
    two, _, _ = read([hist("a", range(5), [10.0] * 5), hist("b", range(5), [10.0] * 5)])
    assert classes(two) == classes(one) and sorted(two[-1]["holders"]) == ["a", "b"] and two[-1]["holders_did"] == "ended"


def test_a_break_needs_every_holder_to_leave():
    ts, _, _ = read([hist("a", range(5), [10.0] * 5), hist("b", range(8), [10.0] * 8)])
    assert classes(ts)[4] == (4, "held") and classes(ts)[-1] == (7, "break")
    mixed, _, _ = read([hist("a", range(5), [10.0] * 5), hist("b", range(8), [10.0] * 5 + [FAR] * 3)])
    assert classes(mixed)[-1] == (4, "break") and mixed[-1]["holders_did"] == "continued_away", "one ended and one went away"


def test_a_stored_position_moved_by_smoothing_alone_is_marked():
    ts, _, _ = read([hist("a", range(6), [10.0] * 5 + [16.0], raw_lats=[10.0] * 6)])
    b = ts[-1]
    assert b["class"] == "break" and b["by_smoothing_alone"] is True and b["marginal"] is True
    assert 500.0 < b["away_km"] <= 750.0


def test_the_radius_and_the_marginal_distance_include_their_bounds():
    q, _ = ref(6)
    h = hist("a", range(6), [10.0] * 5 + [14.0])
    d = float(C.km(14.0, 10.0, 10.0, 10.0))
    assert B.holder_step(h, q, 4, 6, radius_km=d)["outcome"] == "beside"
    away = B.holder_step(h, q, 4, 6, radius_km=d - 1e-6, marginal_km=d)
    assert away["outcome"] == "away" and away["marginal"] is True and away["by_smoothing_alone"] is False
    assert B.holder_step(h, q, 4, 6, radius_km=d - 1e-6, marginal_km=d - 1e-6)["marginal"] is False
    assert B.MARGINAL_KM == 750.0 and B.RADIUS_KM == 500.0
    smoothed = hist("s", range(6), [10.0] * 5 + [16.0], raw_lats=[10.0] * 5 + [14.0])
    assert B.holder_step(smoothed, q, 4, 6, radius_km=d)["by_smoothing_alone"] is True, "an unsmoothed position at the radius is within it"
    assert B.holder_step(smoothed, q, 4, 6, radius_km=d - 1e-6)["by_smoothing_alone"] is False


def test_a_break_with_two_away_holders_takes_the_nearer_and_either_mark():
    near = hist("a", range(6), [10.0] * 5 + [16.0])                                # about 667 km, marginal
    far = hist("b", range(6), [10.0] * 5 + [FAR], raw_lats=[10.0] * 6)             # far, and away by smoothing alone
    ts, _, _ = read([near, far])
    b = ts[-1]
    assert b["class"] == "break" and b["away_km"] == round(float(C.km(16.0, 10.0, 10.0, 10.0)), 1)
    assert b["marginal"] is True and b["by_smoothing_alone"] is True


def test_the_main_holder_is_beside_the_most_observations():
    hs = [hist("a", [3, 4], [10.0] * 2), hist("b", range(5), [10.0] * 5), hist("c", [1, 2, 3, 4], [10.0] * 4)]
    ts, _, _ = read(hs)
    assert ts[-1]["class"] == "break" and ts[-1]["main_holder"] == "b"
    tie, _, _ = read([hist("a", [3, 4], [10.0] * 2), hist("b", [3, 4], [10.0] * 2), hist("c", [2, 3, 4], [FAR, 10.0, 10.0])])
    assert tie[-1]["main_holder"] == "c", "equal counts go to the earlier first step"
    names, _, _ = read([hist("z", [3, 4], [10.0] * 2), hist("y", [3, 4], [10.0] * 2)])
    assert names[-1]["main_holder"] == "y", "and then to the lower identifier"


def test_the_segment_is_measured_to_the_first_atlantic_side_observation():
    ts, _, _ = read([hist("a", range(12), [10.0] * 12)], n=12, extra=6)
    assert [(t["k"], t["hours_to_coast"], t["segment"]) for t in ts][6:] == [(6, 30, "earlier"), (7, 24, "coastal"), (8, 18, "coastal"), (9, 12, "coastal"), (10, 6, "coastal")]
    assert B.segment(24) == "coastal" and B.segment(30) == "earlier" and B.COASTAL_HOURS == 24


def test_a_reference_ending_inside_africa_has_no_segment():
    ts, _, _ = read([hist("a", range(5), [10.0] * 5)], n=8, atlantic=False)
    assert {(t["segment"], t["hours_to_coast"]) for t in ts} == {(None, None)} and classes(ts)[-1] == (4, "break")
    refs = [{"id": 1, "coast_reaching": False, "transitions": ts}]
    assert B.per_reference(refs) == []


def test_longitude_bands_at_their_edges():
    assert [B.band(x) for x in (10.0, 9.99, 0.0, -0.01, -10.0, -10.01)] == ["east_of_10E", "0_to_10E", "0_to_10E", "10W_to_0", "10W_to_0", "west_of_10W"]
    ts, _, _ = read([hist("a", range(12), [10.0] * 12)])
    assert [t["band"] for t in ts][4:7] == ["east_of_10E", "east_of_10E", "0_to_10E"], "the band is the longitude of the observation left"


def test_a_reference_that_is_not_six_hourly_is_refused():
    q, _ = ref(6)
    q["time"] = q["time"].copy(); q["time"][3:] += 0.25
    with pytest.raises(SystemExit, match="six-hourly"):
        B.holder_step(hist("a", range(6), [10.0] * 6), q, 2, 6)


# --- the concentration reading -------------------------------------------------------------------------

def pr(coastal, earlier):
    return {"id": 0, "coastal": list(coastal), "earlier": list(earlier)}


def test_per_reference_counts_breaks_over_held_and_break():
    ts, _, _ = read([hist("a", range(9), [10.0] * 9), hist("b", [10, 11], [10.0] * 2)])
    rows = B.per_reference([{"id": 9, "coast_reaching": True, "transitions": ts}])
    assert rows == [{"id": 9, "coastal": [1, 3], "earlier": [0, 7]}], "the break at 8 is coastal, and the transition at 10 is held"


def test_breaks_confined_to_the_coastal_segment_are_concentrated_there():
    c = B.concentration([pr((1, 2), (0, 4))] * 5)
    assert c["outcome"] == "concentrated near the coast" and c["sign_test"] == [1, 32] and c["pooled"] == {"coastal": [5, 10], "earlier": [0, 20]}
    assert B.concentration([pr((0, 2), (1, 4))] * 5)["outcome"] == "concentrated earlier"


def test_equal_break_shares_are_not_shown():
    c = B.concentration([pr((1, 2), (2, 4))] * 6)
    assert (c["outcome"], c["equal_shares"], c["scored"], c["sign_test"]) == ("not shown", 6, 6, [1, 1])
    mixed = B.concentration([pr((1, 2), (0, 4))] * 3 + [pr((0, 2), (1, 4))] * 3)
    assert mixed["outcome"] == "not shown" and (mixed["higher_share_coastal"], mixed["higher_share_earlier"]) == (3, 3)


def test_fewer_than_five_scored_references_cannot_be_scored():
    assert B.concentration([pr((1, 2), (0, 4))] * 4)["outcome"] == "cannot be scored"
    one_sided = B.concentration([pr((1, 2), (0, 4))] * 4 + [pr((1, 2), (0, 0)), pr((0, 0), (1, 2))])
    assert one_sided["outcome"] == "cannot be scored" and one_sided["transitions_in_one_segment_only"] == 2 and one_sided["scored"] == 4
    assert B.concentration([pr((1, 2), (0, 4))] * 4 + [pr((1, 2), (2, 4))])["outcome"] == "not shown", "a tie is scored, and four of four is 1/16"
    assert B.MIN_SCORED == 5


def test_the_sign_test_must_reach_one_in_twenty():
    assert B.concentration([pr((1, 2), (0, 4))] * 5 + [pr((0, 2), (1, 4))])["outcome"] == "not shown", "five against one is 7/64"
    assert B.concentration([pr((1, 2), (0, 4))] * 7 + [pr((0, 2), (1, 4))])["outcome"] == "concentrated near the coast", "seven against one is 9/256"
    assert B.concentration([pr((0, 2), (1, 4))] * 5 + [pr((1, 2), (0, 4))])["outcome"] == "not shown"
    assert B.concentration([pr((0, 2), (1, 4))] * 7 + [pr((1, 2), (0, 4))])["outcome"] == "concentrated earlier"


def test_the_pooled_share_must_agree_with_the_references():
    rows = [pr((1, 1), (1, 2))] * 5 + [pr((0, 20), (0, 0))]
    c = B.concentration(rows)
    assert c["higher_share_coastal"] == 5 and c["pooled"] == {"coastal": [5, 25], "earlier": [5, 10]} and c["outcome"] == "not shown"
    assert B.concentration([pr((1, 2), (1, 1))] * 5 + [pr((0, 0), (0, 20))])["outcome"] == "not shown"


def test_shares_are_compared_by_reference_and_not_by_count():
    assert B.concentration([pr((1, 4), (2, 4))] * 5)["higher_share_earlier"] == 5
    assert B.concentration([pr((2, 3), (3, 6))] * 5)["higher_share_coastal"] == 5, "two of three is above three of six"


# --- where the primary history stops ----------------------------------------------------------------------

def prow(last, steps=12, **flags):
    p = {"history": "a", "last": last, "start": True, "end": True, "majority": True, "no_long_gap": True}
    p.update(flags)
    return {"steps": steps, "primary": p}


def test_the_stop_of_the_primary_at_its_edges():
    stops = [B.primary_stop(prow(k))["stop"] for k in (11, 10, 7, 6, 3, 2)]
    assert stops == ["at_the_first_atlantic_side_observation", "within_24_hours", "within_24_hours", "30_to_48_hours", "30_to_48_hours", "earlier"]
    got = B.primary_stop(prow(6, end=False))
    assert got == {"history": "a", "last": 6, "hours_before": 30, "stop": "30_to_48_hours", "start": True, "end": False, "majority": True, "no_long_gap": True}
    assert B.primary_stop({"steps": 12, "primary": None}) is None


def place_row(base, sustained=False, primary=None):
    return {"id": base["id"], "outcome": base["outcome"], "steps": base["steps"], "accompanied": base["accompanied"], "best_count": base["longest"],
            "sustained": sustained, "primary": primary}


def test_the_reference_rows_must_be_those_of_the_placement_record():
    hs = [hist("a", range(5), [10.0] * 5)]
    q, row = ref(12)
    base = E.reference_measures(hs, {9: q}, [row])
    by_id = {"a": hs[0]}
    good = place_row(base[0], primary=prow(4)["primary"])
    refs = B.reference_reading(base, [good], {9: q}, by_id)
    assert refs[0]["coast_reaching"] and refs[0]["primary"]["stop"] == "30_to_48_hours" and refs[0]["sustained"] is False and len(refs[0]["transitions"]) == 5
    for key, value in (("accompanied", 4), ("best_count", 3), ("steps", 11), ("outcome", "none"), ("id", 8)):
        with pytest.raises(SystemExit, match="placement record"):
            B.reference_reading(base, [{**good, key: value}], {9: q}, by_id)
    with pytest.raises(SystemExit, match="placement record"):
        B.reference_reading(base, [], {9: q}, by_id)


def test_a_reference_ending_inside_africa_has_no_primary_stop():
    hs = [hist("a", range(5), [10.0] * 5)]
    q, row = ref(8, atlantic=False)
    base = E.reference_measures(hs, {9: q}, [row])
    refs = B.reference_reading(base, [place_row(base[0], primary=prow(4)["primary"])], {9: q}, {"a": hs[0]})
    assert refs[0]["primary"] is None and refs[0]["coast_reaching"] is False


# --- what stands at the next observation -------------------------------------------------------------------

def wave(lat, lon, time=0.0, half=1.0, n=5, points=True):
    lats = np.repeat(np.linspace(-half, half, n), n) + lat
    lons = np.tile(np.linspace(-half, half, n), n) + lon
    if not points:
        lats, lons = np.array([]), np.array([])
    return {"time": time, "lat_mean": float(lat), "lon_mean": float(lon), "lat_wave": lats, "lon_wave": lons, "region": np.zeros((1, 1), bool)}


def test_a_handoff_is_fresh_when_the_receiver_begins_at_that_step():
    pos = {"class": "tracked"}
    assert B.next_state(pos, [], {}, {"id": "n", "step": [7, 8]}, 7, 10.0, 0.0) == {"next": "handed_off_fresh", "receiver": "n"}
    assert B.next_state(pos, [], {}, {"id": "o", "step": [6, 7]}, 7, 10.0, 0.0) == {"next": "handed_off_running", "receiver": "o"}


def test_a_candidate_not_stored_is_classed_by_the_histories_holding_it():
    waves = [wave(10.0, 3.0), wave(10.0, 0.0)]
    pos = {"class": "candidate_not_tracked"}

    def got(fates, at=(10.0, 0.0)):
        return B.next_state(pos, waves, {at: [{"fate": f} for f in fates]}, None, 7, 10.0, 0.5)

    assert got(["pruned_in_loop", "finished"]) == {"next": "candidate_not_stored", "holder_fates": ["finished", "pruned_in_loop"], "not_stored": "held_by_a_finished_history"}
    assert got(["pruned_in_loop"])["not_stored"] == "held_by_pruned_histories_only"
    assert got(["removed_by_final_speed_filter"])["not_stored"] == "held_by_a_history_the_final_filter_removed"
    assert got(["removed_as_single_observation", "pruned_in_loop"])["not_stored"] == "held_by_pruned_histories_only"
    assert got([])["not_stored"] == "not_held"
    assert got(["finished"], at=(10.0, 3.0))["not_stored"] == "not_held", "the holders are those of the candidate nearest the reference"
    with pytest.raises(SystemExit, match="no final candidate within the radius"):
        B.next_state(pos, [wave(30.0, 0.0)], {}, None, 7, 10.0, 0.5)


def test_the_other_stage_classes_pass_through():
    for c in ("placed_away", "signal_without_axis", "masked_by_westerlies", "below_threshold"):
        assert B.next_state({"class": c}, [], {}, None, 7, 10.0, 0.0) == {"next": c}
    assert set(B.NEXT) == {"handed_off_fresh", "handed_off_running", "candidate_not_stored", "placed_away", "signal_without_axis", "masked_by_westerlies", "below_threshold"}


def test_the_recomputed_candidates_must_be_the_positions_the_histories_hold():
    waves = [wave(10.0, 0.0), wave(12.0, 5.0)]
    held = {(10.0, 0.0): [1], (12.0, 5.0): [2]}
    B.check_finals(waves, held, 3, "x")
    with pytest.raises(SystemExit, match="is no final candidate"):
        B.check_finals(waves, {**held, (11.0, 1.0): [3]}, 3, "x")
    with pytest.raises(SystemExit, match="held by no retained history"):
        B.check_finals(waves, {(10.0, 0.0): [1]}, 3, "x")
    B.check_finals(waves + [wave(20.0, 9.0, points=False)], held, 3, "x")       # a candidate without a trough point is never seeded
    B.check_finals(waves + [wave(10.0, 90.0)], held, 3, "x")                    # one outside the retained box is not in the record
    with pytest.raises(SystemExit, match="held by no retained history"):
        B.check_finals(waves + [wave(10.0, 45.0)], held, 3, "x")                # 45 E is inside the box with its margin


# --- the association attempts ----------------------------------------------------------------------------------

def brk(k, step, next_step=None, last_index=11):
    h = {"step": step, "outcome": "ended"}
    if next_step is not None:
        h = {"step": step, "to": k + next_step - step, "next_step": next_step, "outcome": "away"}
    return {"k": k, "main_holder": "a", "last_index": last_index, "holders": {"a": h}}


def test_the_attempts_read_for_a_break():
    assert B.attempts_of(brk(4, 40)) == [("6hr", 41, 5), ("12hr", 42, 6)]
    assert B.attempts_of(brk(4, 40, next_step=41)) == [("6hr", 41, 5)], "a holder that linked at 6 hours has no 12-hour attempt"
    assert B.attempts_of(brk(4, 40, next_step=42)) == [("6hr", 41, 5), ("12hr", 42, 6)], "one that linked at 12 hours failed at 6"
    assert B.attempts_of(brk(10, 40)) == [("6hr", 41, 11)], "the observation 12 hours on lies past the window"
    assert B.attempts_of(brk(9, 40)) == [("6hr", 41, 10), ("12hr", 42, 11)]


SQUARE = [[-1.0, 9.0], [1.0, 9.0], [1.0, 11.0], [-1.0, 11.0], [-1.0, 9.0]]              # lon, lat, closed, area 4


def match(inside=(), chosen=None, speeds=None, pss="6hr", poly=SQUARE):
    return {"birth": "a", "pass": pss, "candidates_inside_polygon": list(inside), "implied_speed_ms": speeds or {str(i): 8.0 for i in inside},
            "distance_to_prediction_nm": {}, "polygon_as_read": poly, "chosen": chosen}


LIVE = {"est_6hr": [10.0, 0.0], "est_12hr": [10.0, -1.0]}


def test_no_candidate_within_500_km_of_the_reference():
    waves = [wave(10.0, 0.0), wave(10.0, -16.0)]                                 # the one not taken is the nearer, about 657 km away
    got = B.attempt_read(match(inside=[0], chosen=0), LIVE, waves, "6hr", 10.0, -10.0)
    assert got == {"pass": "6hr", "chosen": [10.0, 0.0], "candidates_inside": 1, "chosen_km_from_reference": round(float(C.km(10.0, -10.0, 10.0, 0.0)), 1), "class": "no_candidate_near"}
    assert B.attempt_read(match(), LIVE, [], "6hr", 10.0, 0.0)["class"] == "no_candidate_near"
    d = float(C.km(10.0, -10.0, 10.0, -6.0))
    assert B.attempt_read(match(), LIVE, [wave(10.0, -6.0)], "6hr", 10.0, -10.0, radius_km=d)["class"] == "outside_the_search_area"
    assert B.attempt_read(match(), LIVE, [wave(10.0, -6.0)], "6hr", 10.0, -10.0, radius_km=d - 1e-6)["class"] == "no_candidate_near"


def test_a_candidate_outside_the_search_area():
    near = B.attempt_read(match(), LIVE, [wave(10.0, -3.5)], "6hr", 10.0, -3.0)
    assert near["class"] == "outside_the_search_area" and near["chosen"] is None and near["candidates_inside"] == 0
    assert near["polygon_area"] == 4.0 and near["polygon_at_or_below_minimum"] is True and near["inside_if_inflated"] is True
    assert near["candidate_km"] == round(float(C.km(10.0, -3.0, 10.0, -3.5)), 1) and near["prediction_km"] == round(float(C.km(10.0, 0.0, 10.0, -3.5)), 1)
    far = B.attempt_read(match(), LIVE, [wave(10.0, -8.0)], "6hr", 10.0, -8.0)
    assert far["class"] == "outside_the_search_area" and far["inside_if_inflated"] is False, "the inflated square reaches 4.2 W"
    big = [[-10.0, 5.0], [10.0, 5.0], [10.0, 15.0], [-10.0, 15.0], [-10.0, 5.0]]
    wide = B.attempt_read(match(poly=big), LIVE, [wave(10.0, -12.0, half=0.5)], "6hr", 10.0, -12.0)
    assert wide["polygon_area"] == 200.0 and wide["polygon_at_or_below_minimum"] is False and wide["inside_if_inflated"] is False
    exact = [[0.0, 5.0], [7.5, 5.0], [7.5, 15.0], [0.0, 15.0], [0.0, 5.0]]
    at = B.attempt_read(match(poly=exact), LIVE, [wave(10.0, -12.0, half=0.5)], "6hr", 10.0, -12.0)
    assert at["polygon_area"] == 75.0 and at["polygon_at_or_below_minimum"] is True


def test_the_12_hour_pass_inflates_to_its_own_minimum_and_measures_from_its_own_prediction():
    w = [wave(10.0, -4.7, half=0.25)]
    six = B.attempt_read(match(), LIVE, w, "6hr", 10.0, -4.7)
    twelve = B.attempt_read(match(pss="12hr"), LIVE, w, "12hr", 10.0, -4.7)
    assert (six["inside_if_inflated"], twelve["inside_if_inflated"]) == (False, True), "the inflated square reaches 4.2 W at 6 hours and about 4.8 W at 12"
    assert twelve["prediction_km"] == round(float(C.km(10.0, -1.0, 10.0, -4.7)), 1) and six["prediction_km"] == round(float(C.km(10.0, 0.0, 10.0, -4.7)), 1)


def test_inside_the_search_area_the_speed_limit_and_the_choice_decide():
    waves = [wave(10.0, 0.5), wave(10.0, 0.0)]
    assert B.attempt_read(match(inside=[0, 1], chosen=1, speeds={"0": 25.0, "1": 3.0}), LIVE, waves, "6hr", 10.0, 0.6)["class"] == "over_the_speed_limit"
    assert B.attempt_read(match(inside=[0, 1], chosen=1, speeds={"0": 24.99, "1": 3.0}), LIVE, waves, "6hr", 10.0, 0.6)["class"] == "another_candidate_nearer"
    linked = B.attempt_read(match(inside=[0, 1], chosen=0), LIVE, waves, "6hr", 10.0, 0.6)
    assert linked["class"] == "linked" and linked["chosen"] == [10.0, 0.5] and "polygon_area" not in linked
    assert B.attempt_read(match(inside=[1], chosen=1), LIVE, waves, "6hr", 10.0, 0.6)["class"] == "outside_the_search_area", "the nearest to the reference is read, not the one taken"


def test_a_prediction_that_is_not_a_number_has_no_distance():
    got = B.attempt_read(match(), {"est_6hr": [None, None]}, [wave(10.0, -3.5)], "6hr", 10.0, -3.0)
    assert got["class"] == "outside_the_search_area" and got["prediction_km"] is None


def test_a_holder_that_is_not_due_is_refused():
    for m, lv, pss in ((None, LIVE, "6hr"), (match(), None, "6hr"), (match(pss="12hr"), LIVE, "6hr")):
        with pytest.raises(SystemExit, match="was not due"):
            B.attempt_read(m, lv, [wave(10.0, 0.0)], pss, 10.0, 0.0)


def still(_):
    return 0.0


def far(k):
    return wave(-30.0, 130.0, k * STEP)


def schedule():
    """A track seeded at 10 N 0 E that links once at 6 hours, then misses a candidate 2.5 degrees
    on at each of its two attempts. A second track far away stays last in the list, so the first
    keeps the natural polygon the 6-hour pass leaves it."""
    return [[wave(10.0, 0.0, 0 * STEP), far(0)], [wave(10.0, -2.0, 1 * STEP), far(1)], [wave(10.0, -4.5, 2 * STEP), far(2)], [wave(10.0, -7.0, 3 * STEP), far(3)],
            [far(4)], [far(5)]]


def fake_loop(sched):
    def loop(case, grids, ct, ft, flags, options, hook=None):
        return L.registered_loop([k * STEP for k in range(len(sched))], lambda k: (sched[k], still, still), False, hook=hook)
    return loop


def test_the_traced_replay_returns_the_histories_of_the_untraced_one_and_reads_before_the_call():
    from aew.v1port import association as ASSOC
    original = ASSOC.associate_step
    plain = L.registered_loop([k * STEP for k in range(6)], lambda k: (schedule()[k], still, still), False)
    wanted = {1: [("link", "0:0", "6hr", 10.0, -2.0)], 2: [("six", "0:0", "6hr", 10.0, -4.5)], 3: [("twelve", "0:0", "12hr", 10.0, -7.0)]}
    finished, rows, reads = B.traced_loop(None, None, None, None, None, {}, wanted, loop=fake_loop(schedule()))
    assert rows == plain[1] and len(finished) == len(plain[0]) and ASSOC.associate_step is original
    assert reads["link"]["class"] == "linked" and reads["link"]["chosen"] == [10.0, -2.0]
    six = reads["six"]
    assert (six["class"], six["chosen"], six["polygon_area"], six["inside_if_inflated"], six["polygon_at_or_below_minimum"]) == ("outside_the_search_area", None, 4.0, True, True)
    assert six["prediction_km"] == round(float(C.km(10.0, -2.0, 10.0, -4.5)), 1)
    assert reads["twelve"]["class"] == "outside_the_search_area" and reads["twelve"]["pass"] == "12hr" and reads["twelve"]["inside_if_inflated"] is True
    holder = next(r for r in rows if r["birth"] == "0:0")
    h = {"id": "0:0", "step": holder["step"], "raw_lat": holder["lat"], "raw_lon": holder["lon"]}
    for key, step in (("link", 1), ("six", 2), ("twelve", 3)):
        B.check_attempt(reads[key], h, step)
    with pytest.raises(SystemExit, match="disagree on what the holder took"):
        B.check_attempt(reads["six"], h, 1)
    with pytest.raises(SystemExit, match="disagree on what the holder took"):
        B.check_attempt(reads["link"], h, 2)


def test_the_production_function_is_put_back_when_the_replay_fails():
    from aew.v1port import association as ASSOC
    original = ASSOC.associate_step

    def broken(*args, **kwargs):
        assert ASSOC.associate_step is not original, "the read is in place while the loop runs"
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError):
        B.traced_loop(None, None, None, None, None, {}, {}, loop=broken)
    assert ASSOC.associate_step is original
    with pytest.raises(SystemExit, match="was not due"):
        B.traced_loop(None, None, None, None, None, {}, {3: [("x", "0:0", "6hr", 10.0, -7.0)]}, loop=fake_loop(schedule()))
    assert ASSOC.associate_step is original


def test_a_step_that_is_not_wanted_is_not_read():
    _, _, reads = B.traced_loop(None, None, None, None, None, {}, {2: [("six", "0:0", "6hr", 10.0, -4.5)]}, loop=fake_loop(schedule()))
    assert list(reads) == ["six"]


# --- tables ------------------------------------------------------------------------------------------------------

def tr(cls, seg, bd="0_to_10E", **extra):
    return {"k": 0, "lon": 5.0, "band": bd, "hours_to_coast": 6 if seg == "coastal" else 60, "segment": seg, "class": cls, "holders": {}, **extra}


def br(seg, nxt, did="ended", support="supported", first="no_candidate_near", second=None, bd="0_to_10E", **extra):
    a = [{"class": first, "pass": "6hr", **extra.pop("first_extra", {})}]
    if second:
        a.append({"class": second, "pass": "12hr"})
    return tr("break", seg, bd, next=nxt, holders_did=did, support=support, attempts=a, **extra)


def primary(stop, **flags):
    return {"history": "a", "last": 0, "hours_before": 0, "stop": stop, "start": True, "end": True, "majority": True, "no_long_gap": True, **flags}


def refs_fixture():
    out_a = {"class": "outside_the_search_area", "pass": "6hr", "prediction_km": 300.0, "polygon_area": 10.0, "polygon_at_or_below_minimum": True, "inside_if_inflated": True}
    out_b = {**out_a, "prediction_km": 500.0, "polygon_area": 90.0, "polygon_at_or_below_minimum": False, "inside_if_inflated": False}
    r1 = {"id": 1, "coast_reaching": True, "steps": 12, "accompanied": 8, "sustained": False, "primary": primary("within_24_hours", end=False),
          "transitions": [tr("held", "earlier"), tr("held", "earlier", "east_of_10E"),
                          {**br("earlier", "handed_off_fresh", bd="east_of_10E"), "attempts": [out_a]},
                          {**br("coastal", "candidate_not_stored", did="continued_away", support="partial", marginal=True, by_smoothing_alone=False,
                                not_stored="held_by_pruned_histories_only", bd="west_of_10W"), "attempts": [out_b, {"class": "no_candidate_near", "pass": "12hr"}]},
                          tr("held", "coastal", "west_of_10W"), tr("unresolved", "coastal", "west_of_10W")]}
    r2 = {"id": 2, "coast_reaching": True, "steps": 12, "accompanied": 11, "sustained": True, "primary": primary("at_the_first_atlantic_side_observation"),
          "transitions": [tr("held", "earlier"), tr("held", "coastal")]}
    r3 = {"id": 3, "coast_reaching": True, "steps": 12, "accompanied": 0, "sustained": False, "primary": None, "transitions": []}
    r4 = {"id": 4, "coast_reaching": False, "steps": 9, "accompanied": 3, "sustained": False, "primary": None,
          "transitions": [{**tr("held", None), "hours_to_coast": None}, {**br(None, "placed_away", first="no_candidate_near"), "hours_to_coast": None}]}
    r5 = {"id": 5, "coast_reaching": True, "steps": 12, "accompanied": 4, "sustained": False, "primary": primary("earlier", end=False, majority=False, start=False),
          "transitions": [br("earlier", "placed_away", support="unsupported", first="no_candidate_near", second="another_candidate_nearer", did="continued_away",
                             marginal=False, by_smoothing_alone=True)]}
    r6 = {"id": 6, "coast_reaching": True, "steps": 12, "accompanied": 2, "sustained": False, "primary": primary("30_to_48_hours", majority=False), "transitions": []}
    return [r1, r2, r3, r4, r5, r6]


def test_the_tables_of_a_constructed_configuration():
    t = B.tally(refs_fixture())
    assert (t["references"], t["coast_reaching"]) == (6, 5)
    assert t["transitions"] == {"coastal": {"held": 2, "break": 1, "unresolved": 1}, "earlier": {"held": 3, "break": 2, "unresolved": 0}}
    assert t["bands"]["coast_reaching"]["west_of_10W"] == {"held": 1, "break": 1, "unresolved": 1} and t["bands"]["coast_reaching"]["east_of_10E"] == {"held": 1, "break": 1, "unresolved": 0}
    assert t["bands"]["ending_inside_africa"]["0_to_10E"] == {"held": 1, "break": 1, "unresolved": 0}
    assert t["per_reference"] == [{"id": 1, "coastal": [1, 2], "earlier": [1, 3]}, {"id": 2, "coastal": [0, 1], "earlier": [0, 1]},
                                  {"id": 3, "coastal": [0, 0], "earlier": [0, 0]}, {"id": 5, "coastal": [0, 0], "earlier": [1, 1]},
                                  {"id": 6, "coastal": [0, 0], "earlier": [0, 0]}]
    p = t["primary"]
    assert (p["sustained"], p["no_history"], p["others"]) == (1, 1, 3) and p["lacking"] == {"start": 1, "end": 2, "majority": 2, "no_long_gap": 0}
    assert p["stops"] == {"at_the_first_atlantic_side_observation": 1, "within_24_hours": 1, "30_to_48_hours": 1, "earlier": 1}
    assert p["stops_lacking_the_end"] == {"at_the_first_atlantic_side_observation": 0, "within_24_hours": 1, "30_to_48_hours": 0, "earlier": 1}
    b = t["breaks"]
    assert (b["count"], b["references_with_a_break"], b["away_marginal"], b["away_by_smoothing_alone"]) == (3, 2, 1, 1), "the break of the reference ending inside Africa is not attributed"
    assert b["next"]["coastal"]["candidate_not_stored"] == 1 and b["next"]["earlier"] == {**{n: 0 for n in B.NEXT}, "handed_off_fresh": 1, "placed_away": 1}
    assert b["holders_did"] == {"coastal": {"ended": 0, "continued_away": 1}, "earlier": {"ended": 1, "continued_away": 1}}
    assert b["holders_did_by_next"]["placed_away"] == {"ended": 0, "continued_away": 1} and b["not_stored"]["held_by_pruned_histories_only"] == 1
    assert b["support"] == {"supported": 1, "partial": 1, "unsupported": 1} and b["support_by_next"]["candidate_not_stored"] == {"supported": 0, "partial": 1, "unsupported": 0}
    a = t["attempts"]
    assert a["first"] == {**{c: 0 for c in B.ATTEMPTS}, "outside_the_search_area": 2, "no_candidate_near": 1}
    assert a["first_by_next"]["handed_off_fresh"]["outside_the_search_area"] == 1 and a["first_by_segment"]["coastal"]["outside_the_search_area"] == 1
    assert a["second_read"] == 2 and a["second"] == {**{c: 0 for c in B.ATTEMPTS}, "no_candidate_near": 1, "another_candidate_nearer": 1}
    assert a["first_outside"] == {"count": 2, "inside_if_inflated": 1, "polygon_at_or_below_minimum": 1, "median_prediction_km": 400.0, "median_polygon_area": 50.0}
    assert a["candidate_near_at_either_attempt"] == 3
    assert t["concentration"]["outcome"] == "cannot be scored" and t["concentration"]["scored"] == 2


def season_record(year, sha=None):
    return {"generated_by": {"script": "scripts/continuity_break_reading.py", "sha256": sha or B.SCRIPT_SHA256_AT_LOAD}, "year": year,
            "configs": {n: {"references": refs_fixture(), "steps_staged": 3, "attempts_read": 5} for n in B.CONFIGS}}


def write_season(tmp_path, year, **kw):
    with gzip.open(tmp_path / f"continuity_break_reading_{year}.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(season_record(year, **kw), fh)


def test_combine_pools_the_seasons_and_writes_the_tables(tmp_path, capsys):
    for y in B.YEARS:
        write_season(tmp_path, y)
    assert B.main(["combine", "--out-dir", str(tmp_path)]) == 0
    rec = json.load(open(tmp_path / "continuity_break_reading.json", encoding="utf-8"))
    assert set(rec["pooled"]) == set(B.CONFIGS) == {"released", "VCLIP"} and set(rec["seasons"]) == {"1990", "2006"}
    p = rec["pooled"]["VCLIP"]
    assert p["coast_reaching"] == 10 and p["breaks"]["count"] == 6 and p["transitions"]["earlier"] == {"held": 6, "break": 4, "unresolved": 0}
    assert p["concentration"]["scored"] == 4 and rec["seasons"]["2006"]["released"]["breaks"]["count"] == 3
    text = open(tmp_path / "continuity_break_reading.md", encoding="utf-8").read()
    assert text.strip() in capsys.readouterr().out
    assert "| coastal | 4 | 2 | 2 | 2 of 6 (33.3 %) |" in text and "Outcome: CANNOT BE SCORED." in text and "## VCLIP" in text
    assert "| outside the search area | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 4 | 2 | 2 |" in text
    with pytest.raises(SystemExit, match="never overwritten"):
        B.main(["combine", "--out-dir", str(tmp_path)])


def test_combine_refuses_a_season_record_from_another_script(tmp_path):
    write_season(tmp_path, 1990)
    write_season(tmp_path, 2006, sha="0" * 64)
    with pytest.raises(SystemExit, match="not written by the current script"):
        B.main(["combine", "--out-dir", str(tmp_path)])


def test_combine_refuses_a_record_filed_under_another_year(tmp_path):
    write_season(tmp_path, 1990)
    with gzip.open(tmp_path / "continuity_break_reading_2006.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(season_record(1990), fh)
    with pytest.raises(SystemExit, match="not written by the current script"):
        B.main(["combine", "--out-dir", str(tmp_path)])


def test_only_development_seasons_are_read_and_records_are_never_overwritten(tmp_path, monkeypatch):
    for y in (1984, 1992, 1997, 2003, 2009, 2005, 2002):
        with pytest.raises(SystemExit, match="not a development season"):
            B.run_season(y)

    def ran(year):
        raise AssertionError("a season was run over an existing record")

    monkeypatch.setattr(B, "run_season", ran)
    write_season(tmp_path, 1990)
    with pytest.raises(SystemExit, match="never overwritten"):
        B.main(["season", "--year", "1990", "--out-dir", str(tmp_path)])


def records(tmp_path, life_sha=None, place_sha=None, place_life_sha=None, art_sha=None, life_art_sha=None):
    art = tmp_path / "artifact.json"
    art.write_text("{}", encoding="utf-8")
    digest = C.sha256_of(str(art))
    with gzip.open(tmp_path / "continuity_lifecycle_1990.json.gz", "wt", encoding="utf-8") as fh:
        json.dump({"generated_by": {"sha256": life_sha or L.SCRIPT_SHA256_AT_LOAD}, "artifact": {"path": str(art), "sha256": life_art_sha or digest}}, fh)
    with gzip.open(tmp_path / "continuity_placement_development_1990.json.gz", "wt", encoding="utf-8") as fh:
        json.dump({"generated_by": {"sha256": place_sha or B.P.SCRIPT_SHA256_AT_LOAD, "lifecycle_sha256": place_life_sha or L.SCRIPT_SHA256_AT_LOAD},
                   "artifact": {"path": str(art), "sha256": art_sha or digest}}, fh)
    return art


def test_the_retained_records_are_bound_by_digest(tmp_path):
    records(tmp_path)
    life, place = B.load_records(1990, str(tmp_path))
    assert life["artifact"] == place["artifact"]
    records(tmp_path, art_sha="0" * 64, life_art_sha="0" * 64)                   # the two agree with each other and not with the file
    with pytest.raises(SystemExit, match="or it has changed"):
        B.load_records(1990, str(tmp_path))
    for kw, msg in (({"life_sha": "0" * 64}, "lifecycle record"), ({"place_sha": "0" * 64}, "placement development record"),
                    ({"place_life_sha": "0" * 64}, "placement development record"), ({"art_sha": "0" * 64}, "one coast-crossing artifact"),
                    ({"life_art_sha": "1" * 64}, "one coast-crossing artifact")):
        records(tmp_path, **kw)
        with pytest.raises(SystemExit, match=msg):
            B.load_records(1990, str(tmp_path))
    art = records(tmp_path)
    art.write_text("{ }", encoding="utf-8")
    with pytest.raises(SystemExit, match="or it has changed"):
        B.load_records(1990, str(tmp_path))
