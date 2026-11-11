"""Tests for scripts/continuity_recovery.py (`CONTINUITY_ACROSS_AFRICA_2026-11-09.md`, section 20)."""

import copy
import gzip
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import continuity_recovery as V  # noqa: E402
from aew.v1port.association import predict_displacement  # noqa: E402

L, E = V.L, V.E
D0 = 7000.0
STEP = 0.25


def wave(lat, lon, time, half=1.0, n=5):
    lats = np.repeat(np.linspace(-half, half, n), n) + lat
    lons = np.tile(np.linspace(-half, half, n), n) + lon
    return {"time": time, "lat_mean": float(lat), "lon_mean": float(lon), "lat_wave": lats, "lon_wave": lons, "region": np.zeros((1, 1), bool)}


def still(_):
    return 0.0


def westward(_):
    return -8.0


def far(k):
    return wave(-30.0, 130.0, k * STEP)


def run(schedule, u=still, hook=None, recent_days=V.RECENT_DAYS):
    times = [k * STEP for k in range(len(schedule))]
    return L.registered_loop(times, lambda k: (schedule[k], u, still), False, hook=hook, recent_days=recent_days)


def recover(schedule, u=still, **kwargs):
    hook = V.Recovery(**kwargs)
    finished, rows, checks = run(schedule, u=u, hook=hook)
    return finished, {r["birth"]: r for r in rows}, hook.joins


def gap_schedule(at, lon=0.0, n=9, first=(10.0, 0.0), half=1.0):
    """A track seeded at the first step, nothing near it until a candidate at step `at`."""
    out = [[wave(first[0], first[1], 0.0, half=half)]] + [[far(k)] for k in range(1, n)]
    out[at] = [wave(10.0, lon, at * STEP), far(at)]
    return out


# --- the later attempts ----------------------------------------------------------------------

def test_a_track_joins_a_candidate_at_its_position_18_hours_on():
    _, b, joins = recover(gap_schedule(3))
    assert b["0:0"]["step"] == [0, 3] and b["0:0"]["est"] == [[10.0, 0.0, 18]]
    assert "3:0" not in b, "the candidate's seed is absorbed and leaves no history of its own"
    assert joins == [{"track": "0:0", "seed": "3:0", "step": 3, "hours": 18, "km_from_prediction": 0.0}]
    assert b["0:0"]["fate"] == "pruned_in_loop" and b["1:0"]["step"] == list(range(1, 9)), "two observations still fall to the prune, a day after the last"


def test_the_last_attempt_is_at_24_hours():
    _, b, joins = recover(gap_schedule(4))
    assert b["0:0"]["step"] == [0, 4] and b["0:0"]["est"] == [[10.0, 0.0, 24]] and joins[0]["hours"] == 24
    _, b, joins = recover(gap_schedule(5))
    assert b["0:0"]["step"] == [0] and b["5:0"]["step"] == [5] and joins == [], "30 hours is past the horizon"


def test_a_short_track_is_held_until_its_24_hour_attempt_once_the_prune_is_running():
    schedule = [[far(k)] for k in range(16)]
    schedule[8] = [far(8), wave(10.0, 0.0, 8 * STEP)]
    schedule[12] = [far(12), wave(10.0, 0.0, 12 * STEP)]
    _, b, joins = recover(schedule)
    assert b["8:1"]["step"] == [8, 12] and [j["hours"] for j in joins] == [24]
    hook = V.Recovery()
    rows = {r["birth"]: r for r in run(schedule, hook=hook, recent_days=None)[1]}
    assert rows["8:1"]["step"] == [8] and rows["12:1"]["step"] == [12] and hook.joins == [], "production's half day prunes it at 18 hours"


def test_no_later_attempt_is_production():
    for schedule in (gap_schedule(3), gap_schedule(4), gap_schedule(5)):
        plain = L.registered_loop([k * STEP for k in range(len(schedule))], lambda k: (schedule[k], still, still), False)
        hooked = run(schedule, hook=V.Recovery(hours=()), recent_days=None)
        assert hooked[1] == plain[1] and hooked[2] == plain[2] and len(hooked[0]) == len(plain[0])
    assert {r["birth"]: r["step"] for r in plain[1]}["5:0"] == [5]


def test_the_search_area_grows_with_the_hours():
    # production's inflation carries a trough two degrees across to 5.36 W at 125 square degrees and to 5.86 W at 150
    assert recover(gap_schedule(3, lon=-6.3))[2][0]["hours"] == 18, "its nearest points, at 5.3 W, are inside"
    assert recover(gap_schedule(3, lon=-6.6))[2] == [], "at 5.6 W they are outside the 18-hour area"
    assert recover(gap_schedule(4, lon=-6.6))[2][0]["hours"] == 24, "and inside the 24-hour one"
    assert recover(gap_schedule(4, lon=-7.0))[2] == []
    assert recover(gap_schedule(4, lon=-7.0), areas={18: 125.0, 24: 400.0})[2] != [], "the areas are the rule's own parameters"


def test_the_production_speed_maximum_applies_to_a_join():
    wide = dict(first=(10.0, 0.0), half=20.0)                        # a trough 40 degrees across, so the area holds both candidates
    assert recover(gap_schedule(3, lon=14.0, **wide))[2][0]["km_from_prediction"] == pytest.approx(1532.9, abs=1.0), "about 23.7 m/s"
    assert recover(gap_schedule(3, lon=16.0, **wide))[2] == [], "about 27 m/s over 18 hours"
    assert recover(gap_schedule(4, lon=16.0, **wide))[2] != [], "the same distance over 24 hours is under the maximum"


def test_a_candidate_a_production_attempt_claimed_is_not_available():
    schedule = [[wave(10.0, 0.0, 0.0)], [far(1)], [far(2), wave(10.0, -7.0, 2 * STEP)], [far(3), wave(10.0, -3.5, 3 * STEP)]] + [[far(k)] for k in range(4, 9)]
    _, b, joins = recover(schedule)
    assert b["2:1"]["step"] == [2, 3] and b["0:0"]["step"] == [0] and joins == [], "the newer track reached it at 6 hours"
    schedule[2] = [far(2)]
    _, b, joins = recover(schedule)
    assert b["0:0"]["step"] == [0, 3] and [j["seed"] for j in joins] == ["3:1"], "without that track the candidate is free"


def test_one_candidate_serves_one_later_attempt_in_list_order():
    schedule = gap_schedule(3)
    schedule[0] = [wave(10.0, -1.0, 0.0), wave(10.0, 1.0, 0.0)]
    _, b, joins = recover(schedule)
    assert b["0:0"]["step"] == [0, 3] and b["0:1"]["step"] == [0] and len(joins) == 1


def test_the_18_hour_attempts_come_before_the_24_hour_ones():
    schedule = [[wave(10.0, -1.0, 0.0)], [far(1), wave(10.0, 8.0, STEP)], [far(2)], [far(3)], [far(4), wave(10.0, 3.5, 4 * STEP)]] + [[far(k)] for k in range(5, 9)]
    _, b, joins = recover(schedule)
    assert b["1:1"]["step"] == [1, 4] and b["0:0"]["step"] == [0], "the track seen 18 hours ago joins although the older one comes first in the list"
    assert joins == [{"track": "1:1", "seed": "4:1", "step": 4, "hours": 18, "km_from_prediction": pytest.approx(492.7, abs=1.0)}]
    assert recover(schedule, hours=(24,))[1]["0:0"]["step"] == [0, 4], "with only the 24-hour attempt the older one takes it"


def test_each_track_takes_the_candidate_nearest_its_prediction():
    schedule = gap_schedule(3)
    schedule[3] = [wave(10.0, 3.0, 3 * STEP), wave(10.0, -2.0, 3 * STEP), far(3)]
    _, b, joins = recover(schedule)
    assert [j["seed"] for j in joins] == ["3:1"] and b["0:0"]["lon"] == [0.0, -2.0] and b["3:0"]["step"] == [3]


def test_a_joined_track_goes_on_through_the_production_attempts():
    schedule = gap_schedule(3)
    schedule[4] = [wave(10.0, 0.0, 4 * STEP), far(4)]
    _, b, joins = recover(schedule)
    assert b["0:0"]["step"] == [0, 3, 4] and [e[2] for e in b["0:0"]["est"]] == [18, 6] and len(joins) == 1


def test_the_prediction_uses_the_wind_at_the_last_observation():
    _, dlon18 = predict_displacement(-8.0, 0.0, 10.0, 18.0)

    def early_wind(w):
        return -8.0 if w["time"] < 0.1 else 0.0                     # westward at the first step only

    schedule = gap_schedule(3, lon=dlon18)
    _, b, joins = recover(schedule, u=early_wind)
    assert joins[0]["km_from_prediction"] == pytest.approx(0.0, abs=0.2) and b["0:0"]["est"] == [[10.0, round(dlon18, 3), 18]]
    _, b, joins = recover(gap_schedule(3, lon=0.0), u=early_wind)
    assert joins[0]["km_from_prediction"] == pytest.approx(float(E.C.km(10.0, dlon18, 10.0, 0.0)), abs=0.2), "a candidate at the old position is far from the prediction"
    assert recover(gap_schedule(3, lon=dlon18 - 5.0), u=early_wind)[2] != [], "the search area travels with the prediction"
    assert recover(gap_schedule(3, lon=dlon18 - 5.0))[2] == [], "and without the wind that candidate is out of reach"


def test_a_second_join_uses_the_wind_of_the_first_joined_observation():
    def early_wind(w):
        return -8.0 if w["time"] < 0.1 else 0.0

    schedule = gap_schedule(3, n=12)
    schedule[6] = [wave(10.0, 0.0, 6 * STEP), far(6)]
    _, b, joins = recover(schedule, u=early_wind)
    assert b["0:0"]["step"] == [0, 3, 6] and [j["hours"] for j in joins] == [18, 18]
    assert joins[1]["km_from_prediction"] == pytest.approx(0.0, abs=0.2), "still air at the joined observation, so the prediction stays put"


def test_near_takes_only_a_candidate_within_500_km_of_the_prediction():
    _, dlon18 = predict_displacement(-8.0, 0.0, 10.0, 18.0)

    def early_wind(w):
        return -8.0 if w["time"] < 0.1 else 0.0

    here = gap_schedule(3, lon=0.0)                                  # about 518 km from a prediction 4.7 degrees to the west
    assert recover(here, u=early_wind)[2][0]["km_from_prediction"] > 500.0, "RECOVER joins it"
    finished, b, joins = recover(here, u=early_wind, max_km=500.0)
    assert joins == [] and b["0:0"]["step"] == [0] and b["3:0"]["step"] == [3], "NEAR leaves it to seed its own track"
    _, b, joins = recover(gap_schedule(3, lon=dlon18 + 4.5), u=early_wind, max_km=500.0)
    assert b["0:0"]["step"] == [0, 3] and joins[0]["km_from_prediction"] == pytest.approx(492.7, abs=1.0), "4.5 degrees is inside"
    both = gap_schedule(3, lon=0.0)
    both[3] = [wave(10.0, 0.0, 3 * STEP), wave(10.0, dlon18 - 4.0, 3 * STEP), far(3)]
    _, b, joins = recover(both, u=early_wind, max_km=500.0)
    assert [j["seed"] for j in joins] == ["3:1"], "the far candidate is passed over for one within reach"
    rows = list(recover(here, u=early_wind)[1].values())
    assert V.far_joins(rows, 500.0) == 1 and V.far_joins(rows, 600.0) == 0 and V.far_joins(list(recover(here, u=early_wind, max_km=500.0)[1].values()), 500.0) == 0


def test_near_measures_from_the_recorded_prediction_as_a_jump_link_is():
    _, dlon18 = predict_displacement(-8.0, 0.0, 10.0, 18.0)

    def early_wind(w):
        return -8.0 if w["time"] < 0.1 else 0.0

    _, b, joins = recover(gap_schedule(3, lon=dlon18 + 4.5), u=early_wind, max_km=500.0)
    row = b["0:0"]
    hist_row = {"id": "0:0", "time": np.asarray(row["time"]), "lat": np.asarray(row["lat"]), "lon": np.asarray(row["lon"]), "step": row["step"],
                "raw_lat": row["lat"], "raw_lon": row["lon"], "est": row["est"]}
    d = float(E.C.km(row["est"][0][0], row["est"][0][1], row["lat"][1], row["lon"][1]))
    assert not E.jump(hist_row, 1) and d < 500.0
    assert recover(gap_schedule(3, lon=dlon18 + 4.5), u=early_wind, max_km=d)[2] == [], "at exactly the limit the candidate is out"
    assert recover(gap_schedule(3, lon=dlon18 + 4.5), u=early_wind, max_km=d + 1e-6)[2] != []


def fragments():
    """Five observations moving west, a gap of 18 hours, five more."""
    _, d6 = predict_displacement(-8.0, 0.0, 10.0, 6.0)
    _, d18 = predict_displacement(-8.0, 0.0, 10.0, 18.0)
    lons = [k * d6 for k in range(5)]
    lons += [lons[-1] + d18 + k * d6 for k in range(5)]
    steps = list(range(5)) + list(range(7, 12))
    schedule = [[far(k)] for k in range(17)]
    for s, lo in zip(steps, lons):
        schedule[s] = [wave(10.0, lo, s * STEP), far(s)]
    return schedule, steps


def test_two_pruned_fragments_become_one_finished_track():
    schedule, steps = fragments()
    plain = {r["birth"]: r for r in run(schedule, u=westward, recent_days=None)[1]}
    assert (plain["0:0"]["fate"], plain["7:0"]["fate"]) == ("pruned_in_loop", "pruned_in_loop") and len(plain["0:0"]["step"]) == 5
    finished, b, joins = recover(schedule, u=westward)
    assert b["0:0"]["step"] == steps and b["0:0"]["fate"] == "finished" and "7:0" not in b and len(joins) == 1
    assert [f["step"] for f in finished if len(f["step"]) == 10] == [steps]
    assert [e[2] for e in b["0:0"]["est"]] == [6, 6, 6, 6, 18, 6, 6, 6, 6]


def test_the_registry_holds_no_absorbed_seed():
    schedule, _ = fragments()
    hook = V.Recovery()
    _, rows, checks = run(schedule, u=westward, hook=hook)
    births = [r["birth"] for r in rows]
    assert len(births) == len(set(births)) and "7:0" not in births and all(r["fate"] != "live" for r in rows)
    plain = run(schedule, u=westward, recent_days=None)[1]
    assert len(plain) - len(rows) == len(hook.joins) == 1


# --- joins read against QTrack ------------------------------------------------------------------

def q(tid, lats, lons, start=D0):
    n = len(lats)
    return {"id": tid, "time": start + 0.25 * np.arange(n), "lat": np.asarray(lats, float), "lon": np.asarray(lons, float), "n_valid": n}


def hist(hid, steps, lons, est, lat=10.0):
    n = len(steps)
    return {"id": hid, "time": D0 + 0.25 * np.asarray(steps, float), "lat": np.full(n, lat), "lon": np.asarray(lons, float), "step": list(steps),
            "raw_lat": [lat] * n, "raw_lon": [float(x) for x in lons], "est": est, "fate": "finished"}


def census(histories, qs):
    return V.join_census(histories, E.C.index_by_time(qs), E.J.spans(qs))


def test_joins_are_counted_by_status_and_only_joins():
    a = q(1, [10.0] * 6, [0.0, -2.0, -4.0, -6.0, -8.0, -10.0])
    same = hist("same", [0, 1, 4], [0.0, -2.0, -8.0], [[10.0, -2.0, 6], [10.0, -8.0, 18]])          # beside one system on both sides of the gap
    alone = hist("alone", [0, 4], [0.0, 30.0], [[10.0, 0.0, 24]])                                    # the later end has no companion
    plain = hist("plain", [0, 1], [0.0, -2.0], [[10.0, -2.0, 6]])                                    # no join in it
    got = census([same, alone, plain], [a])
    assert got == {"joins": 2, "consistent": 1, "switch": 0, "relay": 0, "undetermined": 0, "unassessed": 1, "supported_false_joins": 0, "jump_joins": 1,
                   "by_hours": {"18": 1, "24": 1}}
    outside = hist("out", [0, 3], [100.0, 98.0], [[10.0, 98.0, 18]])
    assert census([outside], [a])["joins"] == 0, "outside the sector"


def test_a_supported_false_join_needs_a_jump_and_the_old_system_at_the_prediction():
    west = -15.7                                                                                     # about 1,500 km from a prediction at 2 W
    a = q(1, [10.0] * 4, [0.0, -0.7, -1.4, -2.0])                                                    # goes on to where the track was predicted
    b = q(2, [10.0], [west], start=D0 + 0.75)
    jumped = hist("j", [0, 3], [0.0, west], [[10.0, -2.0, 18]])
    assert census([jumped], [a, b])["switch"] == 1 and census([jumped], [a, b])["supported_false_joins"] == 1
    a_far = q(1, [10.0] * 4, [0.0, 4.0, 8.0, 12.0])                                                  # alive, away from the prediction
    assert census([jumped], [a_far, b]) == {**census([jumped], [a, b]), "supported_false_joins": 0}
    near = q(2, [10.0], [-7.5], start=D0 + 0.75)
    drifted = hist("d", [0, 3], [0.0, -7.5], [[10.0, -7.5, 18]])                                     # claimed at its prediction, the old system elsewhere
    got = census([drifted], [a_far, near])
    assert (got["switch"], got["supported_false_joins"], got["jump_joins"]) == (1, 0, 0)
    ended = q(1, [10.0], [0.0])
    assert census([jumped], [ended, b])["relay"] == 1 and census([jumped], [ended, b])["supported_false_joins"] == 0
    beside = q(1, [10.0] * 4, [0.0, -2.0, -4.0, -5.5])                                               # 3.5 degrees from the prediction, 3.5 from the claim
    wide = hist("w", [0, 3], [0.0, -9.0], [[10.0, -2.0, 18]])                                        # a jump of 7 degrees that stays beside one system
    got = census([wide], [beside])
    assert (got["consistent"], got["jump_joins"], got["supported_false_joins"]) == (1, 1, 0), "only a switch can be a supported false join"


def sides(history, qs, hours=V.HOURS):
    return V.earlier_side_census([history], E.C.index_by_time(qs), E.J.spans(qs), hours)


def test_the_constructed_case_of_section_21_reads_as_left_and_supported():
    west = -15.7
    a = q(1, [10.0] * 4, [0.0, -0.7, -1.4, -2.0])                                                    # goes on to where the track was predicted
    jumped = hist("j", [0, 3], [0.0, west], [[10.0, -2.0, 18]])                                      # joins where no QTrack system is
    qi, qs = E.C.index_by_time([a]), E.J.spans([a])
    assert V.join_marks(jumped, 1, qi, qs) == ("unassessed", False), "section 20's reading cannot see it"
    assert V.earlier_side(jumped, 1, qi, qs) == ("left", True)
    assert sides(jumped, [a]) == {"all": 1, "jumps": 1, "links": 1, "kept": 0, "left": 1, "ended": 0, "undetermined": 0, "supported": 1}


def test_the_earlier_side_classes():
    a = q(1, [10.0] * 4, [0.0, -2.0, -4.0, -6.0])
    kept = hist("k", [0, 3], [0.0, -6.0], [[10.0, -6.0, 18]])
    qi, qs = E.C.index_by_time([a]), E.J.spans([a])
    assert V.earlier_side(kept, 1, qi, qs) == ("kept", False)
    left_near = hist("l", [0, 3], [0.0, -13.0], [[10.0, -12.0, 18]])                                 # claimed at its prediction, the system 6 degrees away
    assert V.earlier_side(left_near, 1, qi, qs) == ("left", False), "left without a jump is not supported"
    left_jump_elsewhere = hist("e", [0, 3], [0.0, -20.0], [[10.0, -13.0, 18]])                       # a jump, but the system is not at the prediction
    assert V.earlier_side(left_jump_elsewhere, 1, qi, qs) == ("left", False)
    ended = q(1, [10.0], [0.0])
    assert V.earlier_side(kept, 1, E.C.index_by_time([ended]), E.J.spans([ended])) == ("ended", False)
    gap = q(1, [10.0, 10.0, 10.0, np.nan, 10.0], [0.0, -2.0, -4.0, np.nan, -8.0])
    assert V.earlier_side(kept, 1, E.C.index_by_time([gap]), E.J.spans([gap])) == ("undetermined", False), "alive, without a position at the join"
    apart = q(1, [30.0] * 4, [0.0, -2.0, -4.0, -6.0])
    assert V.earlier_side(kept, 1, E.C.index_by_time([apart]), E.J.spans([apart])) == (None, False), "no system beside the earlier side"
    later_only = q(1, [10.0], [-6.0], start=D0 + 0.75)
    assert V.earlier_side(kept, 1, E.C.index_by_time([later_only]), E.J.spans([later_only])) == (None, False)
    leaving = q(2, [10.0] * 4, [0.3, 3.0, 6.0, 9.0])                                                 # a second system beside the earlier side only
    both = [a, leaving]
    assert V.earlier_side(kept, 1, E.C.index_by_time(both), E.J.spans(both)) == ("kept", False), "one system kept is enough"


def test_the_earlier_side_agrees_with_the_two_sided_reading_where_both_sides_are_accompanied():
    t = hist("t", [0, 3], [0.0, -15.7], [[10.0, -2.0, 18]])
    b = q(2, [10.0], [-15.7], start=D0 + 0.75)
    for first, two_sided, one_sided in ((q(1, [10.0] * 4, [0.0, -0.7, -1.4, -2.0]), "switch", "left"), (q(1, [10.0], [0.0]), "relay", "ended"),
                                        (q(1, [10.0, 10.0, 10.0, np.nan, 10.0], [0.0, 1.0, 2.0, np.nan, 4.0]), "undetermined", "undetermined")):
        qs = [first, b]
        qi, sp = E.C.index_by_time(qs), E.J.spans(qs)
        assert V.join_marks(t, 1, qi, sp)[0] == two_sided and V.earlier_side(t, 1, qi, sp)[0] == one_sided
    assert V.join_marks(t, 1, *[f([q(1, [10.0] * 4, [0.0, -0.7, -1.4, -2.0]), b]) for f in (E.C.index_by_time, E.J.spans)])[1] is True
    assert V.earlier_side(t, 1, *[f([q(1, [10.0] * 4, [0.0, -0.7, -1.4, -2.0]), b]) for f in (E.C.index_by_time, E.J.spans)])[1] is True


def test_the_earlier_side_census_is_by_hours_and_in_the_sector():
    a = q(1, [10.0] * 6, [0.0, -2.0, -4.0, -6.0, -8.0, -10.0])
    mixed = hist("m", [0, 1, 3, 5], [0.0, -2.0, -6.0, 20.0], [[10.0, -2.0, 6], [10.0, -6.0, 12], [10.0, -10.0, 18]])
    assert sides(mixed, [a]) == {"all": 1, "jumps": 1, "links": 1, "kept": 0, "left": 1, "ended": 0, "undetermined": 0, "supported": 1}
    assert sides(mixed, [a], hours=(12,)) == {"all": 1, "jumps": 0, "links": 1, "kept": 1, "left": 0, "ended": 0, "undetermined": 0, "supported": 0}
    assert sides(mixed, [a], hours=(6, 12))["kept"] == 2
    outside = hist("o", [0, 3], [100.0, 98.0], [[10.0, 98.0, 18]])
    assert sides(outside, [q(1, [10.0] * 4, [100.0, 99.0, 98.5, 98.0])])["all"] == 0
    alone = hist("n", [0, 3], [0.0, -6.0], [[30.0, -6.0, 18]], lat=30.0)
    assert sides(alone, [a]) == {"all": 1, "jumps": 0, "links": 0, "kept": 0, "left": 0, "ended": 0, "undetermined": 0, "supported": 0}


# --- the rule -------------------------------------------------------------------------------------

def pooled(carried=(2, 4), longest=(154, 154), accompanied=(170, 170), joins=None, steps=784):
    def ref(i):
        return {"steps": steps, "tracks": 42, "accompanied": accompanied[i], "longest_single_history": longest[i],
                "classes": {"carried": carried[i], "relayed": 0, "picked_up_late": 0, "dropped": 0, "unaccompanied_at_both": 0}}
    j = {"joins": 40, "consistent": 34, "switch": 1, "relay": 5, "undetermined": 0, "unassessed": 0, "supported_false_joins": 1, "jump_joins": 5, "by_hours": {"18": 25, "24": 15}}
    j.update(joins or {})
    return {"released": {"reference": ref(0)}, "recover": {"reference": ref(1), "joins_in_finished_sector_tracks": j}}


def test_the_rule_passes_only_with_every_condition():
    d = V.decide(pooled())
    assert d["carried_to_freezing"] is True and d["assessed_joins"] == 40 and d["flagged_joins"] == 1 and all(d["checks"].values())
    assert d["carried_rise"] == 2 and d["longest_rise"] == 0 and d["accompanied_change"] == 0


def failing(**kwargs):
    d = V.decide(pooled(**kwargs))
    return sorted(k for k, v in d["checks"].items() if not v), d["carried_to_freezing"]


def test_each_condition_at_its_edge():
    assert failing(carried=(2, 3)) == (["benefit"], False)
    assert failing(carried=(2, 3), longest=(154, 193)) == (["benefit"], False), "39 of 784 is under 5 %"
    assert failing(carried=(2, 3), longest=(154, 194)) == ([], True), "40 of 784 is enough without a second carried track"
    assert failing(accompanied=(170, 163)) == ([], True), "7 fewer is within 1 % of 784"
    assert failing(accompanied=(170, 162)) == (["no_loss_of_accompaniment"], False)
    j = {"joins": 19, "consistent": 19, "switch": 0, "relay": 0, "undetermined": 0, "unassessed": 0, "supported_false_joins": 0}
    assert failing(joins=j) == (["joins_consistent", "joins_not_flagged", "joins_not_supported_false", "joins_scored"], False), "too few to score is not a pass"
    assert failing(joins={**j, "joins": 20, "consistent": 20}) == ([], True)
    assert failing(joins={"joins": 60, "unassessed": 20}) == ([], True), "unassessed joins are outside the shares"
    assert failing(joins={"consistent": 32, "relay": 7}) == ([], True), "32 of 40 is exactly four in five"
    assert failing(joins={"consistent": 31, "relay": 8}) == (["joins_consistent"], False)
    assert failing(joins={"switch": 2, "relay": 4}) == ([], True), "2 of 40 is exactly one in twenty"
    assert failing(joins={"switch": 2, "undetermined": 1, "relay": 3}) == (["joins_not_flagged"], False), "an undetermined join counts with the switches"
    assert failing(joins={"supported_false_joins": 2, "switch": 2, "relay": 4}) == (["joins_not_supported_false"], False), "1 of 40 is the most allowed"


def near_pooled(joins=None, base=None, **kwargs):
    p = pooled(**kwargs)
    p["recover"]["joins_in_finished_sector_tracks"]["jump_joins"] = 0
    j = {"all": 500, "jumps": 0, "links": 40, "kept": 32, "left": 5, "ended": 3, "undetermined": 0, "supported": 1}
    b = {"all": 3000, "jumps": 1400, "links": 400, "kept": 330, "left": 40, "ended": 30, "undetermined": 0, "supported": 8}
    j.update(joins or {}); b.update(base or {})
    p["earlier_side"] = {"near_joins": j, "released_12_hour_links": b}
    return p


def near_failing(**kwargs):
    d = V.decide_near(near_pooled(**kwargs))
    return sorted(k for k, v in d["checks"].items() if not v), d["carried_to_freezing"]


def test_section_22_adds_four_conditions_to_section_20s_six():
    d = V.decide_near(near_pooled())
    assert len(d["checks"]) == 10 and d["carried_to_freezing"] is True and d["earlier_side_joins"] == 40 and d["left_or_undetermined_joins"] == 5
    assert near_failing(carried=(2, 3)) == (["benefit"], False), "section 20's conditions still bind"
    assert near_failing(joins={"left": 5, "kept": 32, "ended": 3}) == ([], True), "5 of 40 is exactly 1.25 times 40 of 400"
    assert near_failing(joins={"left": 6, "kept": 31}) == (["left_not_above_production"], False)
    assert near_failing(joins={"left": 4, "undetermined": 2, "kept": 31}) == (["left_not_above_production"], False), "an undetermined join counts with the left"
    assert near_failing(base={"left": 38, "undetermined": 2, "kept": 330}) == ([], True), "and so does an undetermined production link"
    assert near_failing(joins={"supported": 1}) == ([], True), "1 of 40 is exactly 1.25 times 8 of 400"
    assert near_failing(joins={"supported": 2}) == (["supported_not_above_production"], False)
    assert near_failing(joins={"links": 19, "kept": 16, "left": 2, "ended": 1, "supported": 0}) == (
        ["earlier_side_scored", "left_not_above_production", "supported_not_above_production"], False), "too few to score is not a pass"
    assert near_failing(joins={"links": 20, "kept": 17, "left": 2, "ended": 1, "supported": 0}) == ([], True)
    assert near_failing(base={"links": 0, "kept": 0, "left": 0, "ended": 0, "supported": 0})[0] == ["earlier_side_scored", "left_not_above_production", "supported_not_above_production"]
    assert near_failing(joins={"jumps": 1}) == (["no_join_is_a_jump"], False)
    p = near_pooled(); p["recover"]["joins_in_finished_sector_tracks"]["jump_joins"] = 1
    assert V.decide_near(p)["checks"]["no_join_is_a_jump"] is False
    with pytest.raises(SystemExit, match="do not sum to their links"):
        V.decide_near(near_pooled(joins={"kept": 31}))
    with pytest.raises(SystemExit, match="do not sum to their links"):
        V.decide_near(near_pooled(base={"ended": 31}))


def test_the_rule_refuses_mismatched_populations_and_statuses():
    p = pooled(); p["recover"]["reference"]["steps"] = 783
    with pytest.raises(SystemExit, match="different reference observations"):
        V.decide(p)
    with pytest.raises(SystemExit, match="do not sum"):
        V.decide(pooled(joins={"joins": 41}))


# --- the control and the records ------------------------------------------------------------------

def frow(birth, steps, lons, hours, lat=10.0):
    return {"birth": birth, "fate": "finished", "finished_index": 0, "step": list(steps), "time": [D0 + STEP * k for k in steps], "lat": [lat] * len(steps),
            "lon": [float(x) for x in lons], "est": [[lat, float(x), h] for x, h in zip(lons[1:], hours)]}


def reference_files(tmp_path, monkeypatch, year=1990, declared=None, counted=(1, 0), tamper=None):
    import qtrack_pairing_pilot as M
    a = q(1, [10.0] * 4, [0.0, -2.0, -4.0, -6.0])
    monkeypatch.setattr(M, "load_qtrack", lambda path: ([a], None))
    qfile = tmp_path / "qtrack.nc"; qfile.write_text("q")
    art = tmp_path / f"art_{year}.json"
    art.write_text(json.dumps({"inputs": {"qtrack": {"path": str(qfile), "sha256": "0" * 64 if tamper == "qtrack" else E.C.sha256_of(str(qfile))}}}))
    rows = [frow("p", [0, 1, 3], [0.0, -2.0, -6.0], [6, 12]), frow("x", [0, 2], [0.5, 20.0], [12])]
    life = {"generated_by": {"sha256": "0" * 64 if tamper == "life" else L.SCRIPT_SHA256_AT_LOAD, "evaluation_sha256": E.SCRIPT_SHA256_AT_LOAD},
            "artifact": {"path": str(art), "sha256": "0" * 64 if tamper == "artifact" else E.C.sha256_of(str(art))}, "configs": {"released": {"rows": rows}}}
    life_path = tmp_path / f"continuity_lifecycle_{year}.json.gz"
    with gzip.open(life_path, "wt", encoding="utf-8") as fh:
        json.dump(life, fh)
    rec = {"generated_by": {"declared": declared or V.DECLARED["recover"]},
           "lifecycle": {"sha256": "0" * 64 if tamper == "link" else E.C.sha256_of(str(life_path))},
           "recover": {"rows": [frow("r", [0, 3], [0.0, -6.0], [18])], "joins_in_finished_sector_tracks": {"joins": counted[0], "jump_joins": counted[1]}}}
    with gzip.open(tmp_path / f"continuity_recovery_{year}.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(rec, fh)


def test_the_references_are_read_from_the_retained_histories(tmp_path, monkeypatch):
    for y in (1990, 2006):
        reference_files(tmp_path, monkeypatch, year=y)
    refs, inputs = V.earlier_side_references(str(tmp_path))
    zero = {"all": 0, "jumps": 0, "links": 0, "kept": 0, "left": 0, "ended": 0, "undetermined": 0, "supported": 0}
    assert refs == {"released_6_hour_links": {**zero, "all": 2, "links": 2, "kept": 2},
                    "released_12_hour_links": {**zero, "all": 4, "links": 4, "kept": 2, "left": 2},
                    "recover_joins": {**zero, "all": 2, "links": 2, "kept": 2}}
    assert sorted(inputs) == ["1990", "2006"] and len(inputs["1990"]["lifecycle"]) == 64 and inputs["1990"]["recover"] != inputs["1990"]["lifecycle"]
    one, _ = V.earlier_side_references(str(tmp_path), years=(1990,))
    assert one["released_12_hour_links"]["left"] == 1


@pytest.mark.parametrize("kwargs, message", [({"declared": "another section"}, "is not section 20's replay"), ({"tamper": "link"}, "is not section 20's replay"),
                                             ({"counted": (2, 0)}, "do not hold the joins its record counted"), ({"counted": (1, 1)}, "do not hold the joins its record counted"),
                                             ({"tamper": "qtrack"}, "is not the file the 1990 artifact names"), ({"tamper": "artifact"}, "is not the file the 1990 lifecycle record names"),
                                             ({"tamper": "life"}, "1990 lifecycle record was not written by the current scripts")])
def test_the_references_refuse_inputs_that_do_not_bind(tmp_path, monkeypatch, kwargs, message):
    reference_files(tmp_path, monkeypatch, **kwargs)
    with pytest.raises(SystemExit, match=message):
        V.earlier_side_references(str(tmp_path), years=(1990,))


def test_a_near_replay_with_a_far_join_is_refused():
    assert V.Recovery(**V.RULES["near"]).max_km == 500.0 and V.Recovery(**V.RULES["recover"]).max_km is None
    production = [frow("p", [0, 1], [0.0, -6.0], [6])]
    production[0]["est"] = [[10.0, 0.0, 6]]                                                          # a production link far from its prediction is not a join
    V.check_near(1990, production, 500.0)
    rows = [frow("r", [0, 3], [0.0, -6.0], [18])]
    rows[0]["est"] = [[10.0, 0.0, 18]]                                                               # claimed 6 degrees from its prediction
    V.check_near(1990, rows, None)                                                                   # RECOVER has no such condition
    V.check_near(1990, rows, 700.0)
    with pytest.raises(SystemExit, match="made a join 500 km or more from its prediction"):
        V.check_near(1990, rows, 500.0)


def test_the_control_must_be_production(monkeypatch):
    import pilot_margin_replay as MR
    rows = [{"birth": "0:0", "fate": "finished", "step": [0], "time": [D0], "lat": [10.0], "lon": [0.0], "est": [], "finished_index": 0},
            {"birth": "0:1", "fate": "pruned_in_loop", "step": [0], "time": [D0], "lat": [10.0], "lon": [100.0], "est": []}]
    life = {"configs": {"released": {"rows": [copy.deepcopy(rows[0])]}}}
    monkeypatch.setattr(MR, "archive_gate", lambda finished, run: {"passed": True})
    V.check_control(1990, [], rows, None, life)                       # the second history lies outside the retained box
    changed = copy.deepcopy(life); changed["configs"]["released"]["rows"][0]["lon"] = [0.5]
    with pytest.raises(SystemExit, match="not those of the lifecycle census"):
        V.check_control(1990, [], rows, None, changed)
    monkeypatch.setattr(MR, "archive_gate", lambda finished, run: {"passed": False})
    with pytest.raises(SystemExit, match="does not reproduce the stored campaign tracks"):
        V.check_control(1990, [], rows, None, life)


def season_record(year):
    ref = {"steps": 392, "tracks": 21, "accompanied": 90, "longest_single_history": 100, "tracks_accompanied": 12, "histories_on_accompanied": 24,
           "classes": {"carried": 2, "relayed": 1, "picked_up_late": 0, "dropped": 0, "unaccompanied_at_both": 0}}
    flags = {"flagged_tracks": 1, "display_classes": {k: 0 for k in E.DISPLAY_ORDER}, "events": {k: 1 for k in E.EVENT_KINDS}, "exposure_pairs": 10, "systems_left_behind": 1,
             "reference_tracks_with_a_flagged_history": 1, "switch_events_with_a_jump_link": 0, "supported_false_link_candidates": 0, "relay_events_with_a_jump_link": 0,
             "switch_events_by_elapsed": {k: 0 for k in E.ELAPSED}}
    costs = {"tracks_in_sector": 5, "links": 50, "jump_links": 5, "stationary_tracks": 1}
    link = {"6": {k: 1 for k in E.LINK_STATUS}, "18": {k: 2 for k in E.LINK_STATUS}}
    one = {"histories": 3, "reference": ref, "flags": flags, "costs": costs, "link_status": link}
    census = {"STORED": one, "KEEP": copy.deepcopy(one), "ALL": {"histories": 9, "reference": ref}, "breaks": {k: 1 for k in L.BREAKS},
              "chain_limits": {"0": {"longest_single_history": 1, "carried": 1}}, "chain_limits_through_any_history": {"0": {"longest_single_history": 1, "carried": 1,
              "histories_with_a_successor": 0, "successor_links": 0}}, "keep_cost": {"added_by_observations": {"2": 1}},
              "sample": {"observations": {"tracked": 1}, "STORED": {"tracked": 1}, "KEEP": {"tracked": 1}, "ALL": {"tracked": 1}}}
    counts = {k: 10 for k in V.COUNT_KEYS}
    joins = {"joins": 12, "consistent": 8, "switch": 0, "relay": 2, "undetermined": 0, "unassessed": 2, "supported_false_joins": 0, "jump_joins": 3, "by_hours": {"18": 7, "24": 5}}
    return {"year": year, "lifecycle": {"path": "x", "sha256": f"life{year}"}, "control": {"histories": 9}, "released_counts": {k: 7 for k in V.COUNT_KEYS},
            "recover": {"fates": {k: 2 for k in L.FATES}, "prune_checks": {}, "counts": counts, "census": census, "joins_made": {"all": 30, "by_hours": {"18": 20, "24": 10}},
                        "joins_in_finished_sector_tracks": joins, "joins": [], "rows": []}}


def lifecycle_pooled():
    s = season_record(0)["recover"]["census"]["STORED"]
    base = copy.deepcopy(s); base["reference"] = {**base["reference"], "steps": 784, "tracks": 42, "accompanied": 170, "longest_single_history": 154,
                                                    "classes": {**base["reference"]["classes"], "carried": 2}}
    return {"released": {"STORED": base}}


def test_pooling_sums_the_seasons_beside_the_released_baseline():
    p = V.pool([season_record(1990), season_record(2006)], lifecycle_pooled())
    assert p["recover"]["reference"]["steps"] == 784 and p["recover"]["reference"]["classes"]["carried"] == 4 and p["released"]["reference"]["accompanied"] == 170
    assert p["recover"]["joins_in_finished_sector_tracks"] == {"joins": 24, "consistent": 16, "switch": 0, "relay": 4, "undetermined": 0, "unassessed": 4,
                                                                "supported_false_joins": 0, "jump_joins": 6, "by_hours": {"18": 14, "24": 10}}
    assert p["recover"]["joins_made"] == {"all": 60, "by_hours": {"18": 40, "24": 20}} and p["recover"]["counts"]["tracks"] == 20 and p["released"]["counts"]["tracks"] == 14
    assert p["recover"]["link_status"]["18"]["consistent"] == 4 and p["recover"]["fates"]["finished"] == 4
    d = V.decide(p)
    assert d["checks"]["benefit"] is True and d["assessed_joins"] == 20 and d["carried_to_freezing"] is True


def write_records(tmp_path, bad_season=None, bad_life=False, life_seasons=None, rule="recover", season_rule=None):
    life = {"generated_by": {"sha256": "0" * 64 if bad_life else L.SCRIPT_SHA256_AT_LOAD, "evaluation_sha256": E.SCRIPT_SHA256_AT_LOAD,
                             "seasons": life_seasons or {"1990": "life1990", "2006": "life2006"}}, "pooled": lifecycle_pooled()}
    json.dump(life, open(tmp_path / "continuity_lifecycle.json", "w", encoding="utf-8"))
    for y in V.S.YEARS:
        stamp = {"script": "x", "sha256": V.SCRIPT_SHA256_AT_LOAD, "lifecycle_sha256": L.SCRIPT_SHA256_AT_LOAD, "evaluation_sha256": E.SCRIPT_SHA256_AT_LOAD,
                 "rule": season_rule or rule}
        if bad_season == y:
            stamp["lifecycle_sha256"] = "0" * 64
        record = season_record(y)
        if rule == "near":
            record["recover"]["joins_in_finished_sector_tracks"]["jump_joins"] = 0
            record["recover"]["joins_from_the_earlier_side"] = {"all": 12, "jumps": 0, "links": 10, "kept": 8, "left": 1, "ended": 1, "undetermined": 0, "supported": 0}
        with gzip.open(tmp_path / f"{V.STEM[rule]}_{y}.json.gz", "wt", encoding="utf-8") as fh:
            json.dump({"generated_by": stamp, **record}, fh)


def combine(tmp_path, rule="recover"):
    return V.main(["combine", "--out-dir", str(tmp_path), "--lifecycle-dir", str(tmp_path), "--rule", rule])


REFERENCES = {"released_6_hour_links": {"all": 900, "jumps": 30, "links": 200, "kept": 190, "left": 8, "ended": 2, "undetermined": 0, "supported": 0},
              "released_12_hour_links": {"all": 400, "jumps": 180, "links": 100, "kept": 80, "left": 12, "ended": 8, "undetermined": 0, "supported": 3},
              "recover_joins": {"all": 60, "jumps": 35, "links": 30, "kept": 18, "left": 9, "ended": 3, "undetermined": 0, "supported": 2}}


def test_combine_under_near_applies_section_22_beside_the_references(tmp_path, monkeypatch):
    write_records(tmp_path, rule="near")
    monkeypatch.setattr(V, "earlier_side_references", lambda d: (copy.deepcopy(REFERENCES), {"1990": {"lifecycle": "a", "recover": "b"}}))
    assert combine(tmp_path, "near") == 0
    rec = json.load(open(tmp_path / "continuity_recovery_near.json", encoding="utf-8"))
    assert rec["generated_by"]["rule"] == "near" and rec["generated_by"]["declared"].endswith("section 22") and rec["generated_by"]["earlier_side_inputs"]["1990"]["recover"] == "b"
    assert rec["pooled"]["earlier_side"]["near_joins"] == {"all": 24, "jumps": 0, "links": 20, "kept": 16, "left": 2, "ended": 2, "undetermined": 0, "supported": 0}
    assert len(rec["decision"]["checks"]) == 10 and rec["decision"]["carried_to_freezing"] is True, "2 of 20 against 12 of 100, under 1.25 times"
    text = (tmp_path / "continuity_recovery_near.md").read_text()
    assert text.startswith("# Recovery at 18 and 24 hours, NEAR (declared section 22)") and "| NEAR | 180 of 784 (23.0 %) | 4 |" in text
    assert "| NEAR, joins | 24 | 0 | 20 | 16 | 2 | 2 | 0 | 0 |" in text and "| released, 12-hour links | 400 | 180 | 100 | 80 | 12 | 8 | 0 | 3 |" in text
    assert "| RECOVER, joins | 60 | 35 | 30 | 18 | 9 | 3 | 0 | 2 |" in text and "left not above production met" in text and "Histories by fate under NEAR" in text
    assert not os.path.exists(tmp_path / "continuity_recovery.json"), "each rule writes under its own name"
    with pytest.raises(SystemExit, match="never overwritten"):
        combine(tmp_path, "near")


def test_combine_refuses_season_records_of_the_other_rule(tmp_path, monkeypatch):
    write_records(tmp_path, rule="near", season_rule="recover")
    monkeypatch.setattr(V, "earlier_side_references", lambda d: (copy.deepcopy(REFERENCES), {}))
    with pytest.raises(SystemExit, match="under this rule"):
        combine(tmp_path, "near")


def test_combine_writes_the_tables_and_never_overwrites(tmp_path):
    write_records(tmp_path)
    assert combine(tmp_path) == 0
    rec = json.load(open(tmp_path / "continuity_recovery.json", encoding="utf-8"))
    assert rec["decision"]["carried_to_freezing"] is True and rec["decision"]["carried_rise"] == 2 and "rows" not in json.dumps(rec["seasons"])
    text = (tmp_path / "continuity_recovery.md").read_text()
    assert "| RECOVER | 180 of 784 (23.0 %) | 4 |" in text and "| released | 170 of 784 (21.7 %) | 2 |" in text
    assert text.startswith("# Recovery at 18 and 24 hours, RECOVER (declared section 20)") and "read from the earlier side" not in text
    assert "Joins made over the whole domain: 60 (40 at 18 hours, 20 at 24 hours). In finished tracks in the sector: 24 (14 at 18 hours, 10 at 24 hours)" in text
    assert "consistent 16, switch 0, relay 4, undetermined 0, unassessed 4, supported false joins 0" in text and "Carried to freezing: yes" in text
    assert "| RECOVER, 18 | 4 | 4 | 4 | 4 | 4 |" in text and "- 1990: 90 of 392, 2, 100; 8, 0, 2, 0, 2" in text
    with pytest.raises(SystemExit, match="never overwritten"):
        combine(tmp_path)


def test_combine_refuses_records_from_other_scripts_or_other_lifecycle_records(tmp_path):
    write_records(tmp_path, bad_season=2006)
    with pytest.raises(SystemExit, match="2006 season record was not written by the current scripts under this rule"):
        combine(tmp_path)
    for name in os.listdir(tmp_path):
        os.remove(tmp_path / name)
    write_records(tmp_path, bad_life=True)
    with pytest.raises(SystemExit, match="combined lifecycle record was not written"):
        combine(tmp_path)
    for name in os.listdir(tmp_path):
        os.remove(tmp_path / name)
    write_records(tmp_path, life_seasons={"1990": "life1990", "2006": "another"})
    with pytest.raises(SystemExit, match="read different lifecycle records"):
        combine(tmp_path)


def test_season_records_are_never_overwritten_and_other_seasons_are_refused(tmp_path):
    (tmp_path / "continuity_recovery_1990.json.gz").write_text("x")
    with pytest.raises(SystemExit, match="never overwritten"):
        V.main(["season", "--year", "1990", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="not a development season"):
        V.run_season(1984)
    (tmp_path / "continuity_recovery_near_2006.json.gz").write_text("x")
    with pytest.raises(SystemExit, match="continuity_recovery_near_2006.json.gz exists"):
        V.main(["season", "--year", "2006", "--out-dir", str(tmp_path), "--rule", "near"])
