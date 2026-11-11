"""Tests for scripts/continuity_lifecycle.py (`CONTINUITY_ACROSS_AFRICA_2026-11-09.md`, section 18).

The first test is the constructed case section 18 names for expired predictions: a track kept in
the live list is left unchanged while a candidate at its position 18 hours on seeds a new track."""

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
import continuity_lifecycle as L  # noqa: E402
from aew.v1port.association import predict_displacement  # noqa: E402

D0 = 7000.0
STEP = 0.25


def wave(lat, lon, time, half=1.0, n=5, region=True):
    lats = np.repeat(np.linspace(-half, half, n), n) + lat
    lons = np.tile(np.linspace(-half, half, n), n) + lon
    out = {"time": time, "lat_mean": float(lat), "lon_mean": float(lon), "lat_wave": lats, "lon_wave": lons}
    if region:
        out["region"] = np.zeros((1, 1), bool)
    return out


def still(_):
    return 0.0


def westward(_):
    return -8.0


def far(k):
    return wave(-30.0, 130.0, k * STEP)


def run(schedule, u=still, exclusive=False, **kwargs):
    times = [k * STEP for k in range(len(schedule))]
    return L.registered_loop(times, lambda k: (schedule[k], u, still), exclusive, **kwargs)


def by_birth(rows):
    return {r["birth"]: r for r in rows}


# --- the registered loop ---------------------------------------------------------------

def test_a_track_kept_past_its_two_attempts_is_not_continued():
    schedule = [[wave(10.0, 0.0, 0.0)], [far(1)], [far(2)], [wave(10.0, 0.0, 3 * STEP), far(3)], [far(4)], [far(5)], [far(6)],
                [far(7), wave(40.0, 40.0, 7 * STEP)]]
    finished, rows, checks = run(schedule)
    b = by_birth(rows)
    assert b["7:1"]["step"] == [7] and b["7:1"]["fate"] == "removed_as_single_observation", "seeded at the last step, never pruned"
    assert b["0:0"]["step"] == [0] and b["0:0"]["fate"] == "pruned_in_loop", "still in the live list at 18 hours, and unchanged"
    assert b["3:0"]["step"] == [3] and b["3:0"]["fate"] == "pruned_in_loop", "the candidate at its position seeded a new track"
    assert b["1:0"]["step"] == [1, 2, 3, 4, 5, 6, 7] and b["1:0"]["fate"] == "removed_by_final_speed_filter"
    assert finished == [] and checks == {"pruned_with_both_attempts_spent": 2, "pruned_with_an_attempt_stranded": 0}


def test_each_link_records_the_prediction_it_was_matched_under():
    _, dlon6 = predict_displacement(-8.0, 0.0, 10.0, 6.0)
    _, dlon12 = predict_displacement(-8.0, 0.0, 10.0, 12.0)
    schedule = [[wave(10.0, 0.0, 0.0)], [wave(10.0, dlon6, STEP)], [far(2)], [wave(10.0, dlon6 + dlon12, 3 * STEP, half=0.2), far(3)]]
    _, rows, _ = run(schedule, u=westward)
    r = by_birth(rows)["0:0"]
    assert r["step"] == [0, 1, 3]
    assert r["est"] == [[10.0, round(dlon6, 3), 6], [10.0, round(dlon6 + dlon12, 3), 12]], "the six-hour and then the twelve-hour prediction"
    assert [round(x, 3) for x in r["lon"][1:]] == [e[1] for e in r["est"]], "each claimed exactly where predicted"


def test_a_step_without_candidates_strands_an_attempt():
    schedule = [[wave(10.0, 0.0, 0.0)], []] + [[far(k)] for k in range(2, 8)]
    _, rows, checks = run(schedule)
    assert by_birth(rows)["0:0"]["fate"] == "pruned_in_loop"
    assert checks == {"pruned_with_both_attempts_spent": 0, "pruned_with_an_attempt_stranded": 1}


def test_a_prune_with_an_attempt_still_due_is_refused(monkeypatch):
    import aew.v1port.association as assoc
    monkeypatch.setattr(assoc, "prune_stale_tracks", lambda tracks, states, *a, **k: (tracks[:-1], states[:-1]))
    with pytest.raises(SystemExit, match="attempt still due"):
        run([[wave(10.0, 0.0, 0.0)], [far(1)]])


def test_the_exclusive_flag_reaches_the_production_association():
    schedule = [[wave(10.0, 0.0, 0.0), wave(10.0, 0.5, 0.0)], [wave(10.0, 0.2, STEP)]]
    assert sorted(len(r["step"]) for r in run(schedule)[1]) == [2, 2], "version 1 lets two tracks claim one candidate"
    assert sorted(len(r["step"]) for r in run(schedule, exclusive=True)[1]) == [1, 2]


def test_the_hook_sees_each_step_before_the_prune_and_the_recent_clause_can_be_widened():
    schedule = [[wave(10.0, 0.0, 0.0)]] + [[far(k)] for k in range(1, 9)]
    seen = []

    def hook(step, time, waves, winds, tracks, states, registry, order, seeded):
        seen.append((step, time, len(waves), len(tracks), [registry[id(t)]["birth"] for t in seeded], len(order)))
        assert winds[0] is still and len(states) == len(tracks)
        return tracks, states

    _, rows, _ = run(schedule, hook=hook)
    assert seen[0] == (0, 0.0, 1, 1, ["0:0"], 1) and seen[1] == (1, STEP, 1, 2, ["1:0"], 2) and seen[2][4] == []
    assert seen[7][3] == 2 and seen[8][3] == 1, "the stale track is still listed at the eighth step's hook and pruned after it"
    assert by_birth(rows)["0:0"]["fate"] == "pruned_in_loop"
    _, rows, _ = run(schedule, recent_days=2.5)
    assert by_birth(rows)["0:0"]["fate"] == "removed_as_single_observation", "seen within 2.5 days at every prune, so never pruned"
    with pytest.raises(SystemExit, match="left the live list without a fate"):   # what the hook returns is what the loop goes on with
        run(schedule, hook=lambda tracks, states, **k: (tracks[1:], states[1:]) if k["step"] == 3 else (tracks, states))


def test_a_seed_that_cannot_be_traced_to_its_candidate_is_refused():
    with pytest.raises(SystemExit, match="cannot be traced"):
        run([[wave(10.0, 0.0, 0.0, region=False)]])


def moving_schedule(n=10):
    _, dlon = predict_displacement(-8.0, 0.0, 10.0, 6.0)
    return [[wave(10.0 + 0.3 * (k % 2), k * dlon, k * STEP)] for k in range(n)]


def test_a_finished_history_is_rebuilt_from_its_row_exactly():
    finished, rows, checks = run(moving_schedule(), u=westward)
    assert len(finished) == 1 and rows[0]["fate"] == "finished" and rows[0]["finished_index"] == 0 and len(rows[0]["step"]) == 10
    assert checks == {"pruned_with_both_attempts_spent": 0, "pruned_with_an_attempt_stranded": 0}
    sets = L.history_sets(rows)
    out = [{"time": np.asarray(f["time"], float), "lat": np.asarray(f["meanlat"], float), "lon": np.asarray(f["meanlon"], float)} for f in finished]
    L.check_stored("fixture", sets, rows, out)
    assert not np.array_equal(sets["STORED"][0]["lat"], np.asarray(rows[0]["lat"])), "stored positions are smoothed"
    assert sets["ALL"][0]["lat"].tolist() == rows[0]["lat"], "ALL keeps the unsmoothed positions"
    out[0]["lat"] = out[0]["lat"] + 1e-9
    with pytest.raises(SystemExit, match="not stored as the replay finished it"):
        L.check_stored("fixture", sets, rows, out)


# --- the sets ---------------------------------------------------------------------------

def row(birth, steps, lons, fate, lat=10.0, est=None):
    lats = [lat] * len(steps)
    return {"birth": birth, "fate": fate, "step": list(steps), "time": [D0 + STEP * s for s in steps], "lat": lats, "lon": [float(x) for x in lons],
            "est": est if est is not None else [[lat, float(x), 6] for x in lons[1:]], **({"finished_index": 0} if fate == "finished" else {})}


def test_keep_adds_only_pruned_histories_that_pass_the_final_filter():
    rows = [row("f", range(4), [20.0, 18.0, 16.0, 14.0], "finished"),
            row("fast", range(6, 9), [8.0, 6.0, 4.0], "pruned_in_loop"),
            row("single", [9], [2.0], "pruned_in_loop"),
            row("slow", range(10, 13), [0.0, 0.01, 0.0], "pruned_in_loop"),
            row("filtered", range(20, 30), [5.0] * 10, "removed_by_final_speed_filter")]
    sets = L.history_sets(rows)
    assert [h["id"] for h in sets["STORED"]] == ["f"]
    assert [h["id"] for h in sets["KEEP"]] == ["f", "fast"], "a single observation and a track under 2 m/s are not kept"
    assert [h["id"] for h in sets["ALL"]] == ["f", "fast", "single", "slow", "filtered"]


def test_a_finished_row_that_fails_the_filter_is_refused():
    with pytest.raises(SystemExit, match="does not pass the final filter"):
        L.history_sets([row("f", [0], [20.0], "finished")])


def test_the_retained_box_is_the_sector_with_its_margin():
    assert L.in_box(row("a", [0, 1], [46.0, 47.0], "pruned_in_loop")) and not L.in_box(row("b", [0, 1], [46.1, 47.0], "pruned_in_loop"))
    assert L.in_box(row("c", [0], [0.0], "pruned_in_loop", lat=41.0)) and not L.in_box(row("d", [0], [0.0], "pruned_in_loop", lat=41.1))
    assert L.in_box(row("e", [0], [-46.0], "pruned_in_loop", lat=-6.0)) and not L.in_box(row("g", [0], [-46.1], "pruned_in_loop", lat=0.0))


# --- breaks and chains ------------------------------------------------------------------

def h(hid, first, last, lon_first=0.0, lon_last=0.0):
    return {"id": hid, "step": list(range(first, last + 1)), "raw_lat": [10.0] * (last - first + 1),
            "raw_lon": [lon_first] + [lon_last] * (last - first)}


def test_break_classes_at_their_edges():
    a = h("a", 0, 10)
    for first, name in ((9, "concurrent"), (10, "concurrent"), (11, "within_horizon"), (12, "within_horizon"), (13, "18_to_24_hours"),
                        (14, "18_to_24_hours"), (15, "30_to_48_hours"), (18, "30_to_48_hours"), (19, "over_48_hours")):
        assert L.break_class(a, h("b", first, first + 2)) == name, first


def test_breaks_count_changes_of_the_nearest_history_across_unaccompanied_gaps():
    by_id = {"a": h("a", 0, 3), "b": h("b", 6, 8), "c": h("c", 2, 12)}
    rows = [{"nearest": ["a", "a", None, None, "b", "b", "c", "b"]}, {"nearest": [None, "a"]}]
    assert L.breaks(rows, by_id) == {"concurrent": 2, "within_horizon": 0, "18_to_24_hours": 1, "30_to_48_hours": 0, "over_48_hours": 0}


def chain_case():
    by_id = {"a": h("a", 0, 3, 0.0, -6.0), "b": h("b", 6, 9, -12.0, -18.0), "c": h("c", 14, 17, -28.0, -34.0)}
    r = {"by_history": {"a": [0, 1, 2, 3], "b": [6, 7, 8, 9], "c": [14, 15, 16, 17]}, "start": ["a"], "event": ["c"], "outcome": "atlantic_side"}
    return r, by_id


def test_chaining_joins_histories_only_within_the_horizon():
    r, by_id = chain_case()
    assert L.chained(r, by_id, 0) == (4, False)
    assert L.chained(r, by_id, 18) == (8, False), "a to b is 18 hours, b to c is 30"
    assert L.chained(r, by_id, 24) == (8, False)
    assert L.chained(r, by_id, 30) == (12, True)
    assert L.chain_limits([r], by_id) == {"0": {"longest_single_history": 4, "carried": 0}, "18": {"longest_single_history": 8, "carried": 0},
                                           "24": {"longest_single_history": 8, "carried": 0}, "36": {"longest_single_history": 12, "carried": 1},
                                           "48": {"longest_single_history": 12, "carried": 1}}


def test_chaining_never_adds_a_link_the_production_horizon_already_covers():
    by_id = {"a": h("a", 0, 3), "b": h("b", 5, 8), "c": h("c", 4, 4)}
    r = {"by_history": {"a": [0, 1, 2, 3], "b": [5, 6, 7, 8], "c": [4]}, "start": ["a"], "event": ["b"], "outcome": "atlantic_side"}
    assert L.chained(r, by_id, 48) == (4, False), "gaps of 6 and 12 hours were the production association's to make"


def test_chaining_respects_the_speed_maximum_and_the_outcome():
    r, by_id = chain_case()
    by_id["b"] = h("b", 6, 9, -21.0, -27.0)                          # about 1,640 km from a's end in 18 hours, over 25 m/s
    assert L.chained(r, by_id, 18) == (4, False)
    by_id["b"] = h("b", 6, 9, -20.0, -18.0)                          # about 1,530 km, under it
    assert L.chained(r, by_id, 18)[0] == 8
    r, by_id = chain_case()
    r["outcome"] = "inland"
    assert L.chained(r, by_id, 48) == (12, None), "the carried class is for Atlantic-side tracks only"


def test_a_chain_may_pass_through_a_history_that_is_not_beside_the_reference_track():
    a, x, c = h("a", 0, 3), h("x", 6, 8), h("c", 12, 15)            # x never comes near the reference track
    lone = h("lone", 30, 31)
    r = {"by_history": {"a": [0, 1, 2, 3], "c": [12, 13, 14, 15]}, "start": ["a"], "event": ["c"], "outcome": "atlantic_side"}
    by_id = {"a": a, "c": c, "x": x}
    assert L.chained(r, by_id, 48) == (4, False), "a to c is 54 hours when only the histories beside the track may be chained"
    got = L.open_chain_limits([r], [c, lone, x, a])
    assert got["0"] == {"longest_single_history": 4, "carried": 0, "histories_with_a_successor": 0, "successor_links": 0}
    assert got["18"] == {"longest_single_history": 4, "carried": 0, "histories_with_a_successor": 1, "successor_links": 1}, "a to x only"
    assert got["24"] == {"longest_single_history": 8, "carried": 1, "histories_with_a_successor": 2, "successor_links": 2}, "x to c is 24 hours"
    assert got["48"] == got["24"]


def test_the_best_chain_is_taken_when_a_history_has_two_successors():
    by_id = {"a": h("a", 0, 3), "b": h("b", 6, 6), "c": h("c", 6, 9)}
    r = {"by_history": {"a": [0, 1, 2, 3], "b": [6], "c": [6, 7, 8, 9]}, "start": ["a"], "event": ["b"], "outcome": "atlantic_side"}
    assert L.chained(r, by_id, 18) == (8, True), "the longest chain and the carried chain may differ"
    got = L.open_chain_limits([r], list(by_id.values()))["18"]
    assert (got["histories_with_a_successor"], got["successor_links"]) == (1, 2)


# --- one season's census on a constructed season ------------------------------------------

REF_LONS = [20.0 - 2.0 * k for k in range(12)]


def season_fixture():
    """One reference track, four histories beside it: two finished, one pruned of three
    observations and one pruned single, with gaps of 18 hours between the first three."""
    ref = {"id": 9, "time": D0 + STEP * np.arange(12), "lat": np.full(12, 10.0), "lon": np.asarray(REF_LONS), "n_valid": 12}
    q_rows = [{"id": 9, "first": {"lon": 20.0}, "outcome": "atlantic_side", "first_atlantic_side": {"index": 11}, "last_used": {"index": 11}}]
    rows = [row("0:0", range(0, 4), REF_LONS[0:4], "finished"),
            row("6:0", range(6, 9), REF_LONS[6:9], "pruned_in_loop"),
            row("9:0", [9], REF_LONS[9:10], "pruned_in_loop"),
            row("11:0", [11, 12], [REF_LONS[11], REF_LONS[11] - 2.0], "finished")]
    classes = {**{k: "tracked" for k in (0, 1, 2, 3, 11)}, **{k: "candidate_not_tracked" for k in (6, 7, 8, 9)}, **{k: "placed_away" for k in (4, 5, 10)}}
    sample = {"tracks": [{"qtrack_id": 9, "steps": [{"step": k, "lat": 10.0, "lon": REF_LONS[k], "class": classes[k]} for k in range(12)]}]}
    readings = {"association": [{"qtrack_id": 9, "step": k, "holder": "6:0", "fate": "pruned_in_loop", "n_obs": 3} for k in (6, 7, 8)]
                + [{"qtrack_id": 9, "step": 9, "holder": "9:0", "fate": "pruned_in_loop", "n_obs": 1}]}
    return rows, [ref], q_rows, sample, readings, [D0 + STEP * k for k in range(14)]


def test_the_census_of_a_constructed_season():
    rows, qs, q_rows, sample, readings, times = season_fixture()
    c, sets = L.season_census(rows, qs, q_rows, sample, readings, times)
    assert [c[s]["reference"]["accompanied"] for s in L.SETS] == [5, 8, 9]
    assert [c[s]["reference"]["longest_single_history"] for s in L.SETS] == [4, 4, 4]
    assert c["STORED"]["reference"]["classes"]["relayed"] == 1 and c["ALL"]["reference"]["histories_on_accompanied"] == 4
    assert c["breaks"] == {"concurrent": 0, "within_horizon": 2, "18_to_24_hours": 1, "30_to_48_hours": 0, "over_48_hours": 0}
    assert c["chain_limits"]["0"] == {"longest_single_history": 4, "carried": 0}
    assert c["chain_limits"]["18"] == {"longest_single_history": 8, "carried": 1}, "0:0 to 6:0 to 11:0, each 18 hours apart"
    assert c["chain_limits_through_any_history"]["18"] == {"longest_single_history": 8, "carried": 1, "histories_with_a_successor": 2, "successor_links": 2}
    assert c["chain_limits_through_any_history"]["0"]["carried"] == 0
    assert c["STORED"]["link_status"] == {"6": {"consistent": 3, "switch": 0, "relay": 0, "undetermined": 0, "unassessed": 1}}, "the last link ends past the reference track"
    assert c["sample"] == {"observations": {"tracked": 5, "candidate_not_tracked": 4, "placed_away": 3},
                           "STORED": {"tracked": 5, "candidate_not_tracked": 0, "placed_away": 0},
                           "KEEP": {"tracked": 5, "candidate_not_tracked": 3, "placed_away": 0},
                           "ALL": {"tracked": 5, "candidate_not_tracked": 4, "placed_away": 0}}
    assert c["holders"] == [{"qtrack_id": 9, "holder": "6:0", "n_obs": 3, "span_steps": 3, "end": "candidate_present"},
                            {"qtrack_id": 9, "holder": "9:0", "n_obs": 1, "span_steps": 1, "end": "candidate_present"}]
    assert c["keep_cost"] == {"stored_in_sector": 2, "stored_ever_accompanied": 2, "added_in_sector": 1, "added_ever_accompanied": 1,
                              "added_sharing_an_observation": 0, "added_by_observations": {"3": 1}}
    assert c["KEEP"]["costs"]["tracks_in_sector"] == 3 and c["KEEP"]["flags"]["exposure_pairs"] == 3 + 2 + 0
    assert "flags" not in c["ALL"] and len(sets["ALL"]) == 4


def test_a_reference_observation_the_retained_box_does_not_cover_is_refused():
    rows, qs, q_rows, sample, readings, times = season_fixture()
    qs[0]["lon"] = qs[0]["lon"] + 25.0                               # the first used observation is now at 45 E
    with pytest.raises(SystemExit, match="does not cover"):
        L.season_census(rows, qs, q_rows, sample, None, times)


def test_coverage_is_the_box_less_500_km_on_every_side():
    assert L.covered(-0.5, 22.58), "just south of the sector, well inside the retained box"
    assert L.covered(-1.5, 0.0) and not L.covered(-1.6, 0.0), "the box reaches 6 S and 500 km is about 4.5 degrees"
    assert L.covered(36.5, 0.0) and not L.covered(36.6, 0.0)
    assert L.covered(10.0, 41.3) and not L.covered(10.0, 41.5), "about 4.6 degrees of longitude at 14.5 N"
    assert L.covered(10.0, -41.3) and not L.covered(10.0, -41.5)
    assert L.covered(30.0, 40.0) and not L.covered(30.0, 40.6), "the same distance spans more longitude farther north"


def test_the_stored_set_must_accompany_the_sample_as_it_recorded():
    rows, qs, q_rows, sample, readings, times = season_fixture()
    sample["tracks"][0]["steps"][4]["class"] = "tracked"              # the sample says tracked where no stored track is near
    with pytest.raises(SystemExit, match="as the stage sample records"):
        L.season_census(rows, qs, q_rows, sample, readings, times)
    c, _ = L.season_census(rows, qs, q_rows, sample, None, times)
    assert "holders" not in c and c["sample"]["STORED"]["tracked"] == 5, "a second configuration is described without the binding"


def test_holder_ends_and_their_refusals():
    rows, _, _, sample, readings, _ = season_fixture()
    steps = sample["tracks"][0]["steps"]
    steps[9]["class"] = "placed_away"; steps[10]["class"] = "placed_away"
    assert [x["end"] for x in L.holder_ends(rows, sample, readings)] == ["placed_away", "candidate_present"], "the second meets a tracked step at its second attempt"
    steps[11]["class"] = "placed_away"
    assert [x["end"] for x in L.holder_ends(rows, sample, readings)] == ["placed_away", "placed_away"]
    steps[10]["class"] = "tracked"
    assert [x["end"] for x in L.holder_ends(rows, sample, readings)] == ["candidate_present", "candidate_present"]
    late = [row("11:0", [11, 12], [REF_LONS[11], REF_LONS[11] - 2.0], "pruned_in_loop")]
    ends = L.holder_ends(late, sample, {"association": [{"qtrack_id": 9, "step": 11, "holder": "11:0", "fate": "pruned_in_loop", "n_obs": 2},
                                                         {"qtrack_id": 9, "step": 3, "holder": "x", "fate": "finished"}]})
    assert ends == [{"qtrack_id": 9, "holder": "11:0", "n_obs": 2, "span_steps": 2, "end": "beyond_reference"}], "a finished holder is not listed"
    bad = copy.deepcopy(readings); bad["association"][0]["n_obs"] = 4
    with pytest.raises(SystemExit, match="is not reproduced by the replay"):
        L.holder_ends(rows, sample, bad)
    bad = copy.deepcopy(readings); bad["association"][0]["holder"] = "5:5"
    with pytest.raises(SystemExit, match="is not reproduced by the replay"):
        L.holder_ends(rows, sample, bad)
    with pytest.raises(SystemExit, match="is not reproduced by the replay"):
        L.holder_ends(rows[1:2], sample, {"association": bad["association"][:1]})   # even when another pruned history of that length exists


def test_keep_cost_counts_shared_observations_and_companions():
    rows = [row("f", range(4), REF_LONS[0:4], "finished"),
            row("twin", range(1, 4), REF_LONS[1:4], "pruned_in_loop"),          # holds the stored track's own observations
            row("alone", range(0, 3), [30.0, 28.0, 26.0], "pruned_in_loop", lat=30.0),
            row("partly", range(0, 4), [20.5, 18.5, 30.0, 32.0], "pruned_in_loop"),  # beside the reference track at two observations only
            row("outside", range(0, 3), [45.0, 44.0, 43.0], "pruned_in_loop")]
    _, qs, _, _, _, _ = season_fixture()
    k = L.keep_cost(L.history_sets(rows), L.C.index_by_time(qs))
    assert k == {"stored_in_sector": 1, "stored_ever_accompanied": 1, "added_in_sector": 3, "added_ever_accompanied": 2,
                 "added_sharing_an_observation": 1, "added_by_observations": {"3": 2, "4": 1}}


# --- bindings to the false-join test -------------------------------------------------------

def jconf_for(c):
    ref, flags, costs = c["STORED"]["reference"], c["STORED"]["flags"], c["STORED"]["costs"]
    return {"counts": {"accompanied_steps": ref["accompanied"], "qtrack_steps": ref["steps"], "qtrack_classes": dict(ref["classes"]),
                       "tracks_in_sector": costs["tracks_in_sector"], "stationary_tracks": costs["stationary_tracks"]},
            "events": dict(flags["events"]), "classes": {k: flags["display_classes"][k] for k in L.J.CLASS_ORDER}}


def test_the_stored_set_is_bound_to_the_false_join_record():
    rows, qs, q_rows, sample, readings, times = season_fixture()
    c, _ = L.season_census(rows, qs, q_rows, sample, readings, times)
    L.bind("fixture", c, jconf_for(c))
    for path, message in ((("counts", "accompanied_steps"), "reference measures"), (("counts", "qtrack_steps"), "reference measures"),
                          (("counts", "tracks_in_sector"), "sector counts"), (("counts", "stationary_tracks"), "sector counts"),
                          (("events", "switch"), "transitions"), (("classes", "no_transition"), "classes")):
        j = jconf_for(c); j[path[0]][path[1]] += 1
        with pytest.raises(SystemExit, match=message):
            L.bind("fixture", c, j)
    j = jconf_for(c); j["counts"]["qtrack_classes"]["carried"] += 1
    with pytest.raises(SystemExit, match="reference measures"):
        L.bind("fixture", c, j)
    c2 = copy.deepcopy(c); c2["STORED"]["flags"]["events"]["undetermined"] = 1
    j = jconf_for(c2); j["classes"]["relay"] += 1
    L.bind("fixture", c2, j)                                         # with an undetermined transition the two class orders may differ


def test_the_replay_must_reproduce_the_retained_tracks():
    conf = {"tracks": [{"id": 0, "time": [1.0], "lat": [2.0], "lon": [3.0]}], "classes": {"switch": 1}, "counts": {"tracks": 1}}
    rec = {"configs": {n: copy.deepcopy(conf) for n in L.CONFIGS}}
    L.check_retained(1990, rec, copy.deepcopy(rec))
    for key, value in (("tracks", []), ("classes", {"switch": 2}), ("counts", {"tracks": 2})):
        other = copy.deepcopy(rec); other["configs"]["VCLIP"][key] = value
        with pytest.raises(SystemExit, match="does not reproduce the tracks the false-join test retained"):
            L.check_retained(1990, rec, other)


# --- pooling, the marks and the records -------------------------------------------------------

def season_record(year):
    rows, qs, q_rows, sample, readings, times = season_fixture()
    released, _ = L.season_census(rows, qs, q_rows, sample, readings, times)
    vclip, _ = L.season_census(rows, qs, q_rows, sample, None, times)
    one = {"fates": {k: 1 for k in L.FATES}, "prune_checks": {"pruned_with_both_attempts_spent": 2, "pruned_with_an_attempt_stranded": 1},
           "histories_retained_in_record": 4, "rows": rows}
    return {"year": year, "configs": {"released": {**one, "census": released}, "VCLIP": {**one, "census": vclip}}}


def test_pooling_sums_the_seasons_and_counts_holders_once():
    p = L.pool([season_record(1990), season_record(2006)])
    r = p["released"]
    assert r["STORED"]["reference"]["accompanied"] == 10 and r["KEEP"]["reference"]["steps"] == 24 and r["ALL"]["reference"]["classes"]["relayed"] == 2
    assert r["breaks"]["within_horizon"] == 4 and r["chain_limits"]["24"] == {"longest_single_history": 16, "carried": 2}
    assert r["sample"]["KEEP"]["candidate_not_tracked"] == 6 and r["keep_cost"]["added_by_observations"] == {"3": 2}
    assert r["fates"]["pruned_in_loop"] == 2 and r["prune_checks"] == {"pruned_with_both_attempts_spent": 4, "pruned_with_an_attempt_stranded": 2}
    assert r["holders"] == {"pairs": 4, "distinct": 4, "by_observations": {"1": 2, "3": 2}, "ends_by_pair": {"candidate_present": 4, "placed_away": 0, "beyond_reference": 0}}
    assert "holders" not in p["VCLIP"] and p["VCLIP"]["STORED"]["flags"]["exposure_pairs"] == 2 * (3 + 0)


def marks(steps, stored, keep, carried0, carried24, longest0, longest24, through=None):
    one = {"STORED": {"reference": {"steps": steps, "accompanied": stored}}, "KEEP": {"reference": {"steps": steps, "accompanied": keep}},
           "ALL": {"reference": {"steps": steps, "accompanied": keep}},
           "chain_limits": {"0": {"carried": carried0, "longest_single_history": longest0}, "24": {"carried": carried24, "longest_single_history": longest24}}}
    if through is not None:
        one["chain_limits_through_any_history"] = {"0": {"carried": carried0, "longest_single_history": longest0},
                                                   "24": {"carried": through[0], "longest_single_history": through[1]}}
    return {"released": one, "VCLIP": copy.deepcopy(one)}


def test_the_bar_is_read_again_on_the_limit_through_any_history():
    d = L.decide(marks(784, 170, 338, 2, 3, 226, 261, through=(6, 266)))
    assert d["recovery"]["developed_further"] is False, "the declared limit is unchanged"
    assert d["recovery_through_any_history"] == {"horizon_hours": 24, "carried_rise": 4, "longest_rise": 40, "longest_share_rise": 0.051, "bar_met": True}
    assert L.decide(marks(784, 170, 338, 2, 3, 226, 261, through=(3, 265)))["recovery_through_any_history"]["bar_met"] is False
    assert L.decide(marks(784, 170, 338, 2, 3, 226, 261, through=(3, 266)))["recovery_through_any_history"]["bar_met"] is True, "40 observations alone"
    assert L.decide(marks(784, 170, 338, 2, 3, 226, 261, through=(4, 226)))["recovery_through_any_history"]["bar_met"] is True, "two carried alone"
    assert "recovery_through_any_history" not in L.decide(marks(784, 170, 338, 2, 3, 226, 261))


def test_the_two_marks_at_their_edges():
    d = L.decide(marks(140, 30, 37, 2, 3, 50, 56))
    assert d["keep"] == {"accompanied_rise": 7, "share_rise": 0.05, "mark_met": True}, "7 of 140 is exactly 5 points"
    assert d["recovery"]["developed_further"] is False and d["recovery"]["carried_rise"] == 1 and d["recovery"]["longest_rise"] == 6
    assert L.decide(marks(140, 30, 36, 2, 3, 50, 56))["keep"]["mark_met"] is False
    assert L.decide(marks(140, 30, 36, 2, 4, 50, 50))["recovery"]["developed_further"] is True, "two more carried"
    assert L.decide(marks(140, 30, 36, 2, 2, 50, 57))["recovery"]["developed_further"] is True, "7 of 140 more on the longest accompaniment"
    assert L.decide(marks(784, 170, 209, 2, 3, 100, 139))["recovery"]["developed_further"] is False, "39 of 784 is under 5 %"
    assert L.decide(marks(784, 170, 210, 2, 3, 100, 140))["keep"]["mark_met"] is True


def test_the_marks_are_read_on_the_released_configuration_and_one_population():
    m = marks(140, 30, 30, 2, 2, 50, 50)
    m["VCLIP"]["KEEP"]["reference"]["accompanied"] = 130; m["VCLIP"]["chain_limits"]["24"]["carried"] = 9
    d = L.decide(m)
    assert d["keep"]["mark_met"] is False and d["recovery"]["developed_further"] is False
    m["VCLIP"]["ALL"]["reference"]["steps"] = 141
    with pytest.raises(SystemExit, match="different reference observations"):
        L.decide(m)


def write_season(tmp_path, year, sha=None, esha=None):
    rec = {"generated_by": {"script": "x", "sha256": sha or L.SCRIPT_SHA256_AT_LOAD, "evaluation_sha256": esha or L.E.SCRIPT_SHA256_AT_LOAD}, **season_record(year)}
    for c in rec["configs"].values():
        c["census"] = json.loads(json.dumps(c["census"]))
    with gzip.open(tmp_path / f"continuity_lifecycle_{year}.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(rec, fh)


def test_combine_writes_the_tables_from_the_season_records(tmp_path, capsys):
    for y in L.S.YEARS:
        write_season(tmp_path, y)
    assert L.main(["combine", "--out-dir", str(tmp_path)]) == 0
    rec = json.load(open(tmp_path / "continuity_lifecycle.json", encoding="utf-8"))
    assert rec["decision"]["keep"] == {"accompanied_rise": 6, "share_rise": 0.25, "mark_met": True}
    assert rec["decision"]["recovery"] == {"horizon_hours": 24, "carried_rise": 2, "longest_rise": 8, "longest_share_rise": 0.3333, "developed_further": True}
    assert "rows" not in rec["seasons"]["1990"]["released"]
    text = (tmp_path / "continuity_lifecycle.md").read_text()
    assert "| STORED | 10 (41.7 %) | 0 | 2 | 0 | 0 | 0 | 8 | 2 | 2.00 |" in text and "| KEEP | 16 (66.7 %) |" in text
    assert "| 24 | 2 | 16 |" in text and "Recovery developed further: yes" in text and "mark of 5 points met" in text
    assert "| 24 | 2 | 16 | 4 | 4 |" in text and "| 6 | 6 | 0 | 0 | 0 | 2 |" in text and "the same bar is met" in text
    assert rec["decision"]["recovery_through_any_history"]["bar_met"] is True
    assert "within horizon 4" in text and "4 distinct in 4 pairs" in text
    with pytest.raises(SystemExit, match="never overwritten"):
        L.main(["combine", "--out-dir", str(tmp_path)])


def test_combine_refuses_a_season_record_from_other_scripts(tmp_path):
    write_season(tmp_path, 1990); write_season(tmp_path, 2006, sha="0" * 64)
    with pytest.raises(SystemExit, match="not written by the current scripts"):
        L.main(["combine", "--out-dir", str(tmp_path)])
    os.remove(tmp_path / "continuity_lifecycle_2006.json.gz")
    write_season(tmp_path, 2006, esha="0" * 64)
    with pytest.raises(SystemExit, match="not written by the current scripts"):
        L.main(["combine", "--out-dir", str(tmp_path)])


def test_season_records_are_never_overwritten_and_other_seasons_are_refused(tmp_path):
    (tmp_path / "continuity_lifecycle_1990.json.gz").write_text("x")
    with pytest.raises(SystemExit, match="never overwritten"):
        L.main(["season", "--year", "1990", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="not a development season"):
        L.run_season(1984)
