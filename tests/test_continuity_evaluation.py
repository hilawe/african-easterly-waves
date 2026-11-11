"""Tests for scripts/continuity_evaluation.py (`CONTINUITY_ACROSS_AFRICA_2026-11-09.md`, section 18).

The first four are the constructed cases section 18 names: a track with a relay and an undetermined
transition, 935 unaccompanied short tracks added, a relay whose link jumps, and a supported switch
beside a drift."""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import continuity_evaluation as E  # noqa: E402

D0 = 7000.0
FAR_WEST = -15.7                                                     # about 1,500 km west of a prediction at 2 W on 10 N


def q(tid, lats, lons, start=D0):
    n = len(lats)
    return {"id": tid, "time": start + 0.25 * np.arange(n), "lat": np.asarray(lats, float), "lon": np.asarray(lons, float), "n_valid": n}


def hist(hid, lats, lons, start=D0, est=None):
    """A history whose stored and unsmoothed positions agree and whose links match their
    predictions unless `est` says otherwise."""
    n = len(lats)
    if est is None:
        est = [[float(lats[j]), float(lons[j]), 6] for j in range(1, n)]
    return {"id": hid, "time": start + 0.25 * np.arange(n), "lat": np.asarray(lats, float), "lon": np.asarray(lons, float),
            "step": list(range(n)), "raw_lat": [float(x) for x in lats], "raw_lon": [float(x) for x in lons], "est": est, "fate": "finished"}


def measures(histories, q_tracks, ref_rows=()):
    return E.flag_measures(histories, E.C.index_by_time(q_tracks), E.J.spans(q_tracks), list(ref_rows))


def ref_row(tid, first_lon, last, atlantic=True, last_used=None):
    row = {"id": tid, "first": {"lon": first_lon}, "outcome": "atlantic_side" if atlantic else "inland", "last_used": {"index": last if last_used is None else last_used}}
    if atlantic:
        row["first_atlantic_side"] = {"index": last}
    return row


def mixed_case():
    t = hist("t", [10.0] * 4, [0.0, -2.0, -4.0, -6.0])
    a = q(1, [10.0] * 2, [0.0, -2.0])                                # ends, so the next system is a relay
    b = q(2, [10.0, np.nan, 10.0], [-4.0, np.nan, 20.0], start=D0 + 0.5)   # alive at the fourth step without a position
    c = q(3, [10.0], [-6.0], start=D0 + 0.75)
    return t, [a, b, c]


def test_a_relay_and_an_undetermined_transition_are_flagged():
    t, qs = mixed_case()
    ev, _ = E.events_of(t, E.C.index_by_time(qs), E.J.spans(qs))
    assert [e["kind"] for e in ev] == ["relay", "undetermined"]
    assert E.J.track_class(ev) == "relay", "the defect section 17 records, kept in the earlier script"
    assert E.flagged(ev) and E.display_class(ev) == "undetermined"
    flags, _ = measures([t], qs)
    assert flags["flagged_tracks"] == 1 and flags["display_classes"]["undetermined"] == 1 and flags["display_classes"]["relay"] == 0


def test_a_relay_alone_is_not_flagged_and_a_switch_is():
    assert not E.flagged([{"kind": "relay"}, {"kind": "addition"}])
    assert E.flagged([{"kind": "addition"}, {"kind": "switch"}])
    assert E.display_class([{"kind": "relay"}, {"kind": "switch"}]) == "switch"
    assert E.display_class([{"kind": "addition"}]) == "addition_only" and E.display_class([]) == "no_transition"


def switch_case():
    t = hist("t", [10.0] * 4, [0.0, -2.0, -4.0, -6.0])
    a = q(1, [10.0] * 4, [0.0, -2.0, 4.0, 8.0])                      # with the track, then turns away, still alive
    b = q(2, [10.0] * 2, [-4.0, -6.0], start=D0 + 0.5)
    return t, [a, b]


def test_unaccompanied_short_tracks_change_no_flag_and_no_reference_measure():
    t, qs = switch_case()
    ref = q(9, [10.0] * 4, [20.0, 18.0, 16.0, 14.0])
    beside = hist("h", [10.0] * 4, [20.0, 18.0, 16.0, 14.0])
    rows = [ref_row(9, 20.0, 3)]
    qmap = {x["id"]: x for x in qs + [ref]}
    base_sets = [t, beside]
    r0 = E.reference_measures(base_sets, qmap, rows)
    f0, c0 = measures(base_sets, qs + [ref], r0)
    short = [hist(f"s{k}", [30.0, 30.0], [35.0 - 0.01 * (k % 50), 34.0], start=D0 + 20.0 + k) for k in range(935)]
    r1 = E.reference_measures(base_sets + short, qmap, rows)
    f1, c1 = measures(base_sets + short, qs + [ref], r1)
    assert f1 == f0 and E.pool_reference(r1) == E.pool_reference(r0), "nothing gated moves"
    assert f0["flagged_tracks"] == 1 and f0["exposure_pairs"] == 6
    assert c1["tracks_in_sector"] == c0["tracks_in_sector"] + 935 and c1["links"] == c0["links"] + 935
    assert f1["flagged_tracks"] / c1["tracks_in_sector"] < f0["flagged_tracks"] / c0["tracks_in_sector"], "the rate section 15 gated on is diluted"


def test_a_relay_whose_link_jumps_stays_open():
    t = hist("t", [10.0, 10.0], [0.0, FAR_WEST], est=[[10.0, -2.0, 6]])
    a = q(1, [10.0], [0.0])                                          # its record ends at the first step
    b = q(2, [10.0], [FAR_WEST], start=D0 + 0.25)
    flags, costs = measures([t], [a, b])
    assert flags["events"]["relay"] == 1 and flags["relay_events_with_a_jump_link"] == 1
    assert flags["flagged_tracks"] == 0 and flags["supported_false_link_candidates"] == 0, "reported, neither flagged nor cleared"
    assert costs["jump_links"] == 1 and costs["links"] == 1


def test_a_switch_on_a_jump_link_with_the_old_system_at_the_prediction_is_supported():
    t = hist("t", [10.0, 10.0], [0.0, FAR_WEST], est=[[10.0, -2.0, 6]])
    a = q(1, [10.0, 10.0], [0.0, -2.0])                              # goes on where the track was predicted to be
    b = q(2, [10.0], [FAR_WEST], start=D0 + 0.25)
    flags, _ = measures([t], [a, b])
    assert flags["events"]["switch"] == 1 and flags["switch_events_with_a_jump_link"] == 1 and flags["supported_false_link_candidates"] == 1
    assert flags["systems_left_behind"] == 1 and flags["switch_events_by_elapsed"] == {"to_12_hours": 1, "18_to_24_hours": 0, "over_24_hours": 0}


def test_a_drift_between_systems_on_matching_links_is_a_switch_without_support():
    t = hist("t", [10.0] * 4, [0.0, -2.0, -4.0, -6.0])
    a = q(1, [10.0] * 4, [0.0, 6.0, 8.0, 10.0])                      # leaves at once, still alive
    b = q(2, [10.0], [-6.0], start=D0 + 0.75)
    flags, costs = measures([t], [a, b])
    assert flags["events"]["switch"] == 1 and flags["switch_events_with_a_jump_link"] == 0 and flags["supported_false_link_candidates"] == 0
    assert flags["flagged_tracks"] == 1 and flags["switch_events_by_elapsed"]["18_to_24_hours"] == 1 and costs["jump_links"] == 0


def test_a_jump_away_from_a_prediction_the_old_system_is_not_at_is_unsupported():
    t = hist("t", [10.0, 10.0], [0.0, FAR_WEST], est=[[10.0, -2.0, 6]])
    a = q(1, [10.0, 10.0], [0.0, 12.0])                              # still alive, but nowhere near the prediction
    b = q(2, [10.0], [FAR_WEST], start=D0 + 0.25)
    flags, _ = measures([t], [a, b])
    assert flags["switch_events_with_a_jump_link"] == 1 and flags["supported_false_link_candidates"] == 0


def test_only_jump_links_inside_the_interval_count():
    t = hist("t", [10.0] * 4, [0.0, -2.0, -4.0, -6.0], est=[[10.0, -2.0, 6], [10.0, -4.0, 6], [10.0, 9.0, 6]])   # the last link jumps
    a = q(1, [10.0] * 4, [0.0, -2.0, 4.0, 8.0])
    b = q(2, [10.0] * 2, [-4.0, -6.0], start=D0 + 0.5)
    qs = [a, b]
    ev, _ = E.events_of(t, E.C.index_by_time(qs), E.J.spans(qs))
    assert [(e["from"], e["to"]) for e in ev] == [(1, 2)]
    assert E.link_marks(ev[0], t, E.J.spans(qs)) == {"jump_links": 0, "supported": False}, "the jump at the fourth observation is after the switch"


def test_the_thresholds_include_their_bounds():
    t = hist("t", [10.0, 10.0], [0.0, -10.0], est=[[10.0, -2.0, 6]])                # claimed 8 degrees from the prediction
    a = q(1, [10.0, 10.0], [0.0, 4.0])                                              # 6 degrees from the prediction
    b = q(2, [10.0], [-10.0], start=D0 + 0.25)
    d_jump = float(E.C.km(10.0, -2.0, 10.0, -10.0)); d_sys = float(E.C.km(10.0, -2.0, 10.0, 4.0))
    assert E.jump(t, 1, radius_km=d_jump) and not E.jump(t, 1, radius_km=d_jump + 1e-6)
    ev = {"from": 0, "to": 1, "lost": [1]}
    spans = E.J.spans([a, b])
    assert E.link_marks(ev, t, spans, radius_km=d_sys) == {"jump_links": 1, "supported": True}, "a system exactly at the radius supports"
    assert E.link_marks(ev, t, spans, radius_km=d_sys - 1e-6) == {"jump_links": 1, "supported": False}


def test_events_carry_their_interval_and_the_exposure_counts_pairs():
    t = hist("t", [10.0] * 4, [0.0, -2.0, -4.0, -6.0])
    a = q(1, [10.0] * 4, [0.0, 6.0, 8.0, 10.0])
    b = q(2, [10.0], [-6.0], start=D0 + 0.75)
    qs = [a, b]
    ev, pairs = E.events_of(t, E.C.index_by_time(qs), E.J.spans(qs))
    assert [(e["kind"], e["from"], e["to"]) for e in ev] == [("switch", 0, 3)] and pairs == 1
    plain = E.J.transitions(t, E.C.index_by_time(qs), E.J.spans(qs))
    assert [{k: e[k] for k in plain[0]} for e in ev] == plain, "the transitions are section 15's, unchanged"
    assert E.events_of(hist("u", [30.0, 30.0], [30.0, 29.0]), E.C.index_by_time(qs), E.J.spans(qs)) == ([], 0)


def test_link_status_reads_both_ends_of_one_link():
    t = hist("t", [10.0] * 4, [0.0, -2.0, -4.0, -6.0])
    a = q(1, [10.0] * 4, [0.0, -2.0, 4.0, 8.0])                      # beside the first two observations, then away and alive
    b = q(2, [10.0] * 2, [-4.0, -6.0], start=D0 + 0.5)
    qs = [a, b]; idx, spans = E.C.index_by_time(qs), E.J.spans(qs)
    assert [E.link_status(t, j, idx, spans) for j in (1, 2, 3)] == ["consistent", "switch", "consistent"]
    ended = q(1, [10.0] * 2, [0.0, -2.0])
    qs = [ended, b]
    assert E.link_status(t, 2, E.C.index_by_time(qs), E.J.spans(qs)) == "relay"
    gone = q(1, [10.0, 10.0, np.nan, 10.0], [0.0, -2.0, np.nan, 8.0])
    qs = [gone, b]
    assert E.link_status(t, 2, E.C.index_by_time(qs), E.J.spans(qs)) == "undetermined"
    qs = [a]
    assert E.link_status(t, 2, E.C.index_by_time(qs), E.J.spans(qs)) == "unassessed", "the later end has no companion"
    qs = [b]
    assert E.link_status(t, 2, E.C.index_by_time(qs), E.J.spans(qs)) == "unassessed", "the earlier end has no companion"
    both = q(3, [10.0] * 4, [0.3, -1.7, -3.7, -5.7])                 # stays beside the track while the other system leaves
    qs = [a, b, both]
    assert E.link_status(t, 2, E.C.index_by_time(qs), E.J.spans(qs)) == "consistent", "one common system is enough"


def test_the_link_census_is_by_prediction_hours_and_only_in_the_sector():
    t = hist("t", [10.0] * 4, [0.0, -2.0, -4.0, -6.0], est=[[10.0, -2.0, 6], [10.0, -4.0, 12], [10.0, -6.0, 6]])
    away = hist("o", [10.0] * 2, [100.0, 98.0])
    a = q(1, [10.0] * 4, [0.0, -2.0, 4.0, 8.0])
    b = q(2, [10.0] * 2, [-4.0, -6.0], start=D0 + 0.5)
    qs = [a, b]
    got = E.link_census([t, away], E.C.index_by_time(qs), E.J.spans(qs))
    zero = {k: 0 for k in E.LINK_STATUS}
    assert got == {"6": {**zero, "consistent": 2}, "12": {**zero, "switch": 1}}


def test_elapsed_classes():
    t = hist("t", [10.0] * 6, [0.0] * 6)
    for to, name in ((1, "to_12_hours"), (2, "to_12_hours"), (3, "18_to_24_hours"), (4, "18_to_24_hours"), (5, "over_24_hours")):
        assert E.elapsed_class({"from": 0, "to": to}, t) == name


def reference_case():
    lons = [20.0 - 2.0 * k for k in range(12)]
    ref = q(9, [10.0] * 12, lons)
    h1 = hist("h1", [10.0] * 4, lons[:4])
    h2 = hist("h2", [10.0] * 2, lons[10:], start=D0 + 2.5)
    return ref, h1, h2


def test_reference_measures_on_a_relayed_track():
    ref, h1, h2 = reference_case()
    rows = E.reference_measures([h1, h2], {9: ref}, [ref_row(9, 20.0, 11)])
    r = rows[0]
    assert (r["steps"], r["accompanied"], r["longest"], r["class"]) == (12, 6, 4, "relayed")
    assert r["by_history"] == {"h1": [0, 1, 2, 3], "h2": [10, 11]} and r["start"] == ["h1"] and r["event"] == ["h2"]
    assert r["nearest"] == ["h1"] * 4 + [None] * 6 + ["h2"] * 2
    p = E.pool_reference(rows)
    assert p == {"tracks": 1, "steps": 12, "accompanied": 6, "longest_single_history": 4,
                 "classes": {"carried": 0, "relayed": 1, "picked_up_late": 0, "dropped": 0, "unaccompanied_at_both": 0},
                 "tracks_accompanied": 1, "histories_on_accompanied": 2}


def test_the_nearest_history_is_the_closest_and_an_unaccompanied_track_is_not_counted_as_accompanied():
    ref, h1, h2 = reference_case()
    off = hist("a_farther", [12.0] * 4, [20.0, 18.0, 16.0, 14.0])    # about 220 km north of the first four observations
    apart = q(8, [30.0] * 4, [20.0, 18.0, 16.0, 14.0])              # a reference track nothing accompanies
    rows = E.reference_measures([off, h1, h2], {9: ref, 8: apart}, [ref_row(9, 20.0, 11), ref_row(8, 20.0, 3)])
    assert rows[0]["nearest"][:4] == ["h1"] * 4 and rows[0]["by_history"]["a_farther"] == [0, 1, 2, 3]
    p = E.pool_reference(rows)
    assert (p["tracks"], p["tracks_accompanied"], p["histories_on_accompanied"]) == (2, 1, 3)
    assert p["classes"]["unaccompanied_at_both"] == 1 and p["steps"] == 16 and p["accompanied"] == 6


def test_reference_measures_respect_the_band_the_outcome_and_the_used_length():
    ref, h1, h2 = reference_case()
    assert E.reference_measures([h1], {9: ref}, [ref_row(9, 31.0, 11)]) == [], "starts outside 10 to 30 E"
    assert E.reference_measures([h1], {9: ref}, [ref_row(9, 30.0, 11)]) == [], "the band's upper edge is excluded"
    assert len(E.reference_measures([h1], {9: ref}, [ref_row(9, 10.0, 11)])) == 1, "the lower edge is included"
    inland = E.reference_measures([h1, h2], {9: ref}, [ref_row(9, 20.0, 5, atlantic=False)])[0]
    assert "class" not in inland and (inland["steps"], inland["accompanied"]) == (6, 4), "read to the last used observation only"
    early = E.reference_measures([h1, h2], {9: ref}, [ref_row(9, 20.0, 5, last_used=11)])[0]
    assert (early["steps"], early["accompanied"], early["class"]) == (6, 4, "dropped"), "an Atlantic-side track is read to that observation, not to its last"
    whole = hist("w", [10.0] * 12, [20.0 - 2.0 * k for k in range(12)])
    assert E.reference_measures([whole], {9: ref}, [ref_row(9, 20.0, 11)])[0]["class"] == "carried"


def test_a_flagged_history_beside_a_reference_track_is_counted_once_per_track():
    t, qs = switch_case()
    ref = q(9, [10.0] * 4, [0.5, -1.5, -3.5, -5.5])                  # travels beside the flagged track
    rows = E.reference_measures([t], {9: ref}, [ref_row(9, 20.0, 3)])
    flags, _ = measures([t], qs, rows)
    assert flags["reference_tracks_with_a_flagged_history"] == 1
    apart = q(8, [25.0] * 4, [20.0, 18.0, 16.0, 14.0])              # a reference track the flagged history never nears
    clean = hist("c", [25.0] * 4, [20.0, 18.0, 16.0, 14.0])
    rows = E.reference_measures([t, clean], {8: apart}, [ref_row(8, 20.0, 3)])
    assert rows[0]["by_history"] == {"c": [0, 1, 2, 3]}
    assert measures([t, clean], qs + [apart], rows)[0]["reference_tracks_with_a_flagged_history"] == 0


def test_every_system_left_by_a_switch_is_counted_and_a_standing_track_is_a_cost():
    t = hist("t", [10.0] * 4, [0.0, -2.0, -4.0, -6.0])
    a = q(1, [10.0] * 4, [0.0, -2.0, 4.0, 8.0])
    twin = q(3, [10.0] * 4, [0.2, -1.8, 4.2, 8.2])                   # a second system leaving with the first
    b = q(2, [10.0] * 2, [-4.0, -6.0], start=D0 + 0.5)
    flags, costs = measures([t], [a, twin, b])
    assert flags["events"]["switch"] == 1 and flags["systems_left_behind"] == 2
    assert costs["stationary_tracks"] == 0
    standing = hist("s", [10.0] * 13, [0.0] * 13)                    # three days in one place
    assert measures([t, standing], [a, twin, b])[1]["stationary_tracks"] == 1


def test_histories_outside_the_sector_are_not_scored():
    t, qs = switch_case()
    moved = hist("m", [10.0] * 4, [100.0, 98.0, 96.0, 94.0])
    qs2 = [q(1, [10.0] * 4, [100.0, 98.0, 104.0, 108.0]), q(2, [10.0] * 2, [96.0, 94.0], start=D0 + 0.5)]
    flags, costs = measures([moved], qs2)
    assert flags["flagged_tracks"] == 0 and flags["events"]["switch"] == 0 and costs["tracks_in_sector"] == 0
    assert measures([t], qs)[1]["tracks_in_sector"] == 1


def test_a_transition_without_a_previous_accompanied_observation_is_refused(monkeypatch):
    t, qs = switch_case()
    monkeypatch.setattr(E.J, "transitions", lambda *_: [{"time": D0, "kind": "switch", "new": [1], "lost": [2]}])
    with pytest.raises(SystemExit, match="no previous accompanied observation"):
        E.events_of(t, E.C.index_by_time(qs), E.J.spans(qs))
