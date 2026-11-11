"""Tests for scripts/continuity_break_followup.py (`CONTINUITY_ACROSS_AFRICA_2026-11-09.md`, section 29).

The first test is the construction that shows the defect of section 27's sign test: twelve
references with one break each, all in the longer earlier segment."""

import gzip
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import continuity_break_followup as F  # noqa: E402

B = F.B


def pr(coastal, earlier, rid=0):
    return {"id": rid, "coastal": list(coastal), "earlier": list(earlier)}


# --- the position test ------------------------------------------------------------------------------

def test_single_breaks_in_the_longer_segment_are_what_equal_positions_give():
    rows = [pr((0, 4), (1, 16))] * 12
    assert B.concentration(rows)["outcome"] == "concentrated earlier", "the declared sign test reads twelve against none as 1 in 4096"
    t = F.position_test(rows)
    assert t["outcome"] == "not shown" and t["at_most_observed"] == round(0.8 ** 12, 4) == 0.0687
    assert (t["coastal_breaks"], t["expected_coastal_breaks"], t["least"], t["most"], t["at_least_observed"]) == (0, 2.4, 0, 12, 1.0)
    assert (t["near_the_coast_from"], t["earlier_up_to"], t["references_with_a_coastal_break"]) == (6, None, 0), "five or more of twelve at 0.2 is 0.0726 and six or more 0.0194"
    assert F.sign_null(rows) == {"higher_share_coastal": 2.4, "higher_share_earlier": 9.6, "equal_shares": 0.0}


def test_the_arrangements_of_one_reference_and_of_two():
    assert F.exact_positions([pr((0, 1), (1, 3))]) == ({0: 3, 1: 1}, 4)
    assert F.exact_positions([pr((1, 1), (0, 3))]) == ({0: 3, 1: 1}, 4), "the distribution depends on the counts and not on where the break fell"
    assert F.exact_positions([pr((0, 1), (1, 3)), pr((1, 2), (1, 2))]) == ({0: 3, 1: 13, 2: 7, 3: 1}, 24)
    assert F.exact_positions([]) == ({0: 1}, 1)


def test_a_wrong_count_of_arrangements_is_refused(monkeypatch):
    monkeypatch.setattr(F, "comb", lambda n, k: 1)
    with pytest.raises(SystemExit, match="do not add up"):
        F.exact_positions([pr((0, 1), (1, 3))])


def test_the_arrangements_respect_the_size_of_each_segment():
    assert F.exact_positions([pr((2, 3), (1, 1))]) == ({2: 3, 3: 1}, 4), "three breaks cannot leave fewer than two in the coastal segment"
    assert F.exact_positions([pr((0, 0), (2, 5))]) == ({0: 10}, 10), "a reference with one segment only moves nothing"
    assert F.exact_positions([pr((1, 3), (0, 0))]) == ({1: 3}, 3)
    assert F.exact_positions([pr((0, 4), (0, 9))]) == ({0: 1}, 1)
    assert F.exact_positions([pr((2, 2), (0, 3))]) == ({0: 3, 1: 6, 2: 1}, 10)


def test_the_position_test_at_one_in_twenty_both_ways():
    near = F.position_test([pr((1, 1), (0, 19))])
    assert near["outcome"] == "concentrated near the coast" and near["at_least_observed"] == 0.05
    assert (near["near_the_coast_from"], near["earlier_up_to"], near["references_with_a_coastal_break"]) == (1, None, 1)
    assert F.position_test([pr((1, 1), (0, 18))])["outcome"] == "cannot be scored", "one in nineteen is the least either tail can be"
    early = F.position_test([pr((0, 19), (1, 1))])
    assert early["outcome"] == "concentrated earlier" and early["at_most_observed"] == 0.05
    assert (early["near_the_coast_from"], early["earlier_up_to"]) == (None, 0)
    assert F.position_test([pr((0, 18), (1, 1))])["outcome"] == "cannot be scored"


def test_the_position_test_reads_each_tail_with_the_observed_count_in_it():
    five = [pr((0, 4), (1, 4))] * 5
    assert F.position_test(five)["outcome"] == "concentrated earlier", "none of five is 1 in 32"
    assert F.position_test([pr((1, 4), (0, 4))] * 5)["outcome"] == "concentrated near the coast"
    mixed = F.position_test([pr((1, 4), (0, 4))] * 2 + [pr((0, 4), (1, 4))] * 3)
    assert (mixed["outcome"], mixed["at_least_observed"], mixed["at_most_observed"], mixed["expected_coastal_breaks"]) == ("not shown", 0.8125, 0.5, 2.5)
    assert (mixed["near_the_coast_from"], mixed["earlier_up_to"], mixed["references_with_a_coastal_break"]) == (5, 0, 2)
    seven = F.position_test([pr((2, 4), (0, 4))] + [pr((0, 4), (1, 4))] * 6)
    assert (seven["near_the_coast_from"], seven["earlier_up_to"], seven["coastal_breaks"], seven["references_with_a_coastal_break"]) == (7, 1, 2, 1), "58 of 1,792 in either tail"
    one = F.position_test([pr((1, 4), (0, 4))] + [pr((0, 4), (1, 4))] * 4)
    assert (one["outcome"], one["at_most_observed"]) == ("not shown", 0.1875), "one of five or fewer is 6 in 32"
    assert F.position_test([pr((0, 4), (1, 4))] * 4)["outcome"] == "cannot be scored", "none of four is 1 in 16"
    assert F.position_test([pr((0, 4), (1, 4))] * 4 + [pr((0, 0), (1, 9)), pr((1, 3), (0, 0))])["outcome"] == "cannot be scored"


def test_the_expected_references_under_equal_positions():
    assert F.sign_null([pr((1, 2), (1, 2))]) == {"higher_share_coastal": 0.17, "higher_share_earlier": 0.17, "equal_shares": 0.67}
    assert F.sign_null([pr((0, 0), (1, 5)), pr((1, 4), (0, 0))]) == {"higher_share_coastal": 0.0, "higher_share_earlier": 0.0, "equal_shares": 0.0}
    assert F.sign_null([pr((0, 2), (0, 6))]) == {"higher_share_coastal": 0.0, "higher_share_earlier": 0.0, "equal_shares": 1.0}
    assert F.sign_null([pr((0, 1), (1, 3))]) == {"higher_share_coastal": 0.25, "higher_share_earlier": 0.75, "equal_shares": 0.0}
    assert F.sign_null([pr((2, 4), (0, 8))]) == {"higher_share_coastal": 0.58, "higher_share_earlier": 0.42, "equal_shares": 0.0}, "shares are compared, not counts"


# --- the primary's exit ------------------------------------------------------------------------------------

def primary(last=4, stop="earlier", end=False, history="a"):
    return {"history": history, "last": last, "hours_before": 42, "stop": stop, "start": True, "end": end, "majority": False, "no_long_gap": True}


def ref(transitions, sustained=False, coast=True, prim="default", rid=7):
    return {"id": rid, "coast_reaching": coast, "steps": 12, "accompanied": 5, "sustained": sustained, "primary": primary() if prim == "default" else prim, "transitions": transitions}


def tr(k, cls, holders, **extra):
    return {"k": k, "lon": 5.0, "band": "0_to_10E", "hours_to_coast": 6 * (11 - k), "segment": "coastal" if k >= 7 else "earlier", "class": cls, "holders": holders, **extra}


ENDED = {"step": 44, "outcome": "ended"}
AWAY = {"step": 44, "to": 5, "next_step": 45, "outcome": "away", "km": 612.0, "marginal": True, "by_smoothing_alone": False}
BESIDE = {"step": 44, "to": 5, "next_step": 45, "outcome": "beside"}


def test_the_exit_of_a_primary_that_ended():
    t = tr(4, "break", {"a": ENDED}, next="placed_away", main_holder="a", attempts=[{"class": "no_candidate_near"}, {"class": "outside_the_search_area"}])
    assert F.primary_exit(ref([tr(3, "held", {"a": BESIDE}), t])) == {"id": 7, "stop": "earlier", "primary_did": "ended", "reference_then": "placed_away",
                                                                     "attempts": ["no_candidate_near", "outside_the_search_area"]}


def test_the_exit_of_a_primary_that_went_away_while_another_history_stayed():
    t = tr(4, "held", {"a": AWAY, "b": BESIDE})
    assert F.primary_exit(ref([t])) == {"id": 7, "stop": "earlier", "primary_did": "away", "reference_then": "kept_by_another_history",
                                        "km": 612.0, "marginal": True, "by_smoothing_alone": False}


def test_the_exit_names_the_attempts_only_of_a_primary_that_is_the_main_holder():
    t = tr(4, "break", {"a": AWAY, "b": ENDED}, next="handed_off_fresh", main_holder="b", attempts=[{"class": "linked"}])
    got = F.primary_exit(ref([t]))
    assert got["reference_then"] == "handed_off_fresh" and "attempts" not in got and got["primary_did"] == "away"
    past = tr(10, "unresolved", {"a": {"step": 50, "to": 12, "next_step": 52, "outcome": "past_the_window"}})
    assert F.primary_exit(ref([past], prim=primary(last=10, stop="within_24_hours"))) == {"id": 7, "stop": "within_24_hours", "primary_did": "past_the_window", "reference_then": "unresolved"}


def test_only_primaries_lacking_the_end_without_sustained_carriage_have_an_exit():
    t = tr(4, "break", {"a": ENDED}, next="placed_away", main_holder="a", attempts=[{"class": "no_candidate_near"}])
    assert F.primary_exit(ref([t])) is not None
    assert F.primary_exit(ref([t], sustained=True)) is None
    assert F.primary_exit(ref([t], coast=False)) is None
    assert F.primary_exit(ref([t], prim=None)) is None
    assert F.primary_exit(ref([t], prim=primary(end=True))) is None


def test_an_exit_that_the_transitions_do_not_hold_is_refused():
    t = tr(4, "break", {"a": ENDED}, next="placed_away", main_holder="a", attempts=[{"class": "no_candidate_near"}])
    with pytest.raises(SystemExit, match="no transition at its primary's last"):
        F.primary_exit(ref([t], prim=primary(last=5)))
    with pytest.raises(SystemExit, match="no transition at its primary's last"):
        F.primary_exit(ref([t], prim=primary(history="z")))
    with pytest.raises(SystemExit, match="is beside it after"):
        F.primary_exit(ref([tr(4, "held", {"a": BESIDE})]))


# --- the return ----------------------------------------------------------------------------------------------

def test_the_return_of_a_holder_that_went_away():
    t = tr(3, "break", {"a": AWAY, "b": ENDED})
    assert F.return_of(t, {"a": [1, 2, 3, 7], "b": [2, 3]}) == {"hours": 24, "return": "within_24_hours"}
    assert F.return_of(t, {"a": [1, 2, 3, 8], "b": [2, 3]}) == {"hours": 30, "return": "later"}
    assert F.return_of(t, {"a": [1, 2, 3], "b": [2, 3]}) == {"hours": None, "return": "never"}
    assert F.return_of(t, {"a": [1, 2, 3], "b": [2, 3, 5]}) == {"hours": None, "return": "never"}, "a holder that ended is not the one that returns"
    assert F.return_of(tr(3, "break", {"a": ENDED}), {"a": [3, 9]}) is None
    both = tr(3, "break", {"a": AWAY, "b": AWAY})
    assert F.return_of(both, {"a": [3, 9], "b": [3, 5, 6]}) == {"hours": 12, "return": "within_24_hours"}, "the earliest return of either"
    assert F.RETURN_HOURS == 24 and F.RETURNS == ("within_24_hours", "later", "never")


# --- one configuration and the pool ----------------------------------------------------------------------------

def config_fixture():
    r1 = ref([tr(3, "held", {"a": BESIDE}),
              tr(4, "break", {"a": AWAY}, next="candidate_not_stored", main_holder="a", marginal=True, by_smoothing_alone=True, holders_did="continued_away",
                 attempts=[{"class": "over_the_speed_limit"}, {"class": "linked"}], final_km=300.0),
              tr(8, "held", {"a": BESIDE, "e": AWAY})], rid=1)                    # a holder away at a held transition is no break
    r2 = ref([tr(4, "break", {"a": ENDED}, next="placed_away", main_holder="a", holders_did="ended", final_km=750.0,
                 attempts=[{"class": "no_candidate_near"}, {"class": "no_candidate_near"}])], rid=2)
    natural = {"class": "outside_the_search_area", "pass": "12hr", "candidate_km": 269.0, "prediction_km": 552.5, "polygon_area": 7.0, "polygon_at_or_below_minimum": True, "inside_if_inflated": True}
    inflated = {**natural, "pass": "6hr", "polygon_area": 70.71, "inside_if_inflated": False}
    r3 = ref([tr(9, "break", {"c": {**AWAY, "marginal": False}}, next="handed_off_fresh", main_holder="c", marginal=False, by_smoothing_alone=False, holders_did="continued_away",
                 attempts=[inflated, natural], final_km=400.0),
              tr(10, "break", {"f": ENDED}, next="placed_away", main_holder="f", holders_did="ended", attempts=[{"class": "no_candidate_near"}], final_km=None)], sustained=True, rid=3)
    r4 = ref([tr(2, "break", {"d": AWAY}, next="placed_away", main_holder="d", holders_did="continued_away", attempts=[{"class": "linked"}], final_km=1200.0)],
             coast=False, prim=None, rid=4)
    by = {1: {"a": [1, 2, 3, 4, 8, 9], "e": [8]}, 2: {"a": [3, 4]}, 3: {"c": [9], "f": [10]}, 4: {"d": [2, 6]}}
    return [r1, r2, r3, r4], by


def test_the_follow_up_of_a_constructed_configuration():
    refs, by = config_fixture()
    got = F.config_followup(refs, by)
    assert got["breaks"] == 4, "the break of the reference ending inside Africa is not counted"
    assert [x["id"] for x in got["exits"]] == [1, 2] and got["exits"][0]["primary_did"] == "away" and got["exits"][1]["attempts"] == ["no_candidate_near", "no_candidate_near"]
    assert got["groups"] == [{"id": 1, "k": 4, "segment": "earlier", "group": "the_holder_took_the_nearest_candidate", "by_smoothing_alone": True},
                             {"id": 2, "k": 4, "segment": "earlier", "group": "no_candidate_near_at_an_attempt_read", "by_smoothing_alone": False},
                             {"id": 3, "k": 9, "segment": "coastal", "group": "a_candidate_near_was_not_taken", "by_smoothing_alone": False},
                             {"id": 3, "k": 10, "segment": "coastal", "group": "no_candidate_near_at_an_attempt_read", "by_smoothing_alone": False}]
    assert got["placed_away_final_km"] == [750.0], "a break of another class and one with no final candidate at all are left out"
    assert got["outside"] == [{"id": 3, "k": 9, "pass": "6hr", "candidate_km": 269.0, "prediction_km": 552.5, "polygon_area": 70.71, "polygon_at_or_below_minimum": True, "inside_if_inflated": False},
                              {"id": 3, "k": 9, "pass": "12hr", "candidate_km": 269.0, "prediction_km": 552.5, "polygon_area": 7.0, "polygon_at_or_below_minimum": True, "inside_if_inflated": True}]
    assert got["returns"] == [{"id": 1, "k": 4, "segment": "earlier", "next": "candidate_not_stored", "by_smoothing_alone": True, "marginal": True, "hours": 24, "return": "within_24_hours"},
                              {"id": 3, "k": 9, "segment": "coastal", "next": "handed_off_fresh", "by_smoothing_alone": False, "marginal": False, "hours": None, "return": "never"}]
    assert got["per_reference"] == [{"id": 1, "coastal": [0, 1], "earlier": [1, 2]}, {"id": 2, "coastal": [0, 0], "earlier": [1, 1]}, {"id": 3, "coastal": [2, 2], "earlier": [0, 0]}]


def test_the_group_of_a_break_by_its_attempts():
    def g(*classes):
        return F.group_of({"attempts": [{"class": c} for c in classes]})

    assert g("linked") == g("no_candidate_near", "linked") == g("over_the_speed_limit", "linked") == "the_holder_took_the_nearest_candidate"
    assert g("outside_the_search_area") == g("no_candidate_near", "another_candidate_nearer") == g("over_the_speed_limit", "no_candidate_near") == "a_candidate_near_was_not_taken"
    assert g("no_candidate_near") == g("no_candidate_near", "no_candidate_near") == "no_candidate_near_at_an_attempt_read"


def test_the_pool_of_two_seasons():
    refs, by = config_fixture()
    part = F.config_followup(refs, by)
    p = F.pooled([part, part])
    assert p["declared_sign_test"] == {"scored": 2, "higher_share_coastal": 0, "higher_share_earlier": 2, "equal_shares": 0, "sign_test_value": 0.25, "outcome": "cannot be scored"}
    assert p["expected_under_equal_positions"] == {"higher_share_coastal": 0.67, "higher_share_earlier": 1.33, "equal_shares": 0.0}
    assert p["position_test"]["coastal_breaks"] == 4 and p["position_test"]["outcome"] == "cannot be scored" and p["position_test"]["expected_coastal_breaks"] == 4.67
    g = p["groups"]
    assert g["all"] == {"the_holder_took_the_nearest_candidate": 2, "a_candidate_near_was_not_taken": 2, "no_candidate_near_at_an_attempt_read": 4}
    assert g["coastal"] == {"the_holder_took_the_nearest_candidate": 0, "a_candidate_near_was_not_taken": 2, "no_candidate_near_at_an_attempt_read": 2}
    assert g["earlier"]["no_candidate_near_at_an_attempt_read"] == 2 and g["by_smoothing_alone"] == {"the_holder_took_the_nearest_candidate": 2, "a_candidate_near_was_not_taken": 0,
                                                                                                    "no_candidate_near_at_an_attempt_read": 0}
    assert p["placed_away"] == {"count": 2, "within_750_km": 2, "nearest_km": 750.0, "median_km": 750.0, "farthest_km": 750.0}
    assert p["outside"] == {"6hr": {"count": 2, "inside_if_inflated": 0, "polygon_under_the_inflated_size": 0}, "12hr": {"count": 2, "inside_if_inflated": 2, "polygon_under_the_inflated_size": 2}}
    x = p["exits"]
    assert x["count"] == 4 and x["primary_did"] == {"ended": 2, "away": 2, "past_the_window": 0} and (x["away_marginal"], x["away_by_smoothing_alone"]) == (2, 0)
    assert x["reference_then"]["candidate_not_stored"] == 2 and x["reference_then"]["placed_away"] == 2 and x["reference_then"]["kept_by_another_history"] == 0
    assert x["by_stop"]["earlier"] == {"ended": 2, "away": 2, "past_the_window": 0} and x["by_stop"]["within_24_hours"] == {"ended": 0, "away": 0, "past_the_window": 0}
    r = p["returns"]
    assert (r["breaks"], r["with_a_holder_away"]) == (8, 4) and r["return"] == {"within_24_hours": 2, "later": 0, "never": 2}
    assert r["return_by_smoothing_alone"] == {"within_24_hours": 2, "later": 0, "never": 0} and r["return_by_next"]["handed_off_fresh"] == {"within_24_hours": 0, "later": 0, "never": 2}


def test_the_distances_of_placed_away_breaks_are_pooled_in_order():
    refs, by = config_fixture()
    part = F.config_followup(refs, by)
    p = F.pooled([{**part, "placed_away_final_km": [1500.0, 900.0]}, part])
    assert p["placed_away"] == {"count": 3, "within_750_km": 1, "nearest_km": 750.0, "median_km": 900.0, "farthest_km": 1500.0}
    assert F.pooled([{**part, "placed_away_final_km": []}])["placed_away"] == {"count": 0, "within_750_km": 0, "nearest_km": None, "median_km": None, "farthest_km": None}


def test_the_inflated_size_of_each_pass():
    assert round(F.INFLATED["6hr"], 2) == 70.70 and round(F.INFLATED["12hr"], 2) == 94.27
    refs, by = config_fixture()
    part = F.config_followup(refs, by)
    row = part["outside"][1]
    edge = F.pooled([{**part, "outside": [{**row, "polygon_area": 94.28}, {**row, "polygon_area": 94.26}, {**row, "polygon_area": None}, {**row, "pass": "6hr", "polygon_area": 94.26}]}])
    assert edge["outside"]["12hr"] == {"count": 3, "inside_if_inflated": 3, "polygon_under_the_inflated_size": 1}
    assert edge["outside"]["6hr"] == {"count": 1, "inside_if_inflated": 1, "polygon_under_the_inflated_size": 0}


def test_the_recomputed_transitions_must_be_those_of_the_season_record():
    refs, _ = config_fixture()
    plain = [{**r, "transitions": [{k: v for k, v in t.items() if k not in ("next", "attempts")} for t in r["transitions"]]} for r in refs]
    F.check_transitions(plain, refs, "x")
    with pytest.raises(SystemExit, match="are not the season record's"):
        F.check_transitions(plain[:3], refs, "x")
    with pytest.raises(SystemExit, match="are not the season record's"):
        F.check_transitions(list(reversed(plain)), refs, "x")
    short = [{**plain[0], "transitions": plain[0]["transitions"][:2]}] + plain[1:]
    with pytest.raises(SystemExit, match="transitions of reference 1"):
        F.check_transitions(short, refs, "x")
    moved = [{**plain[0], "transitions": [{**plain[0]["transitions"][0], "class": "break"}] + plain[0]["transitions"][1:]}] + plain[1:]
    with pytest.raises(SystemExit, match="transitions of reference 1"):
        F.check_transitions(moved, refs, "x")
    other = [{**plain[0], "transitions": plain[0]["transitions"][:2] + [{**plain[0]["transitions"][2], "holders": {"z": BESIDE}}]}] + plain[1:]
    with pytest.raises(SystemExit, match="transitions of reference 1"):
        F.check_transitions(other, refs, "x")


# --- the records ---------------------------------------------------------------------------------------------------

def seasons_fixture():
    refs, by = config_fixture()
    part = F.config_followup(refs, by)
    return [{"year": y, "record_sha256": "s" * 64, "configs": {n: part for n in B.CONFIGS}} for y in B.YEARS]


def combined(tmp_path, sha=None, breaks=8, outcome="cannot be scored"):
    c = {"scored": 2, "higher_share_coastal": 0, "higher_share_earlier": 2, "equal_shares": 0, "sign_test_value": 0.25, "outcome": outcome}
    rec = {"generated_by": {"sha256": sha or B.SCRIPT_SHA256_AT_LOAD}, "pooled": {n: {"concentration": c, "breaks": {"count": breaks}} for n in B.CONFIGS}}
    json.dump(rec, open(tmp_path / "continuity_break_reading.json", "w", encoding="utf-8"))


def test_the_follow_up_is_written_once_from_the_break_reading_records(tmp_path, monkeypatch, capsys):
    by_year = {s["year"]: s for s in seasons_fixture()}
    monkeypatch.setattr(F, "season_followup", lambda y, artifacts: by_year[y])
    combined(tmp_path)
    assert F.main(["--out-dir", str(tmp_path)]) == 0
    rec = json.load(open(tmp_path / "continuity_break_followup.json", encoding="utf-8"))
    assert set(rec["pooled"]) == {"released", "VCLIP"} and set(rec["seasons"]) == {"1990", "2006"}
    assert rec["generated_by"]["break_reading_sha256"] == B.SCRIPT_SHA256_AT_LOAD and rec["generated_by"]["seasons"] == {"1990": "s" * 64, "2006": "s" * 64}
    text = (tmp_path / "continuity_break_followup.md").read_text(encoding="utf-8")
    assert text.strip() in capsys.readouterr().out
    assert "| in the earlier segment | 2 | 1.33 |" in text and "Outcome: CANNOT BE SCORED." in text and "declared outcome CANNOT BE SCORED" in text
    assert "The coastal breaks lie on 2 references. Concentrated near the coast would be read at no count, concentrated earlier at no count." in text
    assert "| within 24 hours | 2 | 2 |" in text and "| never | 2 | 0 |" in text and "| earlier | 2 | 2 | 0 |" in text
    assert "The reference at that observation: candidate not stored 2, placed away 2." in text
    assert "| a candidate near was not taken | 2 | 2 | 0 | 0 |" in text and "| the holder took the nearest candidate | 2 | 0 | 2 | 2 |" in text
    assert "Breaks classed as placed away: 2, the nearest final candidate at 750.0 to 750.0 km" in text and "within 750 km for 2." in text
    assert ("Attempts with a candidate outside the search area: 6-hour 2 (polygon under its inflated size 0, a trough point inside once inflated 0), "
            "12-hour 2 (polygon under its inflated size 2, a trough point inside once inflated 2).") in text
    with pytest.raises(SystemExit, match="never overwritten"):
        F.main(["--out-dir", str(tmp_path)])


def test_the_follow_up_refuses_records_that_do_not_agree(tmp_path, monkeypatch):
    by_year = {s["year"]: s for s in seasons_fixture()}
    monkeypatch.setattr(F, "season_followup", lambda y, artifacts: by_year[y])
    combined(tmp_path, sha="0" * 64)
    with pytest.raises(SystemExit, match="combined break-reading record was not written"):
        F.main(["--out-dir", str(tmp_path)])
    combined(tmp_path, breaks=7)
    with pytest.raises(SystemExit, match="do not give the combined"):
        F.main(["--out-dir", str(tmp_path)])
    combined(tmp_path, outcome="not shown")
    with pytest.raises(SystemExit, match="do not give the combined"):
        F.main(["--out-dir", str(tmp_path)])
    assert not os.path.exists(tmp_path / "continuity_break_followup.json")


def test_a_season_record_of_another_script_or_year_is_refused(tmp_path):
    def write(year, sha, inside):
        with gzip.open(tmp_path / f"continuity_break_reading_{year}.json.gz", "wt", encoding="utf-8") as fh:
            json.dump({"generated_by": {"sha256": sha}, "year": inside, "configs": {}}, fh)

    write(1990, "0" * 64, 1990)
    with pytest.raises(SystemExit, match="not written by the current break-reading script"):
        F.season_followup(1990, str(tmp_path))
    write(1990, B.SCRIPT_SHA256_AT_LOAD, 2006)
    with pytest.raises(SystemExit, match="not written by the current break-reading script"):
        F.season_followup(1990, str(tmp_path))
