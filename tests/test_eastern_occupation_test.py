"""The occupation test's rules (EASTERN_OCCUPATION_TEST_2026-10-20.md, sections 2 to 4)."""

import datetime as dt
import importlib.util
import json
import os
from fractions import Fraction

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location("occupation_under_test", os.path.join(ROOT, "scripts", "eastern_occupation_test.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(O, cells):
    r = np.zeros(len(O.LONS), dtype=np.int64)
    for x, v in cells.items():
        r[O.S.col_of(x)] = v
    return r


STANDING = {49.5: -2000, 50.0: -2000, 50.5: -2000}


def _occupation(O, rows, days, lons):
    H = np.array(rows)
    strips, by_day = O.S.link_strips(H)
    path = O.S.history_path({"days": days, "lons": lons}, strips, by_day)
    return O.history_occupation(path, O.S.member_classes(path, strips), by_day, H)


def test_pre_position_and_distances():
    O = _load()
    rows = [_row(O, STANDING)] * 3 + [_row(O, {52.0: -2000})] * 2 + [_row(O, {})]
    H = np.array(rows)
    assert O.pre_position(H, 5) == 50, "the median of 50, 50, 50, 52, 52"
    assert O.pre_position(H, 6) == 51 and O.pre_position(H, 2) is None, "four present days give the exact mean of the middle two, and two days are too few"
    assert O.pre_position(np.array([_row(O, {})] * 3 + rows[:3]), 6) == 50 and O.pre_position(np.array([_row(O, {})] * 4 + rows[:2]), 6) is None, "three of five present is enough, two is not"
    a, b = O.S.col_of(49.5), O.S.col_of(50.5)
    assert O.run_distance(a, b, Fraction(50)) == 0 and O.run_distance(a, b, Fraction(52)) == Fraction(3, 2) and O.run_distance(a, b, Fraction(48)) == Fraction(3, 2)
    by_day = [[(a, b, 7)]]
    assert O.occupants(by_day, 0, Fraction(52)) == {7} and O.occupants(by_day, 0, Fraction(105, 2) + Fraction(1, 100)) == set(), "1.5 degrees inclusive"
    assert O.cells_near(_row(O, {51.5: -1200}), Fraction(50), -3.0) and not O.cells_near(_row(O, {51.5: -1199}), Fraction(50), -3.0), "-3 m/s qualifies, -2.9975 does not"
    assert not O.cells_near(_row(O, {52.0: -2000}), Fraction(50), -3.0), "a cell 2 degrees away is outside the tolerance"


def test_held_vacated_and_readings():
    O = _load()
    pre = [_row(O, STANDING)] * 5
    passage = pre + [_row(O, {**STANDING, 60.0: -2000, 60.5: -2000}), _row(O, {**STANDING, 56.0: -2000, 56.5: -2000}), _row(O, {**STANDING, 53.0: -2000, 53.5: -2000}), _row(O, STANDING)]
    o = _occupation(O, passage, [5, 6, 7, 8], [60.0, 56.0, 53.0, 51.5])
    assert [c for c, _ in o["members"]] == ["held_continuous"] * 4 and [i for _, i in o["members"]] == [False, True, True, True]
    assert o["reading"] == "held" and o["sensitivity"] == (4, 4), "a member on no run counts as departed, and the standing strip held 50 before the history entered"
    shift = pre + [_row(O, {48.0: -2000, 48.5: -2000}), _row(O, {46.0: -2000, 46.5: -2000}), _row(O, {44.0: -2000, 44.5: -2000})]
    o = _occupation(O, shift, [5, 6, 7], [48.0, 46.0, 44.0])
    assert [c for c, _ in o["members"]] == ["not_departed", "vacated", "vacated"] and o["reading"] == "vacated", "1.5 degrees from its own run is not departed, and nothing holds 50 once the strip has left"
    tie = pre + [_row(O, {**STANDING, 60.0: -2000, 60.5: -2000}), _row(O, {**STANDING, 56.0: -2000, 56.5: -2000}), _row(O, {53.0: -2000, 53.5: -2000})]
    o = _occupation(O, tie, [6, 7], [56.0, 53.0])
    assert [c for c, _ in o["members"]] == ["held_continuous", "vacated"] and o["reading"] == "tied"
    fresh = pre + [_row(O, {48.0: -2000, 48.5: -2000, 49.0: -2000, 49.5: -2000, 53.0: -2000}),
                   _row(O, {46.5: -2000, 47.0: -2000, 47.5: -2000, 48.0: -2000, 50.0: -2000}),
                   _row(O, {44.5: -2000, 45.0: -2000, 45.5: -2000, 46.0: -2000, 46.5: -2000, 50.0: -2000})]
    o = _occupation(O, fresh, [5, 6, 7], [48.5, 47.0, 45.0])
    assert [c for c, _ in o["members"]] == ["not_departed", "held_new", "held_new"] and o["reading"] == "held", "a run at 50 that is not the strip there before the entry holds it anew"
    assert _occupation(O, [_row(O, {})] * 4 + passage[4:], [5, 6, 7, 8], [60.0, 56.0, 53.0, 51.5])["reading"] == "no_reference", "one present day before the entry is too few"
    assert _occupation(O, passage, [5], [60.0]) is None, "no member inside the window"
    assert _occupation(O, passage, [6], [56.0]) is None, "a member inside the window on a one-member segment is not carried"
    s2 = {46.5: -2000, 47.0: -2000, 47.5: -2000}
    entry = pre + [_row(O, {**s2, x: -2000, x + 0.5: -2000}) for x in (66.0, 63.0, 60.0, 58.0, 56.0, 53.0)]
    o = _occupation(O, entry, [5, 6, 7, 8, 9, 10], [66.0, 63.0, 60.0, 58.0, 56.0, 53.0])
    assert o["reading"] == "held" and [c for c, _ in o["members"]] == ["held_continuous"] * 6, "the position is S2's over the five days before the first member inside the window (47 E), not before the first member (50 E)"
    weak = pre + [_row(O, {48.0: -2000, 48.5: -2000}), _row(O, {46.0: -2000, 46.5: -2000, 50.0: -1300}), _row(O, {44.0: -2000, 44.5: -2000, 50.0: -1300})]
    o = _occupation(O, weak, [5, 6, 7], [48.0, 46.0, 44.0])
    assert o["reading"] == "vacated" and o["sensitivity"] == (2, 2) and o["sensitivity_reading"] == "held", "-3.25 m/s at 50 does not hold it at -4 and does at -3"
    o = _occupation(O, tie, [6, 7], [56.0, 53.0])
    assert o["sensitivity_reading"] == "tied" and _occupation(O, shift, [5, 6, 7], [48.0, 46.0, 44.0])["sensitivity_reading"] == "vacated"


def test_counts_over_histories():
    O = _load()
    pre = [_row(O, STANDING)] * 5
    passage = np.array(pre + [_row(O, {**STANDING, 60.0: -2000, 60.5: -2000}), _row(O, {**STANDING, 56.0: -2000, 56.5: -2000}), _row(O, {**STANDING, 53.0: -2000, 53.5: -2000})])
    shift = np.array(pre + [_row(O, {48.0: -2000, 48.5: -2000}), _row(O, {46.0: -2000, 46.5: -2000}), _row(O, {44.0: -2000, 44.5: -2000})])
    fresh = np.array(pre + [_row(O, {48.0: -2000, 48.5: -2000, 49.0: -2000, 49.5: -2000, 53.0: -2000}),
                            _row(O, {46.5: -2000, 47.0: -2000, 47.5: -2000, 48.0: -2000, 50.0: -2000}),
                            _row(O, {44.5: -2000, 45.0: -2000, 45.5: -2000, 46.0: -2000, 46.5: -2000, 50.0: -2000})])
    Y, b = O.YEARS, "20 to 24.5 N"
    fields = {(Y[0], b): passage, (Y[1], b): shift, (Y[2], b): fresh, (Y[0], "10 to 14.5 N"): passage}
    seasons = {k: O.S.link_strips(v) for k, v in fields.items()}
    hist = {(Y[0], b): [{"days": [5, 6, 7], "lons": [60.0, 56.0, 53.0]}], (Y[1], b): [{"days": [5, 6, 7], "lons": [48.0, 46.0, 44.0]}],
            (Y[2], b): [{"days": [5, 6, 7], "lons": [48.5, 47.0, 45.0]}], (Y[0], "10 to 14.5 N"): [{"days": [5, 6, 7], "lons": [60.0, 56.0, 53.0]}]}
    r = O.occupation_counts(hist, seasons, fields)
    assert r["histories"] == 3 and r["history_reading"]["held"] == 2 and r["history_reading"]["vacated"] == 1, "S1's band is not S2's"
    assert (r["departed_member_days"], r["held_member_days"]) == (7, 5) and r["member_days"]["not_departed"] == 2
    assert r["member_days_in_window"] == {"held_continuous": 2, "held_new": 2, "vacated": 1, "not_departed": 2, "no_reference": 0}, "60 E and 44 E are outside the window"
    assert r["sensitivity"] == {"departed": 7, "held": 5}
    assert r["sensitivity_history_reading"] == {"held": 2, "vacated": 1, "tied": 0, "not_departed": 0, "no_reference": 0}
    weak = np.array(pre + [_row(O, {48.0: -2000, 48.5: -2000}), _row(O, {46.0: -2000, 46.5: -2000, 50.0: -1300}), _row(O, {44.0: -2000, 44.5: -2000, 50.0: -1300})])
    fields[(Y[3], b)] = weak; seasons[(Y[3], b)] = O.S.link_strips(weak); hist[(Y[3], b)] = [{"days": [5, 6, 7], "lons": [48.0, 46.0, 44.0]}]
    r4 = O.occupation_counts(hist, seasons, fields)
    assert r4["history_reading"]["vacated"] == 2 and r4["sensitivity_history_reading"]["held"] == 3 and r4["sensitivity_history_reading"]["vacated"] == 1, "the weak history reads vacated at -4 and held at -3"
    assert r4["reading_transitions"]["vacated"] == {"held": 1, "vacated": 1, "tied": 0, "not_departed": 0, "no_reference": 0}, "one vacated history turns held at -3 and one stays vacated"
    assert O.held_shares(r) == {"held_histories": Fraction(2, 3), "held_member_days": Fraction(5, 7)}
    assert O.reading(r).startswith("among the 3 histories with an in-window carried member that read held or vacated (of 3), held ones outnumber vacated ones, 2 to 1")
    assert O.reading({"histories": 2, "history_reading": {"held": 1, "vacated": 1}}) is None, "an even split carries no summary"
    assert O.reading({"histories": 5, "history_reading": {"held": 0, "vacated": 0}}) is None
    archive_like = {"histories": 63, "history_reading": {"held": 22, "vacated": 29, "tied": 7, "not_departed": 5, "no_reference": 0}}
    sentence = O.reading(archive_like)
    assert sentence.startswith("among the 51 histories with an in-window carried member that read held or vacated (of 63), vacated ones outnumber held ones, 29 to 22")
    assert not sentence.startswith("most"), "the majority is over the histories with either reading, not over every eligible history"
    assert r["reading_transitions"]["held"]["held"] == 2 and r["reading_transitions"]["vacated"]["vacated"] == 1 and sum(sum(v.values()) for v in r["reading_transitions"].values()) == 3


def test_constructions_rule_and_gate(tmp_path):
    O = _load()
    res = json.loads(json.dumps(O.run_constructions(), default=float))
    v = O.apply_rule(res)
    assert v == {"adequate": True, "failures": []}
    assert res["O2"]["history_reading"]["vacated"] > res["O2"]["history_reading"]["held"] and res["O3"]["history_reading"]["vacated"] == 0
    assert res["O6"]["history_reading"]["vacated"] > res["O6"]["history_reading"]["held"], "standing flow below the threshold reads like a shift, the stated blind spot"
    assert {c: (res[c]["histories"], res[c]["history_reading"]["held"], res[c]["history_reading"]["vacated"]) for c in O.CASES} == {
        "O1": (12, 2, 6), "O2": (15, 2, 13), "O3": (14, 14, 0), "O4": (17, 15, 2), "O5": (21, 14, 5), "O6": (22, 4, 9)}, "the constructions' counts, the histories following the moving feature only"
    edge = json.loads(json.dumps(res))
    edge["O2"]["history_reading"].update(held=3, vacated=7); edge["O2"]["held_member_days"] = 1; edge["O2"]["departed_member_days"] = 4
    assert O.apply_rule(edge)["adequate"] is True, "exactly on the O2 bars passes"
    edge["O2"]["history_reading"].update(held=4, vacated=6)
    assert O.apply_rule(edge)["failures"] == ["O2 held_histories_at_most 3/10 not met (0.4)"]
    gone = json.loads(json.dumps(res)); gone["O4"]["history_reading"].update(held=0, vacated=0)
    assert "O4 held_histories_at_least unavailable" in O.apply_rule(gone)["failures"]
    p = tmp_path / "c.json"
    stale = json.loads(json.dumps(res)); stale["O3"]["history_reading"]["held"] = 0; stale["O3"]["history_reading"]["vacated"] = 14
    for rec, run, msg in (({"generated_by": {"sha256": "x"}, "results": res, "verdict": v}, lambda: res, "different version"),
                          ({"generated_by": {"sha256": O.SCRIPT_SHA256_AT_LOAD}, "results": res, "verdict": v}, lambda: stale, "differs from a fresh run"),
                          ({"generated_by": {"sha256": O.SCRIPT_SHA256_AT_LOAD}, "results": stale, "verdict": v}, lambda: stale, "do not pass the rule"),
                          ({"generated_by": {"sha256": O.SCRIPT_SHA256_AT_LOAD}, "results": res, "verdict": {"adequate": True, "failures": ["x"]}}, lambda: res, "stored summary disagrees")):
        p.write_text(json.dumps(rec))
        try:
            O.check_constructions(str(p), rerun=run); assert False
        except SystemExit as ex:
            assert msg in str(ex), str(ex)
    p.write_text(json.dumps({"generated_by": {"sha256": O.SCRIPT_SHA256_AT_LOAD}, "results": res, "verdict": v}))
    assert O.check_constructions(str(p), rerun=lambda: res)["verdict"]["adequate"] is True
    text = O.constructions_text(res, v)
    assert text.count("\n| O") == 6 and "adequate under the rule of section 4, yes" in text


def _records(O, values, years):
    recs = {}
    for y in years:
        for m in O.MONTHS:
            n = O.A.MONTH_DAYS[m]
            times = [f"{(dt.date(y, m, 1) + dt.timedelta(days=d)).isoformat()}T{h:02d}" for d in range(n) for h in (0, 6, 12, 18)]
            bands = {b: np.full((len(times), len(O.LONS)), values.get(b, 0.0)).tolist() for b in O.SC.V.BANDS}
            recs.setdefault(y, {})[m] = {"year": y, "month": m, "times": times, "lons": O.LONS.tolist(), "bands": list(O.SC.V.BANDS), "band_means_700": bands}
    return recs


def test_archive_keeps_bands_apart(monkeypatch):
    O = _load()
    monkeypatch.setattr(O, "YEARS", O.YEARS[:1])
    by = _records(O, {"20 to 24.5 N": -5.0}, O.YEARS)
    hist = {(O.YEARS[0], "20 to 24.5 N"): [{"days": [10, 11, 12], "lons": [52.0, 51.0, 50.0]}], (O.YEARS[0], "25 to 29.5 N"): [{"days": [10, 11, 12], "lons": [52.0, 51.0, 50.0]}]}
    res = O.archive_results(by, hist)
    occ = res["occupation"]
    assert occ["25 to 29.5 N"]["histories"] == 0 and occ["15 to 19.5 N"]["histories"] == 0, "no qualifying flow in those bands"
    assert occ["20 to 24.5 N"]["histories"] == 0 and occ["pooled"]["histories"] == 0, "a flat field gives a strip that does not move, so no carried segment"
    assert res["reading"] is None and "reading under section 2.5, none" in O.archive_text(res)


def test_a_background_moved_before_entry_reads_vacated():
    """The review's construction (section 8): a strong background feature that moves from 54 to
    48 E just before a strong transient enters the window leaves the five-day reference near
    53 E, and the history reads vacated at both thresholds although two features of 10 m/s
    coexist. The reference must stay representative of the background's position."""
    O = _load()
    constant = lambda value: lambda t: value
    background = lambda t: 54.0 if t < 40 else (51.0 + 3.0 * np.cos(np.pi * (t - 40) / 2.0) if t < 42 else 48.0)
    transient = lambda t: 72.0 - 2.0 * (t - 35.0) if 35.0 <= t <= 61.0 else None
    H, centers = O.S.synth([(background, constant(10.0)), (transient, constant(10.0))])
    strips, by_day = O.S.link_strips(H)
    found = []
    for h in O.HI.link(O.HI.rep_minima("RB9", H, O.A.month_index(), O.LONS), "LN"):
        if O.HI.classify(h) == "westward" and O.HI.follows(h, centers[-1]):
            path = O.S.history_path(h, strips, by_day)
            o = O.history_occupation(path, O.S.member_classes(path, strips), by_day, H)
            if o is not None:
                found.append(o)
    assert len(found) == 1 and found[0]["reading"] == "vacated" and found[0]["sensitivity_reading"] == "vacated"
    held = sum(1 for c, _ in found[0]["members"] if c.startswith("held")); vac = sum(1 for c, _ in found[0]["members"] if c == "vacated")
    assert (held, vac) == (7, 13)
