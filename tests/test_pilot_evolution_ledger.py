"""The evolution ledger (`EASTERN_EVOLUTION_DEVELOPMENT_2_2026-10-17.md`, section 2): every
count it prints carries the exact subset predicate it was computed under, and the
classification of a large selected-position change by the alternatives at both ends
distinguishes a lone minimum that shifts from coexisting minima and from a minimum that
appears or is lost."""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("ledger_under_test", os.path.join(ROOT, "scripts", "pilot_evolution_ledger.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _record_and_pairs():
    """A two-row, four-step listing and the pair classes a reader would give it."""
    def m(row, sel, mins):
        return {"row_lat": row, "status": "assigned", "i1": {"lon": sel, "v": -10.0}, "wind_minima": [{"lon": a, "v": b, "label": "I1"} for a, b in mins]}
    steps = [
        {"time": "1999-07-01T00", "annotation": {"i1": {"lon": [44, 63], "rows": [12, 30]}}, "rows": [m(20, 52.0, [(52.0, -10.0)]), m(19, 55.0, [(55.0, -10.0), (50.0, -9.0)])]},
        {"time": "1999-07-01T06", "annotation": {"i1": {"lon": [44, 63], "rows": [12, 30]}}, "rows": [m(20, 50.0, [(50.0, -10.0)]), m(19, 50.0, [(50.0, -11.0), (55.0, -9.0)])]},
        {"time": "1999-07-01T12", "annotation": {"i1": {"lon": [44, 63], "rows": [12, 30]}}, "rows": [m(20, 50.0, [(50.0, -10.0)]), m(19, 58.0, [(58.0, -9.5), (55.0, -9.0)])]},
        {"time": "1999-07-01T18", "annotation": {"i1": {"lon": [44, 63], "rows": [12, 30]}}, "rows": [m(20, 50.0, [(50.0, -10.0)]), m(19, 55.0, [(55.0, -10.0)])]},
    ]
    pairs = {20: [{"row": 20, "k": 0, "from": "1999-07-01T00", "to": "1999-07-01T06", "lon_from": 52.0, "lon_to": 50.0, "move": -2.0, "class": "jump"},
                  {"row": 20, "k": 1, "from": "1999-07-01T06", "to": "1999-07-01T12", "lon_from": 50.0, "lon_to": 50.0, "move": 0.0, "class": "continuous_unique"},
                  {"row": 20, "k": 2, "from": "1999-07-01T12", "to": "1999-07-01T18", "lon_from": 50.0, "lon_to": 50.0, "move": 0.0, "class": "continuous_unique"}],
             19: [{"row": 19, "k": 0, "from": "1999-07-01T00", "to": "1999-07-01T06", "lon_from": 55.0, "lon_to": 50.0, "move": -5.0, "class": "switching"},
                  {"row": 19, "k": 1, "from": "1999-07-01T06", "to": "1999-07-01T12", "lon_from": 50.0, "lon_to": 58.0, "move": 8.0, "class": "jump"},
                  {"row": 19, "k": 2, "from": "1999-07-01T12", "to": "1999-07-01T18", "lon_from": 58.0, "lon_to": 55.0, "move": -3.0, "class": "jump"}]}
    return {"steps": steps}, pairs


def test_counts_carry_their_predicate_and_the_change_classification_reads_both_ends():
    G = _load()
    record, pairs = _record_and_pairs()
    out = G.pair_counts(pairs, rows=(19, 20), t_from="1999-07-01T00", t_to="1999-07-01T12", lon_range=(40.0, 60.0), classes=("jump", "switching", "replacement"))
    assert out["predicate"].startswith("rows 19 to 20") and "lon 40 to 60" in out["predicate"] and "1999-07-01T00" in out["predicate"]
    assert out["counts"] == {"jump": 2, "switching": 1, "replacement": 0}, "the pair ending at 12Z is inside the time predicate and the one ending at 18Z is outside it"
    narrow = G.pair_counts(pairs, rows=(19, 20), t_from="1999-07-01T00", t_to="1999-07-01T12", lon_range=(40.0, 56.0), classes=("jump", "switching", "replacement"))
    assert narrow["counts"] == {"jump": 1, "switching": 1, "replacement": 0}, "the jump to 58 E has one end outside the longitude predicate"
    C = G.classify_changes(record, pairs, depth_ms=-8.0, bound_deg=1.5, lone_deg=3.0, min_abs_move=2.0)
    kinds = {(c["row"], c["from"]): c["kind"] for c in C}
    assert kinds[(20, "1999-07-01T00")] == "lone-shift", "one deep minimum at each end, none within 3 degrees of the other end: a lone minimum that shifted"
    assert kinds[(19, "1999-07-01T00")] == "coexisting", "a deep alternative within the bound of the other end's selection at each end"
    assert kinds[(19, "1999-07-01T06")] == "appeared", "the later selection has no deep minimum near it at the earlier step"
    assert kinds[(19, "1999-07-01T12")] == "lost", "the new selection pre-existed as an alternative while the old one has no deep minimum near it afterwards"
    assert (20, "1999-07-01T06") not in kinds, "a zero move is below the minimum"


def test_counterparts_pair_each_change_with_the_same_row_and_time_in_another_record():
    """The cross-grid table: every classified change in one record is paired with the pair
    at the same row and start time in another record, so the two grids' readings of one
    row-step sit side by side, and a row-step absent from the other record is counted as
    missing rather than dropped."""
    G = _load()
    record, pairs = _record_and_pairs()
    C = G.classify_changes(record, pairs, depth_ms=-8.0, bound_deg=1.5, lone_deg=3.0, min_abs_move=2.0)
    other = {20: [{"row": 20, "k": 0, "from": "1999-07-01T00", "to": "1999-07-01T06", "lon_from": 51.5, "lon_to": 50.5, "move": -1.0, "class": "continuous_unique"}],
             19: [{"row": 19, "k": 0, "from": "1999-07-01T00", "to": "1999-07-01T06", "lon_from": 55.0, "lon_to": 50.0, "move": -5.0, "class": "switching"},
                  {"row": 19, "k": 1, "from": "1999-07-01T06", "to": "1999-07-01T12", "lon_from": 50.0, "lon_to": 57.5, "move": 7.5, "class": "jump"}]}
    T = G.counterparts(C, other)
    by = {(t["row"], t["from"]): t for t in T}
    assert by[(20, "1999-07-01T00")]["other_class"] == "continuous_unique" and by[(20, "1999-07-01T00")]["other_move"] == -1.0
    assert by[(19, "1999-07-01T06")]["other_class"] == "jump" and by[(19, "1999-07-01T06")]["other_move"] == 7.5
    assert by[(19, "1999-07-01T12")]["other_class"] == "missing" and by[(19, "1999-07-01T12")]["other_move"] is None
    table = G.counterpart_table(T)
    assert table[("lone-shift", "continuous_unique")] == 1 and table[("coexisting", "switching")] == 1 and table[("appeared", "jump")] == 1 and table[("lost", "missing")] == 1


def test_competing_minima_and_selection_offsets_are_counted_under_their_predicate():
    """Two listing-level counts the diagnosis cites: row-steps with two or more deep
    in-window minima, and how far the selection in one listing sits from the selection in
    another listing of the same interval (the other grid), both under a stated subset."""
    G = _load()
    record, _ = _record_and_pairs()
    d = G.deep_minima_counts(record, depth_ms=-8.0, rows=(19, 20), t_from="1999-07-01T00", t_to="1999-07-01T12")
    assert d["assigned"] == 6 and d["two_or_more"] == 3, "row 19 carries two deep minima at 00Z, 06Z and 12Z; the 18Z step is outside the predicate"
    assert "rows 19 to 20" in d["predicate"] and "1999-07-01T12" in d["predicate"]
    def shifted(m):
        return {"row_lat": m["row_lat"], "status": "assigned", "i1": {"lon": m["i1"]["lon"] + (0.5 if m["row_lat"] == 20 else 0.0)}}
    other = {"steps": [{"time": s["time"], "rows": [shifted(m) for m in s["rows"]]} for s in record["steps"]]}
    other["steps"][0]["rows"][1]["i1"]["lon"] = 50.0  # row 19 at 00Z: 55 in the record, 50 in the other listing
    other["steps"][2]["rows"][1]["status"] = "abstained"
    o = G.selection_offsets(record, other, rows=(19, 20), t_from="1999-07-01T00", t_to="1999-07-01T12")
    assert o["both_assigned"] == 5 and o["same"] == 1 and o["half_degree"] == 3 and o["farther"] == {5.0: 1}, o


def test_change_classification_takes_only_non_continuous_pairs():
    """Regression from the independent review of 2026-10-17 (F5): under the 2.5-degree
    sensitivity bound a two-degree change is a continuous pair, and the change table must
    not classify it, because a single moving minimum would then read as `coexisting`
    (the selections themselves lie within the enlarged bound). The table carries the
    non-continuous classes its printed predicate names."""
    G = _load()
    record, pairs = _record_and_pairs()
    wide = {20: [{**pairs[20][0], "class": "continuous_unique"}], 19: [dict(p) for p in pairs[19]]}
    C = G.classify_changes(record, wide, depth_ms=-8.0, bound_deg=2.5, lone_deg=3.0, min_abs_move=2.0)
    assert all(c["class"] in ("jump", "replacement", "switching") for c in C), [c["class"] for c in C]
    assert (20, "1999-07-01T00") not in {(c["row"], c["from"]) for c in C}, "a continuous pair of two degrees under the wider bound is not a change to classify"
