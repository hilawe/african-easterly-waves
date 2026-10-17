"""The entry summary's support rule counts a row only when the I1 minimum reaches the
threshold, the primary edge is available and the minimum sits the declared margin inside
the window's western bound."""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("entry_summary_under_test", os.path.join(ROOT, "scripts", "pilot_i1_entry_summary.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(lat, lon, v, edge):
    return {"row_lat": lat, "status": "assigned", "i1": {"lon": lon, "v": v}, "edge": {"primary": {"lon": edge}}, "reversal": {"lon": None}, "zs": None, "za": None}


def test_the_support_rule_applies_threshold_edge_and_margin():
    S = _load()
    step = {"annotation": {"i1": {"lon": [46, 56], "rows": [13, 28]}},
            "rows": [_row(26, 50, -12.0, 48), _row(24, 50, -7.0, 48), _row(22, 50, -12.0, None), _row(20, 47, -12.0, 47), _row(18, 48, -12.0, 47),
                     {"row_lat": 16, "status": "abstained"}]}
    assert S.support_rows(step, -8.0, 2.0) == [26, 18], "weak minimum, missing edge, a minimum at the window bound and an abstained row are excluded"


def test_the_minima_cell_lists_every_minimum_with_its_label_and_marks_the_selected_one():
    S = _load()
    m = {"row_lat": 20, "status": "assigned", "i1": {"lon": 51.0, "v": -11.2},
         "wind_minima": [{"lon": 41.0, "v": -12.5, "label": "W"}, {"lon": 51.0, "v": -11.2, "label": "I1"}, {"lon": 55.0, "v": -9.8, "label": "I1"}, {"lon": 63.0, "v": -1.0, "label": "other"}]}
    assert S.minima_cell(m) == "W41(-12.5) *51(-11.2) 55(-9.8) o63(-1.0)", "every minimum appears in longitude order, the selected one starred, W and other labeled"
    assert S.minima_cell({"row_lat": 20, "status": "abstained", "wind_minima": [{"lon": 41.0, "v": -12.5, "label": "W"}]}) == "abst W41(-12.5)", "an abstained row still lists its minima"


def test_jumps_lists_consecutive_step_moves_of_the_selected_minimum_beyond_the_threshold():
    S = _load()
    def step(t, lon):
        return {"time": t, "rows": [{"row_lat": 22, "status": "assigned", "i1": {"lon": lon, "v": -10.0}, "wind_minima": [{"lon": lon, "v": -10.0, "label": "I1"}]}]}
    rec = {"steps": [step("t0", 58.0), step("t1", 57.0), step("t2", 51.0), step("t3", 51.0)]}
    out = S.jumps(rec, 4.0)
    assert [(j["row"], j["from"], j["to"], j["lon_from"], j["lon_to"]) for j in out] == [(22, "t1", "t2", 57.0, 51.0)], "only the 6-degree move is a jump; the 1-degree move and the standstill are not"
    rec["steps"][2]["rows"][0]["status"] = "abstained"
    assert S.jumps(rec, 4.0) == [], "an abstained step breaks the pair and is not a jump"


def test_arrival_and_coexistence_are_read_from_the_record():
    S = _load()
    def step(t, lon, extra=()):
        mins = [{"lon": lon, "v": -10.0, "label": "I1"}] + [{"lon": x, "v": v, "label": "I1"} for x, v in extra]
        return {"time": t, "rows": [{"row_lat": 22, "status": "assigned", "i1": {"lon": lon, "v": -10.0}, "wind_minima": mins}]}
    rec = {"steps": [step("t0", 58.0), step("t1", 50.0, [(55.0, -9.0)]), step("t2", 56.0), step("t3", 51.0, [(53.0, -8.5), (60.0, -3.0)]), step("t4", 50.0)]}
    assert S.arrival(rec, 46.0, 52.0) == {22: "t3"}, "the arrival is the first step from which the selection stays inside the range to the end, not the first visit"
    assert S.coexistence(rec, -8.0, 3.0) == [(22, "t1", [50.0, 55.0])], "two deep in-window minima at least 3 degrees apart count; t3's pair is 2 degrees apart and its third minimum is shallow"
