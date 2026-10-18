"""The stage E reporting supplement (`EASTERN_STAGE_E_EVALUATION_2026-10-17.md`, section 15,
from the independent review's F7): every table the evaluation cites is printed from the
saved reader output and listing with its predicate, and the switching partition is checked
to be disjoint and exhaustive, with any unassigned pair printed rather than dropped."""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("stage_e_report_under_test", os.path.join(ROOT, "scripts", "pilot_stage_e_report.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pairs():
    def p(row, k, cls, lon0, lon1):
        t0 = f"2000-07-0{1 + (k * 6) // 24}T{(k * 6) % 24:02d}"; t1 = f"2000-07-0{1 + ((k + 1) * 6) // 24}T{((k + 1) * 6) % 24:02d}"
        return {"row": row, "k": k, "from": t0, "to": t1, "lon_from": lon0, "lon_to": lon1, "move": lon1 - lon0, "class": cls}
    return {20: [p(20, 0, "continuous_unique", 50, 50), p(20, 1, "switching", 50, 55), p(20, 2, "switching", 55, 50)],
            19: [p(19, 0, "switching", 52, 56), p(19, 1, "continuous_unique", 56, 56), p(19, 2, "continuous_unique", 56, 55.5)],
            12: [p(12, 0, "switching", 48, 53), p(12, 1, "continuous_unique", 53, 53), p(12, 2, "switching", 53, 48)]}


def test_the_switching_partition_is_disjoint_exhaustive_and_names_what_it_leaves_out():
    R = _load()
    pairs = _pairs()
    predicates = [("rows 19 to 20 N, early", (19, 20), "2000-07-01T00", "2000-07-01T12"), ("rows 12 to 15 N, all times", (12, 15), None, None)]
    part = R.switching_partition(pairs, predicates)
    assert [len(part["groups"][n]) for n, *_ in predicates] == [2, 2], part
    assert [(p["row"], p["from"]) for p in part["unassigned"]] == [(20, "2000-07-01T12")], "the late switch on row 20 is outside both predicates and must be listed, not dropped"
    assert part["total"] == 5 and part["assigned"] == 4
    nested = predicates + [("rows 19 to 20 N, all times, outside the earlier predicates", (19, 20), None, None)]
    part2 = R.switching_partition(pairs, nested)
    assert [len(part2["groups"][n]) for n, *_ in nested] == [2, 2, 1] and part2["unassigned"] == [], "a later predicate takes only what the earlier ones left, so the groups stay disjoint and the late switch is now assigned"
    assert sum(len(g) for g in part2["groups"].values()) == part2["total"] == 5


def test_standing_clipped_to_a_window_keeps_only_the_hours_inside_it():
    R = _load()
    standing = {27: [{"start": "2000-07-02T00", "end": "2000-07-03T00", "hours": 24}], 26: [{"start": "2000-07-02T12", "end": "2000-07-03T12", "hours": 24}]}
    out = R.standing_inside(standing, "2000-07-02T06", "2000-07-03T12")
    assert out[27] == 18 and out[26] == 24, out
