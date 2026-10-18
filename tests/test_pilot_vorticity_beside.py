"""A second-field reading over the saved listings for the twelfth direction discussion
(`EASTERN_DIRECTION_RECOMMENDATION_12_2026-10-17.md`): at every step of a row's westward
phase, the nearest cyclonic vorticity maximum east of the selected wind minimum, and how
its position changes with the selection's. Positions, not features."""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("vorticity_beside_under_test", os.path.join(ROOT, "scripts", "pilot_vorticity_beside.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_nearest_cyclonic_maximum_east_of_the_selection_is_found_and_absence_is_counted():
    V = _load()
    def row(sel, maxima):
        return {"row_lat": 20, "status": "assigned", "i1": {"lon": sel, "v": -10.0}, "zeta_maxima": [{"lon": a, "zeta_1e5": b, "label": "other"} for a, b in maxima]}
    steps = [{"time": "2000-07-01T00", "rows": [row(52.0, [(50.0, 3.0), (54.0, 2.5), (61.0, 4.0)])]},
             {"time": "2000-07-01T06", "rows": [row(51.0, [(53.5, 2.2)])]},
             {"time": "2000-07-01T12", "rows": [row(50.0, [(49.0, 3.0), (58.5, 1.0)])]}]
    out = V.beside(steps, 20, 0, 2, min_zeta=2.0, max_east=8.0)
    assert [o["offset"] for o in out] == [2.0, 2.5, None], "the maximum east of the selection within 8 degrees, and none when the only eastern one is too weak"
    summary = V.summarize(out)
    assert summary["steps"] == 3 and summary["with_maximum"] == 2 and summary["offset_median"] == 2.25


# ---------------------------------------------- part one of the history-layer package (2026-10-17)

def _res_and_listing():
    """A one-row, five-step listing with the frozen reader's labels for it: steps 0 to 3 in
    one chain (motion west, west, standing, standing), step 4 unsupported."""
    def row(sel, maxima, assigned=True):
        return {"row_lat": 20, "status": "assigned" if assigned else "nomin", "i1": {"lon": sel, "v": -10.0} if assigned else None,
                "zeta_maxima": [{"lon": a, "zeta_1e5": b, "label": "other"} for a, b in maxima]}
    # Strengthened at the review of 2026-10-17 (F5): a continuous pair whose maxima are both within
    # the distance but whose offsets differ by 4.5 degrees (steps 1 to 2), a supported step outside
    # every chain beside a switching pair (step 5), and an isolated supported step whose touching
    # pairs are both unassessable (step 7), so that dropping the co-movement bound or folding breaks
    # into unassessable each fails this test.
    steps = [{"time": "2000-07-01T00", "rows": [row(54.0, [(55.5, 3.0)])]},
             {"time": "2000-07-01T06", "rows": [row(53.0, [(54.5, 3.0)])]},
             {"time": "2000-07-01T12", "rows": [row(52.0, [(58.0, 3.0)])]},
             {"time": "2000-07-01T18", "rows": [row(52.0, [(60.5, 3.0)])]},
             {"time": "2000-07-02T00", "rows": [row(None, [], assigned=False)]},
             {"time": "2000-07-02T06", "rows": [row(49.0, [(50.5, 3.0)])]},
             {"time": "2000-07-02T12", "rows": [row(None, [], assigned=False)]},
             {"time": "2000-07-02T18", "rows": [row(47.0, [(48.5, 3.0)])]},
             {"time": "2000-07-03T00", "rows": [row(None, [], assigned=False)]}]
    res = {"motion": {"20": [[0, "west"], [1, "west"], [2, "standing"], [3, "standing"]]},
           "pairs": {"20": [{"row": 20, "k": 0, "class": "continuous_unique"}, {"row": 20, "k": 1, "class": "continuous_unique"}, {"row": 20, "k": 2, "class": "continuous_unique"}, {"row": 20, "k": 3, "class": "unassessable"},
                            {"row": 20, "k": 4, "class": "switching"}, {"row": 20, "k": 5, "class": "unassessable"}, {"row": 20, "k": 6, "class": "unassessable"}, {"row": 20, "k": 7, "class": "unassessable"}]}}
    return res, {"steps": steps}


def test_every_row_step_is_classed_by_the_reader_labels_and_counted_with_its_nearest_maximum():
    V = _load()
    res, rec = _res_and_listing()
    cls = V.classify_steps(res, rec, 20)
    assert cls == ["west", "west", "standing", "standing", "unassessable", "break", "unassessable", "unassessable", "unassessable"], cls
    T = V.beside_all(res, rec, rows=[20], min_zeta=2.0, max_east=8.0)
    by = {t["class"]: t for t in T}
    assert by["west"]["steps"] == 2 and by["west"]["with_maximum"] == 2 and by["west"]["offset_median"] == 1.5
    assert by["standing"]["steps"] == 2 and by["standing"]["with_maximum"] == 1, "at step 3 the only maximum is 8.5 degrees east, beyond the distance"
    assert by["break"]["steps"] == 1 and by["break"]["with_maximum"] == 1, "a supported step beside a switching pair is a break, not unassessable"
    assert by["unassessable"]["steps"] == 4 and by["unassessable"]["with_maximum"] == 1, "the isolated supported step at 02-18 is unassessable because both touching pairs are, and it keeps its maximum"
    # the two continuous pairs starting in the west class both have a maximum at both ends; (0,1) moves -1 and -1,
    # co-moving; (1,2) moves -1 and +3.5, not co-moving within 1.5
    assert by["west"]["continuous_pairs_total"] == 2 and by["west"]["pairs_with_both"] == 2 and by["west"]["pairs_comoving_within_1.5"] == 1, by["west"]
    assert by["standing"]["continuous_pairs_total"] == 1 and by["standing"]["pairs_with_both"] == 0
