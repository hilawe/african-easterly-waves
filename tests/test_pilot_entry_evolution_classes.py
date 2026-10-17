"""The displacement-against-switching classification of consecutive-step evolution on an
entry listing (`EASTERN_DISPLACEMENT_SWITCHING_2026-10-16.md`, section 3): pairs are
unassessable, continuous, switching, replacement or jump under the declared rule, runs of
continuous pairs are displacement, standing or short by duration and rate, and the
outcome criterion is the largest set of adjacent rows with long displacement runs that
overlap in time."""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("evolution_classes_under_test", os.path.join(ROOT, "scripts", "pilot_entry_evolution_classes.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _step(t, bound, rows):
    """rows: {row: (selected_lon_or_None, [(lon, v, label), ...])}"""
    out = {"time": t, "annotation": {"i1": {"lon": [bound, 63], "rows": [12, 30]}}, "rows": []}
    for row, (sel, mins) in rows.items():
        m = {"row_lat": row, "status": "assigned" if sel is not None else "abstained",
             "wind_minima": [{"lon": lon, "v": v, "label": lab} for lon, v, lab in mins]}
        if sel is not None:
            m["i1"] = {"lon": sel, "v": next(v for lon, v, lab in mins if lon == sel)}
        out["rows"].append(m)
    return out


def test_pairs_take_the_five_declared_classes():
    C = _load()
    P = dict(C.PRIMARY)
    # row 20: continuous (1 degree), then switching (both persist), then replacement (old lost), then jump (neither), then unassessable (shallow)
    steps = [
        _step("t0", 44, {20: (55.0, [(55.0, -10.0, "I1")])}),
        _step("t1", 44, {20: (54.0, [(54.0, -10.5, "I1"), (50.0, -8.5, "I1")])}),
        _step("t2", 44, {20: (50.0, [(50.0, -11.0, "I1"), (54.0, -8.2, "I1")])}),
        _step("t3", 44, {20: (57.0, [(57.0, -9.0, "I1")])}),          # 50 lost, 57 did not preexist within 1.5 of 57 at t2? 54 is 3 away: neither -> jump
        _step("t4", 44, {20: (49.0, [(49.0, -9.0, "I1")])}),          # 57 lost, 49 did not preexist -> jump
        _step("t5", 44, {20: (45.0, [(45.0, -9.0, "I1")])}),          # at the bound + 1: unsupported by the margin -> unassessable
    ]
    pairs = C.classify_pairs({"steps": steps}, P)[20]
    assert [p["class"] for p in pairs] == ["continuous", "switching", "jump", "jump", "unassessable"], [p["class"] for p in pairs]
    # replacement: the new minimum preexisted but the old one is gone
    steps2 = [
        _step("t0", 44, {20: (55.0, [(55.0, -10.0, "I1"), (50.0, -8.5, "I1")])}),
        _step("t1", 44, {20: (50.0, [(50.0, -11.0, "I1")])}),
    ]
    p = C.classify_pairs({"steps": steps2}, P)[20][0]
    assert p["class"] == "replacement" and p["detail"] == "old-lost"
    # a shallow selected minimum makes the pair unassessable even when the move is small
    steps3 = [_step("t0", 44, {20: (55.0, [(55.0, -7.9, "I1")])}), _step("t1", 44, {20: (55.0, [(55.0, -10.0, "I1")])})]
    assert C.classify_pairs({"steps": steps3}, P)[20][0]["class"] == "unassessable"


def test_runs_are_displacement_standing_or_short_by_duration_and_rate():
    C = _load()
    P = dict(C.PRIMARY)
    # row 24: 1 degree west every two pairs over 8 pairs (48 h): net -4 over 2 days = 2 deg/day -> displacement
    lons24 = [56, 56, 55, 55, 54, 54, 53, 53, 52]
    # row 22: standing at 50 for 5 pairs (30 h), then a switch, then 2 continuous pairs (short)
    lons22 = [50, 50, 50, 50, 50, 50, 56, 56, 56]
    steps = []
    for k in range(9):
        rows = {24: (float(lons24[k]), [(float(lons24[k]), -10.0, "I1")]),
                22: (float(lons22[k]), [(float(lons22[k]), -10.0, "I1")] + ([(50.0, -8.5, "I1")] if k >= 6 else []) + ([(56.0, -8.5, "I1")] if k == 5 else []))}
        steps.append(_step(f"t{k}", 44, rows))
    R = C.runs({"steps": steps}, P)
    r24 = R[24]; r22 = R[22]
    assert len(r24) == 1 and r24[0]["class"] == "displacement" and r24[0]["pairs"] == 8 and r24[0]["net_move"] == -4.0
    assert [r["class"] for r in r22] == ["standing", "short"], [r["class"] for r in r22]
    assert r22[0]["pairs"] == 5 and r22[1]["pairs"] == 2
    # the pair at the switch is switching, not part of any run
    assert C.classify_pairs({"steps": steps}, P)[22][5]["class"] == "switching"


def test_the_outcome_criterion_needs_adjacent_rows_with_overlapping_long_displacement_runs():
    C = _load()
    P = dict(C.PRIMARY)
    def seq(start, n, step_per_pair):
        return [start - step_per_pair * k for k in range(n)]
    n = 10  # 9 pairs, 54 h of displacement on rows 25 to 28, after which their minima are too shallow to be supported
    moving = [float(x // 1) for x in seq(55, n, 0.5)]                    # 0.5 deg per pair -> 2 deg/day
    steps = []
    for k in range(2 * n):
        rows = {}
        for row in (28, 27, 26, 25, 22):                                 # 22 N displaces too but is not adjacent to the others
            lon = moving[k] if k < n else moving[-1]
            rows[row] = (lon, [(lon, -10.0 if k < n else -5.0, "I1")])
        lon24 = 50.0 if k < n else moving[k - n]                          # row 24 stands first and displaces only after the others end
        rows[24] = (lon24, [(lon24, -10.0, "I1")])
        steps.append(_step(f"t{k}", 44, rows))
    best = C.displacement_set({"steps": steps}, P)
    assert best["rows"] == [28, 27, 26, 25] and best["n_rows"] == 4 and best["meets_criterion"], best
    assert 24 not in best["rows"], "row 24's displacement run does not overlap the others in time, so the overlap rule keeps it out"
    P2 = dict(P); P2["long_run_hours"] = 72
    assert C.displacement_set({"steps": steps}, P2)["n_rows"] == 0, "54-hour runs do not meet a 72-hour requirement"
