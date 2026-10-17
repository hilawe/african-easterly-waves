"""The developed evolution reader (`EASTERN_DIRECTION_RECOMMENDATION_11_2026-10-16.md`,
stage A): continuity is unique or ambiguous depending on nearby deep alternatives, a
displacement phase is found along a chain of continuous pairs by a lagged displacement
rather than a run-averaged rate, so a moving phase inside persistent flow is a phase and
not one standing run, and identical records give identical outputs."""
import copy
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("evolution_phases_under_test", os.path.join(ROOT, "scripts", "pilot_entry_evolution_phases.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _step(t, bound, rows):
    out = {"time": t, "annotation": {"i1": {"lon": [bound, 63], "rows": [12, 30]}}, "rows": []}
    for row, (sel, mins) in rows.items():
        m = {"row_lat": row, "status": "assigned" if sel is not None else "abstained",
             "wind_minima": [{"lon": lon, "v": v, "label": lab} for lon, v, lab in mins]}
        if sel is not None:
            m["i1"] = {"lon": sel, "v": next(v for lon, v, lab in mins if lon == sel)}
        out["rows"].append(m)
    return out


def test_continuity_is_unique_or_ambiguous_by_nearby_deep_alternatives():
    P = _load()
    Q = dict(P.PRIMARY)
    steps = [
        _step("t0", 44, {22: (52.0, [(52.0, -10.0, "I1")])}),
        _step("t1", 44, {22: (51.0, [(51.0, -10.5, "I1"), (53.0, -8.5, "I1")])}),   # 53 lies within 1.5 of the earlier 52: ambiguous
        _step("t2", 44, {22: (50.0, [(50.0, -11.0, "I1"), (56.0, -8.5, "I1")])}),   # 56 is far: unique
        _step("t3", 44, {22: (49.0, [(49.0, -11.0, "I1"), (48.0, -8.1, "I1")])}),   # 48 is 2 degrees from the earlier 50: not a candidate continuation, so unique
        _step("t4", 44, {22: (48.0, [(48.0, -11.0, "I1"), (50.0, -8.1, "I1")])}),   # at t3 the alternative 48 lies within 1.5 of the later 48: the later selection could have come from it, ambiguous
    ]
    pairs = P.classify_pairs({"steps": steps}, Q)[22]
    assert [p["class"] for p in pairs] == ["continuous_ambiguous", "continuous_unique", "continuous_unique", "continuous_ambiguous"], [p["class"] for p in pairs]


def test_a_moving_phase_inside_persistent_flow_is_found_as_a_phase_not_one_standing_run():
    P = _load()
    Q = dict(P.PRIMARY)
    # 2 days standing at 52 E, 3 days moving west at 2 deg/day (0.5 per step, integer positions), 2 days standing at 46 E
    lons = [52.0] * 8 + [float(int(52 - 0.5 * k)) for k in range(1, 13)] + [46.0] * 8
    steps = [_step(f"t{k}", 44, {24: (lons[k], [(lons[k], -10.0, "I1")])}) for k in range(len(lons))]
    out = P.analyze({"steps": steps}, Q)
    phases = [ph for ph in out["phases"][24] if ph["direction"] == "west"]
    assert len(phases) == 1, out["phases"][24]
    ph = phases[0]
    assert 48 <= ph["hours"] <= 90 and -7.0 <= ph["net_move"] <= -4.0, ph
    stand = [s for s in out["standing"][24]]
    assert len(stand) >= 2 and all(s["hours"] >= 24 for s in stand), stand
    assert out["chains"][24][0]["pairs"] == 27, "the whole sequence is one chain of continuous pairs, and the phase is found inside it"


def test_durations_are_elapsed_hours_between_first_and_last_sample():
    """Nine samples span 48 elapsed hours and qualify as a long phase, eight samples span 42
    and do not, a one-sample stretch has zero hours and no rate, and a standing interval
    carries the range of its selections."""
    P = _load()
    Q = dict(P.PRIMARY)
    def rec(n_moving):
        lons = [float(int(55 - 0.5 * k)) for k in range(n_moving)]            # 0.5 deg per step west
        steps = [_step(f"t{k}", 44, {24: (lons[k], [(lons[k], -10.0, "I1")])}) for k in range(n_moving)]
        return {"steps": steps}
    out9 = P.analyze(rec(9), Q); out8 = P.analyze(rec(8), Q)
    ph9 = [p for p in out9["phases"][24] if p["direction"] == "west"]; ph8 = [p for p in out8["phases"][24] if p["direction"] == "west"]
    assert ph9[0]["hours"] == 48 and ph9[0]["n_steps"] == 9 and abs(ph9[0]["rate_deg_per_day"] - (-2.0)) < 1e-9, ph9
    assert ph8[0]["hours"] == 42 and ph8[0]["n_steps"] == 8, ph8
    assert out9["displacement_set"]["n_rows"] == 1 and out8["displacement_set"]["n_rows"] == 0, "48 elapsed hours qualifies, 42 does not"
    # a standing stretch whose selections wobble between two columns carries its range, and a one-sample moving label has no rate
    lons = [50.0, 50.0, 51.0, 51.0, 51.0, 50.0, 50.0, 50.0, 50.0, 50.0, 47.0]
    steps = [_step(f"t{k}", 44, {24: (lons[k], [(lons[k], -10.0, "I1")])}) for k in range(len(lons))]
    out = P.analyze({"steps": steps}, Q)
    stand = out["standing"][24]
    assert stand and stand[0]["lon_min"] == 50.0 and stand[0]["lon_max"] == 51.0 and stand[0]["hours"] == (stand[0]["n_steps"] - 1) * 6, stand
    one = [p for p in out["phases"][24] if p["n_steps"] == 1]
    assert all(p["hours"] == 0 and p["rate_deg_per_day"] is None for p in one), one


def test_identical_records_give_identical_outputs_and_the_set_criterion_still_needs_overlap():
    P = _load()
    Q = dict(P.PRIMARY)
    n = 12
    lons = [float(int(55 - 0.5 * k)) for k in range(n)]
    steps = []
    for k in range(2 * n):
        rows = {r: (lons[k] if k < n else lons[-1], [(lons[k] if k < n else lons[-1], -10.0 if k < n else -5.0, "I1")]) for r in (28, 27, 26, 25)}  # too shallow after the phase
        lon24 = 50.0 if k < n else lons[k - n]                                                                                              # 24 N moves only after the others end
        rows[24] = (lon24, [(lon24, -10.0, "I1")])
        steps.append(_step(f"t{k}", 44, rows))
    rec = {"steps": steps}
    a = P.analyze(rec, Q); b = P.analyze(copy.deepcopy(rec), Q)
    assert a == b
    best = a["displacement_set"]
    assert best["rows"] == [28, 27, 26, 25] and best["n_rows"] == 4, best
    assert 24 not in best["rows"], "row 24's phase does not overlap the others in time, so the overlap rule keeps it out"
