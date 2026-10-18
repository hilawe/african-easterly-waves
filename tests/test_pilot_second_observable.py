"""The second-observable development package's candidates must behave as section 2 of
`EASTERN_SECOND_OBSERVABLE_DEVELOPMENT_2026-10-17.md` declares, including the two defects
the round verified (a standing pattern scores Rmax under C0, a one-flank growth reads as
translation under A0) and the repairs that turn each into a rejection or an abstention;
the scorer must apply the declared adequacy rule and nothing looser."""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("second_observable_under_test", os.path.join(ROOT, "scripts", "pilot_second_observable.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _trough(M, rate, depth=-12.0, growth=0.0, mask=None):
    return M.sample(lambda X, Y, t: M.gauss2(X, Y, 56.0 - rate * t, 20.0, depth + growth * t, 3.0, 2.0), nsteps=5, mask=mask)


def test_a0_recovers_a_clean_translation_and_rejects_a_standing_trough_with_no_tendency():
    M = _load()
    moving = _trough(M, 2.5)
    r = M.advection_fit(moving, 2, 54.5, 20.0)
    assert r["outcome"] == "supported" and abs(r["c_x"] + 2.5) < 0.3 and r["r2"] > 0.85, r
    standing = _trough(M, 0.0)
    r = M.advection_fit(standing, 2, 56.0, 20.0)
    assert r["outcome"] == "rejected" and r["reason"] == "no tendency" and r["S"] == 0.0, r
    assert M.advection_fit(moving, 0, 56.0, 20.0)["reason"] == "window end"


def test_a0_reads_one_flank_growth_as_translation_and_a1_abstains_on_it():
    M = _load()
    one_flank = _trough(M, 0.0, depth=-6.0, growth=-8.0 / 3.0, mask=lambda X, Y, t: X <= 55.0)
    r0 = M.advection_fit(one_flank, 2, 55.0, 20.0)
    assert r0["outcome"] == "supported" and r0["c_x"] < -1.0 and r0["r2"] > 0.8, ("the verified defect: growth on one flank fits as westward translation", r0)
    r1 = M.advection_fit(one_flank, 2, 55.0, 20.0, growth=True, both_flanks=True)
    assert r1["outcome"] == "abstained" and r1["reason"] in ("one flank", "ill-conditioned"), r1
    both = _trough(M, 0.0, depth=-6.0, growth=-8.0 / 3.0)
    r2 = M.advection_fit(both, 2, 56.0, 20.0, growth=True, both_flanks=True)
    assert r2["outcome"] == "rejected" and abs(r2["c_x"]) < 0.3 and r2["g"] > 0.2, ("with both flanks in view the growth term takes the tendency (the trough deepens, so v and its tendency share a sign) and the speed is near zero", r2)
    r3 = M.advection_fit(both, 2, 56.0, 20.0)
    assert r3["outcome"] == "rejected" and abs(r3["c_x"]) < 0.3, ("growth with both flanks in view is orthogonal to the gradients even without the growth term", r3)


def test_c0_scores_a_standing_pattern_at_rmax_and_c1_rejects_it_as_not_westward():
    M = _load()
    standing = _trough(M, 0.0)
    r0 = M.shift_correlation(standing, 2, 56.0, 20.0, 56.0)
    assert r0["outcome"] == "supported" and r0["dx"] == 0.0 and r0["S"] > 0.99, ("the verified defect", r0)
    r1 = M.shift_correlation(standing, 2, 56.0, 20.0, 56.0, refine=True, westward=True)
    assert r1["outcome"] == "rejected" and r1["reason"] == "pattern not westward" and r1["S"] == 0.0, r1
    moving = _trough(M, 2.5)
    r = M.shift_correlation(moving, 2, 54.5, 20.0, 54.0, refine=True, westward=True)
    assert r["outcome"] == "supported" and -1.0 < r["dx_refined"] < -0.5 and r["dx"] in (-0.5, -1.0), ("a 0.625 degree step refines between cells", r)
    flat = M.sample(lambda X, Y, t: np.zeros_like(X) - 9.0, nsteps=3)
    assert M.shift_correlation(flat, 0, 50.0, 20.0, 50.0)["reason"] == "flat"


def test_the_scorer_applies_the_declared_adequacy_rule_exactly():
    M = _load()
    def results(sup, rej, abst, c_x=None):
        rs = [{"outcome": "supported", "reason": "", "S": 2.0} for _ in range(sup)] + [{"outcome": "rejected", "reason": "", "S": 0.1} for _ in range(rej)] + [{"outcome": "abstained", "reason": "flat"} for _ in range(abst)]
        if c_x is not None:
            for r in rs:
                if r["outcome"] != "abstained":
                    r["c_x"] = c_x
        return rs
    neg = {"answer": "none", "speed": None, "kind": "negative"}
    pos = {"answer": "translation", "speed": 2.5, "kind": "positive"}
    assert M.summarize(results(0, 19, 1), neg)["verdict"] == "adequate"
    assert M.summarize(results(1, 9, 0), neg)["verdict"] == "not adequate", "one false support in ten assessed is 0.10"
    assert M.summarize(results(0, 0, 5), neg)["verdict"] == "not assessed", "abstaining everywhere does not pass a negative construction"
    assert M.summarize(results(9, 1, 0, c_x=-2.6), pos)["verdict"] == "adequate"
    assert M.summarize(results(7, 3, 0, c_x=-2.6), pos)["verdict"] == "not adequate"
    assert M.summarize(results(9, 1, 11, c_x=-2.6), pos)["verdict"] == "not adequate", "abstaining on more than half of the asked steps fails the positive rule"
    assert M.summarize(results(9, 1, 0, c_x=-1.8), pos)["verdict"] == "not adequate", "a speed error of 0.7 exceeds the declared 0.5"
    assert M.summarize(results(0, 10, 0), {"answer": "translation", "speed": 6.0, "kind": "limit"})["verdict"] == "reported"


def test_selections_and_pairs_follow_the_listing_and_the_pair_bound():
    M = _load()
    field = M.sample(lambda X, Y, t: M.gauss2(X, Y, 52.0, 20.0, -10.0, 1.5, 2.0) + M.gauss2(X, Y, 58.0, 20.0, -2.0 - 12.0 * t / 3.0, 1.5, 2.0), nsteps=13)
    sel = M.selections(field, rows=[20])[20]
    lons = [None if s is None else s[1] for s in sel]
    assert lons[0] == 52.0 and lons[-1] == 58.0 and all(x in (52.0, 58.0) for x in lons if x is not None), lons
    pairs = M.continuous_pairs(sel)
    assert all(abs(x1 - x0) <= 1.5 for _, x0, x1 in pairs) and len(pairs) == 11, ("the one 6 degree jump is not a continuous pair", pairs)
    smooth = M.smoothed(field)
    assert smooth["v"].shape == field["v"].shape and not smooth["valid"][:, 0, :].any() and smooth["valid"][:, 5:-5, 5:-5].all()
    assert abs(smooth["v"][0, 20, 44] - field["v"][0, 19:22, 43:46].mean()) < 1e-12, "the 3 by 3 mean at a cell"
    shallow = M.sample(lambda X, Y, t: M.gauss2(X, Y, 52.0, 20.0, -5.0, 2.5, 2.0), nsteps=3)
    assert M.selections(shallow, rows=[20])[20] == [None, None, None], "a minimum above the depth is not supported"


def test_the_neighborhood_scale_parameter_reaches_both_candidates():
    M = _load()
    moving = _trough(M, 2.5)
    a = M.advection_fit(moving, 2, 54.5, 20.0, growth=True, both_flanks=True)
    b = M.advection_fit(moving, 2, 54.5, 20.0, growth=True, both_flanks=True, half=(12, 9))
    assert a["n_used"] == 17 * 13 and b["n_used"] == 25 * 19 and b["outcome"] == "supported" and abs(b["c_x"] + 2.5) < 0.3, (a["n_used"], b)
    c = M.shift_correlation(moving, 2, 54.5, 20.0, 54.0, refine=True, westward=True, half=(12, 8))
    assert c["outcome"] == "supported" and -1.0 < c["dx_refined"] < -0.5 and c["n_patch"] == 25 * 17, c
    assert M.shift_correlation(moving, 2, 54.5, 20.0, 54.0)["n_patch"] == 13 * 9
    rows = M.run_scales(ids=("K1",))
    assert [r["half_degrees"] for r in rows] == [[4.0, 3.0], [6.0, 4.5], [8.0, 6.0], [3.0, 2.0], [4.5, 3.0], [6.0, 4.0]]
    assert all(r["summary"]["verdict"] == "adequate" for r in rows), [r["summary"]["verdict"] for r in rows]


def test_degenerate_design_columns_abstain_as_ill_conditioned_instead_of_crashing():
    """Review of 2026-10-18, F1: a field constant along latitude has a zero meridional-gradient
    column, and a pure exponential zonal profile makes v and its zonal gradient proportional."""
    M = _load()
    flat_in_y = M.sample(lambda X, Y, t: -12.0 * np.exp(-0.5 * ((X - (56.0 - 2.5 * t)) / 3.0) ** 2), nsteps=5)
    for growth, flanks in ((True, True), (False, False)):
        r = M.advection_fit(flat_in_y, 2, 54.5, 20.0, growth=growth, both_flanks=flanks)
        assert r["outcome"] == "abstained" and r["reason"] == "ill-conditioned", (growth, r)
    r = M.advection_fit(M.smoothed(flat_in_y), 2, 54.5, 20.0, growth=True, both_flanks=True)
    assert r["outcome"] == "abstained" and r["reason"] == "ill-conditioned", r
    exponential = M.sample(lambda X, Y, t: -(6.0 + 8.0 * t / 3.0) * np.exp(0.5 * (X - 56.0)) * np.exp(-0.5 * ((Y - 20.0) / 2.0) ** 2), nsteps=5)
    r = M.advection_fit(exponential, 2, 56.0, 20.0, growth=True, both_flanks=True)
    assert r["outcome"] == "abstained" and r["reason"] == "ill-conditioned" and r["cond"] > 1e6, ("both flanks in view, nonzero columns, growth and translation underdetermined", r)
    r0 = M.advection_fit(exponential, 2, 56.0, 20.0)
    assert r0["outcome"] != "abstained" and r0["r2"] > 0.99 and r0["c_x"] < -0.5, ("A0 fits the growth as a westward speed with a perfect explained fraction", r0)


def test_the_k12_property_compares_complete_results_and_names_its_start_steps():
    M = _load()
    a = {"outcome": "supported", "reason": "", "S": 2.0, "c_x": -2.5, "c_y": 0.0, "g": 0.1, "r2": 0.9, "row": 20, "k": 5}
    assert M._same(a, dict(a))
    assert not M._same(a, {**a, "g": 0.2}), "a field outside the old subset must count"
    assert not M._same(a, {**a, "c_x": -2.5 + 1e-13}), "exact equality, no tolerance"
    assert not M._same(a, {k: v for k, v in a.items() if k != "g"}), "the same keys are required"
    assert M.K12_EQUAL_FROM == {"step": 5, "pair": 4}


def test_overall_verdicts_and_scale_pass_sets_are_derived_from_the_records():
    M = _load()
    def entry(cid, kind, verdicts):
        return {"id": cid, "truth": {"kind": kind}, "candidates": {c: {"summary": {"verdict": v}} for c, v in verdicts.items()}}
    names = list(M.CANDIDATES)
    out = {"constructions": [entry("K1", "positive", {c: "adequate" for c in names}),
                             entry("K2", "negative", {c: ("not assessed" if c == "A1" else "not adequate" if c == "C0" else "adequate") for c in names}),
                             entry("K14", "limit", {c: "reported" for c in names})]}
    v = M.overall_verdicts(out)
    assert v["A0"] == {"constructions": 2, "adequate": 2, "not adequate": 0, "not assessed": 0, "overall": "adequate overall"}
    assert v["A1"]["not assessed"] == 1 and v["A1"]["overall"] == "not adequate overall"
    assert v["C0"]["not adequate"] == 1 and v["C0"]["overall"] == "not adequate overall"
    rows = [{"id": cid, "candidate": cand, "half_degrees": hd, "summary": {"verdict": vd}} for cand, hd, cid, vd in (
        ("A1", [4.0, 3.0], "K3", "adequate"), ("A1", [4.0, 3.0], "K4", "adequate"), ("A1", [8.0, 6.0], "K3", "adequate"), ("A1", [8.0, 6.0], "K4", "not adequate"),
        ("C1", [3.0, 2.0], "K3", "not adequate"), ("C1", [3.0, 2.0], "K4", "adequate"), ("C1", [6.0, 4.0], "K3", "not adequate"), ("C1", [6.0, 4.0], "K4", "adequate"),
        ("C1", [6.0, 4.0], "K13", "adequate"))]
    assert M.scale_pass_sets(rows) == {"A1": [[4.0, 3.0]], "C1": []}

