"""The row-feature graph prototype: features per row (reversals interpolated, wind minima,
vorticity maxima), row-chains within the spatial tolerance with missing and adjacent
invalid rows recorded, links classed unique, ambiguous or unassessable with branchings
and joinings, histories from unique links only, and the declared synthetic cases passing
noise-free under the IMPLEMENTED scorers, which score the wind-minimum kind only (the
pass expression of `run_synthetic`). As the correction of 2026-10-16 records, that pass
does not establish the declared requirements for the reversal kind, and the grouping and
unassessable-status guarantees of the declaration are not satisfied as written."""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("row_feature_graph_under_test", os.path.join(ROOT, "scripts", "pilot_row_feature_graph.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_row_features_interpolate_the_reversal_and_keep_the_kinds_apart():
    G = _load()
    lons = np.arange(40.0, 50.0)
    v = np.array([-1, -4, -8, -4, -1, 1, 3, 2, 1, 0.5])        # trough at 42 E, crossing between 44 and 45 E at -1 -> +1
    zeta = np.array([0, 0, 0, 1, 3, 2, 0, 0, 0, 0]) * 1e-5
    valid = np.ones(10, bool)
    f = G.row_features(lons, v, valid, zeta, valid)
    assert [x["lon"] for x in f["wind_min"]] == [42.0]
    assert len(f["reversal"]) == 1 and abs(f["reversal"][0]["lon"] - 44.5) < 1e-9
    assert [x["lon"] for x in f["vort_max"]] == [44.0]
    valid2 = valid.copy(); valid2[4] = False                    # an invalid cell at 44 E removes the crossing and the vorticity maximum
    f2 = G.row_features(lons, v, valid2, zeta, valid2)
    assert f2["reversal"] == [] and f2["vort_max"] == []


def test_chains_join_adjacent_rows_within_the_tolerance_and_record_missing_rows():
    G = _load()
    feats = {r: {"reversal": [], "wind_min": [{"lon": 50.0 - 0.5 * (r - 10), "value": -9.0}], "vort_max": []} for r in range(10, 20)}
    feats[15]["wind_min"] = []                                   # a missing row inside the span, with no invalid cell near
    feats[18]["wind_min"].append({"lon": 60.0, "value": -5.0})   # a second feature far away on row 18
    inv = {r: [] for r in range(10, 20)}
    out = G.chains_at_step(feats, range(10, 20), inv, 2.5, 2)["wind_min"]
    main = [c for c in out if c["n_rows"] >= 4]
    assert len(main) == 2 and {tuple(c["rows"]) for c in main} == {(10, 11, 12, 13, 14), (16, 17, 18, 19)}, "the missing row splits the chain under strict adjacency"
    assert all(c["missing_rows"] == [] for c in main)
    far = [c for c in out if any(60.0 in c["lon_by_row"][str(r)] for r in c["rows"])]
    assert far == [], "a lone far feature on one row is below the minimum chain length and forms no chain"
    inv[15] = [47.0, 48.0]                                       # invalid cells near the neighbors' longitudes on the missing row
    out2 = G.chains_at_step(feats, range(10, 20), inv, 2.5, 2)["wind_min"]
    adj = [c for c in out2 if 15 in c["adjacent_invalid_rows"]]
    assert len(adj) == 2, "both chains record the invalid row beyond their span"


def test_links_are_unique_ambiguous_or_unassessable_and_histories_use_unique_links_only():
    G = _load()
    def chain(cid, rows, lon):
        return {"id": cid, "kind": "wind_min", "rows": list(rows), "lon_by_row": {str(r): [lon] for r in rows}, "missing_rows": [], "adjacent_invalid_rows": [], "rows_with_multiple": []}
    t = {"wind_min": [chain("wind_min:0", range(12, 20), 50.0)], "reversal": [], "vort_max": []}
    t1 = {"wind_min": [chain("wind_min:0", range(12, 20), 49.0), chain("wind_min:1", range(12, 20), 52.0)], "reversal": [], "vort_max": []}
    L = G.links_between(t, t1, {}, {}, G.PARAMS)["wind_min"]
    assert len(L["links"]) == 2 and all(l["class"] == "ambiguous" for l in L["links"]) and len(L["branchings"]) == 1 and not L["branchings"][0]["unassessable"]
    t1b = {"wind_min": [chain("wind_min:0", range(12, 15), 49.0), chain("wind_min:1", range(18, 22), 49.0)], "reversal": [], "vort_max": []}
    inv = {15: [48.0, 49.0, 50.0], 16: [49.0], 17: [48.0, 50.0]}
    Lb = G.links_between(t, t1b, {}, inv, G.PARAMS)["wind_min"]
    assert Lb["branchings"][0]["unassessable"] is True and all(l["class"] == "unassessable" for l in Lb["links"])
    steps = [{"step": 0, "chains": t}, {"step": 1, "chains": t1}]
    H = G.histories_and_gaps(steps, [G.links_between(t, t1, {}, {}, G.PARAMS)], G.PARAMS)["wind_min"]["histories"]
    assert all(h["n_steps"] == 1 for h in H) and any(h["end_reason"] == "forward link not unique" for h in H), "no history passes through the branching"


def test_the_declared_synthetic_cases_pass_noise_free(tmp_path):
    G = _load()
    assert G.run_synthetic(str(tmp_path / "syn.json"), noisy=False) == 0
