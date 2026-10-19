"""The minima-histories effort must link, classify and score as section 2 of
`EASTERN_MINIMA_HISTORIES_2026-10-19.md` declares, on hand-built minima (section 3), before
any construction is scored."""
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("minima_histories_under_test", os.path.join(ROOT, "scripts", "eastern_minima_histories.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _m(*lons):
    return [{"lon": float(x), "depth": -5.0} for x in lons]


def test_the_nearest_linker_links_nearest_pairs_first_one_to_one_within_six_degrees():
    M = _load()
    hs = M.link([_m(10, 20), _m(11, 26.5, 21)], "LN")
    assert [h["lons"] for h in hs] == [[10.0, 11.0], [20.0, 21.0], [26.5]], "26.5 is 6.5 degrees from 20 and starts a new history"
    hs = M.link([_m(10, 12), _m(11)], "LN")
    assert [h["lons"] for h in hs] == [[10.0, 11.0], [12.0]], "one-to-one, a tie goes to the history with the lower last longitude"
    hs = M.link([_m(10, 13), _m(13.5, 16)], "LN")
    assert [h["lons"] for h in hs] == [[10.0, 16.0], [13.0, 13.5]], "the nearest pair (13 to 13.5) first, then 10 takes 16 at exactly 6 degrees"
    hs = M.link([_m(10), _m(16)], "LN")
    assert [h["lons"] for h in hs] == [[10.0, 16.0]], "exactly 6 degrees links"
    hs = M.link([_m(10), None, _m(10)], "LN")
    assert [h["lons"] for h in hs] == [[10.0], [10.0]] and [h["days"] for h in hs] == [[0], [2]], "a day without a background ends every history and no gap is bridged"
    hs = M.link([_m(10), [], _m(10)], "LN")
    assert [h["days"] for h in hs] == [[0], [2]], "a day with no minimum ends every history"


def test_the_predicted_linker_follows_the_last_step_within_three_degrees():
    M = _load()
    days = [_m(10), _m(8), _m(8.5, 6.2)]
    ln = M.link(days, "LN"); lp = M.link(days, "LP")
    assert [h["lons"] for h in ln] == [[10.0, 8.0, 8.5], [6.2]], "nearest to the last position"
    assert [h["lons"] for h in lp] == [[10.0, 8.0, 6.2], [8.5]], "nearest to the prediction, 6 degrees, at 0.2"
    lp2 = M.link([_m(10), _m(8), _m(9.5)], "LP")
    assert [h["lons"] for h in lp2] == [[10.0, 8.0], [9.5]], "3.5 degrees from the prediction is beyond 3"
    lp4 = M.link([_m(10), _m(4), _m(-1.0)], "LP")
    assert [h["lons"] for h in lp4] == [[10.0, 4.0, -1.0]], "1 degree from the prediction and 5 from the last position links"
    lp5 = M.link([_m(10), _m(4.5), _m(-1.5)], "LP")
    assert [h["lons"] for h in lp5] == [[10.0, 4.5, -1.5]], "0.5 from the prediction and exactly 6 from the last position links"
    lp6 = M.link([_m(10), _m(5), _m(-1.5)], "LP")
    assert [h["lons"] for h in lp6] == [[10.0, 5.0], [-1.5]], "1.5 from the prediction but 6.5 from the last position does not link"


def test_the_classes_follow_their_thresholds_at_the_boundaries():
    M = _load()
    def h(*x):
        return {"days": list(range(len(x))), "lons": [float(v) for v in x], "depths": [-5.0] * len(x)}
    assert M.classify(h(60, 57, 55, 52)) == "westward", "4 days, net -8, all steps west"
    assert M.classify(h(60, 57, 55, 52.1)) == "other", "net -7.9"
    assert M.classify(h(60, 57, 55)) == "other", "3 days"
    assert M.classify(h(60, 56, 52, 53, 50)) == "westward", "3 of 4 steps west is 0.75"
    assert M.classify(h(60, 56, 57, 53, 52, 50)) == "westward" and M.classify(h(60, 56, 57, 58, 52, 50)) == "other", "3 of 5 steps west is 0.6"
    assert M.classify(h(50, 52, 56, 51, 50)) == "standing" and M.classify(h(50, 52, 56.5, 51, 50)) == "other", "a range of 6 is standing, 6.5 is not"
    assert M.classify(h(50, 51, 52, 50)) == "other", "4 days is too short to stand"
    assert M.classify(h(60, 58, 56, 54), net=-6.0) == "westward" and M.classify(h(60, 58, 56, 54)) == "other" and M.classify(h(60, 57, 55, 52), net=-10.0) == "other"


def test_the_scores_count_what_they_say():
    M = _load()
    tr = np.full(10, np.nan); tr[2:8] = [70.0, 68.0, 66.0, 64.0, 62.0, 60.0]; tr[8] = 75.0
    mins = [[]] * 10; mins = list(mins); mins[5] = None
    follow = {"days": [2, 3, 4, 6, 7], "lons": [70.5, 68.0, 64.5, 62.0, 60.0], "depths": [-5.0] * 5}     # day 4 is 1.5 degrees off the center, a miss
    other = {"days": [2, 3, 4, 5], "lons": [40.0, 38.0, 36.0, 34.0], "depths": [-5.0] * 4}
    r = M.passage_score([follow, other], ["westward", "westward"], tr, mins)
    assert r["present_days"] == 5 and abs(r["share"] - 4 / 5) < 1e-12 and r["unmatched_westward"] == 1 and r["westward"] == 2, r
    assert M.follows(follow, tr) is True and M.follows({"days": [2, 3], "lons": [70.0, 60.0], "depths": [0, 0]}, tr) is True and M.follows({"days": [2, 3, 4], "lons": [70.0, 60.0, 50.0], "depths": [0, 0, 0]}, tr) is False
    r2 = M.passage_score([follow], ["other"], tr, mins)
    assert r2["share"] == 0.0 and r2["unmatched_westward"] == 0, "only westward histories recover a day"


def test_the_rule_and_the_selection_are_applied_as_declared():
    M = _load()
    keys = [M.combo(r, l) for r, l in M.ORDER]
    def results(good):
        def pas(ok):
            return {"present_days": 20, "share": 0.9 if ok else 0.5, "unmatched_westward": 0, "westward": 1}
        return {"C4 x": {"kind": "passage", "meta": {"speed": 1.0}, **{k: pas(k in good) for k in keys}},
                "C3 x": {"kind": "wobble", "meta": {}, **{k: {"westward": 0} for k in keys}},
                "C0 noise alone": {"kind": "noise", "meta": {}, **{k: {"westward": 9} for k in keys}},
                "C4 noisy": {"kind": "passage", "meta": {"variant": True}, **{k: pas(False) for k in keys}}}
    v, sel = M.apply_rule(results(set(keys)))
    assert sel == "RB1|LN", "the first in the declared order, the noise case and the variant do not enter"
    v, sel = M.apply_rule(results({"RB9|LP", "RRAW|LN"}))
    assert sel == "RB9|LP"
    r = results(set(keys)); r["C3 x"]["RB1|LN"]["westward"] = 1; r["C4 x"]["RB1|LP"]["unmatched_westward"] = 1
    v, sel = M.apply_rule(r)
    assert sel == "RB9|LN" and v["RB1|LN"]["failing_cases"] == ["C3 x"] and v["RB1|LP"]["failing_cases"] == ["C4 x"], "a false westward history and an unmatched one each fail"
    v, sel = M.apply_rule(results(set()))
    assert sel is None
    assert M.ORDER == (("RB1", "LN"), ("RB1", "LP"), ("RB9", "LN"), ("RB9", "LP"), ("RRAW", "LN"), ("RRAW", "LP"))
    assert (M.S_MAX, M.S_PRED, M.W_MIN_LEN, M.W_NET, M.W_FRAC, M.S_MIN_LEN, M.S_RANGE, M.TOL_DEG, M.PASS_RECOVERY) == (6.0, 3.0, 4, -8.0, 0.75, 5, 6.0, 1.0, 0.80)
    assert M.REP_SPEC == {"RB1": ("B1", 0.0), "RB9": ("B2_9", 0.0), "RRAW": ("zero", -4.0)}


def test_the_case_matrix_adds_c6_to_the_background_cases():
    M = _load()
    cases = M.construction_cases()
    names = [c[0] for c in cases]
    base = [c[0] for c in M.A.construction_cases()]
    assert names[:len(base)] == base and len(names) == len(base) + 6
    for depth in (5.0, 10.0):
        for c in (1.0, 2.0, 3.0):
            n = f"C6 wobble 3 deg 14 days, transient depth {depth:g}, {c:g} deg/day"
            _, kind, kw, meta = cases[names.index(n)]
            assert kind == "passage" and kw["transient"] == (35.0, 72.0, c, depth) and meta == {"over_standing": True, "depth": depth, "speed": c, "wobble": True}
            assert abs(kw["standing"](3.5) - 53.0) < 1e-12 and abs(kw["standing"](10.5) - 47.0) < 1e-12


def test_the_archive_summaries_and_the_ledger_match():
    M = _load()
    dates = [f"1990-06-{d:02d}" for d in range(1, 11)]
    w1 = {"days": [0, 1, 2, 3, 4], "lons": [60.0, 57.0, 54.0, 51.0, 48.0], "depths": [-5.0] * 5}
    w2 = {"days": [5, 6, 7, 8], "lons": [52.0, 48.0, 44.0, 40.0], "depths": [-5.0] * 4}
    w3 = {"days": [0, 1, 2, 3], "lons": [72.5, 68.5, 64.5, 60.5], "depths": [-5.0] * 4}     # within 3 degrees of the eastern edge, not within 1
    w4 = {"days": [6, 7, 8, 9], "lons": [56.0, 52.0, 48.0, 43.5], "depths": [-5.0] * 4}     # from east of 50 E to 43.5 E, short of 42 E
    st = {"days": [0, 1, 2, 3, 4], "lons": [45.0, 46.0, 45.0, 44.0, 45.0], "depths": [-5.0] * 5}
    per = {M.combo(r, l): [(w1, "westward"), (w2, "westward"), (w3, "westward"), (w4, "westward"), (st, "standing")] for r, l in M.ORDER}
    hist = {(1990, "b"): (dates, per)}
    tabs = M.archive_tables(hist)
    d = tabs["RB1|LN"]["bands"]["b"]
    assert d["westward"] == 4 and d["arc_to_channel"] == 1 and d["ends_at_or_west_of_40"] == 1 and d["cut_east"] == 1 and d["cut_west"] == 0, "43.5 E is not at or west of 42 E, and 72.5 E is within 3 degrees of the eastern edge"
    assert d["by_month_of_first_day"] == {"6": 4, "7": 0, "8": 0, "9": 0} and abs(d["standing_per_season"] - 1 / 24) < 1e-12
    assert d["speed_q"][1] == -4.0 and d["days_q"][1] == 4.0 and d["westward_at_net"] == {"-6": 4, "-10": 4}
    w9 = {"days": [0, 1, 2, 3], "lons": [60.0, 57.0, 54.0, 51.0], "depths": [-5.0] * 4}       # net -9, westward at -6 and -8 but not -10
    only9 = M.archive_tables({(1990, "c"): (dates, {M.combo(r, l): [(w9, "westward")] for r, l in M.ORDER})})
    assert only9["RB1|LN"]["bands"]["c"]["westward_at_net"] == {"-6": 1, "-10": 0}, "a band with no history at -10 still reports a zero there"
    s1 = M.summarize_westward(w1, dates, 1990, "b")
    assert s1["net_deg"] == -12.0 and s1["speed_deg_per_day"] == -3.0 and s1["first_date"] == "1990-06-01" and not s1["arc_to_channel"]
    entries = [{"band": "b", "speeds": {"direction": "westward"}, "days": [{"date": "1990-06-01", "anchored": True, "anchor_lon": 60.5}, {"date": "1990-06-02", "anchored": True, "anchor_lon": 50.0},
                                                                          {"date": "1990-06-03", "anchored": False, "anchor_lon": None}]},
               {"band": "b", "speeds": {"direction": "standing"}, "days": [{"date": "1990-06-01", "anchored": True, "anchor_lon": 45.0}, {"date": "1990-06-02", "anchored": True, "anchor_lon": 30.0}, {"date": "1990-06-03", "anchored": True, "anchor_lon": 54.5}]}]
    led = M.ledger_match(entries, hist)
    assert led["RB1|LN"]["westward"] == {"entries": 1, "matched": 1}, "one of two anchored days within 1 degree is half"
    assert led["RB1|LN"]["standing"] == {"entries": 1, "matched": 0}, "a standing history does not match, and one of three anchored days is less than half"


def test_main_refuses_to_overwrite(tmp_path):
    M = _load()
    (tmp_path / "eval60_hist_constructions.md").write_text("x")
    try:
        M.main(["constructions", "--out-dir", str(tmp_path)]); assert False
    except SystemExit as ex:
        assert "never overwritten" in str(ex)
    (tmp_path / "c.json").write_text(json.dumps({"selected": None}))
    (tmp_path / "eval60_hist_tables.md").write_text("x")
    try:
        M.main(["archive", "--constructions", str(tmp_path / "c.json"), "--records", str(tmp_path), "--ledger", "x", "--out-dir", str(tmp_path)]); assert False
    except SystemExit as ex:
        assert "never overwritten" in str(ex), "the archive runs without a selection but still refuses to overwrite"


def test_the_noise_ensemble_counts_westward_histories_per_noise_season():
    """Added after the declaration: the ensemble runs the declared recipe at each level over
    the seeded seasons, and the matched level scales 2 m/s by the ratio of spreads."""
    M = _load()
    assert M.NOISE_SEEDS == tuple(range(20261019, 20261019 + 24))
    ens = M.noise_ensemble({"a": 2.0}, seeds=(20261019, 20261020))
    per = ens["a"]["per_combination"]
    assert ens["a"]["seasons"] == 2 and set(per) == {M.combo(r, l) for r, l in M.ORDER}
    for k, v in per.items():
        assert 0 <= v["seasons_with_any"] <= 2 and v["max"] >= v["mean"] >= 0
        assert len(v["per_season"]) == 2 and abs(np.mean(v["per_season"]) - v["mean"]) < 1e-12 and v["seasons_with_any"] == sum(1 for x in v["per_season"] if x > 0)
    month_of = M.A.month_index()
    D = M.A.noise_field(2.0).reshape(122, 4, -1).mean(axis=1)
    assert abs(M.b1_anomaly_std(D, month_of) - float(np.nanstd(D - M.A.background("B1", D, month_of)))) < 1e-12
    one = M.noise_ensemble({"x": 2.0}, seeds=(20261019,))["x"]["per_combination"]
    con = M.run_constructions()["C0 noise alone"]
    assert all(one[k]["mean"] == con[k]["westward"] for k in one), "the first seed is the constructions' noise season"
    assert one["RB9|LN"]["mean"] == 1 and M.noise_ensemble({"z": 0.0}, seeds=(20261019,))["z"]["per_combination"]["RB9|LN"]["mean"] == 0, "the level is applied: no noise, no westward history"


def test_the_review_boundaries_are_inclusive():
    """Review of 2026-10-19, F7: exactly 3 degrees from the prediction links, and exactly 0.80
    recovery passes."""
    M = _load()
    assert [h["lons"] for h in M.link([_m(10), _m(8), _m(9.0)], "LP")] == [[10.0, 8.0, 9.0]], "3 degrees from the prediction (6) and 1 from the last position"
    keys = [M.combo(r, l) for r, l in M.ORDER]
    def res(share):
        return {"C4 x": {"kind": "passage", "meta": {"speed": 2.0}, **{k: {"present_days": 10, "share": share, "unmatched_westward": 0, "westward": 1} for k in keys}}}
    assert M.apply_rule(res(0.8))[1] == "RB1|LN" and M.apply_rule(res(0.79))[1] is None


def test_the_ledger_match_needs_one_history_and_a_selection_prints_first():
    """Review of 2026-10-19, F3 and F6."""
    M = _load()
    dates = [f"1990-06-{d:02d}" for d in range(1, 9)]
    h1 = {"days": [0, 1, 2, 3], "lons": [60.0, 57.0, 54.0, 51.0], "depths": [-5.0] * 4}
    h2 = {"days": [4, 5, 6, 7], "lons": [60.0, 57.0, 54.0, 51.0], "depths": [-5.0] * 4}
    hist = {(1990, "b"): (dates, {M.combo(r, l): [(h1, "westward"), (h2, "westward")] for r, l in M.ORDER})}
    anchors = [60.0, 57.0, 40.0, 40.0, 60.0, 57.0, 40.0, 40.0]
    entry = {"band": "b", "speeds": {"direction": "westward"}, "days": [{"date": d, "anchored": True, "anchor_lon": x} for d, x in zip(dates, anchors)]}
    assert M.ledger_match([entry], hist)["RB1|LN"]["westward"] == {"entries": 1, "matched": 0}, "two histories with 2 of 8 days each do not make one match"
    one = dict(entry, days=[{"date": d, "anchored": True, "anchor_lon": x} for d, x in zip(dates, [60.0, 57.0, 54.0, 51.0, 40.0, 40.0, 40.0, 40.0])])
    assert M.ledger_match([one], hist)["RB1|LN"]["westward"] == {"entries": 1, "matched": 1}, "one history with 4 of 8 days matches"
    assert M.ordered_keys("RB9|LP")[0] == "RB9|LP" and M.ordered_keys("RB9|LP")[1:] == [k for k in (M.combo(r, l) for r, l in M.ORDER) if k != "RB9|LP"]
    assert M.ordered_keys(None) == [M.combo(r, l) for r, l in M.ORDER]
    tabs = M.archive_tables(hist); led = M.ledger_match([entry], hist)
    first_row = [l for l in M.archive_text(tabs, led, "RB9|LP").splitlines() if l.startswith("| RB")][0]
    assert first_row.startswith("| RB9|LP |"), first_row


def test_the_matched_level_is_two_times_the_ratio_of_spreads():
    """Review of 2026-10-19, F7: the matched level against values computed outside the
    implementation, through matched_scales on synthetic records."""
    M = _load()
    assert abs(M.matched_level([1.0, 6.0, 2.0], 1.5) - 2.0 * 2.0 / 1.5) < 1e-12, "the median (2), not the mean (3)"
    lons = [20.0, 20.5, 21.0]
    recs = []
    for y in M.YEARS:
        for m in (6, 7, 8, 9):
            nd = {6: 30, 7: 31, 8: 31, 9: 30}[m]
            times, vals = [], []
            for d in range(1, nd + 1):
                for hh in ("00", "06", "12", "18"):
                    tt = (d - 1) + int(hh) / 24.0
                    times.append(f"{y}-{m:02d}-{d:02d}T{hh}"); vals.append([round(np.sin(2 * np.pi * tt / 7.0 + k), 2) for k in range(3)])
            recs.append({"year": y, "month": m, "lons": lons, "times": times, "bands": {"b": [0, 1]}, "band_means_700": {"b": vals}})
    got = M.matched_scales(recs)["b"]
    D = []
    for m in (6, 7, 8, 9):
        r = [x for x in recs if x["year"] == M.YEARS[0] and x["month"] == m][0]
        v = np.array(r["band_means_700"]["b"]); Dm = v.reshape(-1, 4, 3).mean(axis=1); D.append(Dm - Dm.mean(axis=0))
    real = float(np.concatenate(D).std())
    ref = M.A.noise_field(2.0).reshape(122, 4, -1).mean(axis=1); mo = M.A.month_index()
    ref_anom = np.concatenate([ref[mo == m] - ref[mo == m].mean(axis=0) for m in (6, 7, 8, 9)])
    assert abs(got["real_b1_anomaly_std"] - real) < 1e-9 and abs(got["noise_b1_anomaly_std_at_2"] - float(ref_anom.std())) < 1e-9
    assert abs(got["matched_noise_std"] - 2.0 * real / float(ref_anom.std())) < 1e-9


def test_each_representation_keeps_its_own_floor_on_the_exact_path():
    """The raw representation keeps its -4 m/s floor, so a raw minimum of -3 m/s is not a
    minimum, while the anomaly floor of 0 keeps a -3 m/s anomaly."""
    M = _load()
    lons = np.arange(20.0, 75.01, 0.5)
    H = np.zeros((122, len(lons)), dtype=np.int64); H[:, lons == 50.0] = -1200      # a daily mean of -3 m/s at 50 E on every day
    mo = M.A.month_index()
    raw = M.rep_minima("RRAW", H, mo, lons)
    assert all(day == [] for day in raw), "-3 m/s is above the raw floor of -4"
    H2 = H.copy(); H2[40, lons == 60.0] = -1200                                         # one day with a -3 m/s minimum at 60 E against a zero month mean there
    b1 = M.rep_minima("RB1", H2, mo, lons)
    assert [m["lon"] for m in b1[40]] == [60.0], "the standing -3 m/s at 50 E is removed by the month mean, the one-day -3 m/s departure at 60 E is kept"

