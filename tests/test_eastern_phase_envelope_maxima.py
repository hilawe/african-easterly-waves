"""Tests for scripts/eastern_phase_envelope_maxima.py (`EASTERN_PHASE_ENVELOPE_MAXIMA_2026-10-27.md`)."""

import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.path.insert(0, HERE)
import eastern_phase_envelope as PE  # noqa: E402
import eastern_phase_envelope_matched as M  # noqa: E402
import eastern_phase_envelope_maxima as MX  # noqa: E402
import eastern_phase_envelope_probes as PR  # noqa: E402
from test_eastern_phase_envelope_matched import seasons_on  # noqa: E402

LAGD = MX.LAGS * PE.DT_DAYS
ZERO = int(np.flatnonzero(MX.LAGS == 0)[0])


def hand_map(cg, cp=-7.0, sx=10.0, sl=4.0, base=-15.0):
    """R = A(x - cg L) g(L) cos(k (x - cp L)) with Gaussian A and g, wavelength 30 degrees."""
    x, L = M.MLONS[None, :] - base, LAGD[:, None]
    return 2.0 * np.exp(-((x - cg * L) ** 2) / (2 * sx ** 2) - L ** 2 / (2 * sl ** 2)) * np.cos(2 * math.pi / 30.0 * (x - cp * L))


# ------------------------------------------------------------------ the maps

def test_the_block_holds_the_samples_of_the_frozen_analytic_signal():
    S = seasons_on(M.MLONS, -15.0, 0.0, n=1)[0]
    col = int(np.flatnonzero(np.isclose(M.MLONS, -15.0))[0])
    X, r = MX.block(S, S[:, col])
    Z = PE.analytic(S)
    assert X.shape == Z.shape and np.allclose(X, Z.real, atol=1e-10, rtol=0)
    assert np.array_equal(r, X[:, col])


def test_the_map_at_lag_zero_on_the_base_is_the_reference_standard_deviation():
    seasons = seasons_on(M.MLONS, -15.0, 0.0, n=3)
    col = int(np.flatnonzero(np.isclose(M.MLONS, -15.0))[0])
    R = MX.regression(MX.season_stack(seasons, [S[:, col] for S in seasons]))
    pooled = np.concatenate([MX.block(S, S[:, col])[1] for S in seasons])
    assert R.shape == (len(MX.LAGS), M.MLONS.size)
    assert abs(R[ZERO, col] - pooled.std()) < 1e-12


def test_resample_weights_pool_like_repeated_seasons():
    seasons = seasons_on(M.MLONS, -15.0, 3.0, n=3)
    col = int(np.flatnonzero(np.isclose(M.MLONS, -15.0))[0])
    refs = [S[:, col] for S in seasons]
    st = MX.season_stack(seasons, refs)
    rep = MX.season_stack([seasons[0], seasons[0], seasons[2]], [refs[0], refs[0], refs[2]])
    assert np.allclose(MX.regression(st, [2, 0, 1]), MX.regression(rep), atol=1e-12, rtol=0)


def test_a_positive_lag_means_the_field_is_later_than_the_reference():
    rng = np.random.default_rng(3)
    ref = rng.standard_normal(488)
    S = np.column_stack([ref, np.roll(ref, 8)] + [rng.standard_normal(488) for _ in range(3)])
    lons = np.arange(5) * 0.5
    X, r = MX.block(S, ref)
    st = PE.stack([MX.real_sums(X, r)])
    w = np.ones(1)
    n = w @ st["n"]
    C = st["sab"][0] / n[:, None] - (st["sb"][0] / n[:, None]) * (st["sa"][0] / n)[:, None]
    assert lons.size == X.shape[1] and int(MX.LAGS[np.argmax(C[:, 1])]) == 8


# ------------------------------------------------------------------ the rules

def test_row_extrema_are_interior_signed_and_thresholded():
    row = np.array([0.9, 0.2, 0.5, 0.1, -0.4, 0.0, 0.29, 0.1, -0.8])
    assert list(MX.row_extrema(row, 1)) == [2]                # 0.9 is an end column, 0.29 is below the threshold
    assert list(MX.row_extrema(row, -1)) == [4]               # -0.8 is an end column


def test_chains_link_nearest_pairs_one_to_one_within_the_distance():
    lons = np.arange(0.0, 40.01, 0.5)
    R = np.zeros((4, lons.size))
    for li, cols in enumerate(([10, 30], [8, 27], [9, 41], [6, 60])):
        for j in cols:
            R[li, j] = 1.0
    got = sorted(MX.chains(R, lons, 1))
    assert got == sorted([[(0, 10), (1, 8), (2, 9), (3, 6)], [(0, 30), (1, 27)], [(2, 41)], [(3, 60)]])
    R2 = np.zeros((2, lons.size)); R2[0, 10] = R2[0, 14] = R2[1, 12] = 1.0          # two chains reach for one extremum
    assert sorted(MX.chains(R2, lons, 1)) == [[(0, 10), (1, 12)], [(0, 14)]]


def test_chain_maxima_count_inside_peaks_and_each_version_treats_ties_as_recorded():
    lons = np.arange(0.0, 10.01, 0.5)
    lags = np.arange(-2, 3)
    R = np.zeros((5, lons.size)) + 0.01
    for li, v in enumerate((0.5, 0.9, 0.7, 0.65, 0.4)):
        R[li, 10] = v                                           # a chain peaking inside at lag -1
    R[0, 2] = 2.0                                               # the largest value at lag -2, scaling divides by it
    R[1:, 18] = [0.4, 0.5, 0.6, 0.9]                           # peaks at its last point scaled or not, not counted
    # scaled, chain 10 holds its row's largest value at lags -1, 0 and +1, a tied run of three
    got = {v: MX.chain_maxima(R, lons, lags, version=v) for v in range(4)}
    assert got[0] == got[1] == [(-0.25, 5.0, 0.9)]             # the first of the run
    assert got[2] == got[3] == [(0.0, 5.0, 0.7)]               # its middle
    R[0, 10] = 0.95                                             # the unscaled peak moves to the first point
    assert MX.chain_maxima(R, lons, lags, version=0) == []
    assert MX.chain_maxima(R, lons, lags, version=1) == [(-0.25, 5.0, 0.9)]   # 0.95 / 2.0 is less than 0.9 / 0.9
    R[4, 10] = 0.95                                             # the run now reaches the chain's last point
    assert MX.chain_maxima(R, lons, lags, version=2) == [(0.0, 5.0, 0.7)]     # lower middle of four tied points
    assert MX.chain_maxima(R, lons, lags, version=3) == []


def test_the_declared_rule_carries_the_closed_form_pull_and_the_changes_remove_it():
    for cg in (3.0, 0.0):
        pull = cg + 10.0 ** 2 / (4.0 ** 2 * (cg + 7.0))         # cg + sx^2 / (sl^2 (cg - cp))
        R = hand_map(cg)
        for d in MX.RANGES:
            assert abs(MX.read_map(R, M.MLONS, "maxima as declared", d)["drift"] - pull) < 0.1
            assert abs(MX.read_map(R, M.MLONS, "maxima after change 1", d)["drift"] - cg) < 0.25
            assert abs(MX.read_map(R, M.MLONS, "maxima", d)["drift"] - cg) < 0.5
    # change 2 alone takes the middle of tied runs cut by the end of the range read
    assert MX.read_map(hand_map(0.0), M.MLONS, "maxima after change 2", 4)["drift"] > 0.5


def test_the_strongest_rule_skips_end_columns_and_weak_lags():
    lons = np.arange(0.0, 5.01, 0.5)
    lags = np.arange(3)
    R = np.zeros((3, lons.size))
    R[0, 0] = 1.0; R[0, 3] = 0.5            # the largest is an end column, the lag is skipped
    R[1, 4] = 0.2                           # below the threshold
    R[2, 6] = -0.6
    assert MX.strongest_per_lag(R, lons, lags) == [(0.5, 3.0, -0.6)]


def test_drift_needs_three_points_at_two_lags_and_reports_the_largest_lag():
    assert MX.drift([(0.0, 1.0, 1), (0.0, 2.0, 1), (0.0, 3.0, 1)]) is None
    assert MX.drift([(0.0, 1.0, 1), (1.0, 2.0, 1)]) is None
    assert MX.drift([(0.0, 1.0, 1), (1.0, 2.0, 1)], min_count=2) == pytest.approx(1.0)
    assert MX.drift([(0.0, 1.0, 1), (0.0, 2.0, 1)], min_count=2) is None
    assert abs(MX.drift([(0.0, 1.0, 1), (1.0, 3.0, 1), (2.0, 5.0, 1)]) - 2.0) < 1e-12
    r = MX.read_map(hand_map(3.0), M.MLONS, "maxima", 4)
    assert r["pos_lag_days"] == max(p[0] for p in r["points"]) and all(abs(p[0]) <= 4 for p in r["points"])


# ------------------------------------------------------------------ constructions

def test_constructions_are_centered_on_the_base():
    amp = {c: 1.0 for c in PE.CASES}
    for base in (-15.0, -30.0):
        V = MX.construction_signal("U1", 0, base, amp)
        assert M.MLONS[int(np.argmax(V.var(axis=0)))] == base
        V = MX.construction_signal("D07", 0, base, amp)
        assert abs(M.MLONS[int(np.argmax(V.var(axis=0)))] - base) <= 6.0
    assert not MX.construction_signal("N", 0, -15.0, amp).any()


def test_the_noise_is_the_matched_noise_on_this_grid():
    n = MX.noise(0)
    assert n.shape == (488, M.MLONS.size) and abs(n.std() - 3.03) < 1e-9
    assert np.array_equal(n, MX.noise(0)) and not np.array_equal(n, MX.noise(1))
    s = MX.construction_season("N", 2, -15.0, {})
    assert np.array_equal(s, np.round(MX.noise(2), 2))


def fake_results(classes, rules=("maxima",)):
    """Constructions results whose readers `rules` have the class `classes[(case, base, days)]`,
    the constructed class elsewhere."""
    out = {}
    for c in MX.CASES:
        out[c] = {}
        for _, b in M.PRIMARY:
            out[c][f"{b:g}"] = {"readers": {rule: {f"{d}": {"class": classes.get((c, f"{b:g}", d), MX.CONSTRUCTED[c])} for d in MX.RANGES} for rule in rules}}
    return {"cases": out}


def test_adequacy_follows_section_3():
    assert MX.adequacy(fake_results({}))["adequate"]
    assert MX.adequacy(fake_results({("ND7", "-30", 4): "unresolved", ("D07", "-15", 7): "unresolved"}))["adequate"]
    for bad, why in ((("DE7", "-30", 7), "unresolved"), (("ND7", "-15", 4), "unresolved"), (("D07", "-15", 4), "eastward"),
                     (("N", "-30", 7), "westward"), (("U2", "-15", 4), "standing"), (("U1", "-30", 4), "westward")):
        v = MX.adequacy(fake_results({bad: why}))
        assert not v["adequate"] and len(v["failures"]) >= 1
    both = fake_results({("D07", "-15", 4): "eastward"}, rules=("maxima", MX.READER_B))
    both["cases"]["D07"]["-15"]["readers"][MX.READER_B]["4"]["class"] = "standing"
    assert not MX.adequacy(both)["adequate"] and MX.adequacy(both, MX.READER_B)["adequate"]


def constructions_record(tmp_path, b_bad=False, a_shift=False, a_signal_shift=False, digest=None):
    rules = ("maxima", MX.READER_A, MX.READER_B)
    res = fake_results({}, rules)
    old = fake_results({}, rules)
    for r in (res, old):
        for c in MX.CASES[:-1]:
            for b in ("-15", "-30"):
                r["cases"][c][b]["signal_alone"] = {"readers": {MX.READER_A: {"4": 1.0, "7": 1.0}}, "centroid": 0.0}
        r["cases"]["DE7"]["-15"]["signal_alone"] = {"readers": {MX.READER_A: {"4": 2.67, "7": 2.67}}, "centroid": 2.2}
    if a_signal_shift:
        res["cases"]["DE7"]["-15"]["signal_alone"]["readers"][MX.READER_A]["7"] = 2.68
    if b_bad:
        res["cases"]["DE7"]["-30"]["readers"][MX.READER_B]["7"]["class"] = "unresolved"
    if a_shift:
        res["cases"]["U1"]["-15"]["readers"][MX.READER_A]["4"]["class"] = "unresolved"
    path, earlier = tmp_path / "c.json", tmp_path / "e.json"
    path.write_text(json.dumps({"generated_by": {"sha256": digest or MX.SCRIPT_SHA256_AT_LOAD}, "results": res,
                                "verdict": MX.adequacy(res), "amendment_verdict": MX.adequacy(res, MX.READER_B)}))
    earlier.write_text(json.dumps({"results": old}))
    return str(path), str(earlier)


def test_check_constructions_follows_gate_3_of_section_13(tmp_path):
    assert MX.check_constructions(*constructions_record(tmp_path))["amendment_verdict"]["adequate"]
    with pytest.raises(SystemExit, match="different version"):
        MX.check_constructions(*constructions_record(tmp_path, digest="x"))
    with pytest.raises(SystemExit, match="reader B is not adequate"):
        MX.check_constructions(*constructions_record(tmp_path, b_bad=True))
    with pytest.raises(SystemExit, match="reader A's readings of U1 at -15 E differ"):
        MX.check_constructions(*constructions_record(tmp_path, a_shift=True))
    with pytest.raises(SystemExit, match="reader A's readings of DE7 at -15 E differ"):
        MX.check_constructions(*constructions_record(tmp_path, a_signal_shift=True))
    path, earlier = constructions_record(tmp_path)
    rec = json.loads(open(path).read())
    rec["amendment_verdict"] = {"adequate": True, "failures": ["x"]}
    open(path, "w").write(json.dumps(rec))
    with pytest.raises(SystemExit, match="stored verdict disagrees"):
        MX.check_constructions(path, earlier)


@pytest.mark.parametrize("which,cut,why", (
    ("record", lambda r: r.pop("D07"), "holds the cases"),
    ("record", lambda r: r["ND7"].pop("-30"), "holds ND7 at the bases"),
    ("record", lambda r: r["DE7"]["-15"].pop("signal_alone"), "wrong signal-alone readings for DE7 at -15 E"),
    ("record", lambda r: r["N"]["-30"].update(signal_alone={"readers": {}}), "wrong signal-alone readings for N at -30 E"),
    ("record", lambda r: r["U1"]["-15"]["signal_alone"]["readers"][MX.READER_A].pop("7"), "lacks reader A's signal-alone readings for U1"),
    ("record", lambda r: r["U2"]["-30"]["readers"][MX.READER_B].pop("7"), "lacks maxima after change 1, two peaks at both lag ranges for U2"),
    ("record", lambda r: r["U2"]["-30"]["readers"].pop("maxima"), "lacks maxima at both lag ranges for U2"),
    ("earlier", lambda r: r.pop("U1"), "the earlier record holds the cases"),
    ("earlier", lambda r: r["D07"]["-15"].pop("signal_alone"), "the earlier record has the wrong signal-alone readings for D07"),
    ("earlier", lambda r: r["D07"]["-30"]["readers"][MX.READER_A].pop("4"), "the earlier record lacks maxima as declared"),
))
def test_gate_3_refuses_a_record_missing_declared_evidence(tmp_path, which, cut, why):
    path, earlier = constructions_record(tmp_path)
    target = path if which == "record" else earlier
    rec = json.loads(open(target).read())
    cut(rec["results"]["cases"])
    open(target, "w").write(json.dumps(rec))
    with pytest.raises(SystemExit, match=why):
        MX.check_constructions(path, earlier)


def test_reader_b_is_change_1_with_a_two_peak_minimum():
    lons = np.arange(0.0, 10.01, 0.5)
    lags = np.arange(-2, 3)
    R = np.zeros((5, lons.size)) + 0.01
    R[:, 2] = 2.0                                               # every lag's largest value, a chain whose tied run starts at its first point
    R[:, 10] = [0.35, 0.9, 0.4, 0.35, 0.35]                     # peaks inside at lag -1
    R[:, 16] = [-0.35, -0.4, -0.5, -0.95, -0.35]                # peaks inside at lag +1
    pts = MX.POINTS[MX.READER_B](R, lons, lags)
    assert pts == MX.chain_maxima(R, lons, lags, version=1) == [(-0.25, 5.0, 0.9), (0.25, 8.0, -0.95)]
    assert MX.drift(pts) is None and MX.drift(pts, MX.MIN_COUNTS[MX.READER_B]) == pytest.approx(6.0)
    assert MX.MIN_COUNTS == {MX.READER_B: 2} and MX.READ_RULES == ("maxima as declared", MX.READER_B)
    assert MX.READER_B in MX.CONSTRUCTION_RULES and MX.READER_A in MX.CONSTRUCTION_RULES
    full = np.zeros((len(MX.LAGS), lons.size)) + 0.01
    full[:, 2] = 2.0
    full[ZERO - 2:ZERO + 3] = R                                 # the same five lags around lag 0 of the full map
    assert MX.read_map(full, lons, MX.READER_B, 4)["drift"] == pytest.approx(6.0)
    assert MX.read_map(full, lons, "maxima after change 1", 4)["drift"] is None


# ------------------------------------------------------------------ the reading

def test_the_centroid_gate_reproduces_and_refuses_a_changed_stored_reading():
    seasons = seasons_on(M.MLONS, -15.0, 0.0, n=3)
    col = int(np.flatnonzero(np.isclose(M.MLONS, -15.0))[0])
    refs = {MX.BAND_REF: [S[:, col] for S in seasons], MX.POINT_REF: [S[:, col + 2] for S in seasons]}
    band = json.loads(json.dumps(M.results(seasons, (-15.0,), M.MLONS, n_boot=5)))
    point = json.loads(json.dumps(PR.reference_results(seasons, refs[MX.POINT_REF], -15.0, n_boot=5)))
    matched = {"results": {"primary and transect": {"850": band}}}
    reference = {"results": {"850 hPa, -15 E": {"point_reference": point}}}
    got = MX.check_centroid(850, -15.0, seasons, refs, matched, reference, n_boot=5)
    assert got[MX.BAND_REF]["envelope_speed"] == band["-15"]["envelope_speed"] and got[MX.POINT_REF]["envelope_speed"] == point["envelope_speed"]
    point["envelope_interval"][0] += 1e-9
    with pytest.raises(SystemExit, match="point, 10 N reference does not reproduce"):
        MX.check_centroid(850, -15.0, seasons, refs, matched, reference, n_boot=5)


def test_outcome_rows_follow_section_13():
    E, W, U = ["eastward"] * 8, ["westward"] * 8, ["unresolved"] * 8
    assert MX.outcome(E, W) == MX.outcome(E, U) == "A eastward in all eight, B eastward in none"
    assert MX.outcome(E, E) == "A eastward in all eight, B eastward in all eight"
    assert MX.outcome(W, E) == MX.outcome(U, W) == "A eastward in none"
    assert MX.outcome(E[:7] + ["standing"], W) == MX.outcome(E, W[:7] + ["eastward"]) == "any other combination"
    with pytest.raises(SystemExit, match="all eight"):
        MX.outcome(E[:7], W)


def reading_result(cls, b_cls=None):
    r = {"class": cls, "drift": 1.5, "interval": [0.5, 2.5], "undefined": 0.0, "points": [[0.0, -15.0, 1.0], [1.0, -13.0, 0.8], [2.0, -12.0, -0.5]], "pos_lag_days": 2.0}
    return {MX.READER_A: {f"{d}": dict(r) for d in MX.RANGES}, MX.READER_B: {f"{d}": dict(r, **({"class": b_cls} if b_cls else {})) for d in MX.RANGES}}


def fake_reading(b_cls="westward"):
    cent = {"envelope_class": "westward", "envelope_speed": -1.0, "envelope_interval": [-2.0, -0.5]}
    return {f"{lv}": {ref: {"readers": reading_result("eastward", b_cls), "centroid": cent} for ref in MX.REFERENCES} for lv, _ in M.PRIMARY}


def test_the_summary_names_rule_dependence_and_the_texts_name_no_mechanism():
    res = fake_reading()
    summ = MX.summary(res)
    assert summ["outcome"] == "A eastward in all eight, B eastward in none"
    assert MX.summary(fake_reading("eastward"))["outcome"] == "A eastward in all eight, B eastward in all eight"
    assert set(summ["classes"][MX.READER_B].values()) == {"westward"} and len(summ["classes"][MX.READER_A]) == 8
    text = MX.reading_text(res, summ)
    assert "| 700 hPa, -30 E | point, 10 N | maxima as declared | 7 | eastward | +1.50 (+0.50 to +2.50) | 0.000 | 3 | +0.00 to +2.00 | -15.0 to -12.0 |" in text
    assert "| 850 hPa, -15 E | band mean, 8 to 15 N | maxima after change 1, two peaks | 4 | westward |" in text
    rec = {"verdict": {"adequate": False, "failures": ["ND7 -15 E lags to 4 days not westward"]}, "amendment_verdict": {"adequate": True, "failures": []},
           "results": {"cases": {"DE7": {"-15": {"readers": {rule: reading_result("eastward")[MX.READER_A] for rule in MX.CONSTRUCTION_RULES},
                                                 "centroid": {"envelope_class": "eastward", "envelope_speed": 1.7, "envelope_interval": [0.8, 2.3]},
                                                 "signal_alone": {"readers": {rule: {f"{d}": 2.0 for d in MX.RANGES} for rule in MX.CONSTRUCTION_RULES}, "centroid": 2.2}}}}}}
    ctext = MX.constructions_text(rec)
    assert "The maxima rule after change 3 adequate: no. ND7 -15 E lags to 4 days not westward" in ctext and "Reader B of section 13 adequate: yes." in ctext
    assert "| DE7 | -15 | eastward | maxima as declared | 7 |" in ctext
    for word in ("group", "dispers", "packet", "propagat", "energy", "mechanism"):
        assert word not in (text + ctext).lower()


@pytest.mark.parametrize("mode", ("constructions", "read"))
def test_records_are_never_overwritten(tmp_path, mode):
    name = "eval60_phase_env_maxima_constructions.json" if mode == "constructions" else "eval60_phase_env_maxima.json"
    (tmp_path / name).write_text("{}")
    extra = [] if mode == "constructions" else ["--constructions", "c", "--earlier", "e", "--matched", "m", "--reference", "r"]
    with pytest.raises(SystemExit, match="never overwritten"):
        MX.main([mode, "--out-dir", str(tmp_path)] + extra)


def test_the_reading_needs_its_inputs_and_an_adequate_reader_b_before_any_field_is_read(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="required"):
        MX.main(["read", "--out-dir", str(tmp_path), "--constructions", "c", "--matched", "m", "--reference", "r"])
    c, e = constructions_record(tmp_path, b_bad=True)

    def opened(*a, **k):
        raise AssertionError("a field was read")
    monkeypatch.setattr(MX, "era5_inputs", opened)
    with pytest.raises(SystemExit, match="reader B is not adequate"):
        MX.main(["read", "--out-dir", str(tmp_path), "--constructions", c, "--earlier", e, "--matched", "m", "--reference", "r"])


def test_the_figure_title_names_the_amendment_and_fits_the_canvas():
    import matplotlib.pyplot as plt
    res = fake_reading()
    for lv, _ in M.PRIMARY:
        for ref in MX.REFERENCES:
            res[f"{lv}"][ref]["centroid"].update(lags=[0, 4], positions=[-15.0, -16.0])
    maps = {(lv, ref): np.zeros((len(MX.LAGS), len(M.MLONS))) for lv, _ in M.PRIMARY for ref in MX.REFERENCES}
    fig = MX.figure(maps, res)
    fig.canvas.draw()
    box = fig._suptitle.get_window_extent(fig.canvas.get_renderer())
    text = fig._suptitle.get_text()
    plt.close(fig)
    assert "amendment of section 13" in text and "17 seasons" in text and "8 to 15 N" in text and "0.3 m/s" in text
    assert box.x0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, "the title is cut by the canvas"


def test_a_tagged_reading_has_its_own_name(tmp_path):
    (tmp_path / "eval60_phase_env_maxima_maps_r1.png").write_text("x")
    with pytest.raises(SystemExit, match="never overwritten"):
        MX.main(["read", "--out-dir", str(tmp_path), "--tag", "_r1", "--constructions", "c", "--earlier", "e", "--matched", "m", "--reference", "r"])


def test_a_tagged_constructions_record_has_its_own_name(tmp_path, monkeypatch):
    def ran(*a, **k):
        raise AssertionError("the constructions ran")
    monkeypatch.setattr(MX, "run_constructions", ran)
    (tmp_path / "eval60_phase_env_maxima_constructions_r1.json").write_text("{}")
    with pytest.raises(SystemExit, match="never overwritten"):
        MX.main(["constructions", "--out-dir", str(tmp_path), "--tag", "_r1"])


def test_the_reading_reads_readers_a_and_b_only_and_records_the_amendment(tmp_path, monkeypatch):
    seasons = seasons_on(M.MLONS, -15.0, 0.0, n=3)
    cent = {"envelope_class": "westward", "envelope_speed": -1.0, "envelope_interval": [-2.0, -0.5], "lags": [0, 1], "positions": [-15.0, -16.0]}

    def inputs(level, base):
        col = int(np.flatnonzero(np.isclose(M.MLONS, base))[0])
        return seasons, {MX.BAND_REF: [S[:, col] for S in seasons], MX.POINT_REF: [S[:, col + 2] for S in seasons]}
    monkeypatch.setattr(MX, "era5_inputs", inputs)
    monkeypatch.setattr(MX, "check_centroid", lambda *a, **k: {ref: cent for ref in MX.REFERENCES})
    monkeypatch.setattr(MX, "N_SEASONS", 3)
    weights = PE.boot_weights
    monkeypatch.setattr(PE, "boot_weights", lambda n, n_boot=5, seed=PE.BOOT_SEED: weights(n, 5, seed))
    monkeypatch.setattr(M, "input_files", lambda years: [])
    c, e = constructions_record(tmp_path)
    for name in ("m.json", "r.json"):
        (tmp_path / name).write_text("{}")
    out = tmp_path / "out"; out.mkdir()
    MX.main(["read", "--out-dir", str(out), "--constructions", c, "--earlier", e, "--matched", str(tmp_path / "m.json"), "--reference", str(tmp_path / "r.json")])
    rec = json.loads((out / "eval60_phase_env_maxima.json").read_text())
    assert all(set(rec["results"][f"{lv}"][ref]["readers"]) == set(MX.READ_RULES) for lv, _ in M.PRIMARY for ref in MX.REFERENCES)
    assert rec["summary"]["outcome"] in ("A eastward in all eight, B eastward in none", "A eastward in all eight, B eastward in all eight",
                                         "A eastward in none", "any other combination")
    assert rec["generated_by"]["amendment"].startswith("section 13") and rec["generated_by"]["earlier"]["path"] == e
    assert (out / "eval60_phase_env_maxima_maps.png").exists() and "read under the amendment of section 13" in (out / "eval60_phase_env_maxima_tables.md").read_text()


def test_the_constructions_record_stores_both_verdicts(tmp_path, monkeypatch):
    res = fake_results({("DE7", "-30", 7): "unresolved"}, rules=("maxima", MX.READER_A, MX.READER_B))
    res["cases"]["DE7"]["-30"]["readers"][MX.READER_B]["7"]["class"] = "eastward"
    for by in res["cases"].values():
        for v in by.values():
            for rule in MX.CONSTRUCTION_RULES:
                v["readers"].setdefault(rule, {f"{d}": {} for d in MX.RANGES})
                for d in MX.RANGES:
                    v["readers"][rule][f"{d}"].setdefault("class", "unresolved")
                    v["readers"][rule][f"{d}"].update({"drift": 0.0, "interval": [0.0, 0.0], "undefined": 0.0, "points": []})
            v["centroid"] = {"envelope_class": "standing", "envelope_speed": 0.0, "envelope_interval": [0.0, 0.0]}
    monkeypatch.setattr(MX, "run_constructions", lambda: res)
    MX.main(["constructions", "--out-dir", str(tmp_path), "--tag", "_t"])
    rec = json.loads((tmp_path / "eval60_phase_env_maxima_constructions_t.json").read_text())
    assert not rec["verdict"]["adequate"] and rec["amendment_verdict"] == {"adequate": True, "failures": []}
