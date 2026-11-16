"""The input rules of the cloud record as the pipeline applies them, on made-up data."""
import gzip
import json
import os
import sys
import types

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))    # so that this file also runs alone

import build_observed_slots as B  # noqa: E402
import cloud_inputs as C  # noqa: E402
from aew.data import ct as CT  # noqa: E402


# ------------------------------------------------------------------ the recorded inputs
def test_a_recorded_input_with_other_bytes_stops_the_run_and_an_unrecorded_one_is_named(tmp_path, monkeypatch, capsys):
    good, other = tmp_path / "catalog.nc", tmp_path / "other.nc"
    good.write_bytes(b"the recorded bytes")
    other.write_bytes(b"something else")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"class_catalog": {"path": str(good), "sha256": C.sha256(str(good))}}))
    monkeypatch.setattr(C, "MANIFEST", str(manifest))
    assert C.checked(str(good)) == str(good) and "hash as recorded" in capsys.readouterr().out
    assert C.checked(str(other)) == str(other) and "NOT among the recorded ones" in capsys.readouterr().out
    link = tmp_path / "link.nc"                                             # the recorded file reached by another name is still checked
    os.symlink(good, link)
    assert C.checked(str(link)) == str(link) and "hash as recorded" in capsys.readouterr().out
    good.write_bytes(b"the recorded bytes, changed")
    with pytest.raises(SystemExit, match="another SHA-256"):
        C.checked(str(good))
    with pytest.raises(SystemExit, match="does not exist"):
        C.checked(str(tmp_path / "absent.nc"))


def test_the_manifest_names_the_four_inputs_and_the_tracked_calendar_has_its_recorded_hash():
    m = json.load(open(C.MANIFEST))
    assert set(m) == {"class_catalog", "pooled_catalog", "observed_slots", "tracking_file"}
    assert C.sha256(os.path.join(ROOT, m["observed_slots"]["path"])) == m["observed_slots"]["sha256"]
    assert C.CLASS_CATALOG != C.POOLED_CATALOG and all(len(v["sha256"]) == 64 for v in m.values())


# ------------------------------------------------------------------ the calendar builder
def counts_file(path, lines):
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")


def test_the_calendar_builder_reads_one_position_and_refuses_a_broken_pull(tmp_path):
    p = str(tmp_path / "c.txt.gz")
    counts_file(p, ["S MET 1983 7 1 0 40 12 3", "S MET 1983 7 1 3 41 12 3", "S INS 1983 7 1 0 9 2 0", "Y 1983 100"])
    t, n, years = B.read_counts(p, "MET")
    assert list(t) == [pd.Timestamp("1983-07-01 00:00"), pd.Timestamp("1983-07-01 03:00")] and list(n) == [40, 41] and years == [1983]
    with pytest.raises(SystemExit, match="no slot of position"):
        B.read_counts(p, "GMS")
    counts_file(p, ["S MET 1983 7 1 0 40 12", "Y 1983 100"])
    with pytest.raises(SystemExit, match="nine fields"):
        B.read_counts(p, "MET")
    counts_file(p, ["E missing ALLCS1990.out03_18_14", "S MET 1983 7 1 0 40 12 3"])
    with pytest.raises(SystemExit, match="missing year file"):
        B.read_counts(p, "MET")
    slots = B.season_slots([1983, 1984])
    assert len(slots) == 2 * 92 * 8 and slots[0] == pd.Timestamp("1983-07-01") and slots[-1] == pd.Timestamp("1984-09-30 21:00")
    gone = [pd.Timestamp("2000-07-01 00:00"), pd.Timestamp("2000-07-01 03:00"), pd.Timestamp("2000-07-02 12:00")]
    assert B.runs_of(gone) == [("2000-07-01 00:00:00", "2000-07-01 03:00:00", 2), ("2000-07-02 12:00:00", "2000-07-02 12:00:00", 1)] and B.runs_of([]) == []


# ------------------------------------------------------------------ repeated tracking families
def genesis_slice(path, rows, with_family=True):
    ds = nc.Dataset(path, "w")
    ds.createDimension("fam", len(rows))
    ds.createDimension("one", 1)
    cols = ["fam", "ctime", "cslat", "cslon", "csize", "cslife", "tmincl", "cspos"]
    for j, k in enumerate(cols):
        if k == "fam" and not with_family:
            continue
        v = ds.createVariable(k, "f8", ("fam", "one"))
        if k == "ctime":
            v.units = "hours since 2000-07-01 00:00:00"
        v[:, 0] = [r[j] for r in rows]
    ds.close()


def test_a_family_listed_twice_is_counted_once_and_families_that_only_share_a_place_are_kept(tmp_path):
    rows = [(5, 0, 10.0, 3.0, 120, 4, 205, 1),
            (9, 0, 10.0, 3.0, 120, 4, 205, 1),                              # another family at the same time and place
            (5, 0, 10.0, 3.0, 120, 4, 205, 1),                              # the first family again, in every field
            (5, 3, 10.0, 3.0, 120, 4, 205, 1),                              # the same number at another time
            (7, 6, 12.0, 5.0, 150, 9, 190, 1),
            (7, 6, 12.0, 5.0, 150, 9, 190, 1),
            (7, 6, 12.0, 5.0, 150, 9, 190, 1),                              # listed three times
            (7, 6, 12.0, 5.0, 150, 8, 190, 1)]                              # and once with another lifetime
    p = str(tmp_path / "g.nc")
    genesis_slice(p, rows)
    assert list(CT.repeated_rows(p)) == [False, False, True, False, False, True, True, False]
    assert len(CT.from_ct_genesis(p)) == 5 and len(CT.from_ct_genesis(p, drop_repeats=False)) == 8
    assert len(CT.from_ct_genesis(p, deep_core_K=200.0)) == 2               # the deep family, once with each lifetime
    q = str(tmp_path / "no_family.nc")
    genesis_slice(q, rows, with_family=False)
    with pytest.raises(ValueError, match="no family number"):
        CT.repeated_rows(q)
    assert len(CT.from_ct_genesis(q, drop_repeats=False)) == 8


# ------------------------------------------------------------------ the lead-lag series
def test_a_season_with_a_long_unobserved_run_is_left_out_and_a_short_run_is_filled():
    import fig_leadlag as L
    assert L.MAX_GAP_STEPS == 4 and L.FULL_EXPOSURE == 2.0
    times = pd.date_range("2000-01-01", "2002-12-31 18:00", freq="6h")
    rng = np.random.default_rng(1)
    n = times.size
    k = np.arange(n)
    w = np.sin(2 * np.pi * k / 16.0) + 0.3 * rng.standard_normal(n)        # a four-day wave
    c = 50.0 + 20.0 * np.sin(2 * np.pi * (k + 2) / 16.0) + rng.standard_normal(n)
    full = np.full(n, 2.0)
    lags, R0, _, _ = L.seasonal_lag_correlation(w, c, times, L.MAX_LAG, exposure=full, left_out=(out := []))
    assert R0.shape[0] == 3 and out == []
    # no exposure given is the older path, where every grid time counts as observed
    _, Rold, _, _ = L.seasonal_lag_correlation(w, c, times, L.MAX_LAG)
    assert np.allclose(R0, Rold, equal_nan=True)
    e, cc = full.copy(), c.copy()
    short = np.where(times == pd.Timestamp("2000-08-10 00:00"))[0][0]
    e[short:short + 4] = 0.0                                                # four grid times, the longest run that is filled
    cc[short:short + 4] = 0.0
    half = np.where(times == pd.Timestamp("2000-08-20 00:00"))[0][0]
    e[half] = 1.0                                                           # half exposure and half the count
    cc[half] = c[half] / 2.0
    long_ = np.where(times == pd.Timestamp("2001-08-10 00:00"))[0][0]
    e[long_:long_ + 5] = 0.0                                                # five grid times, one too many
    cc[long_:long_ + 5] = 0.0
    _, R1, _, _ = L.seasonal_lag_correlation(w, cc, times, L.MAX_LAG, exposure=e, left_out=(out := []))
    assert out == [2001] and R1.shape[0] == 2
    assert np.nanmax(np.abs(R1[0] - R0[0])) < 0.05 and np.allclose(R1[1], R0[2], equal_nan=True)
    # read as zeros, the same gap moves the correlation of its season more than the filled one does
    _, Rzero, _, _ = L.seasonal_lag_correlation(w, cc, times, L.MAX_LAG)
    assert np.nanmax(np.abs(Rzero[0] - R0[0])) > np.nanmax(np.abs(R1[0] - R0[0]))


# ------------------------------------------------------------------ the wave-following figure
def test_the_figure_null_stops_on_a_placed_trough_whose_bins_leave_the_catalog():
    import fig_wave_following as F
    assert F.REGION_HALF == 20.0 and F.REL_HALF == 12.0 and F.SEARCH == (-10.0, 10.0) and F.REL_HALF >= F.SEARCH[1]
    rel = np.arange(-F.REL_HALF, F.REL_HALF + 0.1, F.REL_STEP)
    lat = np.arange(0.0, 25.1, 2.0)
    band = (lat >= 5) & (lat <= 15)
    t = pd.date_range("2000-07-10", periods=6, freq="6h").values
    tr = types.SimpleNamespace(time=t)
    cs_time = np.repeat(t, 3)
    cs_lon = np.tile([-54.0, 0.0, 54.0], 6)
    cs_lat = np.full(18, 10.0)
    masks = {"all": np.ones(6, bool)}
    inside = [np.array([-41.0, -20.0, 0.0, 10.0, 30.0, 41.0]), np.array([-30.0, -10.0, 0.0, 5.0, 20.0, 40.0])]
    profs, null2d = F.band_profiles_for_draws(tr, masks, lambda b: inside[b], 2, cs_time, cs_lon, cs_lat, rel, lat, band)
    assert profs["all"].shape == (2, rel.size) and null2d.shape == (lat.size, rel.size)
    outside = [inside[0], np.array([-41.5, -20.0, 0.0, 10.0, 30.0, 41.0])]   # one trough half a degree too far west
    with pytest.raises(RuntimeError, match="support assertion failed"):
        F.band_profiles_for_draws(tr, masks, lambda b: outside[b], 2, cs_time, cs_lon, cs_lat, rel, lat, band)


# ------------------------------------------------------------------ the driver and the registry
def test_the_canonical_driver_builds_the_pooled_catalog_as_a_sensitivity_and_can_end_before_the_promotion():
    text = open(os.path.join(ROOT, "scripts", "run_canonical.py")).read()
    body = text[text.index("def main():"):]
    assert 'run("sens_pooled"' in body and '"--csct", cloud_inputs.POOLED_CATALOG' in body
    assert body.index('run("sens_pooled"') < body.index('run("control_model"') < body.index('run("catalog_sens"') < body.index('run("registry"')
    assert '"--alt", "deposit/sens_pooled/canonical_numbers.csv"' in body and '"--out", "catalog_sensitivity.csv"' in body
    assert body.index("if a.no_promote:") < body.index('run("promote_paper"') < body.index('run("checker"')
    assert "~/Downloads" not in body and 'f"{DL}/' not in body               # every figure goes to the folder the run was given
    assert body.index('run("yearstrat_sens"') < body.index('run("scale_split"') < body.index('run("registry"')
    split = body[body.index("    split = ["):body.index('run("scale_split"')]
    assert '"--csct", cloud_inputs.CLASS_CATALOG' in split and '"--deposit", "deposit", "--outdir", "deposit"' in split
    assert 'if not a.skip_sensitivities:\n        split += ["--alt-catalog", cloud_inputs.POOLED_CATALOG]' in split
    # the runner of the comparison of 2026-11-13 holds the driver's steps of that day, which lack the three added since
    kept = os.path.join(ROOT, "docs", "submission", "revision1", "met_alone")
    if not os.path.isdir(kept):
        pytest.skip("the runner of that comparison is kept with the working notes of the revision and is not in this tree")
    sys.path.insert(0, kept)
    import re
    import run_sequence as Q
    labels = re.findall(r'run\("([a-z0-9_]+)"', body) + re.findall(r'\("([a-zA-Z0-9_]+)", \["--', body) + re.findall(r'\("(fig[A-Za-z0-9_]+)", "scripts/', body)
    assert set(labels) - {name for name, _ in Q.steps("F")} == {"promote_paper", "promote_supplement", "checker", "sens_pooled", "catalog_sens", "scale_split",
                                                                    "dedup_counts", "exceedance_sens", "alignment_region"}


def test_the_registry_takes_the_rows_of_the_catalog_sensitivity_the_paper_would_quote(tmp_path, monkeypatch):
    import build_registry as R
    monkeypatch.chdir(tmp_path)
    assert R.rows_from_catalog_sensitivity() is None                        # an absent comparison gives no row
    os.makedirs("deposit")
    rows = [("pooled", 700.0, "lagrangian_rh", -72.0, "%", "contrast", 2.0, 1.0, 3.0, True, 2.8, 1.8, 3.8, True),
            ("pooled", 700.0, "lagrangian_rh", -48.0, "%", "contrast", 1.5, 0.5, 2.5, True, 2.2, 1.2, 3.2, True),
            ("pooled", 850.0, "lagrangian_rh", -72.0, "%", "contrast", 0.9, 0.1, 1.8, True, 1.2, 0.2, 2.2, True),
            ("pooled", 700.0, "antecedent_convection", -48.0, "systems", "contrast", 2.0, 1.5, 2.5, True, 6.7, 5.9, 7.5, True),
            ("pooled", 700.0, "lagrangian_rh_epoch_1983_1993", -72.0, "%", "contrast", 2.2, 0.9, 3.5, True, 4.3, 2.8, 5.8, True),
            ("pooled", 700.0, "thetae", -72.0, "K", "contrast", 0.3, 0.1, 0.5, True, 0.6, 0.4, 0.8, True)]
    cols = ["tier", "level", "statistic", "time_rel_h", "unit", "kind", "diff", "ci_lo", "ci_hi", "significant", "diff_alt", "ci_lo_alt", "ci_hi_alt", "significant_alt"]
    pd.DataFrame(rows, columns=cols).to_csv("deposit/catalog_sensitivity.csv", index=False)
    out = R.rows_from_catalog_sensitivity()
    assert list(out.statistic) == ["catsens_lagrangian_rh", "catsens_antecedent_convection", "catsens_lagrangian_rh_epoch_1983_1993"]
    head = out.iloc[0]
    assert (head["diff"], head.ci_lo, head.ci_hi, head.kind, head.unit, head.level, head.time_rel_h) == (2.8, 1.8, 3.8, "sensitivity", "%", 700.0, -72.0)
    assert out.iloc[1].unit == "systems" and out.iloc[1].time_rel_h == -48.0

def test_the_registry_takes_the_moisture_term_of_each_tier_with_its_sample(tmp_path, monkeypatch):
    import build_registry as R
    monkeypatch.chdir(tmp_path)
    os.makedirs("deposit")
    rows = [("inflow_rh", "primary", "dev", 1214, 146, 0.0033, 0.0267, 0.90, 1.003, 0.952, 1.057),
            ("inflow_rh", "primary", "heldout", 6101, 698, 0.0288, 0.0122, 0.019, 1.029, 1.005, 1.054),
            ("inflow_rh", "full+tcwv", "heldout", 6101, 698, 0.0220, 0.0127, 0.085, 1.022, 0.997, 1.048),
            ("box_rh", "primary", "pooled", 7315, 844, 0.0040, 0.0099, 0.69, 1.004, 0.985, 1.024)]
    cols = ["moisture", "spec", "tier", "n", "waves", "coef", "se", "pvalue", "irr", "irr_lo", "irr_hi"]
    pd.DataFrame(rows, columns=cols).to_csv("deposit/control_model_coeffs.csv", index=False)
    out = R.rows_from_control_model().set_index(["tier", "statistic"])
    assert sorted(out.index) == [("dev", "cm_coef_primary_inflow_rh"), ("heldout", "cm_coef_full_plus_tcwv_inflow_rh"),
                                 ("heldout", "cm_coef_primary_inflow_rh"), ("pooled", "cm_coef_primary_box_rh")]
    r = out.loc[("heldout", "cm_coef_primary_inflow_rh")]
    assert (r.irr, r.irr_lo, r.irr_hi, r.pvalue, r.n_obs, r.n_waves, r.level, r.time_rel_h) == (1.029, 1.005, 1.054, 0.019, 6101, 698, 700.0, -72.0)
    assert out.loc[("heldout", "cm_coef_full_plus_tcwv_inflow_rh")].irr_lo == 0.997


def availability_world(tmp_path, record=None, counts=((1496.0, 841.0), (7332.0, 1788.0), (8828.0, 2629.0))):
    import hashlib
    os.makedirs(tmp_path / "deposit", exist_ok=True)
    os.makedirs(tmp_path / "scripts", exist_ok=True)
    (tmp_path / "scripts" / "met_observed_slots.csv").write_text("time\n1983-07-01 00:00:00\n")
    sha = hashlib.sha256((tmp_path / "scripts" / "met_observed_slots.csv").read_bytes()).hexdigest()
    rec = dict(slots_of_the_seasons=18400, slots_with_a_row=17757, slots_without_a_row=643, slots_taken_as_partial=[dict(rows=k) for k in (1, 3, 4, 4, 8, 10)],
               slots_observed=17751, slots_not_observed=649, partial_fraction=0.25, out_sha256=sha)
    rec.update(record or {})
    (tmp_path / "scripts" / "met_observed_slots_record.json").write_text(json.dumps(rec))
    rows = []
    for tier, (kept, gone) in zip(("dev", "heldout", "pooled"), counts):
        rows += [(tier, "n_corridor_troughs", kept), (tier, "n_window_unobserved", gone), (tier, "n_waves", 7.0)]
    pd.DataFrame(rows, columns=["tier", "statistic", "diff"]).to_csv(tmp_path / "deposit" / "canonical_numbers.csv", index=False)


def test_the_registry_states_the_counts_behind_the_availability_criterion(tmp_path, monkeypatch):
    import build_registry as R
    monkeypatch.chdir(tmp_path)
    assert R.rows_from_observation_rule() is None                            # nothing to read gives no row
    availability_world(tmp_path)
    out = R.rows_from_observation_rule().set_index(["tier", "statistic"])["diff"]
    assert (out[("pooled", "obs_slots_total")], out[("pooled", "obs_slots_no_row")], out[("pooled", "obs_slots_partial")],
            out[("pooled", "obs_slots_passing")], out[("pooled", "obs_slots_with_row")], out[("pooled", "obs_partial_fraction")]) == (18400, 643, 6, 17751, 17757, 0.25)
    assert out[("pooled", "n_calendar_eligible")] == 11457 and out[("pooled", "pct_window_unobserved")] == pytest.approx(100 * 2629 / 11457)
    assert out[("dev", "n_calendar_eligible")] == 2337 and out[("dev", "pct_window_unobserved")] == pytest.approx(100 * 841 / 2337)
    assert out[("heldout", "pct_window_unobserved")] == pytest.approx(100 * 1788 / 9120)
    for plant, message in ((dict(out_sha256="0" * 64), "is not the record of"), (dict(slots_without_a_row=642), "do not add up"),
                           (dict(slots_observed=17752), "do not add up"), (dict(slots_taken_as_partial=[]), "do not add up")):
        availability_world(tmp_path, record=plant)
        with pytest.raises(SystemExit, match=message):
            R.rows_from_observation_rule()
    availability_world(tmp_path)
    d = pd.read_csv("deposit/canonical_numbers.csv")
    pd.concat([d, d[(d.tier == "dev") & (d.statistic == "n_window_unobserved")]]).to_csv("deposit/canonical_numbers.csv", index=False)
    with pytest.raises(SystemExit, match="2 times"):
        R.rows_from_observation_rule()
    d[d.statistic != "n_window_unobserved"].to_csv("deposit/canonical_numbers.csv", index=False)
    assert R.rows_from_observation_rule() is None                            # a deposit built before the criterion


def split_record(R, labels=None, drop=None, twice=None):
    """The paired rows of a made-up record: every declared difference and one the registry does not take."""
    pairs, k = [], 0
    wanted = list(R.SCALE_DIFFERENCES) + [("not_registered", "whole sample", (R._H, "merged 1"), (R._BOX, "merged 1"))]
    for name, sample, first, second in wanted:
        if labels is not None and not {first[1], second[1]} <= set(labels) | {"merged 1"}:
            continue
        for q in R.SCALE_QUANTITIES:
            k += 1
            if (name, q) == drop:
                continue
            row = dict(sample=sample, first=list(first) + [R._BG], second=list(second) + [R._BG], quantity=q, diff=-k / 10.0, low=-k / 10.0 - 2.0,
                       high=-k / 10.0 + 2.0, held_low=-k / 10.0 - 0.5, held_high=-k / 10.0 + 0.5)
            pairs += [row, row] if (name, q) == twice else [row]
    for q in R.SCALE_QUANTITIES:                                             # the same labels over another background and another measure
        pairs.append(dict(sample="whole sample", first=[R._H, "canonical", "classified observations"], second=[R._BOX, "canonical", "classified observations"],
                          quantity=q, diff=9.0, low=8.0, high=10.0, held_low=8.5, held_high=9.5))
        pairs.append(dict(sample="whole sample", first=["inflow RH at -24 h, 700 hPa", "canonical", R._BG], second=[R._BOX, "canonical", R._BG],
                          quantity=q, diff=7.0, low=6.0, high=8.0, held_low=6.5, held_high=7.5))
    with open("deposit/scale_split_record.json", "w") as f:
        json.dump(dict(paired=pairs), f)
    return pairs


def split_table(R, labels=None):
    rows, k = [], 0
    for measure, _, _, _ in R.SCALE_MEASURES + (("column water vapor, mm", "", 0, ""),):
        for sample, _, _ in R.SCALE_SAMPLES:
            for label in (labels or [name for name, _ in R.SCALE_LABELS] + ["merged 1"]):
                for q in R.SCALE_QUANTITIES:
                    for bg in ("every eligible trough", "classified observations"):
                        k += 1
                        rows.append(dict(sample=sample, measure=measure, labels=label, background=bg, quantity=q, diff=k / 100.0, low=k / 100.0 - 1.0,
                                         high=k / 100.0 + 1.0, draws_used=20000, replicate_median=k / 100.0, held_low=k / 100.0 - 0.25,
                                         held_high=k / 100.0 + 0.25, n_quiet=300 + k, n_active=200 + k))
    return pd.DataFrame(rows)


def test_the_registry_takes_the_split_by_scale_with_the_interval_declared_for_each_row(tmp_path, monkeypatch):
    import build_registry as R
    monkeypatch.chdir(tmp_path)
    assert R.rows_from_scale_split() is None
    os.makedirs("deposit")
    t = split_table(R)
    t.to_csv("deposit/scale_split.csv", index=False)
    with pytest.raises(SystemExit, match="scale_split_record.json is not"):  # the table without its record
        R.rows_from_scale_split()
    pairs = split_record(R)
    out = R.rows_from_scale_split().set_index("statistic")
    n_rows = (3 * 5 + 3 * 1 + 2 * 2 + 2 * 3) * 7
    assert len(out) == n_rows + 8 * 7 and out.index.is_unique and set(out.kind) == {"split"} and set(out.tier) == {"pooled"}

    def source(sample, measure, label, q):
        m = (t["sample"] == sample) & (t.measure == measure) & (t["labels"] == label) & (t.quantity == q) & (t.background == "every eligible trough")
        return t[m].iloc[0]
    total, part = source("whole sample", "inflow RH at -72 h, 700 hPa", "canonical", "total"), source("whole sample", "inflow RH at -72 h, 700 hPa", "canonical", "waves")
    r = out.loc["ss_rh_canonical_total"]
    assert (r["diff"], r.ci_lo, r.ci_hi, r.n_quiet, r.n_active, r.time_rel_h, r.level, r.n_boot) == (total["diff"], total.low, total.high, total.n_quiet, total.n_active, -72.0, 700.0, 20000)
    r = out.loc["ss_rh_canonical_waves"]
    assert (r["diff"], r.ci_lo, r.ci_hi) == (part["diff"], part.held_low, part.held_high) and part.held_low != part.low
    east = source("10 E to 40 E", "inflow RH at -72 h, 700 hPa", "canonical", "years and cells")
    assert out.loc["ss_rh_canonical_years_and_cells_east", "diff"] == east["diff"]
    other = source("whole sample", "antecedent count", "every position pooled, year-matched", "passages")
    r = out.loc["ss_antecedent_allpos_yearmatched_passages"]
    assert (r["diff"], r.time_rel_h, r.unit) == (other["diff"], -48.0, "systems")
    assert out.loc["ss_box_yearmatched_total", "time_rel_h"] == -24.0
    assert not any("merged" in s or "tcwv" in s or "not_registered" in s for s in out.index)
    assert "ss_box_canonical_total_east" not in out.index and "ss_rh_yearmatched_total_east" not in out.index and "ss_rh_era_total_1983_1993" not in out.index
    assert "ss_rh_era_waves_1998_2007" in out.index and "ss_rh_allpos_total_1983_1993" in out.index and "ss_box_era_total" in out.index
    assert bool(out.loc["ss_rh_canonical_total", "significant"]) == bool(total.low > 0 or total.high < 0)
    # a difference between two rows comes from its own paired row, a total with the declared interval and a part with the held one
    src = {(tuple(x["first"]), tuple(x["second"]), x["sample"], x["quantity"]): x for x in pairs}
    x = src[((R._H, "canonical", R._BG), (R._H, "canonical", R._BG), "1983 to 1997 less 1998 to 2007", "total")]
    r = out.loc["ssd_eras_canonical_total"]
    assert (r["diff"], r.ci_lo, r.ci_hi, r.time_rel_h, r.significant) == (x["diff"], x["low"], x["high"], -72.0, bool(x["low"] > 0 or x["high"] < 0))
    x = src[((R._H, "canonical", R._BG), (R._H, "every position pooled", R._BG), "whole sample", "years and cells")]
    r = out.loc["ssd_canonical_less_allpos_years_and_cells"]
    assert (r["diff"], r.ci_lo, r.ci_hi) == (x["diff"], x["held_low"], x["held_high"])
    x = src[((R._H, "year-matched", R._BG), (R._BOX, "year-matched", R._BG), "whole sample", "waves")]
    assert out.loc["ssd_rh_less_box_yearmatched_waves", "diff"] == x["diff"] != out.loc["ssd_rh_less_box_canonical_waves", "diff"]
    assert out.loc["ssd_epochs_allpos_total", "diff"] != out.loc["ssd_epochs_canonical_total", "diff"] != out.loc["ssd_eras_era_total", "diff"]
    for plant, message in ((dict(drop=("eras_era", "cells")), "lacks the difference eras_era, cells"), (dict(twice=("epochs_canonical", "total")), "2 times")):
        split_record(R, **plant)
        with pytest.raises(SystemExit, match=message):
            R.rows_from_scale_split()
    # without a second catalog the labelings and the differences from it are absent, and the rest is registered
    kept = ["canonical", "year-matched", "era"]
    split_table(R, labels=kept).to_csv("deposit/scale_split.csv", index=False)
    split_record(R, labels=kept)
    assert len(R.rows_from_scale_split()) == (3 * 3 + 3 * 1 + 2 * 1 + 2 * 2) * 7 + 5 * 7
    split_record(R)
    t.to_csv("deposit/scale_split.csv", index=False)
    first = (t["sample"] == "0 to 10 E") & (t.measure == "inflow RH at -72 h, 700 hPa") & (t["labels"] == "canonical") & (t.quantity == "cells") & (t.background == "every eligible trough")
    t[~first].to_csv("deposit/scale_split.csv", index=False)
    with pytest.raises(SystemExit, match="lacks 0 to 10 E"):
        R.rows_from_scale_split()
    pd.concat([t, t[first]]).to_csv("deposit/scale_split.csv", index=False)
    with pytest.raises(SystemExit, match="2 times"):
        R.rows_from_scale_split()


# ------------------------------------------------------------------ the dry run of the manuscript
def test_the_dry_run_gives_each_tagged_number_its_replacement_and_flags_words_and_bounds(tmp_path):
    kept = os.path.join(ROOT, "docs", "submission", "revision1", "input_rules")
    if not os.path.isdir(kept):
        pytest.skip("the dry run of the manuscript is kept with the working notes of the revision and is not in this tree")
    sys.path.insert(0, kept)
    import manuscript_dry_run as D
    import number_audit as A
    cols = ["tier", "level", "statistic", "time_rel_h", "diff", "ci_lo", "ci_hi"]
    old = pd.DataFrame([["pooled", 700.0, "a", -72.0, 2.86, 1.86, 3.86], ["pooled", 700.0, "b", -72.0, 1.9, 1.13, 2.67],
                        ["pooled", np.nan, "n", np.nan, 8358.0, np.nan, np.nan], ["pooled", 700.0, "c", -72.0, 0.5, 0.1, 0.9]], columns=cols)
    new = pd.DataFrame([["pooled", 700.0, "a", -72.0, 2.30, 1.24, 3.37], ["pooled", 700.0, "b", -72.0, 0.83, 0.04, 1.61],
                        ["pooled", np.nan, "n", np.nan, 6557.0, np.nan, np.nan], ["pooled", 700.0, "c", -72.0, 0.5, -0.1, 0.9]], columns=cols)
    old.to_csv(tmp_path / "old.csv", index=False)
    new.to_csv(tmp_path / "new.csv", index=False)
    o, _ = A.load_canonical(str(tmp_path / "old.csv"))
    n, _ = A.load_canonical(str(tmp_path / "new.csv"))
    text = ("Over 8,358<!--n:pooled::n::diff:u0--> troughs the contrast is +2.9<!--n:pooled:700:a:-72:diff:1-->.\n"
            "At wave scale it is +1.9<!--n:pooled:700:b:-72:diff:1--> (+1.1<!--n:pooled:700:b:-72:ci_lo:1--> to +2.7<!--n:pooled:700:b:-72:ci_hi:1-->),\n"
            "and c is +0.5<!--n:pooled:700:c:-72:diff:1--> (+0.1<!--n:pooled:700:c:-72:ci_lo:1-->).\n")
    rows = D.tags_of("t.md", text, o, n)
    by = {(r["statistic"], r["field"]): r for r in rows}
    assert [r["line"] for r in rows] == [1, 1, 2, 2, 2, 3, 3]
    assert by[("n", "diff")]["rebuilt"] == "6,557" and by[("n", "diff")]["search"] == "8,358<!--n:pooled::n::diff:u0-->" and by[("n", "diff")]["replace"] == "6,557<!--n:pooled::n::diff:u0-->"
    assert by[("a", "diff")]["rebuilt"] == "+2.3" and by[("a", "diff")]["changes"] and not by[("a", "diff")]["word_changes"]
    assert by[("b", "ci_lo")]["rebuilt"] == "+0.0" and by[("b", "ci_lo")]["prints_as_no_difference"] and not by[("b", "ci_hi")]["prints_as_no_difference"]
    assert by[("c", "diff")]["changes"] is False and by[("c", "diff")]["word_changes"] and (by[("c", "diff")]["word_submitted"], by[("c", "diff")]["word_rebuilt"]) == ("resolved", "not resolved")
    assert by[("c", "ci_lo")]["rebuilt"] == "-0.1" and not by[("c", "ci_lo")]["prints_as_no_difference"]
    applied = text
    for r in rows:
        applied = applied.replace(r["search"], r["replace"])
    assert A.audit(applied, n) == [] and len(A.audit(text, n)) == sum(1 for r in rows if r["changes"])
    with pytest.raises(SystemExit, match="does not print the submitted value"):       # a text that is not the submitted one
        D.tags_of("t.md", text.replace("+2.9<", "+3.0<"), o, n)
    with pytest.raises(SystemExit, match="no value in one of the two registries"):     # a row the rebuild lacks
        D.tags_of("t.md", text, o, {k: v for k, v in n.items() if k[2] != "c"})
