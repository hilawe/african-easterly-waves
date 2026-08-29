"""Tests for the common-sample within-between decomposition.

The model is the formal replacement for the exclusive/mixed cohort comparison, so it
gets the write-time discipline: a synthetic record with a KNOWN between-wave signal and
no within-wave signal must recover that structure, and the branched-input guard must
refuse a table on which the estimand is undefined.
"""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

HERE = "scripts/within_between_model.py"


def synth(seed=7, n_waves=120, obs_per_wave=8, between=0.06, within=0.0):
    """Poisson counts driven by the wave-mean of humidity, not the deviation."""
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_waves):
        wave_rh = rng.normal(60.0, 8.0)
        for k in range(obs_per_wave):
            rh = wave_rh + rng.normal(0.0, 4.0)
            mu = np.exp(1.5 + between * (wave_rh - 60.0) + within * (rh - wave_rh))
            rows.append(dict(
                wave=w, time=f"2000-07-{(k % 28) + 1:02d} 00:00:00",
                response=rng.poisson(mu), inflow_rh=rh,
                antecedent=rng.poisson(3.0), amplitude=rng.normal(0, 1),
                shear=rng.normal(0, 1),
                lonmonth=f"c{w % 4}", year=2000 + (w % 5)))
    return pd.DataFrame(rows)


def run_script(design_path, outdir):
    return subprocess.run([sys.executable, HERE, "--design", str(design_path),
                          "--outdir", str(outdir)], capture_output=True, text=True)


def test_recovers_between_wave_signal_and_null_within(tmp_path):
    d = synth()
    p = tmp_path / "design.csv"
    d.to_csv(p, index=False)
    r = run_script(p, tmp_path)
    assert r.returncode == 0, r.stderr
    out = pd.read_csv(tmp_path / "within_between_model.csv")
    bet = out[out.component == "between_wave"].iloc[0]
    wit = out[out.component == "within_wave"].iloc[0]
    eq = out[out.component == "equality"].iloc[0]
    # the planted structure: between resolved and above one, within unresolved
    assert bet.irr > 1.0 and bet.irr_lo > 1.0
    assert wit.irr_lo < 1.0 < wit.irr_hi
    assert eq.pvalue < 0.05


def test_no_planted_signal_resolves_nothing(tmp_path):
    # mutation check: with no signal at all, neither component may be resolved
    d = synth(between=0.0, within=0.0, seed=11)
    p = tmp_path / "design.csv"
    d.to_csv(p, index=False)
    r = run_script(p, tmp_path)
    assert r.returncode == 0, r.stderr
    out = pd.read_csv(tmp_path / "within_between_model.csv")
    for comp in ("between_wave", "within_wave"):
        row = out[out.component == comp].iloc[0]
        assert row.irr_lo < 1.0 < row.irr_hi, comp


def test_branched_table_is_refused(tmp_path):
    # the defect shape: one wave, two rows at the same timestamp
    d = synth()
    dup = d.iloc[[0]].copy()
    dup["inflow_rh"] += 5.0
    d = pd.concat([d, dup], ignore_index=True)
    p = tmp_path / "design.csv"
    d.to_csv(p, index=False)
    r = run_script(p, tmp_path)
    assert r.returncode != 0
    assert "duplicate (wave, time)" in (r.stdout + r.stderr)
