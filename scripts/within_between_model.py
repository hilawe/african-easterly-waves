#!/usr/bin/env python
"""Common-sample within-between decomposition of the inflow-moisture association.

The paper's title rests on the claim that the moisture contrast separates whole waves
rather than passages of the same wave. Until round 9 that claim leaned on two cohort
summaries that are not a formal decomposition: the exclusively-active against
exclusively-quiet contrast (190 and 284 waves) and the paired contrast in a DIFFERENT
cohort of 514 mixed waves. Exclusive classification selects for track length, so the two
numbers are not additive pieces of one estimand (raised by the 2026-08-12
a later check. The cohort summaries stay in the paper as descriptive
context).

This script supplies the formal version on ONE sample, the standard hybrid
(within-between) model. The 700 hPa inflow relative humidity is split into the wave
mean and each observation's deviation from its wave mean, both scaled by the pooled
inflow-humidity standard deviation, and both enter the same Poisson count model used by
scripts/control_model.py, with the same controls (antecedent convection, amplitude,
shear), the same longitude-bin-by-month and year fixed effects, and cluster-robust
standard errors on the wave. The wave-mean coefficient is the between-wave association,
the deviation coefficient the within-wave association, and a Wald test compares them
directly.

Consumes deposit/control_model_design.csv (built by control_model.py), so it runs after
that step in run_canonical.py. Writes deposit/within_between_model.csv.
"""

import argparse
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

CONTROLS = ("antecedent", "amplitude", "shear")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="deposit/control_model_design.csv")
    ap.add_argument("--outdir", default="deposit")
    a = ap.parse_args()

    d = pd.read_csv(a.design)
    d = d[np.isfinite(d[["inflow_rh", *CONTROLS]]).all(axis=1)].reset_index(drop=True)

    # the estimand needs one observation per wave per time; the round-9 deduplication
    # repair guarantees this upstream, and this guard keeps the model honest if it is
    # ever run against a pre-repair table
    dup = int(d.duplicated(["wave", "time"]).sum())
    if dup:
        raise SystemExit(f"design table has {dup} duplicate (wave, time) rows; "
                         f"the within-between estimand is undefined on branched waves")

    sd = float(d.inflow_rh.std(ddof=0))
    d["wb_between"] = (d.groupby("wave").inflow_rh.transform("mean")
                       - d.inflow_rh.mean()) / sd
    d["wb_within"] = (d.inflow_rh
                      - d.groupby("wave").inflow_rh.transform("mean")) / sd
    for c in CONTROLS:
        d[f"z_{c}"] = (d[c] - d[c].mean()) / d[c].std(ddof=0)

    rhs = ("wb_between + wb_within + "
           + " + ".join(f"z_{c}" for c in CONTROLS)
           + " + C(lonmonth) + C(year)")
    res = smf.glm(f"response ~ {rhs}", data=d,
                  family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": d["wave"].values})

    rows = []
    for term, name in (("wb_between", "between_wave"), ("wb_within", "within_wave")):
        ci = res.conf_int().loc[term]
        rows.append(dict(component=name,
                         irr=float(np.exp(res.params[term])),
                         irr_lo=float(np.exp(ci[0])), irr_hi=float(np.exp(ci[1])),
                         pvalue=float(res.pvalues[term])))
    wald = res.wald_test("wb_between - wb_within = 0", scalar=True)
    rows.append(dict(component="equality", irr=np.nan, irr_lo=np.nan, irr_hi=np.nan,
                     pvalue=float(wald.pvalue), chi2=float(wald.statistic)))
    out = pd.DataFrame(rows)
    out["n_obs"] = len(d)
    out["n_waves"] = int(d.wave.nunique())
    out["rh_sd"] = sd
    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, "within_between_model.csv")
    out.to_csv(path, index=False, float_format="%.6f")

    for r in rows[:2]:
        print(f"  {r['component']:13s} IRR {r['irr']:.4f} "
              f"[{r['irr_lo']:.4f}, {r['irr_hi']:.4f}]  p = {r['pvalue']:.3f}")
    print(f"  equality      chi2 {rows[2]['chi2']:.2f}  p = {rows[2]['pvalue']:.3f}")
    print(f"wrote {path} ({len(d)} obs, {d.wave.nunique()} waves)")


if __name__ == "__main__":
    main()
