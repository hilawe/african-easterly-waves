#!/usr/bin/env python
"""Temporal-stratification sensitivity for the Lagrangian moisture-supply contrast.

The canonical split stratifies the response terciles within longitude-by-calendar-month
cells, which matches the two classes in geography and season but NOT in year: the
detection catalog's annual abundance varies severalfold across 1983-2007, and the
MCS-active class sits about four years later on average than the MCS-quiet class. This
harness quantifies what that temporal imbalance contributes to the headline contrast by
re-splitting the SAME eligible sample within year-by-month-by-longitude cells, so the
two classes are matched in year as well, and recomputing the -72 h along-inflow 700 hPa
relative humidity contrast with the wave-cluster bootstrap.

A pure consumer of deposit/eligible_troughs.csv (written by control_model.py on the
repaired pipeline), so it integrates nothing. The within-year split holds fewer
observations per cell, so the 30-observation cell minimum drops more cells; the retained
counts are reported alongside. Writes deposit/yearstrat_sensitivity.csv.
"""

import argparse
import os

import numpy as np
import pandas as pd

from aew.environment import cluster_bootstrap_diff

EDGES = np.arange(-30, 41, 10.0)
MIN_BIN = 30


def stratified_terciles_cells(df, cols):
    """Exact-thirds tercile masks within the given stratification columns, matching the
    canonical convention (inclusive boundaries, cells under MIN_BIN excluded)."""
    lo = np.zeros(len(df), bool)
    hi = np.zeros(len(df), bool)
    for _, g in df.groupby(cols, sort=False):
        if len(g) < MIN_BIN:
            continue
        a, b = np.nanquantile(g.response, [1.0 / 3.0, 2.0 / 3.0])
        pos = df.index.get_indexer(g.index)
        lo[pos[g.response.values <= a]] = True
        hi[pos[g.response.values >= b]] = True
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eligible", default="deposit/eligible_troughs.csv")
    ap.add_argument("--outdir", default="deposit")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d = pd.read_csv(a.eligible).rename(columns={"wave": "traj_id", "inflow_rh": "H"})
    d = d[np.isfinite(d["H"])].copy()
    d["time"] = pd.to_datetime(d["time"])
    d["year"] = d.time.dt.year
    d["month"] = d.time.dt.month
    d["lonbin"] = pd.cut(d.lon, EDGES, right=False, labels=False)
    d = d.reset_index(drop=True)

    rng = np.random.default_rng(a.seed)
    rows = []
    for label, cols in (("monthlon", ["month", "lonbin"]),
                        ("yearmonthlon", ["year", "month", "lonbin"])):
        lo, hi = stratified_terciles_cells(d, cols)
        diff, ci_lo, ci_hi, nq, na = cluster_bootstrap_diff(
            d.traj_id[lo], d.H[lo], d.traj_id[hi], d.H[hi], rng)
        gap = d.year[hi].mean() - d.year[lo].mean()
        rows.append(dict(split=label, diff=diff, ci_lo=ci_lo, ci_hi=ci_hi,
                         n_quiet=int(lo.sum()), n_active=int(hi.sum()),
                         clusters_quiet=nq, clusters_active=na,
                         year_gap=gap,
                         significant=bool(not (ci_lo <= 0 <= ci_hi))))
        print(f"{label:14s} diff {diff:+.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}] "
              f"n {int(lo.sum())}/{int(hi.sum())} year-gap {gap:+.2f}", flush=True)

    out = os.path.join(a.outdir, "yearstrat_sensitivity.csv")
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6f")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
