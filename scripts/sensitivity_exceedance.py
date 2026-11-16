#!/usr/bin/env python
"""The moist-tail result at other percentiles of the inflow humidity.

The deposit reports the share of each class whose -72 h inflow humidity exceeds the
pooled 70th percentile of the classified observations. A reviewer asked whether that
result depends on the percentile. This script repeats the same construction at other
percentiles: the threshold is the percentile of the classified observations taken
together, the share of each class above it is a mean of zeros and ones, and the
difference of the shares carries a wave-cluster bootstrap interval.

A pure consumer of deposit/cases_pooled_700.csv with its own random streams, one for
each percentile, so no canonical number moves. Before it writes, it requires that its
own construction at the canonical percentile reproduce the deposit: the flags must
equal the deposited flags row for row, and the two shares, their difference and the
threshold must equal the deposit's rows. The canonical percentile is used for that check
alone and is not written out a second time with another interval.

Writes deposit/exceedance_sensitivity.csv.
"""

import argparse
import os

import numpy as np
import pandas as pd

from aew.environment import cluster_bootstrap_diff

N_BOOT = 20_000
CANONICAL = 70.0
PERCENTILES = (50.0, 60.0, 75.0, 80.0, 90.0)
QUIET, ACTIVE = "MCS-quiet", "MCS-active"


def refuse(msg):
    raise SystemExit(f"REFUSED: {msg}")


def exceedance(rh, label, wave, percentile, n_boot=None, seed=0):
    """Threshold, shares, ratio and difference at one percentile. An interval when n_boot is given."""
    rh = np.asarray(rh, float)
    label = np.asarray(label)
    wave = np.asarray(wave)
    ok = np.isfinite(rh)
    lo, hi = (label == QUIET) & ok, (label == ACTIVE) & ok
    if not lo.any() or not hi.any():
        refuse("a class holds no observation with a humidity value")
    thr = float(np.nanpercentile(rh[lo | hi], percentile))
    exc = (rh > thr).astype(float)
    p_q, p_a = float(exc[lo].mean()), float(exc[hi].mean())
    out = dict(percentile=float(percentile), threshold=thr, mean_quiet=100.0 * p_q, mean_active=100.0 * p_a,
               diff=100.0 * (p_a - p_q), ratio=p_a / p_q if p_q > 0 else np.nan,
               n_quiet=int(lo.sum()), n_active=int(hi.sum()), flags=exc)
    if n_boot:
        rng = np.random.default_rng([int(seed), 7, int(round(10 * percentile))])
        d, a, b, nq, na = cluster_bootstrap_diff(wave[lo], exc[lo], wave[hi], exc[hi], rng, n_boot=n_boot)
        if abs(100.0 * d - out["diff"]) > 1e-9:
            refuse("the bootstrap's point value differs from the direct difference")
        out.update(ci_lo=100.0 * a, ci_hi=100.0 * b, clusters_quiet=int(nq), clusters_active=int(na), n_boot=int(n_boot),
                   significant=bool(not (a <= 0.0 <= b)))
    return out


def bind_to_deposit(cases, dep, tolerance=1e-4):
    """The construction at the canonical percentile must be the deposit's. Stops on a mismatch."""
    got = exceedance(cases.rh_m72.values, cases.label.values, cases.traj_id.values, CANONICAL)
    kept = np.isin(cases.label.values, (QUIET, ACTIVE)) & np.isfinite(cases.rh_m72.values)
    if not np.array_equal(got["flags"][kept], cases.exceed_p70.values[kept].astype(float)):
        refuse("the flags at the canonical percentile differ from the deposited flags")
    r = pd.read_csv(os.path.join(dep, "canonical_numbers.csv"))
    r = r[(r.tier == "pooled") & np.isclose(r.level, 700.0) & np.isclose(r.time_rel_h, -72.0)]

    def one(stat):
        m = r.statistic == stat
        if int(m.sum()) != 1:
            refuse(f"the deposit states {stat} {int(m.sum())} times")
        return r[m].iloc[0]
    row = one("exceedance_p70")
    pairs = (("mean_quiet", got["mean_quiet"], row.mean_quiet), ("mean_active", got["mean_active"], row.mean_active),
             ("diff", got["diff"], row["diff"]), ("threshold", got["threshold"], one("exceedance_threshold")["diff"]),
             ("ratio", got["ratio"], one("exceedance_ratio")["diff"]))
    for name, mine, theirs in pairs:
        if not (np.isfinite(mine) and np.isfinite(theirs) and abs(mine - theirs) <= tolerance):
            refuse(f"{name} at the canonical percentile is {mine:.6f} and the deposit states {theirs:.6f}")
    if (got["n_quiet"], got["n_active"]) != (int(row.n_quiet), int(row.n_active)):
        refuse("the class sizes differ from the deposit's")
    return {name: [float(mine), float(theirs)] for name, mine, theirs in pairs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposit", default="deposit")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cases = pd.read_csv(os.path.join(a.deposit, "cases_pooled_700.csv"))
    checked = bind_to_deposit(cases, a.deposit)
    print(f"the construction at the {CANONICAL:.0f}th percentile reproduces the deposit ({len(checked)} values)")
    rows = []
    for p in PERCENTILES:
        r = exceedance(cases.rh_m72.values, cases.label.values, cases.traj_id.values, p, n_boot=a.n_boot, seed=a.seed)
        r.pop("flags")
        rows.append(r)
        print(f"p{p:.0f}: threshold {r['threshold']:.1f}, active {r['mean_active']:.1f} against quiet {r['mean_quiet']:.1f}, "
              f"ratio {r['ratio']:.2f}, difference {r['diff']:+.1f} ({r['ci_lo']:+.1f} to {r['ci_hi']:+.1f})", flush=True)
    path = os.path.join(a.deposit, "exceedance_sensitivity.csv")
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.6f")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
