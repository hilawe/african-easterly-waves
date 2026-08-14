#!/usr/bin/env python
"""Deduplication tie-rule sensitivity.

The deduplication keeps the LONGEST member trajectory of each merged wave, and where two
or more members are co-longest the rule breaks the tie by smallest original identifier.
That tie-break is arbitrary, and it is not rare. A code audit (2026-08-13)
found 614 of 2,972 merged components carry a tie, so the arbitrary part of the rule
decides which track represents about a fifth of the merged waves.

This compares the canonical deposit against a full pooled rebuild under the opposite tie
rule (largest original identifier), produced by

    scripts/build_deposit.py --tiers pooled --outdir deposit/sens_tierule --tie-rule largest

and writes one row per pooled contrast, so the comparison is deposited rather than
asserted. Reading the two directories is all this does; it fits no model and draws no
random numbers, so it cannot perturb any canonical value.

Writes deposit/tierule_sensitivity.csv.
"""

import argparse
import os

import numpy as np
import pandas as pd

KEY = ["tier", "level", "statistic", "time_rel_h", "unit", "kind"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="deposit/canonical_numbers.csv")
    ap.add_argument("--alt", default="deposit/sens_tierule/canonical_numbers.csv")
    ap.add_argument("--outdir", default="deposit")
    a = ap.parse_args()

    if not os.path.exists(a.alt):
        raise SystemExit(
            f"{a.alt} absent. Run:\n"
            f"  .venv/bin/python scripts/build_deposit.py --tiers pooled "
            f"--outdir deposit/sens_tierule --tie-rule largest")

    base = pd.read_csv(a.base)
    alt = pd.read_csv(a.alt)
    m = base.merge(alt, on=KEY, suffixes=("", "_alt"), validate="one_to_one")
    c = m[m["kind"] == "contrast"].dropna(subset=["diff", "diff_alt"]).copy()
    if c.empty:
        raise SystemExit("no comparable contrast rows; the two deposits do not align")

    c["delta"] = c["diff_alt"] - c["diff"]
    # A verdict flip is the thing a reader needs, so it is a column rather than prose.
    c["verdict_flips"] = (c["significant"].astype(str)
                          != c["significant_alt"].astype(str))

    out = c[KEY + ["diff", "ci_lo", "ci_hi", "significant",
                   "diff_alt", "ci_lo_alt", "ci_hi_alt", "significant_alt",
                   "delta", "verdict_flips"]].sort_values(KEY)
    path = os.path.join(a.outdir, "tierule_sensitivity.csv")
    out.to_csv(path, index=False, float_format="%.6f")

    d = out["delta"].abs()
    flips = out[out["verdict_flips"]]
    print(f"tie-rule sensitivity over {len(out)} pooled contrasts")
    print(f"  max |change| {d.max():.4f} on "
          f"{out.loc[d.idxmax(), 'statistic']}")
    print(f"  median |change| {d.median():.4f}")
    print(f"  significance verdict flips: {len(flips)}")
    for _, r in flips.iterrows():
        print(f"    {r['statistic']} {r['level']:.0f} hPa t={r['time_rel_h']:.0f}: "
              f"{r['diff']:+.3f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] -> "
              f"{r['diff_alt']:+.3f} [{r['ci_lo_alt']:+.2f}, {r['ci_hi_alt']:+.2f}]")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
