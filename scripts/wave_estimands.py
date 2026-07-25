#!/usr/bin/env python
"""The three R4 wave-level estimands (REPAIR_SPEC.md), from canonical inputs.

Estimand 1 (wave-unit) reads the control model's cached design matrix (every eligible
corridor trough with its response and along-inflow moisture), so it covers all waves,
not only the extreme classes. Estimands 2 and 3 (paired within-wave, between-only)
read the deposit's pooled 700 hPa case table. Writes deposit/wave_estimands.csv.

Run AFTER the canonical rerun of build_deposit.py and control_model.py (with --cache),
so every input reflects the repaired estimands.
"""

import argparse
import os

import numpy as np
import pandas as pd

from aew.wave_level import (
    between_contrast,
    paired_contrast,
    wave_table,
    wave_unit_contrast,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eligible", default="deposit/eligible_troughs.csv",
                    help="unfiltered eligible-trough table written by control_model.py")
    ap.add_argument("--cases", default="deposit/cases_pooled_700.csv")
    ap.add_argument("--outdir", default="deposit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=20000)
    a = ap.parse_args()

    rows = []
    waves = None
    if os.path.exists(a.eligible):
        d = pd.read_csv(a.eligible).rename(
            columns={"wave": "traj_id", "inflow_rh": "H"})
        waves = wave_table(d[["traj_id", "time", "lon", "response", "H"]])
        r = wave_unit_contrast(waves, rng=np.random.default_rng(a.seed),
                               n_boot=a.n_boot)
        rows.append(r)
        print(f"wave_unit: {r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] "
              f"(two-stage season sens [{r['sens_season_lo']:+.3f}, "
              f"{r['sens_season_hi']:+.3f}]; {r['n_top']}/{r['n_bottom']} waves, "
              f"{r['n_dropped_missing_H']} dropped missing H, "
              f"{r['n_dropped_small_cell']} in small cells)")
    else:
        raise SystemExit(f"{a.eligible} absent; run control_model.py first "
                         "(the R4 estimands need the unfiltered eligible table)")

    cases = pd.read_csv(a.cases)
    rng = np.random.default_rng(a.seed + 1)
    for min_pc, tag in ((1, "primary"), (3, "sensitivity")):
        r = paired_contrast(cases, rng=rng, n_boot=a.n_boot, min_per_class=min_pc)
        r["note"] = tag
        rows.append(r)
        print(f"paired (min {min_pc}/class, {tag}): {r['diff']:+.3f} "
              f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] over {r['n_waves']} mixed waves "
              f"(season sens [{r['sens_season_lo']:+.3f}, {r['sens_season_hi']:+.3f}])")
    r = between_contrast(cases, waves, rng=np.random.default_rng(a.seed + 2),
                         n_boot=a.n_boot)
    rows.append(r)
    print(f"between-only (cell-standardized): {r['diff']:+.3f} "
          f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] "
          f"({r['n_active_only']} active-only vs {r['n_quiet_only']} quiet-only)")

    out = os.path.join(a.outdir, "wave_estimands.csv")
    for r in rows:
        # the count that actually RAN. Recording the module default here meant a
        # --n-boot override was silently mislabelled (round 8).
        r["n_boot"] = a.n_boot
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6f")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
