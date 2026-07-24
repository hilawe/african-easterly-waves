#!/usr/bin/env python
"""One validated sequence for the canonical record (implementation-review fold).

The canonical outputs were previously produced by four manually ordered drivers, so a
stale or skipped step could leave a mixed-generation deposit. This orchestrator runs
the full sequence, the R2 sensitivity variants, the figure nulls (cache-validated),
the registry merge, and the manuscript checker, stopping at the first failure. Full
atomic directory promotion is recorded in REPAIR_SPEC.md as deferred; this sequence
plus the manifest-validated caches and the registry's duplicate-key gate are the
enforced part.

    .venv/bin/python scripts/run_canonical.py [--skip-sensitivities] [--allow-untagged]
"""

import argparse
import os
import subprocess
import sys
import time

PY_ = sys.executable
DL = os.path.expanduser("~/Downloads")


def run(label, cmd):
    t0 = time.time()
    print(f"\n=== {label}: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"FAILED at {label} (exit {r.returncode}); "
                         "nothing after this step ran")
    print(f"=== {label} done ({(time.time() - t0) / 60:.1f} min)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sensitivities", action="store_true",
                    help="skip the delta-0/50 and static-footprint variant runs")
    ap.add_argument("--allow-untagged", action="store_true",
                    help="pass the lint transition flag to the checker")
    a = ap.parse_args()

    run("deposit", [PY_, "scripts/build_deposit.py"])
    if not a.skip_sensitivities:
        for name, extra in (
                ("sens_delta0", ["--delta-hpa", "0"]),
                ("sens_delta50", ["--delta-hpa", "50"]),
                ("sens_static", ["--footprint", "static"])):
            run(name, [PY_, "scripts/build_deposit.py", "--tiers", "pooled",
                       "--outdir", f"deposit/{name}"] + extra)
    run("control_model", [PY_, "scripts/control_model.py",
                          "--cache", "deposit/control_model_design.csv"])
    run("wave_estimands", [PY_, "scripts/wave_estimands.py"])
    run("fig2_null", [PY_, "scripts/fig_wave_following.py",
                      "--out", f"{DL}/aew_wave_following_pooled.png"])
    run("fig3_null", [PY_, "scripts/fig_ct_wave_following.py",
                      "--out", f"{DL}/aew_ct_wave_following_pooled.png"])
    run("leadlag", [PY_, "scripts/fig_leadlag.py",
                    "--out", f"{DL}/aew_leadlag_pooled.png"])
    run("registry", [PY_, "tools/build_registry.py"])
    check = [PY_, "tools/check_manuscript_figures.py"]
    if a.allow_untagged:
        check.append("--allow-untagged")
    run("checker", check)
    print("\ncanonical sequence complete")


if __name__ == "__main__":
    main()
