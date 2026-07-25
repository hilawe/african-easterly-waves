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
    run("response_sens", [PY_, "scripts/sensitivity_response.py"])
    # feeds the registry's yearstrat_* rows, which the abstract and section 2d quote.
    # It sat OUTSIDE this sequence, so a canonical rerun regenerated everything around
    # it and merged its previous-generation CSV: the mixed-generation state this
    # orchestrator exists to prevent (found while folding round 6).
    run("yearstrat_sens", [PY_, "scripts/sensitivity_yearstrat.py"])
    run("fig2_null", [PY_, "scripts/fig_wave_following.py",
                      "--out", f"{DL}/aew_wave_following_pooled.png"])
    run("fig3_null", [PY_, "scripts/fig_ct_wave_following.py",
                      "--out", f"{DL}/aew_ct_wave_following_pooled.png"])
    run("leadlag", [PY_, "scripts/fig_leadlag.py",
                    "--out", f"{DL}/aew_leadlag_pooled.png"])
    run("registry", [PY_, "tools/build_registry.py"])
    # every estimand-bearing manuscript figure is regenerated HERE, as a canonical
    # consumer of the tables just written, so an image can never lag the numbers
    # (the full-access review caught F5/F6 stale after the repair rerun). F1 and S1
    # are frozen validation figures with no estimand content and stay outside.
    for label, script, out in (
            ("fig5_eulerian", "scripts/fig05_eulerian.py", "aew_fig_eulerian.png"),
            ("fig6_supply", "scripts/fig_supply_contrast.py",
             "aew_fig_supply_contrast.png"),
            ("fig7_fingerprint", "scripts/fig_fingerprint.py",
             "aew_fig_fingerprint.png"),
            ("fig8_organization", "scripts/fig_organization_axis.py",
             "aew_fig_organization_axis.png"),
            ("fig9_schematic", "scripts/fig_schematic.py",
             "aew_fig_schematic_v5draft.png"),
            ("figS2_exceedance", "scripts/fig_exceedance.py", "aew_exceedance.png"),
            ("figS3_control", "scripts/fig_control_model.py",
             "aew_fig_control_model.png")):
        run(label, [PY_, script, "--out", f"{DL}/{out}"])
    # PROMOTE the regenerated images into docs/paper/figures before checking. The
    # scripts above write to ~/Downloads and the REPO copy is what the PDF embeds, so
    # without this the sequence could finish green over stale repo figures: exactly the
    # gap the 2026-07-25 full-access review found still open after the previous round
    # moved figure regeneration into this driver. The builders own copy_figure(), so
    # running them here is the promotion.
    run("promote_paper", [PY_, "tools/build_pandoc_paper.py"])
    run("promote_supplement", [PY_, "tools/build_supplement.py"])
    # --strict: a comparison that could not run is a failure here, because this
    # driver just regenerated the sources it is comparing against.
    check = [PY_, "tools/check_manuscript_figures.py", "--strict"]
    if a.allow_untagged:
        check.append("--allow-untagged")
    run("checker", check)
    print("\ncanonical sequence complete")


if __name__ == "__main__":
    main()
