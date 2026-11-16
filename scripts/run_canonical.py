#!/usr/bin/env python
"""One validated sequence for the canonical record (implementation fold).

The canonical outputs were previously produced by four manually ordered drivers, so a
stale or skipped step could leave a mixed-generation deposit. This orchestrator runs
the full sequence, the R2 sensitivity variants, the figure nulls (cache-validated),
the registry merge, and the manuscript checker, stopping at the first failure. Full
atomic directory promotion is recorded in REPAIR_SPEC.md as deferred; this sequence
plus the manifest-validated caches and the registry's duplicate-key gate are the
enforced part.

    .venv/bin/python scripts/run_canonical.py [--skip-sensitivities] [--allow-untagged]

A rebuild whose manuscript has not been revised yet, made in a clone of the repository so
that the deposit of the submitted paper stays as it is, ends before the promotion:

    python scripts/run_canonical.py --figures DIR --no-promote --record run_record.json

That writes every figure to DIR, leaves the manuscript's figure folder and the PDFs alone,
does not run the manuscript checker, which compares the manuscript with the registry and
would fail by design until the text is revised, and records every step with its exit code
and its minutes.
"""

import argparse
import os
import subprocess
import sys
import time

import cloud_inputs

PY_ = sys.executable
DL = os.path.expanduser("~/Downloads")
RECORD = dict(path=None, data=None)


def _write_record():
    if RECORD["path"]:
        import json
        with open(RECORD["path"], "w") as f:
            json.dump(RECORD["data"], f, indent=1)


def run(label, cmd):
    t0 = time.time()
    print(f"\n=== {label}: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if RECORD["data"] is not None:
        RECORD["data"]["steps"].append(dict(label=label, arguments=cmd[1:], exit=r.returncode,
                                            minutes=round((time.time() - t0) / 60.0, 2)))
        _write_record()
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
    ap.add_argument("--figures", default=DL, help="folder for every figure of the run")
    ap.add_argument("--no-promote", action="store_true",
                    help="end after the figures, without promoting them into the "
                         "manuscript and without the manuscript checker")
    ap.add_argument("--record", default=None,
                    help="write every step with its exit code and minutes to this file")
    a = ap.parse_args()
    fig = a.figures
    os.makedirs(fig, exist_ok=True)
    if a.record:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        RECORD["path"] = a.record
        RECORD["data"] = dict(commit=commit, python=sys.version.split()[0],
                              started=time.strftime("%Y-%m-%d %H:%M:%S"),
                              inputs={k: cloud_inputs.sha256(cloud_inputs.path_of(k))
                                      for k in ("class_catalog", "pooled_catalog",
                                                "observed_slots", "tracking_file")},
                              promoted=not a.no_promote, steps=[])

    run("deposit", [PY_, "scripts/build_deposit.py"])
    if not a.skip_sensitivities:
        for name, extra in (
                ("sens_delta0", ["--delta-hpa", "0"]),
                ("sens_delta50", ["--delta-hpa", "50"]),
                ("sens_static", ["--footprint", "static"]),
                # the deduplication tie-break is arbitrary and 614 of 2,972 merged
                # components carry a tie, so the opposite rule is a variant run like
                # the terrain ones rather than an assumption
                ("sens_tierule", ["--tie-rule", "largest"])):
            run(name, [PY_, "scripts/build_deposit.py", "--tiers", "pooled",
                       "--outdir", f"deposit/{name}"] + extra)
        # the classes are counted from the detections of one satellite position, and
        # the catalog that pools every position is kept as a sensitivity: the same
        # pooled-tier rebuild on the same eligible troughs, with the counts from it
        run("sens_pooled", [PY_, "scripts/build_deposit.py", "--tiers", "pooled",
                            "--outdir", "deposit/sens_pooled",
                            "--csct", cloud_inputs.POOLED_CATALOG])
    # The cache is the ELIGIBLE cohort and is NOT the published design table. Pointing
    # --cache at deposit/control_model_design.csv (which control_model.py now writes
    # itself, post-filter) would make the guard compare that file against a cohort it is
    # a strict subset of, and every deposited trough outside it would read as missing.
    run("control_model", [PY_, "scripts/control_model.py",
                          "--cache", "deposit/control_model_design_eligible.csv"])
    # the common-sample within-between decomposition; consumes the design
    # table control_model just wrote, so it must follow that step
    run("within_between", [PY_, "scripts/within_between_model.py"])
    run("wave_estimands", [PY_, "scripts/wave_estimands.py"])
    run("response_sens", [PY_, "scripts/sensitivity_response.py"])
    # feeds the registry's yearstrat_* rows, which the abstract and section 2d quote.
    # It sat OUTSIDE this sequence, so a canonical rerun regenerated everything around
    # it and merged its previous-generation CSV: the mixed-generation state this
    # orchestrator exists to prevent (found while folding an earlier revision).
    run("yearstrat_sens", [PY_, "scripts/sensitivity_yearstrat.py"])
    # reads the canonical and tie-rule deposits and writes the comparison; skipped
    # cleanly when --skip-sensitivities meant the variant was never built
    if not a.skip_sensitivities:
        run("tierule_sens", [PY_, "scripts/sensitivity_tierule.py"])
        run("catalog_sens", [PY_, "scripts/sensitivity_tierule.py",
                             "--alt", "deposit/sens_pooled/canonical_numbers.csv",
                             "--out", "catalog_sensitivity.csv",
                             "--label", "catalog of every position pooled"])
    # the class contrasts split exactly among years, cells, waves and passages, a
    # consumer of the tables above with its own random streams, so no number above
    # moves. With the pooled catalog's build present, the same split is made under
    # classes counted from it on the same troughs.
    split = [PY_, "scripts/scale_split.py", "--deposit", "deposit", "--outdir", "deposit",
             "--csct", cloud_inputs.CLASS_CATALOG]
    if not a.skip_sensitivities:
        split += ["--alt-catalog", cloud_inputs.POOLED_CATALOG]
    run("scale_split", split)
    # three small consumers a reviewer asked for, each with its own output and no shared
    # random stream: the counts before and after deduplication, the moist-tail result at
    # other percentiles, and the cross-reanalysis alignment on the regional fields. The
    # fourth, an interval for the centroid difference of the first-detection figure, is
    # computed by that figure's own step below.
    run("dedup_counts", [PY_, "scripts/dedup_counts.py"])
    run("exceedance_sens", [PY_, "scripts/sensitivity_exceedance.py"])
    run("alignment_region", [PY_, "scripts/check_track_alignment_region.py"])
    run("fig2_null", [PY_, "scripts/fig_wave_following.py",
                      "--out", f"{fig}/aew_wave_following_pooled.png"])
    run("fig3_null", [PY_, "scripts/fig_ct_wave_following.py",
                      "--out", f"{fig}/aew_ct_wave_following_pooled.png"])
    run("leadlag", [PY_, "scripts/fig_leadlag.py",
                    "--out", f"{fig}/aew_leadlag_pooled.png"])
    run("registry", [PY_, "tools/build_registry.py"])
    # every estimand-bearing manuscript figure is regenerated HERE, as a canonical
    # consumer of the tables just written, so an image can never lag the numbers
    # (F5/F6 were found stale after the repair rerun). F1 and S1
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
        run(label, [PY_, script, "--out", f"{fig}/{out}"])
    # PROMOTE the regenerated images into the manuscript's figure directory before
    # checking. The scripts above write to a delivery directory and the REPO copy is
    # what the PDF embeds, so
    # without this the sequence could finish green over stale repo figures: exactly the
    # gap was found still open on 2026-07-25 after an earlier change
    # moved figure regeneration into this driver. The builders own copy_figure(), so
    # running them here is the promotion.
    if a.no_promote:
        if RECORD["data"] is not None:
            RECORD["data"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            RECORD["data"]["inputs_at_the_end"] = {
                k: cloud_inputs.sha256(cloud_inputs.path_of(k)) for k in RECORD["data"]["inputs"]}
            _write_record()
        print("\ncanonical sequence complete up to the figures; nothing was promoted "
              "and the manuscript checker was not run")
        return
    run("promote_paper", [PY_, "tools/build_pandoc_paper.py"])
    run("promote_supplement", [PY_, "tools/build_supplement.py"])
    # --strict: a comparison that could not run is a failure here, because this
    # driver just regenerated the sources it is comparing against.
    check = [PY_, "tools/check_manuscript_figures.py", "--strict"]
    if a.allow_untagged:
        check.append("--allow-untagged")
    run("checker", check)
    if RECORD["data"] is not None:
        RECORD["data"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _write_record()
    print("\ncanonical sequence complete")


if __name__ == "__main__":
    main()
