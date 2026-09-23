#!/usr/bin/env python3
"""Capture the checker's refusal of every credited pair, as a failing integration baseline.

STEP 1 OF THE SEQUENCED REPAIR, and deliberately the smallest one. The published accounting
artifact reads fifteen of fifteen. The current checker refuses all fifteen, because the
phase B amendment made `parameters.discrepancy_times` mandatory for any pair whose finished
tracks disagree in ways `divergence_steps` cannot express, and every case artifact predates
that requirement.

WHY CAPTURE THE FAILURE RATHER THAN FIX IT. A regeneration run now would be spent against a
contract that is still being repaired, so it would have to be thrown away and repeated.
Recording the refusal instead gives an explicit failing integration baseline to repair
against, and it makes the disagreement between the published claim and the code a fact in
the repository rather than a paragraph in a handoff.

THE CHECKER IS NOT WEAKENED TO RESTORE THE COUNT, and the eventual repair is not required to
end at fifteen. Regenerating the declarations may leave fewer credits, and that would be the
answer rather than a problem.

    .venv/bin/python3 scripts/capture_closure_refusal.py --out <artifact>
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402
from residue_membership import (derived_discrepancies, read_manifest,  # noqa: E402
                                verified_outputs)

ARTIFACTS = "docs/aewc_v2/artifacts"
PUBLISHED = "residue_membership_2026-09-22.json"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    published = json.load(open(os.path.join(ARTIFACTS, PUBLISHED)))
    # the field was named for one kind while it held every kind; both spellings are read
    credited = published.get("explained_pairs") or published.get("explained_v1_extra_pairs") or []
    outputs, problems = verified_outputs(
        sorted(glob.glob("docs/aewc_v2/evidence/reference_output_*.mat"))
        + ["docs/aewc_v2/evidence/port_output_9a72b9272ad9.mat",
           "docs/aewc_v2/evidence/tracker_port.mat"],
        os.path.join(ARTIFACTS, "reference_logs.json"))
    residuals = json.load(open(os.path.join(ARTIFACTS,
                                            "tracker_oracle_residuals_2026-09-18.json")))
    by_pair = {p["v1_index"]: p for p in residuals["pairs"]}

    rows, cases_by_pair = {}, {}
    for path in sorted(glob.glob(os.path.join(ARTIFACTS, "*_case_*.json"))):
        case = json.load(open(path))
        for index in (case.get("explains") or {}).get("v1_extra_pairs") or ():
            cases_by_pair.setdefault(index, []).append(os.path.basename(path))

    for index in credited:
        files = cases_by_pair.get(index, [])
        declared = None
        for name in files:
            case = json.load(open(os.path.join(ARTIFACTS, name)))
            if (case.get("parameters") or {}).get("discrepancy_times") is not None:
                declared = name
        pair = by_pair.get(index) or {}
        ref = next((o for o in outputs.values() if o.get("kind") == "reference"), None)
        port = next((o for o in outputs.values() if o.get("kind") == "port"), None)
        found = None
        if ref and port and pair.get("port_index") is not None:
            try:
                found = derived_discrepancies(ref["tracks"][index],
                                              port["tracks"][pair["port_index"]])
            except (IndexError, KeyError, TypeError):
                found = None
        rows[str(index)] = {
            "case_files": files,
            "declares_discrepancy_times": declared,
            "derived": {k: len(v) for k, v in (found or {}).items()} if found else None,
            "refused_because": ("no case artifact declares discrepancy_times while the "
                                "finished tracks hold displaced observations")
            if found and found["displaced"] and declared is None else None,
        }

    refused = [k for k, v in rows.items() if v["refused_because"]]
    payload = {
        "generated_by": "scripts/capture_closure_refusal.py",
        "step": "1 of the sequenced repair: contain the false closure claim, change nothing",
        "published_artifact": PUBLISHED,
        "published_artifact_sha256": X.digest(os.path.join(ARTIFACTS, PUBLISHED)),
        "published_claim": {
            "explained_v1_extra_pairs": credited,
            "counts": published["counts"],
        },
        "status_of_that_claim": "STALE AND CURRENTLY UNVERIFIED. It is preserved as "
                               "historical evidence of what was produced, not withdrawn "
                               "from the repository, and it is NOT reproducible under the "
                               "checker as it stands.",
        "pairs_refused_now": sorted(int(k) for k in refused),
        "pairs_refused_count": len(refused),
        "per_pair": rows,
        "output_verification_problems": problems,
        "what_this_does_not_do": [
            "it does not regenerate any case artifact, because the contract is still under "
            "repair and a regeneration now would be repeated",
            "it does not weaken the checker to restore the published count",
            "it does not assume the repaired result should still be fifteen credits",
        ],
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    print(f"published claim: {len(credited)} pairs credited")
    print(f"refused by the checker as it stands: {len(refused)}")
    print(f"  {sorted(int(k) for k in refused)}")
    missing_derivation = [k for k, v in rows.items() if v["derived"] is None]
    if missing_derivation:
        print(f"  could not derive discrepancies for: {sorted(missing_derivation)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
