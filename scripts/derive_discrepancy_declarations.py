#!/usr/bin/env python3
"""Add the derived discrepancy declaration to every case artifact that lacks one.

STEP 5 OF THE SEQUENCED REPAIR, the part that needs NO RERUN. The phase B amendment made
`parameters.discrepancy_times` mandatory for any pair whose finished tracks disagree in
ways `divergence_steps` cannot express. Every credited pair has displaced observations and
every case predates the requirement, so the checker refuses all fifteen. The declaration is
derived from the two pinned finished outputs each case already names, by the same function
the checker uses, so it cannot disagree with the derivation.

THIS SEPARATES DERIVABLE DECLARATIONS FROM EVIDENCE THAT WAS NEVER RECORDED. A declaration
is derivable when the case's pinned outputs are retained, which they are for all fifteen.
Control RECEIPTS for the thirteen injection cases already exist as requested_times,
applied_at and injections_applied per run. The two operation cases get theirs from
scripts/write_operation_cases.py. Nothing here requires rerunning an experiment.

The published closure count is NOT a target. Whatever the regenerated accounting says is
the answer.

    .venv/bin/python3 scripts/derive_discrepancy_declarations.py [--dry-run]
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402
from residue_membership import derived_discrepancies, examined_explains  # noqa: E402
from write_operation_cases import pinned_tracks  # noqa: E402

ARTIFACTS = "docs/aewc_v2/artifacts"
EVIDENCE = "docs/aewc_v2/evidence"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    residuals = json.load(open(os.path.join(ARTIFACTS, "tracker_oracle_residuals_2026-09-18.json")))
    port_of = {p["v1_index"]: p["port_index"] for p in residuals["pairs"]}
    by_digest = {X.digest(p): p for p in glob.glob(os.path.join(EVIDENCE, "*.mat"))}
    cache = {}
    changed, skipped = [], []
    for path in sorted(glob.glob(os.path.join(ARTIFACTS, "*_case_*.json"))):
        case = json.load(open(path))
        params = case.setdefault("parameters", {})
        # THE PAIRS A CASE EXAMINED, BY THE CHECKER'S OWN SCOPING, not the pairs it claims.
        # A first version read `explains`, and the checker then refused two negative-result
        # cases that examined pairs 30 and 63 without claiming them, because a discrepancy
        # declaration describes the pair rather than the claim. Reusing `examined_explains`
        # is what stops this script and the checker from disagreeing about scope again.
        examined, _ = examined_explains(os.path.basename(path), case, residuals)
        pairs = list(examined.get("v1_extra_pairs") or ())
        if not pairs or "discrepancy_times" in params or "discrepancy_times_by_pair" in params:
            skipped.append((os.path.basename(path),
                            "already declared" if pairs else "examines no pair"))
            continue
        digests = case["input_sha256"]
        ref = by_digest.get(digests.get("tracker_octave_instrumented.mat"))
        port = by_digest.get(digests.get("tracker_port.mat"))
        if not ref or not port or any(pair not in port_of for pair in pairs):
            skipped.append((os.path.basename(path), "pinned outputs not retained"))
            continue
        for key in (ref, port):
            cache.setdefault(key, pinned_tracks(key))
        declared = {}
        for pair in pairs:
            found = derived_discrepancies(cache[ref][pair], cache[port][port_of[pair]])
            declared[str(pair)] = {k: [float(t) for t in v] for k, v in found.items()}
        # ONE PAIR GETS THE SINGLE FORM, SEVERAL GET THE PER-PAIR FORM. Three cases here claim
        # two or four pairs each, eight of the fifteen, and one set of three lists cannot
        # describe two pairs; a first version assumed it could and would have left them
        # refused.
        if len(pairs) == 1:
            params["discrepancy_times"] = declared[str(pairs[0])]
        else:
            params["discrepancy_times_by_pair"] = declared
        params["discrepancy_times_derived_by"] = "scripts/derive_discrepancy_declarations.py"
        changed.append((os.path.basename(path),
                        {p: {k: len(v) for k, v in d.items()} for p, d in declared.items()}))
        if not args.dry_run:
            json.dump(case, open(path, "w"), indent=1, sort_keys=True)
    print(f"{'would update' if args.dry_run else 'updated'} {len(changed)} cases")
    for name, counts in changed:
        print(f"  {name}: {counts}")
    for name, why in skipped:
        print(f"  skipped {name}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
