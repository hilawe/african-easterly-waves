#!/usr/bin/env python3
"""The explained and remaining residue of the identical-input comparison, as explicit
index lists derived from the residual artifact and the worked-case artifacts, never
typed.

A review found the handoff's coverage totals overstated (two of five unmatched tracks
explained where the cases had walked one) and its remaining-work list mixing categories.
This script reads the residual classification and each worked case's declared
membership, checks every claimed index against the category it is claimed for, and
writes the counts beside the lists they come from. A case's membership is what its
artifact declares under `explains`; a case artifact without that block explains
nothing here.

    .venv/bin/python scripts/residue_membership.py \\
        --residuals docs/aewc_v2/artifacts/tracker_oracle_residuals_2026-09-18.json \\
        --cases docs/aewc_v2/artifacts/sahara_case_2026-09-18.json ... --out <json>
"""
import argparse
import hashlib
import json
import os
import sys


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


BINDING_FILES = ("tracker_case.mat", "tracker_port.mat")


def binding_problems(residuals, cases):
    """The reasons a case artifact is NOT a case from the residual artifact's own
    comparison. Track indices are local to a run, so a case must agree with the
    residual artifact on the case identifier, the exported-case digest and the
    port-output digest before any of its claimed indices can count. The reference
    output digests are NOT compared, because instrumented runs with different dump
    schedules legitimately carry different producer metadata over identical tracks.
    A review copied a genuine case artifact, renamed its case and replaced these
    digests, and the first version credited its claims."""
    problems = []
    r_case = residuals.get("case_id")
    r_hashes = residuals.get("input_sha256") or {}
    for name, case in cases.items():
        c_case = case.get("case_id")
        c_hashes = case.get("input_sha256") or {}
        if r_case is None or c_case is None:
            problems.append(f"{name}: a case identifier is missing on one side")
        elif c_case != r_case:
            problems.append(f"{name}: case {c_case!r} is not the residual artifact's {r_case!r}")
        for f in BINDING_FILES:
            if f not in r_hashes or f not in c_hashes:
                problems.append(f"{name}: the digest of {f} is missing on one side")
            elif c_hashes[f] != r_hashes[f]:
                problems.append(f"{name}: {f} differs from the residual artifact's, so the "
                                f"case comes from another export or another port run")
    return problems


def membership(residuals, cases):
    binding = binding_problems(residuals, cases)
    if binding:
        return {"problems": binding}
    unmatched_no_eligible = sorted(u["index"] for u in residuals["v1_unmatched"]
                                   if not u["eligible_counterpart_exists"])
    unmatched_eligible = sorted(u["index"] for u in residuals["v1_unmatched"]
                                if u["eligible_counterpart_exists"])
    by_kind = {}
    for p in residuals["pairs"]:
        by_kind.setdefault(p["extra_kind"], []).append(p["v1_index"])
    for k in by_kind:
        by_kind[k] = sorted(by_kind[k])
    explained_unmatched, explained_pairs, problems, per_case = [], [], [], {}
    for name, case in cases.items():
        ex = case.get("explains") or {}
        per_case[name] = ex
        for i in ex.get("unmatched_v1_tracks", []):
            if i not in unmatched_no_eligible:
                problems.append(f"{name} claims unmatched track {i}, which is not in the "
                                f"no-eligible-counterpart set")
            elif i in explained_unmatched:
                problems.append(f"unmatched track {i} is claimed by more than one case")
            else:
                explained_unmatched.append(i)
        for i in ex.get("v1_extra_pairs", []):
            if i not in by_kind.get("extra_v1_only", []):
                problems.append(f"{name} claims pair {i}, which is not a version-1-extra pair")
            elif i in explained_pairs:
                problems.append(f"pair {i} is claimed by more than one case")
            else:
                explained_pairs.append(i)
    return {"unmatched_v1_no_eligible_counterpart": unmatched_no_eligible,
            "unmatched_v1_with_eligible_counterpart": unmatched_eligible,
            "pairs_by_extra_kind": by_kind,
            "explained_unmatched": sorted(explained_unmatched),
            "remaining_unmatched": sorted(set(unmatched_no_eligible) - set(explained_unmatched)),
            "explained_v1_extra_pairs": sorted(explained_pairs),
            "remaining_v1_extra_pairs": sorted(set(by_kind.get("extra_v1_only", []))
                                               - set(explained_pairs)),
            "counts": {"unmatched_no_eligible": len(unmatched_no_eligible),
                       "unmatched_explained": len(explained_unmatched),
                       "v1_extra_pairs": len(by_kind.get("extra_v1_only", [])),
                       "v1_extra_pairs_explained": len(explained_pairs),
                       "both_sides_extra_pairs": len(by_kind.get("extra_both_sides", []))},
            "per_case": per_case, "problems": problems}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--residuals", required=True)
    ap.add_argument("--cases", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    with open(args.residuals) as fh:
        residuals = json.load(fh)
    cases = {}
    for path in args.cases:
        with open(path) as fh:
            cases[os.path.basename(path)] = json.load(fh)
    result = membership(residuals, cases)
    if result["problems"]:
        print("REFUSED: " + "; ".join(result["problems"]), flush=True)
        return 2
    print(json.dumps(result["counts"]))
    print("remaining unmatched:", result["remaining_unmatched"])
    print("remaining version-1-extra pairs:", result["remaining_v1_extra_pairs"])
    if args.out:
        result.update({"generated_by": "scripts/residue_membership.py",
                       "input_sha256": {os.path.basename(p): _sha256(p)
                                        for p in [args.residuals] + list(args.cases)},
                       "source_sha256": {"scripts/residue_membership.py":
                                         _sha256(os.path.abspath(__file__))}})
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1, sort_keys=True)
        print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
