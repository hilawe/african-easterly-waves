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
import math
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


REQUIRED_RUNS = ("baseline", "intervention", "control", "shape_control")


def experiment_problems(intervention, divergence_step, control_step=None,
                        control_step_problems=(), injected_axes=None, shape_axes=None):
    """Why a recorded four-run experiment is not one, checked the SAME WAY wherever it is
    read. This is the shared contract: the trace calls it before it will state anything
    about an intervention, and the membership checker calls it before it will credit one.

    A review is why it is shared rather than written twice. The checker had its own
    weaker copy: it required a receipt to agree with the time its own replay requested,
    and never asked whether that time was the case's DIVERGENCE STEP, so moving an
    intervention ten hours away kept all four of its credits while the trace refused the
    same artifact. It also ignored the recorded control-step problems and let the whole
    shape-control record disappear. The expected divergence time is passed in explicitly,
    never read from a module global."""
    problems = []
    if not intervention or not intervention.get("baseline"):
        return ["the artifact records no intervention"]
    absent = [r for r in REQUIRED_RUNS if not intervention.get(r)]
    if absent:
        problems.append("the experiment records no " + " or ".join(absent) + " replay")
    problems.extend(control_step_problems or [])

    def finite(x):
        try:
            return math.isfinite(float(x))
        except (TypeError, ValueError):
            return False

    for label in ("intervention", "control", "shape_control"):
        run = intervention.get(label)
        if not run:
            continue
        applied, at, when = (run.get("injections_applied"), run.get("applied_at") or [],
                             run.get("injected_at"))
        if applied != 1 or len(at) != 1:
            problems.append(f"the {label} replay applied {applied} of the 1 injection it "
                            f"requested")
            continue
        if not finite(at[0]) or not finite(when) or abs(float(at[0]) - float(when)) > 1e-6:
            problems.append(f"the {label} replay recorded an application at {at} for an "
                            f"injection requested at {when}")
            continue
        if label in ("intervention", "shape_control"):
            if not finite(divergence_step) \
                    or abs(float(when) - float(divergence_step)) > 1e-6:
                problems.append(f"the {label} replay injected at {when} and the case's "
                                f"divergence step is {divergence_step}")
        else:
            if finite(divergence_step) and abs(float(when) - float(divergence_step)) <= 1e-6:
                problems.append(f"the time control injected at the divergence step {when}")
            if control_step is not None and finite(control_step) \
                    and abs(float(when) - float(control_step)) > 1e-6:
                problems.append(f"the time control injected at {when} and the case declares "
                                f"its control step as {control_step}")
    if injected_axes and shape_axes \
            and all(a.get("lon") == b.get("lon") and a.get("lat") == b.get("lat")
                    for a, b in zip(injected_axes, shape_axes)):
        problems.append("the shape control's vertices are the intervention's own")
    return problems


def outcome_for(intervention, index):
    """What a case's intervention DID for one index, read from its own records.

    "reproduced" means the intervention replay reproduces that reference track exactly
    while neither the untouched replay nor a control does. "track_survival" is the weaker
    outcome the Sahara case has: nothing reproduces the track, and the intervention's
    count of finished tracks in the western box differs from every control's. Anything
    else is no outcome at all.

    A REVIEW READING THIS COLD FOUND WHY THE DISTINCTION MATTERS. The first version of
    this gate required only that the three replays MENTION the index, so a case whose
    intervention reproduced nothing was credited beside cases that reproduced their tracks
    exactly, and the weaker case was presented as the stronger one. Requiring an examined
    index was a better stand-in for an outcome, not an outcome."""
    labels = [l for l in REQUIRED_RUNS if intervention.get(l)]
    exact, west = {}, {}
    for label in labels:
        run = intervention[label]
        for rec in run.get("reference_tracks") or []:
            if rec.get("reference_index") == index:
                exact[label] = bool(rec.get("reproduced_exactly"))
        tracks = run.get("finished_tracks_in_the_western_box")
        west[label] = None if tracks is None else len(tracks)
    if set(exact) != set(labels):
        missing = sorted(set(labels) - set(exact))
        return None, f"its replays {missing} record no outcome for it"
    # THE NULL CONTROL IS THE ONE IN TIME, the same injection at a step the two sides
    # already agree on, and it is the one that must not reproduce the track. The SHAPE
    # control is a second intervention rather than a null: it injects an axis where the
    # port had none, with the vertices moved off the crossing, and on three of the four
    # cases it reproduces the tracks too. That is a refinement of the claim, since it
    # shows the port's loss is the ABSENCE of an axis rather than the absence of that
    # line, and treating it as a disqualifier would refuse a case for being better
    # understood.
    nulls = [l for l in labels if l == "control"]
    if exact["intervention"] and not exact["baseline"] \
            and not any(exact[l] for l in nulls):
        return "reproduced", None
    if exact["baseline"] or any(exact[l] for l in nulls):
        return None, ("the untouched replay reproduces it" if exact["baseline"]
                      else "the time control reproduces it as well as the intervention")
    if west.get("intervention") is None or west["baseline"] is None:
        return None, "no finished-track measurement to fall back on"
    others = [west[l] for l in ["baseline"] + nulls if west.get(l) is not None]
    # SURVIVAL MEANS MORE TRACKS, not a different number of them. A review set a genuine
    # artifact's baseline and control counts to two and its intervention's to zero, and
    # the first version credited that LOSS under a label that says the opposite.
    if len(others) == len(nulls) + 1 and all(west["intervention"] > o for o in others):
        return "track_survival", None
    if len(others) == len(nulls) + 1 and all(west["intervention"] < o for o in others):
        return None, ("its intervention REMOVES finished tracks from the western box, "
                      "which is not the survival outcome")
    return None, "neither reproduces it nor adds finished tracks to its western box"


def walk_problems(name, case, claimed):
    """Why a case may not be credited for an index it claims, and how it is credited when
    it may. Returns (problems, outcomes).

    THE EXPERIMENT IS VALIDATED FIRST, under the same contract the trace applies, with the
    expected divergence and control steps taken from the case's own declared parameters."""
    intervention = case.get("intervention") or {}
    if not intervention.get("baseline"):
        return ([f"{name} claims {sorted(claimed)} and records no intervention, so nothing "
                 f"in it examined those indices"], {})
    parameters = case.get("parameters") or {}
    broken = experiment_problems(
        intervention, parameters.get("divergence_step"), parameters.get("control_step"),
        intervention.get("control_step_problems"), intervention.get("injected_axes"),
        (intervention.get("shape_control") or {}).get("axes"))
    if broken:
        return ([f"{name} claims {sorted(claimed)} and {w}" for w in broken], {})
    problems, outcomes = [], {}
    for index in sorted(claimed):
        outcome, why = outcome_for(intervention, index)
        if outcome is None:
            problems.append(f"{name} claims {index}, which {why}")
        else:
            outcomes[index] = outcome
    return problems, outcomes


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
    by_outcome = {}
    for name, case in cases.items():
        ex = case.get("explains") or {}
        per_case[name] = ex
        claimed = list(ex.get("unmatched_v1_tracks", [])) + list(ex.get("v1_extra_pairs", []))
        if claimed:
            why, outcomes = walk_problems(name, case, claimed)
            problems.extend(why)
            by_outcome.update(outcomes)
        for i in ex.get("unmatched_v1_tracks", []):
            if i not in unmatched_no_eligible:
                problems.append(f"{name} claims unmatched track {i}, which is not in the "
                                f"no-eligible-counterpart set")
            elif i in explained_unmatched:
                problems.append(f"unmatched track {i} is claimed by more than one case")
            elif i in by_outcome:
                explained_unmatched.append(i)
        for i in ex.get("v1_extra_pairs", []):
            if i not in by_kind.get("extra_v1_only", []):
                problems.append(f"{name} claims pair {i}, which is not a version-1-extra pair")
            elif i in explained_pairs:
                problems.append(f"pair {i} is claimed by more than one case")
            elif i in by_outcome:
                # A CREDIT IS GRANTED ONLY WHERE AN OUTCOME WAS READ. The lists used to be
                # built from the declaration alone, so an index whose experiment had just
                # been refused above still appeared among the explained.
                explained_pairs.append(i)
    def split(indices):
        return {"reproduced": sorted(i for i in indices
                                     if by_outcome.get(i) == "reproduced"),
                "track_survival": sorted(i for i in indices
                                         if by_outcome.get(i) == "track_survival")}

    return {"unmatched_v1_no_eligible_counterpart": unmatched_no_eligible,
            # THE OUTCOME EACH CREDIT RESTS ON, beside the credit, so a case that changes
            # a track count is never read as one that reproduced a track
            "explained_by_outcome": {"unmatched": split(explained_unmatched),
                                     "v1_extra_pairs": split(explained_pairs)},
            "unmatched_v1_with_eligible_counterpart": unmatched_eligible,
            "pairs_by_extra_kind": by_kind,
            "explained_unmatched": sorted(explained_unmatched),
            "remaining_unmatched": sorted(set(unmatched_no_eligible) - set(explained_unmatched)),
            "explained_v1_extra_pairs": sorted(explained_pairs),
            "remaining_v1_extra_pairs": sorted(set(by_kind.get("extra_v1_only", []))
                                               - set(explained_pairs)),
            "counts": {"unmatched_no_eligible": len(unmatched_no_eligible),
                       "unmatched_explained": len(explained_unmatched),
                       "unmatched_explained_by_exact_reproduction":
                           len([i for i in explained_unmatched
                                if by_outcome.get(i) == "reproduced"]),
                       "v1_extra_pairs_explained_by_exact_reproduction":
                           len([i for i in explained_pairs
                                if by_outcome.get(i) == "reproduced"]),
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
