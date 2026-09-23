#!/usr/bin/env python3
"""Write pairs 30 and 63 as case artifacts the accounting can read.

WHY THIS IS A TRANSCRIPTION AND NOT A RERUN. Both results are already established and
retained: complete finished trajectories for every run, beside the digests of the case file,
the reference output, the producing script and the repository head. What was missing was an
artifact in the shape `residue_membership.py` reads, which until 2026-09-22 could only
express an INJECTION. This reads the retained runs and writes that artifact.

NOTHING HERE RECOMPUTES AN OUTCOME. The accounting recomputes every `reproduced_exactly`
flag from the recorded tracks against its own pinned reference and refuses the artifact if a
flag disagrees, so a flag written here that were wrong would be caught rather than believed.
The tracks are copied verbatim from the retained runs.

    .venv/bin/python3 scripts/write_operation_cases.py
"""

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402
from residue_membership import (_same_track, derived_discrepancies,  # noqa: E402
                                verified_exchange)

ARTIFACTS = "docs/aewc_v2/artifacts"
EVIDENCE = "docs/aewc_v2/evidence"
SCRIPTS = os.path.dirname(os.path.abspath(__file__))

# Which retained run plays which part in the contract's four-run shape. The retained files
# use the diagnostics' own names; the contract's names are on the left.
CASES = {
    "pair63_reordering": {
        "runs": "pair63_reorder_runs_2026-09-22.json",
        "out": "pair63_reordering_case_2026-09-22.json",
        "operation": "reordering",
        "pair": 63,
        "source_case": "south_atlantic_start_case_2026-09-21.json",
        "diagnostic": "pair63_finished_track.py",
        "result_flag": "exact_reproduction_by_intervention_alone",
        "roles": {"baseline": "baseline", "intervention": "intervention",
                  "representation_control": "representation",
                  "negative_control": "negative"},
        "changed_at": 33030.5,
        "changed_at_basis": (
            "the one step at which version 1's finished track holds an observation the "
            "port's does not, read from the residual record; the merge trace then located "
            "the pair whose order decides it"),
        "fields": {"moved_from": 65, "moved_to": 58, "positions_unchanged": True,
                   "population_before": 77, "population_after": 77},
        # RECEIPTS, derived from the frozen diagnostic rather than recorded by it. The
        # representation run reinserted the feature's axis AT ITS OWN INDEX; the negative
        # run moved the first axis after the pair, index 66, to the front. Both are what
        # scripts/pair63_finished_track.py does in its "identity" and "control" modes, and
        # that script's digest is in the retained runs' identity block.
        "receipts": {
            "representation_control": {"identity": True, "population_before": 77,
                                       "population_after": 77, "moved_from": 65,
                                       "moved_to": 65},
            "negative_control": {"population_before": 77, "population_after": 77,
                                 "moved_from": 66, "moved_to": 0}},
        "note": None,
    },
    "pair30_removal": {
        "runs": "pair30_removal_runs_2026-09-22.json",
        "out": "pair30_removal_case_2026-09-22.json",
        "operation": "removal",
        "pair": 30,
        "source_case": "peru_case_2026-09-20.json",
        "diagnostic": "pair30_merge_input.py",
        "result_flag": "exact_reproduction_with_controls_failing",
        "roles": {"baseline": "baseline", "intervention": "narrowed",
                  "representation_control": "control", "negative_control": "narrow ctrl"},
        "changed_at": 33026.5,
        "changed_at_basis": (
            "leave-one-out over all 75 axes at this step, the only one whose removal makes "
            "the port's extra candidate disappear; the step itself came from following the "
            "port's pruned fragment, and is NOT derivable from the residual record, whose "
            "port-extra times for this pair are empty"),
        "fields": {"removed": {"lat": -3.71, "lon": -87.37},
                   "identified_by": "leave-one-out over all 75 axes at this step",
                   "population_before": 75, "population_after": 74},
        # THE INDICES ARE RE-DERIVED, not remembered. The diagnostic recorded the culprit
        # and spare axes as CENTROIDS; the checker bounds indices to the population, so the
        # index of each centroid is located in the port's own axis list at that step,
        # deterministically, and the centroid must match to 0.01 degrees or this refuses.
        "locate": {"removed_index": (-3.71, -87.37), "negative_removed_index": (-4.28, -74.26)},
        "receipts": {
            "representation_control": {"identity": True, "population_before": 75,
                                       "population_after": 75},
            "negative_control": {"population_before": 75, "population_after": 74}},
        "note": None,
    },
}


def as_track(run_track):
    """A retained trajectory in the key names the accounting's track reader uses."""
    return {"time": list(run_track["time"]), "lat": list(run_track["meanlat"]),
            "lon": list(run_track["meanlon"])}


def in_western_box(tracks, west, life):
    """The finished tracks in the western box during the feature's life.

    THE SAME PREDICATE THE ACCOUNTING RECOMPUTES, written here to produce a list it will
    agree with. It recomputes this from the recorded tracks and refuses the artifact when
    the recorded list disagrees, so a stub of zero, which a first version wrote, is caught
    rather than believed."""
    if not west or not life:
        return []
    low, high = float(life[0]), float(life[1])
    out = []
    for tr in tracks:
        for t, la, lo in zip(tr["time"], tr["lat"], tr["lon"]):
            lat_range, lon_range = west["lat"], west["lon"]
            if (float(lat_range[0]) <= float(la) <= float(lat_range[1])
                    and float(lon_range[0]) <= float(lo) <= float(lon_range[1])
                    and low <= float(t) <= high):
                out.append({"steps": len(tr["time"])})
                break
    return out


def locate_axis(step, centroid):
    """The index of the port's axis whose centroid matches, at this step, or refuse."""
    import numpy as np
    from scipy.io import loadmat
    from validation_phase_b import axes_at
    raw = loadmat(os.path.join(EVIDENCE, "tracker_case.mat"))
    case = {k: np.asarray(v) for k, v in raw.items() if not k.startswith("__")}
    axes = axes_at(case, step)
    gaps = [abs(float(np.mean(a)) - centroid[0]) + abs(float(np.mean(b)) - centroid[1])
            for a, b in axes]
    k = int(min(range(len(gaps)), key=gaps.__getitem__))
    if gaps[k] > 0.01:
        raise SystemExit(f"no axis at {step} has centroid {centroid}; nearest is "
                         f"{gaps[k]:.3f} degrees off, so the index cannot be trusted")
    return k, len(axes)


def pinned_tracks(path):
    from scipy.io import loadmat
    import numpy as np
    m = loadmat(path)
    n = int(np.asarray(m["n"]).ravel()[0])
    return [{"time": np.asarray(m[f"time{i}"], float).ravel().tolist(),
             "lat": np.asarray(m[f"lat{i}"], float).ravel().tolist(),
             "lon": np.asarray(m[f"lon{i}"], float).ravel().tolist()} for i in range(n)]


def declared_discrepancies(source, pair):
    """The case's discrepancy declaration, DERIVED from the pinned outputs it declares.

    The checker refuses every pre-amendment artifact because none declares this and every
    one has displaced observations. The declaration is read from the same two finished
    outputs the checker itself reads, by the same function, so it cannot disagree with the
    derivation except by the artifacts being wrong, which is what the checker is for."""
    residuals = json.load(open(os.path.join(ARTIFACTS,
                                            "tracker_oracle_residuals_2026-09-18.json")))
    port_index = next(p["port_index"] for p in residuals["pairs"] if p["v1_index"] == pair)
    digests = source["input_sha256"]
    ref = next(p for p in glob.glob(os.path.join(EVIDENCE, "*.mat"))
               if X.digest(p) == digests["tracker_octave_instrumented.mat"])
    port = next(p for p in glob.glob(os.path.join(EVIDENCE, "*.mat"))
                if X.digest(p) == digests["tracker_port.mat"])
    found = derived_discrepancies(pinned_tracks(ref)[pair], pinned_tracks(port)[port_index])
    return {k: [float(t) for t in v] for k, v in found.items()}


def retained_output(digest):
    """The retained evidence file with this digest, or None."""
    return next((p for p in glob.glob(os.path.join(EVIDENCE, "*.mat"))
                 if X.digest(p) == digest), None)


def same_indexed_tracks(a, b):
    """Equal-length collections whose tracks agree index for index, by the checker's own
    exact per-track equality. IN FINAL-INDEX ORDER, because the checker judges a replay
    against `ref["tracks"][i]` and the identity names an index. A first version compared
    the two as sorted multisets, and a confirmation round swapped tracks 30 and 63 in the
    judged file and was accepted, with index 63 then naming a different track."""
    return len(a) == len(b) and all(_same_track(x, y) for x, y in zip(a, b))


def identity_problems(spec, retained, source):
    """Why the retained runs are NOT the experiment this case transcribes.

    A review mutated the retained identity block, a case digest of sixty-four zeroes, a
    reference digest of ones, the reference index of another pair, and a diagnostic digest
    of twos, left every trajectory untouched, and both this producer and the accounting
    credited pair 63 through `main(argv)`. The producer had checked only that the named runs
    existed and then COPIED the contradiction into `evidence_binding`, where nothing reads
    it. Correct arithmetic against a reference does not establish that the recorded
    experiment belongs to this input, this pair or this step, so those are checked here,
    against the case being superseded and the specification, before anything is copied.

    The reference output is accepted by CONTENT, not by digest alone. Four retained reference
    files hold identical finished tracks under different digests, and the runs were judged
    against one of them while the source case pins another, so the file the digest names
    must exist and its finished tracks must equal the pinned reference's exactly."""
    identity = retained["identity"]
    pair = spec["pair"]
    problems = []
    pinned = source["input_sha256"]
    # THE CASE INPUT IS RESOLVED THROUGH THE CHECKER'S OWN EXCHANGE TABLE, not compared as
    # a digest. Sixteen cases pin the case input under its earlier serialization and the
    # retained runs name the rebuilt file, which the manifest records as a verified
    # reserialization of it with the same recomputed content identity. A digest comparison
    # refused pair 30 on that alone the first time this check ran. Both digests must be
    # ones the manifest resolves, and they must resolve to the same content.
    exchange, why = verified_exchange()
    problems.extend(why)
    runs_input = exchange.get(str(identity.get("case_sha256")))
    case_input = exchange.get(str(pinned.get("tracker_case.mat")))
    if runs_input is None:
        problems.append(f"the retained runs were produced from case input "
                        f"{str(identity.get('case_sha256'))[:12]}, which is not a retained "
                        f"exchange input or a verified reserialization of one")
    elif case_input is None:
        problems.append(f"the case being superseded pins case input "
                        f"{str(pinned.get('tracker_case.mat'))[:12]}, which is not a "
                        f"retained exchange input or a verified reserialization of one")
    elif runs_input["identity"] != case_input["identity"]:
        problems.append(f"the retained runs were produced from case input "
                        f"{str(identity.get('case_sha256'))[:12]} and the case being "
                        f"superseded pins {str(pinned.get('tracker_case.mat'))[:12]}, and "
                        f"the two resolve to different content")
    if identity.get("reference_track_index") != pair:
        problems.append(f"the retained runs were judged against reference track "
                        f"{identity.get('reference_track_index')} and this case is pair {pair}")
    # THE STEP IS REQUIRED, not checked when present. A confirmation round deleted it and
    # the producer supplied its own `changed_at` under a comment saying the step had been
    # checked, which established nothing about the retained experiment's step.
    try:
        step = float(identity["step"])
        if step != step:
            raise ValueError("nan")
    except (KeyError, TypeError, ValueError):
        problems.append("the retained runs record no numeric intervention step, so nothing "
                        "says the experiment acted where this case declares")
    else:
        if step != float(spec["changed_at"]):
            problems.append(f"the retained runs intervene at {identity['step']} and this "
                            f"case declares {spec['changed_at']}")
    # EVERY FIELD THE CASE COPIES FROM THE IDENTITY IS JUDGED HERE, or it is not copied.
    # Two no-folder reviews found `comparison`, `case_file`, `script` and the result flag
    # copied or ignored after four rounds had repaired the same class one key at a time.
    # `git_head` and the free-text intervention descriptions are not copied and not read.
    if identity.get("case_file") != "tracker_case.mat":
        problems.append(f"the retained runs name their case input "
                        f"{identity.get('case_file')!r} rather than tracker_case.mat")
    if identity.get("comparison") != X.COMPARISON:
        problems.append("the retained runs do not record the exact whole-array comparison "
                        "this transcription requires")
    if identity.get("script") != spec["diagnostic"]:
        problems.append(f"the retained runs were produced by {identity.get('script')!r} "
                        f"and this case transcribes {spec['diagnostic']}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(identity.get("script_sha256"))):
        problems.append("the retained runs record no digest for the diagnostic that "
                        "produced them")
    if identity.get(spec["result_flag"]) is not True:
        problems.append(f"the retained runs do not record {spec['result_flag']} as true, "
                        f"and this case would credit the pair on exactly that reading")
    judged = retained_output(identity.get("reference_output_sha256"))
    pinned_ref = retained_output(pinned.get("tracker_octave_instrumented.mat"))
    if judged is None:
        problems.append(f"no retained evidence file has the digest the runs were judged "
                        f"against, {str(identity.get('reference_output_sha256'))[:12]}")
    elif os.path.basename(judged) != identity.get("reference_output"):
        # THE NAME AND THE DIGEST MUST AGREE. A confirmation round changed only the name to
        # the port output's and the artifact then named the wrong reference file, with the
        # count unchanged because the digest had resolved the right one.
        problems.append(f"the retained runs name reference output "
                        f"{identity.get('reference_output')} and the digest they record "
                        f"is {os.path.basename(judged)}")
    elif pinned_ref is None:
        problems.append("the case being superseded pins a reference output that is not "
                        "retained")
    elif not same_indexed_tracks(pinned_tracks(judged), pinned_tracks(pinned_ref)):
        problems.append(f"the reference output the runs were judged against, "
                        f"{os.path.basename(judged)}, does not hold the same finished "
                        f"tracks AT THE SAME INDICES as the one the case pins, "
                        f"{os.path.basename(pinned_ref)}")
    return problems


def diagnostic_source_status(identity):
    """Whether the diagnostic the runs record is the source on disk today, as a fact.

    The receipts are declared by this producer from the retained run records and the
    specification, not recovered from a verified frozen source. Pair 30's recorded
    diagnostic digest differs from the current script, which does not invalidate a
    historical run, and does mean the current source cannot stand in for the one that ran.
    Stated in the artifact rather than refused."""
    script = identity.get("script")
    path = os.path.join(SCRIPTS, script) if script else None
    current = X.digest(path) if path and os.path.exists(path) else None
    return {"diagnostic_script": script,
            "diagnostic_script_sha256_at_the_runs": identity.get("script_sha256"),
            "diagnostic_script_sha256_now": current,
            "diagnostic_script_matches_current_source":
                bool(current) and current == identity.get("script_sha256")}


def build(spec):
    retained = json.load(open(os.path.join(ARTIFACTS, spec["runs"])))
    source = json.load(open(os.path.join(ARTIFACTS, spec["source_case"])))
    pair = spec["pair"]
    identity = retained["identity"]

    missing = [name for name in spec["roles"].values()
               if name is not None and name not in retained["runs"]]
    if missing:
        raise SystemExit(f"{spec['runs']} holds no run named {missing}")
    problems = identity_problems(spec, retained, source)
    if problems:
        raise SystemExit(f"REFUSED, {spec['runs']} is not the experiment "
                         f"{spec['out']} would transcribe: " + "; ".join(problems))
    status = diagnostic_source_status(identity)

    intervention = {"operation": spec["operation"], "partial_runs": [],
                    # THE STANDING OF THESE RECEIPTS, stated rather than implied. They are
                    # NEW DECLARATIONS made by this producer from the retained run records
                    # and its own specification, and the removed indices are re-derived
                    # from the port's own axes. They are not recovered from a verified
                    # frozen source and none of it is execution evidence. The checker grades
                    # them as the replay's claim.
                    "receipts_basis": (
                        "declared by scripts/write_operation_cases.py from the retained "
                        "run records and its specification. The diagnostic the runs "
                        f"record, {status['diagnostic_script']}, "
                        + ("matches" if status["diagnostic_script_matches_current_source"]
                           else "does NOT match")
                        + " the current source. Not bound to independently checked "
                        "execution evidence.")}
    for role, name in spec["roles"].items():
        if name is None:
            continue
        tracks = [as_track(t) for t in retained["runs"][name]]
        reproduced = role == "intervention"
        intervention[role] = {
            "retained_run": name,
            "reference_tracks": [{"reference_index": pair,
                                  "reproduced_exactly": reproduced}],
            "finished_tracks": tracks,
            "finished_tracks_in_the_western_box": in_western_box(
                tracks, source["parameters"].get("west"),
                source["parameters"].get("feature_life")),
        }
    intervention["intervention"].update(spec["fields"])
    intervention["intervention"]["changed_at"] = spec["changed_at"]
    intervention["intervention"]["changed_at_basis"] = spec["changed_at_basis"]
    receipts = json.loads(json.dumps(spec["receipts"]))
    if spec.get("locate"):
        removed, population = locate_axis(spec["changed_at"], spec["locate"]["removed_index"])
        spare, _ = locate_axis(spec["changed_at"], spec["locate"]["negative_removed_index"])
        assert population == spec["fields"]["population_before"], population
        intervention["intervention"]["removed_index"] = removed
        receipts["negative_control"]["removed_index"] = spare
    for label, receipt in receipts.items():
        receipt["step"] = spec["changed_at"]
        intervention[label]["receipt"] = receipt

    case = {
        "case_id": source["case_id"],
        "case_name": spec["out"].replace(".json", ""),
        "generated_by": "scripts/write_operation_cases.py",
        "explains": {"unmatched_v1_tracks": [], "v1_extra_pairs": [pair]},
        # THE SOURCE CASE'S PARAMETERS WHOLE, not a chosen subset. A first version copied
        # five keys and dropped `axis_box`, which is part of the effective SELECTION REGION,
        # so the vertices this case recorded were selected under a different region than the
        # accounting expected and it refused. Copying the block keeps the region identical.
        "parameters": dict(source["parameters"]),
        "intervention": intervention,
        # VERSION 1'S DUMPED VERTICES AT THE DECLARED STEPS, carried over from the case this
        # one supersedes. A removal or a reordering injects nothing, so these are not an
        # expectation any run is checked against here, but the accounting requires every case
        # to record what its reference log holds at the steps it declares, and that check is
        # COMMON rather than injection-specific. Carrying them keeps it intact; exempting
        # these operations from it would have been the easier repair and the wrong one.
        "at_divergence": source.get("at_divergence"),
        "input_sha256": dict(source["input_sha256"]),
        "evidence_binding": {
            "retained_runs": spec["runs"],
            "retained_runs_sha256": X.digest(os.path.join(ARTIFACTS, spec["runs"])),
            "reference_output_the_runs_were_judged_against": identity["reference_output"],
            "reference_output_sha256": identity["reference_output_sha256"],
            "reference_track_index": identity["reference_track_index"],
            "case_sha256_at_the_runs": identity["case_sha256"],
            "comparison": identity["comparison"],
            # CHECKED, NOT COPIED: `identity_problems` refused above unless the case input,
            # the reference index, the step and the judged reference's content all agree
            # with the case being superseded
            "identity_checked_against": spec["source_case"],
            **status,
        },
        "what_this_does_not_establish": [
            "that the port is wrong to produce what the intervention changes; the "
            "intervention is synthetic and shows SUFFICIENCY in this window only",
            "that the same holds in any other window, which the six preselected windows "
            "exist to test",
            "that any other stage of the two programs agrees, since association, pruning "
            "and finalization were exercised here rather than compared",
        ],
    }
    case["parameters"]["reference_tracks"] = [pair]
    case["parameters"]["discrepancy_times"] = declared_discrepancies(source, pair)
    # `agreeing_steps` in the source case named times both sides HOLD an observation, which
    # the review showed are not times they AGREE at: two of pair 30's have different
    # positions. The name is retired here rather than carried under a false meaning.
    case["parameters"].pop("agreeing_steps", None)
    case["parameters"]["shared_observation_times_note"] = (
        "times both sides hold an observation are not times they agree; see "
        "discrepancy_times.displaced")
    if spec["note"]:
        case["what_this_does_not_establish"].insert(0, spec["note"])
    return case


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None)
    args = ap.parse_args(argv)
    for key, spec in CASES.items():
        if args.only and args.only != key:
            continue
        case = build(spec)
        path = os.path.join(ARTIFACTS, spec["out"])
        with open(path, "w") as fh:
            json.dump(case, fh, indent=1, sort_keys=True)
        runs = [r for r in case["intervention"] if isinstance(case["intervention"][r], dict)]
        print(f"wrote {path}")
        print(f"  operation {spec['operation']}, pair {spec['pair']}, "
              f"runs {sorted(runs)}")
        if spec["note"]:
            print(f"  NOTE: {spec['note'].split('.')[0]}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
