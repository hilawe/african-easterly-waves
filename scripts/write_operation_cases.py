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
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402

ARTIFACTS = "docs/aewc_v2/artifacts"
EVIDENCE = "docs/aewc_v2/evidence"

# Which retained run plays which part in the contract's four-run shape. The retained files
# use the diagnostics' own names; the contract's names are on the left.
CASES = {
    "pair63_reordering": {
        "runs": "pair63_reorder_runs_2026-09-22.json",
        "out": "pair63_reordering_case_2026-09-22.json",
        "operation": "reordering",
        "pair": 63,
        "source_case": "south_atlantic_start_case_2026-09-21.json",
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
        "note": None,
    },
    "pair30_removal": {
        "runs": "pair30_removal_runs_2026-09-22.json",
        "out": "pair30_removal_case_2026-09-22.json",
        "operation": "removal",
        "pair": 30,
        "source_case": "peru_case_2026-09-20.json",
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


def build(spec):
    retained = json.load(open(os.path.join(ARTIFACTS, spec["runs"])))
    source = json.load(open(os.path.join(ARTIFACTS, spec["source_case"])))
    pair = spec["pair"]
    identity = retained["identity"]

    missing = [name for name in spec["roles"].values()
               if name is not None and name not in retained["runs"]]
    if missing:
        raise SystemExit(f"{spec['runs']} holds no run named {missing}")

    intervention = {"operation": spec["operation"], "partial_runs": []}
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
