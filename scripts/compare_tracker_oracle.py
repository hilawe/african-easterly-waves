#!/usr/bin/env python3
"""Compare version 1's own finished tracks against the port's, on identical input.

WHAT MAKES THIS DIFFERENT FROM EVERY EARLIER COMPARISON. The published version 1 record
is the output of a different reanalysis subset, a different climatology and a full year
of association state, so a disagreement with it has many possible causes and none of them
is isolated. Here version 1's `find_ews_f.m` has been handed the SAME raw fields the port
was handed, over the same contiguous window. That removes the reanalysis subset, the
climatology and the association history as explanations for a disagreement, which the
published record cannot. It does NOT narrow the difference to association and prune:
detection, contouring, merging, the smoothing and ordinary numerical differences are all
still in play, and the `contourc` substitution for version 1's `contours` is a known one.

MATCHING. Tracks are compared by position over the timesteps they share. A pair is
eligible when it overlaps by at least MIN_OVERLAP timesteps and its mean separation over
that overlap is within TOLERANCE_DEG. Pairs are then assigned one-to-one so that the
NUMBER OF MATCHES IS MAXIMIZED, with total separation as the tie-break. One version 1
track cannot absorb several of the port's, which would hide the duplication defect
version 1's own published record carries (1.90 tracks per distinct wave). The many-to-one
count is reported separately for that reason.

NOT GREEDY. The first version of this assigned pairs best-first, which is not
maximum-cardinality: taking the closest pair can consume a track that was the only
partner available to two others, so the port could be reported as reproducing fewer
tracks than it does. A review constructed a three-edge case where greedy returns one pair
and two exist.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment

MIN_OVERLAP = 2
TOLERANCE_DEG = 5.0


def read_tracks(path):
    """Read the (n, lat<i>, lon<i>, time<i>) layout both harness sides write."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} is absent. The run that writes it did not finish.")
    raw = loadmat(path)
    # A DECLARED COUNT IS NOT A NUMBER OF TRACKS UNTIL IT IS ONE. A review set both
    # outputs' counts to -1 and watched the comparison succeed on zero tracks a side, with
    # both provenance records still verified, and a fractional count was truncated.
    declared = np.asarray(raw["n"]).ravel()[0]
    if not np.isfinite(declared) or float(declared) < 0 \
            or float(declared) != int(float(declared)):
        raise SystemExit(
            f"{path} declares {declared!r} tracks, which is not a whole count. A file whose "
            f"count cannot be read holds an unknown number of tracks, not zero.")
    n = int(float(declared))
    if "case_id" not in raw:
        raise SystemExit(
            f"{path} carries no case id, so nothing establishes which exported window it "
            f"was computed from. Re-export and re-run.")
    tracks = []
    for i in range(n):
        tracks.append({
            "lat": np.asarray(raw[f"lat{i}"], dtype=float).ravel(),
            "lon": np.asarray(raw[f"lon{i}"], dtype=float).ravel(),
            "time": np.asarray(raw[f"time{i}"], dtype=float).ravel(),
        })
    return tracks, str(np.asarray(raw["case_id"]).ravel()[0]).strip()


# THE FAITHFUL ORACLE IS THE DEFAULT, and that is not a preference. `find_ews_f.m` builds
# its association polygons with `convhull(...,'simplify',true)`, which MATLAB R2026a accepts
# and Octave refuses, leaving an oversized fallback in its place. The run made with a
# working hull is therefore the one that reproduces version 1 as published; the other is
# the same program degraded by the substitution. Defaulting to the degraded file was how
# 82.1 percent got quoted as the port's fidelity for most of a session.
FAITHFUL_ORACLE = "tracker_octave_hullfixed.mat"
DEGRADED_ORACLE = "tracker_octave.mat"
# The instrumented runner (scripts/octave/run_tracker_instrumented.m) writes this name.
# A NAME DECIDES NOTHING ABOUT FAITHFULNESS. Whether an oracle run is the faithful one
# (the convex-hull call repaired, which Octave otherwise refuses) is read from the
# PRODUCER RECORD the runner writes inside its output at run time, with the hashes of
# the source files that executed. A review copied outputs beside a fake source and
# watched the first version of this artifact call them faithful by file name; that
# path now reports the provenance as unverified and the faithfulness as unknown.
REPAIRED_INSTRUMENTED_ORACLE = "tracker_octave_instrumented.mat"
KNOWN_ORACLE_NAMES = (FAITHFUL_ORACLE, REPAIRED_INSTRUMENTED_ORACLE)


def read_producer(path):
    """The producer record a harness side wrote into its output, or None."""
    raw = loadmat(path, variable_names=["producer_json"])
    if "producer_json" not in raw:
        return None
    text = str(np.asarray(raw["producer_json"]).ravel()[0])
    try:
        return json.loads(text)
    except ValueError:
        return {"unparseable": text[:200]}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


HEX64 = re.compile(r"^[0-9a-f]{64}$")
ORACLE_REQUIRED = {"producer": str, "case_id": str, "instrumented": bool,
                   "repaired_convhull": bool, "unrepaired_convhull_sites": int,
                   "executed_source_sha256": dict, "runner_sha256": str}
PORT_REQUIRED = {"producer": str, "case_id": str, "git_dirty": bool,
                 "source_sha256": dict, "settings": dict}
# The entries a port record must name for its source inventory to identify the
# producing code: the tracker's pipeline module and the exporter that ran it. Any one
# path under the package is not enough (a review named a nonexistent module there).
PORT_REQUIRED_SOURCES = ("src/aew/v1port/pipeline.py", "scripts/export_tracker_case.py")


def _digest_problems(mapping, label):
    problems = []
    if not mapping:
        problems.append(f"{label} is empty")
        return problems
    for key, digest in mapping.items():
        if not isinstance(digest, str) or not HEX64.match(digest):
            problems.append(f"{label}[{key}] is not a sha256 digest")
    return problems


def validate_oracle_record(rec, outer_case, exported_case):
    """The reasons an oracle producer record is NOT a valid record of THIS output.

    A record is a claim about what produced the tracks it travels with. It is valid
    only when it has the fields the runner writes, with their types, when its case
    identifier is the case the output and the export carry, when every digest it
    names is well formed and the executed find_ews_f.m is among them, and when its
    repair flag agrees with its own count of unrepaired sites. A review fed the
    first version a record naming another case, an empty source mapping, and a
    digest that was not one, and each came back verified. Validity is about the
    record's own consistency and is kept separate from whether the tree beside the
    output still matches it.
    """
    problems = []
    if not isinstance(rec, dict):
        return ["record is not a mapping"]
    for key, typ in ORACLE_REQUIRED.items():
        if key not in rec:
            problems.append(f"missing {key}")
        elif typ is int and (isinstance(rec[key], bool) or not isinstance(rec[key], (int, float))
                             or int(rec[key]) != rec[key]):
            problems.append(f"{key} is not an integer")
        elif typ is not int and not isinstance(rec[key], typ):
            problems.append(f"{key} is not {typ.__name__}")
    if problems:
        return problems
    if rec["case_id"] != outer_case or rec["case_id"] != exported_case:
        problems.append(f"record case {rec['case_id']!r} is not the output's "
                        f"{outer_case!r} or the export's {exported_case!r}")
    problems += _digest_problems(rec["executed_source_sha256"], "executed_source_sha256")
    if "v1_instrumented__find_ews_f_m" not in rec["executed_source_sha256"]:
        problems.append("executed_source_sha256 does not name find_ews_f.m")
    if not HEX64.match(rec["runner_sha256"]):
        problems.append("runner_sha256 is not a sha256 digest")
    if rec["repaired_convhull"] != (int(rec["unrepaired_convhull_sites"]) == 0):
        problems.append("repaired_convhull disagrees with unrepaired_convhull_sites")
    return problems


def validate_port_record(rec, outer_case, exported_case, output_settings):
    """The reasons a port producer record is NOT a valid record of THIS output: the
    fields the exporter writes, the case identifier of this output and export, a
    nonempty source mapping of well-formed digests naming the tracker package, a head
    that is a commit identifier when present, and settings that agree with the flags
    the output itself carries."""
    problems = []
    if not isinstance(rec, dict):
        return ["record is not a mapping"]
    for key, typ in PORT_REQUIRED.items():
        if key not in rec:
            problems.append(f"missing {key}")
        elif not isinstance(rec[key], typ):
            problems.append(f"{key} is not {typ.__name__}")
    if problems:
        return problems
    if rec["case_id"] != outer_case or rec["case_id"] != exported_case:
        problems.append(f"record case {rec['case_id']!r} is not the output's "
                        f"{outer_case!r} or the export's {exported_case!r}")
    problems += _digest_problems(rec["source_sha256"], "source_sha256")
    for required in PORT_REQUIRED_SOURCES:
        if required not in rec["source_sha256"]:
            problems.append(f"source_sha256 does not name {required}")
    head = rec.get("git_head")
    if head is not None and not re.match(r"^[0-9a-f]{40}$", str(head)):
        problems.append("git_head is not a commit identifier")
    for flag in ("exclusive", "absorb"):
        if flag in output_settings and flag in rec["settings"] \
                and bool(rec["settings"][flag]) != bool(output_settings[flag]):
            problems.append(f"settings.{flag} {rec['settings'][flag]!r} disagrees with the "
                            f"output's {output_settings[flag]!r}")
        elif flag not in rec["settings"]:
            problems.append(f"settings lacks {flag}")
    return problems


def oracle_provenance(oracle_dir, oracle_name, outer_case, exported_case):
    """What the oracle output says produced it, validated, then checked against the
    tree beside it.

    `status` is one of "unverified" (no record), "invalid" (a record that fails its
    own consistency checks, with `problems`), "inconsistent" (a valid record whose
    executed-source digests differ from the files now beside the output, with
    `tree_mismatches`), or "verified" (valid and the tree matches). `faithful` is the
    record's detected repair mode when the record is valid and None otherwise. The
    record itself is returned whatever the status, because a valid record of a run
    whose source has since changed is still the record of that run.
    """
    rec = read_producer(os.path.join(oracle_dir, oracle_name))
    if rec is None:
        return {"status": "unverified", "faithful": None, "record": None, "problems": []}
    problems = validate_oracle_record(rec, outer_case, exported_case)
    if problems:
        return {"status": "invalid", "faithful": None, "record": rec, "problems": problems}
    mismatches = []
    here = os.path.dirname(os.path.abspath(__file__))
    for key, digest in rec["executed_source_sha256"].items():
        # the runner encodes "dir/file.m" as "dir__file_m", since a struct field name
        # cannot carry a slash or a dot
        rel = key.replace("__", "/")
        rel = rel[:-2] + ".m" if rel.endswith("_m") else rel
        candidate = os.path.join(oracle_dir, rel) if rel.startswith("v1_instrumented/") \
            else os.path.join(here, "octave", rel)
        if not os.path.exists(candidate) or _sha256(candidate) != digest:
            mismatches.append(rel)
    runner = os.path.join(here, "octave", "run_tracker_instrumented.m")
    if not os.path.exists(runner) or _sha256(runner) != rec["runner_sha256"]:
        mismatches.append("scripts/octave/run_tracker_instrumented.m")
    status = "verified" if not mismatches else "inconsistent"
    return {"status": status, "faithful": bool(rec["repaired_convhull"]), "record": rec,
            "problems": [], "tree_mismatches": mismatches}


def port_provenance(oracle_dir, port_name, outer_case, exported_case, output_settings,
                    repo=None):
    """What the port output says produced it, validated, then its declared source
    inventory compared file by file against the source tree of THIS checkout.

    `status` mirrors the oracle side: "unverified" (no record), "invalid" (fails its
    own consistency checks, with `problems`), "inconsistent" (valid, but a declared
    file is absent from the tree or hashes differently now, with `tree_mismatches`;
    the record is kept, because it remains the record of that run), or "verified"
    (valid and every declared file matches the tree). A review fed the first version
    a well-formed digest that matched no file and a path that did not exist, and both
    came back verified, because syntax was all that was checked.
    """
    rec = read_producer(os.path.join(oracle_dir, port_name))
    if rec is None:
        return {"status": "unverified", "record": None, "problems": []}
    problems = validate_port_record(rec, outer_case, exported_case, output_settings)
    if problems:
        return {"status": "invalid", "record": rec, "problems": problems}
    repo = repo or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mismatches = []
    for rel, digest in rec["source_sha256"].items():
        path = os.path.join(repo, rel)
        if not os.path.exists(path) or _sha256(path) != digest:
            mismatches.append(rel)
    return {"status": "verified" if not mismatches else "inconsistent", "record": rec,
            "problems": [], "tree_mismatches": sorted(mismatches)}


def load_case(oracle_dir, port_name="tracker_port.mat", oracle_name=FAITHFUL_ORACLE):
    """Both track sets, checked against the EXPORTED case rather than only each other.

    Factored out because three later analysis scripts each read the same two files and
    compared only their two identifiers, which a review defeated by planting outputs
    labeled for one window beside an export for another: every one of them ran to
    completion and reported numbers. Checking the outputs against one another proves they
    agree with each other, not that either read the export.

    Returns (v1 tracks, port tracks, case id, port settings).
    """
    path = os.path.join(oracle_dir, oracle_name)
    if not os.path.exists(path) and oracle_name == FAITHFUL_ORACLE:
        raise SystemExit(
            f"{path} is absent. That is the oracle run with a working convex hull, which "
            f"is the one that reproduces version 1 as published. Build it with "
            f"`make_instrumented_v1.py --repair-convhull` and run it, or pass "
            f"--oracle {DEGRADED_ORACLE} and say so in whatever you report.")
    v1, v1_case = read_tracks(path)
    port, port_case = read_tracks(os.path.join(oracle_dir, port_name))
    if oracle_name not in KNOWN_ORACLE_NAMES:
        print(f"  NOTE: comparing against {oracle_name}, not one of the harness's own "
              f"output names ({' or '.join(KNOWN_ORACLE_NAMES)})")
    case_path = os.path.join(oracle_dir, "tracker_case.mat")
    if not os.path.exists(case_path):
        raise SystemExit(f"{case_path} is absent, so nothing says what was exported.")
    exported = loadmat(case_path, variable_names=["case_id"])
    if "case_id" not in exported:
        raise SystemExit(f"{case_path} carries no case id. Re-export and re-run.")
    exported_case = str(np.asarray(exported["case_id"]).ravel()[0]).strip()
    if not (v1_case == port_case == exported_case):
        raise SystemExit(
            f"these files are not one case: the export carries {exported_case}, "
            f"version 1's tracks carry {v1_case} and the port's carry {port_case}. "
            f"Re-export and re-run.")
    raw = loadmat(os.path.join(oracle_dir, port_name),
                  variable_names=["exclusive", "absorb"])
    settings = {k: bool(np.asarray(raw[k]).ravel()[0]) for k in ("exclusive", "absorb")
                if k in raw}
    return v1, port, exported_case, settings


def overlap_separation(a, b):
    """Timesteps shared by two tracks, and their mean separation over them.

    Returns (0, inf) when they never coexist. Separation is in degrees on the plate
    carree grid the tracker itself works on, weighted by cos(latitude) in longitude so a
    fixed degree tolerance means roughly the same distance across the wave belt.
    """
    shared, ia, ib = np.intersect1d(a["time"], b["time"], return_indices=True)
    if shared.size == 0:
        return 0, np.inf
    dlat = a["lat"][ia] - b["lat"][ib]
    mean_lat = np.radians(0.5 * (a["lat"][ia] + b["lat"][ib]))
    dlon = (a["lon"][ia] - b["lon"][ib]) * np.cos(mean_lat)
    return int(shared.size), float(np.mean(np.hypot(dlat, dlon)))


def match(reference, candidate):
    """Maximum-cardinality one-to-one assignment, separation as the tie-break.

    Ineligible pairs are given a cost far above any eligible one, so a solution that
    leaves a track unmatched always costs more than one that matches it, and the
    minimum-cost assignment is therefore also a maximum-cardinality one. Ineligible
    assignments are dropped afterwards.
    """
    eligible = {}
    for i, r in enumerate(reference):
        for j, c in enumerate(candidate):
            n, sep = overlap_separation(r, c)
            if n >= MIN_OVERLAP and sep <= TOLERANCE_DEG:
                eligible[(i, j)] = (n, sep)
    if not eligible:
        return [], eligible
    # Strictly greater than the largest total an all-eligible solution could reach, so
    # trading one match away for any number of cheaper ones is never profitable.
    forbidden = TOLERANCE_DEG * (min(len(reference), len(candidate)) + 1) + 1.0
    cost = np.full((len(reference), len(candidate)), forbidden, dtype=float)
    for (i, j), (_, sep) in eligible.items():
        cost[i, j] = sep
    rows, cols = linear_sum_assignment(cost)
    assigned = [(int(i), int(j), eligible[(int(i), int(j))][0],
                 eligible[(int(i), int(j))][1])
                for i, j in zip(rows, cols) if (int(i), int(j)) in eligible]
    return assigned, eligible


# A GENUINE PARTITION, tested in order with the last case unconditional. The first
# version of this listed three latitude-longitude boxes and called them a partition in
# the prose, and they were not: a track north of 30 degrees inside the African longitudes
# fell in no box at all, so the table silently described 116 of 117 tracks. Ordering the
# tests and ending with a catch-all makes the claim structurally true rather than true by
# inspection, and the totals are asserted below.
REGIONS = (
    ("the African wave belt, 0-30N 40W-40E",
     lambda lat, lon: 0.0 <= lat <= 30.0 and -40.0 <= lon <= 40.0),
    ("elsewhere in the northern hemisphere", lambda lat, lon: lat >= 0.0),
    ("the southern hemisphere", lambda lat, lon: True),
)


def region_of(track):
    """The name of the one region a track's mean position falls in."""
    lat, lon = float(np.mean(track["lat"])), float(np.mean(track["lon"]))
    for name, test in REGIONS:
        if test(lat, lon):
            return name
    raise AssertionError("the last region must be unconditional")


def describe(name, tracks):
    # An empty side is a real outcome, not a bug to crash on: a window with no waves, or
    # an Octave run that finished and found nothing, both produce it, and reporting zero
    # is more useful than a traceback from min() on an empty sequence.
    if not tracks:
        print(f"  {name:>28}: no tracks")
        return
    lengths = np.array([t["time"].size for t in tracks], dtype=float)
    lons = np.array([float(np.mean(t["lon"])) for t in tracks])
    print(f"  {name:>28}: {len(tracks):4d} tracks, "
          f"median {np.median(lengths):.0f} steps "
          f"(range {lengths.min():.0f} to {lengths.max():.0f}), "
          f"median longitude {np.median(lons):+.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle-dir", default=os.environ.get("AEW_ORACLE_DIR"))
    ap.add_argument("--oracle", default=FAITHFUL_ORACLE,
                    help="which version 1 run to compare against")
    ap.add_argument("--out", default=None,
                    help="write the summary numbers and every input's digest as JSON")
    args = ap.parse_args(argv)
    if not args.oracle_dir:
        ap.error("set AEW_ORACLE_DIR or pass --oracle-dir")

    v1, port, exported_case, port_settings = load_case(args.oracle_dir,
                                                       oracle_name=args.oracle)
    v1_case = port_case = exported_case
    # ALL THREE FILES, not just the two outputs. Checking the outputs against each other
    # only proves they agree with one another, and a review planted a `tracker_case.mat`
    # for a different window beside two agreeing outputs: the comparison printed
    # "identical raw fields, same window" and exited zero. The exported case is the thing
    # both sides were supposed to have read, so it is the one they must both match.
    case_path = os.path.join(args.oracle_dir, "tracker_case.mat")
    if not os.path.exists(case_path):
        raise SystemExit(f"{case_path} is absent, so nothing says what was exported.")
    exported = loadmat(case_path, variable_names=["case_id"])
    if "case_id" not in exported:
        raise SystemExit(f"{case_path} carries no case id. Re-export and re-run.")
    exported_case = str(np.asarray(exported["case_id"]).ravel()[0]).strip()
    if not (v1_case == port_case == exported_case):
        raise SystemExit(
            f"these files are not one case: the export carries {exported_case}, "
            f"version 1's tracks carry {v1_case} and the port's carry {port_case}. "
            f"Both sides write to fixed paths, so a run that died part-way leaves an "
            f"earlier window's answer in place. Re-export and re-run.")

    print("VERSION 1'S OWN TRACKER VS THE PORT, identical raw fields, same window")
    print(f"case {v1_case}\n")
    describe("version 1, find_ews_f.m", v1)
    describe("the port, track_year", port)

    assigned, eligible = match(v1, port)
    matched_v1 = {i for i, _, _, _ in assigned}
    matched_port = {j for _, j, _, _ in assigned}
    seps = np.array([s for _, _, _, s in assigned]) if assigned else np.array([])

    prov = oracle_provenance(args.oracle_dir, args.oracle, v1_case, exported_case)
    port_prov = port_provenance(args.oracle_dir, "tracker_port.mat", port_case,
                                exported_case, port_settings)
    port_rec = port_prov["record"]
    if prov["status"] != "verified":
        why = {"unverified": "carries no producer record",
               "invalid": "carries a producer record that fails its own consistency "
                          "checks: " + "; ".join(prov.get("problems", [])),
               "inconsistent": "was produced by source that differs from the tree "
                               "beside it"}[prov["status"]]
        print(f"  PROVENANCE {prov['status'].upper()}: the oracle output {why}. Its "
              f"faithfulness is {'unknown' if prov['faithful'] is None else prov['faithful']} "
              f"and this comparison is reported as such.")
    if port_prov["status"] != "verified":
        print(f"  PORT PROVENANCE {port_prov['status'].upper()}: "
              + ("no producer record" if port_prov["status"] == "unverified" else
                 "; ".join(port_prov["problems"]) if port_prov["status"] == "invalid" else
                 f"declared source differs from or is absent in this checkout: "
                 f"{port_prov['tree_mismatches'][:5]}"))
    if prov["status"] == "verified" and not prov["faithful"]:
        print("  NOTE: the oracle's producer record says the convex-hull call was NOT "
              "repaired, so this is the degraded oracle whatever its file name.")
    summary = {"case_id": v1_case, "oracle_file": args.oracle,
               "oracle_provenance": prov["status"],
               "oracle_faithful": prov["faithful"],
               "oracle_producer": prov.get("record"),
               "oracle_problems": prov.get("problems", []),
               "oracle_tree_mismatches": prov.get("tree_mismatches"),
               "port_producer": port_rec,
               "port_provenance": port_prov["status"],
               "port_problems": port_prov["problems"],
               "port_tree_mismatches": port_prov.get("tree_mismatches"),
               "v1_tracks": len(v1), "port_tracks": len(port),
               "v1_matched": len(matched_v1), "port_matched": len(matched_port),
               "v1_unmatched": len(v1) - len(matched_v1),
               "port_unmatched": len(port) - len(matched_port),
               "min_overlap": MIN_OVERLAP, "tolerance_deg": TOLERANCE_DEG}
    print(f"\nMAXIMUM-CARDINALITY ONE-TO-ONE MATCHING, overlap of at least "
          f"{MIN_OVERLAP} timesteps and mean")
    print(f"separation within {TOLERANCE_DEG:.0f} degrees:\n")
    print(f"  version 1 tracks the port reproduces  {len(matched_v1):4d} / {len(v1):4d}"
          f"  ({100.0 * len(matched_v1) / max(len(v1), 1):.1f}%)")
    print(f"  port tracks matching a version 1 one  {len(matched_port):4d} / "
          f"{len(port):4d}  ({100.0 * len(matched_port) / max(len(port), 1):.1f}%)")
    print(f"  version 1 tracks with no counterpart  {len(v1) - len(matched_v1):4d}")
    print(f"  port tracks with no counterpart       {len(port) - len(matched_port):4d}")
    if seps.size:
        print(f"\n  separation over matched pairs: median {np.median(seps):.2f} deg, "
              f"90th {np.percentile(seps, 90):.2f}, worst {seps.max():.2f}")
        # THE WHOLE TRACK, NOT THE SHARED PART OF IT. Two earlier versions got this
        # wrong in the same direction. The first counted a mean separation below 1e-6
        # degrees as identical, which is a tolerance and cannot support the word. The
        # second required exactly zero but still measured only over the timesteps the two
        # tracks SHARE, so a three-step track and a two-step track agreeing on their two
        # common steps was counted as identical when one of them runs a step longer.
        # Identical now means the time, latitude and longitude arrays are equal, same
        # length included.
        identical = sum(
            1 for i, j, _, _ in assigned
            if v1[i]["time"].shape == port[j]["time"].shape
            and np.array_equal(v1[i]["time"], port[j]["time"])
            and np.array_equal(v1[i]["lat"], port[j]["lat"])
            and np.array_equal(v1[i]["lon"], port[j]["lon"]))
        zero_on_shared = int(np.sum(seps == 0.0))
        near = int(np.sum(seps < 1e-6))
        print(f"  matched pairs IDENTICAL over the whole track: "
              f"{identical} / {seps.size}")
        summary.update({"identical_pairs": int(identical),
                        "separation_median_deg": float(np.median(seps)),
                        "separation_p90_deg": float(np.percentile(seps, 90)),
                        "separation_worst_deg": float(seps.max())})
        # THE NON-IDENTICAL PAIRS ON THEIR OWN, because the write-up quotes a median for
        # them and the median printed above is over ALL 96, which is a different number
        # whenever the identical pairs are a large share. A review caught the write-up
        # attributing the all-pairs median to the subset.
        rest = np.array([s for (i, j, _, s), keep in zip(
            assigned,
            [not (v1[i]["time"].shape == port[j]["time"].shape
                  and np.array_equal(v1[i]["time"], port[j]["time"])
                  and np.array_equal(v1[i]["lat"], port[j]["lat"])
                  and np.array_equal(v1[i]["lon"], port[j]["lon"]))
             for i, j, _, _ in assigned]) if keep])
        if rest.size:
            print(f"    of the {rest.size} that are NOT identical: median "
                  f"{np.median(rest):.2f} deg, worst {rest.max():.2f}")
            summary.update({"nonidentical_pairs": int(rest.size),
                            "nonidentical_median_deg": float(np.median(rest)),
                            "nonidentical_worst_deg": float(rest.max())})
        print(f"  pairs at zero separation on shared steps only: "
              f"{zero_on_shared} / {seps.size}")
        print(f"  pairs within 1e-6 degrees on shared steps:     {near} / {seps.size}")

    # Duplication, allowing many-to-one: how many port tracks are eligible for each
    # version 1 track. Version 1's own published record carries 1.90 of these.
    per_v1 = {}
    for i, j in eligible:
        per_v1.setdefault(i, set()).add(j)
    if per_v1:
        counts = np.array([len(v) for v in per_v1.values()], dtype=float)
        print(f"\n  port tracks eligible per version 1 track: mean {counts.mean():.2f}, "
              f"max {counts.max():.0f}")

    if len(v1) - len(matched_v1):
        print("\n  VERSION 1 TRACKS THE PORT DOES NOT PRODUCE, longest first:")
        missing = sorted((i for i in range(len(v1)) if i not in matched_v1),
                         key=lambda i: -v1[i]["time"].size)
        for i in missing[:12]:
            t = v1[i]
            print(f"    {t['time'].size:3d} steps, "
                  f"lat {t['lat'].mean():+6.2f}, lon {t['lon'].mean():+7.2f}, "
                  f"times {t['time'].min():.2f} to {t['time'].max():.2f}")
        if len(missing) > 12:
            print(f"    ... and {len(missing) - 12} more")

    # WHERE THE AGREEMENT IS, because the whole-domain rate mixes the wave belt with the
    # far field, and only the first of those is what the record is used for. Matching is
    # NOT redone per region; the one-to-one assignment above is partitioned by the
    # version 1 track's mean position, so a region cannot gain a match by losing a
    # competitor.
    print("\n  BY REGION, partitioning the same assignment:")
    print(f"    {'region':>37} {'matched':>9} {'of':>5} {'rate':>7}")
    placed = {name: [] for name, _ in REGIONS}
    for i in range(len(v1)):
        placed[region_of(v1[i])].append(i)
    total_tracks = total_matched = 0
    for label, _ in REGIONS:
        idx = placed[label]
        hit = sum(1 for i in idx if i in matched_v1)
        total_tracks += len(idx)
        total_matched += hit
        rate = f"{100.0 * hit / len(idx):.1f}%" if idx else "n/a"
        print(f"    {label:>37} {hit:9d} {len(idx):5d} {rate:>7}")
    # The regions are a partition or this table is not the one described. Asserted rather
    # than eyeballed, because the previous version's shortfall was one track and nobody
    # added the column up.
    assert total_tracks == len(v1), f"{total_tracks} placed, {len(v1)} tracks"
    assert total_matched == len(matched_v1), (
        f"{total_matched} placed matches, {len(matched_v1)} matched")
    print(f"    {'':>37} {total_matched:9d} {total_tracks:5d}  (totals check)")

    # WHETHER THE WAVE BELT'S RATE IS DISTINGUISHABLE AT ALL, computed here rather than
    # by hand, because a number that appears only in prose has nothing keeping it true
    # when the window changes. The belt is the region the record is used for, so a lower
    # rate there would matter, and this is what says whether the sample can show one.
    belt = REGIONS[0][0]
    inside = placed[belt]
    outside = [i for name, _ in REGIONS[1:] for i in placed[name]]
    hit_in = sum(1 for i in inside if i in matched_v1)
    hit_out = sum(1 for i in outside if i in matched_v1)
    if inside and outside:
        from scipy import stats
        p = stats.fisher_exact([[hit_in, len(inside) - hit_in],
                                [hit_out, len(outside) - hit_out]])[1]
        print(f"\n  the wave belt ({hit_in}/{len(inside)}) against everywhere else "
              f"({hit_out}/{len(outside)}):")
        print(f"    two-sided Fisher exact p = {p:.3f}"
              f"{'' if p < 0.05 else ', so this sample does not separate them'}")
    if args.out:
        # EVERY INPUT IS BOUND BY DIGEST, including the source the oracle ran from when
        # it is the instrumented copy, so a reader can tell a repaired-hull run from a
        # degraded one without trusting the file name.
        files = {name: os.path.join(args.oracle_dir, name)
                 for name in ("tracker_case.mat", "tracker_port.mat", args.oracle)}
        inst = os.path.join(args.oracle_dir, "v1_instrumented", "find_ews_f.m")
        if os.path.exists(inst):
            files["v1_instrumented/find_ews_f.m"] = inst
        here = os.path.dirname(os.path.abspath(__file__))
        try:
            head = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            head = None
        summary.update({"generated_by": "scripts/compare_tracker_oracle.py",
                        "input_sha256": {k: _sha256(v) for k, v in files.items()},
                        "source_sha256": {"scripts/compare_tracker_oracle.py":
                                          _sha256(os.path.abspath(__file__))},
                        # the checkout this COMPARISON ran in, named as such; the head
                        # that PRODUCED the port's tracks is in port_producer
                        "comparison_git_head": head,
                        "what_this_is": "one-to-one maximum-cardinality track matching of "
                                        "version 1's own tracker against the port on "
                                        "identical raw fields over one window; equal "
                                        "final tracks do not establish that the "
                                        "intermediate stages agreed"})
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=1, sort_keys=True)
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
