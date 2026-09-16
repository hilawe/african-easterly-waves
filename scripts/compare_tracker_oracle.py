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
import os
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
    n = int(np.asarray(raw["n"]).ravel()[0])
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
    if oracle_name != FAITHFUL_ORACLE:
        print(f"  NOTE: comparing against {oracle_name}, not the faithful "
              f"{FAITHFUL_ORACLE}")
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
    args = ap.parse_args(argv)
    if not args.oracle_dir:
        ap.error("set AEW_ORACLE_DIR or pass --oracle-dir")

    v1, port, exported_case, _ = load_case(args.oracle_dir, oracle_name=args.oracle)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
