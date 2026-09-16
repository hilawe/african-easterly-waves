#!/usr/bin/env python3
"""Why the unmatched tracks sit over Africa on one side and the Atlantic on the other.

THE OBSERVATION THIS EXISTS TO EXPLAIN. On the 60-timestep oracle window the two programs
agree exactly on where their MATCHED tracks are (median longitude -48.8 on both sides),
and disagree in opposite directions on the rest: version 1's 21 unmatched tracks have a
median longitude of +0.7, over Africa, while the port's 22 have -63.5, out over the
Atlantic. A single detection difference would not do that, because it would move both
populations the same way. Something is removing tracks in the east and adding them in the
west, or one thing is doing both.

THE HYPOTHESES, each with the measurement that would support or refute it.

  H1 NEAR MISSES. The two populations are the same waves, failing only the matching
     criteria (five degrees, two shared timesteps). Refuted if relaxing both leaves the
     counts roughly where they were.
  H2 DOMAIN EDGE. The tracked domain stops at 40E. A wave crossing that boundary is
     truncated, and if the two programs truncate differently the eastern population is an
     edge artifact rather than a detection difference. Supported if version 1's unmatched
     tracks crowd the eastern boundary.
  H3 WINDOW EDGE. The window is 60 steps cut out of a year, so tracks alive at either end
     are fragments. Supported if the unmatched tracks crowd the first or last timesteps.
  H4 SPLITTING. The port's Atlantic extras are fragments of waves it ALREADY matched,
     which is the duplication defect the project has measured at the record level.
     Supported if an unmatched port track sits close in space and time to a matched one.
  H5 LIFETIME. Short tracks are structurally harder to match and version 1's prune and
     speed filters cut differently. Supported if the unmatched populations are short.

None of these is exclusive, and the point is to price them against each other rather than
to find one winner.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_tracker_oracle import (FAITHFUL_ORACLE, MIN_OVERLAP,  # noqa: E402
                                    TOLERANCE_DEG, load_case, match,
                                    overlap_separation)

EAST_EDGE = 40.0
WEST_EDGE = -140.0


def spans(track):
    """The track's longitude range, latitude mean, and first and last timestep."""
    return (float(track["lon"].min()), float(track["lon"].max()),
            float(np.mean(track["lat"])),
            float(track["time"].min()), float(track["time"].max()))


def describe_population(label, tracks, window):
    if not tracks:
        print(f"  {label:>34}: none")
        return
    lons = np.array([float(np.mean(t["lon"])) for t in tracks])
    lens = np.array([t["time"].size for t in tracks], dtype=float)
    lats = np.array([float(np.mean(t["lat"])) for t in tracks])
    east = np.array([float(t["lon"].max()) for t in tracks])
    starts = np.array([float(t["time"].min()) for t in tracks])
    ends = np.array([float(t["time"].max()) for t in tracks])
    touches_start = int((starts <= window[0] + 1e-9).sum())
    touches_end = int((ends >= window[1] - 1e-9).sum())
    near_east = int((east >= EAST_EDGE - 5.0).sum())
    print(f"  {label:>34}: n={len(tracks):3d}  median lon {np.median(lons):+7.1f}  "
          f"median lat {np.median(lats):+5.1f}  median length {np.median(lens):4.0f}")
    print(f"  {'':>34}  within 5 deg of the 40E edge: {near_east:2d}   "
          f"alive at window start: {touches_start:2d}   at window end: {touches_end:2d}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle-dir", default=os.environ.get("AEW_ORACLE_DIR"))
    ap.add_argument("--oracle", default=None, help="which version 1 run to use")
    args = ap.parse_args(argv)
    if not args.oracle_dir:
        ap.error("set AEW_ORACLE_DIR or pass --oracle-dir")

    v1, port, v1_case, settings = load_case(args.oracle_dir,
                                             oracle_name=args.oracle or FAITHFUL_ORACLE)
    if not v1 or not port:
        raise SystemExit("one side has no tracks, so there is nothing to split.")
    all_times = np.concatenate([t["time"] for t in v1 + port])
    window = (float(all_times.min()), float(all_times.max()))

    assigned, _ = match(v1, port)
    matched_v1 = {i for i, _, _, _ in assigned}
    matched_port = {j for _, j, _, _ in assigned}
    lost = [v1[i] for i in range(len(v1)) if i not in matched_v1]
    extra = [port[j] for j in range(len(port)) if j not in matched_port]

    print(f"case {v1_case}, window {window[0]:.2f} to {window[1]:.2f}\n")
    print("THE FOUR POPULATIONS")
    describe_population("version 1, matched", [v1[i] for i in matched_v1], window)
    describe_population("version 1, UNMATCHED (the east)", lost, window)
    describe_population("the port, matched", [port[j] for j in matched_port], window)
    describe_population("the port, UNMATCHED (the west)", extra, window)

    # --- H1, near misses ------------------------------------------------------------
    print(f"\nH1  NEAR MISSES. Matching again at relaxed criteria, to see whether these")
    print(f"    are the same waves failing a threshold rather than absent waves.\n")
    print(f"    {'tolerance':>10} {'min overlap':>12} {'matched':>8} "
          f"{'v1 left':>8} {'port left':>10}")
    import compare_tracker_oracle as C
    base = (C.TOLERANCE_DEG, C.MIN_OVERLAP)
    for tol, ovl in ((5.0, 2), (8.0, 2), (12.0, 2), (20.0, 2), (5.0, 1), (20.0, 1)):
        C.TOLERANCE_DEG, C.MIN_OVERLAP = tol, ovl
        a, _ = match(v1, port)
        print(f"    {tol:10.0f} {ovl:12d} {len(a):8d} {len(v1) - len(a):8d} "
              f"{len(port) - len(a):10d}")
    C.TOLERANCE_DEG, C.MIN_OVERLAP = base

    # --- H4, splitting --------------------------------------------------------------
    # An unmatched port track that overlaps in time with a MATCHED port track and sits
    # near it is a fragment of a wave the port already has, which is the duplication
    # shape. Measured against the port's own matched tracks, not against version 1.
    print(f"\nH4  SPLITTING. How close each UNMATCHED port track sits to the port's own")
    print(f"    MATCHED tracks, in degrees over the steps they share:\n")
    near = []
    for t in extra:
        best = np.inf
        for j in matched_port:
            n, sep = overlap_separation(t, port[j])
            if n >= 1:
                best = min(best, sep)
        near.append(best)
    near = np.array(near)
    finite = near[np.isfinite(near)]
    print(f"    unmatched port tracks that coexist with a matched one: "
          f"{finite.size} of {near.size}")
    if finite.size:
        for cut in (5.0, 10.0, 20.0):
            print(f"    within {cut:4.0f} degrees of a matched port track: "
                  f"{int((finite <= cut).sum()):3d}")
        print(f"    median distance to the nearest matched port track: "
              f"{np.median(finite):.1f} deg")

    # the same question asked of version 1's unmatched, as the control: if version 1
    # fragments at the same rate, splitting is not what distinguishes the two.
    near_v1 = []
    for t in lost:
        best = np.inf
        for i in matched_v1:
            n, sep = overlap_separation(t, v1[i])
            if n >= 1:
                best = min(best, sep)
        near_v1.append(best)
    near_v1 = np.array(near_v1)
    fin1 = near_v1[np.isfinite(near_v1)]
    print(f"\n    CONTROL, the same for version 1's unmatched against its own matched:")
    print(f"    coexisting: {fin1.size} of {near_v1.size}", end="")
    if fin1.size:
        print(f", within 5 deg: {int((fin1 <= 5.0).sum())}, "
              f"median {np.median(fin1):.1f} deg")
    else:
        print()

    # --- H2, where exactly ----------------------------------------------------------
    print(f"\nH2  WHERE THEY SIT, in 20 degree longitude bins, mean position:\n")
    edges = np.arange(-140.0, 41.0, 20.0)
    lost_lon = np.array([float(np.mean(t["lon"])) for t in lost])
    extra_lon = np.array([float(np.mean(t["lon"])) for t in extra])
    m_lon = np.array([float(np.mean(v1[i]["lon"])) for i in matched_v1])
    print(f"    {'bin':>16} {'v1 matched':>11} {'v1 LOST':>9} {'port EXTRA':>11}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        a = int(((m_lon >= lo) & (m_lon < hi)).sum())
        b = int(((lost_lon >= lo) & (lost_lon < hi)).sum())
        c = int(((extra_lon >= lo) & (extra_lon < hi)).sum())
        if a or b or c:
            print(f"    {f'{lo:+.0f} to {hi:+.0f}':>16} {a:11d} {b:9d} {c:11d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
