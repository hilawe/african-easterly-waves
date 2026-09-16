#!/usr/bin/env python3
"""The oracle comparison at the level of distinct waves rather than tracks.

WHY THE TRACK-LEVEL NUMBER CANNOT BE THE FIDELITY MEASURE. Matching tracks one to one
against version 1 scores 85.8 percent with the port in version 1's own configuration, and
42.5 percent with the port's DUPLICATION REPAIR turned on. The repair is the correction
the whole project exists to make, and on the eastern Pacific cluster it plainly does the
right thing, cutting the port from 12 tracks to 3 where version 1 has 4. A metric that
falls when the known defect is fixed is measuring agreement with the defect, not fidelity.

The cause is structural rather than a threshold choice. Version 1's published record
carries 1.90 tracks per distinct wave, and on this window 74 percent of its tracks sit
within 5 degrees of another of its own. When the reference counts one wave twice and the
port counts it once, one-to-one assignment MUST leave one of the reference's copies
unmatched, and the port is penalized for not duplicating.

WHAT THIS DOES INSTEAD. Each side's tracks are collapsed into distinct waves first, by the
same rule used to identify duplicates (within DUPLICATE_DEG over at least DUPLICATE_OVERLAP
shared timesteps), taking connected components so a chain of near-copies becomes one wave.
Waves are then matched one to one between the programs. A wave matches when any of its
tracks meets any of the other wave's tracks under the ordinary criteria.

WHAT IT STILL DOES NOT SETTLE. Collapsing by proximity cannot distinguish one wave counted
twice from two genuinely distinct waves passing close together, and the choice of
DUPLICATE_DEG moves the counts. The sweep at the end reports the sensitivity rather than
hiding it behind one number, and the honest reading is the shape of that sweep, not any
single row of it.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_tracker_oracle import (FAITHFUL_ORACLE, MIN_OVERLAP,  # noqa: E402
                                    TOLERANCE_DEG, load_case,
                                    overlap_separation)
from scipy.optimize import linear_sum_assignment  # noqa: E402

DUPLICATE_OVERLAP = 2


def collapse_to_waves(tracks, duplicate_deg):
    """Connected components of the near-copy relation, as lists of track indices."""
    parent = list(range(len(tracks)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            n, sep = overlap_separation(tracks[i], tracks[j])
            if n >= DUPLICATE_OVERLAP and sep <= duplicate_deg:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    groups = {}
    for i in range(len(tracks)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def wave_separation(wave_a, tracks_a, wave_b, tracks_b):
    """The closest any track of one wave comes to any track of the other."""
    best, shared = np.inf, 0
    for i in wave_a:
        for j in wave_b:
            n, sep = overlap_separation(tracks_a[i], tracks_b[j])
            if n >= MIN_OVERLAP and sep < best:
                best, shared = sep, n
    return shared, best


def match_waves(waves_a, tracks_a, waves_b, tracks_b):
    """Maximum-cardinality one-to-one assignment between waves, as for tracks."""
    eligible = {}
    for x, wa in enumerate(waves_a):
        for y, wb in enumerate(waves_b):
            n, sep = wave_separation(wa, tracks_a, wb, tracks_b)
            if n >= MIN_OVERLAP and sep <= TOLERANCE_DEG:
                eligible[(x, y)] = sep
    if not eligible:
        return []
    forbidden = TOLERANCE_DEG * (min(len(waves_a), len(waves_b)) + 1) + 1.0
    cost = np.full((len(waves_a), len(waves_b)), forbidden, dtype=float)
    for (x, y), sep in eligible.items():
        cost[x, y] = sep
    rows, cols = linear_sum_assignment(cost)
    return [(int(x), int(y)) for x, y in zip(rows, cols)
            if (int(x), int(y)) in eligible]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle-dir", default=os.environ.get("AEW_ORACLE_DIR"))
    ap.add_argument("--oracle", default=None, help="which version 1 run to use")
    ap.add_argument("--port", default="tracker_port.mat",
                    help="which port output to read, so a repaired run can be compared")
    args = ap.parse_args(argv)
    if not args.oracle_dir:
        ap.error("set AEW_ORACLE_DIR or pass --oracle-dir")

    v1, port, v1_case, settings = load_case(args.oracle_dir, args.port,
                                             args.oracle or FAITHFUL_ORACLE)
    if not v1 or not port:
        raise SystemExit("one side has no tracks, so there is nothing to compare.")

    # SAY WHICH RUN THIS IS from the file's own metadata rather than its name, since the
    # repaired and unrepaired outputs differ only by a setting and are easy to confuse.
    shown = ", ".join(f"{k}={v}" for k, v in sorted(settings.items())) or "not recorded"
    print(f"case {v1_case}, port file {args.port} ({shown})\n")
    print(f"  version 1: {len(v1)} tracks;  the port: {len(port)} tracks\n")
    print(f"  {'collapse (deg)':>15} {'v1 waves':>9} {'port waves':>11} "
          f"{'matched':>8} {'of v1':>7} {'v1 tracks/wave':>15}")
    for cut in (0.5, 1.0, 2.0, 3.0, 5.0):
        wv = collapse_to_waves(v1, cut)
        wp = collapse_to_waves(port, cut)
        m = match_waves(wv, v1, wp, port)
        print(f"  {cut:15.1f} {len(wv):9d} {len(wp):11d} {len(m):8d} "
              f"{100.0 * len(m) / max(len(wv), 1):6.1f}% "
              f"{len(v1) / max(len(wv), 1):15.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
