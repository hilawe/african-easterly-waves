#!/usr/bin/env python3
"""Whether the unmatched tracks are each program's own duplicates rather than misses.

THE LEAD THIS FOLLOWS. The first split of the oracle window looked at longitude and found
version 1's unmatched tracks over Africa and the port's over the Atlantic. Two other
columns of that table are sharper. Version 1's unmatched tracks sit a MEDIAN OF 2.0
DEGREES from one of its OWN matched tracks, while the port's sit 9.3 degrees from one of
its own, and the port's unmatched tracks have a median latitude of +19.8 against -0.6 for
everything else.

WHY THE FIRST OF THOSE MATTERS MORE THAN IT LOOKS. The comparison assigns tracks ONE TO
ONE, deliberately, so that one version 1 track cannot absorb several of the port's. If
version 1 emits two tracks for a single wave and the port emits one, then exactly one of
version 1's two is unmatched BY CONSTRUCTION, and the port has not missed anything. The
project has measured version 1's published record at 1.90 tracks per distinct wave, so
this is not a hypothetical mechanism, and a chunk of the 18 percent shortfall may be it
rather than a detection difference.

WHAT IS MEASURED HERE. Duplication WITHIN each program's own output on this window, using
the same distance and overlap rule the cross-program matching uses, so the two numbers are
comparable. Then the unmatched populations are split by whether they are duplicates of a
matched track on their own side.

THIS DOES NOT ESTABLISH WHICH TRACK IS THE REAL WAVE. Two tracks close together may be one
wave counted twice or two genuinely distinct nearby waves, and nothing here separates
those. What it does establish is how much of the unmatched count is structurally forced by
one-to-one assignment against a duplicating reference.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_tracker_oracle import (FAITHFUL_ORACLE, load_case,  # noqa: E402
                                    match,
                                    overlap_separation)

DUPLICATE_DEG = 5.0
DUPLICATE_OVERLAP = 2


def duplicate_partners(tracks, subset=None, against=None):
    """For each track in `subset`, the closest track in `against` that could be it again.

    Returns a list of (index, partner index, shared steps, separation), one per track in
    `subset` that has any partner meeting the overlap rule, with the closest chosen.
    """
    subset = range(len(tracks)) if subset is None else subset
    against = range(len(tracks)) if against is None else against
    out = []
    for i in subset:
        best = None
        for j in against:
            if i == j:
                continue
            n, sep = overlap_separation(tracks[i], tracks[j])
            if n >= DUPLICATE_OVERLAP and (best is None or sep < best[2]):
                best = (j, n, sep)
        if best is not None:
            out.append((i, best[0], best[1], best[2]))
    return out


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
        raise SystemExit("one side has no tracks, so there is nothing to compare.")

    assigned, _ = match(v1, port)
    matched_v1 = {i for i, _, _, _ in assigned}
    matched_port = {j for _, j, _, _ in assigned}
    sides = (("version 1", v1, matched_v1), ("the port", port, matched_port))

    print(f"case {v1_case}\n")
    print(f"DUPLICATION WITHIN EACH PROGRAM'S OWN OUTPUT, a partner within "
          f"{DUPLICATE_DEG:.0f} degrees")
    print(f"over at least {DUPLICATE_OVERLAP} shared timesteps:\n")
    print(f"  {'':>12} {'tracks':>7} {'with a partner':>15} {'rate':>7} "
          f"{'median sep':>11}")
    for name, tracks, _ in sides:
        partners = duplicate_partners(tracks)
        close = [p for p in partners if p[3] <= DUPLICATE_DEG]
        seps = np.array([p[3] for p in close]) if close else np.array([])
        rate = 100.0 * len(close) / max(len(tracks), 1)
        med = f"{np.median(seps):.1f}" if seps.size else "n/a"
        print(f"  {name:>12} {len(tracks):7d} {len(close):15d} {rate:6.1f}% {med:>11}")

    print(f"\nTHE UNMATCHED POPULATIONS, split by whether they duplicate a track that DID")
    print(f"match on their own side. A track in the first row is one the other program")
    print(f"has once and this program has twice, so one-to-one assignment must leave it")
    print(f"over, and it is NOT evidence of a missed wave.\n")
    for name, tracks, matched in sides:
        unmatched = [i for i in range(len(tracks)) if i not in matched]
        partners = duplicate_partners(tracks, subset=unmatched, against=matched)
        dup = [p for p in partners if p[3] <= DUPLICATE_DEG]
        dup_idx = {p[0] for p in dup}
        rest = [i for i in unmatched if i not in dup_idx]
        print(f"  {name}: {len(unmatched)} unmatched")
        print(f"    duplicates of one of its own MATCHED tracks: {len(dup):3d}")
        print(f"    not explained that way:                      {len(rest):3d}")
        if rest:
            lat = np.array([float(np.mean(tracks[i]["lat"])) for i in rest])
            lon = np.array([float(np.mean(tracks[i]["lon"])) for i in rest])
            ln = np.array([tracks[i]["time"].size for i in rest], dtype=float)
            print(f"    those {len(rest)}: median lat {np.median(lat):+5.1f}, "
                  f"lon {np.median(lon):+7.1f}, length {np.median(ln):.0f} steps")
        print()

    print("WHAT THE HEADLINE BECOMES. Removing only the structurally forced cases:\n")
    forced_v1 = len([p for p in duplicate_partners(
        v1, subset=[i for i in range(len(v1)) if i not in matched_v1],
        against=matched_v1) if p[3] <= DUPLICATE_DEG])
    real_misses = len(v1) - len(matched_v1) - forced_v1
    denom = len(v1) - forced_v1
    print(f"  version 1 tracks                                     {len(v1):4d}")
    print(f"  of which duplicates of another version 1 track the")
    print(f"  port matched, so unmatchable one to one               {forced_v1:4d}")
    print(f"  version 1 tracks the port genuinely does not have     {real_misses:4d}")
    print(f"\n  reproduced, counting every version 1 track:          "
          f"{100.0 * len(matched_v1) / len(v1):.1f}%")
    print(f"  reproduced, excluding the forced duplicates:          "
          f"{100.0 * len(matched_v1) / max(denom, 1):.1f}%")
    print(f"\nTHE SECOND FIGURE IS NOT A BETTER SCORE, it is a different question: how much")
    print(f"of the shortfall is a detection difference rather than an artifact of comparing")
    print(f"one to one against a reference that counts some waves twice. Both belong in any")
    print(f"write-up, and neither says which of a duplicate pair is the real wave.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
