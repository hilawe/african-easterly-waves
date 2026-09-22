#!/usr/bin/env python3
"""Compare version 1's internal detection and association state against the port's.

THE QUESTION. The port detects the eastern Africa wave at nine of ten timesteps and its
association still fragments the feature into three tracks, which the prune discards
(THE_EASTERN_AFRICA_WAVE.md). Two candidates were left open: version 1's trough axes may
differ from the port's, or version 1's association may bridge a gap the port's does not.

Both are questions about version 1's INTERNAL state, which its return value does not carry.
`scripts/make_instrumented_v1.py` builds a copy that prints it, and
`scripts/octave/run_tracker_instrumented.m` runs that copy on the same exported case. This
reads the dump and puts the two programs' intermediates side by side at the same timesteps.

WHAT IS COMPARED, at each of the feature's ten timesteps:

  1. THE RAW AXES, count and extent. If version 1's contour parsing yields shorter axes
     than the port's, its candidate positions are better and the divergence is upstream of
     association.
  2. THE MERGED CENTERS, coarse and fine, and how close the nearest one comes to the
     feature. This is what each association actually consumes.
  3. THE LIVE TRACK SET, so version 1's own fragmentation, if any, is visible. The port
     builds three fragments of 3, 7 and 2 observations here. If version 1 also fragments
     and keeps the pieces anyway, the difference is the prune's input, not the association.

WHAT THIS CANNOT SETTLE. The dump records positions, not the association's internal
decisions, so a link version 1 makes and the port does not is visible as an outcome rather
than as a reason. Narrowing further would need the polygon and prediction state printed too.
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from aew.v1port.geometry import great_circle_distance  # noqa: E402

# version 1's track 41, the eastern Africa wave this was first written for. It is the
# DEFAULT and no longer the only case: --feature takes another track's positions, so the
# same instrument serves any pair whose difference is downstream of detection. Added
# 2026-09-21 for pair 30, rather than writing a second tool for the same question.
FEATURE = [(33027.00, -8.25, 33.50), (33027.50, -9.25, 32.67),
           (33027.75, -10.55, 32.10), (33028.25, -11.10, 30.60),
           (33028.50, -11.40, 29.10), (33028.75, -11.50, 27.50),
           (33029.00, -12.20, 26.80), (33029.25, -11.00, 26.40),
           (33029.75, -8.70, 26.80), (33030.25, -7.33, 28.33)]


def read_dump(path):
    """Parse the AXIS / COARSE / FINE / TRACK lines the instrumented copy prints."""
    axes, coarse, fine, tracks = {}, {}, {}, {}
    patterns = {
        "AXIS": re.compile(r"^AXIS ([\d.]+) (\d+) (-?[\d.]+) (-?[\d.]+) "
                           r"(-?[\d.]+) (-?[\d.]+)$"),
        "COARSE": re.compile(r"^COARSE ([\d.]+) (-?[\d.]+) (-?[\d.]+)$"),
        "FINE": re.compile(r"^FINE ([\d.]+) (-?[\d.]+) (-?[\d.]+)$"),
        "TRACK": re.compile(r"^TRACK ([\d.]+) (\d+) (\d+) (-?[\d.]+) (-?[\d.]+) "
                            r"(-?[\d.]+)$"),
    }
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            kind = line.split(" ", 1)[0] if " " in line else ""
            m = patterns.get(kind)
            if m is None:
                continue
            g = m.match(line)
            if g is None:
                continue
            t = float(g.group(1))
            if kind == "AXIS":
                axes.setdefault(t, []).append(
                    (int(g.group(2)), float(g.group(3)), float(g.group(4)),
                     float(g.group(5)), float(g.group(6))))
            elif kind == "COARSE":
                coarse.setdefault(t, []).append((float(g.group(2)), float(g.group(3))))
            elif kind == "FINE":
                fine.setdefault(t, []).append((float(g.group(2)), float(g.group(3))))
            else:
                tracks.setdefault(t, []).append(
                    (int(g.group(2)), int(g.group(3)), float(g.group(4)),
                     float(g.group(5)), float(g.group(6))))
    return axes, coarse, fine, tracks


def nearest_km(lat, lon, points):
    if not points:
        return np.inf
    arr = np.asarray(points, dtype=float)
    return float(np.min(great_circle_distance(lat, lon, arr[:, 0], arr[:, 1])))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", default=None,
                    help="the instrumented run's log (default $AEW_ORACLE_DIR/../instrumented.log)")
    ap.add_argument("--feature", default=None,
                    help="the track to examine, as time,lat,lon triples separated by "
                         "semicolons. Defaults to the eastern Africa wave.")
    ap.add_argument("--label", default="THE EASTERN AFRICA FEATURE",
                    help="what to call the feature in the printed heading")
    ap.add_argument("--port-note", default=None,
                    help="one line stating what the PORT does at these steps, for the "
                         "closing comparison. Without it the closing note is generic, "
                         "because the default one describes the eastern Africa wave only.")
    args = ap.parse_args(argv)
    feature = FEATURE
    if args.feature:
        feature = []
        for chunk in args.feature.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            bits = chunk.split(",")
            if len(bits) != 3:
                raise SystemExit(f"{chunk!r} is not a time,lat,lon triple")
            feature.append(tuple(float(x) for x in bits))
        if not feature:
            raise SystemExit("--feature was given and parsed to nothing")
    dump = args.dump or os.path.join(
        os.path.dirname(os.environ["AEW_ORACLE_DIR"].rstrip("/")), "instrumented.log")
    if not os.path.exists(dump):
        raise SystemExit(f"{dump} is absent. Run run_tracker_instrumented.m first.")
    axes, coarse, fine, tracks = read_dump(dump)
    if not axes:
        raise SystemExit(f"{dump} holds no AXIS lines; the instrumentation did not fire.")

    print(f"VERSION 1'S OWN INTERMEDIATE STATE AT {args.label}\n")
    missing = [t for t, _, _ in feature if t not in axes]
    if missing:
        raise SystemExit(
            f"the dump holds no records at {len(missing)} of the {len(feature)} requested "
            f"steps, first {missing[0]}. Rerun run_tracker_instrumented.m with "
            f"AEW_DUMP_TIMES covering them, rather than reading a table with holes in it.")
    print(f"  {'time':>9} {'axes':>5} {'longest axis':>13} {'median span':>12} | "
          f"{'coarse':>7} {'nearest':>8} | {'fine':>5} {'nearest':>8}")
    for t, la, lo in feature:
        a = axes.get(t, [])
        spans = np.array([x[3] for x in a]) if a else np.array([0.0])
        longest = spans.max() if a else 0.0
        c = coarse.get(t, [])
        f = fine.get(t, [])
        print(f"  {t:9.2f} {len(a):5d} {longest:12.1f}d {np.median(spans):11.1f}d | "
              f"{len(c):7d} {nearest_km(la, lo, c):7.0f}k | "
              f"{len(f):5d} {nearest_km(la, lo, f):7.0f}k")

    print("\n  'longest axis' and 'median span' are degrees of latitude. 'nearest' is how")
    print("  close version 1's nearest merged center comes to the feature, in km.\n")

    print("VERSION 1'S LIVE TRACKS NEAR THE FEATURE, within 500 km at each timestep:\n")
    print(f"  {'time':>9} {'live':>5} {'near':>5} | {'the ones near it: obs, last position'}")
    for t, la, lo in feature:
        live = tracks.get(t, [])
        near = [x for x in live
                if float(great_circle_distance(la, lo, x[2], x[3])) <= 500.0]
        detail = "; ".join(f"{n} obs at ({y:+.1f},{z:+.1f})" for _, n, y, z, _ in near)
        print(f"  {t:9.2f} {len(live):5d} {len(near):5d} | {detail}")

    if args.port_note:
        print(f"\n  {args.port_note}")
    elif feature is FEATURE:
        print("\n  The port builds three fragments here, of 3, 7 and 2 observations, and its")
        print("  prune discards all three because none exceeds eight.")
    print("\n  HOW TO READ THE TRACK ROWS. If version 1 carries ONE track with a growing")
    print("  observation count through these rows where the port builds several short ones,")
    print("  its association is holding together what the port's splits, and the divergence")
    print("  is in association rather than detection. If version 1 ALSO fragments and its")
    print("  prune keeps the pieces, the divergence is in the prune's input instead.")
    print("  THIS DUMP RECORDS POSITIONS, NOT DECISIONS, so a link version 1 makes and the")
    print("  port does not appears here as an outcome and never as a reason.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
