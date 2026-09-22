#!/usr/bin/env python3
"""Pair 30, substituting version 1's captured merge input at 33026.50 and then narrowing.

THE EXPERIMENT CONTRACT, stated here rather than cited. Measurement is the FINISHED TRACK,
compared against a pinned reference track by exact float equality over complete arrays. Phase
one substitutes version 1's whole captured merge input at one step and claims only that the
input is sufficient. Phase two narrows to a single axis. The representation control must
reproduce the baseline exactly or the run refuses; the narrow control must fail to restore
the reference track. A restored candidate is not an observation and not a track.

WHAT IS BEING TESTED. The port's fragment takes an extra observation at 33026.50 near
(-1.0, -84.0) that version 1's track does not take, which strands it and gets it pruned.
Version 1's own merge discards the candidate it holds nearest that spot. The merge code is
shared and verified identical, so the difference is the candidate list.

HOW THE SUBSTITUTION WORKS. The port sets each candidate's position to the MEAN of its axis
vertices and uses the vertices for nothing else, so a candidate list is injected as one
SINGLE-VERTEX axis per candidate. `--check` verifies that representation by re-expressing
the port's OWN candidates that way and requiring detect_troughs to return exactly what it
returns untouched.

    .venv/bin/python3 scripts/pair30_merge_input.py --log <dir>/pair30b.log
"""

import argparse
import glob
import os
import sys

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aew.v1port import climatology as clim  # noqa: E402
from aew.v1port import detection as D  # noqa: E402
from aew.v1port import pipeline as P  # noqa: E402
from aew.v1port.association import (associate_step, finalize_tracks,  # noqa: E402
                                    prune_stale_tracks)
from aew.v1port.geometry import great_circle_distance as gc  # noqa: E402

CASE = "docs/aewc_v2/evidence/tracker_case.mat"
STEP = 33026.5
# The port's extra candidate, the one version 1's merge does not keep.
EXTRA = (-1.0, -84.0)
# The ONE axis whose removal alone makes that candidate disappear, found by leave-one-out
# over all 75 axes at this step. Its centroid is 480 km from where its merged candidate ends
# up, which is why filtering axes by proximity to the merged position finds nothing: pass 1
# replaces the centroid with the median of the region it seeds.
CULPRIT = (-3.71, -87.37)
# An unrelated axis, for the narrowed run's control. Removing it must NOT restore the track.
# It is a REAL AXIS CENTROID of similar size, 5 vertices against the culprit's 4, taken from
# the same step. A first version used a merged candidate's position here and the guard
# refused it, correctly: an axis centroid and a merged center are different quantities, which
# is the same distinction that made the culprit unfindable by proximity to its own output.
SPARE = (-4.28, -74.26)
# Version 1's track 30, which the port's fragment fails to become.
V1_START = 33025.25


def version1_candidates(path, when=STEP):
    rows = []
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) > 4 and parts[0] == "POTWV" and abs(float(parts[1]) - when) < 1e-6:
                rows.append((int(parts[2]), float(parts[3]), float(parts[4])))
    rows.sort()
    if not rows:
        raise SystemExit(f"{path} holds no POTWV records at {when}")
    if [r[0] for r in rows] != list(range(1, len(rows) + 1)):
        raise SystemExit("the captured indices are not 1..n, so the order cannot be trusted")
    return [(r[1], r[2]) for r in rows]


def as_axes(points):
    """One single-vertex axis per candidate, whose mean is that candidate's position."""
    return [(np.array([a], dtype=float), np.array([b], dtype=float)) for a, b in points]


def hook(original, now, mode, v1points, record):
    def spy(latgrid, longrid, advection, level=D.TROUGH_LEVEL):
        out = list(original(latgrid, longrid, advection, level))
        if mode == "none" or now[0] is None or abs(now[0] - STEP) > 1e-6:
            return out
        if mode == "control":
            replaced = as_axes([(float(np.mean(a)), float(np.mean(b))) for a, b in out])
        elif mode == "v1":
            replaced = as_axes(v1points)
        elif mode in ("drop", "dropother"):
            target = CULPRIT if mode == "drop" else SPARE
            gaps = [float(gc(target[0], target[1], np.mean(a), np.mean(b), "km"))
                    for a, b in out]
            k = int(np.argmin(gaps))
            if gaps[k] > 60.0:
                raise SystemExit(f"no axis within 60 km of {target}, nearest {gaps[k]:.0f} km, "
                                 f"so this run would be the baseline under another name")
            replaced = [a for j, a in enumerate(out) if j != k]
        else:
            raise SystemExit(f"unknown mode {mode}")
        record.update({"applied": True, "before": len(out), "after": len(replaced)})
        return replaced
    return spy


def run(case, mode, v1points=None):
    times = case["time"].ravel()
    lat_c, lon_c = case["lat_c"].ravel(), case["lon_c"].ravel()
    latgrid_c, longrid_c = np.meshgrid(lat_c, lon_c, indexing="ij")
    ct, ft = P.thresholds_for("ERA-Int", 700)
    original, now, record = D.trough_axes, [None], {"applied": False}
    D.trough_axes = hook(original, now, mode, v1points, record)
    tracks, states = [], []
    try:
        for step in range(times.size):
            t = float(times[step])
            now[0] = t
            waves = D.detect_troughs(
                t, latgrid_c, longrid_c, case["u_c"][step], case["currv_anom_c"][step],
                case["advcurrv_anom_c"][step], case["latgrid"], case["longrid"],
                case["currv_anom"][step], coarse_threshold=ct, fine_threshold=ft,
                absorb=False)
            um = P._median_over(clim.smooth9(case["u"][step]))
            vm = P._median_over(clim.smooth9(case["v"][step]))
            tracks, states = associate_step(tracks, states, waves, step, um, vm,
                                            exclusive=False)
            tracks, states = prune_stale_tracks(tracks, states, step, t, waves)
    finally:
        D.trough_axes = original
    if mode != "none" and not record["applied"]:
        raise SystemExit(f"the {mode} substitution never fired")
    return finalize_tracks(tracks, total_steps=times.size), record


def check_representation(case):
    """Collapsing an axis to its centroid must change nothing, or every number is an artifact."""
    times = case["time"].ravel()
    index = int(np.argmin(np.abs(times - STEP)))
    lat_c, lon_c = case["lat_c"].ravel(), case["lon_c"].ravel()
    latgrid_c, longrid_c = np.meshgrid(lat_c, lon_c, indexing="ij")
    ct, ft = P.thresholds_for("ERA-Int", 700)
    args = (STEP, latgrid_c, longrid_c, case["u_c"][index], case["currv_anom_c"][index],
            case["advcurrv_anom_c"][index], case["latgrid"], case["longrid"],
            case["currv_anom"][index])
    kw = dict(coarse_threshold=ct, fine_threshold=ft, absorb=False)
    plain = D.detect_troughs(*args, **kw)
    original, now, record = D.trough_axes, [STEP], {"applied": False}
    D.trough_axes = hook(original, now, "control", None, record)
    try:
        collapsed = D.detect_troughs(*args, **kw)
    finally:
        D.trough_axes = original
    same = len(plain) == len(collapsed) and all(
        a["lat_mean"] == b["lat_mean"] and a["lon_mean"] == b["lon_mean"]
        for a, b in zip(plain, collapsed))
    return same, len(plain), len(collapsed)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", required=True, help="the instrumented dump holding POTWV at 33026.50")
    ap.add_argument("--narrow", action="store_true",
                    help="also run phase two, dropping only the port's extra candidate")
    ap.add_argument("--retain", default=None,
                    help="write the complete trajectories of every run, with the identities "
                         "they were judged against, to this JSON path")
    args = ap.parse_args(argv)

    raw = loadmat(CASE)
    case = {k: np.asarray(v) for k, v in raw.items() if not k.startswith("__")}
    v1points = version1_candidates(args.log)
    ref_path = X.reference_output()
    idx, vt, vl, vo = X.reference_track(ref_path, V1_START, -4.0, -78.5)

    ok, n_plain, n_collapsed = check_representation(case)
    print(f"REPRESENTATION CHECK: collapsing the port's own axes to their centroids gives "
          f"{n_collapsed} candidates against {n_plain} untouched, identical: {ok}")
    if not ok:
        raise SystemExit("the single-vertex representation is not faithful, so every "
                         "substitution below would be an artifact of it")
    print(f"version 1's captured merge input at {STEP}: {len(v1points)} candidates")
    print(f"reference: {os.path.basename(ref_path)}, track {idx}, {len(vt)} observations, "
          f"{vt.min():.2f} to {vt.max():.2f}\n")

    modes = [("none", "baseline"), ("control", "control"), ("v1", "diagnostic")]
    if args.narrow:
        modes += [("drop", "narrowed"), ("dropother", "narrow ctrl")]
    finals = {}
    print("PHASE ONE, the real-data diagnostic")
    for mode, label in modes:
        if mode == "drop":
            print("\nPHASE TWO, the narrowed intervention")
        final, record = run(case, mode, v1points)
        finals[label] = final
        note = (f"{record['before']} axes -> {record['after']}") if record["applied"] else ""
        print(f"  {label:13s} {len(final):3d} tracks  {note}")

    # THE REPRESENTATION CONTROL IS A GATE. Re-expressing the port's own axes must leave the
    # WHOLE run untouched, every track and every observation, or no substitution below can be
    # attributed to version 1's positions rather than to the representation.
    X.require_identical(finals["baseline"], finals["control"], "the baseline", "the control")
    print(f"\n  THE CONTROL REPRODUCES THE BASELINE EXACTLY, over all "
          f"{len(finals['baseline'])} finished tracks and every observation in them.")

    verdicts = {}
    for label in finals:
        verdicts[label] = X.holds_exactly(finals[label], vt, vl, vo)
        print(f"  {label:13s} reproduces version 1's complete track {idx} exactly: "
              f"{verdicts[label]}")

    # THE NARROW CONTROL IS NOT REQUIRED TO REPRODUCE THE BASELINE WHOLE, because removing a
    # different axis may legitimately change other tracks at that step. What it must not do
    # is restore THIS track, and that is checked rather than assumed.
    ok = (verdicts["diagnostic"] and not verdicts["baseline"] and not verdicts["control"])
    if "narrowed" in verdicts:
        if verdicts.get("narrow ctrl"):
            raise SystemExit("REFUSED: the narrow control restored the reference track, so "
                             "removing any axis does it and the narrowed result is "
                             "unattributable.")
        ok = ok and verdicts["narrowed"]
    print(f"\n  EXACT REPRODUCTION BY THE INTERVENTIONS, WITH EVERY CONTROL FAILING: {ok}")

    if args.retain:
        identity = X.save_run(args.retain, finals, CASE, ref_path, idx, __file__,
                              extra={"step": STEP,
                                     "interventions": ["substitute version 1's captured merge "
                                                       "input at one step",
                                                       "remove one axis at one step"],
                                     "exact_reproduction_with_controls_failing": ok})
        print(f"  retained {len(finals)} runs to {args.retain}")
        print(f"    case {identity['case_sha256'][:12]}, reference "
              f"{identity['reference_output_sha256'][:12]}, track index {idx}")
    if not ok:
        return 1
    print("\n  WHAT THIS SUPPORTS, and no more. Removing one axis at one step is SUFFICIENT "
          "to\n  reproduce this track in this window. It does not show the port is wrong to "
          "draw it,\n  nor that any other stage of the two programs agrees.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
