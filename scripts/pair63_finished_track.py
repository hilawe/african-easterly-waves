#!/usr/bin/env python3
"""Carry pair 63's reordering through association to a FINISHED TRACK.

Stage 2 is not stage 5. `scripts/pair63_merge_input.py` showed that moving one candidate
ahead of its remover restores the coarse-merge candidate at version 1's exact position, and
that is a detected candidate, not an observation and not a track. This runs the port's full
pipeline over the whole window three times and asks what the FINISHED tracks hold.

    baseline        the port untouched
    intervention    at 33030.50 ONLY, the feature's axis moved ahead of its remover's
    control         at 33030.50 ONLY, an unrelated axis moved, the two left in their order

The control is what makes the intervention mean anything. It performs a reordering of the
same kind at the same step without changing the relative order of the pair, so a difference
between it and the baseline would be reordering as such rather than this reordering.

NOTHING BUT ORDER CHANGES. No axis is added, removed or moved in space. The hook reorders
the list `trough_axes` returns and nothing else, at one timestep and no other.

WHAT A FINISHED OBSERVATION IS. `finalize_tracks` applies version 1's `smooth(x,5)`, so a
finished position is a running mean of up to five raw centers and generally does NOT equal
the detected candidate. The comparison below is against version 1's own FINISHED tracks in
the retained reference output, which carry the same smoothing.

    .venv/bin/python3 scripts/pair63_finished_track.py
"""

import argparse
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

CASE = "docs/aewc_v2/evidence/tracker_case.mat"
STEP = 33030.5
FEATURE = (-14.158, -20.973)
REMOVER = (-12.990, -15.240)
# What version 1's FINISHED track holds at this step and the port's does not.
#
# THIS IS NOT (-13.0, -19.0), and the difference is the whole reason this script exists.
# (-13.0, -19.0) is version 1's FINE-MERGE CANDIDATE here, a stage 3 position. Its finished
# track sits at (-15.8, -19.8), because finalize_tracks applies smooth(x,5) and a finished
# position is a running mean of up to five raw centers. A first version of this script
# compared the finished tracks against the stage 3 position and reported the observation
# ABSENT in every run including version 1's own, which is the same stage confusion that has
# already cost this project one published claim.
MISSING = (-15.8, -19.8)
TOLERANCE = 0.6


def nearest_index(axes, target):
    d = [np.hypot(np.mean(a) - target[0], np.mean(b) - target[1]) for a, b in axes]
    return int(np.argmin(d)), float(min(d))


def reorder_hook(original, now, mode, record):
    """Reorder the axis list at STEP only. `mode` is 'none', 'pair' or 'control'."""
    def hook(latgrid, longrid, advection, level=D.TROUGH_LEVEL):
        out = list(original(latgrid, longrid, advection, level))
        if mode == "none" or now[0] is None or abs(now[0] - STEP) > 1e-6:
            return out
        feature, df = nearest_index(out, FEATURE)
        remover, dr = nearest_index(out, REMOVER)
        if df > 0.2 or dr > 0.2:
            raise SystemExit(f"the pair is not where this expects it at {STEP}: the feature "
                             f"is {df:.3f} deg away and the remover {dr:.3f} deg. Refusing "
                             f"rather than reordering something else.")
        if mode == "pair":
            if feature < remover:
                raise SystemExit("the feature already precedes its remover, so there is "
                                 "nothing for this intervention to change")
            out.insert(remover, out.pop(feature))
            record.update({"applied": True, "moved_from": feature, "moved_to": remover})
        else:
            # An unrelated axis, chosen so it is neither of the pair and so moving it cannot
            # change their relative order.
            spare = next(k for k in range(len(out)) if k not in (feature, remover)
                         and k > max(feature, remover))
            out.insert(0, out.pop(spare))
            after_f, _ = nearest_index(out, FEATURE)
            after_r, _ = nearest_index(out, REMOVER)
            if (after_f < after_r) != (feature < remover):
                raise SystemExit("the control changed the pair's relative order, which is "
                                 "the one thing it must not do")
            record.update({"applied": True, "moved_from": spare, "moved_to": 0})
        return out
    return hook


def run(case, mode):
    """The port's full pipeline over the window, returning its finished tracks."""
    times = case["time"].ravel()
    lat_c, lon_c = case["lat_c"].ravel(), case["lon_c"].ravel()
    latgrid_c, longrid_c = np.meshgrid(lat_c, lon_c, indexing="ij")
    coarse_threshold, fine_threshold = P.thresholds_for("ERA-Int", 700)
    original, now, record = D.trough_axes, [None], {"applied": False}
    D.trough_axes = reorder_hook(original, now, mode, record)
    tracks, states = [], []
    try:
        for step in range(times.size):
            t = float(times[step])
            now[0] = t
            waves = D.detect_troughs(
                t, latgrid_c, longrid_c, case["u_c"][step], case["currv_anom_c"][step],
                case["advcurrv_anom_c"][step], case["latgrid"], case["longrid"],
                case["currv_anom"][step], coarse_threshold=coarse_threshold,
                fine_threshold=fine_threshold, absorb=False)
            um = P._median_over(clim.smooth9(case["u"][step]))
            vm = P._median_over(clim.smooth9(case["v"][step]))
            tracks, states = associate_step(tracks, states, waves, step, um, vm,
                                            exclusive=False)
            tracks, states = prune_stale_tracks(tracks, states, step, t, waves)
    finally:
        D.trough_axes = original
    if mode != "none" and not record["applied"]:
        raise SystemExit(f"the {mode} reordering was never applied, so this run is not the "
                         f"experiment it claims to be")
    return finalize_tracks(tracks, total_steps=times.size), record


def candidates_at_step(case, mode, when=STEP):
    """The DETECTED candidates at one step, which is stage 3 and not the finished track."""
    times = case["time"].ravel()
    index = int(np.argmin(np.abs(times - when)))
    lat_c, lon_c = case["lat_c"].ravel(), case["lon_c"].ravel()
    latgrid_c, longrid_c = np.meshgrid(lat_c, lon_c, indexing="ij")
    coarse_threshold, fine_threshold = P.thresholds_for("ERA-Int", 700)
    original, now, record = D.trough_axes, [when], {"applied": False}
    D.trough_axes = reorder_hook(original, now, mode, record)
    try:
        waves = D.detect_troughs(
            when, latgrid_c, longrid_c, case["u_c"][index], case["currv_anom_c"][index],
            case["advcurrv_anom_c"][index], case["latgrid"], case["longrid"],
            case["currv_anom"][index], coarse_threshold=coarse_threshold,
            fine_threshold=fine_threshold, absorb=False)
    finally:
        D.trough_axes = original
    return [(w["lat_mean"], w["lon_mean"]) for w in waves]


def at_step(final, when=STEP):
    """Every finished observation at one timestep, with the length of its track.

    THE TIMESTAMP TEST IS EXACT. It was `abs(tt - when) < 1e-6`, which is a tolerance
    standing in for a comparison nobody had checked. Version 1's finished times are float64
    values present exactly in the case's own time array, so the tolerance bought nothing and
    hid the question of whether the two runs were being lined up on the same step at all.
    """
    out = []
    for tr in final:
        for tt, la, lo in zip(tr["time"], tr["meanlat"], tr["meanlon"]):
            if float(tt) == when:
                out.append((float(la), float(lo), len(tr["time"])))
    return sorted(out)


def version1_at_step(path, when=STEP):
    """Every version 1 finished observation at one timestep. The timestamp test is EXACT."""
    ref = loadmat(path)
    n = int(np.asarray(ref["n"]).ravel()[0])
    out = []
    for i in range(n):
        tt = np.asarray(ref[f"time{i}"], dtype=float).ravel()
        la = np.asarray(ref[f"lat{i}"], dtype=float).ravel()
        lo = np.asarray(ref[f"lon{i}"], dtype=float).ravel()
        for a, b, c in zip(tt, la, lo):
            if float(a) == when:
                out.append((float(b), float(c), len(tt)))
    return sorted(out), n


def version1_track_containing(path, when, near, span=TOLERANCE):
    """Version 1's finished track holding an observation at `when` near `near`.

    THE WINDOW HERE IDENTIFIES A TRACK, IT DOES NOT DECIDE ANYTHING. Picking which of
    version 1's 123 tracks this case is about is a lookup. The verdict is whether a port run
    reproduces that track's COMPLETE arrays exactly, which is `exact_tracks.holds_exactly`
    and applies no tolerance at all. Keeping the two apart is the point: an earlier version
    of this file used a 0.6 degree radius and `np.allclose` in the verdict itself.
    """
    ref = loadmat(path)
    for i in range(int(np.asarray(ref["n"]).ravel()[0])):
        tt = np.asarray(ref[f"time{i}"], dtype=float).ravel()
        la = np.asarray(ref[f"lat{i}"], dtype=float).ravel()
        lo = np.asarray(ref[f"lon{i}"], dtype=float).ravel()
        for a, b, c in zip(tt, la, lo):
            if float(a) == when and abs(b - near[0]) <= span and abs(c - near[1]) <= span:
                return i, tt, la, lo
    raise SystemExit(f"no version 1 track holds an observation at {when} near {near}")


def nearest_to(rows, target):
    """Reporting only. How close the nearest row comes, never a verdict."""
    if not rows:
        return None, float("nan")
    d = [float(np.hypot(a - target[0], b - target[1])) for a, b, _ in rows]
    k = int(np.argmin(d))
    return rows[k], d[k]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", default=CASE)
    ap.add_argument("--retain", default=None,
                    help="write the complete trajectories of every run, with the identities "
                         "they were judged against, to this JSON path")
    args = ap.parse_args(argv)
    raw = loadmat(args.case)
    case = {k: np.asarray(v) for k, v in raw.items() if not k.startswith("__")}

    ref_path = X.reference_output()
    v1rows, v1n = version1_at_step(ref_path)
    idx, vt, vl, vo = version1_track_containing(ref_path, STEP, MISSING)
    row, gap = nearest_to(v1rows, MISSING)
    print(f"version 1's finished tracks, from {os.path.basename(ref_path)}: {v1n} tracks, "
          f"{len(v1rows)} observations at {STEP}")
    print(f"  the reference track is index {idx}, {len(vt)} observations, "
          f"{vt.min():.2f} to {vt.max():.2f}, holding ({row[0]:.3f}, {row[1]:.3f}) here")

    print(f"\nSTAGE 3, the detected candidate at {STEP}, against version 1's (-13.0, -19.0)")
    for mode, label in (("none", "baseline"), ("pair", "intervention"), ("control", "control")):
        pts = candidates_at_step(case, mode)
        d = [float(np.hypot(a + 13.0, b + 19.0)) for a, b in pts]
        k = int(np.argmin(d))
        print(f"  {label:13s} {len(pts):3d} candidates, "
              f"{'present at (%.3f, %.3f)' % pts[k] if d[k] <= TOLERANCE else 'absent'}"
              f"  (nearest {d[k]:.2f} deg; stage 3 is reported, not judged)")

    print("\nSTAGE 5, the finished tracks")
    finals, records = {}, {}
    for mode, label in (("none", "baseline"), ("pair", "intervention"), ("control", "control")):
        final, record = run(case, mode)
        finals[label], records[label] = final, record
        rows = at_step(final)
        row, gap = nearest_to(rows, MISSING)
        extra = "" if mode == "none" else f", reordering applied at index {record['moved_from']}"
        print(f"  {label:13s} {len(final):3d} tracks, {len(rows):2d} observations at {STEP}"
              f"{extra}; nearest to the reference observation {gap:.2f} deg")

    # THE CONTROL IS A GATE, NOT A ROW IN A TABLE. It must reproduce the baseline's COMPLETE
    # trajectories exactly, every track and every observation, or there is nothing to report.
    X.require_identical(finals["baseline"], finals["control"], "the baseline", "the control")
    print("\n  THE CONTROL REPRODUCES THE BASELINE EXACTLY, over all "
          f"{len(finals['baseline'])} finished tracks and every observation in them.")

    verdicts = {}
    for label in ("baseline", "control", "intervention"):
        verdicts[label] = X.holds_exactly(finals[label], vt, vl, vo)
        print(f"  {label:13s} reproduces version 1's complete track {idx} exactly: "
              f"{verdicts[label]}")

    ok = verdicts["intervention"] and not verdicts["baseline"] and not verdicts["control"]
    print(f"\n  EXACT REPRODUCTION BY THE INTERVENTION ALONE: {ok}")
    if args.retain:
        identity = X.save_run(args.retain, finals, args.case, ref_path, idx, __file__,
                              extra={"step": STEP, "intervention": "reorder one axis ahead "
                                     "of its remover at one step",
                                     "exact_reproduction_by_intervention_alone": ok})
        print(f"  retained {len(finals)} runs to {args.retain}")
        print(f"    case {identity['case_sha256'][:12]}, reference "
              f"{identity['reference_output_sha256'][:12]}, track index {idx}")
    if not ok:
        return 1
    print("\n  WHAT THIS SUPPORTS, and no more. A reordering of one axis at one step is "
          "SUFFICIENT\n  to reproduce this track in this window. It does not show the "
          "port's order is wrong,\n  nor that any other stage of the two programs agrees.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
