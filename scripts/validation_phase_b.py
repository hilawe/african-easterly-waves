#!/usr/bin/env python3
"""Phase B of the validation: the case investigations, in the fixed order and budget.

THE CONTRACT THIS IMPLEMENTS, the phase B amendment of 2026-09-22, in substance. Every
nonidentical pair is admitted whatever its category, reference-extra, port-extra, both
sides or displaced only, and phase A's reference-extra rule is not merely mirrored. Two
kinds of time are kept apart: DISCREPANCY TIMES are where the two finished outputs differ
and are derived from retained evidence, and an INTERVENTION TIME is where an experiment
acts, is the case's own, and carries a recorded basis. An outcome is EXPLAINED only by
exact whole-array reproduction of the reference track with a representation control that
reproduces the baseline and a negative control that does not reproduce the reference,
UNSUPPORTED when no valid negative control exists for the operation, and UNEXPLAINED
WITHIN BUDGET otherwise.

WHAT THIS AUTOMATES, AND THE LIMIT THAT MATTERS MOST. The budget allows three intervention
kinds per item, each attempted ONCE. An "attempt" here is a DEFINED SEARCH, not a hand-built
experiment, and that is the honest reading of a fixed budget applied to 48 items. So an item
reported UNEXPLAINED WITHIN BUDGET means THIS SEARCH did not construct an intervention that
worked. It does NOT mean no intervention exists. Pair 30's removal took a human following a
pruned fragment to a step that appears in none of its discrepancy sets, and no search
specified in advance would have gone there. The distinction is reported rather than smoothed.

THE SEARCH WINDOW, declared here because the amendment left step 5's mechanics open. For
removal and reordering the steps tried are the item's discrepancy times together with the
two steps either side of each, which is what contains pair 30's 33026.50 against
discrepancies at 33025.25 to 33026.25. Widening it after seeing a result would be tuning.

    .venv/bin/python3 scripts/validation_phase_b.py <work-dir> --window A --out <artifact>
"""

import argparse
import collections
import time
import glob
import gzip
import json
import os
import sys

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import exact_tracks as X  # noqa: E402

from aew.v1port import climatology as clim  # noqa: E402
from aew.v1port import detection as D  # noqa: E402
from aew.v1port import pipeline as P  # noqa: E402
from aew.v1port.association import (associate_step, finalize_tracks,  # noqa: E402
                                    prune_stale_tracks)
from aew.v1port.geometry import great_circle_distance as gc  # noqa: E402

WINDOWS = {"A": (1983, "jul"), "B": (1983, "sep"), "C": (1995, "jul"),
           "D": (1995, "sep"), "E": (2007, "jul"), "F": (2007, "sep")}
PER_WINDOW_BUDGET = 8
NEIGHBORHOOD = 2          # steps either side of a discrepancy time, declared in advance
INJECT_RADIUS_DEG = 5.0   # an injected axis must come within this of the reference position
REORDER_TRIALS = 400      # a cap, declared in advance, so one item cannot eat the budget


def load_window(work, name):
    year, month = WINDOWS[name]
    directory = os.path.join(work, f"{name}_{year}_{month}")
    raw = loadmat(os.path.join(directory, "tracker_case.mat"))
    case = {k: np.asarray(v) for k, v in raw.items() if not k.startswith("__")}
    residuals = json.load(open(os.path.join(directory, "residuals.json")))
    reference = loadmat(os.path.join(directory, "tracker_octave_instrumented.mat"))
    return directory, case, residuals, reference


def reference_track(reference, index):
    return (np.asarray(reference[f"time{index}"], float).ravel(),
            np.asarray(reference[f"lat{index}"], float).ravel(),
            np.asarray(reference[f"lon{index}"], float).ravel())


# WHERE THE TIME GOES, counted at the two calls that cost anything: a full replay of the
# window and a detection screen at one step. Windows B to F cost fifteen times their
# estimate and the record could not say how much of that was screening and how much was
# replay, so a benchmark of an exhausted item was asked for with the two reported
# separately. `investigate` snapshots these before and after each item.
COST = {"replay_calls": 0, "replay_seconds": 0.0, "screen_calls": 0, "screen_seconds": 0.0}


def _cost_snapshot():
    return dict(COST)


def _cost_since(before):
    return {k: (round(COST[k] - before[k], 3) if isinstance(COST[k], float)
                else COST[k] - before[k]) for k in COST}


def replay(case, hook=None):
    """The port's full pipeline over the window, with an optional axis hook."""
    started = time.process_time()
    try:
        return _replay(case, hook)
    finally:
        COST["replay_calls"] += 1
        COST["replay_seconds"] += time.process_time() - started


def _replay(case, hook=None):
    times = case["time"].ravel()
    lat_c, lon_c = case["lat_c"].ravel(), case["lon_c"].ravel()
    latgrid, longrid = np.meshgrid(lat_c, lon_c, indexing="ij")
    ct, ft = P.thresholds_for("ERA-Int", 700)
    original, now = D.trough_axes, [None]
    if hook is not None:
        D.trough_axes = hook(original, now)
    tracks, states = [], []
    try:
        for step in range(times.size):
            t = float(times[step])
            now[0] = t
            waves = D.detect_troughs(
                t, latgrid, longrid, case["u_c"][step], case["currv_anom_c"][step],
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
    return finalize_tracks(tracks, total_steps=times.size)


def axes_at(case, when):
    """The port's own axes at one step, for a search that does not replay the pipeline."""
    index = int(np.argmin(np.abs(case["time"].ravel() - when)))
    lat_c = case["lat_c"].ravel()
    latgrid, longrid = np.meshgrid(lat_c, case["lon_c"].ravel(), indexing="ij")
    ct, _ = P.thresholds_for("ERA-Int", 700)
    wind = D.smooth9(np.asarray(case["u_c"][index], float))
    curvature = D._prepare(np.asarray(case["currv_anom_c"][index], float), lat_c)
    advection = D.smooth9(np.asarray(case["advcurrv_anom_c"][index], float))
    westerly = wind > D.MAX_ZONAL_WIND
    advection = np.where(westerly, np.nan, advection)
    curvature = np.where(westerly, np.nan, curvature)
    advection = np.where(curvature < ct, np.nan, advection)
    return D.trough_axes(latgrid, longrid, advection)


def edit_hook(when, edit):
    """An axis hook that applies `edit` at one step and at no other."""
    def make(original, now):
        def spy(latgrid, longrid, advection, level=D.TROUGH_LEVEL):
            out = list(original(latgrid, longrid, advection, level))
            if now[0] is not None and float(now[0]) == when:
                return edit(out)
            return out
        return spy
    return make


def search_steps(discrepancy, times):
    """The discrepancy times plus NEIGHBORHOOD steps either side, declared in advance."""
    ordered = sorted(float(t) for t in times)
    wanted = set()
    for key in discrepancy:
        value = float(key)
        near = int(np.argmin(np.abs(np.asarray(ordered) - value)))
        for k in range(max(0, near - NEIGHBORHOOD),
                       min(len(ordered), near + NEIGHBORHOOD + 1)):
            wanted.add(ordered[k])
    return sorted(wanted)


def candidate_signature(waves):
    """Everything downstream consumes, IN ORDER, so the screen cannot miss a real change.

    A FIRST VERSION RETURNED `sorted((lat_mean, lon_mean))` AND WAS WRONG TWICE OVER. It
    SORTED, so an edit that changed only the ORDER of the candidates looked like no change,
    and association iterates them in order. And it kept only the centers, so an edit that
    changed a candidate's SPATIAL FOOTPRINT while leaving its center alone looked like no
    change, and association builds its hull polygons from `lat_wave` and `lon_wave` and
    tests membership against them. Either would make the screen report "cannot change
    anything downstream" about an edit that changes plenty, which turns a search into a
    generator of false negatives.

    The signature is therefore the ordered tuple of every field association reads."""
    out = []
    for w in waves:
        out.append((float(w["time"]), float(w["lat_mean"]), float(w["lon_mean"]),
                    tuple(float(x) for x in np.asarray(w["lat_wave"], float).ravel()),
                    tuple(float(x) for x in np.asarray(w["lon_wave"], float).ravel())))
    return out


def detect_at(case, when, edit=None):
    """The detected candidates at one step, optionally with the axis list edited."""
    index = int(np.argmin(np.abs(case["time"].ravel() - when)))
    lat_c = case["lat_c"].ravel()
    latgrid, longrid = np.meshgrid(lat_c, case["lon_c"].ravel(), indexing="ij")
    ct, ft = P.thresholds_for("ERA-Int", 700)
    original = D.trough_axes
    if edit is not None:
        D.trough_axes = lambda *a, **k: edit(list(original(*a, **k)))
    started = time.process_time()
    try:
        waves = D.detect_troughs(
            float(when), latgrid, longrid, case["u_c"][index], case["currv_anom_c"][index],
            case["advcurrv_anom_c"][index], case["latgrid"], case["longrid"],
            case["currv_anom"][index], coarse_threshold=ct, fine_threshold=ft, absorb=False)
    finally:
        D.trough_axes = original
        COST["screen_calls"] += 1
        COST["screen_seconds"] += time.process_time() - started
    return candidate_signature(waves)


def attempt_removal(case, target, steps, baseline):
    """Leave-one-out at the declared steps, SCREENED AT DETECTION, then replayed.

    A full replay per candidate axis is about nine seconds and there are of order a hundred
    axes a step, so screening matters. An axis whose removal leaves the DETECTED candidate
    set unchanged cannot change the finished tracks either, since detection is the only
    thing the edit touches, so only axes that survive the screen are replayed. This is the
    two-stage procedure that found pair 30's culprit."""
    t, la, lo = target
    for when in steps:
        base_axes = axes_at(case, when)
        if not base_axes:
            continue
        plain = detect_at(case, when)
        for k in range(len(base_axes)):
            def drop(out, k=k):
                return [a for j, a in enumerate(out) if j != k]
            if detect_at(case, when, drop) == plain:
                continue                     # cannot change anything downstream
            final = replay(case, edit_hook(when, drop))
            if X.holds_exactly(final, t, la, lo):
                return {"step": when, "index": k, "final": final,
                        "centroid": [float(np.mean(base_axes[k][0])),
                                     float(np.mean(base_axes[k][1]))]}
    return None


def attempt_injection(case, target, axes_by_step, steps):
    """Inject version 1's own dumped axes at the reference's extra steps, all together.

    THIS IS THE KIND THAT EXPLAINED THIRTEEN OF FIFTEEN ITEMS IN 1990, so a phase B that
    skipped it would report almost everything unexplained and would be measuring its own
    omission. The axes come from an instrumented Octave capture of the same window."""
    t, la, lo = target
    usable = {k: v for k, v in axes_by_step.items() if v and float(k) in
              {float(s) for s in steps}}
    if not usable:
        return None

    def make(original, now):
        def spy(latgrid, longrid, advection, level=D.TROUGH_LEVEL):
            out = list(original(latgrid, longrid, advection, level))
            if now[0] is None:
                return out
            extra = usable.get(f"{float(now[0]):.4f}")
            return out + list(extra) if extra else out
        return spy

    final = replay(case, make)
    if X.holds_exactly(final, t, la, lo):
        # THE SCHEDULE ACTUALLY INJECTED IS RETURNED, not the candidate set it was filtered
        # from. A first version sized the negative control on the candidate set `picked`
        # while the intervention used this narrower `usable`, so the control injected MORE
        # groups at MORE steps than the experiment it controlled. That is the same defect as
        # truncating it, in the other direction, and it survived the truncation fix because
        # the control was sized off the wrong variable.
        return {"steps": sorted(usable), "n_axes": sum(len(v) for v in usable.values()),
                "final": final, "usable": usable}
    return None


def attempt_reordering(case, target, steps):
    """Move each axis ahead of an earlier one, SCREENED AT DETECTION before replaying.

    A FIRST VERSION PREFILTERED ON THE MERGE DISTANCE BETWEEN AXIS CENTROIDS AND WAS WRONG.
    The merge compares POST-PASS-1 centers, which are the medians of grown regions and not
    the centroids the axes carry. Pair 63's two axes are 5.69 degrees apart while the waves
    they become are 4.38 apart, so that filter would have excluded the ONE reordering in
    this project known to work. An axis and the wave it becomes are different quantities,
    which is the same distinction that made pair 30's culprit unfindable by proximity to its
    own output.

    The detection screen replaces it and is both cheaper and correct, because it asks the
    question directly: a reordering that leaves the detected candidate set unchanged cannot
    change the finished tracks, since detection is the only thing the edit touches. At a
    typical step 18 of 68 axis pairs sit within the merge distance and NONE of them changes
    the detected set, so the screen turns a search of minutes into one of seconds."""
    t, la, lo = target
    trials = 0
    for when in steps:
        base = axes_at(case, when)
        plain = detect_at(case, when)
        for i in range(len(base)):
            for j in range(len(base)):
                if i <= j:
                    continue
                if trials >= REORDER_TRIALS:
                    return "cap"
                def move(out, i=i, j=j):
                    out = list(out)
                    out.insert(j, out.pop(i))
                    return out
                if detect_at(case, when, move) == plain:
                    continue                     # cannot change anything downstream
                trials += 1
                final = replay(case, edit_hook(when, move))
                if X.holds_exactly(final, t, la, lo):
                    return {"step": when, "moved_from": i, "moved_to": j, "final": final}
    return None


def axes_from_capture(path):
    """Version 1's dumped axis vertices per step, from an instrumented capture log."""
    out = collections.defaultdict(list)
    if not path or not os.path.exists(path):
        return {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if not parts or parts[0] != "AXISPTS":
                continue
            values = [float(x) for x in parts[3:]]
            out[f"{float(parts[1]):.4f}"].append(
                (np.array(values[0::2]), np.array(values[1::2])))
    return dict(out)


def near_reference(axes_by_step, target, radius=INJECT_RADIUS_DEG):
    """Only axes coming within `radius` of the reference's own position at that step.

    A DECLARED SELECTION RULE, not a judgment per item. Injecting all of version 1's roughly
    120 axes at a step would be an intervention nobody could attribute anything to."""
    t, la, lo = target
    where = {f"{float(a):.4f}": (b, c) for a, b, c in zip(t, la, lo)}
    picked = {}
    for key, groups in axes_by_step.items():
        here = where.get(key)
        if here is None:
            continue
        keep = [g for g in groups
                if any(float(gc(here[0], here[1], float(p), float(q), "nm")) / 60.0 <= radius
                       for p, q in zip(g[0], g[1]))]
        if keep:
            picked[key] = keep
    return picked


def injection_hook(axes_by_step):
    def make(original, now):
        def spy(latgrid, longrid, advection, level=D.TROUGH_LEVEL):
            out = list(original(latgrid, longrid, advection, level))
            if now[0] is None:
                return out
            extra = axes_by_step.get(f"{float(now[0]):.4f}")
            return out + list(extra) if extra else out
        return spy
    return make


def negative_steps(target, item, times, wanted):
    """Steps the two sides ALREADY AGREE ON, for an injection's negative control.

    A CONTROL CARRIES AS MANY INJECTIONS AS THE EXPERIMENT IT CONTROLS, which is the 1990
    contract's rule and which a first version here broke twice. It capped the list at three
    and then zipped it against the intervention's steps, so a four-step intervention was
    controlled by a three-step control, and the axes injected at a control step were
    whichever group `enumerate` happened to pair with it rather than the group that step was
    meant to carry. Returns None when not enough agreed steps exist, so the caller refuses
    rather than controlling with a truncated schedule."""
    t, _, _ = target
    disagreement = set(item["discrepancy_all"])
    shared = sorted(float(x) for x in t if f"{float(x):.4f}" not in disagreement)
    return shared[:wanted] if len(shared) >= wanted else None


def investigate(case, residuals, reference, item, baseline, axes_by_step=None):
    """One residue item, through the fixed order, with its controls."""
    index = item["index"]
    t, la, lo = reference_track(reference, index)
    target = (t, la, lo)
    times = case["time"].ravel()
    out = {"index": index, "kind": item["kind"],
           "discrepancy_counts": item["discrepancy_counts"]}
    cost_before, item_started = _cost_snapshot(), time.process_time()
    try:
        return _investigate(case, residuals, reference, item, baseline, axes_by_step,
                            target, times, out)
    finally:
        out["cost"] = _cost_since(cost_before)
        out["cost"]["item_seconds"] = round(time.process_time() - item_started, 3)
        out["cost"]["other_seconds"] = round(out["cost"]["item_seconds"]
                                             - out["cost"]["replay_seconds"]
                                             - out["cost"]["screen_seconds"], 3)


def _investigate(case, residuals, reference, item, baseline, axes_by_step, target, times, out):
    index = item["index"]
    t, la, lo = target

    if X.holds_exactly(baseline, t, la, lo):
        out["outcome"] = "ANOMALOUS"
        out["why"] = "the untouched baseline already reproduces this reference track"
        return out

    steps = search_steps(item["discrepancy_all"], times)
    out["search_steps"] = len(steps)

    # INJECTION is not attempted here. It needs version 1's dumped axes at the reference's
    # extra steps, which requires an instrumented Octave capture per window, and for an item
    # with no reference-extra time it cannot be constructed at all.
    if not item["discrepancy"]["v1_extra"]:
        out["injection"] = "not constructible, the reference holds no extra observation"
    elif not axes_by_step:
        out["injection"] = "not available, no axis capture was supplied for this window"
    else:
        picked = near_reference(axes_by_step, target)
        if not picked:
            out["injection"] = "the capture holds no axis near the reference at these steps"
        else:
            found = attempt_injection(case, target, picked, item["discrepancy"]["v1_extra"])
            out["injection"] = ("reproduced the reference track" if found else
                                f"injected {sum(len(v) for v in picked.values())} axes at "
                                f"{len(picked)} steps, did not reproduce it")
            if found:
                rep = replay(case, edit_hook(float(sorted(picked)[0]), lambda o: list(o)))
                try:
                    X.require_identical(baseline, rep, "the baseline",
                                        "the representation control")
                except SystemExit as why:
                    out["outcome"], out["why"] = "INVALID", str(why)
                    return out
                # THE NEGATIVE CONTROL, which a first version of this driver did not run for
                # injection or reordering and issued credits without. It injects the SAME
                # axes at steps the two sides already agree on, which is the 1990 design's
                # time control, and it must not reproduce the reference track.
                injected = found["usable"]
                negative = negative_steps(target, item, times, len(injected))
                if negative is None:
                    # NO CREDIT WITHOUT A NEGATIVE CONTROL. A first version fell through to
                    # EXPLAINED here, which credited an item on a representation control
                    # alone, and the amendment requires both.
                    out["outcome"] = "UNSUPPORTED"
                    out["why"] = (f"the intervention injects at {len(injected)} steps and the "
                                  f"two sides agree at too few steps to control it with the "
                                  f"same number of injections, so no negative control can "
                                  f"be built and no credit is issued")
                    return out
                ordered = sorted(injected)
                moved = {f"{float(negative[k]):.4f}": injected[key]
                         for k, key in enumerate(ordered)}
                # THE CONTROL CARRIES THE SAME SCHEDULE AS THE EXPERIMENT, asserted rather
                # than assumed: as many steps, and the same total number of axes.
                assert len(moved) == len(injected)
                assert sum(len(v) for v in moved.values()) == sum(
                    len(v) for v in injected.values())
                neg = replay(case, injection_hook(moved))
                out["negative_control_steps"] = sorted(moved)
                if X.holds_exactly(neg, t, la, lo):
                    out["outcome"] = "ANOMALOUS"
                    out["why"] = ("the same axes injected at steps the two sides agree "
                                  "on reproduce it too, so the intervention is not "
                                  "specific to where it acted")
                    return out
                out["runs"] = {"intervention": found["final"],
                               "representation_control": rep, "negative_control": neg}
                out["outcome"] = "EXPLAINED"
                out["operation"] = "injection"
                out["detail"] = {"steps": found["steps"], "n_axes": found["n_axes"]}
                return out

    found = attempt_removal(case, target, steps, baseline)
    if found:
        out["runs"] = {"intervention": found["final"]}
        # THE CONTROLS. Representation removes and reinserts at the same index, which is a
        # null edit; negative removes a DIFFERENT axis at the same step.
        rep = replay(case, edit_hook(found["step"], lambda out: list(out)))
        try:
            X.require_identical(baseline, rep, "the baseline", "the representation control")
        except SystemExit as why:
            out["outcome"], out["why"] = "INVALID", str(why)
            return out
        population = len(axes_at(case, found["step"]))
        # NO CREDIT WITHOUT A VALID NEGATIVE CONTROL. A first version chose `spare = 0 if
        # index else 1`, so for a one-axis list it removed a nonexistent index, its output
        # WAS the baseline, and the item was credited automatically. The negative target
        # must exist and differ from the intervention's, or the item is UNSUPPORTED.
        spare = next((k for k in range(population) if k != found["index"]), None)
        if spare is None:
            out["outcome"] = "UNSUPPORTED"
            out["why"] = ("only one axis at this step, so no comparable removal of a "
                          "different axis exists to serve as a negative control")
            return out
        neg = replay(case, edit_hook(
            found["step"], lambda o, s=spare: [a for j, a in enumerate(o) if j != s]))
        out["receipts"] = {
            "intervention": {"step": found["step"], "population_before": population,
                             "population_after": population - 1,
                             "removed_index": found["index"]},
            "representation_control": {"step": found["step"], "identity": True,
                                       "population_before": population,
                                       "population_after": population},
            "negative_control": {"step": found["step"], "population_before": population,
                                 "population_after": population - 1,
                                 "removed_index": spare}}
        # RETAINED BEFORE ANY VERDICT IS RETURNED. The confirmation review found the
        # anomalous return preceded this assignment, so on exactly the path where the
        # negative trajectory matters most it was discarded, and the claim that every
        # control run is retained was false there.
        out["runs"]["representation_control"] = rep
        out["runs"]["negative_control"] = neg
        if X.holds_exactly(neg, t, la, lo):
            out["outcome"] = "ANOMALOUS"
            out["why"] = "a negative control removing a different axis reproduces it too"
            return out
        out["outcome"] = "EXPLAINED"
        out["operation"] = "removal"
        out["detail"] = {k: found[k] for k in ("step", "index", "centroid")}
        return out

    found = attempt_reordering(case, target, steps)
    if found == "cap":
        # AN EXHAUSTED SEARCH IS NOT AN EMPTY ONE, and reporting both as "unexplained within
        # budget" hides which happened. On window A's items 0 and 8 the previous, less
        # sensitive screen DID reach an edit that reproduces the reference track, and the
        # corrected screen lets more edits through so the cap bites before the search
        # reaches it. That is a fact about the budget, not about the trackers.
        out["outcome"] = "UNEXPLAINED, SEARCH BUDGET EXHAUSTED"
        out["why"] = (f"the reordering search reached its cap of {REORDER_TRIALS} replayed "
                      f"trials before finishing. No conclusion about whether a reordering "
                      f"exists follows from this.")
        return out
    if found:
        rep = replay(case, edit_hook(found["step"], lambda out: list(out)))
        try:
            X.require_identical(baseline, rep, "the baseline", "the representation control")
        except SystemExit as why:
            out["outcome"], out["why"] = "INVALID", str(why)
            return out
        # THE NEGATIVE CONTROL: the same KIND of move, a different axis, the same step and
        # destination. Several are tried because one alternative failing is weak evidence
        # that the effect is specific.
        base_axes = axes_at(case, found["step"])
        population = len(base_axes)
        alternatives = [k for k in range(population)
                        if k not in (found["moved_from"], found["moved_to"])][-5:]
        if not alternatives:
            # A two-axis list leaves nothing to move, the loop ran zero times, and a first
            # version credited the item with negative_controls_tried=0.
            out["outcome"] = "UNSUPPORTED"
            out["why"] = ("no third axis exists at this step to move in the negative "
                          "control, so the effect cannot be shown specific to this pair")
            return out
        reproduced_by, negatives = [], {}
        for k in alternatives:
            def move(out, k=k, j=found["moved_to"]):
                out = list(out)
                out.insert(j, out.pop(k))
                return out
            final = replay(case, edit_hook(found["step"], move))
            # EVERY NEGATIVE TRAJECTORY IS RETAINED with its move, so the negative verdict
            # can be recomputed from what was kept. A first version passed each straight to
            # the verdict and discarded it, leaving a count nobody could audit.
            negatives[f"negative_control_{k}_to_{found['moved_to']}"] = final
            if X.holds_exactly(final, t, la, lo):
                reproduced_by.append(k)
        out["negative_controls_tried"] = len(alternatives)
        out["receipts"] = {
            "intervention": {"step": found["step"], "population_before": population,
                             "population_after": population,
                             "moved_from": found["moved_from"], "moved_to": found["moved_to"]},
            "representation_control": {"step": found["step"], "identity": True,
                                       "population_before": population,
                                       "population_after": population},
            "negative_controls": [{"step": found["step"], "population_before": population,
                                   "population_after": population, "moved_from": k,
                                   "moved_to": found["moved_to"]} for k in alternatives]}
        out["runs"] = {"intervention": found["final"], "representation_control": rep,
                       **negatives}
        if reproduced_by:
            out["outcome"] = "ANOMALOUS"
            out["why"] = (f"moving a different axis to the same place reproduces it too "
                          f"({reproduced_by}), so the effect is not specific to this pair")
            return out
        out["outcome"] = "EXPLAINED"
        out["operation"] = "reordering"
        out["detail"] = {k: found[k] for k in ("step", "moved_from", "moved_to")}
        return out

    out["outcome"] = "UNEXPLAINED WITHIN BUDGET"
    out["why"] = ("the declared search constructed no removal or reordering that reproduces "
                  "the reference track; injection was not available for this item")
    return out


def build_items(residuals, reference, case):
    """Residue items in the fixed order, reference track index ascending."""
    from residue_membership import derived_discrepancies
    port = {}
    items = []
    for pair in residuals.get("pairs") or ():
        items.append({"index": pair["v1_index"], "kind": pair["extra_kind"],
                      "port_index": pair.get("port_index")})
    for unmatched in residuals.get("v1_unmatched") or ():
        items.append({"index": unmatched["index"], "kind": "unmatched_v1", "port_index": None})
    items.sort(key=lambda r: r["index"])
    return items


MERGE_MUST_AGREE = ("contract", "window", "budget", "search_neighborhood_steps",
                    "inject_radius_deg", "reorder_trial_cap", "axis_capture",
                    "inputs_sha256", "driver_sha256", "git_head_at_launch", "budget_set",
                    "items_total", "items_not_investigated", "what_unexplained_means_here")


def cost_totals(results):
    """Screening, replay and everything else, summed over items, from each item's own
    record, so the split can be read from a window artifact and not only per job."""
    keys = ("replay_calls", "replay_seconds", "screen_calls", "screen_seconds",
            "other_seconds")
    total = {k: 0 for k in keys}
    for r in results:
        c = r.get("cost") or {}
        for k in keys:
            total[k] += c.get(k, 0)
    return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in total.items()}


def baseline_agreement(jobs):
    """Every job replayed the baseline itself. Their retained baselines must hold the same
    trajectories, canonically, or the jobs did not run in one environment on one input."""
    canon = {}
    for job in jobs:
        rdir = job.get("retention_directory")
        for name, recorded in (job.get("retained_files") or {}).items():
            path = os.path.join(rdir, name)
            if not os.path.isfile(path):
                raise SystemExit(f"REFUSED: retained file {path} named by a job is missing")
            if X.digest(path) != recorded:
                raise SystemExit(f"REFUSED: retained file {path} has changed since its job "
                                 f"recorded it")
            if "_baseline_" in name:
                runs = json.load(open(path))["runs"]["baseline"]
                canon[name] = [(tuple(t["time"]), tuple(t["meanlat"]), tuple(t["meanlon"]))
                               for t in runs]
    if not canon:
        return {"baselines_compared": 0, "identical": None}
    first = next(iter(canon.values()))
    differing = sorted(n for n, c in canon.items() if c != first)
    if differing:
        raise SystemExit(f"REFUSED: the per-job baselines do not agree: {differing} differ "
                         f"from {next(iter(canon))}")
    return {"baselines_compared": len(canon), "identical": True,
            "tracks": len(first)}


def publish(path, payload):
    """An artifact is published exclusively, like retained evidence. A retry that named an
    existing --out silently replaced it, and an --out pointed at a retained baseline
    destroyed that baseline in the same successful invocation."""
    try:
        X.publish_json(path, payload, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {path} already exists and artifacts are never "
                         f"overwritten. Name a fresh output")


def merge_jobs(directory, window, out):
    """One window artifact from the per-job artifacts of a window split across jobs.

    Every job must have run the same frozen search, so every field the procedure fixes
    must agree across the jobs, and the items the jobs selected must cover the budget set
    exactly once. Anything else is refused rather than merged around."""
    paths = sorted(glob.glob(os.path.join(directory, f"phase_b_{window}_job*.json")))
    if not paths:
        raise SystemExit(f"REFUSED: no per-job artifacts for window {window} in {directory}")
    jobs = [json.load(open(p)) for p in paths]
    first = jobs[0]
    for p, job in zip(paths, jobs):
        for key in MERGE_MUST_AGREE:
            if key not in job:
                raise SystemExit(f"REFUSED: {os.path.basename(p)} records no {key}")
            if job.get(key) != first.get(key):
                raise SystemExit(f"REFUSED: {os.path.basename(p)} disagrees with "
                                 f"{os.path.basename(paths[0])} on {key}")
        if not job.get("items_selected_for_this_job"):
            raise SystemExit(f"REFUSED: {os.path.basename(p)} did not run under --items, "
                             f"so it cannot be one part of a split window")
    covered = [i for job in jobs for i in job["items_selected_for_this_job"]]
    if sorted(covered) != sorted(first["budget_set"]):
        raise SystemExit(f"REFUSED: the jobs cover {sorted(covered)} and the budget set is "
                         f"{sorted(first['budget_set'])}, and every item exactly once is required")
    results = sorted((r for job in jobs for r in job["results"]), key=lambda r: r["index"])
    ran = [r["index"] for r in results]
    if ran != sorted(first["budget_set"]):
        raise SystemExit(f"REFUSED: results cover {ran}, not the budget set")
    tally = collections.Counter(r["outcome"] for r in results)
    agreement = baseline_agreement(jobs)
    payload = {k: v for k, v in first.items()
               if k not in ("results", "timing", "outcomes", "items_selected_for_this_job",
                            "items_investigated", "launched_at", "retained_files",
                            "retention_directory")}
    payload.update({
        "items_investigated": len(results),
        "outcomes": {k: tally[k] for k in sorted(tally)},
        "results": results,
        "timing": {"processor_seconds": round(sum(j["timing"]["processor_seconds"]
                                                  for j in jobs), 1),
                   **cost_totals(results),
                   "elapsed_seconds_longest_job": round(max(j["timing"]["elapsed_seconds"]
                                                            for j in jobs), 1),
                   "note": "summed processor time over the jobs. Elapsed is the longest "
                           "single job, since the jobs ran concurrently"},
        "merged_from": [{"file": os.path.basename(p), "sha256": X.digest(p),
                         "items": job["items_selected_for_this_job"],
                         "launched_at": job.get("launched_at"),
                         "retained_files": job.get("retained_files"),
                         "retention_directory": job.get("retention_directory")}
                        for p, job in zip(paths, jobs)],
        "baseline_agreement": agreement,
    })
    publish(out, payload)
    print(f"window {window}: merged {len(jobs)} jobs covering {len(results)} items")
    for key in sorted(tally):
        print(f"  {key:36s} {tally[key]}")
    print(f"  wrote {out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("work")
    ap.add_argument("--window", required=True, choices=sorted(WINDOWS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=PER_WINDOW_BUDGET)
    ap.add_argument("--retain", default=None,
                    help="directory for the intervention and control outputs")
    ap.add_argument("--axes", default=None,
                    help="an instrumented capture log holding AXISPTS at this window's "
                         "reference-extra steps, for the injection attempt")
    ap.add_argument("--items", default=None,
                    help="comma-separated reference indices to run in THIS job, each of "
                         "which must be within the budget set. The budget and the order "
                         "are unchanged, so several jobs together cover exactly one window")
    ap.add_argument("--merge", default=None,
                    help="a directory of per-job artifacts for this window, merged into "
                         "--out; no replay runs in this mode")
    args = ap.parse_args(argv)
    if args.merge:
        return merge_jobs(args.merge, args.window, args.out)
    # THE DRIVER'S IDENTITY IS CAPTURED AT LAUNCH, not when the artifact is written. Window
    # A's frozen rerun took three and a half hours, the driver was edited while it ran, and
    # the artifact then named the digest of the edited file rather than the code that ran,
    # with the repository head of the last commit made during the run. Both are read here,
    # before anything else, and carried through to the artifact and every retained run.
    launched = {"driver_sha256": X.digest(__file__),
                "git_head_at_launch": X.repository_head(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "launched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    directory, case, residuals, reference = load_window(args.work, args.window)
    # THE INPUTS ARE IDENTIFIED BY CONTENT, before anything is computed from them. A review
    # merged two jobs that had read different axis captures under one basename, and two
    # that had read different case files, because the artifact named inputs by name. The
    # digests are part of the launch identity and every job of a window must agree on them.
    if args.axes and not os.path.isfile(args.axes):
        raise SystemExit(f"REFUSED: the axis capture {args.axes} cannot be read")
    consumed = {name: os.path.join(directory, name)
                for name in ("tracker_case.mat", "residuals.json",
                             "tracker_octave_instrumented.mat", "tracker_port.mat")}
    if args.axes:
        consumed["axis_capture"] = args.axes
    launched["inputs_sha256"] = {name: X.digest(path) for name, path in consumed.items()}
    from residue_membership import derived_discrepancies
    baseline = replay(case)
    port_out = loadmat(os.path.join(directory, "tracker_port.mat"))

    items = build_items(residuals, reference, case)
    for item in items:
        t, la, lo = reference_track(reference, item["index"])
        v1 = {"time": t, "lat": la, "lon": lo}
        if item["port_index"] is not None:
            p = item["port_index"]
            counterpart = {"time": np.asarray(port_out[f"time{p}"], float).ravel(),
                           "lat": np.asarray(port_out[f"lat{p}"], float).ravel(),
                           "lon": np.asarray(port_out[f"lon{p}"], float).ravel()}
            found = derived_discrepancies(v1, counterpart)
        else:
            found = {"v1_extra": [f"{float(x):.4f}" for x in t], "port_extra": [],
                     "displaced": []}
        item["discrepancy"] = found
        item["discrepancy_counts"] = {k: len(v) for k, v in found.items()}
        item["discrepancy_all"] = sorted(set(found["v1_extra"]) | set(found["port_extra"])
                                         | set(found["displaced"]))

    investigated, deferred = items[:args.budget], items[args.budget:]
    # ONE JOB MAY TAKE A SUBSET OF THE BUDGET SET, never an item outside it, so that a
    # window split across scheduler slots is still the same frozen search. The merge mode
    # requires the jobs to cover the budget set exactly, once.
    selected = None
    if args.items:
        selected = sorted({int(x) for x in args.items.split(",") if x.strip()})
        budget_set = {i["index"] for i in investigated}
        outside = [i for i in selected if i not in budget_set]
        if outside:
            raise SystemExit(f"REFUSED: --items names {outside}, which are not in window "
                             f"{args.window}'s budget set {sorted(budget_set)}")
        investigated = [i for i in investigated if i["index"] in selected]
    axes_by_step = axes_from_capture(args.axes)

    # RETAIN WHAT EACH CREDIT RESTS ON, AS EACH ITEM FINISHES. A verdict whose runs nobody
    # kept is a number in a table, and a run that retains only at the end cannot be
    # inspected while it works or resumed if it stops. Window A's frozen rerun retained
    # nothing for its first hour and there was no first item to check. The baseline is
    # shared by every item and is retained once, before the first item.
    case_path = os.path.join(directory, "tracker_case.mat")
    ref_path = os.path.join(directory, "tracker_octave_instrumented.mat")
    # EACH JOB RETAINS ITS OWN BASELINE UNDER ITS OWN NAME, and no retained file is ever
    # overwritten. A review ran two disjoint jobs into one directory and the second job's
    # baseline replaced the first's, launch identity and all, while the merge accepted
    # both. The merge compares the per-job baselines by content instead.
    job_tag = "-".join(str(i) for i in selected) if selected else "all"
    retained_files = {}
    provenance = {"script_sha256": launched["driver_sha256"],
                  "git_head": launched["git_head_at_launch"]}

    def retain(name, runs, index, extra):
        # EXCLUSIVE AT THE WRITE. A check before the write let two overlapping retries
        # both pass and the second truncated the first's retained baseline. The serializer
        # publishes by an atomic link and refuses an existing target itself.
        path = os.path.join(args.retain, name)
        try:
            X.save_run(path, runs, case_path, ref_path, index, __file__,
                       extra={"window": args.window, **launched, **extra}, exclusive=True,
                       **provenance)
        except FileExistsError:
            raise SystemExit(f"REFUSED: {path} already exists and retained evidence is "
                             f"never overwritten. Use a fresh retention directory")
        retained_files[name] = X.digest(path)

    if args.retain:
        os.makedirs(args.retain, exist_ok=True)
        retain(f"{args.window}_baseline_job{job_tag}.json", {"baseline": baseline}, -1,
               {"role": "baseline replayed by this job"})
    wall0, cpu0 = time.perf_counter(), time.process_time()
    results = []
    for n, item in enumerate(investigated, 1):
        r = investigate(case, residuals, reference, item, baseline, axes_by_step)
        results.append(r)
        if args.retain and r.get("runs"):
            retain(f"{args.window}_item{r['index']}.json", r["runs"], r["index"],
                   {"operation": r.get("operation"), "outcome": r["outcome"],
                    "detail": r.get("detail"), "receipts": r.get("receipts")})
        print(f"  item {n}/{len(investigated)} index {r['index']} {r['kind']}: "
              f"{r['outcome']} by {r.get('operation')} "
              f"({time.process_time() - cpu0:.0f} s processor so far)", flush=True)
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    tally = collections.Counter(r["outcome"] for r in results)
    cost = cost_totals(results)

    payload = {
        "generated_by": "scripts/validation_phase_b.py",
        "contract": ("phase B amendment of 2026-09-22: every category admitted, discrepancy "
                     "times derived from evidence and kept apart from the intervention "
                     "time, UNSUPPORTED when no negative control exists"),
        "window": args.window, "budget": args.budget,
        "search_neighborhood_steps": NEIGHBORHOOD,
        "inject_radius_deg": INJECT_RADIUS_DEG,
        "reorder_trial_cap": REORDER_TRIALS,
        "axis_capture": os.path.basename(args.axes) if args.axes else None,
        **launched,
        "timing": {"elapsed_seconds": round(wall, 1), "processor_seconds": round(cpu, 1),
                   **cost,
                   "note": "elapsed exceeds processor time when other work shares the "
                           "machine; processor time is the figure to plan with"},
        "items_total": len(items),
        "items_investigated": len(investigated),
        "items_not_investigated": [i["index"] for i in deferred],
        "budget_set": [i["index"] for i in items[:args.budget]],
        "items_selected_for_this_job": selected,
        "retained_files": retained_files or None,
        "retention_directory": os.path.abspath(args.retain) if args.retain else None,
        "outcomes": {k: tally[k] for k in sorted(tally)},
        "results": [{k: v for k, v in r.items() if k not in ("final", "runs")}
                    for r in results],
        "what_unexplained_means_here": (
            "the DECLARED SEARCH constructed no intervention that worked. It does not mean "
            "none exists. Pair 30's removal acts at a step no search specified in advance "
            "would have reached."),
    }
    publish(args.out, payload)
    print(f"window {args.window}: {len(investigated)} of {len(items)} items investigated, "
          f"{len(deferred)} not investigated")
    print(f"  elapsed {wall/60:.1f} min, processor {cpu/60:.1f} min")
    for key in sorted(tally):
        print(f"  {key:26s} {tally[key]}")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
