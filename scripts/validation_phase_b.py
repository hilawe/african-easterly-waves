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


def replay(case, hook=None):
    """The port's full pipeline over the window, with an optional axis hook."""
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
    try:
        waves = D.detect_troughs(
            float(when), latgrid, longrid, case["u_c"][index], case["currv_anom_c"][index],
            case["advcurrv_anom_c"][index], case["latgrid"], case["longrid"],
            case["currv_anom"][index], coarse_threshold=ct, fine_threshold=ft, absorb=False)
    finally:
        D.trough_axes = original
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
    args = ap.parse_args(argv)

    directory, case, residuals, reference = load_window(args.work, args.window)
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
    axes_by_step = axes_from_capture(args.axes)
    wall0, cpu0 = time.perf_counter(), time.process_time()
    results = [investigate(case, residuals, reference, item, baseline, axes_by_step)
               for item in investigated]
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    tally = collections.Counter(r["outcome"] for r in results)

    # RETAIN WHAT EACH CREDIT RESTS ON. A verdict whose runs nobody kept is a number in a
    # table. The baseline is shared by every item so it is retained once.
    if args.retain:
        os.makedirs(args.retain, exist_ok=True)
        case_path = os.path.join(directory, "tracker_case.mat")
        ref_path = os.path.join(directory, "tracker_octave_instrumented.mat")
        X.save_run(os.path.join(args.retain, f"{args.window}_baseline.json"),
                   {"baseline": baseline}, case_path, ref_path, -1, __file__,
                   extra={"window": args.window, "role": "shared baseline"})
        for r in results:
            if not r.get("runs"):
                continue
            X.save_run(
                os.path.join(args.retain, f"{args.window}_item{r['index']}.json"),
                r["runs"], case_path, ref_path, r["index"], __file__,
                extra={"window": args.window, "operation": r.get("operation"),
                       "outcome": r["outcome"], "detail": r.get("detail"),
                       "receipts": r.get("receipts")})

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
        "driver_sha256": X.digest(__file__),
        "timing": {"elapsed_seconds": round(wall, 1), "processor_seconds": round(cpu, 1),
                   "note": "elapsed exceeds processor time when other work shares the "
                           "machine; processor time is the figure to plan with"},
        "items_total": len(items),
        "items_investigated": len(investigated),
        "items_not_investigated": [i["index"] for i in deferred],
        "outcomes": {k: tally[k] for k in sorted(tally)},
        "results": [{k: v for k, v in r.items() if k not in ("final", "runs")}
                    for r in results],
        "what_unexplained_means_here": (
            "the DECLARED SEARCH constructed no intervention that worked. It does not mean "
            "none exists. Pair 30's removal acts at a step no search specified in advance "
            "would have reached, and injection was not attempted in this run at all."),
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    print(f"window {args.window}: {len(investigated)} of {len(items)} items investigated, "
          f"{len(deferred)} not investigated")
    print(f"  elapsed {wall/60:.1f} min, processor {cpu/60:.1f} min")
    for key in sorted(tally):
        print(f"  {key:26s} {tally[key]}")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
