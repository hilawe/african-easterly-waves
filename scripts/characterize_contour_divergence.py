#!/usr/bin/env python3
"""Classify, at each explained item's intervention step, why the port's tracer lacks the axis.

THE BRIEF WAS WRITTEN FIRST, before any run, with the question, the measurement stage,
the controls and the stopping condition stated there. This reads retained artifacts and captures, computes no replay, and reuses the Sahara
trace's field and crossing tools unchanged. It classifies, it does not judge which program
is right, and it says nothing about items it was not given.

THREE QUESTIONS PER ITEM, in order. Are the two masked coarse advection fields the same at
the step, cell for cell over the whole grid (FIELD if not)? Which of version 1's axes at
that step has no port counterpart and is nearest the observation the intervention
produced? Do that axis's vertices lie on zero crossings of the port's field, and are
those crossings closed (CLOSED CROSSING, the 1990 class), open (OPEN CROSSING) or not
crossings of the port's field at all (OFF CROSSING)? The control step's reading is taken
the same way, so a class that appears equally there is a property of the measurement.

    .venv/bin/python3 scripts/characterize_contour_divergence.py <work> --artifacts A.json B.json --out <artifact>
"""

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402
import substitution_pilot as S  # noqa: E402
import validation_phase_b as B  # noqa: E402
from trace_sahara_case import (compare_fields, crossing_neighborhoods,  # noqa: E402
                               reference_field_at, vertices_on_crossings,
                               zero_crossings)

MATCH_DEG = 0.5      # CANDIDATE SELECTION: a centroid within this box is not offered as a candidate
NEAR_DEG = 12.0      # CANDIDATE SELECTION: the search bound around the observation
VERTEX_TOL = 5e-5 + 1e-12   # the four-decimal print tolerance the crossing test uses, per coordinate
FIELD_REL_TOL = 1e-9        # what compare_fields calls the same field: masks equal, values within this

# WHAT THE TWO SELECTION RULES ARE. Centroid matching cannot establish that two lines are
# the same (a review showed the same segment sampled at different points called unmatched,
# and two perpendicular segments through one point called matched), and the search bound
# only decides what is looked at. They select a CANDIDATE. Geometric identity is a separate
# reading: the distinct vertices of one axis equal those of another within the print
# tolerance, with multiplicity reported apart, since version 1 repeats its endpoints.


def port_field_at(case, step):
    """The exact masked advection array the port's tracer receives at `step`."""
    seen = {}
    original = B.D.trough_axes

    def spy(latgrid, longrid, advection, level=B.D.TROUGH_LEVEL):
        seen["advection"] = np.array(advection, dtype=float)
        return original(latgrid, longrid, advection, level)
    B.D.trough_axes = spy
    try:
        B.detect_at(case, step)
    finally:
        B.D.trough_axes = original
    return seen["advection"]


def field_block(case, step):
    lat_c, lon_c = case["lat_c"].ravel(), case["lon_c"].ravel()
    field = port_field_at(case, step)
    grid = [[None if not np.isfinite(x) else float(x) for x in row] for row in field]
    return {"rows_lat": [float(x) for x in lat_c], "cols_lon": [float(x) for x in lon_c],
            "masked_smoothed_advection": grid, "grid": grid}


def distinct_vertices(lat, lon):
    """The distinct vertices of a polyline within the print tolerance, and how many were repeated."""
    out = []
    for la, lo in zip(lat, lon):
        if not any(abs(la - a) <= VERTEX_TOL and abs(lo - b) <= VERTEX_TOL for a, b in out):
            out.append((float(la), float(lo)))
    return out, len(lat) - len(out)


def geometric_match(axis, other_axes):
    """The index of an axis in `other_axes` whose distinct vertex set equals this one's within
    the print tolerance, or None. Says the two tracers drew the same points, in any order."""
    mine, _ = distinct_vertices(axis["lat"], axis["lon"])
    for k, (a, b) in enumerate(other_axes):
        theirs, _ = distinct_vertices([float(x) for x in a], [float(x) for x in b])
        if len(mine) != len(theirs):
            continue
        # ONE TO ONE, not any-to-any. A confirmation round matched two of my vertices to one
        # of theirs and called the lines identical; each of theirs may be used once.
        unused = list(range(len(theirs)))
        matched = True
        for p in mine:
            hit = next((j for j in unused if abs(p[0] - theirs[j][0]) <= VERTEX_TOL
                        and abs(p[1] - theirs[j][1]) <= VERTEX_TOL), None)
            if hit is None:
                matched = False
                break
            unused.remove(hit)
        if matched:
            return k
    return None


def unmatched_axes(v1_axes, port_axes):
    """CENTROID-UNMATCHED candidates: axes with no centroid on the other side within the
    selection box. A selection heuristic, labelled as such in every record."""
    def cen(a):
        return (float(np.mean(a[0])), float(np.mean(a[1])))
    port = [cen(a) for a in port_axes]
    out = []
    for a in v1_axes:
        c = cen(a)
        if not any(abs(c[0] - p[0]) < MATCH_DEG and abs(c[1] - p[1]) < MATCH_DEG for p in port):
            out.append({"centroid": c, "points": int(len(a[0])),
                        "lat": [float(x) for x in a[0]], "lon": [float(x) for x in a[1]],
                        "geometrically_identical_to_a_port_axis": geometric_match(
                            {"lat": [float(x) for x in a[0]], "lon": [float(x) for x in a[1]]}, port_axes) is not None})
    return out


def vertex_overlap(axis, port_axes):
    """Which of the axis's DISTINCT vertices some port axis carries within the print
    tolerance, with the port axes that carry each, so the overlap can be audited. Overlap
    is a count, none, partial or complete. It establishes no connectivity and no joining."""
    mine, repeated = distinct_vertices(axis["lat"], axis["lon"])
    carriers = []
    for la, lo in mine:
        who = sorted({k for k, (a, b) in enumerate(port_axes)
                      if any(abs(la - float(x)) <= VERTEX_TOL and abs(lo - float(y)) <= VERTEX_TOL
                             for x, y in zip(a, b))})
        carriers.append(who)
    shared = sum(1 for w in carriers if w)
    kind = "none" if shared == 0 else ("complete" if shared == len(mine) else "partial")
    return {"distinct_vertices": len(mine), "repeated_vertices": repeated, "shared": shared,
            "overlap": kind, "port_axes_carrying_each_vertex": carriers,
            "port_axes_involved": sorted({k for w in carriers for k in w})}


def classify(axis, block, crossings, neighborhoods):
    """The class of one unmatched axis against the port's field, and the per-vertex record."""
    on = vertices_on_crossings([{"lat": axis["lat"], "lon": axis["lon"]}], crossings)
    closed_by = {(round(n["lat"], 4), round(n["lon"], 4)): n["closed_off"] for n in neighborhoods}
    verdicts = []
    for v in on:
        if not v["on_a_crossing"] or v["nearest_crossing"] is None:
            verdicts.append("off")
            continue
        key = (round(v["nearest_crossing"]["lat"], 4), round(v["nearest_crossing"]["lon"], 4))
        closed = closed_by.get(key)
        verdicts.append("undetermined" if closed is None else ("closed" if closed else "open"))
    counts = collections.Counter(verdicts)
    # LOCAL VERTEX PROPERTIES, and no more. CLOSED: every sampled vertex is within the print
    # tolerance of a crossing obstructed on both sides by the measured mask or boundary.
    # OPEN: every sampled vertex is on a crossing with a path on at least one side, which
    # is local availability and not a continuous path along version 1's polyline. OFF: no
    # sampled vertex meets the crossing tolerance, not that no segment meets a contour.
    if counts["off"] == len(verdicts):
        klass = "OFF CROSSING"
    elif counts["off"]:
        klass = "MIXED, some vertices off the port's crossings"
    elif counts["undetermined"]:
        klass = "UNDETERMINED, a crossing at the grid's edge"
    elif counts["open"] == 0:
        klass = "CLOSED CROSSING"
    elif counts["closed"] == 0:
        klass = "OPEN CROSSING"
    else:
        klass = "MIXED, open and closed crossings"
    return klass, {"vertices": len(verdicts), **{k: counts[k] for k in ("closed", "open", "off", "undetermined")}}


def read_step(case, capture_text, step, near, control=False):
    """Everything the brief asks at one step, near one position."""
    times = case["time"].ravel()
    key = f"{step:.4f}"
    block = field_block(case, step)
    ref = reference_field_at_text(capture_text, step)
    comparison = compare_fields(block["rows_lat"], block["cols_lon"], block["grid"], ref["cells"]) \
        if ref["cells"] and not ref["problems"] else {"available": False, "reason": ref["problems"] or "no FIELD records at this step"}
    crossings = zero_crossings(block)
    neighborhoods = crossing_neighborhoods(block["rows_lat"], block["cols_lon"], block["grid"],
                                           crossings, domain_complete=True)
    v1_axes, _, _ = S.ordered_capture(capture_text, times)
    port_axes = B.axes_at(case, step)
    unmatched = unmatched_axes(v1_axes.get(key, []), port_axes)
    port_only = unmatched_axes(port_axes, v1_axes.get(key, []))
    out = {"step": step, "fields_same": comparison.get("same_field"),
           "fields_agreement": (
               "not compared" if not comparison.get("available") else
               ("masks equal and values within a relative difference of "
                f"{FIELD_REL_TOL} (worst seen {comparison.get('worst_relative_difference')})"
                if comparison.get("same_field") else
                f"masks differ or values beyond a relative difference of {FIELD_REL_TOL} "
                f"(worst seen {comparison.get('worst_relative_difference')}, cells unmasked only "
                f"in the port {comparison.get('cells_unmasked_only_in_the_port')})")),
           "port_axes_without_v1_counterpart": len(port_only),
           "port_only_axes_within_range_of_observation": None if near is None else sum(
               1 for a in port_only if abs(a["centroid"][0] - near[0]) + abs(a["centroid"][1] - near[1]) <= NEAR_DEG),
           "v1_only_axes_within_range_of_observation": None if near is None else sum(
               1 for a in unmatched if abs(a["centroid"][0] - near[0]) + abs(a["centroid"][1] - near[1]) <= NEAR_DEG),
           "field_comparison": {k: comparison.get(k) for k in
                                ("available", "cells_unmasked_on_both_sides",
                                 "cells_unmasked_only_in_the_port", "cells_unmasked_only_in_version_1",
                                 "worst_relative_difference", "same_field", "reason")},
           "v1_axes": len(v1_axes.get(key, [])), "port_axes": len(port_axes),
           "v1_axes_without_port_counterpart": len(unmatched)}
    # THE PRIMARY READING IS GATED ON THE FIELDS. Without a comparison there is no basis
    # for a contour class, and with a failed one the difference is upstream of the tracer.
    if not comparison.get("available"):
        out["primary"] = "UNDETERMINED, the fields could not be compared: " + str(comparison.get("reason"))
    elif not comparison.get("same_field"):
        out["primary"] = "FIELD, the fields differ beyond the tolerance or in their masks"
    else:
        out["primary"] = "fields agree within tolerance"
    if near is not None:
        candidates = sorted(unmatched, key=lambda a: abs(a["centroid"][0] - near[0]) + abs(a["centroid"][1] - near[1]))
        if candidates and abs(candidates[0]["centroid"][0] - near[0]) + abs(candidates[0]["centroid"][1] - near[1]) <= NEAR_DEG:
            axis = candidates[0]
            klass, detail = classify(axis, block, crossings, neighborhoods)
            overlap = vertex_overlap(axis, port_axes)
            if out["primary"] != "fields agree within tolerance":
                klass = out["primary"]
            out["candidate"] = {"selected_by": "nearest centroid-unmatched axis within the search bound, a heuristic",
                                "centroid": axis["centroid"], "points": axis["points"],
                                "distance_deg": abs(axis["centroid"][0] - near[0]) + abs(axis["centroid"][1] - near[1]),
                                "geometrically_identical_to_a_port_axis": axis["geometrically_identical_to_a_port_axis"],
                                "class": klass, "vertices": detail, "overlap": overlap}
        else:
            out["candidate"] = None
            out["candidate_note"] = "no centroid-unmatched candidate within the search bound"
    return out


def reference_field_at_text(text, step):
    """`reference_field_at` reads a path; the capture is already in memory."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
        fh.write(text)
        path = fh.name
    try:
        return reference_field_at(path, step)
    finally:
        os.unlink(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("work")
    ap.add_argument("--artifacts", nargs="+", required=True,
                    help="substitution artifacts whose EXPLAINED items are characterized")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    launched = {"script_sha256": X.digest(__file__), "launched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "git_head_at_launch": X.repository_head(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}
    items = []
    for p in args.artifacts:
        a = json.load(open(p))
        for r in a["results"]:
            if r["outcome"] == "EXPLAINED":
                items.append((r["window"], r["index"], r["intervention"]["step"],
                              r["controls"]["negative"]["selected_step"], os.path.basename(p)))
    results, inputs, cpu0 = [], {}, time.process_time()
    for w in sorted({i[0] for i in items}):
        directory, case, residuals, reference, port, text, digests = S.load_window_snapshot(args.work, w)
        inputs[w] = digests
        for (ww, index, step, neg_step, source) in sorted(i for i in items if i[0] == w):
            t, la, lo = B.reference_track(reference, index)
            k = int(np.argmin(np.abs(np.asarray(t, float) - step)))
            near = (float(la[k]), float(lo[k]))
            rec = {"window": w, "index": index, "source": source, "reference_position_at_step": near,
                   "intervention": read_step(case, text, step, near),
                   "control": read_step(case, text, neg_step, near, control=True)}
            results.append(rec)
            ax = rec["intervention"].get("candidate")
            cx = rec["control"].get("candidate")
            def word(a):
                return "no candidate within the bound" if not a else f"{a['class']} at {tuple(round(x, 2) for x in a['centroid'])}, {a['points']} pts, {a['distance_deg']:.1f} deg, overlap {a['overlap']['overlap']} {a['overlap']['shared']}/{a['overlap']['distinct_vertices']}"
            print(f"  {w} {index} at {step}: {rec['intervention']['primary']} | candidate {word(ax)} | control {neg_step}: {word(cx)}", flush=True)
    tally = collections.Counter((r["intervention"].get("candidate") or {}).get("class", "no candidate within the bound") for r in results)
    control_tally = collections.Counter((r["control"].get("candidate") or {}).get("class", "no candidate within the bound") for r in results)
    distinct_steps = len({(r["window"], r["intervention"]["step"]) for r in results})
    payload = {"generated_by": "scripts/characterize_contour_divergence.py", **launched,
               "brief": "BRIEF_contour_stage_characterization.md", "inputs_sha256": inputs,
               "candidate_selection": {"centroid_box_deg": MATCH_DEG, "search_bound_deg": NEAR_DEG,
                                       "standing": "heuristics that select a candidate; not geometric identity"},
               "vertex_tolerance_deg": VERTEX_TOL, "field_relative_tolerance": FIELD_REL_TOL,
               "classes_at_intervention_steps": dict(tally), "classes_at_control_steps": dict(control_tally),
               "items": len(results), "distinct_intervention_steps": distinct_steps,
               "fields_agree_within_tolerance_at_intervention": collections.Counter(str(r["intervention"]["fields_same"]) for r in results),
               "results": results, "processor_seconds": round(time.process_time() - cpu0, 1),
               "what_this_does_not_establish": [
                   "which program is right",
                   "anything about items not given to it",
                   "that the selected candidate is the population difference the merge acted on, or its cause",
                   "geometric identity of any two axes beyond the distinct-vertex test, or line identity from centroids",
                   "connectivity: that the port traced the segments between shared vertices, or joined the same crossings",
                   "step specificity: the same classes occur at the negative-control steps, where no track changed",
                   "exact field equality: masks equal and values within the stated relative tolerance"]}
    X.publish_json(args.out, payload, exclusive=True)
    print(f"classes at intervention steps: {dict(tally)} | at control steps: {dict(control_tally)} | items {len(results)}, distinct steps {distinct_steps} | {payload['processor_seconds']:.1f} processor-s")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
