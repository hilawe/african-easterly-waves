#!/usr/bin/env python3
"""The substitution pilot: version 1's captured merge input, in place of the port's, at one step.

THE OPERATION, FROZEN AS A DATED AMENDMENT TO THE PHASE B PROCEDURE BEFORE ANY RUN. At one
timestep the port's detected trough axes are REPLACED by version 1's captured axes at that
step, in the order version 1's merge received them, and the port's own merge, association
and finalization then run on them. Nothing else changes at any other step. The frozen
phase B injection APPENDS version 1's axes to the port's, and window E item 3 showed that
appending does not produce the candidate that substitution does.

WHAT IS SUBSTITUTED, STATED AT ITS TRUE WIDTH. The capture holds `AXISPTS` (each axis's
vertices, written to FOUR DECIMALS) and `POTWV` (each potential wave's center at full
precision, with the index version 1's merge loop used). What the port's merge is handed is
the captured VERTICES, in `POTWV` order, and it computes each center itself as the mean of
those rounded vertices. That center can differ from version 1's full-precision center by up
to 5e-5 degrees. This script requires, at every step it uses, that the `POTWV` records are
a complete contiguous sequence, that every coordinate is finite, that the counts match, and
that every derived center agrees with its `POTWV` center to within 1e-4, and it records the
largest deviation it saw. It does NOT claim to hand the merge version 1's exact input. A
capture written at full precision would close that gap and is a separate change.

TIMESTEPS ARE RESOLVED, NOT ASSUMED. A capture or discrepancy time is used only if it equals
exactly one of the case's timesteps, the same resolved value is used for the screen and the
replay, and the edit is counted: an intervention or control replay whose edit did not fire
exactly once is ANOMALOUS. A review showed a capture at a time between two timesteps
passing the screen (which reads the nearest field) and never firing in the replay (which
requires equality), so a control that never acted was credited.

INPUTS ARE READ ONCE, AS BYTES. Each input file is read into memory, digested from those
bytes, and parsed from those bytes, and the digests are what the retained runs and the
artifact carry. Nothing re-reads a path after the run has started.

CANDIDATE STEPS are the item's discrepancy times (reference-extra, port-extra and displaced)
that have a usable capture, in time order. Each is SCREENED first: a substitution that
leaves the detected candidate signature unchanged cannot change the finished tracks and is
not replayed. Each survivor is replayed once. The first exact reproduction stops the search.

CONTROLS. The REPRESENTATION control is an identity edit at the success step and must
reproduce the baseline exactly. The NEGATIVE control is the same substitution at a usable
captured step OUTSIDE the item's discrepancy times and their neighborhood, chosen BEFORE
the search as the earliest such step where the screen shows the substitution changes the
detected signature, so the control is an operation that acts. It must not reproduce the
reference. The eligible steps and which of them act are recorded before the search, so a
capture that would change the choice is visible. One non-reproducing control supports the
comparison with that step, and no wider claim.

OUTCOMES: EXPLAINED (exact reproduction, both controls executed and passed), UNEXPLAINED
(every candidate step screened out or replayed without reproduction, or the negative
control also reproduces), UNSUPPORTED (no usable candidate step, or no negative step
available), ANOMALOUS (the identity edit does not reproduce the baseline, or an edit fired
other than exactly once). Every item records which controls were available, selected,
executed and passed. NONE IS AN ACCOUNTING CREDIT, and no phase B result changes.

    .venv/bin/python3 scripts/substitution_pilot.py <work> --items E:3,B:25 --out <artifact> --retain <dir>
"""

import argparse
import collections
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402
import validation_phase_b as B  # noqa: E402
from residue_membership import derived_discrepancies  # noqa: E402

OPERATION = ("substitute version 1's captured axis vertices (AXISPTS, four decimals) in the "
             "order of its merge input (POTWV) for the port's detected axes at one step")
# THE MERGE-INPUT SUBSTITUTION, separately named. The port's merge reads only each
# candidate's center (lat_mean, lon_mean, time), computed in `detect_troughs` as the mean
# of an axis's vertices. A single-point axis therefore hands the merge that point EXACTLY,
# so version 1's full-precision POTWV centers can be supplied as they were recorded, in
# their order, with no rounding in between. Whether they reach the merge exactly is not
# assumed: a spy on the merge records what it received at the step and the run is
# ANOMALOUS unless it equals the supplied centers float for float.
OPERATION_CENTERS = ("substitute version 1's captured merge-input centers (POTWV, full "
                     "precision, in its order) as single-point axes for the port's detected "
                     "axes at one step, so the port's merge receives those centers exactly")
CENTER_TOLERANCE_DEG = 1e-4     # the four-decimal rounding of the vertices bounds this at 5e-5
INPUT_NAMES = ("tracker_case.mat", "residuals.json", "tracker_octave_instrumented.mat",
               "tracker_port.mat", "axes_capture.log")


def parse_capture(text):
    """Version 1's axes and merge-input centers per step, from the capture's text.

    Returns (axes, pot, problems): axes[key] is the AXISPTS list in file order as
    (lat_array, lon_array), pot[key] is the list of (index, lat, lon) POTWV records."""
    axes, pot = collections.defaultdict(list), collections.defaultdict(list)
    problems = collections.defaultdict(list)     # keyed by the RAW time, or "*" when unreadable
    for n, line in enumerate(text.splitlines(), 1):
        p = line.split()
        if not p or p[0] not in ("AXISPTS", "POTWV"):
            continue
        # THE TIME IS KEPT RAW. A first version rounded it to four decimals here, so a
        # record at 3.00004 became "3.0000" and resolved to the grid step 3.0, and a
        # control could act on data that was never at that step.
        try:
            raw = float(p[1])
        except (ValueError, IndexError):
            problems["*"].append(f"line {n}: {p[0]} record with no readable time")
            continue
        try:
            if p[0] == "AXISPTS":
                count = int(p[2])
                values = [float(x) for x in p[3:]]
                if len(values) != 2 * count or count < 1:
                    raise ValueError(f"declares {count} points and carries {len(values)} values")
                axes[raw].append((np.array(values[0::2]), np.array(values[1::2])))
            else:
                pot[raw].append((int(p[2]), float(p[3]), float(p[4])))
        except (ValueError, IndexError) as e:
            # A MALFORMED RECORD DISQUALIFIES ITS STEP. Dropping the line and keeping the
            # step let a capture with one bad axis still earn a credit at that step.
            problems[raw].append(f"line {n}: unreadable {p[0]} record ({e})")
    return dict(axes), dict(pot), dict(problems)


def merge_spy(step, expected):
    """A wrapper for the port's merge that records the FIRST-pass candidates it received at
    `step` and whether they equal `expected` (a list of (lat, lon)) exactly."""
    original = B.D.merge_contours
    seen = {"calls_at_step": 0, "received": None, "exact": None}

    def spy(candidates, *args, **kwargs):
        if candidates and all("region" not in c for c in candidates) \
                and candidates[0].get("time") is not None \
                and abs(float(candidates[0]["time"]) - step) < 1e-9:
            seen["calls_at_step"] += 1
            if seen["calls_at_step"] == 1:
                got = [(float(c["lat_mean"]), float(c["lon_mean"])) for c in candidates]
                seen["received"] = len(got)
                seen["exact"] = (len(got) == len(expected)
                                 and all(g[0] == e[0] and g[1] == e[1]
                                         for g, e in zip(got, expected)))
        return original(candidates, *args, **kwargs)
    return spy, seen


def ordered_capture(text, times=None):
    """Version 1's axes per GRID step in merge-input order, and why a step is unusable.

    Every raw capture time must equal exactly one case timestep when `times` is given,
    and a raw time that resolves to no step, or to a step another raw time also resolves
    to, is refused. A step with any malformed record is refused. A step is usable only if
    its POTWV indices are exactly 1..n with no gap or repeat, its AXISPTS count is n, every
    coordinate on both sides is finite, and the center the port will compute from each
    axis's rounded vertices agrees with the POTWV center at that index to within
    CENTER_TOLERANCE_DEG. Keys are the resolved grid steps formatted to four decimals, the
    largest deviation per step is returned, and if the capture has records whose time
    cannot be read at all the whole capture is refused."""
    axes, pot, problems = parse_capture(text)
    usable, refused, deviation = {}, {}, {}
    if "*" in problems:
        return {}, {"_capture": problems["*"]}, {}
    resolved = {}
    for raw in sorted(set(axes) | set(pot) | set(problems), key=float):
        step = resolve_step(times, raw) if times is not None else float(raw)
        if step is None:
            refused[f"raw {raw!r}"] = "no case timestep equals this capture time exactly"
            continue
        resolved.setdefault(step, []).append(raw)
    for step, raws in resolved.items():
        key = f"{step:.4f}"
        if len(raws) > 1:
            refused[key] = f"several capture times {raws} resolve to this one step"
            continue
        raw = raws[0]
        if raw in problems:
            refused[key] = "; ".join(problems[raw])
            continue
        entries, records = axes.get(raw, []), sorted(pot.get(raw, []))
        indices = [r[0] for r in records]
        if indices != list(range(1, len(records) + 1)):
            refused[key] = f"POTWV indices {indices[:6]}... are not 1..{len(records)} without gap or repeat"
            continue
        if len(entries) != len(records):
            refused[key] = f"{len(entries)} AXISPTS axes and {len(records)} POTWV records"
            continue
        if not all(np.all(np.isfinite(a)) and np.all(np.isfinite(b)) for a, b in entries) \
                or not all(np.isfinite(r[1]) and np.isfinite(r[2]) for r in records):
            refused[key] = "a coordinate is not finite"
            continue
        worst = 0.0
        for (a, b), (_, la, lo) in zip(entries, records):
            worst = max(worst, abs(float(np.mean(a)) - la), abs(float(np.mean(b)) - lo))
        if worst > CENTER_TOLERANCE_DEG:
            refused[key] = (f"a center derived from the rounded vertices differs from its "
                            f"POTWV center by {worst:.2e} degrees, beyond {CENTER_TOLERANCE_DEG}")
            continue
        usable[key] = [(np.asarray(a, float), np.asarray(b, float)) for a, b in entries]
        deviation[key] = worst
    return usable, refused, deviation


def ordered_centers(text, times=None):
    """Version 1's full-precision merge-input centers per usable grid step, in order."""
    axes, pot, _ = parse_capture(text)
    usable, refused, _ = ordered_capture(text, times)
    centers = {}
    for key in usable:
        step = float(key)
        raw = next(r for r in pot if (resolve_step(times, r) if times is not None else float(r)) == step)
        centers[key] = [(la, lo) for _, la, lo in sorted(pot[raw])]
    return centers, refused


def as_single_point_axes(centers):
    return [(np.array([la], float), np.array([lo], float)) for la, lo in centers]


def port_centers(case, step):
    """The port's own merge input at a step, as the centers it would compute."""
    return [(float(np.mean(a)), float(np.mean(b))) for a, b in B.axes_at(case, step)]


def load_window_snapshot(work, name):
    """The window's inputs read ONCE as bytes, digested from those bytes, parsed from them."""
    from scipy.io import loadmat
    year, month = B.WINDOWS[name]
    directory = os.path.join(work, f"{name}_{year}_{month}")
    raw, digests = {}, {}
    for file in INPUT_NAMES:
        with open(os.path.join(directory, file), "rb") as fh:
            raw[file] = fh.read()
        digests[file] = hashlib.sha256(raw[file]).hexdigest()
    mat = loadmat(io.BytesIO(raw["tracker_case.mat"]))
    case = {k: np.asarray(v) for k, v in mat.items() if not k.startswith("__")}
    residuals = json.loads(raw["residuals.json"].decode())
    reference = loadmat(io.BytesIO(raw["tracker_octave_instrumented.mat"]))
    port = loadmat(io.BytesIO(raw["tracker_port.mat"]))
    capture_text = raw["axes_capture.log"].decode(errors="replace")
    return directory, case, residuals, reference, port, capture_text, digests


def resolve_step(times, value):
    """The one case timestep equal to `value`, or None when there is none or several."""
    hits = [float(t) for t in times if abs(float(t) - float(value)) < 1e-9]
    return hits[0] if len(hits) == 1 else None


def item_discrepancy(reference, port, item):
    t, la, lo = B.reference_track(reference, item["index"])
    if item["port_index"] is None:
        return {"v1_extra": [f"{float(x):.4f}" for x in t], "port_extra": [], "displaced": []}
    q = item["port_index"]
    counterpart = {k: np.asarray(port[f"{k}{q}"], float).ravel() for k in ("time", "lat", "lon")}
    return derived_discrepancies({"time": t, "lat": la, "lon": lo}, counterpart)


def counted_substitution(axes):
    """An edit that replaces the axes and counts how many times it fired."""
    hits = [0]

    def edit(out):
        hits[0] += 1
        return list(axes)
    return edit, hits


def counted_identity():
    hits = [0]

    def edit(out):
        hits[0] += 1
        return list(out)
    return edit, hits


def investigate(case, reference, item, disc, baseline, capture, times, mode="axes"):
    """One item under the frozen substitution operation. Returns the record and the runs.

    `mode` is "axes" (the rounded vertices, the pilot as first run) or "centers" (the
    full-precision merge-input centers as single-point axes, boundary-verified)."""
    operation = OPERATION if mode == "axes" else OPERATION_CENTERS

    def payload_for(key):
        return capture[key] if mode == "axes" else as_single_point_axes(capture[key])
    index = item["index"]
    t, la, lo = B.reference_track(reference, index)
    all_disc = sorted({float(x) for v in disc.values() for x in v})
    unresolved = [s for s in all_disc if resolve_step(times, s) is None]
    resolved_disc = [resolve_step(times, s) for s in all_disc if resolve_step(times, s) is not None]
    capture_steps = {}
    for key in capture:
        r = resolve_step(times, float(key))
        if r is not None:
            capture_steps[r] = key
    candidates = [s for s in resolved_disc if s in capture_steps]
    excluded = set(B.search_steps([f"{s:.4f}" for s in resolved_disc], times))
    controls = {"representation": {"available": None, "executed": False, "passed": None},
                "negative": {"available": False, "selected_step": None, "executed": False,
                             "passed": None, "eligible_steps": []}}
    out = {"index": index, "kind": item["kind"], "operation": operation, "mode": mode,
           "discrepancy_counts": {k: len(v) for k, v in disc.items()},
           "discrepancy_times_not_on_the_grid": unresolved,
           "candidate_steps": candidates, "steps_tried": [], "screened_out": [],
           "controls": controls, "achieved_grade": "none"}
    runs = {"baseline": baseline}
    cost0, wall0 = B._cost_snapshot(), time.perf_counter()

    def finish(outcome, why=None):
        out["outcome"] = outcome
        if why:
            out["why"] = why
        cost = B._cost_since(cost0)
        out["replays_after_baseline"], out["screens"] = cost["replay_calls"], cost["screen_calls"]
        out["processor_seconds"] = round(cost["replay_seconds"] + cost["screen_seconds"], 1)
        out["elapsed_seconds"] = round(time.perf_counter() - wall0, 1)
        return out, runs

    if X.holds_exactly(baseline, t, la, lo):
        return finish("ANOMALOUS", "the untouched baseline already reproduces this track")
    if not candidates:
        return finish("UNSUPPORTED", "no candidate step of this item has a usable capture on "
                                     "the grid, so the operation cannot be attempted")
    # THE NEGATIVE CONTROL IS CHOSEN BEFORE THE SEARCH, and the eligible population is
    # recorded, so a capture that would change the choice is visible in the record.
    for s in sorted(capture_steps):
        if s in excluded:
            continue
        plain = B.detect_at(case, s)
        changed = B.detect_at(case, s, counted_substitution(payload_for(capture_steps[s]))[0])
        acts = changed != plain
        controls["negative"]["eligible_steps"].append({"step": s, "acts": acts})
        if acts and controls["negative"]["selected_step"] is None:
            controls["negative"]["selected_step"] = s
            controls["negative"]["available"] = True
            controls["negative"]["operation"] = {
                "step": s, "operation": operation,
                "detected_signature_entries_changed": sum(
                    1 for a, b in zip(plain, changed) if a != b) + abs(len(plain) - len(changed)),
                "candidates_before": len(plain), "candidates_after": len(changed)}
    if controls["negative"]["selected_step"] is None:
        return finish("UNSUPPORTED", "no usable captured step outside the item's discrepancy "
                                     "neighborhood changes the detected candidates under "
                                     "substitution, so no negative control is available")
    found = None
    for s in candidates:
        key = capture_steps[s]
        plain = B.detect_at(case, s)
        if B.detect_at(case, s, counted_substitution(payload_for(key))[0]) == plain:
            out["screened_out"].append(s)
            continue
        out["steps_tried"].append(s)
        edit, hits = counted_substitution(payload_for(key))
        if mode == "centers":
            spy, seen = merge_spy(s, capture[key])
            original_merge = B.D.merge_contours
            B.D.merge_contours = spy
            try:
                final = B.replay(case, B.edit_hook(s, edit))
            finally:
                B.D.merge_contours = original_merge
            out.setdefault("merge_boundary", {})[f"{s:.4f}"] = seen
            if not seen["exact"]:
                return finish("ANOMALOUS", f"the merge at {s} did not receive the supplied "
                                           f"centers exactly ({seen})")
        else:
            final = B.replay(case, B.edit_hook(s, edit))
        if hits[0] != 1:
            return finish("ANOMALOUS", f"the substitution at {s} fired {hits[0]} times in the "
                                       f"replay rather than exactly once")
        if X.holds_exactly(final, t, la, lo):
            found = (s, final)
            break
    if found is None:
        return finish("UNEXPLAINED", "substitution at every usable candidate step either left "
                                     "detection unchanged or did not reproduce the track")
    s, final = found
    runs["intervention"] = final
    out["intervention"] = {"step": s, "operation": operation, "edit_fired": 1,
                           "axes_substituted": len(capture[capture_steps[s]]),
                           "port_axes_replaced": len(B.axes_at(case, s))}
    controls["representation"]["available"] = True
    if mode == "centers":
        # THE REPRESENTATION CONTROL FOR THE CENTERS PATH is the port's OWN merge input
        # carried through the same single-point mechanism, which must reproduce the
        # baseline exactly, so the mechanism itself is shown to move nothing.
        edit, hits = counted_substitution(as_single_point_axes(port_centers(case, s)))
        controls["representation"]["operation"] = ("the port's own centers at the step as "
                                                   "single-point axes")
    else:
        edit, hits = counted_identity()
        controls["representation"]["operation"] = "identity edit at the step"
    rep = B.replay(case, B.edit_hook(s, edit))
    controls["representation"]["executed"] = True
    controls["representation"]["step"] = s
    controls["representation"]["edit_fired"] = hits[0]
    controls["representation"]["passed"] = (hits[0] == 1
                                            and X.canonical(rep) == X.canonical(baseline))
    runs["representation_control"] = rep
    if hits[0] != 1:
        return finish("ANOMALOUS", f"the identity edit at {s} fired {hits[0]} times")
    if not controls["representation"]["passed"]:
        return finish("ANOMALOUS", "the identity edit at the intervention step does not reproduce "
                                   "the baseline, so the apparatus moves the run by itself")
    ns = controls["negative"]["selected_step"]
    edit, hits = counted_substitution(payload_for(capture_steps[ns]))
    neg = B.replay(case, B.edit_hook(ns, edit))
    runs["negative_control"] = neg
    controls["negative"]["executed"] = True
    controls["negative"]["edit_fired"] = hits[0]
    controls["negative"]["reproduces_reference"] = X.holds_exactly(neg, t, la, lo)
    controls["negative"]["equals_baseline"] = X.canonical(neg) == X.canonical(baseline)
    controls["negative"]["passed"] = (hits[0] == 1
                                      and not controls["negative"]["reproduces_reference"])
    if hits[0] != 1:
        return finish("ANOMALOUS", f"the negative control's edit at {ns} fired {hits[0]} times")
    if controls["negative"]["reproduces_reference"]:
        return finish("UNEXPLAINED", "the negative control also reproduces the reference, so "
                                     "the reproduction is not specific to the intervention step")
    out["achieved_grade"] = ("two-control: exact reproduction at the intervention step, the "
                             "identity edit there reproduces the baseline, and the same "
                             "substitution at the selected unrelated acting step does not "
                             "reproduce the reference. Not an accounting credit.")
    return finish("EXPLAINED")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("work")
    ap.add_argument("--items", required=True, help="comma-separated WINDOW:INDEX, e.g. E:3,B:25")
    ap.add_argument("--out", required=True)
    ap.add_argument("--retain", required=True)
    ap.add_argument("--replay-ceiling", type=int, default=None,
                    help="a hard ceiling on full replays for the whole command, baselines "
                         "included. An item is not started if the replays already made plus "
                         "its worst case (candidate steps and two controls) would exceed it, "
                         "and is recorded as NOT_RUN")
    ap.add_argument("--operation", choices=("axes", "centers"), default="axes",
                    help="axes: the rounded vertices in merge-input order (the pilot). "
                         "centers: the full-precision merge-input centers as single-point "
                         "axes, boundary-verified")
    args = ap.parse_args(argv)
    operation = OPERATION if args.operation == "axes" else OPERATION_CENTERS
    launched = {"pilot_sha256": X.digest(__file__),
                "driver_sha256": X.digest(B.__file__),
                "git_head_at_launch": X.repository_head(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "launched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    wanted = collections.defaultdict(list)
    for token in args.items.split(","):
        w, i = token.strip().split(":")
        wanted[w].append(int(i))
    os.makedirs(args.retain, exist_ok=True)
    results, retained_files, inputs, capture_report = [], {}, {}, {}
    total0, wall0, cpu0 = B._cost_snapshot(), time.perf_counter(), time.process_time()
    for w in sorted(wanted):
        directory, case, residuals, reference, port, capture_text, digests = \
            load_window_snapshot(args.work, w)
        inputs[w] = digests
        capture, refused_steps, deviation = ordered_capture(capture_text, case["time"].ravel())
        if args.operation == "centers":
            capture, refused_steps = ordered_centers(capture_text, case["time"].ravel())
        capture_report[w] = {"usable_steps": sorted(capture, key=float),
                             "refused_steps": refused_steps,
                             "largest_center_deviation_deg": max(deviation.values(), default=None)}
        items = {i["index"]: i for i in B.build_items(residuals, reference, case)}
        unknown = [i for i in wanted[w] if i not in items]
        if unknown:
            raise SystemExit(f"REFUSED: window {w} has no residual items {unknown}")
        baseline = B.replay(case)
        provenance = {"script_sha256": launched["pilot_sha256"],
                      "git_head": launched["git_head_at_launch"],
                      "case_sha256": digests["tracker_case.mat"],
                      "reference_output_sha256": digests["tracker_octave_instrumented.mat"]}

        def retain(name, runs, index, extra):
            path = os.path.join(args.retain, name)
            try:
                X.save_run(path, runs, os.path.join(directory, "tracker_case.mat"),
                           os.path.join(directory, "tracker_octave_instrumented.mat"),
                           index, __file__,
                           extra={"window": w, "inputs_sha256": digests, **launched, **extra},
                           exclusive=True, **provenance)
            except FileExistsError:
                raise SystemExit(f"REFUSED: {path} exists and retained evidence is never "
                                 f"overwritten. Use a fresh retention directory")
            retained_files[name] = X.digest(path)

        retain(f"{w}_baseline_pilot.json", {"baseline": baseline}, -1,
               {"role": "baseline replayed by this pilot"})
        for index in sorted(wanted[w]):
            item = items[index]
            disc = item_discrepancy(reference, port, item)
            if args.replay_ceiling is not None:
                # THE CEILING IS ENFORCED BEFORE AN ITEM STARTS, from the worst case it could
                # cost, so no item is cut off half way and the ceiling is never exceeded.
                worst = len([s for s in {float(x) for v in disc.values() for x in v}
                             if resolve_step(case["time"].ravel(), s) is not None
                             and f"{resolve_step(case['time'].ravel(), s):.4f}" in capture]) + 2
                made = B._cost_since(total0)["replay_calls"]
                if made + worst > args.replay_ceiling:
                    results.append({"index": index, "kind": item["kind"], "window": w,
                                    "outcome": "NOT_RUN", "operation": operation,
                                    "why": f"the replay ceiling of {args.replay_ceiling} would be "
                                           f"exceeded: {made} made, up to {worst} more needed",
                                    "candidate_steps": [], "steps_tried": [], "screened_out": [],
                                    "replays_after_baseline": 0, "screens": 0,
                                    "processor_seconds": 0.0, "elapsed_seconds": 0.0,
                                    "achieved_grade": "none", "controls": {}})
                    print(f"  {w} item {index}: NOT_RUN, replay ceiling", flush=True)
                    continue
            record, runs = investigate(case, reference, item, disc, baseline, capture,
                                       case["time"].ravel(), mode=args.operation)
            record["window"] = w
            results.append(record)
            retain(f"{w}_item{index}_pilot.json", runs, index,
                   {"operation": operation, "outcome": record["outcome"],
                    "merge_boundary": record.get("merge_boundary"),
                    "intervention": record.get("intervention"),
                    "controls": record["controls"], "achieved_grade": record["achieved_grade"]})
            print(f"  {w} item {index} {item['kind']}: {record['outcome']} "
                  f"(candidate steps {len(record['candidate_steps'])}, tried "
                  f"{len(record['steps_tried'])}, replays after baseline "
                  f"{record['replays_after_baseline']}, {record['processor_seconds']:.0f} s "
                  f"processor, {record['elapsed_seconds']:.0f} s elapsed)", flush=True)
    tally = collections.Counter(r["outcome"] for r in results)
    total = B._cost_since(total0)
    payload = {
        "generated_by": "scripts/substitution_pilot.py",
        "amendment": "phase B substitution amendment of 2026-09-24, pilot only",
        "operation": operation,
        "what_is_substituted": (
            "version 1's captured axis VERTICES, written to four decimals, in the order of "
            "its merge input (POTWV), handed to the port's own merge, which computes each "
            "center from those rounded vertices. Every used step's derived centers agree "
            "with the POTWV centers to within 1e-4 degrees and the largest deviation seen is "
            "recorded per window. This is not version 1's exact merge input."
            if args.operation == "axes" else
            "version 1's captured merge-input CENTERS (POTWV) at full precision, in its "
            "order, as single-point axes, so the port's merge receives exactly those "
            "centers. That they reached the merge exactly is verified by a spy on the merge "
            "during every intervention replay and recorded per step."),
        "protocol": ("EXPLAINED requires exact reproduction at the intervention step, an "
                     "identity edit there reproducing the baseline, and the same "
                     "substitution at the pre-selected unrelated acting step not reproducing "
                     "the reference, each edit fired exactly once. Each item's achieved "
                     "grade and control states are in its record. Nothing here is an "
                     "accounting credit, and no phase B result changes."),
        **launched,
        "inputs_sha256": inputs,
        "capture": capture_report,
        "items": {w: sorted(v) for w, v in wanted.items()},
        "replay_ceiling": args.replay_ceiling,
        "outcomes": {k: tally[k] for k in sorted(tally)},
        "results": results,
        "retained_files": retained_files,
        "retention_directory": os.path.abspath(args.retain),
        "timing": {"processor_seconds": round(time.process_time() - cpu0, 1),
                   "elapsed_seconds": round(time.perf_counter() - wall0, 1),
                   "replays_including_baselines": total["replay_calls"],
                   "replays_after_baselines": sum(r["replays_after_baseline"] for r in results),
                   "screens": total["screen_calls"],
                   "replay_seconds": total["replay_seconds"],
                   "screen_seconds": total["screen_seconds"]},
    }
    try:
        X.publish_json(args.out, payload, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {args.out} exists and artifacts are never overwritten")
    print(f"pilot: {len(results)} items, {json.dumps(payload['outcomes'])}, "
          f"{payload['timing']['replays_including_baselines']} replays including baselines, "
          f"{payload['timing']['processor_seconds']/60:.1f} processor-min, "
          f"{payload['timing']['elapsed_seconds']/60:.1f} elapsed-min")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
