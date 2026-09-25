#!/usr/bin/env python3
"""One matched pair's vertex list, version 1's for the port's, at one step.

THE QUESTION. The whole-population comparison of the four unclassified cases found that
in three of them the version-1 axis nearest the produced observation and the port axis
nearest it share four points, and the port's line runs on past the point where version
1's ends and repeats its endpoint. Whether that difference is what the merge acts on was
not established, since no replay was run. This script asks it, and nothing wider.

THE OPERATION. At the step, the port's ordered axis population is kept as it is except
that the paired port axis's vertex list is replaced by version 1's captured vertex list
(`AXISPTS`, four decimals) for its paired axis, at the port axis's own position. Every
other axis is untouched. The edit refuses to act unless the population it is handed has
the expected count and the axis at that position carries exactly the vertex arrays the
pair was chosen against. That binds the target of the replacement and does not
establish that the rest of the population is unchanged.

WHAT THE MERGE RECEIVES is recorded rather than assumed. The port's merge reads only each
candidate's center, the mean of its vertices, so the replacement reaches everything
downstream through that one number. A spy on the merge records the centers it received
at the step, and the run is ANOMALOUS unless the center at the replaced position equals
the mean of the supplied vertices exactly. The deviation of that center from version 1's
full-precision `POTWV` center for the same axis is recorded. The exact `POTWV` center is
also screened as a single-point axis at the same position and replayed only if its
detected signature differs from the vertex list's, as the precision reading.

WHAT THIS TESTS. The selected axis representation. Replacing the vertex list changes the
line's extent and the weighting of its centroid together. A reproduction establishes that
this representation change at this one axis is sufficient under the tested conditions. It
does not identify the extra endpoints, the repeats or the extent as the cause.

ONE INTERVENTION, SEVERAL OUTCOMES. Items that share a step and an observation share the
pair and the replay, and each item's track is judged from that one replay.

DETECTION FIRST. If the replacement leaves the detected candidate signature unchanged it
changes nothing consumed downstream, and the intervention stops with no replay.

CONTROLS, BOTH ALWAYS RUN. The representation control is an identity edit at the step and
must reproduce the baseline exactly. The negative control is the same operation at a
captured step outside every paired item's discrepancy times and their neighborhood,
anchored on the same observation position, the earliest such step where the screen shows
the replacement acts. It must not reproduce any paired item's track. The negative step is
chosen before the intervention replay, and the eligible steps are recorded.

CEILING. A hard number of full replays for the whole command, baselines included, enforced
before each replay. Inputs are read once as bytes and digested, retention is exclusive,
and nothing here is an accounting credit.

    .venv/bin/python3 scripts/pair_vertex_substitution.py <work> \\
        --interventions "C:34882.75:23,27;E:39262.5:17" \\
        --record docs/aewc_v2/artifacts/contour_divergence_wholepop_2026-09-25.json \\
        --out <artifact> --retain <dir> --replay-ceiling 10
"""

import argparse
import collections
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402
import substitution_pilot as S  # noqa: E402
import validation_phase_b as B  # noqa: E402

OPERATION = ("replace, at one step, the vertex list of the port's axis nearest the produced "
             "observation with version 1's captured vertex list (AXISPTS, four decimals) of "
             "its axis nearest that observation, at the port axis's position in the port's "
             "ordered population, leaving every other axis unchanged")
PAIRING = ("the nearest axis on each side to the observation by the sum of absolute latitude "
           "and longitude differences, as the whole-population comparison recorded it")
RECORD_TOL = 1e-9         # the computed pair must reproduce the recorded centroids to this
NEAR_MERGED_DEG = 6.0     # merged candidates within this of the observation are listed


class ReplayCeiling(Exception):
    pass


class EditTargetMismatch(Exception):
    pass


def centroid(axis):
    return (float(np.mean(axis[0])), float(np.mean(axis[1])))


def l1(c, obs):
    return abs(c[0] - obs[0]) + abs(c[1] - obs[1])


def nearest_index(axes, obs):
    """The index of the axis whose centroid is nearest `obs`, and how many tie for it."""
    if not axes:
        return None, 0
    d = [l1(centroid(a), obs) for a in axes]
    best = min(d)
    ties = sum(1 for x in d if abs(x - best) < 1e-12)
    return d.index(best), ties


def pair_at(port_axes, v1_axes, obs):
    """The matched pair at one step under the recorded pairing rule."""
    kp, tp = nearest_index(port_axes, obs)
    kv, tv = nearest_index(v1_axes, obs)
    if kp is None or kv is None:
        return None
    return {"rule": PAIRING,
            "port": {"position": kp, "of": len(port_axes), "centroid": centroid(port_axes[kp]),
                     "points": int(len(port_axes[kp][0])), "ties": tp,
                     "distance_deg": l1(centroid(port_axes[kp]), obs),
                     "lat": [float(x) for x in port_axes[kp][0]],
                     "lon": [float(x) for x in port_axes[kp][1]]},
            "v1": {"position": kv, "of": len(v1_axes), "centroid": centroid(v1_axes[kv]),
                   "points": int(len(v1_axes[kv][0])), "ties": tv,
                   "distance_deg": l1(centroid(v1_axes[kv]), obs),
                   "lat": [float(x) for x in v1_axes[kv][0]],
                   "lon": [float(x) for x in v1_axes[kv][1]]}}


def pair_reproduces_record(pair, record):
    """Whether the computed pair equals the recorded nearest axes on both sides."""
    if record is None:
        return False, "no recorded comparison for this item"
    rec = record.get("nearest_axis_any_side") or {}
    problems = []
    for side in ("port", "v1"):
        got = pair[side]
        want = rec.get(side)
        if not want:
            problems.append(f"the record has no nearest {side} axis")
            continue
        _, (la, lo), n = want
        if abs(got["centroid"][0] - la) > RECORD_TOL or abs(got["centroid"][1] - lo) > RECORD_TOL:
            problems.append(f"{side} centroid {got['centroid']} differs from the recorded ({la}, {lo})")
        if got["points"] != n:
            problems.append(f"{side} axis has {got['points']} points, the record says {n}")
    return (not problems), "; ".join(problems)


def replacement_edit(position, expected_count, expected_axis, vertices):
    """An edit that puts `vertices` at `position` and counts how many times it fired.

    It refuses, by raising, unless the population it is handed has `expected_count` axes
    and the axis at `position` carries exactly the vertex arrays of `expected_axis`, so
    the replacement can only land on the axis the pair was chosen against. A review
    showed the earlier centroid check accepted a different axis with the same centroid.
    Neither check establishes that the other axes are the ones the pair was chosen
    beside, and no claim here rests on that."""
    hits = [0]
    lat, lon = np.asarray(vertices[0], float), np.asarray(vertices[1], float)
    want_lat, want_lon = np.asarray(expected_axis[0], float), np.asarray(expected_axis[1], float)

    def edit(out):
        hits[0] += 1
        if len(out) != expected_count:
            raise EditTargetMismatch(f"handed {len(out)} axes, expected {expected_count}")
        if not (np.array_equal(np.asarray(out[position][0], float), want_lat)
                and np.array_equal(np.asarray(out[position][1], float), want_lon)):
            raise EditTargetMismatch(f"the axis at {position} does not carry the vertex arrays "
                                     f"the pair was chosen against (centroid "
                                     f"{centroid(out[position])})")
        new = list(out)
        new[position] = (lat.copy(), lon.copy())
        return new
    return edit, hits


def observation_at(reference, index, step):
    t, la, lo = B.reference_track(reference, index)
    m = np.abs(t - step) < 1e-9
    if m.sum() != 1:
        return None
    return (float(la[m][0]), float(lo[m][0]))


def merged_near(signature, obs):
    return [[round(l1((s[1], s[2]), obs), 4), s[1], s[2]] for s in signature
            if l1((s[1], s[2]), obs) <= NEAR_MERGED_DEG]


def signature_difference(plain, changed):
    return {"entries_changed": sum(1 for a, b in zip(plain, changed) if a != b)
            + abs(len(plain) - len(changed)),
            "candidates_before": len(plain), "candidates_after": len(changed)}


class Budget:
    """The replay ceiling, enforced BEFORE each replay, over the whole command."""

    def __init__(self, ceiling):
        self.ceiling, self.made = ceiling, 0

    def replay(self, case, hook=None):
        if self.ceiling is not None and self.made + 1 > self.ceiling:
            raise ReplayCeiling(f"the replay ceiling of {self.ceiling} would be exceeded")
        self.made += 1
        return B.replay(case, hook)


def investigate(case, reference, port, items, step, capture, centers, times, record,
                baseline, budget):
    """One intervention: the pair at `step`, judged for every item in `items`."""
    key = f"{step:.4f}"
    out = {"step": step, "items": [i["index"] for i in items],
           "kinds": {str(i["index"]): i["kind"] for i in items},
           "operation": OPERATION, "outcomes": {}, "controls": {
               "representation": {"available": None, "executed": False, "passed": None},
               "negative": {"available": False, "selected_step": None, "executed": False,
                            "passed": None, "eligible_steps": []}}}
    runs = {"baseline": baseline}
    cost0, wall0 = B._cost_snapshot(), time.perf_counter()
    tracks = {i["index"]: B.reference_track(reference, i["index"]) for i in items}

    def finish(outcome_all, why=None):
        for i in items:
            out["outcomes"].setdefault(str(i["index"]), outcome_all)
        if why:
            out["why"] = why
        cost = B._cost_since(cost0)
        out["replays_after_baseline"], out["screens"] = cost["replay_calls"], cost["screen_calls"]
        out["processor_seconds"] = round(cost["replay_seconds"] + cost["screen_seconds"], 1)
        out["elapsed_seconds"] = round(time.perf_counter() - wall0, 1)
        return out, runs

    for i in items:
        if X.holds_exactly(baseline, *tracks[i["index"]]):
            return finish("ANOMALOUS", f"the untouched baseline already reproduces item {i['index']}")
    observations = {i["index"]: observation_at(reference, i["index"], step) for i in items}
    if any(o is None for o in observations.values()) or len({observations[k] for k in observations}) != 1:
        return finish("UNSUPPORTED", f"the items do not share one observation at {step}: {observations}")
    obs = next(iter(observations.values()))
    out["observation"] = list(obs)
    if key not in capture:
        return finish("UNSUPPORTED", f"no usable capture at {step}")
    port_axes, v1_axes = B.axes_at(case, step), capture[key]
    pair = pair_at(port_axes, v1_axes, obs)
    if pair is None or pair["port"]["ties"] != 1 or pair["v1"]["ties"] != 1:
        return finish("UNSUPPORTED", f"the pair cannot be formed without a tie: {pair}")
    ok, why = pair_reproduces_record(pair, record)
    if not ok:
        return finish("UNSUPPORTED", "the computed pair does not reproduce the recorded comparison: " + why)
    kp, kv = pair["port"]["position"], pair["v1"]["position"]
    exact_center = centers[key][kv]
    pair["v1"]["potwv_center_full_precision"] = list(exact_center)
    out["pair"] = pair

    # THE DETECTION-STAGE CHECK FIRST, before any replay and before a control is chosen.
    # A replacement that changes nothing the merge produces needs no control, and a first
    # run ordered the negative selection ahead of this and left one intervention without
    # its detection reading when no unrelated step acted.
    plain = B.detect_at(case, step)
    edit, _ = replacement_edit(kp, len(port_axes), (pair["port"]["lat"], pair["port"]["lon"]),
                               (pair["v1"]["lat"], pair["v1"]["lon"]))
    try:
        changed = B.detect_at(case, step, edit)
    except EditTargetMismatch as e:
        return finish("ANOMALOUS", f"the screen was handed a population the pair was not chosen against: {e}")
    exact_edit, _ = replacement_edit(kp, len(port_axes), (pair["port"]["lat"], pair["port"]["lon"]),
                                     ([exact_center[0]], [exact_center[1]]))
    exact = B.detect_at(case, step, exact_edit)
    out["detection_stage"] = {
        "vertex_list": {**signature_difference(plain, changed), "acts": changed != plain,
                        "merged_candidates_near_observation_before": merged_near(plain, obs),
                        "merged_candidates_near_observation_after": merged_near(changed, obs)},
        "exact_potwv_center_as_single_point": {
            **signature_difference(plain, exact), "acts": exact != plain,
            "same_signature_as_vertex_list": exact == changed}}
    if changed == plain:
        if exact == plain:
            return finish("UNEXPLAINED", "screened out at detection: the replacement, and the exact "
                                         "center at the same position, change nothing the merge "
                                         "produces at this step, so no replay was made")
        out["note"] = ("the vertex list changes nothing at detection while the exact center "
                       "does, so only the exact center is replayed, as the precision reading")
    # THE NEGATIVE STEP IS CHOSEN BEFORE THE INTERVENTION REPLAY, from the captured steps
    # outside the union of every item's discrepancy neighborhood, anchored on the same
    # observation position, and the eligible population is recorded.
    excluded = set()
    for i in items:
        disc = S.item_discrepancy(reference, port, i)
        resolved = [S.resolve_step(times, s) for v in disc.values() for s in v]
        excluded |= set(B.search_steps([f"{s:.4f}" for s in resolved if s is not None], times))
    neg = out["controls"]["negative"]
    for other in sorted(float(k) for k in capture):
        if other in excluded or S.resolve_step(times, other) is None:
            continue
        p = pair_at(B.axes_at(case, other), capture[f"{other:.4f}"], obs)
        if p is None or p["port"]["ties"] != 1 or p["v1"]["ties"] != 1:
            continue
        edit, _ = replacement_edit(p["port"]["position"], p["port"]["of"], (p["port"]["lat"], p["port"]["lon"]),
                                   (p["v1"]["lat"], p["v1"]["lon"]))
        # NAMED APART from the intervention's `plain` and `changed`, which the variant
        # selection below reads after this loop.
        plain_other = B.detect_at(case, other)
        try:
            changed_other = B.detect_at(case, other, edit)
        except EditTargetMismatch as e:
            return finish("ANOMALOUS", f"the screen at {other} was handed a population the pair "
                                       f"was not chosen against: {e}")
        acts = changed_other != plain_other
        neg["eligible_steps"].append({"step": other, "acts": acts})
        if acts and neg["selected_step"] is None:
            neg["selected_step"], neg["available"] = other, True
            neg["operation"] = {"step": other, "operation": OPERATION, "pair": p,
                                **signature_difference(plain_other, changed_other)}
    if neg["selected_step"] is None:
        return finish("UNSUPPORTED", "no usable captured step outside the items' discrepancy "
                                     "neighborhoods changes the detected candidates under the "
                                     "operation, so no negative control is available")

    variants = []
    if changed != plain:
        variants.append(("intervention", edit_factory(kp, len(port_axes), (pair["port"]["lat"], pair["port"]["lon"]),
                                                      (pair["v1"]["lat"], pair["v1"]["lon"])),
                         centroid((pair["v1"]["lat"], pair["v1"]["lon"]))))
    if exact != changed:
        variants.append(("intervention_exact_center", edit_factory(kp, len(port_axes), (pair["port"]["lat"], pair["port"]["lon"]),
                                                                   ([exact_center[0]], [exact_center[1]])),
                         tuple(exact_center)))
    reproduced = {str(i["index"]): {} for i in items}
    for name, factory, supplied in variants:
        edit, hits = factory()
        expected = S.port_centers(case, step)
        expected[kp] = supplied
        spy, seen = S.merge_spy(step, expected)
        original_merge = B.D.merge_contours
        B.D.merge_contours = spy
        try:
            final = budget.replay(case, B.edit_hook(step, edit))
        except ReplayCeiling as e:
            return finish("NOT_RUN", str(e))
        except EditTargetMismatch as e:
            return finish("ANOMALOUS", f"the replay was handed a population the pair was not chosen against: {e}")
        finally:
            B.D.merge_contours = original_merge
        runs[name] = final
        received = {"first_pass_calls_at_step": seen["calls_at_step"], "candidates": seen["received"],
                    "received_center_at_position": list(supplied) if seen["exact"] else None,
                    "equals_supplied_exactly": seen["exact"],
                    "deviation_from_potwv_center_deg": [abs(supplied[0] - exact_center[0]),
                                                        abs(supplied[1] - exact_center[1])]}
        out.setdefault("merge_received", {})[name] = received
        if hits[0] != 1:
            return finish("ANOMALOUS", f"the {name} edit fired {hits[0]} times rather than exactly once")
        if not seen["exact"]:
            return finish("ANOMALOUS", f"the merge at {step} did not receive the supplied centers exactly ({seen})")
        for i in items:
            reproduced[str(i["index"])][name] = X.holds_exactly(final, *tracks[i["index"]])
    out["reproduced"] = reproduced

    # BOTH CONTROLS, whether or not a track was reproduced.
    rep = out["controls"]["representation"]
    rep["available"], rep["operation"], rep["step"] = True, "identity edit at the step", step
    edit, hits = S.counted_identity()
    try:
        r = budget.replay(case, B.edit_hook(step, edit))
    except ReplayCeiling as e:
        return finish("NOT_RUN", str(e))
    runs["representation_control"] = r
    rep["executed"], rep["edit_fired"] = True, hits[0]
    rep["passed"] = hits[0] == 1 and X.canonical(r) == X.canonical(baseline)
    if hits[0] != 1:
        return finish("ANOMALOUS", f"the identity edit fired {hits[0]} times")
    if not rep["passed"]:
        return finish("ANOMALOUS", "the identity edit at the step does not reproduce the baseline, "
                                   "so the apparatus moves the run by itself")
    ns, p = neg["selected_step"], neg["operation"]["pair"]
    edit, hits = replacement_edit(p["port"]["position"], p["port"]["of"], (p["port"]["lat"], p["port"]["lon"]),
                                  (p["v1"]["lat"], p["v1"]["lon"]))
    try:
        n = budget.replay(case, B.edit_hook(ns, edit))
    except ReplayCeiling as e:
        return finish("NOT_RUN", str(e))
    except EditTargetMismatch as e:
        return finish("ANOMALOUS", f"the negative control's replay was handed a population its pair "
                                   f"was not chosen against: {e}")
    runs["negative_control"] = n
    neg["executed"], neg["edit_fired"] = True, hits[0]
    neg["reproduces_reference"] = {str(i["index"]): X.holds_exactly(n, *tracks[i["index"]]) for i in items}
    neg["equals_baseline"] = X.canonical(n) == X.canonical(baseline)
    neg["passed"] = hits[0] == 1 and not any(neg["reproduces_reference"].values())
    if hits[0] != 1:
        return finish("ANOMALOUS", f"the negative control's edit fired {hits[0]} times")
    for i in items:
        k = str(i["index"])
        if neg["reproduces_reference"][k]:
            out["outcomes"][k] = "UNEXPLAINED"
            out.setdefault("why_per_item", {})[k] = ("the negative control also reproduces this "
                                                    "track, so the reproduction is not specific "
                                                    "to the intervention step")
        elif any(reproduced[k].values()):
            out["outcomes"][k] = "EXPLAINED"
        else:
            out["outcomes"][k] = "UNEXPLAINED"
            out.setdefault("why_per_item", {})[k] = ("the replacement changes the detected candidates "
                                                    "and does not reproduce this track")
    out["achieved_grade"] = ("per item: EXPLAINED means exact reproduction of the complete track "
                             "from the intervention replay, the identity edit at the step "
                             "reproducing the baseline, and the same operation at the pre-selected "
                             "unrelated acting step not reproducing the track. Sufficient under "
                             "the tested conditions, not a cause, not an accounting credit.")
    return finish(None)


def edit_factory(position, count, expected_axis, vertices):
    return lambda: replacement_edit(position, count, expected_axis, vertices)


def parse_interventions(text):
    """'C:34882.75:23,27;E:39262.5:17' to a list of (window, step, [items])."""
    out = []
    for token in text.split(";"):
        w, s, items = token.strip().split(":")
        out.append((w, float(s), sorted(int(x) for x in items.split(","))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("work")
    ap.add_argument("--interventions", required=True)
    ap.add_argument("--record", required=True,
                    help="the whole-population comparison artifact whose pairing must be reproduced")
    ap.add_argument("--out", required=True)
    ap.add_argument("--retain", required=True)
    ap.add_argument("--replay-ceiling", type=int, default=10)
    args = ap.parse_args(argv)
    launched = {"script_sha256": X.digest(__file__), "pilot_sha256": X.digest(S.__file__),
                "driver_sha256": X.digest(B.__file__),
                "git_head_at_launch": X.repository_head(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "launched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(args.record, "rb") as fh:
        record_bytes = fh.read()
    record_sha256 = hashlib.sha256(record_bytes).hexdigest()
    records = {r["item"]: r for r in json.loads(record_bytes.decode())["results"]}
    os.makedirs(args.retain, exist_ok=True)
    budget = Budget(args.replay_ceiling)
    results, retained_files, inputs = [], {}, {}
    wall0, cpu0, cost0 = time.perf_counter(), time.process_time(), B._cost_snapshot()
    for w, step, wanted in parse_interventions(args.interventions):
        directory, case, residuals, reference, port, capture_text, digests = \
            S.load_window_snapshot(args.work, w)
        inputs[w] = digests
        times = case["time"].ravel()
        capture, _, _ = S.ordered_capture(capture_text, times)
        centers, _ = S.ordered_centers(capture_text, times)
        items_by_index = {i["index"]: i for i in B.build_items(residuals, reference, case)}
        unknown = [i for i in wanted if i not in items_by_index]
        if unknown:
            raise SystemExit(f"REFUSED: window {w} has no residual items {unknown}")
        items = [items_by_index[i] for i in wanted]
        # ITEMS THAT SHARE AN INTERVENTION MUST SHARE A RECORDED PAIR, or the run refuses.
        recs = [records[f"{w} {i}"] for i in wanted if f"{w} {i}" in records]
        if recs and any(r.get("nearest_axis_any_side") != recs[0].get("nearest_axis_any_side")
                        for r in recs):
            raise SystemExit(f"REFUSED: the recorded comparisons for {w} {wanted} disagree")
        record = recs[0] if recs else None
        try:
            baseline = budget.replay(case)
        except ReplayCeiling as e:
            results.append({"window": w, "step": step, "items": wanted, "outcomes": {
                str(i): "NOT_RUN" for i in wanted}, "why": str(e)})
            continue
        provenance = {"script_sha256": launched["script_sha256"], "git_head": launched["git_head_at_launch"],
                      "case_sha256": digests["tracker_case.mat"],
                      "reference_output_sha256": digests["tracker_octave_instrumented.mat"]}

        def retain(name, runs, index, extra):
            path = os.path.join(args.retain, name)
            try:
                X.save_run(path, runs, os.path.join(directory, "tracker_case.mat"),
                           os.path.join(directory, "tracker_octave_instrumented.mat"),
                           index, __file__,
                           extra={"window": w, "inputs_sha256": digests,
                                  "record_sha256": record_sha256, **launched, **extra},
                           exclusive=True, **provenance)
            except FileExistsError:
                raise SystemExit(f"REFUSED: {path} exists and retained evidence is never overwritten")
            retained_files[name] = X.digest(path)
        retain(f"{w}_baseline_pair.json", {"baseline": baseline}, -1,
               {"role": "baseline replayed by this command"})
        rec, runs = investigate(case, reference, port, items, step, capture, centers, times,
                                record, baseline, budget)
        rec["window"] = w
        results.append(rec)
        retain(f"{w}_{step:.4f}_pair.json", runs, wanted[0],
               {"items": wanted, "operation": OPERATION, "outcomes": rec["outcomes"],
                "pair": rec.get("pair"), "merge_received": rec.get("merge_received"),
                "detection_stage": rec.get("detection_stage"), "controls": rec["controls"],
                "achieved_grade": rec.get("achieved_grade")})
        print(f"  {w} at {step}: {json.dumps(rec['outcomes'])} ({rec.get('why', '')}) replays after "
              f"baseline {rec.get('replays_after_baseline')}, {rec.get('processor_seconds')} s", flush=True)
    total = B._cost_since(cost0)
    payload = {
        "generated_by": "scripts/pair_vertex_substitution.py",
        "brief": ("written before the run: two interventions covering three items, the pairing "
                  "rule, the operation, detection first, both controls, a ceiling of ten replays"),
        "operation": OPERATION, "pairing": PAIRING,
        "what_this_tests": ("the selected axis representation: replacing one axis's vertex list "
                            "changes its extent and the weighting of its centroid together, and "
                            "the merge consumes it through the centroid alone. A reproduction says "
                            "the change is sufficient under the tested conditions and does not "
                            "identify the extra endpoints, the repeats or the extent as the cause"),
        **launched, "record": args.record, "record_sha256": record_sha256,
        "inputs_sha256": inputs, "replay_ceiling": args.replay_ceiling,
        "results": results, "retained_files": retained_files,
        "retention_directory": os.path.abspath(args.retain),
        "timing": {"processor_seconds": round(time.process_time() - cpu0, 1),
                   "elapsed_seconds": round(time.perf_counter() - wall0, 1),
                   "replays_including_baselines": total["replay_calls"],
                   "screens": total["screen_calls"],
                   "replay_seconds": total["replay_seconds"], "screen_seconds": total["screen_seconds"]},
    }
    try:
        X.publish_json(args.out, payload, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {args.out} exists and artifacts are never overwritten")
    print(f"pair substitution: {payload['timing']['replays_including_baselines']} replays including "
          f"baselines, {payload['timing']['processor_seconds']/60:.1f} processor-min; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
