#!/usr/bin/env python3
"""The explained and remaining residue of the identical-input comparison, as explicit
index lists derived from the residual artifact and the worked-case artifacts, never
typed.

A review found the handoff's coverage totals overstated (two of five unmatched tracks
explained where the cases had walked one) and its remaining-work list mixing categories.
This script reads the residual classification and each worked case's declared
membership, checks every claimed index against the category it is claimed for, and
writes the counts beside the lists they come from. A case's membership is what its
artifact declares under `explains`; a case artifact without that block explains
nothing here.

    .venv/bin/python scripts/residue_membership.py \\
        --residuals docs/aewc_v2/artifacts/tracker_oracle_residuals_2026-09-18.json \\
        --cases docs/aewc_v2/artifacts/sahara_case_2026-09-18.json ... --out <json>
"""
import argparse
import gzip
import hashlib
import math
import json
import os
import sys


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


BINDING_FILES = ("tracker_case.mat", "tracker_port.mat")


def binding_problems(residuals, cases):
    """The reasons a case artifact is NOT a case from the residual artifact's own
    comparison. Track indices are local to a run, so a case must agree with the
    residual artifact on the case identifier, the exported-case digest and the
    port-output digest before any of its claimed indices can count. The reference
    output digests are NOT compared, because instrumented runs with different dump
    schedules legitimately carry different producer metadata over identical tracks.
    A review copied a genuine case artifact, renamed its case and replaced these
    digests, and the first version credited its claims."""
    problems = []
    r_case = residuals.get("case_id")
    r_hashes = residuals.get("input_sha256") or {}
    for name, case in cases.items():
        c_case = case.get("case_id")
        c_hashes = case.get("input_sha256") or {}
        if r_case is None or c_case is None:
            problems.append(f"{name}: a case identifier is missing on one side")
        elif c_case != r_case:
            problems.append(f"{name}: case {c_case!r} is not the residual artifact's {r_case!r}")
        for f in BINDING_FILES:
            if f not in r_hashes or f not in c_hashes:
                problems.append(f"{name}: the digest of {f} is missing on one side")
            elif c_hashes[f] != r_hashes[f]:
                problems.append(f"{name}: {f} differs from the residual artifact's, so the "
                                f"case comes from another export or another port run")
    return problems


REQUIRED_RUNS = ("baseline", "intervention", "control", "shape_control")

REFERENCE_MANIFEST = "docs/aewc_v2/artifacts/reference_logs.json"


def _log_text(path):
    """The log's contents and the digest of them. The retained copies are compressed and
    a case artifact records the digest of the DECOMPRESSED bytes, so both come from here
    rather than from the file on disk."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as fh:
        raw = fh.read()
    return raw.decode(), hashlib.sha256(raw).hexdigest()


def _within(lat, lon, box):
    return (box["lat"][0] <= lat <= box["lat"][1]
            and box["lon"][0] <= lon <= box["lon"][1])


def axes_from_log(text, window, box, axis_box):
    """Version 1's dumped axis vertices per timestep, parsed OUT OF THE ORIGINAL LOG.

    THIS IS A SECOND, INDEPENDENT PARSE, and that is the point of it. The trace has its
    own reader, and importing that one would make the two agree by construction rather
    than by evidence. It reproduces the same selection from the case's own declared
    parameters: a dump inside the window, and an axis kept when ANY of its vertices lies
    in the axis box, or in the case box when no axis box was declared."""
    where = axis_box or box
    out, problems = {}, []
    for n, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts or parts[0] != "AXISPTS":
            continue
        # A LOG LINE THAT WILL NOT PARSE IS A REFUSAL, not an exception. The first version
        # raised on one, which is an unhandled failure rather than a verdict, and a
        # caller cannot tell a crash from a judgment.
        try:
            t = float(parts[1])
            vals = [float(x) for x in parts[3:]]
        except (IndexError, ValueError):
            problems.append(f"line {n} of the reference log is not a readable axis dump")
            continue
        if not window[0] <= t <= window[1]:
            continue
        if not vals or len(vals) % 2 or not all(math.isfinite(v) for v in vals):
            problems.append(f"line {n} of the reference log holds an odd or nonfinite "
                            f"set of coordinates")
            continue
        lat, lon = vals[0::2], vals[1::2]
        if any(_within(la, lo, where) for la, lo in zip(lat, lon)):
            out.setdefault(f"{t:.4f}", []).append({"lat": lat, "lon": lon})
    return out, problems


def _read_finished_tracks(path):
    """The finished tracks an output holds, by FINAL index, as plain arrays. The reader is
    the comparison's own MAT reader, since the independence that matters here is from the
    case artifact, not from the file format."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import compare_tracker_oracle as C
    tracks, case_id = C.read_tracks(path)
    return [{"time": [float(x) for x in tr["time"]], "lat": [float(x) for x in tr["lat"]],
             "lon": [float(x) for x in tr["lon"]]} for tr in tracks], case_id


def verified_outputs(paths, manifest_path=None):
    """The retained finished outputs, keyed by the digest of the file, with the reasons
    any is not usable. Version 1's reference tracks come from these BY FINAL INDEX. A
    review found the first version of the unmatched check reading TRACK lines out of the
    log, whose index is a live list position at that loop time and not a finished track:
    at the Sahara step the two were on different continents."""
    accepted, problems = read_manifest(manifest_path)
    if problems:
        return {}, problems
    pinned = {e["sha256"]: e for e in accepted.get("__outputs__", [])}
    out = {}
    for path in paths or ():
        if not os.path.exists(path):
            problems.append(f"the finished output {path} is not present")
            continue
        digest = _sha256(path)
        if digest not in pinned:
            problems.append(f"the finished output {path} has digest {digest[:12]}, which the "
                            f"manifest does not name as one this project accepts")
            continue
        try:
            tracks, case_id = _read_finished_tracks(path)
        except (SystemExit, KeyError, ValueError, OSError) as e:
            problems.append(f"the finished output {path} cannot be read: {e}")
            continue
        # and the pinned side is held to the same shape, since a ragged reference would
        # make every comparison against it meaningless in the same silent way
        bad = [w for tr in tracks for w in _track_problems(tr)]
        if bad:
            problems.append(f"the finished output {path} holds a track that cannot be "
                            f"compared: {bad[0]}")
            continue
        out[digest] = {"path": path, "kind": pinned[digest].get("kind"), "tracks": tracks,
                       "case_id": case_id}
    return out, problems


def _times(track):
    return {f"{float(t):.4f}" for t in track["time"]}


def _at(track, key):
    """The (lat, lon) a track holds at a step key, or None."""
    for t, la, lo in zip(track["time"], track["lat"], track["lon"]):
        if f"{float(t):.4f}" == key:
            return float(la), float(lo)
    return None


def evidence_problems(name, case, residuals, explains, outputs):
    """Why the case's scope is not the residual's own, read from the RETAINED OUTPUTS.

    What is derived here, and from what, for every index the case claims:
      the pair's extra times: version 1's finished track holds an observation and the
        port's finished counterpart does not, from the two pinned outputs, and both the
        residual record and the case's declared steps must equal that set exactly;
      an unmatched track's steps: every declared step is a time version 1's finished
        track holds an observation (WHICH of its times the case declares is its own);
      the time-control steps: version 1's finished track holds an observation, and for
        a pair so does the port's counterpart, since "steps both sides already agree on"
        is what the prose claims and this is the first place it is read rather than
        declared;
      the region: the EFFECTIVE selector, the axis box where one is declared and the case
        box otherwise, contains version 1's own observation at every declared step. A
        review changed only the axis box, selected nine South American axes for the
        Mozambique pair, and the guard on the case box was satisfied throughout."""
    problems = []
    params = case.get("parameters") or {}
    declared_ref = (case.get("input_sha256") or {}).get("tracker_octave_instrumented.mat")
    declared_port = (case.get("input_sha256") or {}).get("tracker_port.mat")
    ref = outputs.get(declared_ref) if declared_ref else None
    port = outputs.get(declared_port) if declared_port else None
    if ref is None or ref.get("kind") != "reference":
        return [f"{name} declares reference output {(declared_ref or '')[:12] or '(none)'}, "
                f"which was not supplied to this command"]
    if port is None or port.get("kind") != "port":
        return [f"{name} declares port output {(declared_port or '')[:12] or '(none)'}, "
                f"which was not supplied to this command"]
    steps = sorted(f"{float(t):.4f}" for t in (params.get("divergence_steps") or ()))
    controls = sorted(f"{float(t):.4f}" for t in (params.get("control_steps") or ()))
    region = params.get("axis_box") or params.get("box") or {}
    by_pair = {p["v1_index"]: p for p in residuals.get("pairs") or ()}

    def track(side, i, what):
        tracks = side["tracks"]
        if not isinstance(i, int) or not 0 <= i < len(tracks):
            problems.append(f"{name} claims {what}, which the retained output does not hold")
            return None
        return tracks[i]

    def region_holds(v1, what):
        for key in steps:
            pos = _at(v1, key)
            if pos is None:
                continue                          # reported by the step checks
            if not region or not _within(pos[0], pos[1], region):
                problems.append(f"{name} claims {what} and its effective selection region "
                                f"does not contain version 1's own observation at {key} "
                                f"({pos[0]:.2f}, {pos[1]:.2f})")

    for i in explains.get("v1_extra_pairs") or ():
        pair = by_pair.get(i)
        if pair is None:
            continue
        v1 = track(ref, i, f"pair {i}")
        counterpart = track(port, pair.get("port_index"), f"pair {i}'s port counterpart")
        if v1 is None or counterpart is None:
            continue
        extra = sorted(_times(v1) - _times(counterpart))
        recorded = sorted(f"{float(t):.4f}" for t in
                          ((pair.get("v1_extra") or {}).get("times") or ()))
        if recorded != extra:
            problems.append(f"{name} claims pair {i}, whose residual record says version 1's "
                            f"extra observations are at {recorded} and the retained outputs "
                            f"say {extra}")
        if steps != extra:
            problems.append(f"{name} claims pair {i} and declares divergence steps {steps}, "
                            f"while the retained outputs put version 1's extra observations "
                            f"at {extra}")
        for key in controls:
            if key not in _times(v1) or key not in _times(counterpart):
                problems.append(f"{name} claims pair {i} and declares control step {key}, at "
                                f"which the two sides do not both hold an observation")
        region_holds(v1, f"pair {i}")

    for i in explains.get("unmatched_v1_tracks") or ():
        v1 = track(ref, i, f"unmatched track {i}")
        if v1 is None:
            continue
        for key in steps:
            if key not in _times(v1):
                problems.append(f"{name} claims unmatched track {i} and declares step {key}, "
                                f"at which version 1's finished track holds no observation")
        for key in controls:
            if key not in _times(v1):
                problems.append(f"{name} claims unmatched track {i} and declares control "
                                f"step {key}, at which version 1's finished track holds no "
                                f"observation")
        region_holds(v1, f"unmatched track {i}")
    return problems


def _track_problems(tr):
    """Why a recorded track is not one that can be counted or compared: its three arrays
    are not the same nonzero length, or a coordinate is not a finite number. A review
    deleted the final latitude of a genuine Mozambique replay track, and appended one,
    and the comparison below still called it an exact reproduction, because it checked
    the length of the time array alone and zipped the others, and a zip stops at the
    shorter side."""
    if not isinstance(tr, dict):
        return ["a recorded track is not a record"]
    arrays = [list(tr.get(k) or []) for k in ("time", "lat", "lon")]
    if not arrays[0] or len({len(a) for a in arrays}) != 1:
        return [f"a recorded track holds {len(arrays[0])} times, {len(arrays[1])} latitudes "
                f"and {len(arrays[2])} longitudes"]
    for a in arrays:
        for x in a:
            try:
                if not math.isfinite(float(x)):
                    return ["a recorded track holds a coordinate that is not finite"]
            except (TypeError, ValueError):
                return ["a recorded track holds a coordinate that is not a number"]
    return []


def _same_track(a, b):
    """Exact equality, step for step, the comparison the trace's own `track_equal`
    makes, written again here rather than imported. COMPLETE ARRAYS, every one of the
    three checked for length on both sides, since a zip over arrays of different length
    compares a prefix and calls it equal."""
    for k in ("time", "lat", "lon"):
        u, v = list(a.get(k) or []), list(b.get(k) or [])
        if len(u) != len(v) or not u:
            return False
        if any(float(x) != float(y) for x, y in zip(u, v)):
            return False
    return True


def recomputed_outcomes(name, case, explains, outputs):
    """The two outcomes, RECOMPUTED from each replay's recorded finished tracks against the
    pinned reference, and the reasons the recomputation could not be made or disagrees
    with what the replay recorded. Returns (problems, {label: {"exact": {index: bool},
    "west": int}}).

    A recorded track collection is still the case's own. What this establishes is that
    the flags are arithmetic on declared tracks against an independent reference, not
    that the replay produced those tracks. The grade is DERIVED."""
    problems, out = [], {}
    params = case.get("parameters") or {}
    west_box, life = params.get("west"), params.get("feature_life")
    declared_ref = (case.get("input_sha256") or {}).get("tracker_octave_instrumented.mat")
    ref = outputs.get(declared_ref) if declared_ref else None
    if ref is None:
        return [f"{name} declares no retained reference output to recompute against"], {}
    claimed = list(explains.get("unmatched_v1_tracks") or ()) \
        + list(explains.get("v1_extra_pairs") or ())
    intervention = case.get("intervention") or {}
    labels = [l for l in REQUIRED_RUNS if intervention.get(l)] \
        + list(intervention.get("partial_runs") or ())
    for label in labels:
        run = intervention.get(label) or {}
        tracks = run.get("finished_tracks")
        if not isinstance(tracks, list):
            problems.append(f"the {label} replay records no finished tracks, so its outcome "
                            f"cannot be recomputed")
            continue
        # EVERY RECORDED TRACK IS VALIDATED BEFORE ANYTHING COUNTS OR COMPARES IT. A track
        # with a missing or extra coordinate, or one that is not a number, refuses the
        # replay outright rather than being counted as whatever the comparison makes of it.
        malformed = [w for tr in tracks for w in _track_problems(tr)]
        if malformed:
            problems.append(f"the {label} replay records a finished track that cannot be "
                            f"compared: {malformed[0]}")
            continue
        exact = {}
        for i in claimed:
            if not isinstance(i, int) or not 0 <= i < len(ref["tracks"]):
                continue
            exact[i] = any(_same_track(tr, ref["tracks"][i]) for tr in tracks)
            flag = next((r.get("reproduced_exactly") for r in run.get("reference_tracks") or ()
                         if r.get("reference_index") == i), None)
            if flag is None:
                # A RECOMPUTATION CHECKS A RECORDED OUTCOME, it does not stand in for one.
                # An index the replay never examined must not become credited because the
                # recomputation happened to find its track among the recorded ones.
                problems.append(f"the {label} replay records no outcome for {i}")
                continue
            if bool(flag) != exact[i]:
                problems.append(f"the {label} replay records reproduced_exactly={flag} for "
                                f"{i} and its own finished tracks give {exact[i]}")
        if not west_box or not life:
            problems.append(f"{name} declares no western box or feature life, so the "
                            f"finished-track count cannot be recomputed")
            west = None
        else:
            west = sum(1 for tr in tracks
                       if any(_within(float(la), float(lo), west_box)
                              and float(life[0]) <= float(t) <= float(life[1])
                              for t, la, lo in zip(tr["time"], tr["lat"], tr["lon"])))
            recorded = run.get("finished_tracks_in_the_western_box")
            if recorded is not None and len(recorded) != west:
                problems.append(f"the {label} replay records {len(recorded)} finished tracks "
                                f"in the western box and its own finished tracks give {west}")
        out[label] = {"exact": exact, "west": west}
    return problems, out


def read_manifest(path=None):
    """The reference runs this project accepts, and why they are pinned here.

    A digest the candidate supplies names its claimed input. The accepted identities live
    in a retained record beside the evidence, and what that buys is exactly what the
    manifest says of itself: a reproducibility check against retained evidence. It does
    not protect against a party able to rewrite the repository, the artifacts and this
    manifest together."""
    path = path or REFERENCE_MANIFEST
    if not os.path.exists(path):
        return {}, [f"the reference-log manifest {path} is not present, so no log can be "
                    f"recognised as one this project accepts"]
    with open(path) as fh:
        manifest = json.load(fh)
    accepted = {e["sha256"]: e for e in manifest.get("logs") or ()}
    accepted["__outputs__"] = list(manifest.get("outputs") or ())
    return accepted, []


def verified_logs(paths, manifest_path=None):
    """The reference logs, keyed by the digest of their contents, with the reasons any of
    them is not usable evidence. A log the manifest does not name is refused, because an
    unpinned log is one the candidate could have supplied along with everything else."""
    accepted, problems = read_manifest(manifest_path)
    if problems:
        return {}, problems
    logs = {}
    for path in paths or ():
        if not os.path.exists(path):
            problems.append(f"the reference log {path} is not present")
            continue
        text, digest = _log_text(path)
        if digest not in accepted or digest == "__outputs__":
            problems.append(f"the reference log {path} has digest {digest[:12]}, which the "
                            f"manifest does not name as one this project accepts")
            continue
        logs[digest] = {"path": path, "text": text}
    return logs, problems


def source_problems(name, case, logs):
    """Why a case's recorded axis vertices are not the ones its reference log holds.

    THE CASE CANNOT AUTHENTICATE ITS OWN COPY. A review translated a case's embedded
    vertices and every replay's axes together by one degree, left the recorded digests and
    outcomes untouched, and the accounting credited the pair: every field the checker
    observed still agreed with every other. The vertices are therefore re-read from the
    log the case declares, and a case whose copy does not match it is refused."""
    declared = (case.get("input_sha256") or {}).get("octave1.log")
    if not declared:
        return [f"{name} declares no reference-log digest, so its recorded vertices "
                f"cannot be traced to a run"]
    if declared not in logs:
        return [f"{name} declares reference log {declared[:12]}, which was not supplied "
                f"to this command"]
    parameters = case.get("parameters") or {}
    window, box = parameters.get("window"), parameters.get("box")
    if not window or not box:
        return [f"{name} declares no window or no box, so the log's axes cannot be "
                f"selected the way the case selected them"]
    from_log, problems = axes_from_log(logs[declared]["text"], window, box,
                                       parameters.get("axis_box"))
    problems = [f"{name}: {w}" for w in problems]
    recorded = dumped_vertices((case.get("at_divergence") or {}).get("v1"))
    for key in sorted(recorded):
        want = from_log.get(key)
        if not want:
            problems.append(f"{name} records axis vertices at {key} and its reference log "
                            f"holds none there")
        elif not _axes_equal(recorded[key], want):
            problems.append(f"{name}'s recorded axis vertices at {key} are not the ones "
                            f"its reference log holds there")
    # NO TRACK-LINE CHECK HERE. A first version read `TRACK <time> <index>` lines out of
    # the log as the finished track's observations; the index there is a live list
    # position at that loop time, and at the Sahara step the two were on different
    # continents. The finished tracks come from the pinned outputs, in `evidence_problems`.
    for key in sorted(set(from_log) - set(recorded)):
        if any(abs(float(key) - float(t)) < 1e-6
               for t in (parameters.get("divergence_steps") or ())):
            problems.append(f"{name} records no axis vertices at {key}, which its "
                            f"reference log holds and the case declares as a divergence "
                            f"step")
    return problems




def dumped_vertices(v1_at_divergence):
    """Version 1's own dumped axis vertices per declared step, read from the records the
    case carries for that step. THIS is what the injected axes are bound to, because a
    map the artifact also carries beside them is only a duplicate declaration, and a
    review made the duplicate and the run agree with each other while agreeing with
    nothing else.

    Takes the step-keyed map of version 1 records, which is what both readers hold. The
    trace has it while it is building the case, and the membership checker reads it back
    out of `at_divergence.v1`."""
    out = {}
    for key, records in (v1_at_divergence or {}).items():
        groups = [{"lat": list(r.get("lat") or []), "lon": list(r.get("lon") or [])}
                  for r in (records or ()) if r.get("kind") == "AXISPTS"]
        if groups:
            out[str(key)] = groups
    return out


def expected_axes(steps, control_steps, reference_vertices, offset):
    """What each run's axes MUST be, derived rather than read from the artifact.

    A review is why this is derived. The first version compared each run against a map
    the artifact itself carried, so deleting the map disabled the comparison, and writing
    the wrong geometry into BOTH the run and its map made the two agree with each other
    while agreeing with nothing else. The expectation therefore rests on the case's own
    DUMPED REFERENCE VERTICES, the declared step pairing and the recorded translation.

    Returns (expectations_by_label, problems)."""
    problems = []
    keys = [f"{t:.4f}" for t in steps]
    reference = {}
    for t, key in zip(steps, keys):
        groups = (reference_vertices or {}).get(key)
        if not groups:
            problems.append(f"the case records no dumped reference vertices at {key}, so "
                            f"there is nothing to bind the injected axes to")
        elif not _usable(groups):
            # A REVIEW WROTE "NaN" INTO THE DUMPED LONGITUDES and kept the credit, because
            # the comparison below rejects a coordinate only when the difference EXCEEDS a
            # tolerance, and a difference from a not-a-number is never greater than
            # anything. Only the recorded run coordinates were checked for finiteness.
            problems.append(f"the case's dumped reference vertices at {key} are empty, "
                            f"ragged or not finite, so nothing can be checked against them")
        else:
            reference[key] = groups
    if problems:
        return {}, problems
    controls = [float(c) for c in (control_steps or [])]
    if len(controls) != len(steps):
        return {}, [f"the case declares {len(controls)} control steps for {len(steps)} "
                    f"divergence steps"]
    try:
        shift = float(offset)
    except (TypeError, ValueError):
        shift = float("nan")
    if not math.isfinite(shift) or shift == 0.0:
        return {}, ["the shape control's recorded translation is zero or not a finite "
                    "number, so its vertices are not shown to differ from the "
                    "intervention's"]
    control_map = {f"{c:.4f}": reference[k] for k, c in zip(keys, controls)}
    shape_map = {k: [{"lat": list(g["lat"]),
                      "lon": [float(x) - shift for x in g["lon"]]}
                     for g in reference[k]] for k in keys}
    # A TRANSLATION TOO SMALL TO SEE IS NOT A TRANSLATION. A review passed 1e-12, which is
    # finite and nonzero and moves nothing at the tolerance the comparison uses, and the
    # shape control then carried the intervention's own geometry under another name.
    if any(_axes_equal(shape_map[k], reference[k]) for k in keys):
        return {}, [f"the shape control's translation of {shift} degrees leaves its "
                    f"vertices equal to the intervention's at the tolerance this "
                    f"comparison uses"]
    out = {"intervention": reference, "control": control_map, "shape_control": shape_map}
    for k in keys:
        out[f"partial_at_{k}"] = {k: reference[k]}
    return out, problems


def axis_problems(label, run, expected_by_step):
    """Why a run's recorded axes are not the geometry it claims to have injected.

    A review emptied every axis group in a genuine artifact, left the time receipts and
    outcomes alone, and both readers accepted it and credited the pair: the contract
    counted applications and never asked what was injected. It also swapped the translated
    shape-control axes into the time control, which kept the credit while the generated
    sentence still called them the same axes. The expectation passed here is DERIVED from
    the dumped vertices, never read from the artifact, and it is required rather than
    optional, so a missing expectation is a refusal and not a skipped comparison."""
    problems = []
    recorded = run.get("axes_by_requested_step")
    if not isinstance(recorded, dict) or not recorded:
        return [f"the {label} replay records no axes for the steps it injected at"]
    requested = {f"{float(t):.4f}" for t in (run.get("requested_times") or [])}
    if set(recorded) != requested:
        return [f"the {label} replay records axes for {sorted(recorded)} and injected at "
                f"{sorted(requested)}"]
    if not expected_by_step:
        return [f"the {label} replay has no expected axes to be checked against"]
    if set(expected_by_step) != set(recorded):
        return [f"the {label} replay records axes for {sorted(recorded)} and the case's "
                f"own vertices give {sorted(expected_by_step)}"]
    for step, groups in sorted(recorded.items()):
        if not groups:
            problems.append(f"the {label} replay records no axis at {step}")
            continue
        for group in groups:
            lat, lon = group.get("lat") or [], group.get("lon") or []
            if not lat or not lon or len(lat) != len(lon):
                problems.append(f"the {label} replay's axis at {step} has "
                                f"{len(lat)} latitudes and {len(lon)} longitudes")
            elif not all(math.isfinite(float(x)) for x in list(lat) + list(lon)):
                problems.append(f"the {label} replay's axis at {step} is not finite")
        want = expected_by_step.get(step)
        if want is not None and not _axes_equal(groups, want):
            problems.append(f"the {label} replay's axes at {step} are not the ones the "
                            f"case's dumped vertices give for it")
    return problems


def _usable(groups):
    """Whether a set of axis groups is something a comparison can rest on: nonempty, with
    as many latitudes as longitudes, and every coordinate a finite number."""
    if not groups:
        return False
    for g in groups:
        lat, lon = list((g or {}).get("lat") or []), list((g or {}).get("lon") or [])
        if not lat or len(lat) != len(lon):
            return False
        for x in lat + lon:
            try:
                if not math.isfinite(float(x)):
                    return False
            except (TypeError, ValueError):
                return False
    return True


def _axes_equal(a, b, tol=1e-9):
    """Whether two sets of axis groups hold the same finite coordinates.

    EQUALITY IS STATED POSITIVELY, as finite and within tolerance, rather than as the
    absence of a difference exceeding it. A review put a not-a-number into one side, where
    every difference test is false, and the negative form read that as agreement."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        for coord in ("lat", "lon"):
            u, v = list(x.get(coord) or []), list(y.get(coord) or [])
            if len(u) != len(v):
                return False
            for p, q in zip(u, v):
                try:
                    p, q = float(p), float(q)
                except (TypeError, ValueError):
                    return False
                if not (math.isfinite(p) and math.isfinite(q) and abs(p - q) <= tol):
                    return False
    return True


def experiment_problems(intervention, divergence_steps, control_steps=None,
                        control_step_problems=(), reference_vertices=None):
    """Why a recorded experiment set is not one, checked the SAME WAY wherever it is read.
    This is the shared contract: the trace calls it before it will state anything about an
    experiment, and the membership checker calls it before it will credit one.

    A review is why it is shared rather than written twice. The checker had its own weaker
    copy: it required a receipt to agree with the time its own replay requested, and never
    asked whether that time was a DECLARED DIVERGENCE STEP, so moving an intervention ten
    hours away kept all its credits while the trace refused the same artifact. The
    expected steps are passed in explicitly, never read from a module global.

    A CONTROL CARRIES AS MANY INJECTIONS AS THE EXPERIMENT IT CONTROLS. A two-injection
    experiment compared against a one-injection control is not a comparison, so the counts
    are required to match rather than merely to be nonzero."""
    problems = []
    if not intervention or not intervention.get("baseline"):
        return ["the artifact records no intervention"]
    absent = [r for r in REQUIRED_RUNS if not intervention.get(r)]
    if absent:
        problems.append("the experiment records no " + " or ".join(absent) + " replay")
    problems.extend(control_step_problems or [])
    steps = sorted(float(x) for x in (divergence_steps or []))
    if not steps:
        problems.append("the case declares no divergence step")
        return problems

    def finite(x):
        try:
            return math.isfinite(float(x))
        except (TypeError, ValueError):
            return False

    # THE SCHEDULE THE CASE DECLARES, not the list the artifact happens to carry. A review
    # deleted both single-step records and their names, changed one of them to test the
    # other's step, and repeated a control time, and every version was accepted: the
    # reader took its required runs from the artifact's own optional list.
    required_partials = ([f"partial_at_{t:.4f}" for t in steps] if len(steps) > 1 else [])
    named = list(intervention.get("partial_runs") or ())
    if sorted(named) != sorted(required_partials):
        problems.append(f"the experiment names single-step replays {sorted(named)} and the "
                        f"case declares {sorted(required_partials)}")
    if control_steps is not None and len(control_steps) != len(steps):
        # A DECLARED CONTROL SCHEDULE THAT IS NOT THE EXPERIMENT'S SHAPE. Without this,
        # a longer declared list lets a control inject at a SUBSET of it and still pass a
        # membership test, which is why equality rather than membership is required below.
        problems.append(f"the case declares {len(control_steps)} control steps for "
                        f"{len(steps)} divergence steps")
    # WHAT EACH RUN'S AXES MUST BE, derived from the case's own dumped vertices, the
    # declared pairing and the recorded translation, never read from the artifact.
    expectations, why = expected_axes(
        steps, control_steps, reference_vertices,
        (intervention.get("shape_control") or {}).get("longitude_offset_deg"))
    problems.extend(why)
    for label in ("intervention", "control", "shape_control") + tuple(required_partials):
        run = intervention.get(label)
        if not run:
            problems.append(f"the experiment records no {label} replay")
            continue
        problems.extend(axis_problems(label, run, expectations.get(label)))
        requested = [x for x in (run.get("requested_times") or [])]
        applied = [x for x in (run.get("applied_at") or [])]
        if not requested:
            problems.append(f"the {label} replay requested no injection")
            continue
        if run.get("injections_applied") != len(requested) or len(applied) != len(requested):
            problems.append(f"the {label} replay applied {run.get('injections_applied')} of "
                            f"the {len(requested)} injections it requested")
            continue
        if not all(finite(a) and finite(r) and abs(float(a) - float(r)) <= 1e-6
                   for a, r in zip(sorted(applied), sorted(requested))):
            problems.append(f"the {label} replay recorded applications at {applied} for "
                            f"injections requested at {requested}")
            continue
        if not steps:
            continue
        same_as_declared = (len(requested) == len(steps)
                            and all(abs(float(r) - t) <= 1e-6
                                    for r, t in zip(sorted(requested), sorted(steps))))
        if label in ("intervention", "shape_control") and not same_as_declared:
            problems.append(f"the {label} replay injected at {sorted(requested)} and the "
                            f"case declares {sorted(steps)}")
        elif label == "control":
            if len(requested) != len(steps):
                problems.append(f"the time control carries {len(requested)} injections "
                                f"against the experiment's {len(steps)}")
            if any(any(abs(float(r) - t) <= 1e-6 for t in steps) for r in requested):
                problems.append(f"the time control injected at a declared divergence step "
                                f"({sorted(requested)})")
            if len(set(float(r) for r in requested)) != len(requested):
                problems.append(f"the time control repeats a step ({sorted(requested)}), so "
                                f"it tests fewer steps than the case declares")
            if control_steps and sorted(float(r) for r in requested) != sorted(
                    float(c) for c in control_steps):
                problems.append(f"the time control injected at {sorted(requested)} and the "
                                f"case declares its control steps as {sorted(control_steps)}")
        elif label.startswith("partial_at_"):
            # A RUN NAMED FOR A STEP MUST TEST THAT STEP. A review renamed nothing and
            # simply changed one single-step run to inject at the other's step, leaving
            # the second step untested while both names remained.
            named_step = float(label[len("partial_at_"):])
            if len(requested) != 1:
                problems.append(f"the {label} replay carries {len(requested)} injections "
                                f"where a single-step run carries one")
            elif abs(float(requested[0]) - named_step) > 1e-6:
                problems.append(f"the {label} replay injected at {requested[0]} and is "
                                f"named for {named_step}")
            elif not any(abs(float(requested[0]) - t) <= 1e-6 for t in steps):
                problems.append(f"the {label} replay injected at {requested[0]}, which the "
                                f"case does not declare")
    return problems


def outcome_for(intervention, index, recomputed=None):
    """What a case's intervention DID for one index, read from its own records.

    "reproduced" means the intervention replay reproduces that reference track exactly
    while neither the untouched replay nor the TIME control does. The SHAPE control may
    reproduce it, and Mozambique's does. That wording matters and an earlier version of
    this docstring got it wrong, saying no control may: the time control is the null one,
    and the shape control is a second intervention whose result narrows the claim rather
    than disqualifying it. See the comment in the body.

    "track_survival" is the weaker outcome the Sahara case has, where nothing reproduces
    the track and the intervention's count of finished tracks in the western box differs
    from every control's. Anything else is no outcome at all.

    A REVIEW READING THIS COLD FOUND WHY THE DISTINCTION MATTERS. The first version of
    this gate required only that the three replays MENTION the index, so a case whose
    intervention reproduced nothing was credited beside cases that reproduced their tracks
    exactly, and the weaker case was presented as the stronger one. Requiring an examined
    index was a better stand-in for an outcome, not an outcome."""
    labels = [l for l in REQUIRED_RUNS if intervention.get(l)]
    exact, west = {}, {}
    for label in labels:
        run = intervention[label]
        if recomputed is not None:
            # THE RECOMPUTED VALUES, from the replay's recorded tracks against the pinned
            # reference, never the flags. The flags were required to agree with them
            # before this is reached; here they are not read at all.
            got = recomputed.get(label) or {}
            if index in (got.get("exact") or {}):
                exact[label] = bool(got["exact"][index])
            west[label] = got.get("west")
            continue
        for rec in run.get("reference_tracks") or []:
            if rec.get("reference_index") == index:
                exact[label] = bool(rec.get("reproduced_exactly"))
        tracks = run.get("finished_tracks_in_the_western_box")
        west[label] = None if tracks is None else len(tracks)
    if set(exact) != set(labels):
        missing = sorted(set(labels) - set(exact))
        return None, f"its replays {missing} record no outcome for it"
    # THE NULL CONTROL IS THE ONE IN TIME, the same injection at a step the two sides
    # already agree on, and it is the one that must not reproduce the track. The SHAPE
    # control is a second intervention rather than a null: it injects an axis where the
    # port had none, with the vertices moved off the crossing, and on three of the four
    # cases with a performed experiment it reproduces the tracks too. What that shows is
    # as narrow as the test: THE REFERENCE VERTICES ARE NOT NECESSARY AMONG THE PLACEMENTS
    # TESTED, one offset in one direction at one magnitude, which says nothing about
    # arbitrary placements. An earlier wording here read it as showing the loss is "the
    # absence of an axis rather than the absence of that line", which is wider than one
    # placement supports and is withdrawn. Treating shape success as a disqualifier would
    # still refuse a case for being better understood, so it does not.
    nulls = [l for l in labels if l == "control"]
    if not nulls:
        # AN ABSENT NULL IS NOT A PASSED NULL. `not any([])` is True, so without this an
        # experiment with no time control at all read as one the control failed to
        # reproduce. The command never reaches here without one, since the experiment
        # contract refuses first, but this function is public and must hold on its own.
        return None, "no time control was recorded, so nothing tests the null"
    if exact["intervention"] and not exact["baseline"] \
            and not any(exact[l] for l in nulls):
        return "reproduced", None
    if exact["baseline"] or any(exact[l] for l in nulls):
        return None, ("the untouched replay reproduces it" if exact["baseline"]
                      else "the time control reproduces it as well as the intervention")
    if west.get("intervention") is None or west["baseline"] is None:
        return None, "no finished-track measurement to fall back on"
    others = [west[l] for l in ["baseline"] + nulls if west.get(l) is not None]
    # SURVIVAL MEANS MORE TRACKS, not a different number of them. A review set a genuine
    # artifact's baseline and control counts to two and its intervention's to zero, and
    # the first version credited that LOSS under a label that says the opposite.
    if len(others) == len(nulls) + 1 and all(west["intervention"] > o for o in others):
        return "track_survival", None
    if len(others) == len(nulls) + 1 and all(west["intervention"] < o for o in others):
        return None, ("its intervention REMOVES finished tracks from the western box, "
                      "which is not the survival outcome")
    return None, "neither reproduces it nor adds finished tracks to its western box"



def scope_problems(name, case, residuals, explains):
    """Why the steps a case declares are not the steps its claim is ABOUT.

    Two outside reviews found the same thing: the checker verified the case's vertices
    against the log INSIDE A GATE THE CASE DREW ITSELF. Narrow the box or drop a divergence
    step and a missing observation leaves the contract, and a case can then explain the
    part of a pair it can fix and be credited for the whole. The residual artifact records,
    for every version-1-extra pair, the times at which version 1 holds an observation and
    the port does not, and those are the steps the case must declare, exactly.

    For an unmatched track the residual record carries only its span, so the check there
    is the weaker one, that every declared step lies inside it. That is stated as the
    limit it is rather than dressed as the same guarantee."""
    problems = []
    steps = sorted(f"{float(t):.4f}" for t in
                   ((case.get("parameters") or {}).get("divergence_steps") or ()))
    if not steps:
        return [f"{name} declares no divergence step"]
    by_pair = {p["v1_index"]: p for p in residuals.get("pairs") or ()}
    for i in explains.get("v1_extra_pairs") or ():
        pair = by_pair.get(i)
        if pair is None:
            continue                     # the category check reports this one
        want = sorted(f"{float(t):.4f}" for t in
                      ((pair.get("v1_extra") or {}).get("times") or ()))
        if not want:
            problems.append(f"{name} claims pair {i}, for which the residual artifact "
                            f"records no version-1-extra times")
        elif steps != want:
            problems.append(f"{name} claims pair {i} and declares divergence steps "
                            f"{steps}, while the pair's version-1-extra observations are "
                            f"at {want}")
    return problems


def _retired_span_and_box_checks():
    """Retired. The span and mean-location checks that stood here judged an unmatched
    track by a lifetime that admits steps it was never observed at, and a box that is not
    the selector. Both are now read from the retained outputs in `evidence_problems`."""


def walk_problems(name, case, claimed, recomputed=None):
    """Why a case may not be credited for an index it claims, and how it is credited when
    it may. Returns (problems, outcomes).

    THE EXPERIMENT IS VALIDATED FIRST, under the same contract the trace applies, with the
    expected divergence and control steps taken from the case's own declared parameters."""
    intervention = case.get("intervention") or {}
    if not intervention.get("baseline"):
        return ([f"{name} claims {sorted(claimed)} and records no intervention, so nothing "
                 f"in it examined those indices"], {})
    parameters = case.get("parameters") or {}
    # `control_step_problems` is NOT read from the artifact here. It could only ever add
    # refusals, so an empty list hid nothing, but it is the judged object's own verdict on
    # its control steps and has no place in the contract the checker applies. The trace
    # still computes and records it for its own refusal.
    broken = experiment_problems(
        intervention, parameters.get("divergence_steps"), parameters.get("control_steps"),
        (), dumped_vertices((case.get("at_divergence") or {}).get("v1")))
    if broken:
        return ([f"{name} claims {sorted(claimed)} and {w}" for w in broken], {})
    problems, outcomes = [], {}
    for index in sorted(claimed):
        outcome, why = outcome_for(intervention, index, recomputed)
        if outcome is None:
            problems.append(f"{name} claims {index}, which {why}")
        else:
            outcomes[index] = outcome
    return problems, outcomes



CREDIT_BASIS = {
    "read_from_retained_evidence": [
        "the case is bound to the residual artifact's exported window by the digests of "
        "the tracker input and the port output",
        "every claimed index is a member of the category it is claimed for",
        "for a PAIR, the divergence steps are the times version 1's finished track holds "
        "an observation and the port's finished counterpart does not, read from the two "
        "pinned outputs by final index, and both the residual record and the case's "
        "declaration must equal that set exactly",
        "for an UNMATCHED TRACK, every declared step is a time version 1's finished track "
        "holds an observation, read from the pinned reference output by final index",
        "every time-control step is one at which version 1's finished track holds an "
        "observation, and for a pair so does the port's counterpart",
        "the axis vertices the case recorded at the judged steps are every axis its "
        "declared reference log holds there under the effective selection region, "
        "re-read from the retained log by this module's own parser"],
    "derived_under_a_declared_region": [
        "the effective selection region, the axis box where declared and the case box "
        "otherwise, is the case's own; it is required to contain version 1's observation "
        "at every declared step, and its extent beyond that is not determined by anything "
        "independent",
        "for an unmatched track, WHICH of the reference track's observation times the "
        "case declares as divergence steps is the case's own"],
    "claimed_by_the_replay_records": [
        "that each injection was applied at the time requested (the receipts are the "
        "replay's own, required to be complete and to agree with the schedule)",
        "that each replay produced the finished tracks it records"],
    "recomputed_from_the_replay_records_against_the_pinned_reference": [
        "reproduced_exactly for every claimed index and every replay, as exact equality "
        "between a recorded finished track and the pinned reference track, with the "
        "replay's own flag required to agree",
        "the count of finished tracks in the western box during the feature's life, for "
        "every replay, with the replay's own list required to agree"],
    "what_a_credit_therefore_means": (
        "a case whose scope, control-step agreement and injected geometry are the retained "
        "runs' own, and whose outcomes are arithmetic on the tracks it records against an "
        "independent reference. It does not establish that the replay produced those "
        "tracks; that needs a rerun, which is a separate audit gate. Read the counts as "
        "recomputed from recorded replays, not as independently reproduced.")}


def membership(residuals, cases, logs=None, log_problems=(), outputs=None):
    """The accounting, which grants no credit without the reference logs.

    `logs` is the verified reference runs, keyed by the digest of their contents, and it
    is REQUIRED rather than optional. A mode that produces the same answer without them
    would be the escape hatch the ordinary path takes, and a run that skipped the check
    must never report the result of one."""
    binding = binding_problems(residuals, cases)
    if binding:
        return {"problems": binding}
    if log_problems:
        return {"problems": list(log_problems)}
    if not logs:
        return {"problems": ["no reference log was supplied, so no case's recorded axis "
                             "vertices can be traced to the run that produced them"]}
    if not outputs:
        return {"problems": ["no finished output was supplied, so no case's scope or "
                             "outcome can be read from the retained runs"]}
    unmatched_no_eligible = sorted(u["index"] for u in residuals["v1_unmatched"]
                                   if not u["eligible_counterpart_exists"])
    unmatched_eligible = sorted(u["index"] for u in residuals["v1_unmatched"]
                                if u["eligible_counterpart_exists"])
    by_kind = {}
    for p in residuals["pairs"]:
        by_kind.setdefault(p["extra_kind"], []).append(p["v1_index"])
    for k in by_kind:
        by_kind[k] = sorted(by_kind[k])
    explained_unmatched, explained_pairs, problems, per_case = [], [], [], {}
    by_outcome = {}
    for name, case in cases.items():
        ex = case.get("explains") or {}
        per_case[name] = ex
        claimed = list(ex.get("unmatched_v1_tracks", [])) + list(ex.get("v1_extra_pairs", []))
        if claimed:
            # THE SCOPE AND THE SOURCE ARE CHECKED BEFORE THE EXPERIMENT. A case judged at
            # steps it chose, or on vertices its reference log does not hold, has nothing
            # for the experiment checks to rest on, however consistent its own fields are
            # with each other. The category checks below run regardless, since they are
            # independent of both and a refusal should name every reason it has.
            bad = (scope_problems(name, case, residuals, ex)
                   + evidence_problems(name, case, residuals, ex, outputs)
                   + source_problems(name, case, logs))
            problems.extend(bad)
            if not bad:
                why, recomputed = recomputed_outcomes(name, case, ex, outputs)
                problems.extend(why)
                if not why:
                    why2, outcomes = walk_problems(name, case, claimed, recomputed)
                    problems.extend(why2)
                    by_outcome.update(outcomes)
        for i in ex.get("unmatched_v1_tracks", []):
            if i not in unmatched_no_eligible:
                problems.append(f"{name} claims unmatched track {i}, which is not in the "
                                f"no-eligible-counterpart set")
            elif i in explained_unmatched:
                problems.append(f"unmatched track {i} is claimed by more than one case")
            elif i in by_outcome:
                explained_unmatched.append(i)
        for i in ex.get("v1_extra_pairs", []):
            if i not in by_kind.get("extra_v1_only", []):
                problems.append(f"{name} claims pair {i}, which is not a version-1-extra pair")
            elif i in explained_pairs:
                problems.append(f"pair {i} is claimed by more than one case")
            elif i in by_outcome:
                # A CREDIT IS GRANTED ONLY WHERE AN OUTCOME WAS READ. The lists used to be
                # built from the declaration alone, so an index whose experiment had just
                # been refused above still appeared among the explained.
                explained_pairs.append(i)
    def split(indices):
        return {"reproduced": sorted(i for i in indices
                                     if by_outcome.get(i) == "reproduced"),
                "track_survival": sorted(i for i in indices
                                         if by_outcome.get(i) == "track_survival")}

    return {"unmatched_v1_no_eligible_counterpart": unmatched_no_eligible,
            # THE OUTCOME EACH CREDIT RESTS ON, beside the credit, so a case that changes
            # a track count is never read as one that reproduced a track
            "explained_by_outcome": {"unmatched": split(explained_unmatched),
                                     "v1_extra_pairs": split(explained_pairs)},
            "unmatched_v1_with_eligible_counterpart": unmatched_eligible,
            "pairs_by_extra_kind": by_kind,
            "explained_unmatched": sorted(explained_unmatched),
            "remaining_unmatched": sorted(set(unmatched_no_eligible) - set(explained_unmatched)),
            "explained_v1_extra_pairs": sorted(explained_pairs),
            "remaining_v1_extra_pairs": sorted(set(by_kind.get("extra_v1_only", []))
                                               - set(explained_pairs)),
            "counts": {"unmatched_no_eligible": len(unmatched_no_eligible),
                       "unmatched_explained": len(explained_unmatched),
                       "unmatched_explained_by_exact_reproduction":
                           len([i for i in explained_unmatched
                                if by_outcome.get(i) == "reproduced"]),
                       "v1_extra_pairs_explained_by_exact_reproduction":
                           len([i for i in explained_pairs
                                if by_outcome.get(i) == "reproduced"]),
                       "v1_extra_pairs": len(by_kind.get("extra_v1_only", [])),
                       "v1_extra_pairs_explained": len(explained_pairs),
                       "both_sides_extra_pairs": len(by_kind.get("extra_both_sides", []))},
            "per_case": per_case, "problems": problems,
            # WHAT A CREDIT HERE RESTS ON, stated in the artifact because the field names
            # above say "explained" and a reader will take that at its width.
            "credit_basis": CREDIT_BASIS,
            # WHICH GRADE APPLIED TO WHICH INDEX, because a static block cannot say that
            # the pair-scope check ran for a pair and the weaker one for a track
            "credit_basis_by_index": {
                **{str(i): "pair: scope fixed by the residual pair's own extra times"
                   for i in explained_pairs},
                **{str(i): "unmatched track: declared steps checked against the log's "
                           "track lines and the recorded span, the choice among them "
                           "the case's own" for i in explained_unmatched}}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--residuals", required=True)
    ap.add_argument("--cases", nargs="*", default=[])
    ap.add_argument("--reference-log", nargs="+", required=True, dest="reference_logs",
                    help="the instrumented version 1 run(s) the cases rest on, retained "
                         "under docs/aewc_v2/evidence and pinned in "
                         "docs/aewc_v2/artifacts/reference_logs.json")
    ap.add_argument("--reference-output", nargs="+", required=True, dest="reference_outputs",
                    help="the retained finished outputs, version 1's reference tracks and "
                         "the port's, pinned in the same manifest")
    ap.add_argument("--manifest", default=None,
                    help="the record of accepted reference runs (default "
                         "docs/aewc_v2/artifacts/reference_logs.json)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    with open(args.residuals) as fh:
        residuals = json.load(fh)
    cases = {}
    for path in args.cases:
        with open(path) as fh:
            cases[os.path.basename(path)] = json.load(fh)
    logs, log_problems = verified_logs(args.reference_logs, args.manifest)
    outputs, out_problems = verified_outputs(args.reference_outputs, args.manifest)
    result = membership(residuals, cases, logs, list(log_problems) + list(out_problems),
                        outputs)
    if result["problems"]:
        print("REFUSED: " + "; ".join(result["problems"]), flush=True)
        return 2
    print(json.dumps(result["counts"]))
    print("credit basis: scope, agreement and geometry read from the retained outputs and "
          "log; outcomes recomputed from each replay's recorded tracks against the pinned "
          "reference; the tracks themselves are the replay's own record "
          "(see credit_basis in the artifact)")
    print("remaining unmatched:", result["remaining_unmatched"])
    print("remaining version-1-extra pairs:", result["remaining_v1_extra_pairs"])
    if args.out:
        result.update({"generated_by": "scripts/residue_membership.py",
                       "input_sha256": {os.path.basename(p): _sha256(p)
                                        for p in [args.residuals] + list(args.cases)},
                       # the runs whose axis vertices every credited case was checked
                       # against, by the digest of the log's own contents
                       "reference_logs_sha256": sorted(logs),
                       "reference_outputs_sha256": sorted(outputs),
                       "source_sha256": {"scripts/residue_membership.py":
                                         _sha256(os.path.abspath(__file__))}})
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1, sort_keys=True)
        print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
