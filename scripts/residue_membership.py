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


def binding_problems(residuals, cases, exchange=None):
    """The reasons a case artifact is NOT a case from the residual artifact's own
    comparison. Track indices are local to a run, so a case must agree with the
    residual artifact on the case identifier, the exported-case digest and the
    port-output digest before any of its claimed indices can count. The reference
    output digests are NOT compared, because instrumented runs with different dump
    schedules legitimately carry different producer metadata over identical tracks.
    A review copied a genuine case artifact, renamed its case and replaced these
    digests, and the first version credited its claims.

    TWO DIGESTS MAY DIFFER AND STILL NAME THE SAME EVIDENCE. `savemat` is not
    deterministic, so re-exporting an identical window writes identical numbers under a
    new digest, and on 2026-09-21 that is exactly what happened after the scratch holding
    the original was cleared. A digest that a RETAINED exchange input records as a
    verified reserialization of is therefore accepted, and the acceptance is recorded so
    the artifact says which binding held. The file digests are still required to agree
    with each other: the relaxation is only that a recorded, recomputed content
    equivalence stands in for byte identity."""
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
            elif c_hashes[f] != r_hashes[f] and not same_exchange(
                    c_hashes[f], r_hashes[f], f, exchange):
                problems.append(f"{name}: {f} differs from the residual artifact's, so the "
                                f"case comes from another export or another port run")
            elif not equivalent_exchange(c_hashes[f], f, exchange):
                problems.append(f"{name}: {f} names digest {c_hashes[f][:12]}, which is "
                                f"neither a retained exchange input nor a verified "
                                f"reserialization of one")
    return problems


def same_exchange(a, b, role, exchange):
    """Whether two DIFFERENT digests name the same retained exchange input, one of them as
    a recorded reserialization of the other.

    This is what lets a case built after the 2026-09-21 rebuild bind to a residual
    artifact written before it. Both digests must resolve, through the retained manifest,
    to the same file in the same role, whose content identity was recomputed when the
    manifest was read."""
    if not exchange:
        return False
    x, y = exchange.get(a), exchange.get(b)
    return bool(x and y) and x["role"] == role and y["role"] == role \
        and x["path"] == y["path"]


def equivalent_exchange(digest, role, exchange):
    """Whether a declared exchange digest names evidence this project retains, either as
    itself or as something a retained file is a verified reserialization of.

    With no retained exchange supplied at all the question is not asked, which keeps every
    caller that does not carry the exchange working as before."""
    if not exchange:
        return True
    known = exchange.get(digest)
    return bool(known) and known.get("role") == role


REQUIRED_RUNS = ("baseline", "intervention", "control", "shape_control")

# THE SMALLEST TRANSLATION A SHAPE CONTROL MAY DECLARE, in degrees of longitude. The check
# below once required only that the shifted vertices differ from the intervention's by more
# than 1e-9, so a shift of 1e-8 degrees, which changes nothing any stage computes, counted
# as a placement test. THIS FLOOR IS AN OPERATIONAL CONVENTION, NOT A PHYSICAL THRESHOLD.
# An earlier comment justified it as the tracker's minimum wave extent, and that was wrong:
# MIN_EXTENT_DEG is a north-south span test on a merged wave, and it says nothing about how
# far an axis must move before the merge or the association would treat it differently.
# The value is one degree because it is well below the four degrees every retained case
# declares and well above anything indistinguishable from no translation. Changing it is
# a decision, and it is Hilawe's.
MIN_SHAPE_TRANSLATION_DEG = 1.0

# THE OPERATION AN EXPERIMENT PERFORMS, added 2026-09-22. Until then every check here was
# written for an INJECTION, which adds version 1's axes at declared steps and is controlled
# in TIME and in SHAPE. Two final-pair results are a REORDERING and a REMOVAL, and neither
# fits: a reordering supplies no geometry for the axis check to re-read, and a shape control
# is not an operation on a permutation. The contract named them as excluded, which kept the
# counts honest and left the results uncarried. This carries them, with checks specific to
# each operation and EVERY COMMON EVIDENCE CHECK PRESERVED: the exchange binding, the scope
# and examined-index rules, the recomputation of outcomes from recorded tracks against the
# pinned reference, and the refusal of malformed tracks all apply unchanged.
OPERATIONS = ("injection", "removal", "reordering")

# A REPRESENTATION CONTROL performs the same mechanical operation with NO semantic change
# and MUST reproduce the baseline exactly; when it does not, the apparatus moves the run by
# itself and the experiment is unattributable. A NEGATIVE CONTROL performs a comparable
# operation on an UNRELATED target and MUST NOT reproduce the reference track; it is not
# required to reproduce the baseline whole, because acting elsewhere may legitimately change
# other tracks. They are different instruments and the first version of this file's successor
# proposal called both "the control".
REQUIRED_RUNS_BY_OPERATION = {
    "injection": REQUIRED_RUNS,
    "removal": ("baseline", "intervention", "representation_control", "negative_control"),
    "reordering": ("baseline", "intervention", "representation_control", "negative_control"),
}


def operation_of(intervention):
    """Which operation an experiment records, defaulting to injection.

    THE DEFAULT IS WHAT MAKES EVERY EXISTING ARTIFACT STILL VALID. Sixteen cases were
    written before this field existed and none carries it; they are injections and are read
    as injections. A new artifact states its operation explicitly."""
    return ((intervention or {}).get("operation") or "injection")


def required_runs_for(intervention):
    return REQUIRED_RUNS_BY_OPERATION.get(operation_of(intervention), REQUIRED_RUNS)

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


def declared_discrepancy_for(params, index):
    """The discrepancy declaration a case makes for one pair, in either of two forms.

    A case claiming ONE pair declares `discrepancy_times` as three lists. A case claiming
    SEVERAL, of which this project has three holding eight of the fifteen credited pairs,
    declares `discrepancy_times_by_pair` keyed by the pair index, because one set of three
    lists cannot describe two pairs. A first version of the deriver assumed the single form
    and would have left those eight refused. The per-pair form wins where both are present."""
    by_pair = (params or {}).get("discrepancy_times_by_pair") or {}
    if str(index) in by_pair:
        return by_pair[str(index)]
    return (params or {}).get("discrepancy_times")


def _any_discrepancy_declared(params):
    single = (params or {}).get("discrepancy_times") or {}
    if any(single.get(k) for k in ("v1_extra", "port_extra", "displaced")):
        return True
    return any(any((d or {}).get(k) for k in ("v1_extra", "port_extra", "displaced"))
               for d in ((params or {}).get("discrepancy_times_by_pair") or {}).values())


def derived_discrepancies(v1, counterpart):
    """Where two finished tracks disagree, read from the tracks themselves.

    THREE KINDS, AND THEY ARE NOT THE SAME QUESTION. `v1_extra` are times the reference
    holds an observation and the port does not, `port_extra` the reverse, and `displaced`
    times BOTH hold one and the positions are not identical. A pair can have any
    combination, and the 1990 window happens to contain only the first and the third, which
    is why the contract was written as though the first were the only one.

    DISPLACEMENT IS EXACT INEQUALITY, not a threshold. The residual record classifies a pair
    as displaced using its own tolerance, which is the right instrument for grouping pairs.
    For SCOPE the question is where the two finished tracks differ at all, and a tolerance
    there would put the boundary of a case's scope at a number nobody chose for that purpose.

    THIS IS NOT AN INTERVENTION SCHEDULE. These are the times the two OUTPUTS differ. Where
    an intervention must act to change that is a separate question this cannot answer, and
    on pair 30 the answer was a step that appears in none of these three sets, because the
    port's extra candidate was taken by a fragment that was later pruned. The contract keeps
    the two apart deliberately and grades them differently."""
    v1_times, port_times = _times(v1), _times(counterpart)
    displaced = []
    for key in sorted(v1_times & port_times):
        a, b = _at(v1, key), _at(counterpart, key)
        if a is not None and b is not None and (a[0] != b[0] or a[1] != b[1]):
            displaced.append(key)
    return {"v1_extra": sorted(v1_times - port_times),
            "port_extra": sorted(port_times - v1_times),
            "displaced": displaced}


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
        # THE FULL DISCREPANCY, all three kinds, where the case declares it. Required of any
        # pair whose disagreement is not purely reference-extra, because `divergence_steps`
        # cannot express those and reading its emptiness as "no disagreement" is exactly the
        # failure validation phase A found. A pair that IS purely reference-extra may declare
        # it and is checked against the same derivation if it does.
        found = derived_discrepancies(v1, counterpart)
        declared_disc = declared_discrepancy_for(params, i)
        purely_v1_extra = not found["port_extra"] and not found["displaced"]
        if declared_disc is None:
            if not purely_v1_extra:
                problems.append(
                    f"{name} claims pair {i}, whose finished tracks disagree in ways "
                    f"`divergence_steps` cannot express ({len(found['port_extra'])} port-extra "
                    f"and {len(found['displaced'])} displaced times), and declares no "
                    f"discrepancy_times")
        else:
            want = {k: sorted(f"{float(t):.4f}" for t in (declared_disc.get(k) or ()))
                    for k in ("v1_extra", "port_extra", "displaced")}
            for kind in ("v1_extra", "port_extra", "displaced"):
                if want[kind] != found[kind]:
                    problems.append(
                        f"{name} claims pair {i} and declares {kind} discrepancy times "
                        f"{want[kind]}, while the retained outputs give {found[kind]}")
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


def _canonical_tracks(tracks):
    """A track collection as sorted exact tuples, PRESERVING DUPLICATE MULTIPLICITY.

    THE CHECK THIS REPLACES WAS `all(any(_same_track(x, y) for y in mirror) for x in
    baseline)` BESIDE A LENGTH TEST, and it did not consume matches. A baseline of [A, A]
    therefore passed against a control of [A, B]: both copies of A matched the control's
    single A and B was never examined. Version 1 DUPLICATES TRACKS, holding several at
    identical positions, so this is the shape its own output takes rather than a contrived
    one, and a representation control that silently swapped a track for another would have
    been accepted.

    A multiset comparison is enough BECAUSE `_same_track` IS EXACT. If it ever gained a
    tolerance this would have to become a complete matching rather than a sort, since
    tolerant equality is not transitive and sorting would then depend on order.
    """
    out = []
    for tr in tracks:
        out.append((tuple(float(x) for x in (tr.get("time") or [])),
                    tuple(float(x) for x in (tr.get("lat") or [])),
                    tuple(float(x) for x in (tr.get("lon") or []))))
    return sorted(out)


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
    labels = [l for l in required_runs_for(intervention) if intervention.get(l)] \
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


# WHERE THE CODE WAS CHECKED OUT, not what ran. `source_sha256` inside the same record
# pins the executed sources by content, so these two are strictly weaker provenance and
# are the only fields that move when an identical export is reserialized.
EXCHANGE_LABEL_FIELDS = ("git_head", "git_dirty")


def _mat_payload(path, drop=()):
    """A MAT file's fields, normalized back to the shapes the writer held before
    serialization: singleton dimensions squeezed, string arrays returned as strings.

    A MAT round trip turns a vector into a row and a scalar into a one-by-one, so a digest
    taken over the loaded arrays is not the digest the writer took. Squeezing recovers
    every shape in these payloads, which is checked by recomputing a known identifier.

    THE PRECONDITION: squeezing recovers the original shape only where the original held
    no singleton dimension of its own. That holds for these payloads, whose arrays are
    grids and stacks, and it is verified rather than assumed, because a caller compares
    the recomputed identifier against the one the file stores and REFUSES on disagreement.
    A payload that broke the precondition would be refused, not silently accepted."""
    import numpy as np
    from scipy.io import loadmat
    out = {}
    for name, value in loadmat(path).items():
        if name.startswith("__") or name in drop:
            continue
        array = np.asarray(value)
        out[name] = (str(array.ravel()[0]) if array.dtype.kind in "US"
                     else np.squeeze(array))
    return out


def exchange_content_identity(path, role):
    """What an exchange input IS, computed from its loaded fields, never read out of it.

    A review of this project's own binding found it resting on FILE DIGESTS while the
    thing it stands for is content. `savemat` is not deterministic, so re-exporting an
    identical window yields identical numbers under a different digest, and the sixteen
    committed cases would have been refused for a rewrite rather than for any difference
    in evidence. Measured on the 2026-09-21 rebuild: every field identical, the port's 114
    tracks identical track for track, and only `git_head` and `git_dirty` moved.

    THE STORED IDENTIFIER IS NOT TRUSTED. For the exported window the identity is
    `export_tracker_case.case_id` RECOMPUTED over the loaded payload with the stored
    `case_id` removed, which is a digest over every exported field. For the port output it
    is the case identifier it carries, its finished tracks, its settings and the provenance
    that says what ran, with the two label fields above excluded.

    Returns (identity, problems)."""
    import hashlib
    import numpy as np
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from export_tracker_case import case_id
    if not os.path.exists(path):
        return {}, [f"the exchange input {path} is not present"]
    try:
        payload = _mat_payload(path, drop=("case_id",))
        stored = _mat_payload(path).get("case_id")
        if role == "tracker_case.mat":
            identity = {"case_id": case_id(payload)}
        else:
            h = hashlib.sha256()
            for name in sorted(k for k in payload if k[:3] in ("lat", "lon", "tim")):
                h.update(name.encode())
                h.update(np.ascontiguousarray(
                    np.asarray(payload[name], dtype=float)).tobytes())
            record = json.loads(payload.get("producer_json") or "{}")
            # EVERY OTHER FIELD, not a chosen few. The settings a run was made under sit
            # at the top level beside the tracks, and an identity that named only the
            # tracks and the producer record would have called two runs with different
            # settings the same evidence.
            other = {}
            for name in sorted(payload):
                if name == "producer_json" or name[:3] in ("lat", "lon", "tim"):
                    continue
                value = payload[name]
                other[name] = (value if isinstance(value, str)
                               else np.asarray(value, dtype=float).ravel().tolist())
            identity = {"tracks_sha256": h.hexdigest(),
                        "fields": other,
                        "provenance": {k: v for k, v in sorted(record.items())
                                       if k not in EXCHANGE_LABEL_FIELDS}}
    except (ImportError, KeyError, ValueError, TypeError, OSError) as e:
        return {}, [f"the exchange input {path} cannot be read as {role}: {e}"]
    if role == "tracker_case.mat" and str(stored) != identity["case_id"]:
        # the stored label disagreeing with the recomputed identity is the one case where
        # the file is refused outright, since it cannot be both
        return {}, [f"the exchange input {path} stores case identifier {stored} and its "
                    f"own fields give {identity['case_id']}"]
    return identity, []


def verified_exchange(manifest_path=None):
    """The retained exchange inputs, keyed by the FILE digest a case may declare for them,
    including the predecessor digests each one is a verified reserialization of.

    The manifest's own record of a content identity is not taken on trust: each retained
    file's identity is recomputed here and must agree with it."""
    accepted, problems = read_manifest(manifest_path)
    if problems:
        return {}, problems
    out = {}
    for entry in accepted.get("__exchange__", []):
        path, role = entry.get("file"), entry.get("role")
        identity, why = exchange_content_identity(path, role)
        if why:
            problems.extend(why)
            continue
        digest = _sha256(path)
        if digest != entry.get("sha256"):
            problems.append(f"the retained exchange input {path} has digest "
                            f"{digest[:12]} and the manifest names {str(entry.get('sha256'))[:12]}")
            continue
        if entry.get("content_identity") != identity:
            problems.append(f"the retained exchange input {path} does not hold the "
                            f"content identity the manifest records for it")
            continue
        for known in [digest] + list(entry.get("reserialization_of") or ()):
            out[known] = {"role": role, "path": path, "identity": identity,
                          "is_reserialization": known != digest}
    return out, problems


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
    accepted["__exchange__"] = list(manifest.get("exchange") or ())
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
    if abs(shift) < MIN_SHAPE_TRANSLATION_DEG:
        return {}, [f"the shape control's recorded translation of {shift} degrees is "
                    f"below the floor of {MIN_SHAPE_TRANSLATION_DEG}, an operational "
                    f"convention, so it cannot test placement"]
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


def _in_range(value, population):
    return isinstance(value, int) and isinstance(population, int) and 0 <= value < population


def _control_receipt_problems(operation, intervention, population):
    """Why a removal's or reordering's CONTROLS are not the controls they are labelled.

    AN INDEPENDENT REVIEW SHOWED THE CHECKER ACCEPTED A BASELINE COPY LABELLED
    `negative_control`. Nothing asked what the control CHANGED; a dictionary with that name
    and finished tracks that fail to reproduce the reference was enough. It also admitted an
    intervention declaring moved_from=-500 and moved_to=500 against a population of 77. Both
    are the same defect: the artifact asserted an operation and nothing checked that one
    occurred.

    A control therefore carries a RECEIPT of what it did, and this checks the receipt:
      - the REPRESENTATION control acted at the intervention's step and left the population
        and every position unchanged (an identity edit, so its tracks must equal the baseline,
        which the caller checks from the tracks themselves);
      - the NEGATIVE control acted at the same step on a DIFFERENT target, with the same
        population arithmetic as the intervention, and every index inside the population.

    WHAT IS DELIBERATELY NOT REQUIRED: that a negative control's finished output differ from
    the baseline. A real input change can legitimately leave the output unchanged, and the
    review was explicit that refusing on that would reject sound controls.

    WHAT A RECEIPT ESTABLISHES, AND NO MORE. A receipt is a DECLARATION of a consistent
    operation. A confirmation review copied a baseline into `negative_control`, invented
    internally consistent receipt fields, and was credited, which is correct behaviour for a
    consistency check and wrong to describe as showing the control "acted". Whether the
    declared operation was executed is the replay's own claim, exactly as the tracks
    themselves are, and it is bound only by the provenance the case records: a retained
    runs file with its digest, produced by a frozen diagnostic whose digest that file
    carries. A case must say which of those two standings its receipts have, in
    `receipts_basis`, and the credit basis grades them as claimed by the replay records."""
    problems = []
    run = intervention.get("intervention") or {}
    if operation == "removal":
        removed = run.get("removed_index")
        if not _in_range(removed, population):
            problems.append(f"the removal records removed_index {removed!r}, which is not an "
                            f"index inside its population of {population}")
    if operation == "reordering":
        for key in ("moved_from", "moved_to"):
            if not _in_range(run.get(key), population):
                problems.append(f"the reordering records {key} {run.get(key)!r}, which is "
                                f"not an index inside its population of {population}")
    step = run.get("changed_at")
    basis = str(intervention.get("receipts_basis") or "").strip()
    if not basis:
        problems.append("the experiment records no receipts_basis, so it does not say "
                        "whether its control receipts are bound to retained execution "
                        "evidence or declared from a frozen diagnostic")
    for label in ("representation_control", "negative_control"):
        ctl = intervention.get(label) or {}
        receipt = ctl.get("receipt")
        if not isinstance(receipt, dict):
            problems.append(f"the {label} records no receipt of what it changed, so it "
                            f"cannot be told from a baseline under another name")
            continue
        if receipt.get("step") != step:
            problems.append(f"the {label} acted at {receipt.get('step')!r}, not at the "
                            f"intervention's step {step!r}")
        b, a = receipt.get("population_before"), receipt.get("population_after")
        if not isinstance(b, int) or not isinstance(a, int) or b != population:
            problems.append(f"the {label} records population {b!r} -> {a!r} against the "
                            f"intervention's {population}")
            continue
        if label == "representation_control":
            if receipt.get("identity") is not True or a != b:
                problems.append("the representation control does not record an identity "
                                "edit leaving the population unchanged")
            continue
        # the negative control, which must be the same KIND of change to a DIFFERENT target
        if operation == "removal":
            if a != b - 1:
                problems.append(f"the negative control records {b} -> {a}, which is not the "
                                f"removal of exactly one")
            other = receipt.get("removed_index")
            if not _in_range(other, population):
                problems.append(f"the negative control's removed_index {other!r} is not "
                                f"inside the population")
            elif other == run.get("removed_index"):
                problems.append("the negative control removed the SAME axis as the "
                                "intervention, so it controls nothing")
        if operation == "reordering":
            if a != b:
                problems.append(f"the negative control records {b} -> {a}, so it did not "
                                f"only reorder")
            src, dst = receipt.get("moved_from"), receipt.get("moved_to")
            if not _in_range(src, population) or not _in_range(dst, population):
                problems.append(f"the negative control's move {src!r} -> {dst!r} is not "
                                f"inside the population")
            elif src == run.get("moved_from"):
                problems.append("the negative control moved the SAME axis as the "
                                "intervention, so it controls nothing")
            elif src == dst:
                problems.append("the negative control moved an axis to its own index, "
                                "which is an identity and not a negative control")
    return problems


def operation_problems(operation, intervention, steps, parameters=None, window=None):
    """Why a REMOVAL or a REORDERING is not a recorded experiment of that kind.

    These are the checks an injection's geometry rules cannot stand in for. What they do
    NOT replace is the common evidence: the exchange binding, the scope and examined-index
    rules, the refusal of malformed tracks and the recomputation of every outcome from
    recorded tracks against the pinned reference all run for these operations exactly as
    they do for an injection, because they are about the artifact rather than the
    operation."""
    problems = []
    run = intervention.get("intervention") or {}
    changed = run.get("changed_at")
    try:
        changed = float(changed)
    except (TypeError, ValueError):
        problems.append(f"the {operation} records no readable changed_at step")
        changed = None
    # THE INTERVENTION STEP IS THE CASE'S OWN, and requiring it to be a DECLARED DIVERGENCE
    # STEP was wrong. The first version of this check did require that, and writing pair 30's
    # artifact is what exposed it: that case's removal acts at 33026.50, where VERSION 1 HAS
    # NOTHING, which is the entire finding. Divergence steps are the times version 1 holds an
    # observation and the port does not, so a removal's target can never be one of them, and
    # the residual record's port_extra times are empty for that pair because the extra
    # candidate was taken by a FRAGMENT that was later pruned rather than by the paired track.
    # The step is therefore not derivable from retained evidence at all. What can be required
    # is that it lies in the region the case declares and that the case says HOW it was found,
    # and the credit basis grades it as the case's own rather than as read from evidence.
    region = life if (life := (parameters or {}).get("feature_life")) else window
    if changed is not None and region and len(region) == 2:
        low, high = float(region[0]), float(region[1])
        if not low - 1e-6 <= changed <= high + 1e-6:
            problems.append(f"the {operation} changes the run at {changed}, outside the "
                            f"region the case declares, {low} to {high}")
    if not str(run.get("changed_at_basis") or "").strip():
        problems.append(f"the {operation} does not record HOW its step was chosen, which is "
                        f"the case's own judgment and cannot be read from the residual record")
    before, after = run.get("population_before"), run.get("population_after")
    if not isinstance(before, int) or not isinstance(after, int) or before <= 0:
        problems.append(f"the {operation} records no candidate population before and after")
    else:
        # THE POPULATION ARITHMETIC IS WHAT MAKES THE OPERATION THE ONE IT CLAIMS. A removal
        # that changed the count by anything but one, or a reordering that changed it at
        # all, is a different experiment under this experiment's name.
        if operation == "removal" and after != before - 1:
            problems.append(f"the removal records {before} candidates before and {after} "
                            f"after, which is not the removal of exactly one")
        if operation == "reordering" and after != before:
            problems.append(f"the reordering records {before} candidates before and {after} "
                            f"after, so it did not only reorder")
    if operation == "removal":
        if not run.get("removed"):
            problems.append("the removal does not record WHICH candidate it removed")
        if not run.get("identified_by"):
            problems.append("the removal does not record how the removed candidate was "
                            "identified, so the choice cannot be told from a search for one "
                            "that works")
    if operation == "reordering":
        moved_from, moved_to = run.get("moved_from"), run.get("moved_to")
        if not isinstance(moved_from, int) or not isinstance(moved_to, int):
            problems.append("the reordering does not record the index it moved from and to")
        elif moved_from == moved_to:
            problems.append("the reordering records a move to the index it moved from, "
                            "which changes nothing")
        if run.get("positions_unchanged") is not True:
            problems.append("the reordering does not assert that every candidate POSITION is "
                            "unchanged, which is the whole claim of a reordering")
    problems.extend(_control_receipt_problems(operation, intervention, before))
    # THE REPRESENTATION CONTROL MUST REPRODUCE THE BASELINE EXACTLY, and that is checked
    # here from the recorded tracks rather than taken from a flag, because a control the
    # apparatus moved is an experiment nothing can be attributed to.
    baseline = (intervention.get("baseline") or {}).get("finished_tracks")
    mirror = (intervention.get("representation_control") or {}).get("finished_tracks")
    if not isinstance(baseline, list) or not isinstance(mirror, list):
        problems.append("the baseline or the representation control records no finished "
                        "tracks, so the control cannot be checked against the baseline")
    else:
        bad = [w for tr in list(baseline) + list(mirror) for w in _track_problems(tr)]
        if bad:
            problems.append(f"a recorded track cannot be compared: {bad[0]}")
        elif _canonical_tracks(baseline) != _canonical_tracks(mirror):
            problems.append("the representation control does not reproduce the baseline "
                            "exactly, so the intervention beside it is unattributable")
    return problems


def experiment_problems(intervention, divergence_steps, control_steps=None,
                        control_step_problems=(), reference_vertices=None,
                        parameters=None, window=None):
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
    runs = required_runs_for(intervention)
    if not intervention or not any(isinstance(intervention.get(l), dict) for l in runs):
        return ["the artifact records no intervention"]
    operation = operation_of(intervention)
    if operation not in OPERATIONS:
        return [f"the experiment declares operation {operation!r}, which is not one of "
                f"{list(OPERATIONS)}"]
    absent = [r for r in runs if not intervention.get(r)]
    if absent:
        # reported ONCE. The label loop below used to append the same absence again.
        problems.append("the experiment records no " + " or ".join(absent) + " replay")
    problems.extend(control_step_problems or [])
    steps = sorted(float(x) for x in (divergence_steps or []))
    if not steps:
        # AN EMPTY REFERENCE-EXTRA SET IS NOT AN EMPTY DISAGREEMENT, which is what phase A
        # of the validation found. A pair whose extras are on the PORT's side, or which is
        # merely displaced, has no divergence step by the old definition and is still a real
        # residue item. An INJECTION still requires one, because there is nothing to inject
        # without it. A removal or a reordering does not, provided the case declares where
        # the finished tracks actually disagree.
        if operation == "injection" or not _any_discrepancy_declared(parameters):
            problems.append(
                "the case declares no divergence step" if operation == "injection" else
                "the case declares neither a divergence step nor any discrepancy time, so "
                "it names no disagreement for an experiment to be about")
            return problems
        problems.extend(operation_problems(operation, intervention, steps,
                                           parameters=parameters, window=window))
        return problems

    def finite(x):
        try:
            return math.isfinite(float(x))
        except (TypeError, ValueError):
            return False

    if operation != "injection":
        # EVERY CHECK ABOVE IS COMMON and has already run: the runs are present, the control
        # steps agree, and the case declares divergence steps. What follows is specific to
        # adding geometry, so a removal or a reordering takes its own checks instead.
        problems.extend(operation_problems(operation, intervention, steps,
                                           parameters=parameters, window=window))
        return problems

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
            if label not in absent:            # the sweep above already named it
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
    labels = [l for l in required_runs_for(intervention) if intervention.get(l)]
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
    # WHICH RUNS MUST FAIL depends on the operation. An injection is controlled in TIME,
    # and its shape control may reproduce the track, which narrows the claim rather than
    # disqualifying it. A REMOVAL or a REORDERING has no shape control and instead carries
    # two runs that must BOTH fail: the NEGATIVE control, a comparable operation on an
    # unrelated target, and the REPRESENTATION control, the same operation with no semantic
    # change, which reproduces the baseline and therefore must not reproduce the reference.
    if operation_of(intervention) == "injection":
        nulls = [l for l in labels if l == "control"]
        missing_null = "no time control was recorded, so nothing tests the null"
    else:
        nulls = [l for l in labels if l in ("negative_control", "representation_control")]
        missing_null = ("no negative or representation control was recorded, so nothing "
                        "tests the null")
        if len(nulls) < 2:
            # BOTH ARE REQUIRED AND THEY ARE NOT INTERCHANGEABLE. One shows the apparatus is
            # inert, the other that the effect is specific to what was changed, and an
            # experiment carrying only one of them establishes only half of that.
            return None, ("a removal or reordering needs BOTH a representation control and "
                          "a negative control, and this records " + str(sorted(nulls)))
    if not nulls:
        # AN ABSENT NULL IS NOT A PASSED NULL. `not any([])` is True, so without this an
        # experiment with no time control at all read as one the control failed to
        # reproduce. The command never reaches here without one, since the experiment
        # contract refuses first, but this function is public and must hold on its own.
        return None, missing_null
    if exact["intervention"] and not exact["baseline"] \
            and not any(exact[l] for l in nulls):
        return "reproduced", None
    if exact["baseline"] or any(exact[l] for l in nulls):
        if exact["baseline"]:
            return None, "the untouched replay reproduces it"
        # NAMED AS THE READER KNOWS THEM. The injection's null run is stored under the
        # label "control" and has always been reported as "the time control"; keeping that
        # wording matters because it is what the refusal has always said.
        spoken = {"control": "time control", "negative_control": "negative control",
                  "representation_control": "representation control"}
        failed = sorted(spoken.get(l, l) for l in nulls if exact[l])
        return None, (f"the {' and the '.join(failed)} reproduces it as well as the "
                      f"intervention")
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



def experiment_recorded(case):
    """Whether a case records ANY replay evidence.

    A case that records some is a performed OR AN INCOMPLETE experiment and is validated
    as one. Only a case that records none at all is exempt, which is what a deliberately
    unperformed case looks like: an intervention block naming what was attempted and why
    it was not, with no replays. A review deleted one case's BASELINE ALONE, left its
    intervention, both controls and its partial runs in place, and the whole validation
    path was skipped, because the test for a performed experiment was the baseline."""
    iv = case.get("intervention") or {}
    if not isinstance(iv, dict):
        return False
    labels = list(required_runs_for(iv)) + list(iv.get("partial_runs") or ())
    if any(isinstance(iv.get(l), dict) for l in labels):
        return True
    return bool(iv.get("injected_axes")) or bool(iv.get("declared_steps"))


def examined_explains(name, case, residuals, claimed=()):
    """The indices a case's experiment EXAMINED, in the shape `explains` has, classified
    by the residual artifact rather than by the case, with the reasons the declaration is
    not usable. Returns (explains-shaped, problems).

    A review found that a case claiming nothing was never checked at all, and then that
    the first repair could be walked round three ways: by deleting the declaration, by
    declaring an index the residual artifact does not know (which the first version
    silently filtered away to an empty set), and by deleting one replay so the case read
    as unperformed. A NEGATIVE RESULT RESTS ON THE SAME CONTROLS A POSITIVE ONE DOES, so
    a performed experiment must say what it examined, every index must be one the residual
    artifact classifies, the declaration must agree with what the replays recorded
    outcomes for, and anything claimed must be among them."""
    declared = list((case.get("parameters") or {}).get("reference_tracks") or ())
    problems = []
    # EVERY NONIDENTICAL PAIR, whatever its category. This once kept only `extra_v1_only`,
    # so the phase B amendment's new categories could be validated by a helper and never
    # reach the artifact, which an independent review called "true of a helper and false of
    # the artifact". The category still matters for INJECTION, which needs a reference-extra
    # step, and that is enforced where injections are checked rather than by hiding pairs.
    pairs = {p["v1_index"] for p in residuals.get("pairs") or ()}
    tracks = {u["index"] for u in residuals.get("v1_unmatched") or ()}
    out = {"v1_extra_pairs": [], "unmatched_v1_tracks": []}
    if not declared:
        problems.append(f"{name} records replays and declares no reference tracks, so "
                        f"nothing says which indices its experiment examined")
    for i in declared:
        if i in pairs:
            out["v1_extra_pairs"].append(i)
        elif i in tracks:
            out["unmatched_v1_tracks"].append(i)
        else:
            # REFUSED, NOT FILTERED. A review set this list to a single unknown index and
            # the first version quietly produced an empty examined set, which checked
            # nothing at all.
            problems.append(f"{name} declares reference track {i}, which the residual "
                            f"artifact classifies as neither a version-1-extra pair nor "
                            f"an unmatched track")
    iv = case.get("intervention") or {}
    recorded = {r.get("reference_index")
                for l in list(required_runs_for(iv)) + list(iv.get("partial_runs") or ())
                for r in ((iv.get(l) or {}).get("reference_tracks") or ())}
    if recorded and recorded != set(declared):
        problems.append(f"{name} declares reference tracks {sorted(declared)} and its "
                        f"replays record outcomes for {sorted(recorded)}")
    outside = sorted(i for i in claimed if i not in declared)
    if outside:
        problems.append(f"{name} claims {outside}, which its experiment does not declare "
                        f"among the reference tracks it examined")
    return out, problems


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
    params = case.get("parameters") or {}
    steps = sorted(f"{float(t):.4f}" for t in (params.get("divergence_steps") or ()))
    if not steps and not _any_discrepancy_declared(params):
        # A PAIR WITH NO REFERENCE-EXTRA STEP IS STILL A PAIR, if it declares where its
        # finished tracks disagree. Refusing here unconditionally is what kept port-extra
        # and displaced-only pairs out of the public path.
        return [f"{name} declares neither a divergence step nor any discrepancy time"]
    by_pair = {p["v1_index"]: p for p in residuals.get("pairs") or ()}
    for i in explains.get("v1_extra_pairs") or ():
        pair = by_pair.get(i)
        if pair is None:
            continue                     # the category check reports this one
        want = sorted(f"{float(t):.4f}" for t in
                      ((pair.get("v1_extra") or {}).get("times") or ()))
        if not want and not declared_discrepancy_for(params, i):
            # A pair with no reference-extra time is refused ONLY if the case also declares
            # no discrepancy at all. With discrepancy_times declared, an empty v1_extra is
            # what a port-only or displaced-only pair looks like, and `steps` must then
            # equal `want`, both empty, which the branch below checks.
            problems.append(f"{name} claims pair {i}, for which the residual artifact "
                            f"records no version-1-extra times and the case declares no "
                            f"discrepancy times")
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
    if not intervention.get("baseline") and not experiment_recorded(case):
        return ([f"{name} claims {sorted(claimed)} and records no intervention, so nothing "
                 f"in it examined those indices"], {})
    parameters = case.get("parameters") or {}
    # `control_step_problems` is NOT read from the artifact here. It could only ever add
    # refusals, so an empty list hid nothing, but it is the judged object's own verdict on
    # its control steps and has no place in the contract the checker applies. The trace
    # still computes and records it for its own refusal.
    broken = experiment_problems(
        intervention, parameters.get("divergence_steps"), parameters.get("control_steps"),
        (), dumped_vertices((case.get("at_divergence") or {}).get("v1")),
        parameters=parameters, window=parameters.get("window"))
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
        "A PAIR'S FULL DISCREPANCY IS THREE SETS, not one, derived from the two finished "
        "tracks: v1_extra as above, PORT_EXTRA the reverse, and DISPLACED where both hold "
        "an observation and the positions are not identical. Displacement here is EXACT "
        "INEQUALITY rather than a threshold, because a tolerance would put the boundary of "
        "a case's scope at a number chosen for grouping pairs and not for this. A pair "
        "whose disagreement is not purely v1_extra MUST declare discrepancy_times, and the "
        "declaration must equal the derivation exactly. Added 2026-09-22 after the "
        "validation's phase A measured that the 1990 window contains only v1_extra and "
        "displaced pairs, so a rule reading an empty v1_extra set as no disagreement was "
        "invisible in the window every instrument was built against.",
        "for an UNMATCHED TRACK, every declared step and every control step is a time "
        "version 1's finished track holds an observation, read from the pinned reference "
        "output by final index",
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
        "case declares as divergence steps is the case's own",
        "A DISCREPANCY TIME AND AN INTERVENTION TIME ARE DIFFERENT QUANTITIES AND ARE "
        "GRADED DIFFERENTLY. Discrepancy times are where the two OUTPUTS differ and are "
        "read from retained evidence. An intervention time is where an experiment ACTS, is "
        "the case's own, and neither constrains nor is constrained by the other. Pair 30 is "
        "the instance: its removal acts at a step in none of its discrepancy sets.",
        "for a REMOVAL or a REORDERING, the STEP THE INTERVENTION ACTS AT is the case's "
        "own. It is required to lie inside the region the case declares and to carry a "
        "recorded basis saying how it was chosen, and it is NOT derived from anything "
        "here. Requiring it to be a declared divergence step was tried and was WRONG: a "
        "divergence step is a time version 1 holds an observation and the port does not, "
        "so a removal's target can never be one, and on the case that forced this the "
        "residual record carries no port-extra time either, because the extra candidate "
        "was taken by a fragment that was later pruned rather than by the paired track. "
        "Read a removal's step as a claim the case makes and states its grounds for."],
    "claimed_by_the_replay_records": [
        "that each injection was applied at the time requested (the receipts are the "
        "replay's own, required to be complete and to agree with the schedule)",
        "that each replay produced the finished tracks it records",
        "that a control performed the operation its RECEIPT declares. A receipt is checked "
        "for consistency, step, target and bounds, and a consistent receipt can be written "
        "by hand, which a confirmation review demonstrated. The experiment's "
        "receipts_basis says whether the receipts are bound to a retained runs file with a "
        "recorded digest or declared from a frozen diagnostic, and neither standing is "
        "execution evidence"],
    "recomputed_from_the_replay_records_against_the_pinned_reference": [
        "reproduced_exactly for every claimed index and every replay, as exact equality "
        "between a recorded finished track and the pinned reference track, with the "
        "replay's own flag required to agree",
        "the count of finished tracks in the western box during the feature's life, for "
        "every replay, with the replay's own list required to agree"],
    "intervention_kinds_this_contract_admits": [
        "INJECTION, REMOVAL and REORDERING. An experiment declares its operation, and an "
        "artifact that declares none is read as an INJECTION, which is what every case "
        "written before 2026-09-22 is.",
        "AN INJECTION adds version 1's dumped axes at declared steps and is controlled in "
        "TIME and in SHAPE. Its axis-geometry check re-reads the injected vertices from the "
        "reference log, and its receipts must show each injection applied at the time "
        "requested. The shape control MAY reproduce the track, which narrows the claim "
        "rather than disqualifying it.",
        "A REMOVAL takes one candidate out, and a REORDERING moves one within the list. "
        "Neither supplies geometry to re-read and neither has a meaningful shape control, "
        "since translating a permutation four degrees west is not an operation. Each is "
        "checked on its own terms instead: the step it changes is the case's own, graded "
        "under derived_under_a_declared_region; a removal must change the candidate count by exactly one and "
        "record WHICH candidate and HOW it was identified, so a principled choice can be "
        "told from a search for one that works; a reordering must change the count by "
        "nothing, name the indices it moved between, and assert every position unchanged.",
        "BOTH CARRY TWO CONTROLS THAT ARE NOT INTERCHANGEABLE. A REPRESENTATION CONTROL "
        "performs the same mechanical operation with no semantic change and MUST reproduce "
        "the baseline exactly, checked here from the recorded tracks rather than from a "
        "flag; when it does not, the apparatus moves the run by itself and the experiment "
        "is unattributable. A NEGATIVE CONTROL performs a comparable operation on an "
        "UNRELATED target and MUST NOT reproduce the reference track; it is not required to "
        "reproduce the baseline whole, because acting elsewhere may legitimately change "
        "other tracks. A credit requires BOTH, and an experiment recording only one "
        "establishes half of what it claims."],
    "what_the_common_checks_are": [
        "EVERY OPERATION TAKES THEM, because they are about the artifact rather than the "
        "operation: the exchange binding by content identity, the membership of every "
        "claimed index in the category claimed for it, the scope rules that fix the "
        "divergence steps from the two pinned outputs, the requirement that a performed "
        "experiment declare a valid non-empty examined set, the refusal of any malformed "
        "recorded track, and the recomputation of every outcome from recorded tracks "
        "against the pinned reference with the replay's own flag required to agree.",
        "SO EXTENDING THE CONTRACT ADDED CHECKS AND REMOVED NONE. A removal or reordering "
        "credit rests on the same evidence an injection credit does, plus the checks above "
        "that only its own operation can be given."],
    "what_a_credit_therefore_means": (
        "a case whose scope, control-step observation availability (both finished tracks "
        "hold an observation there, which is not agreement of position) and, for an "
        "injection only, injected geometry are read from retained evidence, and whose "
        "outcomes are arithmetic on the tracks it records against an "
        "independent reference. It does not establish that the replay produced those "
        "tracks. That needs a rerun, which is a separate audit gate. Read the counts as "
        "recomputed from recorded replays, not as independently reproduced. The scope is "
        "the three operations named above, each with its own checks and all of them with "
        "the common ones.")}


def membership(residuals, cases, logs=None, log_problems=(), outputs=None, exchange=None):
    """The accounting, which grants no credit without the reference logs.

    `logs` is the verified reference runs, keyed by the digest of their contents, and it
    is REQUIRED rather than optional. A mode that produces the same answer without them
    would be the escape hatch the ordinary path takes, and a run that skipped the check
    must never report the result of one."""
    binding = binding_problems(residuals, cases, exchange)
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
    # WHICH OPERATION EACH CREDITED INDEX WAS EARNED BY, recorded as the cases are walked so
    # the per-index basis can name it. Read from the case's own experiment record, and
    # absent means injection, which is what every case written before the field existed is.
    operation_by_index = {}
    for name, case in cases.items():
        ex = case.get("explains") or {}
        per_case[name] = ex
        for i in (list(ex.get("unmatched_v1_tracks", []))
                  + list(ex.get("v1_extra_pairs", []))):
            operation_by_index[i] = operation_of(case.get("intervention"))
        claimed = list(ex.get("unmatched_v1_tracks", [])) + list(ex.get("v1_extra_pairs", []))
        # A CASE THAT PERFORMED AN EXPERIMENT IS CHECKED WHETHER OR NOT IT CLAIMS ONE.
        # The indices come from the case's claim where it makes one and from the
        # experiment's own reference tracks where it does not. A case that declares NO
        # experiment stays exempt, since there is nothing to validate.
        performed = experiment_recorded(case)
        if claimed or performed:
            # THE SCOPE AND THE SOURCE ARE CHECKED BEFORE THE EXPERIMENT. A case judged at
            # steps it chose, or on vertices its reference log does not hold, has nothing
            # for the experiment checks to rest on, however consistent its own fields are
            # with each other. The category checks below run regardless, since they are
            # independent of both and a refusal should name every reason it has.
            # EVERY EXAMINED INDEX IS CHECKED, not only the claimed ones, and a claim
            # outside what the experiment examined is refused above.
            checking, bad = examined_explains(name, case, residuals, claimed)
            if not bad:
                bad = (scope_problems(name, case, residuals, checking)
                       + evidence_problems(name, case, residuals, checking, outputs)
                       + source_problems(name, case, logs))
            problems.extend(bad)
            if not bad:
                why, recomputed = recomputed_outcomes(name, case, checking, outputs)
                problems.extend(why)
                if not why:
                    # the EXPERIMENT contract holds for every performed case; only a
                    # CLAIMED index goes on to be credited
                    problems.extend(walk_problems(name, case, [], recomputed)[0])
                    if claimed:
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
        all_pairs = {i for kind in by_kind.values() for i in kind}
        for i in ex.get("v1_extra_pairs", []):
            if i not in all_pairs:
                problems.append(f"{name} claims pair {i}, which is not a nonidentical pair "
                                f"in the residual record")
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
            "explained_pairs": sorted(explained_pairs),
            # REMAINING IS OVER EVERY CATEGORY NOW, and the per-kind breakdown beside it says
            # which categories the credits and the remainder fall in, so a count of credits
            # cannot be read as covering a category the walk never reached.
            "remaining_pairs": sorted(
                {i for kind in by_kind.values() for i in kind} - set(explained_pairs)),
            "explained_pairs_by_kind": {
                kind: sorted(i for i in members if i in explained_pairs)
                for kind, members in sorted(by_kind.items())},
            "remaining_pairs_by_kind": {
                kind: sorted(i for i in members if i not in explained_pairs)
                for kind, members in sorted(by_kind.items())},
            "counts": {"unmatched_no_eligible": len(unmatched_no_eligible),
                       "unmatched_explained": len(explained_unmatched),
                       "unmatched_explained_by_exact_reproduction":
                           len([i for i in explained_unmatched
                                if by_outcome.get(i) == "reproduced"]),
                       "pairs_explained_by_exact_reproduction":
                           len([i for i in explained_pairs
                                if by_outcome.get(i) == "reproduced"]),
                       "v1_extra_pairs": len(by_kind.get("extra_v1_only", [])),
                       "nonidentical_pairs_all_kinds":
                           sum(len(v) for v in by_kind.values()),
                       "pairs_explained": len(explained_pairs),
                       # BY OPERATION, because an injection credit rests on a time and a
                       # shape control failing and a removal or reordering credit on a
                       # negative and a representation control, which are not one kind
                       # of evidence under one label
                       "pairs_explained_by_operation": {
                           op: len([i for i in explained_pairs
                                    if operation_by_index.get(i, "injection") == op])
                           for op in sorted({operation_by_index.get(i, "injection")
                                             for i in explained_pairs})},
                       # RESTRICTED TO ITS OWN KIND, so that beside `v1_extra_pairs` it is
                       # a ratio and not a count of every credit under a narrower name
                       "v1_extra_pairs_explained":
                           len([i for i in explained_pairs
                                if i in by_kind.get("extra_v1_only", [])]),
                       "both_sides_extra_pairs": len(by_kind.get("extra_both_sides", []))},
            "per_case": per_case, "problems": problems,
            # WHAT A CREDIT HERE RESTS ON, stated in the artifact because the field names
            # above say "explained" and a reader will take that at its width.
            "credit_basis": CREDIT_BASIS,
            # THE GRADE OF EVERY CREDIT ABOVE, on the same object as the counts, because a
            # reader who takes "explained" at its width has not read credit_basis
            "credit_grade": ("recorded replay: outcomes recomputed from the tracks each "
                             "replay records against the pinned reference. That the "
                             "replay executed as declared is not independently audited"),
            # WHICH GRADE APPLIED TO WHICH INDEX, because a static block cannot say that
            # the pair-scope check ran for a pair and the weaker one for a track
            # WHICH OPERATION EARNED EACH CREDIT, because since 2026-09-22 they are not
            # all injections and a reader cannot tell from the count. An index credited by
            # a REORDERING or a REMOVAL rests on different operation-specific checks than
            # one credited by an injection, and the per-index line is where that shows.
            "credit_basis_by_index": {
                **{str(i): (f"pair, by {operation_by_index.get(i, 'injection')}: "
                            + ("scope fixed by the residual pair's own extra times"
                               if operation_by_index.get(i, "injection") == "injection"
                               else "scope fixed by the discrepancy times derived from "
                                    "the pinned outputs, and the step the operation acts "
                                    "at is the case's own"))
                   for i in explained_pairs},
                **{str(i): (f"unmatched track, by {operation_by_index.get(i, 'injection')}"
                            ": declared steps and control steps checked "
                            "against version 1's finished track in the pinned reference "
                            "output, the choice among its observation times the case's own")
                   for i in explained_unmatched}}}


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
    exchange, ex_problems = verified_exchange(args.manifest)
    result = membership(residuals, cases, logs,
                        list(log_problems) + list(out_problems) + list(ex_problems),
                        outputs, exchange)
    if result["problems"]:
        print("REFUSED: " + "; ".join(result["problems"]), flush=True)
        return 2
    print("recorded-replay credit, not execution-audited:", json.dumps(result["counts"]))
    print("credit basis: scope and shared control-step observations read from the retained "
          "outputs, injected geometry from the log for injections only. Outcomes recomputed "
          "from each replay's recorded tracks against the pinned reference. The tracks "
          "themselves are the replay's own record (see credit_basis in the artifact)")
    print("remaining unmatched:", result["remaining_unmatched"])
    print("remaining nonidentical pairs, every kind:", result["remaining_pairs"])
    print("remaining by kind:", json.dumps(result["remaining_pairs_by_kind"]))
    if args.out:
        result.update({"generated_by": "scripts/residue_membership.py",
                       # THE VALIDATOR'S OWN DIGEST, so an artifact can be told stale. The
                       # published fifteen-of-fifteen was produced, then the validator was
                       # changed two commits later, and nothing re-ran it; a gate comparing
                       # this against the current file catches that shape mechanically.
                       "validator_sha256": _sha256(os.path.abspath(__file__)),
                       # THE MANIFEST IS AN INPUT, pinned like the rest
                       "manifest_sha256": _sha256(args.manifest or REFERENCE_MANIFEST),
                       "input_sha256": {os.path.basename(p): _sha256(p)
                                        for p in [args.residuals] + list(args.cases)},
                       # the runs whose axis vertices every credited case was checked
                       # against, by the digest of the log's own contents
                       "reference_logs_sha256": sorted(logs),
                       "reference_outputs_sha256": sorted(outputs),
                       # WHICH BINDING HELD for each exchange input: its own bytes, or a
                       # recomputed content equivalence with a retained reserialization
                       "exchange_binding": {
                           d: ("reserialization of the retained "
                               f"{v['path']}" if v["is_reserialization"]
                               else f"the retained {v['path']}")
                           for d, v in sorted(exchange.items())
                           if d in {h for c in cases.values()
                                    for h in (c.get("input_sha256") or {}).values()}},
                       "source_sha256": {"scripts/residue_membership.py":
                                         _sha256(os.path.abspath(__file__))}})
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1, sort_keys=True)
        print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
