#!/usr/bin/env python3
"""Walk one track version 1 produces and the port does not through both trackers'
intermediate state, timestep by timestep: version 1's track 19 of the 1990 window
(20N 4W over the Sahara, 33024.00 to 33027.50), the "Sahara case".

WHAT IT DOES. On the port side it replays detection, association and the in-loop prune
on the exported window with the port's own functions and logs every candidate inside a
box around the feature with the track that took it. On the version 1 side it reads the
instrumented run's dumped axes and candidates (AXIS, COARSE, FINE lines) inside the same
box from the Octave log. It then isolates the one timestep on which the two sides
diverge, prints the masked advection field there, and runs the port's axis finder on a
synthetic field that has only that field's shape, a lone unmasked column bounded by
masked cells. The Octave side of the synthetic check is scripts/octave/lone_column_check.m,
run here when octave-cli is available and recorded as not run otherwise.

A STAGE DIFFERENCE IS NOT A MECHANISM WITHOUT AN INTERVENTION. When a case declares the
version 1 tracks it says the port would reproduce (`--reference-tracks`), the window is
replayed twice more: once with version 1's own dumped axis vertices added to the port's
axis set at the divergence step alone, and once with the same vertices added at another
timestep as a control. Each replay must actually inject what it requested, the control
step must be a timestep of the window and not the divergence step, and the artifact
reports what each replay reproduced beside the untouched replay. A case that declares no
reference tracks records that the intervention was not attempted and claims no mechanism.

AND WHOSE FIELD IT IS, because a missing port axis can be a masking difference rather
than a contouring one. The instrumented copy dumps version 1's own masked coarse
advection field cell by cell at the dump times, and this compares it against the port's
over the whole grid. A reference log without those records leaves the question open and
the artifact says so.

EVIDENCE IS VALIDATED BEFORE ANYTHING IS CONCLUDED. A review ran this with an empty
reference log and the first version still wrote its conclusion. Now the reference log
must belong to the oracle output beside it (its dumped timesteps equal the producer
record's dump list, its returned track count equals the output's), the three files must
carry one case id, and every observation a conclusion rests on must be present: both
sides' candidates at the agreeing steps, version 1's axes and candidate at the
divergence step, the port's masked field there, and the port's western fragments with
the prune events that removed them. Missing or mismatched evidence is a refusal, and
the conclusion is derived from the recorded observations rather than typed.

WHAT IT ESTABLISHES, and the width is deliberate. The two sides detect the same western
feature at 33024.00, 33024.50 and 33024.75. At 33025.00 version 1 has a coarse
candidate at 18N 9W and the port has none, and on the masked advection field at that
step the only finite cells in the box are one column at 9W with signs plus, minus, plus.
Version 1's contouring under Octave returns two degenerate three-point axes in such a
column; the port's returns none. The port's western track therefore loses its
prediction chain, its later detections seed short tracks the lifetime prune removes,
and version 1's single track reaches eleven observations, survives, and joins the main
trough for its last three steps. Whether MATLAB's own contouring behaves as Octave's
does on a masked-bounded column is NOT established here.

    AEW_ORACLE_DIR=<exchange dir> .venv/bin/python scripts/trace_sahara_case.py \\
        --out <artifact.json>
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from aew.v1port import climatology as clim  # noqa: E402
from aew.v1port import detection as D  # noqa: E402
from aew.v1port import pipeline as P  # noqa: E402
from aew.v1port.association import associate_step, finalize_tracks, prune_stale_tracks  # noqa: E402

sys.path.insert(0, HERE)
import compare_tracker_oracle as C  # noqa: E402
import residue_membership as RM  # noqa: E402  (the shared experiment contract)

# EVERY MODULE WHOSE CODE DECIDES WHAT THIS ARTIFACT SAYS, fingerprinted into it, because
# a later run under different detection, merge or validation code would otherwise carry
# the same script fingerprint (a review's finding). The last entry was added after a
# review pointed out that the shared experiment contract had become a dependency of this
# trace without joining the record of what produced it: the membership artifact hashed
# that file, so the accounting chain as a whole named the version used, but an individual
# case record did not.
REPLAY_SOURCES = ("src/aew/v1port/detection.py", "src/aew/v1port/contours.py",
                  "src/aew/v1port/association.py", "src/aew/v1port/pipeline.py",
                  "src/aew/v1port/climatology.py", "scripts/compare_tracker_oracle.py",
                  "scripts/residue_membership.py")

# THE CASE PARAMETERS, as module globals so the same validated code walks another case
# by overriding them from the command line; the defaults are the Sahara case. `BOX` is
# where candidates are logged, `WEST` is the history whose port fragments are recorded
# (for the Sahara case its northern edge sits at 20.5N and its eastern at 3W on purpose,
# since the main trough's two long port tracks pass 21N 2W at 33026.25 and a first box
# that reached them listed them as entering the west), `FEATURE_LIFE` bounds the steps
# on which entering that box counts, and the steps name where the two sides agree and
# where they diverge.
V1_TRACK = 19
BOX = {"lat": (12.0, 24.0), "lon": (-14.0, -3.0)}
WEST = {"lat": (16.0, 20.5), "lon": (-12.0, -3.0)}
FEATURE_LIFE = (33024.0, 33026.5)
WINDOW = (33023.5, 33027.5)
AGREEING_STEPS = (33024.0, 33024.5, 33024.75)
DIVERGENCE_STEP = 33025.0
AXIS_BOX = None                  # where axis vertices are compared; None means BOX. The
                                 # third case needed a tight candidate box around the
                                 # missing observation and a wide one for the junction
FIELD_ROWS = (14.0, 22.0)        # the printed sub-box of the masked field, latitudes
FIELD_COLS = (-13.0, -5.0)       # and longitudes


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def in_box(lat, lon, box=None):
    box = BOX if box is None else box
    return box["lat"][0] <= lat <= box["lat"][1] and box["lon"][0] <= lon <= box["lon"][1]


def in_axis_box(lat, lon):
    return in_box(lat, lon, AXIS_BOX if AXIS_BOX is not None else BOX)


def read_producer_text(path):
    """The producer record as stored (text) and parsed, with the outer case id and the
    declared count; the text is what the runner hashed into its RUN line."""
    raw = loadmat(path, variable_names=["producer_json", "case_id", "n"])
    text = str(np.asarray(raw["producer_json"]).ravel()[0]) if "producer_json" in raw else None
    rec = None
    if text is not None:
        try:
            rec = json.loads(text)
        except ValueError:
            rec = {"unparseable": text[:200]}
    return text, rec, str(np.asarray(raw["case_id"]).ravel()[0]).strip(), \
        int(np.asarray(raw["n"]).ravel()[0]) if "n" in raw else None


def parse_log(octave_log):
    """The dumped records and the header facts of an instrumented run's log, including
    the RUN digest the runner prints from its own producer record."""
    records, dumped_times, returned, run_digest = [], set(), None, None
    with open(octave_log) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] in ("AXIS", "AXISPTS", "COARSE", "FINE", "TRACK"):
                dumped_times.add(round(float(parts[1]), 4))
                records.append(parts)
            elif line.startswith("find_ews_f returned"):
                returned = int(parts[2])
            elif parts[0] == "RUN" and len(parts) == 2:
                run_digest = parts[1]
    return records, dumped_times, returned, run_digest


def finished_tracks(path):
    """The finished tracks an output holds, refusing a file whose declared count is not
    backed by complete arrays; returns (tracks, reason)."""
    try:
        tracks, _case = C.read_tracks(path)
    except (SystemExit, KeyError, ValueError, IndexError) as e:
        return None, f"{os.path.basename(path)} does not hold complete finished-track "\
                     f"arrays for its declared count ({e})"
    return tracks, None


def validate_evidence(oracle_dir, case_id, octave_log):
    """Every reason the reference log and the two outputs are NOT evidence for this
    case, or an empty list. Reuses the comparison's full producer contract on both
    outputs (structure, case, digests, settings, tree consistency), requires complete
    finished-track arrays behind the declared counts, and binds the log to the oracle
    output by the RUN digest the runner printed from its own producer record. Dump
    schedule and returned count remain consistency checks."""
    problems = []
    oracle_path = os.path.join(oracle_dir, "tracker_octave_instrumented.mat")
    port_path = os.path.join(oracle_dir, "tracker_port.mat")
    text, oracle_rec, oracle_case, oracle_n = read_producer_text(oracle_path)
    _ptext, port_rec, port_case, port_n = read_producer_text(port_path)
    if oracle_case != case_id or port_case != case_id:
        problems.append("the outputs do not carry the exported case id")
    port_settings = {}
    raw = loadmat(port_path, variable_names=["exclusive", "absorb"])
    for flag in ("exclusive", "absorb"):
        if flag in raw:
            port_settings[flag] = bool(np.asarray(raw[flag]).ravel()[0])
    # THE REPLAY HARD-CODES THESE, so an output produced under other settings would be
    # diagnosed by a computation that is not the one whose digest the artifact records. A
    # review validated exactly such an output without a word.
    for flag, supported in (("exclusive", False), ("absorb", False)):
        if port_settings.get(flag, supported) != supported:
            problems.append(f"the port output was produced with {flag}={port_settings[flag]}, "
                            f"and this replay only reproduces {flag}={supported}")
    oprov = C.oracle_provenance(oracle_dir, "tracker_octave_instrumented.mat", oracle_case,
                                case_id)
    pprov = C.port_provenance(oracle_dir, "tracker_port.mat", port_case, case_id,
                              port_settings)
    if oprov["status"] != "verified":
        problems.append(f"oracle provenance {oprov['status']}: "
                        + "; ".join(oprov.get("problems") or oprov.get("tree_mismatches")
                                    or ["no producer record"]))
    elif not oprov["faithful"]:
        problems.append("the oracle output's producer record says the convex-hull call "
                        "was not repaired")
    if pprov["status"] != "verified":
        problems.append(f"port provenance {pprov['status']}: "
                        + "; ".join(pprov.get("problems") or pprov.get("tree_mismatches")
                                    or ["no producer record"]))
    for path, n in ((oracle_path, oracle_n), (port_path, port_n)):
        tracks, why = finished_tracks(path)
        if why:
            problems.append(why)
        elif n is not None and len(tracks) != n:
            problems.append(f"{os.path.basename(path)} declares {n} tracks but holds "
                            f"{len(tracks)}")
    _records, dumped_times, returned, run_digest = parse_log(octave_log)
    if run_digest is None:
        problems.append("the reference log carries no RUN digest, so it cannot be bound "
                        "to an output")
    elif text is None or hashlib.sha256(text.encode()).hexdigest() != run_digest:
        problems.append("the reference log's RUN digest is not the digest of the oracle "
                        "output's producer record, so the log belongs to another run")
    if oracle_rec is None or "unparseable" in oracle_rec:
        # A RECORD THAT DID NOT PARSE IS NOT A RECORD. The reader returns a placeholder on
        # a parse failure, and the dump-schedule check below was guarded by the presence of
        # its key, so an uninterpretable record SKIPPED that check instead of failing it.
        problems.append("the oracle output's producer record does not parse, so its dump "
                        "list cannot be read")
    elif "dump_times" not in oracle_rec:
        problems.append("the oracle output's producer record names no dump list")
    else:
        expected = {round(float(t), 4) for t in oracle_rec["dump_times"]}
        if not dumped_times:
            problems.append("the reference log holds no dumped records")
        elif dumped_times != expected:
            problems.append(f"the reference log's dumped timesteps ({len(dumped_times)}) are "
                            f"not the producer record's dump list ({len(expected)})")
        if not {round(t, 4) for t in AGREEING_STEPS + (DIVERGENCE_STEP,)} <= expected:
            problems.append("the dump list does not cover the agreeing and divergence steps")
    if returned is None:
        problems.append("the reference log does not report the returned track count")
    elif oracle_n is not None and returned != oracle_n:
        problems.append(f"the reference log reports {returned} tracks but the output holds "
                        f"{oracle_n}")
    return problems, {"oracle": oprov, "port": pprov}


def axis_spy(original, inject, now, captured, applied=None):
    """The detection hook the replay installs: it records the masked field and the axes
    the port drew, and, when the replay is intervening, adds the injected axes at THE
    NAMED TIMESTEP ONLY. A hook that added them at every step would make the control
    reproduce the case as well as the intervention does, and the pair would say nothing.

    EVERY APPLICATION IS RECORDED in `applied`, because a review passed a control time
    that lay outside the window, watched nothing be injected, and read the result as an
    injection that had no effect. An injection that did not happen is not evidence."""
    def spy(latgrid, longrid, advection, level=D.TROUGH_LEVEL):
        out = original(latgrid, longrid, advection, level)
        if inject is not None and now[0] is not None \
                and abs(now[0] - inject["time"]) < 1e-6:
            out = list(out) + [(np.asarray(a["lat"], dtype=float),
                                np.asarray(a["lon"], dtype=float))
                               for a in inject["axes"]]
            if applied is not None:
                applied.append(float(now[0]))
        captured["field"], captured["axes"] = advection, out
        return out
    return spy


def replay_port(case, inject=None, reference_field=None):
    """Port candidates in the box per step with their fate, the masked advection field
    plus axes at the divergence step, the full history of every port track that
    ever enters the western box with the step that pruned it, and the finished tracks,
    from the port's own functions.

    `inject`, when given, adds axes to the port's own axis set at ONE timestep, which is
    how this script intervenes: `{"time": t, "axes": [{"lat": [...], "lon": [...]}, ...]}`.
    The axes are version 1's dumped vertices, read from the reference log, never the
    port's own, and they are added at the named step and nowhere else. Returns the times
    at which an injection was actually applied as its last value, so a caller can refuse
    a replay in which nothing was injected.

    `reference_field`, when given, is version 1's own masked field at the divergence step
    and is compared cell by cell against the port's there.
    """
    times = case["time"].ravel()
    lat_c, lon_c = case["lat_c"].ravel(), case["lon_c"].ravel()
    latgrid_c, longrid_c = np.meshgrid(lat_c, lon_c, indexing="ij")
    ct, ft = P.thresholds_for("ERA-Int", 700)
    captured = {}
    original = D.trough_axes
    now, applied = [None], []
    D.trough_axes = axis_spy(original, inject, now, captured, applied)
    tracks, states, log, divergence = [], [], [], None
    # Histories are keyed by a STABLE tag written into each track dictionary when it is
    # first seen, never by the object's identity: a first version keyed on id(track),
    # Python reuses an identity once a pruned track is freed, and a later track
    # inherited a fragment's record and moved its prune step from 33025.5 to 33031.0.
    histories, next_tag = {}, [0]
    try:
        for step in range(times.size):
            t = float(times[step])
            now[0] = t
            waves = D.detect_troughs(t, latgrid_c, longrid_c, case["u_c"][step],
                                     case["currv_anom_c"][step], case["advcurrv_anom_c"][step],
                                     case["latgrid"], case["longrid"], case["currv_anom"][step],
                                     coarse_threshold=ct, fine_threshold=ft, absorb=False)
            if abs(t - DIVERGENCE_STEP) < 1e-6:
                ri = np.where((lat_c <= FIELD_ROWS[1]) & (lat_c >= FIELD_ROWS[0]))[0]
                ci = np.where((lon_c >= FIELD_COLS[0]) & (lon_c <= FIELD_COLS[1]))[0]
                field = captured["field"][np.ix_(ri, ci)]
                divergence = {
                    "time": t, "rows_lat": lat_c[ri].tolist(), "cols_lon": lon_c[ci].tolist(),
                    "masked_smoothed_advection": [[None if np.isnan(x) else float(x) for x in r]
                                                  for r in field],
                    "finite_cells": int(np.isfinite(field).sum()),
                    "port_axes_in_box": [{"n": int(len(a[0])), "lat_mean": float(np.mean(a[0])),
                                          "lon_mean": float(np.mean(a[1])),
                                          "lat": [float(x) for x in a[0]],
                                          "lon": [float(x) for x in a[1]]}
                                         for a in captured["axes"]
                                         if any(in_axis_box(la, lo)
                                                for la, lo in zip(a[0], a[1]))],
                    "port_waves_in_box": [{"lat_mean": w["lat_mean"], "lon_mean": w["lon_mean"]}
                                          for w in waves if in_box(w["lat_mean"], w["lon_mean"])]}
                # THE TWO SIDES' FIELDS, over the whole coarse grid rather than the
                # printed box, since a contour comes from quads that can straddle a box
                # edge. Without version 1's own field dumped, this records that it was
                # not available rather than assuming the fields agree.
                full = [[None if not np.isfinite(x) else float(x) for x in row]
                        for row in captured["field"]]
                divergence["field_comparison"] = field_comparison(lat_c, lon_c, full,
                                                                   reference_field)
                # THE WHOLE GRID travels with the record, because the crossing geometry
                # cannot be read off the printed crop: the cells beyond it decide whether
                # a crossing is closed or merely unobserved.
                divergence["full_field"] = {"rows_lat": [float(x) for x in lat_c],
                                            "cols_lon": [float(x) for x in lon_c],
                                            "grid": full}
                # THE CROSSING GEOMETRY IS COMPUTED HERE, where the whole grid is in hand,
                # because it cannot be read off the printed crop: the cells beyond the
                # crop decide whether a crossing is closed or merely unobserved.
                divergence["zero_crossings"] = zero_crossings(divergence)
                divergence["crossing_neighborhoods"] = crossing_neighborhoods(
                    [float(x) for x in lat_c], [float(x) for x in lon_c], full,
                    divergence["zero_crossings"], domain_complete=True)
            um = P._median_over(clim.smooth9(case["u"][step]))
            vm = P._median_over(clim.smooth9(case["v"][step]))
            tracks, states = associate_step(tracks, states, waves, step, um, vm, exclusive=False)
            if WINDOW[0] <= t <= WINDOW[1]:
                for w in waves:
                    if not in_box(w["lat_mean"], w["lon_mean"]):
                        continue
                    holders = [len(tr["time"]) for tr in tracks
                               if tr["step"] and tr["step"][-1] == step
                               and abs(tr["meanlat"][-1] - w["lat_mean"]) < 1e-9
                               and abs(tr["meanlon"][-1] - w["lon_mean"]) < 1e-9]
                    log.append({"time": t, "lat_mean": w["lat_mean"], "lon_mean": w["lon_mean"],
                                "n_points": int(np.asarray(w["lat_wave"]).size),
                                "taken_by_tracks_of_length": holders})
            for tr in tracks:
                if "_trace_tag" not in tr:
                    tr["_trace_tag"] = next_tag[0]
                    next_tag[0] += 1
                if any(in_box(la, lo, WEST) and FEATURE_LIFE[0] <= tt <= FEATURE_LIFE[1]
                       for la, lo, tt in zip(tr["meanlat"], tr["meanlon"], tr["time"])):
                    rec = histories.setdefault(tr["_trace_tag"],
                                               {"observations": [], "pruned_at": None})
                    rec["observations"] = [[float(a), float(b), float(c)] for a, b, c
                                           in zip(tr["time"], tr["meanlat"], tr["meanlon"])]
            before_tags = {tr["_trace_tag"] for tr in tracks}
            tracks, states = prune_stale_tracks(tracks, states, step, t, waves)
            for gone in before_tags - {tr["_trace_tag"] for tr in tracks}:
                if gone in histories:
                    histories[gone]["pruned_at"] = t
    finally:
        D.trough_axes = original
    final = finalize_tracks(tracks, total_steps=times.size)
    west = [{"steps": len(tr["time"]), "first": float(min(tr["time"])),
             "last": float(max(tr["time"]))} for tr in final
            if any(in_box(la, lo, WEST) and FEATURE_LIFE[0] <= tt <= FEATURE_LIFE[1]
                   for la, lo, tt in zip(tr["meanlat"], tr["meanlon"], tr["time"]))]
    western = sorted(histories.values(), key=lambda h: h["observations"][0][0])
    return log, divergence, west, western, final, applied


# The reference log prints vertices to four decimals, so rounding can move EACH
# COORDINATE by at most half of the last digit. A first version allowed that much as a
# EUCLIDEAN distance, which is ten times too loose per coordinate, and a review showed it
# accepting a vertex displaced by 4e-4 in one coordinate that no rounding could explain.
VERTEX_PRINT_TOLERANCE = 5e-5
FIELD_RELATIVE_TOLERANCE = 1e-9   # the two sides smooth and mask the same inputs


def reference_field_at(octave_log, t):
    """Version 1's OWN masked coarse advection field at one timestep, cell by cell, from
    the instrumented copy's FIELD records. A cell absent from those records is one
    version 1 masked, so the records carry the mask as well as the values.

    A review is why this exists. It built a second masked field, differing from the
    port's by one restored cell, on which Octave returns the same contour vertices and
    the port's tracer draws a line. So a dumped vertex lying on a zero crossing of the
    port's field is consistent with the two sides holding DIFFERENT fields, and only the
    fields themselves separate a contouring difference from a masking one."""
    out, problems, records = {}, [], 0
    with open(octave_log) as fh:
        for line in fh:
            parts = line.split()
            # THE TOKEN, not the spelling. A review sent a tab-delimited record, which the
            # literal prefix test skipped, so a contradictory value for a cell already read
            # went unseen and the comparison reported agreement.
            if not parts or parts[0] != "FIELD":
                continue
            if len(parts) != 5:
                problems.append(f"a FIELD record with {len(parts) - 1} fields")
                continue
            try:
                when, lat, lon, value = (float(parts[1]), float(parts[2]),
                                         float(parts[3]), float(parts[4]))
            except ValueError:
                problems.append("a FIELD record whose numbers do not parse")
                continue
            # A NOT-A-NUMBER TIMESTAMP passes every comparison against the step, so it
            # would join whatever step is being read; a review put one through.
            if not all(np.isfinite(x) for x in (when, lat, lon)):
                problems.append("a FIELD record whose time or position is not finite")
                continue
            if abs(when - t) > 1e-6:
                continue
            records += 1
            key = (round(lat, 4), round(lon, 4))
            if key in out and out[key] != value:
                # TWO RECORDS FOR ONE CELL are contradictory evidence, and taking the last
                # silently resolves the contradiction in favour of whichever came last.
                problems.append(f"two different values for the cell at {key}")
            out[key] = value
    return {"cells": out, "records": records, "problems": problems}


def field_comparison(lat_c, lon_c, port_field, reference):
    """The comparison, or the reason there is none. A reference record set that carries
    problems is refused rather than compared, and an absent one is recorded as absent."""
    if not reference or not reference.get("cells"):
        return {"available": False,
                "reason": "the reference log holds no FIELD records at this step, so "
                          "version 1's own masked field is unknown"}
    if reference.get("problems"):
        return {"available": False, "reason": "; ".join(sorted(set(reference["problems"])))}
    out = compare_fields(lat_c, lon_c, port_field, reference["cells"])
    out["reference_records_read"] = reference.get("records")
    return out


def relative_difference(a, b):
    """The relative difference between two field values, defined at zero. A first version
    divided by the larger magnitude, which raised on two equal zeros (a legitimate value
    of the field being contoured) and, worse, returned not-a-number against an infinite
    reference value, where the comparison's `d > worst` test then left the worst
    difference at zero and the verdict at agreement."""
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale else 0.0


def compare_fields(lat_c, lon_c, field, reference, examples=10):
    """The port's masked field against version 1's over the WHOLE coarse grid: the cells
    each side left unmasked, and the largest relative difference where both did. The
    comparison is over the whole grid rather than the printed box, because a contour is
    drawn from quads that can straddle a box edge.

    EVERY REFERENCE CELL IS ACCOUNTED FOR, including one whose coordinates are not on the
    port's grid at all, because a comparison that walks the port's cells alone cannot see
    what the other side holds elsewhere and would call that agreement. Reference values
    that are not finite are refused rather than compared, for the same reason."""
    both, worst, worst_cell = 0, 0.0, None
    port_only, v1_only, unusable = [], [], []
    seen = set()
    for i, lat in enumerate(lat_c):
        for j, lon in enumerate(lon_c):
            key = (round(float(lat), 4), round(float(lon), 4))
            seen.add(key)
            p = field[i][j]
            v = reference.get(key)
            p = None if p is None or not np.isfinite(p) else float(p)
            if v is not None and not np.isfinite(v):
                unusable.append([float(lat), float(lon)])
                continue
            if p is None and v is None:
                continue
            if v is None:
                port_only.append([float(lat), float(lon)])
            elif p is None:
                v1_only.append([float(lat), float(lon)])
            else:
                both += 1
                d = relative_difference(p, v)
                if d > worst:
                    worst, worst_cell = d, [float(lat), float(lon)]
    off_grid = sorted(k for k in reference if k not in seen)
    for lat, lon in off_grid:
        (unusable if not np.isfinite(reference[(lat, lon)]) else v1_only).append([lat, lon])
    return {"available": True, "cells_unmasked_on_both_sides": both,
            "cells_unmasked_only_in_the_port": len(port_only),
            "cells_unmasked_only_in_version_1": len(v1_only),
            "reference_cells_off_the_port_grid": len(off_grid),
            "reference_cells_not_finite": len(unusable),
            "example_port_only_cells": port_only[:examples],
            "example_version_1_only_cells": v1_only[:examples],
            "example_unusable_cells": unusable[:examples],
            "reference_cells_dumped": len(reference),
            "worst_relative_difference": worst, "worst_cell": worst_cell,
            "same_field": (not port_only and not v1_only and not unusable and both > 0
                           and worst <= FIELD_RELATIVE_TOLERANCE)}


def zero_crossings(field):
    """The linear zero crossings on the edges between neighboring finite cells of the
    printed field, each named by the two cells it lies between.

    Version 1's contouring interpolates the same way on the same quantity, so a dumped
    vertex that coincides with one of these was drawn on a field the port also holds.
    That is what separates a CONTOURING difference from a MASKING difference: without it,
    a missing port axis could equally mean the port masked cells version 1 kept."""
    rows, cols = field["rows_lat"], field["cols_lon"]
    grid = field["masked_smoothed_advection"]
    out = []
    for i, lat in enumerate(rows):
        for j, lon in enumerate(cols):
            a = grid[i][j]
            if a is None:
                continue
            if j + 1 < len(cols):
                b = grid[i][j + 1]
                if b is not None and (a > 0) != (b > 0):
                    out.append({"lat": lat, "lon": lon + (cols[j + 1] - lon) * (a / (a - b)),
                                "between": [[lat, lon], [lat, cols[j + 1]]]})
            if i + 1 < len(rows):
                b = grid[i + 1][j]
                if b is not None and (a > 0) != (b > 0):
                    out.append({"lat": lat + (rows[i + 1] - lat) * (a / (a - b)), "lon": lon,
                                "between": [[lat, lon], [rows[i + 1], lon]]})
    return out


def crossing_neighborhoods(rows, cols, grid, crossings, domain_complete):
    """For each zero crossing, the two quads that share its edge and how many corners of
    each are masked.

    THE PROSE HAS BEEN WRONG ABOUT THIS TWICE AND THE FIRST GENERATED VERSION ONCE, so
    read the three corrections together. The first prose said the port draws nothing "in a
    one-cell-wide run of unmasked cells bounded by masked cells", which is a slogan rather
    than an obstruction. The second said each adjacent quad carries a masked corner, and a
    reviewer showed the port's tracer runs with `corner_mask=True` and traces the unmasked
    TRIANGLE of a quad that has lost ONE corner. What leaves no path is two or more masked
    corners in the quads on BOTH sides of the edge.

    THE THIRD CORRECTION IS WHY `domain_complete` IS AN ARGUMENT. The first generated
    version ran over the PRINTED CROP and treated a quad reaching outside it as no path at
    all. A crop says nothing about the cells beyond it: a reviewer cropped a field so that
    the quad completing a crossing fell outside, watched this report the crossing closed,
    and then traced that very crossing on the full field. An absent quad is therefore only
    an obstruction when the field really ends there, and otherwise the crossing is
    UNDETERMINED (`closed_off` is None) rather than established."""
    index = {(round(float(la), 4), round(float(lo), 4)): (i, j)
             for i, la in enumerate(rows) for j, lo in enumerate(cols)}

    def corner(i, j):
        if not (0 <= i < len(rows) and 0 <= j < len(cols)):
            return None                      # no cell here: absent, which is not "masked"
        return grid[i][j] is None

    out = []
    for c in crossings:
        (i0, j0), (i1, j1) = (index[(round(a, 4), round(b, 4))] for a, b in c["between"])
        horizontal = i0 == i1
        quads = []
        for step in (-1, 1):
            corners = ([(i0 + step, j0), (i0 + step, j1), (i0, j0), (i0, j1)] if horizontal
                       else [(i0, j0 + step), (i1, j0 + step), (i0, j0), (i1, j1)])
            states = [corner(i, j) for i, j in corners]
            absent = any(x is None for x in states)
            quads.append({"outside_the_field": absent,
                          "beyond_the_domain": absent and domain_complete,
                          "unobserved": absent and not domain_complete,
                          "masked_corners": sum(1 for x in states if x)})
        if any(q["unobserved"] for q in quads):
            closed = None                    # the field handed in does not say
        else:
            closed = all(q["beyond_the_domain"] or q["masked_corners"] >= 2 for q in quads)
        out.append({"lat": c["lat"], "lon": c["lon"], "between": c["between"],
                    "quads": quads, "closed_off": closed})
    return out


def vertices_on_crossings(v1_points, crossings):
    """Each dumped vertex with the nearest zero crossing of the port's field and the
    distance to it, so the statement rests on the measured distance rather than on the
    two numbers looking alike."""
    out = []
    for rec in v1_points:
        for lat, lon in zip(rec["lat"], rec["lon"]):
            nearest, best, coords = None, None, None
            for c in crossings:
                dlat, dlon = abs(c["lat"] - lat), abs(c["lon"] - lon)
                d = float(np.hypot(dlat, dlon))
                if best is None or d < best:
                    nearest, best, coords = c, d, (dlat, dlon)
            out.append({"vertex": [lat, lon], "nearest_crossing": nearest,
                        "distance_deg": best,
                        "latitude_difference": None if coords is None else coords[0],
                        "longitude_difference": None if coords is None else coords[1],
                        # PER COORDINATE, because that is what four-decimal printing
                        # bounds; a Euclidean bound of the same size is ten times looser
                        "on_a_crossing": coords is not None
                        and coords[0] <= VERTEX_PRINT_TOLERANCE + 1e-12
                        and coords[1] <= VERTEX_PRINT_TOLERANCE + 1e-12})
    return out


def track_equal(port_track, reference):
    """Whether a finished port track is the reference track: equal step for step in time,
    latitude and longitude, with no tolerance. The tie-rule correction showed that a port
    track which reproduces a reference track reproduces it exactly, so a tolerance here
    would let a near miss read as a reproduction."""
    return (np.array_equal(np.asarray(port_track["time"], dtype=float), reference["time"])
            and np.array_equal(np.asarray(port_track["meanlat"], dtype=float), reference["lat"])
            and np.array_equal(np.asarray(port_track["meanlon"], dtype=float), reference["lon"]))


def reproduction(final, reference):
    """For each reference track, whether some finished port track equals it exactly, and
    how near the closest port track that HOLDS EVERY ONE OF ITS TIMESTEPS comes.

    The covering requirement is not decoration. A first version took the smallest worst
    separation over any track sharing any step, and a one-step fragment that happened to
    coincide reported a separation of zero for a reference track it reproduced no part
    of."""
    out = []
    for index, ref in reference:
        best, covering, best_steps = None, 0, None
        for tr in final:
            t = np.asarray(tr["time"], dtype=float)
            i = np.searchsorted(t, ref["time"])
            if np.any(i >= t.size) or not np.array_equal(t[np.minimum(i, t.size - 1)],
                                                         ref["time"]):
                continue
            covering += 1
            worst = float(np.max(np.hypot(
                np.asarray(tr["meanlat"], dtype=float)[i] - ref["lat"],
                np.asarray(tr["meanlon"], dtype=float)[i] - ref["lon"])))
            if best is None or worst < best:
                best, best_steps = worst, int(t.size)
        out.append({"reference_index": index, "reference_steps": int(ref["time"].size),
                    "reproduced_exactly": any(track_equal(tr, ref) for tr in final),
                    "port_tracks_covering_its_steps": covering,
                    # the nearest track's OWN length, because the distance is measured on
                    # the reference's steps alone and a longer track can sit at zero
                    # separation there while holding observations the reference does not
                    "nearest_port_track_steps": best_steps,
                    "nearest_worst_step_deg": best})
    return out


def observations_at(final, t):
    """The distinct positions the finished port tracks hold at one timestep inside the
    box, which is the observation this case says the port loses."""
    return sorted({(round(float(la), 6), round(float(lo), 6))
                   for tr in final
                   for tt, la, lo in zip(tr["time"], tr["meanlat"], tr["meanlon"])
                   if abs(float(tt) - t) < 1e-6 and in_box(la, lo)})


def control_step_problems(case, control_time):
    """Why a control time is not a control. It has to be a timestep of the exported
    window, or nothing is injected and the replay is the untouched one under another
    name, and it has to differ from the divergence step, or it is the intervention."""
    times = [float(t) for t in np.asarray(case["time"]).ravel()]
    problems = []
    if control_time is None or not np.isfinite(control_time):
        return [f"the control step {control_time} is not a finite time"]
    if abs(control_time - DIVERGENCE_STEP) < 1e-6:
        problems.append(f"the control step {control_time} is the divergence step")
    if not any(abs(control_time - t) < 1e-6 for t in times):
        problems.append(f"the control step {control_time} is not a timestep of the "
                        f"exported window ({times[0]} to {times[-1]})")
    return problems


SHAPE_CONTROL_OFFSET_DEG = 4.0    # two coarse cells west, recorded in the artifact


def intervene(case, v1_axis_points, reference, control_time, baseline_final,
              baseline_west, shape_offset=SHAPE_CONTROL_OFFSET_DEG):
    """THE INTERVENTION, because a difference observed at a stage is not a mechanism
    until an intervention at that stage changes the output and one elsewhere does not.

    Version 1's own dumped axis vertices at the divergence step, read from the reference
    log and never from the port's axes, are added to the port's axis set at that step
    alone and the whole window is replayed. The CONTROL adds the same vertices at another
    timestep instead, so the effect has to be specific to the step this case names rather
    than to the act of adding an axis. Both are reported beside the untouched replay."""
    axes = [{"lat": list(r["lat"]), "lon": list(r["lon"])} for r in v1_axis_points]
    runs = {"injected_axes": axes,
            "control_step_problems": control_step_problems(case, control_time),
            "baseline": {"injected_at": None, "injections_applied": 0,
                         "applied_at": [],
                         "reference_tracks": reproduction(baseline_final, reference),
                         "observations_at_divergence_in_box":
                             observations_at(baseline_final, DIVERGENCE_STEP),
                         "finished_tracks_in_the_western_box": baseline_west}}
    # THE SHAPE CONTROL, at the divergence step with the same vertex count moved off the
    # crossing. The time control shows the effect belongs to this timestep; it cannot show
    # the effect belongs to THESE vertices, and a reader coming in cold asked for that.
    moved = [{"lat": list(a["lat"]), "lon": [lo - shape_offset for lo in a["lon"]]}
             for a in axes]
    for label, t, what in (("intervention", DIVERGENCE_STEP, axes),
                           ("control", control_time, axes),
                           ("shape_control", DIVERGENCE_STEP, moved)):
        _log, _div, west, _hist, final, applied = replay_port(
            case, inject={"time": t, "axes": what})
        runs[label] = {"injected_at": t, "injections_applied": len(applied),
                       "applied_at": applied,
                       "reference_tracks": reproduction(final, reference),
                       "observations_at_divergence_in_box":
                           observations_at(final, DIVERGENCE_STEP),
                       # the TRACK-level effect beside the observation-level one: the
                       # Sahara case's injection restores the missing observation without
                       # reproducing the reference track, and only this distinguishes the
                       # two outcomes
                       "finished_tracks_in_the_western_box": west}
    runs["shape_control"]["longitude_offset_deg"] = shape_offset
    runs["shape_control"]["axes"] = moved
    return runs


def read_v1_dumps(octave_log):
    """Version 1's dumped axes and candidates inside the box across the window."""
    out = []
    with open(octave_log) as fh:
        for line in fh:
            parts = line.split()
            if not parts or parts[0] not in ("AXIS", "AXISPTS", "COARSE", "FINE"):
                continue
            t = float(parts[1])
            if not WINDOW[0] <= t <= WINDOW[1]:
                continue
            if parts[0] == "AXISPTS":
                # every vertex of one axis; kept when any vertex is in the box
                vals = [float(x) for x in parts[3:]]
                lats, lons = vals[0::2], vals[1::2]
                if any(in_axis_box(la, lo) for la, lo in zip(lats, lons)):
                    out.append({"kind": "AXISPTS", "time": t, "n_points": int(parts[2]),
                                "lat": lats, "lon": lons})
                continue
            if parts[0] == "AXIS":
                n, lat, lon = int(parts[2]), float(parts[3]), float(parts[4])
                # the fifth and sixth fields are RANGES (maximum minus minimum), which the
                # instrumented copy prints; a first version named them standard deviations
                rec = {"kind": "AXIS", "time": t, "n_points": n, "lat_mean": lat, "lon_mean": lon,
                       "lat_range": float(parts[5]), "lon_range": float(parts[6])}
            else:
                lat, lon = float(parts[2]), float(parts[3])
                rec = {"kind": parts[0], "time": t, "lat_mean": lat, "lon_mean": lon}
            if in_box(lat, lon):
                out.append(rec)
    return out


def intervention_statements(runs):
    """What the intervention and its control did, derived from the two replays' own
    counts. A run that was not performed says nothing: an absent intervention is never
    reported as one that changed nothing."""
    statements, missing = [], []
    if runs is None:
        return statements, missing
    counts = {}
    for label in ("baseline", "intervention", "control", "shape_control"):
        refs = runs.get(label, {}).get("reference_tracks")
        if not refs:
            missing.append(f"the {label} replay's reference-track comparison")
            continue
        counts[label] = (sum(1 for r in refs if r["reproduced_exactly"]), len(refs))
    # AN INJECTION THAT DID NOT HAPPEN IS NOT A NEGATIVE RESULT. A review passed a control
    # time outside the window, nothing was injected, and the first version reported the
    # untouched replay as an injection that reproduced nothing.
    missing.extend(RM.experiment_problems(
        runs, DIVERGENCE_STEP, runs.get("control", {}).get("injected_at"),
        runs.get("control_step_problems"), runs.get("injected_axes"),
        (runs.get("shape_control") or {}).get("axes")))
    # AN ABSENT MEASUREMENT IS NOT A ZERO, and this statement reports the counts, so it
    # refuses when one is missing. A review deleted all three records and watched the
    # first version report "untouched 0 ... intervention 0 ... control 0", which is the
    # one comparison the Sahara case's claim rests on.
    for label in ("baseline", "intervention", "control", "shape_control"):
        if runs.get(label, {}).get("finished_tracks_in_the_western_box") is None:
            missing.append(f"the {label} replay's finished tracks in the western box")
    if missing:
        return statements, missing
    n_axes = len(runs["injected_axes"])
    statements.append(
        f"injecting version 1's {n_axes} dumped axis (axes) into the port's axis set at "
        f"{DIVERGENCE_STEP} alone reproduces {counts['intervention'][0]} of "
        f"{counts['intervention'][1]} reference tracks exactly, against "
        f"{counts['baseline'][0]} of {counts['baseline'][1]} with the port untouched")
    statements.append(
        f"the same axes injected at {runs['control']['injected_at']} instead reproduce "
        f"{counts['control'][0]} of {counts['control'][1]}, and the same vertex count moved "
        f"{runs['shape_control']['longitude_offset_deg']} degrees west of the crossing at "
        f"{DIVERGENCE_STEP} reproduces {counts['shape_control'][0]} of "
        f"{counts['shape_control'][1]}")
    counts_west = {label: len(runs[label].get("finished_tracks_in_the_western_box") or [])
                   for label in ("baseline", "intervention", "control", "shape_control")}
    statements.append(
        f"finished port tracks in the western box: untouched {counts_west['baseline']}, "
        f"with the injection at {DIVERGENCE_STEP} {counts_west['intervention']}, with the "
        f"time control {counts_west['control']}, with the shape control "
        f"{counts_west['shape_control']}")
    statements.append(
        f"positions the finished port tracks hold at {DIVERGENCE_STEP} in the box: "
        f"untouched {runs['baseline']['observations_at_divergence_in_box']}, "
        f"with the injection at {DIVERGENCE_STEP} "
        f"{runs['intervention']['observations_at_divergence_in_box']}, "
        f"with the control injection "
        f"{runs['control']['observations_at_divergence_in_box']}")
    return statements, missing


def field_statements(divergence):
    """What the two sides' masked fields were measured to be at the divergence step, or
    that the measurement is absent. An absent measurement is stated, never skipped."""
    cmp_field = (divergence or {}).get("field_comparison") or {"available": False}
    if not cmp_field.get("available"):
        return ["version 1's own masked field at this step was not dumped, so whether the "
                "port's axes and the port's candidates differ from version 1's by "
                "contouring or by masking is not established here"]
    if cmp_field["same_field"]:
        return [f"version 1's own masked field at this step, dumped cell by cell, is "
                f"unmasked in exactly the same {cmp_field['cells_unmasked_on_both_sides']} "
                f"cells of the coarse grid as the port's and agrees with it to a relative "
                f"difference of at most {cmp_field['worst_relative_difference']:.1e}, so "
                f"the two sides hold one field and differ in what they draw through it"]
    return [f"version 1's masked field and the port's differ at this step: "
            f"{cmp_field['cells_unmasked_only_in_the_port']} cells unmasked only in the "
            f"port, {cmp_field['cells_unmasked_only_in_version_1']} only in version 1, "
            f"{cmp_field.get('reference_cells_not_finite', 0)} reference cells not usable, "
            f"worst relative difference {cmp_field['worst_relative_difference']:.1e} over "
            f"{cmp_field['cells_unmasked_on_both_sides']} shared cells, so what they draw "
            f"is not established as contouring alone"]


def derive_conclusion(log, v1, v1_at, port_at, divergence, west, western, intervention=None):
    """The conclusion as a list of statements, each derived from a recorded observation,
    and the list of observations that were needed and absent. Nothing here is typed
    from memory of the case: a statement appears only when its observation does."""
    missing, statements = [], []
    for t in AGREEING_STEPS:
        v1_c = [r for r in v1 if r["kind"] == "COARSE" and abs(r["time"] - t) < 1e-6]
        p_c = [r for r in log if abs(r["time"] - t) < 1e-6]
        if not v1_c:
            missing.append(f"version 1 coarse candidate in the box at {t}")
        if not p_c:
            missing.append(f"port candidate in the box at {t}")
    if len(missing) == 0:
        statements.append(f"both sides detect a candidate in the box at each of "
                          f"{list(AGREEING_STEPS)}")
    v1_axes = [r for r in v1_at if r["kind"] == "AXIS"]
    v1_coarse = [r for r in v1_at if r["kind"] == "COARSE"]
    if not v1_axes or not v1_coarse:
        missing.append(f"version 1 axes and coarse candidate in the box at {DIVERGENCE_STEP}")
    if divergence is None:
        missing.append(f"the port's masked field at {DIVERGENCE_STEP}")
    if not missing:
        v1_fine = [r for r in v1_at if r["kind"] == "FINE"]
        if port_at and not divergence["port_axes_in_box"]:
            statements.append(f"the port also has a candidate in the box at "
                              f"{DIVERGENCE_STEP} and draws no axis, so this step is "
                              f"not the divergence")
        elif port_at:
            # BOTH SIDES HOLD A CANDIDATE: the divergence is where each side's fine
            # candidate landed, which depends on which vertices its line carried
            for pc in port_at:
                for vf in v1_fine:
                    d = float(np.hypot(vf["lat_mean"] - pc["lat_mean"],
                                       (vf["lon_mean"] - pc["lon_mean"])
                                       * np.cos(np.radians(0.5 * (vf["lat_mean"]
                                                                  + pc["lat_mean"])))))
                    statements.append(
                        f"at {DIVERGENCE_STEP} both sides hold a candidate in the box: "
                        f"version 1's fine candidate at ({vf['lat_mean']}, {vf['lon_mean']}) "
                        f"and the port's at ({pc['lat_mean']}, {pc['lon_mean']}), "
                        f"{d:.1f} degrees apart")
            v1_pts = [r for r in v1_at if r["kind"] == "AXISPTS"]
            if v1_pts and divergence["port_axes_in_box"]:
                cmp = compare_vertices(v1_pts, divergence["port_axes_in_box"])
                divergence["vertex_comparison"] = cmp
                statements.append(
                    f"both sides draw axes in the axis box: version 1 {cmp['v1_lines']} "
                    f"line(s) over {cmp['v1_vertices']} vertices, the port "
                    f"{cmp['port_lines']} line(s) over {cmp['port_vertices']} vertices; "
                    f"{cmp['shared_vertices']} distinct vertices are shared, "
                    f"{cmp['v1_only_vertices']} are version 1's only and "
                    f"{cmp['port_only_vertices']} the port's only")
                if cmp["v1_only_vertices"] == 0 and cmp["port_only_vertices"] == 0 \
                        and not cmp["same_partition"]:
                    # DESCRIPTIVE ONLY. A first version said the candidates therefore
                    # differed "by partition alone"; a review showed the same vertices
                    # partitioned differently with identical candidates, and the South
                    # Atlantic case's real cause lay in a later stage. What a partition
                    # difference does to a candidate is a question for an intervention.
                    statements.append("the two sides share every distinct vertex in the "
                                      "axis box and join them into different lines; "
                                      "whether that changes a candidate is not "
                                      "established by this comparison")
        elif divergence["port_axes_in_box"]:
            # BOTH SIDES DRAW AXES HERE, so the question is which vertices each joined
            # into a line. The vertex sets are compared as sets rounded to four decimals,
            # in the box only, and the statement says what they share and how each side
            # partitioned them, since a candidate is the mean of a line and a different
            # partition moves it.
            v1_pts = [r for r in v1_at if r["kind"] == "AXISPTS"]
            if not v1_pts:
                missing.append(f"version 1 axis vertices (AXISPTS) at {DIVERGENCE_STEP}")
            else:
                cmp = compare_vertices(v1_pts, divergence["port_axes_in_box"])
                divergence["vertex_comparison"] = cmp
                statements.append(
                    f"at {DIVERGENCE_STEP} both sides draw axes in the box: version 1 "
                    f"{cmp['v1_lines']} line(s) over {cmp['v1_vertices']} vertices, the port "
                    f"{cmp['port_lines']} line(s) over {cmp['port_vertices']} vertices; "
                    f"{cmp['shared_vertices']} distinct vertices are shared, "
                    f"{cmp['v1_only_vertices']} are version 1's only and "
                    f"{cmp['port_only_vertices']} the port's only; the port has no "
                    f"candidate in the box")
                if cmp["v1_only_vertices"] == 0 and cmp["port_only_vertices"] == 0 \
                        and not cmp["same_partition"]:
                    # DESCRIPTIVE ONLY. A first version said the candidates therefore
                    # differed "by partition alone"; a review showed the same vertices
                    # partitioned differently with identical candidates, and the South
                    # Atlantic case's real cause lay in a later stage. What a partition
                    # difference does to a candidate is a question for an intervention.
                    statements.append("the two sides share every distinct vertex in the "
                                      "axis box and join them into different lines; "
                                      "whether that changes a candidate is not "
                                      "established by this comparison")
        else:
            statements.append(
                f"at {DIVERGENCE_STEP} version 1 dumps {len(v1_axes)} axes and "
                f"{len(v1_coarse)} coarse candidate in the box and the port has no "
                f"candidate and no axis there; the port's masked field holds "
                f"{divergence['finite_cells']} finite cells in the box")
            v1_pts = [r for r in v1_at if r["kind"] == "AXISPTS"]
            if v1_pts:
                crossings = divergence.get("zero_crossings") or zero_crossings(divergence)
                on = vertices_on_crossings(v1_pts, crossings)
                divergence["zero_crossings"] = crossings
                divergence["v1_vertices_on_port_crossings"] = on
                # READ FROM THE RECORD THE REPLAY WROTE, which computed them over the whole
                # grid. A conclusion that recomputed them here would only have the crop.
                hoods = divergence.get("crossing_neighborhoods") or []
                closed = [h for h in hoods if h["closed_off"] is True]
                undetermined = [h for h in hoods if h["closed_off"] is None]
                if hoods:
                    statements.append(
                        f"of the {len(hoods)} zero crossings on the port's field in the "
                        f"printed box, {len(closed)} lie on an edge whose quads on both "
                        f"sides hold two or more masked corners, which is what leaves the "
                        f"tracer no triangle to trace through (a quad with one masked "
                        f"corner still offers one), and {len(undetermined)} are "
                        f"undetermined because a neighbouring quad was not observed; the "
                        f"count is taken over the WHOLE coarse grid, not the printed box")
                worst = max((v["distance_deg"] for v in on if v["distance_deg"] is not None),
                            default=None)
                if on and all(v["on_a_crossing"] for v in on):
                    cell = on[0]["nearest_crossing"]["between"]
                    statements.append(
                        f"version 1's {len(on)} dumped vertices sit on a zero crossing of "
                        f"the port's own masked field, within {worst:.1e} degrees in each "
                        f"coordinate (the first between the cells at {cell[0]} and "
                        f"{cell[1]}), which locates where version 1 drew and does not by "
                        f"itself say the two fields are the same")
                else:
                    statements.append(
                        f"version 1's dumped vertices do not all sit on zero crossings of "
                        f"the port's own masked field (worst distance "
                        f"{'none' if worst is None else format(worst, '.2f')} degrees)")
    # WHOSE FIELD IS IT, reported WHATEVER the two sides drew at this step. A review
    # proved that dumped vertices cannot separate a contouring difference from a masking
    # one, by building a second masked field on which Octave returns the same vertices
    # while the port's tracer draws a line. It also found this statement living inside
    # the branch where the port draws no axis, so the third case, where both sides draw,
    # recorded nothing about the fields at all.
    statements += field_statements(divergence)
    if west:
        statements.append(f"{len(west)} finished port track(s) sit in the western box")
    else:
        statements.append("no finished port track sits in the western box")
    pruned = [h for h in western if h["pruned_at"] is not None]
    if not western:
        missing.append("port tracks entering the western box during the feature's life")
    else:
        statements.append("port tracks entering the western box: "
                          + "; ".join(f"{len(h['observations'])} observations from "
                                      f"{h['observations'][0][0]} to {h['observations'][-1][0]}"
                                      + (f", pruned at {h['pruned_at']}" if h["pruned_at"]
                                         is not None else ", never pruned")
                                      for h in western))
        if len(pruned) == len(western) and not west:
            statements.append("every port track that entered the western box was removed "
                              "by the in-loop prune, so the western history survives on "
                              "neither side as a finished port track")
    more, absent = intervention_statements(intervention)
    return statements + more, missing + absent


def compare_vertices(v1_axes, port_axes, decimals=4):
    """Distinct vertices in the box on each side, their overlap, and the line counts."""
    def distinct(axes):
        return {(round(la, decimals), round(lo, decimals))
                for a in axes for la, lo in zip(a["lat"], a["lon"]) if in_axis_box(la, lo)}
    v, p = distinct(v1_axes), distinct(port_axes)
    # THE PARTITION ITSELF, as a set of lines each given by its vertex set in the box,
    # because two sides can hold the same number of lines over the same vertices and
    # still have joined them differently (the third case: two lines each side, and
    # the southern segment joined to a different branch on each)
    def partition(axes):
        return {frozenset((round(la, decimals), round(lo, decimals))
                          for la, lo in zip(a["lat"], a["lon"]) if in_axis_box(la, lo))
                for a in axes}
    return {"v1_lines": len(v1_axes), "port_lines": len(port_axes),
            "v1_vertices": len(v), "port_vertices": len(p),
            "shared_vertices": len(v & p), "v1_only_vertices": len(v - p),
            "port_only_vertices": len(p - v),
            "same_partition": partition(v1_axes) == partition(port_axes),
            "v1_line_means": [[float(np.mean(a["lat"])), float(np.mean(a["lon"])), a["n_points"]]
                              for a in v1_axes],
            "port_line_means": [[a["lat_mean"], a["lon_mean"], a["n"]] for a in port_axes]}


def lone_column_field():
    """The Sahara divergence step's shape in isolation: a five by five masked field
    whose only finite cells are one column with signs plus, minus, plus; and the same
    with a second finite column beside it."""
    lat = np.array([22.0, 20.0, 18.0, 16.0, 14.0])
    lon = np.array([-13.0, -11.0, -9.0, -7.0, -5.0])
    f = np.full((5, 5), np.nan)
    f[1, 2], f[2, 2], f[3, 2] = 1.3e-11, -3.5e-11, 7.4e-11
    g = f.copy()
    g[1, 3], g[2, 3], g[3, 3] = 1.0e-11, -2.0e-11, 5.0e-11
    return lat, lon, f, g


def lone_row_field():
    """The eastern Pacific divergence step's shape in isolation: one finite ROW with
    signs plus, plus, minus, the sign change on an edge whose neighboring cells above
    and below are masked."""
    lat = np.array([-4.0, -6.0, -8.0, -10.0, -12.0])
    lon = np.array([-141.0, -139.0, -137.0, -135.0, -133.0])
    f = np.full((5, 5), np.nan)
    f[1, 1], f[1, 2], f[1, 3] = 2.0e-11, 1.0e-11, -3.0e-11
    return lat, lon, f


def interior_change_row_field():
    """The South American divergence step's shape in isolation: one finite row of five
    cells whose sign changes in its INTERIOR rather than at its end, masked above, with
    two cells of one sign below its eastern end. The values are the ones the port's own
    field holds at that step, so the crossing the isolation draws is the crossing the
    real step has."""
    lat = np.array([-28.0, -30.0, -32.0, -34.0, -36.0])
    lon = np.array([-65.0, -63.0, -61.0, -59.0, -57.0, -55.0])
    f = np.full((5, 6), np.nan)
    f[2, 0], f[2, 1], f[2, 2] = 4.071516838295352e-10, 3.615261478456151e-10, \
        1.202971274689068e-10
    f[2, 3], f[2, 4] = -2.3182610628854524e-10, -1.3041326376378817e-10
    f[3, 4], f[3, 5] = -8.94e-10, -1.46e-09
    return lat, lon, f


def port_synthetic():
    lat, lon, f, g = lone_column_field()
    LG, NG = np.meshgrid(lat, lon, indexing="ij")
    rlat, rlon, r = lone_row_field()
    RLG, RNG = np.meshgrid(rlat, rlon, indexing="ij")
    ilat, ilon, i = interior_change_row_field()
    ILG, ING = np.meshgrid(ilat, ilon, indexing="ij")
    return {"lone_column_axes": [{"n": int(len(a[0]))} for a in D.trough_axes(LG, NG, f)],
            "two_column_axes": [{"n": int(len(a[0]))} for a in D.trough_axes(LG, NG, g)],
            "lone_row_axes": [{"n": int(len(a[0]))} for a in D.trough_axes(RLG, RNG, r)],
            "interior_change_row_axes": [{"n": int(len(a[0]))}
                                         for a in D.trough_axes(ILG, ING, i)]}


def octave_synthetic():
    exe = shutil.which("octave-cli")
    script = os.path.join(HERE, "octave", "lone_column_check.m")
    if exe is None:
        return {"ran": False, "reason": "octave-cli not on PATH"}
    try:
        res = subprocess.run([exe, "--quiet", script], capture_output=True, text=True,
                             timeout=300, check=True, cwd=os.path.join(HERE, ".."))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        return {"ran": False, "reason": repr(e)[:300]}
    return {"ran": True, "output": res.stdout.strip().splitlines(),
            "script_sha256": _sha256(script)}


def main(argv=None):
    global V1_TRACK, BOX, WEST, FEATURE_LIFE, WINDOW, AGREEING_STEPS, DIVERGENCE_STEP
    global FIELD_ROWS, FIELD_COLS, AXIS_BOX
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--oracle-dir", default=os.environ.get("AEW_ORACLE_DIR"))
    ap.add_argument("--octave-log", default=None,
                    help="the instrumented run's log (default <oracle-dir>/octave3.log)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--case", default="sahara",
                    help="a name recorded in the artifact for the case being walked")
    ap.add_argument("--v1-track", type=int, default=V1_TRACK)
    ap.add_argument("--box", type=float, nargs=4, metavar=("LAT0", "LAT1", "LON0", "LON1"),
                    default=None, help="where candidates are logged")
    ap.add_argument("--west", type=float, nargs=4, metavar=("LAT0", "LAT1", "LON0", "LON1"),
                    default=None, help="the history whose port fragments are recorded")
    ap.add_argument("--feature-life", type=float, nargs=2, default=None)
    ap.add_argument("--window", type=float, nargs=2, default=None)
    ap.add_argument("--agreeing", type=float, nargs="+", default=None)
    ap.add_argument("--divergence", type=float, default=None)
    ap.add_argument("--axis-box", type=float, nargs=4, metavar=("LAT0", "LAT1", "LON0", "LON1"),
                    default=None, help="where axis vertices are compared (default the box)")
    ap.add_argument("--explains-unmatched", type=int, nargs="*", default=[],
                    help="the version 1 unmatched tracks (no eligible counterpart) this "
                         "case accounts for; the membership script checks each claim")
    ap.add_argument("--explains-pairs", type=int, nargs="*", default=[],
                    help="the version-1-extra pairs (by version 1 index) this case "
                         "accounts for")
    ap.add_argument("--field-rows", type=float, nargs=2, default=None)
    ap.add_argument("--field-cols", type=float, nargs=2, default=None)
    ap.add_argument("--reference-tracks", type=int, nargs="*", default=[],
                    help="version 1 track indices this case says the port would "
                         "reproduce with the missing axis; giving them runs the "
                         "intervention and its control, and giving none runs neither")
    ap.add_argument("--shape-control-offset", type=float, default=SHAPE_CONTROL_OFFSET_DEG,
                    help="how far west to move the injected vertices for the shape "
                         "control, which runs at the divergence step")
    ap.add_argument("--control-step", type=float, default=None,
                    help="the timestep the control injects the same axes at (default the "
                         "last agreeing step)")
    args = ap.parse_args(argv)
    if not args.oracle_dir:
        ap.error("set AEW_ORACLE_DIR or pass --oracle-dir")
    V1_TRACK = args.v1_track
    if args.box:
        BOX = {"lat": (args.box[0], args.box[1]), "lon": (args.box[2], args.box[3])}
    if args.west:
        WEST = {"lat": (args.west[0], args.west[1]), "lon": (args.west[2], args.west[3])}
    if args.feature_life:
        FEATURE_LIFE = tuple(args.feature_life)
    if args.window:
        WINDOW = tuple(args.window)
    if args.agreeing:
        AGREEING_STEPS = tuple(args.agreeing)
    if args.divergence is not None:
        DIVERGENCE_STEP = args.divergence
    if args.axis_box:
        AXIS_BOX = {"lat": (args.axis_box[0], args.axis_box[1]),
                    "lon": (args.axis_box[2], args.axis_box[3])}
    if args.field_rows:
        FIELD_ROWS = tuple(args.field_rows)
    if args.field_cols:
        FIELD_COLS = tuple(args.field_cols)
    case_path = os.path.join(args.oracle_dir, "tracker_case.mat")
    octave_log = args.octave_log or os.path.join(args.oracle_dir, "octave3.log")
    for p in (case_path, octave_log):
        if not os.path.exists(p):
            print(f"REFUSED: {p} is absent", flush=True)
            return 2
    case = loadmat(case_path)
    case_id = str(np.asarray(case["case_id"]).ravel()[0]).strip()
    oracle_path = os.path.join(args.oracle_dir, "tracker_octave_instrumented.mat")
    port_path = os.path.join(args.oracle_dir, "tracker_port.mat")
    for p in (oracle_path, port_path):
        if not os.path.exists(p):
            print(f"REFUSED: {p} is absent", flush=True)
            return 2
    problems, provenance = validate_evidence(args.oracle_dir, case_id, octave_log)
    if problems:
        print("REFUSED: the reference evidence is not evidence for this case: "
              + "; ".join(problems), flush=True)
        return 2
    _t, oracle_rec, _c, _n = read_producer_text(oracle_path)
    _t, port_rec, _c, _n = read_producer_text(port_path)
    _r, dumped_times, returned, run_digest = parse_log(octave_log)
    reference_field = reference_field_at(octave_log, DIVERGENCE_STEP)
    if reference_field["problems"]:
        print("REFUSED: the reference log's FIELD records are not usable evidence: "
              + "; ".join(sorted(set(reference_field["problems"]))), flush=True)
        return 2
    log, divergence, west, western, final, _applied = replay_port(
        case, reference_field=reference_field)
    v1 = read_v1_dumps(octave_log)
    v1_at = [r for r in v1 if abs(r["time"] - DIVERGENCE_STEP) < 1e-6]
    port_at = [r for r in log if abs(r["time"] - DIVERGENCE_STEP) < 1e-6]
    intervention, control_used = None, None
    not_attempted = "no reference tracks were declared"
    if args.reference_tracks:
        v1_pts = [r for r in v1_at if r["kind"] == "AXISPTS"]
        reference, absent = [], []
        if not v1_pts:
            absent.append(f"version 1 axis vertices (AXISPTS) at {DIVERGENCE_STEP}, which "
                          f"the intervention injects")
        tracks, why = finished_tracks(oracle_path)
        if why:
            absent.append(why)
        else:
            for i in args.reference_tracks:
                if not 0 <= i < len(tracks):
                    absent.append(f"reference track {i}, which the oracle output does not hold")
                else:
                    reference.append((i, tracks[i]))
        if absent:
            print("REFUSED: the intervention this case declares cannot run: "
                  + "; ".join(absent), flush=True)
            return 2
        control_used = (args.control_step if args.control_step is not None
                        else AGREEING_STEPS[-1])
        intervention = intervene(case, v1_pts, reference, control_used, final,
                                 west, args.shape_control_offset)
    conclusion, missing = derive_conclusion(log, v1, v1_at, port_at, divergence, west,
                                            western, intervention)
    if missing:
        print("REFUSED: the observations this case's conclusion needs are absent: "
              + "; ".join(missing), flush=True)
        return 2
    synthetic = {"port": port_synthetic(), "octave": octave_synthetic()}
    result = {"case_id": case_id, "case_name": args.case,
              "explains": {"unmatched_v1_tracks": sorted(args.explains_unmatched),
                           "v1_extra_pairs": sorted(args.explains_pairs)},
              "parameters": {"v1_track": V1_TRACK, "box": BOX, "axis_box": AXIS_BOX, "west": WEST,
                             "feature_life": list(FEATURE_LIFE), "window": list(WINDOW),
                             "agreeing_steps": list(AGREEING_STEPS),
                             "divergence_step": DIVERGENCE_STEP,
                             # the printed sub-box of the masked field and the
                             # intervention's arguments, which a reader otherwise has to
                             # infer from the recorded rows and columns
                             "field_rows": list(FIELD_ROWS), "field_cols": list(FIELD_COLS),
                             "reference_tracks": sorted(args.reference_tracks),
                             "control_step": control_used},
              "evidence_binding": {"oracle_provenance": provenance["oracle"]["status"],
                                   "oracle_faithful": provenance["oracle"]["faithful"],
                                   "port_provenance": provenance["port"]["status"],
                                   "log_run_digest": run_digest,
                                   "oracle_producer_dump_times": len(oracle_rec["dump_times"]),
                                   "log_dumped_timesteps": len(dumped_times),
                                   "log_returned_tracks": returned,
                                   "port_producer_git_head": port_rec.get("git_head")},
              "western_port_tracks": western,
              "conclusion": conclusion,
              # AN INTERVENTION THAT DID NOT RUN SAYS SO, and says nothing else. A case
              # that declares no reference tracks has not shown a mechanism, and its
              # artifact must not read as one that intervened and found no effect.
              "intervention": intervention if intervention is not None
              else {"attempted": False, "reason": not_attempted},

              "port_candidates_in_box": log, "port_finished_tracks_in_western_box": west,
              "v1_dumps_in_box": v1,
              "at_divergence": {"v1": v1_at, "port": port_at, "field": divergence},
              "synthetic": synthetic,
              "what_this_does_not_establish": "whether MATLAB's contouring matches Octave's "
                                              "on a masked-bounded column, and any count of "
                                              "how many other unmatched tracks share this "
                                              "mechanism"}
    for st in conclusion:
        print("  -", st)
    print(f"case {result['case_id']}: port candidates in box {len(log)}, v1 dump records in "
          f"box {len(v1)}; at {DIVERGENCE_STEP}: v1 {len(v1_at)} records, port {len(port_at)} "
          f"candidates, finite cells in the box {divergence['finite_cells'] if divergence else None}, "
          f"port axes in box {divergence['port_axes_in_box'] if divergence else None}")
    print(f"port finished tracks ever in the western box: {west}")
    print(f"synthetic lone column: port axes {synthetic['port']['lone_column_axes']}, "
          f"two columns {synthetic['port']['two_column_axes']}; octave "
          f"{synthetic['octave']}")
    if args.out:
        repo = os.path.join(HERE, "..")
        result.update({"generated_by": "scripts/trace_sahara_case.py",
                       "input_sha256": {"tracker_case.mat": _sha256(case_path),
                                        "tracker_octave_instrumented.mat": _sha256(oracle_path),
                                        "tracker_port.mat": _sha256(port_path),
                                        os.path.basename(octave_log): _sha256(octave_log)},
                       "source_sha256": {"scripts/trace_sahara_case.py":
                                         _sha256(os.path.abspath(__file__)),
                                         **{rel: _sha256(os.path.join(repo, rel))
                                            for rel in REPLAY_SOURCES}}})
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1, sort_keys=True, default=str)
        print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
