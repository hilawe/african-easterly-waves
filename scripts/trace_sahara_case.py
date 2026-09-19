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

# The port modules the replay EXECUTES, fingerprinted into the artifact, because a later
# replay under different detection or merge code would otherwise carry the same script
# fingerprint (a review's finding).
REPLAY_SOURCES = ("src/aew/v1port/detection.py", "src/aew/v1port/contours.py",
                  "src/aew/v1port/association.py", "src/aew/v1port/pipeline.py",
                  "src/aew/v1port/climatology.py", "scripts/compare_tracker_oracle.py")

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
    if oracle_rec and "dump_times" in oracle_rec:
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


def replay_port(case):
    """Port candidates in the box per step with their fate, the masked advection field
    plus axes at the divergence step, and the full history of every port track that
    ever enters the western box with the step that pruned it, from the port's own
    functions."""
    times = case["time"].ravel()
    lat_c, lon_c = case["lat_c"].ravel(), case["lon_c"].ravel()
    latgrid_c, longrid_c = np.meshgrid(lat_c, lon_c, indexing="ij")
    ct, ft = P.thresholds_for("ERA-Int", 700)
    captured = {}
    original = D.trough_axes

    def spy(latgrid, longrid, advection, level=D.TROUGH_LEVEL):
        out = original(latgrid, longrid, advection, level)
        captured["field"], captured["axes"] = advection, out
        return out
    D.trough_axes = spy
    tracks, states, log, divergence = [], [], [], None
    # Histories are keyed by a STABLE tag written into each track dictionary when it is
    # first seen, never by the object's identity: a first version keyed on id(track),
    # Python reuses an identity once a pruned track is freed, and a later track
    # inherited a fragment's record and moved its prune step from 33025.5 to 33031.0.
    histories, next_tag = {}, [0]
    try:
        for step in range(times.size):
            t = float(times[step])
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
    return log, divergence, west, western


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


def derive_conclusion(log, v1, v1_at, port_at, divergence, west, western):
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
    return statements, missing


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
    signs plus, plus, minus, the sign change on an edge whose neighbouring cells above
    and below are masked."""
    lat = np.array([-4.0, -6.0, -8.0, -10.0, -12.0])
    lon = np.array([-141.0, -139.0, -137.0, -135.0, -133.0])
    f = np.full((5, 5), np.nan)
    f[1, 1], f[1, 2], f[1, 3] = 2.0e-11, 1.0e-11, -3.0e-11
    return lat, lon, f


def port_synthetic():
    lat, lon, f, g = lone_column_field()
    LG, NG = np.meshgrid(lat, lon, indexing="ij")
    rlat, rlon, r = lone_row_field()
    RLG, RNG = np.meshgrid(rlat, rlon, indexing="ij")
    return {"lone_column_axes": [{"n": int(len(a[0]))} for a in D.trough_axes(LG, NG, f)],
            "two_column_axes": [{"n": int(len(a[0]))} for a in D.trough_axes(LG, NG, g)],
            "lone_row_axes": [{"n": int(len(a[0]))} for a in D.trough_axes(RLG, RNG, r)]}


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
    ap.add_argument("--explains-unmatched", type=int, nargs="*", default=[19],
                    help="the version 1 unmatched tracks (no eligible counterpart) this "
                         "case accounts for; the membership script checks each claim")
    ap.add_argument("--explains-pairs", type=int, nargs="*", default=[],
                    help="the version-1-extra pairs (by version 1 index) this case "
                         "accounts for")
    ap.add_argument("--field-rows", type=float, nargs=2, default=None)
    ap.add_argument("--field-cols", type=float, nargs=2, default=None)
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
    log, divergence, west, western = replay_port(case)
    v1 = read_v1_dumps(octave_log)
    v1_at = [r for r in v1 if abs(r["time"] - DIVERGENCE_STEP) < 1e-6]
    port_at = [r for r in log if abs(r["time"] - DIVERGENCE_STEP) < 1e-6]
    conclusion, missing = derive_conclusion(log, v1, v1_at, port_at, divergence, west, western)
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
                             "divergence_step": DIVERGENCE_STEP},
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
