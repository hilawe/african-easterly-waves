#!/usr/bin/env python3
"""The lineage reading of `EASTERN_EVALUATION_ADDENDUM_2006_2026-10-10.md`, section 4.

Reads the declaration, the traced 2006 replay with its merge lineage at the scored maps, and
the baseline measurement record, and at each scored map with no final candidate within the
tolerance (and separately within the alternative) follows every relevant input axis through
the two merges. It recomputes nothing and changes nothing.

TWO PROXIMITIES, kept apart. An input axis's proximity is the smallest distance from any of
its vertices to the map's read geometry. Its merge-input center's proximity is the distance
of the axis's mean position, the point the coarse merge seeds its region from. Both use the
measurement's distance (Euclidean degrees to the nearest point of the read longitude, a
value or a range, by the read latitude range). An axis is RELEVANT at a tolerance when its
vertex proximity is within it.

THE STAGE OF AN AXIS is the first point at which its position leaves the tolerance or it is
removed, in the merges' own order: the merge-input center, the coarse region's position
(the region median, with the level landed and whether the ladder ran out), the coarse
duplicate pass, the coarse extent test, the coarse refinement, then the same four for the
fine merge, and finally the final candidate. Every position is recorded, so a later return
inside the tolerance stays visible. A broken chain is UNRESOLVED.

THE STAGE OF A MAP is "no trough axis within tolerance before the merges" when no axis is
relevant, the shared stage when every relevant axis has the same one, and MIXED, with each
axis's stage, when they differ. No single cause is forced.

    python3 scripts/pilot_eastern_lineage_reading.py --reference <declaration.json> --trace <trace.json.gz> \\
        --trace-digest <first 12> --measurement <baseline record.json> --out <reading.json>
"""
import argparse
import gzip
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_eastern_measurement as E  # noqa: E402
import pilot_r1_measurement as M  # noqa: E402

MERGE_DISTANCE_DEG = 5.0
NONE_NEAR = "no trough axis within tolerance before the merges"


def arc_deg(lat1, lon1, lat2, lon2):
    """Great-circle distance in degrees of arc, the merge's own measure."""
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    c = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _pos(sel):
    r = (sel or {}).get("region") or {}
    return (r.get("median_lat"), r.get("median_lon"))


def follow(axis, lineage, candidates, m, tol):
    """One axis through both merges, every position with its distance, and its stage."""
    d = lambda la, lo: None if la is None or lo is None else round(E.ref_distance(la, lo, m), 3)   # noqa: E731
    coarse = {e["input"]: e for e in lineage["coarse"]}
    fine = {e["input"]: e for e in lineage["fine"]}
    vertex = min(E.ref_distance(la, lo, m) for la, lo in zip(axis["lat"], axis["lon"]))
    rec = {"axis": axis["index"], "n_vertices": axis["n_vertices"], "vertex_distance_deg": round(vertex, 3),
           "center": [axis["lat_mean"], axis["lon_mean"]], "center_distance_deg": d(axis["lat_mean"], axis["lon_mean"]), "positions": []}
    stage = None

    def leave(label):
        nonlocal stage
        if stage is None:
            stage = label

    if rec["center_distance_deg"] > tol:
        leave("the merge-input center lies beyond tolerance")
    e = coarse.get(axis["index"])
    if e is None:
        rec["stage"] = "unresolved, no coarse lineage entry for the axis"
        return rec
    if "absorbed_by" in e:
        rec["stage"] = stage or "coarse merge, absorbed into another input's region"
        return rec
    sel = e.get("select") or {}
    la, lo = _pos(sel)
    rec["positions"].append({"stage": "coarse region", "lat": la, "lon": lo, "distance_deg": d(la, lo), "level_landed": sel.get("level_landed"),
                             "last_level": sel.get("last_level"), "exhausted": sel.get("exhausted"), "cells": (sel.get("region") or {}).get("cells"),
                             "lat_extent": sel.get("selected_lat_extent"), "lon_extent": sel.get("selected_lon_extent"), "region_sha256": (sel.get("region") or {}).get("region_sha256")})
    if d(la, lo) is None or d(la, lo) > tol:
        leave("coarse merge, the selected region's position lies beyond tolerance")
    if not e.get("lone_wave"):
        if e.get("kept_after_pass_two") is False:
            near = []
            for other in lineage["coarse"]:
                if other is e or "absorbed_by" in other or not other.get("kept_after_pass_two"):
                    continue
                ola, olo = _pos(other.get("select"))
                if ola is None or la is None:
                    continue
                g = arc_deg(la, lo, ola, olo)
                if g <= MERGE_DISTANCE_DEG:
                    out_o = other.get("output")
                    fo = fine.get(out_o) if out_o is not None else None
                    near.append({"input": other["input"], "pass_one": [ola, olo], "arc_deg": round(g, 3), "output": out_o,
                                 "output_position": None if fo is None else [fo["lat_mean"], fo["lon_mean"]],
                                 "output_distance_deg": None if fo is None else d(fo["lat_mean"], fo["lon_mean"])})
            rec["dropped_near"] = near
            rec["stage"] = stage or "coarse merge, dropped in the duplicate pass"
            return rec
        if e.get("rejected_min_extent"):
            rec["stage"] = stage or "coarse merge, rejected for extent"
            return rec
    out = e.get("output")
    fe = fine.get(out) if out is not None else None
    if fe is None:
        rec["stage"] = stage or "unresolved, the coarse output has no fine-merge entry"
        return rec
    rec["positions"].append({"stage": "coarse output", "lat": fe["lat_mean"], "lon": fe["lon_mean"], "distance_deg": d(fe["lat_mean"], fe["lon_mean"])})
    if d(fe["lat_mean"], fe["lon_mean"]) > tol:
        leave("coarse merge, refinement moved the position beyond tolerance")
    if "absorbed_by" in fe:
        rec["stage"] = stage or "fine merge, absorbed into another input's region"
        return rec
    fsel = fe.get("select") or {}
    fla, flo = _pos(fsel)
    rec["positions"].append({"stage": "fine region", "lat": fla, "lon": flo, "distance_deg": d(fla, flo), "level_landed": fsel.get("level_landed"),
                             "exhausted": fsel.get("exhausted"), "cells": (fsel.get("region") or {}).get("cells")})
    if d(fla, flo) is None or d(fla, flo) > tol:
        leave("fine merge, the selected region's position lies beyond tolerance")
    if not fe.get("lone_wave"):
        if fe.get("kept_after_pass_two") is False:
            rec["stage"] = stage or "fine merge, dropped in the duplicate pass"
            return rec
        if fe.get("rejected_min_extent"):
            rec["stage"] = stage or "fine merge, rejected for extent"
            return rec
    fout = fe.get("output")
    cand = next((c for c in candidates if c["index"] == fout), None) if fout is not None else None
    if cand is None:
        rec["stage"] = stage or "unresolved, the fine output is not a logged candidate"
        return rec
    rec["positions"].append({"stage": "final candidate", "candidate": fout, "lat": cand["lat_mean"], "lon": cand["lon_mean"], "distance_deg": d(cand["lat_mean"], cand["lon_mean"])})
    if d(cand["lat_mean"], cand["lon_mean"]) > tol:
        leave("fine merge, refinement moved the position beyond tolerance")
    rec["stage"] = stage or "reaches a final candidate within tolerance"
    return rec


def read_map(entry, m, tol):
    lineage = entry.get("lineage")
    if lineage is None:
        raise SystemExit(f"REFUSED: the scored map {m['date']} carries no lineage")
    axes = lineage["axes"]
    nearest = min((min(E.ref_distance(la, lo, m) for la, lo in zip(a["lat"], a["lon"])) for a in axes), default=None)
    relevant = [a for a in axes if min(E.ref_distance(la, lo, m) for la, lo in zip(a["lat"], a["lon"])) <= tol]
    rows = [follow(a, lineage, entry["candidates"], m, tol) for a in relevant]
    stages = sorted({r["stage"] for r in rows})
    if not rows:
        label = NONE_NEAR
    elif len(stages) == 1:
        label = stages[0]
    else:
        label = "mixed"
    return {"date": m["date"], "segment": m["segment"], "west_of_40E": E.west_of_40(m), "axes_total": len(axes),
            "nearest_axis_vertex_distance_deg": None if nearest is None else round(nearest, 3), "relevant_axes": rows, "stage": label,
            "stages_present": stages}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--reference", "--trace", "--trace-digest", "--measurement", "--out"):
        ap.add_argument(name, required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    digest = M.sha256_decompressed(args.trace)
    if len(args.trace_digest) < 12 or not digest.startswith(args.trace_digest):
        raise SystemExit("REFUSED: the trace's digest does not begin with the one the replay logged")
    meas = json.load(open(args.measurement))
    if meas["inputs"]["trace"]["sha256_decompressed"] != digest or meas["inputs"]["reference"]["sha256"] != M.sha256_of(args.reference):
        raise SystemExit("REFUSED: the measurement record was not made from this trace and declaration")
    decl = json.load(open(args.reference))
    with gzip.open(args.trace, "rb") as fh:
        trace = json.loads(fh.read())
    if trace["trace"].get("families_sha256") != M.sha256_of(args.reference):
        raise SystemExit("REFUSED: the trace was not run with this declaration")
    entries = {M.date_key(e["date"]): e for e in trace["trace"]["steps"]}
    maps = {(m["date"], m["segment"]): m for m in decl["reference"]["maps"]}
    readings = {}
    for name in ("tolerance", "alternative"):
        t = float(meas["results"][name]["degrees"])
        rows = []
        for r in meas["results"][name]["maps"]:
            if not r["scored"] or r["detection"]["within"]:
                continue
            rows.append(read_map(entries[r["date"]], maps[(r["date"], r["segment"])], t))
        counts = {}
        for row in rows:
            counts[row["stage"]] = counts.get(row["stage"], 0) + 1
        readings[name] = {"degrees": t, "uncovered_scored_maps": rows, "stage_counts": counts}
    out = {"generated_by": "scripts/pilot_eastern_lineage_reading.py", "script_sha256": M.sha256_of(os.path.abspath(__file__)),
           "inputs": {"reference": {"path": args.reference, "sha256": M.sha256_of(args.reference)}, "trace": {"path": args.trace, "sha256_decompressed": digest},
                      "measurement": {"path": args.measurement, "sha256": M.sha256_of(args.measurement)}},
           "readings": readings,
           "limits": "the stage locates where correspondence with the read geometry is lost inside the frozen detector; it says nothing about physical identity, and a map with no relevant axis is not explained further here"}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for name, r in readings.items():
        print(f"[{name} {r['degrees']} deg] {len(r['uncovered_scored_maps'])} uncovered scored maps; stages {r['stage_counts']}")
        for row in r["uncovered_scored_maps"]:
            print(f"  {row['date']} {row['segment']}: {row['stage']} (relevant axes {len(row['relevant_axes'])}, nearest axis vertex {row['nearest_axis_vertex_distance_deg']} deg)")
    print(f"Record {args.out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
