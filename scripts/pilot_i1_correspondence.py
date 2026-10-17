#!/usr/bin/env python3
"""The correspondence table of the frozen candidate against the I1 entry
(`EASTERN_I1_ENTRY_2026-10-15.md`, section 5; `EASTERN_DIRECTION_RECOMMENDATION_8_2026-10-15.md`,
section 8), from a traced replay record and the entry listing record. Per logged step in
the entry window it lists every candidate in the family box with its representative, its
extent, the nearest integer row (both rows at a tie, the larger distance taken), the row's
counterpart under the declared precedence (the reversal where available, else the
assigned vorticity maximum), the longitude distance and its class (on within 3 degrees,
near within 5, off beyond), whether the representative is on another structure (within 3
degrees of a W or E wind minimum on its row while off I1), the row's I1 wind minimum for
description, and the association events (seeds, claims, matching decisions with the pass,
the implied speed and the chosen flag, prunings). The association follows the declared
rule, the history with the longest overlap of the supported steps, held fixed, and the
verdict follows the declared convention. Distances are reported on every row; the verdict
uses the entry's tested rows and supported steps only. Stored finished positions are
listed beside the per-step claimed positions and never substituted for them.

    python3 pilot_i1_correspondence.py TRACE_GZ ENTRY_LISTING_JSON --family AUG_I1 --out OUT_JSON --table OUT_MD
"""
import argparse
import datetime as dt
import gzip
import json

import numpy as np

ON, NEAR = 3.0, 5.0
TESTED_ROWS = (24, 25, 26, 27, 28)
SUPPORTED = ("1990-08-10T18", "1990-08-13T12")
EPOCH = dt.datetime(1900, 1, 1)


def stamp_of(step_rec):
    if "time" in step_rec:
        return (EPOCH + dt.timedelta(days=float(step_rec["time"]))).strftime("%Y-%m-%dT%H")
    return dt.datetime.strptime(step_rec["date"], "%Y-%m-%d %HZ").strftime("%Y-%m-%dT%H")


def rows_of(lat):
    r = round(lat)
    if abs(lat - np.floor(lat) - 0.5) < 1e-9:
        return [int(np.floor(lat)), int(np.ceil(lat))]
    return [int(r)]


def counterpart(row_rec):
    if row_rec is None or row_rec.get("status") != "assigned":
        return None, "row not assigned to I1"
    rv = row_rec.get("reversal") or {}
    if rv.get("lon") is not None:
        return float(rv["lon"]), "reversal"
    z = row_rec.get("zs") or row_rec.get("za")
    if z:
        return float(z["lon"]), "vorticity maximum (reversal unavailable)"
    return None, "no counterpart available on the row"


def classify(d):
    if d is None:
        return "unassessable"
    a = abs(d)
    return "on" if a <= ON else "near" if a <= NEAR else "off"


def other_structure(row_rec, lon):
    if row_rec is None or row_rec.get("status") != "assigned":
        return []
    hits = []
    for m in row_rec.get("wind_minima", []):
        if m["label"] in ("W", "E") and abs(m["lon"] - lon) <= ON:
            hits.append(f"{m['label']} {m['lon']:g}E")
    return hits


def lineage_in_box(L, box):
    """Axes and merge entries of a lineage record whose mean position lies inside the
    family box, plus any entry sharing a selected-region digest with one of them (a
    retained parent may lie outside the box), summarized without arrays: for an axis its
    vertex count and ranges, for a merge entry its input index, seed, pass-two survival,
    its own pass-one index, output index, minimum-extent rejection, the inputs sharing
    its region and which kept an output, and the selected region's cells, extents and
    median. A shared digest shows that several axes selected the same coarse region and
    that duplicate removal kept one; it establishes nothing about membership."""
    (la0, la1), (lo0, lo1) = box["lat"], box["lon"]

    def inside(la, lo):
        return la0 <= la <= la1 and lo0 <= lo <= lo1
    axes = [{"index": a["index"], "lat_mean": a["lat_mean"], "lon_mean": a["lon_mean"], "n_vertices": a["n_vertices"],
             "lat_range": [min(a["lat"]), max(a["lat"])], "lon_range": [min(a["lon"]), max(a["lon"])]} for a in L["axes"] if inside(a["lat_mean"], a["lon_mean"])]

    def entries(rows):
        """Entries inside the box, and every entry anywhere that shares a selected-region
        digest with one of them (the retained parent may lie outside the box), each with
        the inputs sharing its digest and which of them kept an output. `merged` is the
        entry's own pass-one index, not the index of an entry that absorbed it."""
        def digest(e):
            return ((e.get("select") or {}).get("region") or {}).get("region_sha256")
        by_digest = {}
        for e in rows:
            d = digest(e)
            if d:
                by_digest.setdefault(d, []).append(e)
        keep = [e for e in rows if inside(e["lat_mean"], e["lon_mean"])]
        extra = [e for d, group in by_digest.items() if any(inside(x["lat_mean"], x["lon_mean"]) for x in group) for e in group if e not in keep]
        out = []
        for e in keep + extra:
            sel = e.get("select") or {}
            reg = sel.get("region") or {}
            seed = e.get("seed")
            d = digest(e)
            group = by_digest.get(d, [e]) if d else [e]
            out.append({"input": e["input"], "mean": [e["lat_mean"], e["lon_mean"]], "seed": [seed.get("lat"), seed.get("lon")] if isinstance(seed, dict) else seed,
                        "inside_box": inside(e["lat_mean"], e["lon_mean"]), "parent_outside_box": (e in extra),
                        "kept_after_pass_two": e.get("kept_after_pass_two"), "pass_one_index": e.get("merged"), "output": e.get("output"), "rejected_min_extent": e.get("rejected_min_extent"),
                        "shares_region_with": [x["input"] for x in group if x is not e], "retained_input_of_group": [x["input"] for x in group if x.get("output") is not None],
                        "selected_cells": sel.get("selected_cells"), "selected_lat_extent": sel.get("selected_lat_extent"), "selected_lon_extent": sel.get("selected_lon_extent"),
                        "level_landed": sel.get("level_landed"), "exhausted": sel.get("exhausted"), "region_median": [reg.get("median_lat"), reg.get("median_lon")], "region_cells": reg.get("cells")})
        return out
    return {"axes": axes, "coarse": entries(L["coarse"]), "fine": entries(L["fine"])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace"); ap.add_argument("entry"); ap.add_argument("--family", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--table", required=True)
    a = ap.parse_args(argv)
    r = json.load(gzip.open(a.trace)); t = r["trace"]
    e = json.load(open(a.entry))
    entry_by = {s["time"]: {int(m["row_lat"]): m for m in s["rows"]} for s in e["steps"]}
    fam = next(f for f in t["families_traced"] if f["name"] == a.family)
    steps = sorted(t["steps"], key=lambda s: s["step"])
    finished = r.get("finished")
    fin_by_birth = {}
    if isinstance(finished, dict):
        fin_by_birth = finished
    elif isinstance(finished, list):
        for h in finished:
            b = h.get("birth") or h.get("id")
            if b is not None:
                fin_by_birth[str(b)] = h
    table = []
    per_history = {}
    for s in steps:
        stamp = stamp_of(s)
        if stamp not in entry_by:
            continue
        rows = entry_by[stamp]
        supported_step = SUPPORTED[0] <= stamp <= SUPPORTED[1]
        claims_by_cand = {}
        for c in s.get("claims", []):
            claims_by_cand.setdefault(c["candidate"], []).append(c["birth"])
        seeds_by_cand = {}
        for c in s.get("seeds", []):
            seeds_by_cand.setdefault(c["candidate"], []).append(c["birth"])
        match_by_cand = {}
        for m in s.get("matching", []):
            for ci in m.get("candidates_inside_polygon", []):
                sp = (m.get("implied_speed_ms") or {}).get(str(ci))
                match_by_cand.setdefault(ci, []).append({"birth": m["birth"], "chosen": m.get("chosen") == ci, "pass": m.get("pass"), "implied_speed_ms": sp})
        cands = []
        for c in s["candidates"]:
            if a.family not in (c.get("in_family_box") or []):
                continue
            lat, lon = float(c["lat_mean"]), float(c["lon_mean"])
            rr = rows_of(lat)
            worst = None
            for row in rr:
                cp, kind = counterpart(rows.get(row))
                d = None if cp is None else lon - cp
                rec = {"row": row, "counterpart_lon": cp, "counterpart_kind": kind, "distance_deg": d, "class": classify(d),
                       "tested_row": row in TESTED_ROWS, "other_structure": other_structure(rows.get(row), lon),
                       "i1_min_lon": (rows.get(row) or {}).get("i1", {}).get("lon") if rows.get(row) and rows[row].get("i1") else None}
                if worst is None or (rec["distance_deg"] is not None and (worst["distance_deg"] is None or abs(rec["distance_deg"]) > abs(worst["distance_deg"]))):
                    worst = rec
            cands.append({"index": c["index"], "lat": lat, "lon": lon, "lat_extent": c["lat_extent"], "lon_extent": c["lon_extent"], "n_points": c["n_points"],
                          "v_median": c.get("v_median"), "rows_considered": rr, "assessment": worst,
                          "claimed_by": claims_by_cand.get(c["index"], []), "seeded": seeds_by_cand.get(c["index"], []), "matching": match_by_cand.get(c["index"], [])})
            for b in claims_by_cand.get(c["index"], []):
                per_history.setdefault(b, []).append({"time": stamp, "supported": supported_step, "candidate": c["index"], "lat": lat, "lon": lon, "assessment": worst})
        lin = lineage_in_box(s["lineage"], fam["box"]) if "lineage" in s else None
        table.append({"time": stamp, "supported_step": supported_step, "n_candidates_in_box": len(cands), "candidates": cands, "lineage_in_box": lin,
                      "pruned": [p for p in s.get("pruned", []) if any(p["birth"] in (cc["claimed_by"] + cc["seeded"]) for cc in cands) or True],
                      "n_live_before": s.get("n_live_before"), "has_lineage": "lineage" in s})
    # association by the longest overlap of the supported steps
    overlap = {b: sum(1 for x in v if x["supported"]) for b, v in per_history.items()}
    assoc = max(overlap, key=lambda b: (overlap[b], -int(b) if str(b).isdigit() else 0)) if overlap and max(overlap.values()) > 0 else None
    verdict = {"associated_history": assoc, "overlap_steps": overlap.get(assoc, 0) if assoc else 0, "rule": "longest overlap of the 12 supported steps, held fixed"}
    n_sup = sum(1 for x in table if x["supported_step"])
    if assoc is not None:
        seq = [x for x in per_history[assoc] if x["supported"]]
        on_near = sum(1 for x in seq if x["assessment"]["tested_row"] and x["assessment"]["class"] in ("on", "near"))
        off_other = sum(1 for x in seq if x["assessment"]["tested_row"] and x["assessment"]["class"] == "off" and x["assessment"]["other_structure"])
        off_tested_rows = sum(1 for x in seq if not x["assessment"]["tested_row"])
        unassessable = sum(1 for x in seq if x["assessment"]["tested_row"] and x["assessment"]["class"] == "unassessable")
        missing = n_sup - len(seq)
        verdict.update({"supported_steps": n_sup, "steps_with_a_claim": len(seq), "on_or_near_on_tested_rows": on_near, "off_and_on_another_structure": off_other,
                        "claims_on_rows_outside_the_tested_rows": off_tested_rows, "unassessable_on_tested_rows": unassessable, "supported_steps_without_a_claim": missing})
        times = [x["time"] for x in seq]
        gaps = []
        idx = {x["time"]: i for i, x in enumerate([y for y in table if y["supported_step"]])}
        pos = sorted(idx[tm] for tm in times if tm in idx)
        for p, q in zip(pos, pos[1:]):
            if q - p > 1:
                gaps.append(q - p - 1)
        verdict["largest_recovery_gap_steps"] = max(gaps) if gaps else 0
        assessable = on_near + off_other + sum(1 for x in seq if x["assessment"]["tested_row"] and x["assessment"]["class"] == "off" and not x["assessment"]["other_structure"])
        if assessable == 0:
            verdict["verdict"] = "UNRESOLVED"
            verdict["why"] = "no assessable supported step on the tested rows for the associated history"
        elif on_near == assessable and missing <= 2 and verdict["largest_recovery_gap_steps"] <= 2 and off_tested_rows == 0:
            verdict["verdict"] = "SUPPORTED"; verdict["why"] = "on or near at every assessable supported step, gaps within the association's recovery"
        elif off_other > assessable / 2:
            verdict["verdict"] = "CONTRADICTED"; verdict["why"] = "off I1 and on another structure at a majority of the assessable supported steps"
        else:
            verdict["verdict"] = "UNRESOLVED"; verdict["why"] = "mixed, or claims on rows outside the tested rows, or gaps beyond the association's recovery"
    else:
        verdict.update({"supported_steps": n_sup, "verdict": "UNRESOLVED or CONTRADICTED pending the lineage read",
                        "why": "no history claims a candidate in the box at any supported step; the lineage at the declared steps decides whether an axis on the counterpart was extracted and removed"})
    out = {"generated_by": "scripts/pilot_i1_correspondence.py", "trace": a.trace, "entry_listing": a.entry, "family": fam,
           "conventions": {"on_deg": ON, "near_deg": NEAR, "tested_rows": TESTED_ROWS, "supported_steps": SUPPORTED, "precedence": "reversal, else assigned vorticity maximum",
                           "other_structure": "within 3 degrees of a W or E wind minimum on the row while off I1", "rows": "nearest integer row, both at a tie, larger distance taken",
                           "positions": "per-step claimed candidate representatives (association-time); stored finished positions listed beside them"},
           "steps": table, "histories": per_history, "verdict": verdict,
           "finished_positions_of_associated_history": (fin_by_birth.get(str(assoc)) if assoc is not None else None)}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    lines = ["# correspondence of the frozen candidate with the I1 entry; per candidate: index, representative (lat, lon), lon extent, row, counterpart kind and lon, distance, class, other-structure hits, claimed-by / seeded / matching (birth:chosen:pass:speed)", "",
             "| step | supported | candidates in box | details |", "| --- | --- | --- | --- |"]
    for x in table:
        det = []
        for c in x["candidates"]:
            w = c["assessment"]
            cp = "-" if w["counterpart_lon"] is None else f"{w['counterpart_lon']:.1f}"
            d = "-" if w["distance_deg"] is None else f"{w['distance_deg']:+.1f}"
            mt = ",".join(f"{m['birth']}:{'C' if m['chosen'] else 'c'}:{m['pass']}:{m['implied_speed_ms']:.1f}" for m in c["matching"] if m.get("implied_speed_ms") is not None)
            det.append(f"#{c['index']} ({c['lat']:.1f}, {c['lon']:.1f}) lon {c['lon_extent'][0]:g}..{c['lon_extent'][1]:g} row {w['row']}{'*' if w['tested_row'] else ''} {w['counterpart_kind'].split(' ')[0]} {cp} d {d} {w['class']}"
                       + (f" [{' '.join(w['other_structure'])}]" if w["other_structure"] else "") + f" claimed {c['claimed_by'] or '-'} seeded {c['seeded'] or '-'} match {mt or '-'}")
        lines.append(f"| {x['time']} | {'yes' if x['supported_step'] else 'no'} | {x['n_candidates_in_box']} | " + "; ".join(det) + " |")
    lines += ["", "## lineage at the declared steps, inside the family box", ""]
    for x in table:
        lin = x.get("lineage_in_box")
        if not lin:
            continue
        lines.append(f"### {x['time']}")
        lines.append("axes: " + ("; ".join(f"#{a['index']} mean ({a['lat_mean']:.1f}, {a['lon_mean']:.1f}) lon {a['lon_range'][0]:.1f}..{a['lon_range'][1]:.1f} lat {a['lat_range'][0]:.1f}..{a['lat_range'][1]:.1f} v{a['n_vertices']}" for a in lin["axes"]) or "none"))
        for name in ("coarse", "fine"):
            lines.append(f"{name}: " + ("; ".join(f"in{e['input']} mean ({e['mean'][0]:.1f}, {e['mean'][1]:.1f}){' OUTSIDE BOX, shares a region with an inside input' if e['parent_outside_box'] else ''} pass2={e['kept_after_pass_two']} out={e['output']} shares={e['shares_region_with']} retained={e['retained_input_of_group']} minext={e['rejected_min_extent']} sel={e['selected_cells']} lon {e['selected_lon_extent']} lat {e['selected_lat_extent']} med {e['region_median']}" for e in lin[name]) or "none"))
        lines.append("")
    lines += ["", f"verdict: {json.dumps(verdict)}"]
    with open(a.table, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(json.dumps(verdict, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
