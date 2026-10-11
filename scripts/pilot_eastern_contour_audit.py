#!/usr/bin/env python3
"""Connectivity audit of the sign-change branches (`EASTERN_VERTICAL_600_700_2026-10-11.md`,
section 10). It traces the zero contour of the same meridional-wind fields with
contourpy (matplotlib's two-dimensional contour extractor, terrain-invalid points masked),
assigns every original bracket to the contour line crossing its row between its two
points, and reports, at the 12 shared-region maps, whether the original branches fragment
connected contour lines and whether the apparent multiplicity survives on contour lines.
Where a contour passes a saddle cell, its connectivity there depends on the extractor's
choice and is reported as unresolved. No physical identity is claimed.

    python3 scripts/pilot_eastern_contour_audit.py --vertical-record <eval60_vertical_2006.json> --declaration <vertical json> \\
        --sp-file <sp> --out <json>

The near-read, near-output and on-region branch identities come from the retained record,
and every recorded branch must equal its recomputation, or the run refuses.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_alledge_replay as A  # noqa: E402
import pilot_eastern_measurement as EM  # noqa: E402
import pilot_eastern_vertical as VT  # noqa: E402
import pilot_field_sequence as FS  # noqa: E402


def contour_lines(v, valid, lat1, lon1):
    """The v = 0 contour lines, terrain-invalid points masked, as (n, 2) arrays of (lon, lat)."""
    import contourpy
    z = np.ma.masked_array(np.asarray(v, float), mask=~(np.asarray(valid, bool) & np.isfinite(v)))
    g = contourpy.contour_generator(np.asarray(lon1, float), np.asarray(lat1, float), z, line_type=contourpy.LineType.Separate)
    return [np.asarray(x, float) for x in g.lines(0.0)]


def bracket_line(b, lines, lon1):
    """The contour line with a vertex on the bracket's row strictly between its two
    columns (or on its east point when v is exactly zero there), or None."""
    lo, hi = float(lon1[b["j"]]), float(lon1[b["j"] + 1])
    for k, ln in enumerate(lines):
        on_row = np.abs(ln[:, 1] - b["lat"]) < 1e-9
        if np.any(on_row & (ln[:, 0] > lo - 1e-9) & (ln[:, 0] <= hi + 1e-9)):
            return k
    return None


def saddle_cells(v, valid):
    """Grid cells whose four corners alternate in sign, where contour connectivity is the
    extractor's choice."""
    s = np.asarray(v, float) > 0
    ok = np.asarray(valid, bool)
    a, b, c, d = s[:-1, :-1], s[:-1, 1:], s[1:, :-1], s[1:, 1:]
    okc = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1] & ok[1:, 1:]
    return okc & (a == d) & (b == c) & (a != b)


def line_saddles(line, saddles, lat1, lon1):
    """The saddle cells a contour line passes through (by the midpoints of its segments)."""
    out = set()
    lat1 = np.asarray(lat1, float); lon1 = np.asarray(lon1, float)
    for p, q in zip(line[:-1], line[1:]):
        x, y = (p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0
        j = int(np.searchsorted(lon1, x)) - 1
        i = int(np.argmin(np.abs(lat1 - y)))
        i = i if (lat1[i] >= y and i + 1 < lat1.size and lat1[i + 1] <= y) else i - 1
        if 0 <= i < saddles.shape[0] and 0 <= j < saddles.shape[1] and saddles[i, j]:
            out.add((i, j))
    return out


def audit_level(brs, lines, lon1, saddles, lat1, R, S, G):
    """Map original branches to contour lines and recompute the three descriptions on lines."""
    by = {br["id"]: br for br in brs}
    line_of = {bid: sorted({x for x in (bracket_line(b, lines, lon1) for b in by[bid]["brackets"]) if x is not None}) for bid in set(R) | set(S) | set(G)}
    LR = {x for bid in R for x in line_of[bid]}
    LS = {x for bid in S for x in line_of[bid]}
    LG = {x for bid in G for x in line_of[bid]}
    extra = LR - LS
    sad = {k: line_saddles(lines[k], saddles, lat1, lon1) for k in LR | LS}
    unresolved_pairs = [(r, s) for r in extra for s in LS if sad[r] & sad[s]]
    involved = set(R) | set(S) | set(G)
    lines_used = {x for bid in involved for x in line_of[bid]}
    return {"branches_involved": len(involved), "lines_they_lie_on": len(lines_used),
            "branches_sharing_a_line": sum(1 for bid in involved if any(set(line_of[bid]) & set(line_of[o]) for o in involved if o != bid)),
            "branches_on_several_lines": sum(1 for bid in involved if len(line_of[bid]) > 1),
            "branches_without_a_line": sum(1 for bid in involved if not line_of[bid]),
            "lines_near_read": sorted(LR), "lines_near_output": sorted(LS), "lines_on_region": sorted(LG),
            "additional_on_lines": bool(extra), "additional_on_lines_on_region": bool(extra & LG), "and_output_line_present": bool(extra & LG) and bool(LS),
            "separations_through_a_shared_saddle": [[int(a), int(b)] for a, b in unresolved_pairs],
            "saddle_cells_on_relevant_lines": {str(k): len(v) for k, v in sad.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--vertical-record", "--declaration", "--sp-file", "--out"):
        ap.add_argument(name, required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    rec = json.load(open(args.vertical_record))
    if rec["declaration"]["sha256"] != A.X.digest(args.declaration):
        raise SystemExit("REFUSED: the vertical record was not made under this declaration")
    decl = json.load(open(args.declaration))
    g = decl["grid"]
    lat1 = np.arange(g["lat"][1], g["lat"][0] - 0.5, -1.0); lon1 = np.arange(g["lon"][0], g["lon"][1] + 0.5, 1.0)
    out = []
    for x in rec["shared_regions"]:
        d = x["date"]
        day = VT.day_of(d)
        sp = FS.surface_pressure_on(args.sp_file, lat1, lon1, [day])[0]
        row = {"date": d, "segment": x["segment"]}
        for lev in ("700", "600"):
            v = FS.field_on(decl["v_files"]["2006"][lev], "v", "m s**-1", lat1, lon1, [day], float(lev))[0]
            valid = np.isfinite(v) & (sp > int(lev) * 100.0)
            brs = VT.branches(VT.brackets(v, valid, lat1, lon1), v.shape)
            by = {br["id"]: br for br in brs}
            for summ in x[lev]["branches"]:                                   # the recomputed branches are the recorded ones
                b = by.get(summ["id"])
                if b is None or VT.summarize_branch(b) != summ:
                    raise SystemExit(f"REFUSED: at {d} {lev} hPa the recomputed branch {summ['id']} is not the recorded one")
            lines = contour_lines(v, valid, lat1, lon1)
            row[lev] = audit_level(brs, lines, lon1, saddle_cells(v, valid), lat1, x[lev]["near_read"], x[lev]["near_surviving_output"], x[lev]["branches_on_region"])
            row[lev]["original"] = {"additional": x[lev]["additional_structure"],
                                    "on_region": bool((set(x[lev]["near_read"]) - set(x[lev]["near_surviving_output"])) & set(x[lev]["branches_on_region"]))}
        out.append(row)
    summary = {}
    for lev in ("700", "600"):
        summary[lev] = {"maps": len(out),
                        "branches_involved": sum(r[lev]["branches_involved"] for r in out), "lines_they_lie_on": sum(r[lev]["lines_they_lie_on"] for r in out),
                        "maps_with_branches_sharing_a_line": sum(1 for r in out if r[lev]["branches_sharing_a_line"]),
                        "original_additional": sum(1 for r in out if r[lev]["original"]["additional"]), "additional_on_lines": sum(1 for r in out if r[lev]["additional_on_lines"]),
                        "original_on_region": sum(1 for r in out if r[lev]["original"]["on_region"]), "on_lines_on_region": sum(1 for r in out if r[lev]["additional_on_lines_on_region"]),
                        "on_lines_on_region_with_output_line": sum(1 for r in out if r[lev]["and_output_line_present"]),
                        "maps_with_a_separation_through_a_shared_saddle": sum(1 for r in out if r[lev]["separations_through_a_shared_saddle"])}
    res = {"generated_by": "scripts/pilot_eastern_contour_audit.py", "script_sha256": A.X.digest(os.path.abspath(__file__)),
           "inputs": {"vertical_record": {"path": args.vertical_record, "sha256": A.X.digest(args.vertical_record)}, "declaration": args.declaration},
           "extractor": "contourpy, the v = 0 contour with terrain-invalid points masked", "maps": out, "summary": summary,
           "limits": "contour connectivity at saddle cells is the extractor's choice and is reported as unresolved; no physical identity is claimed"}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"Record {args.out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
