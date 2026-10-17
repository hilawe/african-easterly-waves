#!/usr/bin/env python3
"""Summaries derived from an entry listing record (`pilot_i1_entry_listing.py`): per row,
the twelve-hourly sequence of the I1 wind minimum, the primary-threshold edge, the
reversal and the assigned vorticity maximum, and per step the rows that meet a declared
support rule (I1 minimum at or below the primary threshold, the primary edge available,
and the minimum at least a declared margin inside the window's western bound, so that a
minimum at the stripe boundary is not counted). A listing of positions, not a track.

    python3 scripts/pilot_i1_entry_summary.py RECORD --rows 16 20 24 28 --every-hours 12 --margin-deg 2
"""
import argparse
import json


def support_rows(step, threshold, margin):
    out = []
    ann = step.get("annotation") or {}
    lo = (ann.get("i1") or {}).get("lon", [None])[0]
    for m in step["rows"]:
        if m["status"] != "assigned":
            continue
        if m["i1"]["v"] <= threshold and m["edge"]["primary"]["lon"] is not None and lo is not None and m["i1"]["lon"] >= lo + margin:
            out.append(int(m["row_lat"]))
    return out


def minima_cell(m, below=None):
    """Every wind minimum of one row at one step in longitude order, the selected one
    starred, W and other labeled by their first letter, so competing minima are read
    beside the selection and not replaced by it. `below` keeps only minima at or under
    that value, a display choice that discards nothing from the record. An abstained
    row still lists its minima. Added 2026-10-16 for the I2 entry."""
    mins = sorted(m.get("wind_minima", []), key=lambda x: x["lon"])
    if below is not None:
        mins = [x for x in mins if x["v"] <= below]
    sel = (m.get("i1") or {}).get("lon") if m.get("status") == "assigned" else None
    parts = []
    for x in mins:
        tag = "*" if sel is not None and x["lon"] == sel else ("W" if x["label"] == "W" else "E" if x["label"] == "E" else "o" if x["label"] == "other" else "")
        parts.append(f"{tag}{x['lon']:g}({x['v']:.1f})")
    head = "" if m.get("status") == "assigned" else ("abst" if m.get("status") == "abstained" else "nomin")
    return " ".join(([head] if head else []) + parts) if (head or parts) else "none"


def jumps(record, min_deg):
    """Consecutive-step moves of the selected minimum on one row by at least `min_deg`
    degrees, with every minimum listed at both steps, so a move of the selection between
    two coexisting minima is read as a change of which is deeper and not as translation.
    Pairs with an unassigned row at either step are skipped. Added 2026-10-16."""
    out = []
    steps = record["steps"]
    for s0, s1 in zip(steps, steps[1:]):
        for m0 in s0["rows"]:
            row = int(m0["row_lat"])
            m1 = next((x for x in s1["rows"] if int(x["row_lat"]) == row), None)
            if m1 is None or m0["status"] != "assigned" or m1["status"] != "assigned":
                continue
            d = m1["i1"]["lon"] - m0["i1"]["lon"]
            if abs(d) >= min_deg:
                out.append({"row": row, "from": s0["time"], "to": s1["time"], "lon_from": m0["i1"]["lon"], "lon_to": m1["i1"]["lon"], "move_deg": d,
                            "minima_from": minima_cell(m0), "minima_to": minima_cell(m1)})
    return out


def arrival(record, lo, hi):
    """Per row, the first step from which the selected minimum stays inside [lo, hi] at
    every later assigned step to the end of the record, or no entry when it never does.
    A position read from the record, not a track. Added 2026-10-16."""
    rows = sorted({int(m["row_lat"]) for s in record["steps"] for m in s["rows"]}, reverse=True)
    out = {}
    for row in rows:
        seq = [(s["time"], m) for s in record["steps"] for m in s["rows"] if int(m["row_lat"]) == row and m["status"] == "assigned"]
        for k, (t, m) in enumerate(seq):
            if all(lo <= x["i1"]["lon"] <= hi for _, x in seq[k:]):
                out[row] = t
                break
    return out


def coexistence(record, below, min_sep):
    """Row-steps whose window holds at least two minima at or under `below` separated by at
    least `min_sep` degrees, with those minima's longitudes, read from the record so that
    a selection between coexisting minima is never mistaken for one feature. Added 2026-10-16."""
    out = []
    for s in record["steps"]:
        for m in s["rows"]:
            deep = sorted(x["lon"] for x in m.get("wind_minima", []) if x["label"] == "I1" and x["v"] <= below)
            if len(deep) >= 2 and deep[-1] - deep[0] >= min_sep:
                out.append((int(m["row_lat"]), s["time"], deep))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("record"); ap.add_argument("--rows", nargs="+", type=int, required=True)
    ap.add_argument("--every-hours", type=int, default=12); ap.add_argument("--margin-deg", type=float, default=2.0)
    ap.add_argument("--with-minima", action="store_true", help="also print, per row and step, every wind minimum with its label and the selected one starred")
    ap.add_argument("--minima-below", type=float, default=None, help="with --with-minima, print only minima at or under this value (m/s); the record keeps all")
    ap.add_argument("--jumps", type=float, default=None, metavar="DEG", help="also list consecutive-step moves of the selected minimum of at least DEG degrees on any row, with every minimum at both steps")
    ap.add_argument("--arrival", nargs=2, type=float, default=None, metavar=("LO", "HI"), help="also print, per row, the first step from which the selected minimum stays inside [LO, HI] to the end")
    ap.add_argument("--coexist", nargs=2, type=float, default=None, metavar=("BELOW", "SEP"), help="also list row-steps with two or more in-window minima at or under BELOW m/s at least SEP degrees apart")
    a = ap.parse_args(argv)
    r = json.load(open(a.record))
    thr = float(r["annotation"]["content"]["edge_thresholds_ms"]["primary"])
    every = max(1, a.every_hours // int(r["declared"]["every_hours"]))
    steps = r["steps"][::every]
    print(f"# rows, every {a.every_hours} h, from {steps[0]['time']} to {steps[-1]['time']}; per cell: I1 min lon | edge({thr:g}) | reversal | ZA/ZS; '-' unavailable, 'abst' abstained")
    for row in a.rows:
        cells = []
        for s in steps:
            m = next((x for x in s["rows"] if int(x["row_lat"]) == row), None)
            if m is None or m["status"] != "assigned":
                cells.append("abst" if (m is None or m["status"] == "abstained") else "nomin")
                continue
            e = m["edge"]["primary"]["lon"]; rv = m["reversal"]["lon"]
            z = m["zs"] or m["za"]
            cells.append(f"{m['i1']['lon']:g}|{'-' if e is None else f'{e:g}'}|{'-' if rv is None else f'{rv:.1f}'}|{'-' if not z else f'{z['lon']:g}'}")
        print(f"{row} N: " + "  ".join(cells))
    if a.with_minima:
        print(f"# every wind minimum per row and step{'' if a.minima_below is None else f' at or under {a.minima_below:g} m/s'}, lon(v): '*' the selected (deepest inside the window), 'W' west of the window, 'E' second window, 'o' other; competing minima are read here, not from the selection")
        for row in a.rows:
            print(f"{row} N:")
            for s in steps:
                m = next((x for x in s["rows"] if int(x["row_lat"]) == row), None)
                print(f"  {s['time']}: " + (minima_cell(m, a.minima_below) if m else "no row"))
    if a.jumps is not None:
        J = jumps(r, a.jumps)
        print(f"# moves of the selected minimum of at least {a.jumps:g} degrees between consecutive steps ({len(J)}), with every minimum at both steps")
        for j in J:
            print(f"{j['row']} N {j['from']} -> {j['to']}: {j['lon_from']:g} -> {j['lon_to']:g} ({j['move_deg']:+g}) | before: {j['minima_from']} | after: {j['minima_to']}")
    if a.arrival is not None:
        A = arrival(r, a.arrival[0], a.arrival[1])
        print(f"# per row, the first step from which the selected minimum stays within {a.arrival[0]:g} to {a.arrival[1]:g} E to the end of the record ('never' otherwise)")
        for row in sorted({int(m['row_lat']) for s in r['steps'] for m in s['rows']}, reverse=True):
            print(f"{row} N: {A.get(row, 'never')}")
    if a.coexist is not None:
        C = coexistence(r, a.coexist[0], a.coexist[1])
        print(f"# row-steps with two or more in-window minima at or under {a.coexist[0]:g} m/s at least {a.coexist[1]:g} degrees apart ({len(C)})")
        for row, t, deep in C:
            print(f"{row} N {t}: {' '.join(f'{x:g}' for x in deep)}")
    print(f"# supported rows per step (I1 min <= {thr:g} m/s, primary edge available, min >= window west bound + {a.margin_deg:g} deg)")
    for s in r["steps"]:
        rows = support_rows(s, thr, a.margin_deg)
        rev = [int(m["row_lat"]) for m in s["rows"] if int(m["row_lat"]) in rows and m["reversal"]["lon"] is not None]
        zs = [int(m["row_lat"]) for m in s["rows"] if int(m["row_lat"]) in rows and (m["zs"] or m["za"])]
        print(f"{s['time']}: {len(rows)} rows {rows[0] if rows else '-'}..{rows[-1] if rows else '-'} N; reversal on {len(rev)}; vorticity max on {len(zs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
