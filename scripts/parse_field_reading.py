#!/usr/bin/env python3
"""Parse one alternative reading of the field sequences into per-map geometry, and screen
its form (`EASTERN_ALTERNATIVE_READINGS_2026-10-13.md`, section 1).

A reading's per-map table has one row per map,

    Sequence A, 2006-07-30 00Z: identified; 50; 10 to 20; northerlies west, southerlies east

with a status (identified, weak, indeterminate, none), an axis longitude as one value or a
range ("45 to 47"), a latitude range, and a signature. Rows with status indeterminate or
none carry no geometry. Two rows at one date are two geometries. Nothing is filled in,
corrected or dropped: the parsed record holds every row as read.

The form screen reports, per sequence, the maps in the figure, the rows read, duplicated
and missing dates, the status counts, the distinct latitude ranges, and the longitude step
between consecutive scored maps. It is reported, never used to exclude a reading.

    python3 scripts/parse_field_reading.py --reading <md> --source <label> --sequences A=<fieldseq name>,... \\
        --evidence <extent dir> --out <json>
"""
import argparse
import hashlib
import json
import os
import re
import statistics

ROW = re.compile(r"^\s*Sequence ([A-Z]),\s*(\d{4}-\d{2}-\d{2}) (\d{2})Z:\s*(identified|weak|indeterminate|none)\b\s*;?\s*(.*?)\s*$", re.I)
STATUSES = ("identified", "weak", "indeterminate", "none")


def parse_value(text):
    """A longitude or latitude field: one number, a range "a to b", or nothing."""
    t = text.strip().rstrip(".").replace("\u2014", "").replace("\u2013", "to").strip()
    if not t:
        return None
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*(?:to|-)\s*(-?\d+(?:\.\d+)?)", t)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return [min(a, b), max(a, b)]
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*[EN]?", t)
    return float(m.group(1)) if m else None


def parse_rows(text):
    rows = []
    for line in text.splitlines():
        m = ROW.match(line)
        if not m:
            continue
        seq, day, hour, status, rest = m.group(1).upper(), m.group(2), m.group(3), m.group(4).lower(), m.group(5)
        parts = [p.strip() for p in rest.split(";")]
        row = {"sequence": seq, "date": f"{day}T{hour}", "status": status, "text": line.strip()}
        if status in ("identified", "weak"):
            lon = parse_value(parts[0]) if parts else None
            lat = parse_value(parts[1]) if len(parts) > 1 else None
            row["signature"] = "; ".join(parts[2:]) if len(parts) > 2 else ""
            if lon is None or not isinstance(lat, list):
                row["status"] = "unparsed"
                row["problem"] = f"status {status} without a longitude and a latitude range"
            else:
                if isinstance(lon, list):
                    row["axis_lon_range"] = lon
                else:
                    row["axis_lon"] = lon
                row["lat_range"] = lat
        else:
            row["signature"] = rest.strip()
        rows.append(row)
    return rows


def form_screen(rows, figure_dates):
    """Per sequence, the figure against the rows."""
    out = {}
    for seq, dates in figure_dates.items():
        mine = [r for r in rows if r["sequence"] == seq]
        by_date = {}
        for r in mine:
            by_date.setdefault(r["date"], []).append(r)
        scored = {d: [r for r in rs if r["status"] in ("identified", "weak")] for d, rs in by_date.items()}
        lat_ranges = sorted({tuple(r["lat_range"]) for rs in scored.values() for r in rs})
        steps = []
        for a, b in zip(dates[:-1], dates[1:]):
            if len(scored.get(a, [])) == 1 and len(scored.get(b, [])) == 1:
                la, lb = scored[a][0], scored[b][0]
                ca = la.get("axis_lon", sum(la.get("axis_lon_range", [0, 0])) / 2.0)
                cb = lb.get("axis_lon", sum(lb.get("axis_lon_range", [0, 0])) / 2.0)
                steps.append(cb - ca)
        out[seq] = {"maps_in_figure": len(dates), "rows": len(mine), "dates_read": len(by_date),
                    "dates_with_two_or_more_rows": sorted(d for d, rs in by_date.items() if len(rs) > 1),
                    "dates_missing": [d for d in dates if d not in by_date], "dates_not_in_figure": sorted(d for d in by_date if d not in dates),
                    "status_counts": {s: sum(1 for r in mine if r["status"] == s) for s in STATUSES + ("unparsed",)},
                    "distinct_latitude_ranges": [list(x) for x in lat_ranges],
                    "longitude_step_deg": {"n": len(steps), "mean": round(statistics.mean(steps), 2) if steps else None,
                                           "spread": round(statistics.pstdev(steps), 2) if len(steps) > 1 else None,
                                           "restarts_over_10_deg": sum(1 for s in steps if s > 10.0)},
                    "longitudes": {"min": min((r.get("axis_lon", sum(r.get("axis_lon_range", [0, 0])) / 2.0) for rs in scored.values() for r in rs), default=None),
                                   "max": max((r.get("axis_lon", sum(r.get("axis_lon_range", [0, 0])) / 2.0) for rs in scored.values() for r in rs), default=None)}}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--reading", "--source", "--sequences", "--evidence", "--out"):
        ap.add_argument(name, required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    with open(args.reading, "rb") as fh:
        blob = fh.read()
    rows = parse_rows(blob.decode("utf-8"))
    if not rows:
        raise SystemExit("REFUSED: no per-map rows found")
    figure_dates = {}
    for item in args.sequences.split(","):
        letter, name = item.split("=")
        rec = json.load(open(os.path.join(args.evidence, f"eval60_fieldseq_{name}.json")))
        figure_dates[letter] = [d.replace(" ", "T").replace("Z", "") for f in rec["figures"] for d in f["dates"]]
    maps = {}
    for r in rows:
        maps.setdefault(r["date"], []).append({k: r[k] for k in ("status", "axis_lon", "axis_lon_range", "lat_range", "signature", "sequence") if k in r})
    out = {"generated_by": "scripts/parse_field_reading.py", "source": args.source, "reading": {"path": args.reading, "sha256": hashlib.sha256(blob).hexdigest()},
           "label": "an additional model interpretation of the fields, not a scientific adjudication and not a calibrated uncertainty estimate",
           "sequences": {k: v for k, v in sorted(figure_dates.items())}, "rows": rows, "maps": maps, "form": form_screen(rows, figure_dates)}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out["form"], indent=1))
    print(f"{len(rows)} rows, {len(maps)} dates. Record {args.out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
