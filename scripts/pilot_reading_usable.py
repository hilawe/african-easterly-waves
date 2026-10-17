#!/usr/bin/env python3
"""One usable reference per scored map per reading, separated by sequence
(`EASTERN_ALTERNATIVE_READINGS_2026-10-13.md`, section 3).

A parsed reading keys its rows by date, and two sequences of the package show the same
map at some dates. Here only the rows of the sequence that OWNS a map's segment apply to
that map. Exactly one geometry row (status identified or weak) in the owning sequence
gives the map its usable reference. Two or more geometry rows there give "two answers"
and no reference. No geometry row gives the row's own status, indeterminate or none, and
no row at all gives "missing". Rows from the other sequence at the same date are kept
beside the map and not used. The output has the parsed reading's schema, so the
measurement reads it unchanged and lists every map without a reference with its status,
and it adds a usable table per segment. No row is corrected, dropped or adjudicated.

    python3 scripts/pilot_reading_usable.py --parsed <json> --declaration <json> \\
        --owner 1137_standing=A 1137_moving=A R8_east=B R9_first=C --out <json>
"""
import argparse
import hashlib
import json
import os

STATUSES = ("usable", "two answers", "indeterminate", "none", "missing")


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def usable_rows(rows, owner):
    """The owning sequence's rows at one date, reduced to one usable row or one status
    row, and the other sequences' rows kept apart."""
    mine = [r for r in rows if r.get("sequence") == owner]
    others = [r for r in rows if r.get("sequence") != owner]
    geoms = [r for r in mine if r.get("status") in ("identified", "weak")]
    if len(geoms) == 1:
        return "usable", [dict(geoms[0])], others
    if len(geoms) > 1:
        return "two answers", [{"status": "two answers", "sequence": owner, "answers": [dict(g) for g in geoms]}], others
    if mine:
        status = mine[0].get("status", "unparsed")
        return status, [dict(mine[0])], others
    return "missing", [{"status": "missing", "sequence": owner}], others


def build(parsed, decl, owners):
    maps, other, table = {}, {}, {}
    for m in decl["reference"]["maps"]:
        seg = m["segment"]
        if seg not in owners:
            raise SystemExit(f"REFUSED: segment {seg} has no owning sequence")
        status, rows, others = usable_rows(parsed["maps"].get(m["date"]) or [], owners[seg])
        maps[m["date"]] = rows
        if others:
            other[m["date"]] = others
        if m["scored"]:
            t = table.setdefault(seg, {s: [] for s in STATUSES})
            t.setdefault(status if status in STATUSES else "other", []).append(m["date"])
    for seg, t in table.items():
        t["counts"] = {s: len(v) for s, v in t.items() if s != "counts"}
    return maps, other, table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--parsed", "--declaration", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--owner", nargs="+", required=True, metavar="SEGMENT=SEQUENCE")
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    owners = dict(item.split("=", 1) for item in args.owner)
    parsed = json.load(open(args.parsed))
    decl = json.load(open(args.declaration))
    maps, other, table = build(parsed, decl, owners)
    out = {"generated_by": "scripts/pilot_reading_usable.py", "script_sha256": sha256_of(os.path.abspath(__file__)),
           "derived_from": {"path": args.parsed, "sha256": sha256_of(args.parsed)}, "declaration": {"path": args.declaration, "sha256": sha256_of(args.declaration)},
           "source": parsed.get("source"), "owners": owners, "rule": "one usable reference per scored map from the owning sequence; two geometry rows there are two answers and no reference; absent, indeterminate and missing answers kept distinct; other sequences' rows at the same date preserved and not used",
           "label": parsed.get("label"), "sequences": parsed.get("sequences"), "form": parsed.get("form"), "maps": maps, "other_sequence_rows": other, "usable": table}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for seg, t in table.items():
        print(seg, t["counts"])
    print(f"Record {args.out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
