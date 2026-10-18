#!/usr/bin/env python3
"""The stage E reporting supplement (`EASTERN_STAGE_E_EVALUATION_2026-10-17.md`, section 15,
from the independent review's finding F7). The two saved ledgers print the primary pair
classifications and the deep-minimum counts, and not the other tables the evaluation
cites. This prints those tables from the same unchanged saved reader output and listing,
each with its predicate, and checks that the switching partition the record states is
disjoint and exhaustive, printing any pair it leaves unassigned. It computes nothing from
the fields and changes no record.

    python3 scripts/pilot_stage_e_report.py --phases PH.json --listing LIST.json --out REPORT.md
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ("continuous_unique", "continuous_ambiguous", "switching", "replacement", "jump", "unassessable")
SIX_TIMES = ["1987-09-15T00", "1987-09-16T00", "1987-09-17T00", "1987-09-18T00", "1987-09-19T00", "1987-09-20T00"]
# The record's E3 partition, as named predicates over (rows, from, to), both ends inside.
E3_PREDICATES = [
    ("inside the sheet's expected coexistence subset, rows 15 to 26 N, 17 September 12Z to 18 September 18Z", (15, 26), "1987-09-17T12", "1987-09-18T18"),
    ("in the broader subset, rows 16 to 23 N, 17 September 00Z to 19 September 06Z, outside the expected time bounds", (16, 23), "1987-09-17T00", "1987-09-19T06"),
    ("rows 17 to 21 N on 15 and 16 September", (17, 21), "1987-09-15T00", "1987-09-16T18"),
    ("rows 12 to 15 N, all times", (12, 15), None, None),
    ("rows 20 to 21 N on 19 and 20 September", (20, 21), "1987-09-19T00", "1987-09-20T12"),
]
LATE_WINDOW = ("1987-09-19T06", "1987-09-20T12")


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _ledger():
    spec = importlib.util.spec_from_file_location("ledger_for_report", os.path.join(HERE, "pilot_evolution_ledger.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _hours(a, b):
    f = "%Y-%m-%dT%H"
    return (dt.datetime.strptime(b, f) - dt.datetime.strptime(a, f)).total_seconds() / 3600.0


def _in(p, rows, t0, t1):
    return rows[0] <= int(p["row"]) <= rows[1] and (t0 is None or p["from"] >= t0) and (t1 is None or p["to"] <= t1)


def switching_partition(pairs_by_row, predicates):
    """The switching pairs assigned to named predicates in order, each pair to the FIRST
    predicate it satisfies, so a later predicate is read as excluding the earlier ones and
    the groups are disjoint by construction. Every pair no predicate claims is listed as
    unassigned rather than dropped, which is the exhaustiveness check."""
    sw = [p for ps in pairs_by_row.values() for p in ps if p["class"] == "switching"]
    groups = {name: [] for name, *_ in predicates}
    unassigned = []
    for p in sw:
        hits = [name for name, rows, t0, t1 in predicates if _in(p, rows, t0, t1)]
        if hits:
            groups[hits[0]].append(p)
        else:
            unassigned.append(p)
    return {"groups": groups, "unassigned": unassigned, "total": len(sw), "assigned": sum(len(g) for g in groups.values())}


def standing_inside(standing_by_row, t0, t1):
    """Per row, the elapsed hours of retained standing intervals that fall inside the window."""
    out = {}
    for row, ss in standing_by_row.items():
        h = 0.0
        for s in ss:
            a, b = max(s["start"], t0), min(s["end"], t1)
            if b > a:
                h += _hours(a, b)
        out[int(row)] = h
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phases", required=True); ap.add_argument("--listing", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"REFUSED: {a.out} exists and a record is never overwritten")
    G = _ledger()
    ph = json.load(open(a.phases)); rec = json.load(open(a.listing))
    assert ph["record"]["sha256"] == sha256_of(a.listing), "the phases output was not computed from this listing"
    times = [s["time"] for s in rec["steps"]]
    L = [f"# stage E reporting supplement over {os.path.basename(a.phases)} (sha256 {sha256_of(a.phases)[:16]}) and {os.path.basename(a.listing)} (sha256 {ph['record']['sha256'][:16]}), reader {ph['script_sha256'][:16]}, listing grid {rec['declared'].get('grid')} ({rec['declared'].get('lon_spacing_deg'):g} deg)", "",
         "Every table is computed from the two saved files above and carries its predicate. Nothing here changes the method or the saved outputs.", ""]
    L.append("## Pairs by class, supported row-steps, phases and the adjacent-row set, under every parameter set (all rows, all times)")
    L.append("| set | unique | ambiguous | switching | replacement | jump | unassessable | supported | in west phases | in standing | west phases of 48 h or more | set rows | overlap h | ambiguous pairs in the set |")
    L.append("| --- |" + " --- |" * 13)
    for name, res in ph["results"].items():
        P = res["params"]; d = res["displacement_set"]
        ov = (min(p["i_end"] for p in d["phases"]) - max(p["i_start"] for p in d["phases"])) * P["step_hours"] if d["phases"] else None
        amb = sum(p["ambiguous_pairs"] for p in d["phases"]); den = sum(p["n_steps"] - 1 for p in d["phases"])
        tot = {c: sum(v["pairs_by_class"][c] for v in res["groups"].values()) for c in CLASSES}
        sup = sum(v["supported_row_steps"] for v in res["groups"].values()); west = sum(v["row_steps_in_west_phases"] for v in res["groups"].values()); stand = sum(v["row_steps_in_standing"] for v in res["groups"].values())
        long = sum(1 for r in res["phases"].values() for p in r if p["direction"] == "west" and p["hours"] >= 48)
        L.append(f"| {name} | " + " | ".join(str(tot[c]) for c in CLASSES) + f" | {sup} | {west} | {stand} | {long} | {d['n_rows']} {d['rows']} | {ov} | {amb} of {den} |")
    L.append("")
    res = ph["results"]["primary"]
    idx = {s["time"]: s for s in rec["steps"]}

    def pos(t, row):
        m = next((m for m in idx[t]["rows"] if int(m["row_lat"]) == row), None)
        return None if m is None or m["status"] != "assigned" else m["i1"]["lon"]
    rows = sorted({int(r) for r in res["pairs"]}, reverse=True)
    L.append("## Selected positions (the deepest in-window minimum per row, assigned rows, whether or not supported) at six times")
    L.append("| time | " + " | ".join(str(r) for r in rows) + " |"); L.append("| --- |" + " --- |" * len(rows))
    for t in SIX_TIMES:
        L.append(f"| {t[5:]} | " + " | ".join("-" if pos(t, r) is None else f"{pos(t, r):g}" for r in rows) + " |")
    L.append("")
    L.append("## Westward phases of 12 elapsed hours or more, primary set, by row")
    for r in rows:
        ps = [p for p in res["phases"][str(r)] if p["direction"] == "west" and p["hours"] >= 12]
        if ps:
            L.append(f"  {r} N: " + "; ".join(f"{p['start'][5:]}..{p['end'][5:]} {p['hours']} h {p['lon_start']:g}->{p['lon_end']:g} net {p['net_move']:+g} rate {p['rate_deg_per_day']:+.2f}/day ambiguous {p['ambiguous_pairs']}" for p in ps))
    L.append("")
    L.append("## Eastward phases of 24 elapsed hours or more, primary set, by row")
    for r in rows:
        ps = [p for p in res["phases"][str(r)] if p["direction"] == "east" and p["hours"] >= 24]
        if ps:
            L.append(f"  {r} N: " + "; ".join(f"{p['start'][5:]}..{p['end'][5:]} {p['hours']} h {p['lon_start']:g}->{p['lon_end']:g}" for p in ps))
    L.append("")
    L.append("## Retained standing intervals of 24 elapsed hours or more, primary set, by row")
    for r in rows:
        ss = [s for s in res["standing"][str(r)] if s["hours"] >= 24]
        if ss:
            L.append(f"  {r} N: " + "; ".join(f"{s['start'][5:]}..{s['end'][5:]} {s['hours']} h at {s['lon_min']:g} to {s['lon_max']:g} E" for s in ss))
    L.append("")
    si = standing_inside(res["standing"], *LATE_WINDOW)
    L.append(f"## Hours of retained standing inside the sheet's late window, {LATE_WINDOW[0]} to {LATE_WINDOW[1]}, by row (rows with none omitted)")
    L.append("  " + ", ".join(f"{r} N {si[r]:g} h" for r in rows if si.get(r, 0) > 0))
    L.append("")
    part = switching_partition(res["pairs"], E3_PREDICATES)
    L.append(f"## The E3 switching partition, {part['total']} switching pairs, each assigned to the first named predicate it satisfies (both ends inside), {part['assigned']} assigned and {len(part['unassigned'])} unassigned")
    for name, *_ in E3_PREDICATES:
        g = part["groups"][name]
        L.append(f"  {len(g)} {name}: " + ", ".join(f"{p['row']} N {p['from'][5:]} {p['lon_from']:g}->{p['lon_to']:g}" for p in sorted(g, key=lambda p: (-p['row'], p['from']))))
    L.append("  unassigned: " + (", ".join(f"{p['row']} N {p['from'][5:]}" for p in part["unassigned"]) if part["unassigned"] else "none"))
    L.append("")
    L.append("## Switching and replacement pairs before 16 September 00Z on rows 19 to 23 N (the reorganization of 15 September)")
    for r in rows:
        if 19 <= r <= 23:
            for p in res["pairs"][str(r)]:
                if p["class"] in ("switching", "replacement") and p["from"] < "1987-09-16T00":
                    L.append(f"  {r} N {p['from'][5:]} -> {p['to'][5:]} {p['lon_from']:g}->{p['lon_to']:g} {p['class']}")
    L.append("")
    L.append("## Row 22 N: motion labels and selected positions under the primary set, and its westward phases under the rate sensitivity")
    lab = dict(res["motion"]["22"])
    L.append("  " + ", ".join(f"{times[i][5:]} {lab.get(i, '-')[0]} {pos(times[i], 22):g}" for i in range(len(times))))
    r05 = ph["results"]["rate_0.5"]
    L.append("  rate 0.5 west phases of 48 h or more: " + "; ".join(f"{p['start'][5:]}..{p['end'][5:]} {p['hours']} h {p['lon_start']:g}->{p['lon_end']:g}" for p in r05["phases"]["22"] if p["direction"] == "west" and p["hours"] >= 48))
    L.append("")
    L.append("## Ambiguous pairs by row group under the primary set and the 2.5-degree bound")
    for name in ("primary", "bound_2.5"):
        g = ph["results"][name]["groups"]
        L.append(f"  {name}: " + ", ".join(f"{k} {v['pairs_by_class']['continuous_ambiguous']}" for k, v in g.items()) + f", total {sum(v['pairs_by_class']['continuous_ambiguous'] for v in g.values())}")
    L.append("")
    pc = G.pair_counts(res["pairs"], (18, 24), "1987-09-15T12", "1987-09-18T12", (40.0, 60.0), ("jump", "replacement", "switching"))
    dm = G.deep_minima_counts(rec, -8.0, (18, 24), "1987-09-15T12", "1987-09-18T12")
    L.append(f"## The ledger's sheet-subset counts, reproduced here: {pc['predicate']}: " + ", ".join(f"{c} {v}" for c, v in pc["counts"].items()) + f"; {dm['predicate']}: assigned {dm['assigned']}, two or more deep minima {dm['two_or_more']}")
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(l for l in L if l.startswith("## The E3") or l.startswith("  unassigned") or l.startswith("| primary") or l.startswith("| rate")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
