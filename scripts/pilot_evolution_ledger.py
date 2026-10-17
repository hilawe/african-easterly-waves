#!/usr/bin/env python3
"""The evolution ledger of the second development package
(`EASTERN_EVOLUTION_DEVELOPMENT_2_2026-10-17.md`, section 2). Every number it prints is
computed from a saved reader output and its saved listing under a subset predicate that is
printed beside it, so that prose cites the ledger instead of transcribing counts. It
computes nothing from the fields and changes no record.

    python3 scripts/pilot_evolution_ledger.py --phases PH.json --listing LIST.json --set primary \
        --rows 16 30 --t-from 1983-06-13T12 --t-to 1983-06-17T12 --lon 40 56 --out LEDGER.md
"""
import argparse
import hashlib
import json
import os

CLASSES = ("continuous_unique", "continuous_ambiguous", "switching", "replacement", "jump", "unassessable")
GROUPS = ((25, 30), (19, 24), (12, 18))


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _predicate(rows, t_from, t_to, lon_range, classes):
    parts = [f"rows {rows[0]} to {rows[1]} N"]
    if t_from or t_to:
        parts.append(f"pairs from {t_from or 'start'} to {t_to or 'end'}")
    if lon_range:
        parts.append(f"both ends lon {lon_range[0]:g} to {lon_range[1]:g} E")
    if classes:
        parts.append("classes " + ", ".join(classes))
    return ", ".join(parts)


def _select(pairs_by_row, rows, t_from=None, t_to=None, lon_range=None, classes=None):
    out = []
    for row, pairs in pairs_by_row.items():
        r = int(row)
        if not (rows[0] <= r <= rows[1]):
            continue
        for p in pairs:
            if t_from and p["from"] < t_from:
                continue
            if t_to and p["to"] > t_to:
                continue
            if classes and p["class"] not in classes:
                continue
            if lon_range:
                if p["lon_from"] is None or p["lon_to"] is None:
                    continue
                if not (lon_range[0] <= p["lon_from"] <= lon_range[1] and lon_range[0] <= p["lon_to"] <= lon_range[1]):
                    continue
            out.append(p)
    return out


def pair_counts(pairs_by_row, rows, t_from=None, t_to=None, lon_range=None, classes=None):
    """Counts by class under the stated predicate, the predicate printed with them."""
    classes = tuple(classes) if classes else CLASSES
    sel = _select(pairs_by_row, rows, t_from, t_to, lon_range, classes)
    return {"predicate": _predicate(rows, t_from, t_to, lon_range, classes), "counts": {c: sum(1 for p in sel if p["class"] == c) for c in classes}, "n": len(sel)}


def _row_at(step, row):
    return next((m for m in step["rows"] if int(m["row_lat"]) == row), None)


def _deep(m, depth_ms):
    return sorted(x["lon"] for x in m.get("wind_minima", []) if x["label"] == "I1" and x["v"] <= depth_ms)


def classify_changes(record, pairs_by_row, depth_ms=-8.0, bound_deg=1.5, lone_deg=3.0, min_abs_move=2.0, rows=(10, 30), t_from=None, t_to=None, lon_range=None):
    """Every non-continuous assessable pair (jump, replacement or switching, the classes
    the printed predicate names) whose selected position moves by at least `min_abs_move`,
    classified by the deep in-window minima at both ends. The class filter was added at
    the independent review of 2026-10-17 (F5): without it, under the 2.5-degree sensitivity
    bound a two-degree continuous pair was classified too, and a single moving minimum
    read as `coexisting` because the selections themselves lie within the enlarged bound.
    The six retained primary ledgers are unchanged by the filter, because a two-degree
    change exceeds the primary bound. `coexisting`: a deep minimum
    persists within the bound of the old selection at the later step and one pre-existed
    within the bound of the new selection at the earlier step. `lone-shift`: neither, the
    move is at most `lone_deg`, and no other deep minimum lies within `lone_deg` of the
    selection at either end, so one minimum shifted. `appeared`: the new selection had no
    deep minimum within the bound at the earlier step. `lost`: the old selection has no deep
    minimum within the bound at the later step while the new one pre-existed. `other`
    covers the rest. The classification reads listed minima and establishes no identity."""
    steps = record["steps"]
    out = []
    for p in _select(pairs_by_row, rows, t_from, t_to, lon_range, ("jump", "replacement", "switching")):
        if p["lon_from"] is None or p["lon_to"] is None or abs(p["move"]) < min_abs_move:
            continue
        m0, m1 = _row_at(steps[p["k"]], int(p["row"])), _row_at(steps[p["k"] + 1], int(p["row"]))
        D0, D1 = _deep(m0, depth_ms), _deep(m1, depth_ms)
        L0, L1 = p["lon_from"], p["lon_to"]
        old_persists = any(abs(x - L0) <= bound_deg for x in D1)
        new_preexisted = any(abs(x - L1) <= bound_deg for x in D0)
        others0 = [x for x in D0 if x != L0 and abs(x - L0) <= lone_deg]
        others1 = [x for x in D1 if x != L1 and abs(x - L1) <= lone_deg]
        if old_persists and new_preexisted:
            kind = "coexisting"
        elif not old_persists and not new_preexisted and abs(p["move"]) <= lone_deg and not others0 and not others1:
            kind = "lone-shift"
        elif not new_preexisted:
            kind = "appeared"
        elif not old_persists:
            kind = "lost"
        else:
            kind = "other"
        out.append({"row": int(p["row"]), "from": p["from"], "to": p["to"], "lon_from": L0, "lon_to": L1, "move": p["move"], "class": p["class"], "kind": kind, "deep_before": D0, "deep_after": D1})
    return out


def c_pos(c):
    return f"{c['lon_from']:g}->{c['lon_to']:g} move {c['move']:+g} {c['class']}"


def _steps_in(record, t_from, t_to):
    for s in record["steps"]:
        if t_from and s["time"] < t_from:
            continue
        if t_to and s["time"] > t_to:
            continue
        yield s


def deep_minima_counts(record, depth_ms=-8.0, rows=(10, 30), t_from=None, t_to=None):
    """Assigned row-steps under the subset, and how many of them list two or more deep
    in-window minima (competing selections at one row-step)."""
    assigned = two = 0
    for s in _steps_in(record, t_from, t_to):
        for m in s["rows"]:
            if not (rows[0] <= int(m["row_lat"]) <= rows[1]) or m["status"] != "assigned":
                continue
            assigned += 1
            if len(_deep(m, depth_ms)) >= 2:
                two += 1
    return {"predicate": f"rows {rows[0]} to {rows[1]} N, steps from {t_from or 'start'} to {t_to or 'end'}, depth {depth_ms:g} m/s", "assigned": assigned, "two_or_more": two}


def selection_offsets(record, other, rows=(10, 30), t_from=None, t_to=None):
    """How far the other listing's selection sits from this listing's at every row-step
    both assign, under the subset: the same column, half a degree off, or farther (by
    distance). Establishes nothing about features; it measures the two samplings' readings."""
    o_idx = {(s["time"], int(m["row_lat"])): m for s in other["steps"] for m in s["rows"]}
    same = half = both = 0; farther = {}
    for s in _steps_in(record, t_from, t_to):
        for m in s["rows"]:
            r = int(m["row_lat"])
            if not (rows[0] <= r <= rows[1]) or m["status"] != "assigned":
                continue
            om = o_idx.get((s["time"], r))
            if om is None or om["status"] != "assigned":
                continue
            both += 1
            d = abs(om["i1"]["lon"] - m["i1"]["lon"])
            if d == 0:
                same += 1
            elif d == 0.5:
                half += 1
            else:
                farther[d] = farther.get(d, 0) + 1
    return {"predicate": f"rows {rows[0]} to {rows[1]} N, steps from {t_from or 'start'} to {t_to or 'end'}", "both_assigned": both, "same": same, "half_degree": half, "farther": farther}


def counterparts(changes, other_pairs_by_row):
    """Each classified change paired with the pair at the same row and start time in another
    reader output (the other grid's reading of the same row-step). A row-step the other
    output has no pair for is `missing`, counted and never dropped."""
    idx = {(int(row), p["from"]): p for row, ps in other_pairs_by_row.items() for p in ps}
    out = []
    for c in changes:
        o = idx.get((c["row"], c["from"]))
        out.append({**c, "other_class": o["class"] if o else "missing", "other_move": o["move"] if o else None,
                    "other_lon_from": o["lon_from"] if o else None, "other_lon_to": o["lon_to"] if o else None})
    return out


def counterpart_table(T):
    """Counts of (kind in this record, class in the other record)."""
    tab = {}
    for t in T:
        key = (t["kind"], t["other_class"])
        tab[key] = tab.get(key, 0) + 1
    return tab


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phases", required=True); ap.add_argument("--listing", required=True); ap.add_argument("--set", default="primary")
    ap.add_argument("--rows", nargs=2, type=int, default=(16, 30)); ap.add_argument("--t-from"); ap.add_argument("--t-to"); ap.add_argument("--lon", nargs=2, type=float, default=None)
    ap.add_argument("--min-move", type=float, default=2.0); ap.add_argument("--out", required=True)
    ap.add_argument("--counterpart-phases", help="another reader output over the same interval (the other grid), whose pair at each changed row-step is tabulated beside the change")
    ap.add_argument("--counterpart-set", default=None, help="parameter set in the counterpart (default: the same as --set)")
    ap.add_argument("--counterpart-listing", help="the counterpart's listing, for the selection-offset counts")
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"REFUSED: {a.out} exists and a record is never overwritten")
    ph = json.load(open(a.phases)); rec = json.load(open(a.listing))
    assert ph["record"]["sha256"] == sha256_of(a.listing), "the phases output was not computed from this listing"
    res = ph["results"][a.set]; P = res["params"]
    pairs = res["pairs"]
    L = [f"# evolution ledger, {os.path.basename(a.phases)} (set {a.set}) over {os.path.basename(a.listing)}", "",
         f"phases sha256 {sha256_of(a.phases)[:16]}, listing sha256 {ph['record']['sha256'][:16]}, reader sha256 {ph['script_sha256'][:16]}, listing grid {rec['declared'].get('grid', 'integer')} ({rec['declared'].get('lon_spacing_deg', 1.0):g} deg)",
         f"parameters: depth {P['depth_ms']:g} m/s, pair bound {P['pair_bound_deg']:g} deg, lag {P['lag_steps'] * P['step_hours']} h, rate {P['rate_deg_per_day']:g} deg/day, long phase {P['long_phase_hours']} h, overlap {P['overlap_hours']} h", ""]
    L.append("## Pairs by class, by row group (all times, all longitudes)")
    L.append("| group | " + " | ".join(c.replace("continuous_", "") for c in CLASSES) + " | assessable |")
    L.append("| --- |" + " --- |" * (len(CLASSES) + 1))
    for lo, hi in GROUPS:
        pc = pair_counts(pairs, (lo, hi))
        n_ass = sum(v for c, v in pc["counts"].items() if c != "unassessable")
        L.append(f"| {lo} to {hi} N | " + " | ".join(str(pc["counts"][c]) for c in CLASSES) + f" | {n_ass} |")
    L.append("")
    dm = deep_minima_counts(rec, P["depth_ms"], tuple(a.rows), a.t_from, a.t_to)
    L.append(f"## Competing deep minima under the subset: {dm['predicate']}")
    L.append(f"assigned row-steps {dm['assigned']}, with two or more deep in-window minima {dm['two_or_more']}")
    L.append("")
    if a.counterpart_listing:
        orec = json.load(open(a.counterpart_listing))
        so = selection_offsets(rec, orec, tuple(a.rows), a.t_from, a.t_to)
        L.append(f"## Selection offsets against {os.path.basename(a.counterpart_listing)} (grid {orec['declared'].get('grid', 'integer')}), under: {so['predicate']}")
        L.append(f"row-steps both assign {so['both_assigned']}: same column {so['same']}, half a degree off {so['half_degree']}, farther {so['both_assigned'] - so['same'] - so['half_degree']} " + "(" + ", ".join(f"{k:g} deg x{v}" for k, v in sorted(so["farther"].items())) + ")")
        L.append("")
    lon_range = tuple(a.lon) if a.lon else None
    pc = pair_counts(pairs, tuple(a.rows), a.t_from, a.t_to, lon_range, ("jump", "replacement", "switching"))
    L.append(f"## Non-continuous assessable pairs under the subset: {pc['predicate']}")
    L.append(f"total {pc['n']}: " + ", ".join(f"{c} {v}" for c, v in pc["counts"].items()))
    L.append("")
    C = classify_changes(rec, pairs, P["depth_ms"], P["pair_bound_deg"], 3.0, a.min_move, tuple(a.rows), a.t_from, a.t_to, lon_range)
    L.append(f"## Selected-position changes of at least {a.min_move:g} degrees in that subset, classified by the deep minima at both ends (depth {P['depth_ms']:g} m/s, bound {P['pair_bound_deg']:g} deg, lone distance 3 deg)")
    kinds = ("lone-shift", "coexisting", "appeared", "lost", "other")
    L.append("| |move| | " + " | ".join(kinds) + " | total |"); L.append("| --- |" + " --- |" * (len(kinds) + 1))
    for mv in sorted({abs(c["move"]) for c in C}):
        sub = [c for c in C if abs(c["move"]) == mv]
        L.append(f"| {mv:g} | " + " | ".join(str(sum(1 for c in sub if c["kind"] == k)) for k in kinds) + f" | {len(sub)} |")
    L.append("| all | " + " | ".join(str(sum(1 for c in C if c["kind"] == k)) for k in kinds) + f" | {len(C)} |")
    L.append("")
    L.append("by row, class and kind:")
    for c in sorted(C, key=lambda c: (-c["row"], c["from"])):
        L.append(f"  {c['row']} N {c['from'][5:]} -> {c['to'][5:]} move {c['move']:+g} {c['class']} {c['kind']}: deep {c['deep_before']} -> {c['deep_after']}")
    L.append("")
    if a.counterpart_phases:
        oph = json.load(open(a.counterpart_phases)); oset = a.counterpart_set or a.set
        ores = oph["results"][oset]
        T = counterparts(C, ores["pairs"]); tab = counterpart_table(T)
        oclasses = [c for c in CLASSES if any(k[1] == c for k in tab)] + (["missing"] if any(k[1] == "missing" for k in tab) else [])
        L.append(f"## The same row-steps in {os.path.basename(a.counterpart_phases)} (set {oset}, reader sha256 {oph['script_sha256'][:16]}, listing sha256 {oph['record']['sha256'][:16]}): kind here by class there")
        L.append("| kind here | " + " | ".join(c.replace("continuous_", "") for c in oclasses) + " | total |"); L.append("| --- |" + " --- |" * (len(oclasses) + 1))
        for k in kinds:
            if any(t["kind"] == k for t in T):
                L.append(f"| {k} | " + " | ".join(str(tab.get((k, c), 0)) for c in oclasses) + f" | {sum(1 for t in T if t['kind'] == k)} |")
        L.append("| all | " + " | ".join(str(sum(v for kk, v in tab.items() if kk[1] == c)) for c in oclasses) + f" | {len(T)} |")
        L.append("")
        L.append("by row-step, this record then the other:")
        for t in sorted(T, key=lambda t: (-t["row"], t["from"])):
            om = "missing" if t["other_move"] is None else f"{t['other_lon_from']:g}->{t['other_lon_to']:g} move {t['other_move']:+g} {t['other_class']}"
            L.append(f"  {t['row']} N {t['from'][5:]}: {c_pos(t)} {t['kind']} | {om}")
        L.append("")
    L.append(f"## Westward phases of 48 elapsed hours or more (set {a.set}), and the largest adjacent-row set")
    for row in sorted(res["phases"], key=int, reverse=True):
        for p in res["phases"][row]:
            if p["direction"] == "west" and p["hours"] >= 48:
                L.append(f"  {row} N {p['start'][5:]}..{p['end'][5:]} {p['hours']} h ({p['n_steps']} samples) {p['lon_start']:g}->{p['lon_end']:g} rate {p['rate_deg_per_day']:+.2f}/day ambiguous {p['ambiguous_pairs']}")
    ds = res["displacement_set"]
    ov = (min(p["i_end"] for p in ds["phases"]) - max(p["i_start"] for p in ds["phases"])) * P["step_hours"] if ds["phases"] else None
    L.append(f"largest adjacent-row set: {ds['n_rows']} rows {ds['rows']}, common overlap {ov} h, ambiguous pairs in its phases {sum(p['ambiguous_pairs'] for p in ds['phases'])} of {sum(p['n_steps'] - 1 for p in ds['phases'])}")
    L.append("")
    L.append("## Displacement set under every parameter set")
    for name, r in ph["results"].items():
        d = r["displacement_set"]
        o = (min(p["i_end"] for p in d["phases"]) - max(p["i_start"] for p in d["phases"])) * r["params"]["step_hours"] if d["phases"] else None
        L.append(f"  {name}: {d['n_rows']} rows {d['rows']}, overlap {o} h")
    L.append("")
    L.append(f"## Retained standing intervals of 48 elapsed hours or more (set {a.set}), with the range of selections")
    for row in sorted(res["standing"], key=int, reverse=True):
        for s in res["standing"][row]:
            if s["hours"] >= 48:
                L.append(f"  {row} N {s['start'][5:]}..{s['end'][5:]} {s['hours']} h selections {s['lon_min']:g} to {s['lon_max']:g} E")
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(l for l in L if l.startswith("total") or l.startswith("| all") or l.startswith("largest") or l.startswith("  primary") or l.startswith("  bound")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
