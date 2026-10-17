#!/usr/bin/env python3
"""Displacement against switching, a declared classification of consecutive-step evolution
on an entry listing (`EASTERN_DISPLACEMENT_SWITCHING_2026-10-16.md`, section 3). Reads a
listing record of `pilot_i1_entry_listing.py` and computes nothing from the fields.

Per row and consecutive pair of steps the pair is UNASSESSABLE (either step not supported:
not assigned, selected minimum shallower than the depth threshold, or selected longitude
inside the margin of the window's western bound), CONTINUOUS (the selected longitude moves
by at most the pair bound), and otherwise SWITCHING (a deep in-window minimum persists
near the old position and one pre-existed near the new one), REPLACEMENT (exactly one of
those) or JUMP (neither). A RUN of consecutive continuous pairs is DISPLACEMENT when it
lasts at least the minimum run duration and its net rate is at or above the rate
threshold, STANDING when it lasts that long with a smaller rate, and SHORT otherwise. The
outcome criterion is the largest set of adjacent rows each holding a displacement run of
at least the long-run duration, pairwise overlapping by at least the overlap duration.
Positions, not features. Nothing here is a track, an identity or a physical statement.

    python3 scripts/pilot_entry_evolution_classes.py RECORD --out OUT.json --text OUT.txt [--extra-rows LO HI]
"""
import argparse
import hashlib
import json
import os

PRIMARY = {"depth_ms": -8.0, "pair_bound_deg": 1.5, "margin_deg": 2.0, "min_run_hours": 24, "rate_deg_per_day": 1.0,
           "long_run_hours": 48, "overlap_hours": 24, "adjacent_rows": 4, "step_hours": 6}
SENSITIVITIES = {"depth_-6": {"depth_ms": -6.0}, "bound_2.5": {"pair_bound_deg": 2.5}, "rate_0.5": {"rate_deg_per_day": 0.5}}
GROUPS = {"rows_25_30": (25, 30), "rows_19_24": (19, 24), "rows_12_18": (12, 18)}
PAIR_CLASSES = ("continuous", "switching", "replacement", "jump", "unassessable")


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _rows_of(record):
    return sorted({int(m["row_lat"]) for s in record["steps"] for m in s["rows"]}, reverse=True)


def _row_at(step, row):
    return next((m for m in step["rows"] if int(m["row_lat"]) == row), None)


def _bound(step):
    ann = step.get("annotation") or {}
    return (ann.get("i1") or {}).get("lon", [None])[0]


def supported(m, bound, P):
    """Assigned, selected minimum at or under the depth threshold at full precision, and at
    least the margin east of the window's western bound."""
    if m is None or m.get("status") != "assigned" or bound is None:
        return False
    return m["i1"]["v"] <= P["depth_ms"] and m["i1"]["lon"] >= bound + P["margin_deg"]


def deep_set(m, P):
    return [x["lon"] for x in m.get("wind_minima", []) if x["label"] == "I1" and x["v"] <= P["depth_ms"]]


def classify_pairs(record, P):
    """{row: [pair, ...]} in step order, one entry per consecutive pair of steps."""
    steps = record["steps"]
    out = {}
    for row in _rows_of(record):
        pairs = []
        for k in range(len(steps) - 1):
            s0, s1 = steps[k], steps[k + 1]
            m0, m1 = _row_at(s0, row), _row_at(s1, row)
            pair = {"row": row, "k": k, "from": s0["time"], "to": s1["time"], "lon_from": None, "lon_to": None, "move": None, "class": None, "detail": ""}
            if not (supported(m0, _bound(s0), P) and supported(m1, _bound(s1), P)):
                pair["class"] = "unassessable"
                pairs.append(pair)
                continue
            L0, L1 = m0["i1"]["lon"], m1["i1"]["lon"]
            pair.update({"lon_from": L0, "lon_to": L1, "move": L1 - L0})
            if abs(L1 - L0) <= P["pair_bound_deg"]:
                pair["class"] = "continuous"
            else:
                near = P["pair_bound_deg"]
                old_persists = any(abs(x - L0) <= near for x in deep_set(m1, P))
                new_preexisted = any(abs(x - L1) <= near for x in deep_set(m0, P))
                if old_persists and new_preexisted:
                    pair["class"] = "switching"
                    pair["detail"] = f"deep minima at {sorted(deep_set(m0, P))} then {sorted(deep_set(m1, P))}"
                elif old_persists or new_preexisted:
                    pair["class"] = "replacement"
                    pair["detail"] = "new-appeared" if old_persists else "old-lost"
                else:
                    pair["class"] = "jump"
            pairs.append(pair)
        out[row] = pairs
    return out


def runs(record, P, pairs_by_row=None):
    """{row: [run, ...]}, each run a maximal sequence of consecutive continuous pairs."""
    pairs_by_row = pairs_by_row or classify_pairs(record, P)
    steps = record["steps"]
    out = {}
    for row, pairs in pairs_by_row.items():
        row_runs = []
        k = 0
        while k < len(pairs):
            if pairs[k]["class"] != "continuous":
                k += 1
                continue
            j = k
            while j + 1 < len(pairs) and pairs[j + 1]["class"] == "continuous":
                j += 1
            n = j - k + 1
            hours = n * P["step_hours"]
            lon_start, lon_end = pairs[k]["lon_from"], pairs[j]["lon_to"]
            rate = (lon_end - lon_start) / (hours / 24.0)
            cls = "short" if hours < P["min_run_hours"] else ("displacement" if abs(rate) >= P["rate_deg_per_day"] else "standing")
            row_runs.append({"row": row, "i_start": pairs[k]["k"], "i_end": pairs[j]["k"] + 1, "start": steps[pairs[k]["k"]]["time"], "end": steps[pairs[j]["k"] + 1]["time"],
                             "pairs": n, "hours": hours, "lon_start": lon_start, "lon_end": lon_end, "net_move": lon_end - lon_start, "rate_deg_per_day": rate, "class": cls})
            k = j + 1
        out[row] = row_runs
    return out


def displacement_set(record, P, runs_by_row=None):
    """The largest set of adjacent rows each holding a displacement run of at least the
    long-run duration, with every pair of chosen runs overlapping by at least the overlap
    duration. Returns rows (descending), the chosen runs and whether the criterion is met."""
    runs_by_row = runs_by_row or runs(record, P)
    need = P["overlap_hours"] / P["step_hours"]
    long = {row: [r for r in rr if r["class"] == "displacement" and r["hours"] >= P["long_run_hours"]] for row, rr in runs_by_row.items()}
    best = {"rows": [], "n_rows": 0, "runs": [], "meets_criterion": False}

    def overlap(a, b):
        return min(a["i_end"], b["i_end"]) - max(a["i_start"], b["i_start"])

    def extend(row, chosen):
        nonlocal best
        if len(chosen) > best["n_rows"]:
            best = {"rows": [r for r, _ in chosen], "n_rows": len(chosen), "runs": [x for _, x in chosen], "meets_criterion": len(chosen) >= P["adjacent_rows"]}
        for run in long.get(row - 1, []):
            if all(overlap(run, x) >= need for _, x in chosen):
                extend(row - 1, chosen + [(row - 1, run)])

    for row in sorted(long, reverse=True):
        for run in long[row]:
            extend(row, [(row, run)])
    return best


def summarize(record, P, extra=None):
    pairs_by_row = classify_pairs(record, P)
    runs_by_row = runs(record, P, pairs_by_row)
    groups = dict(GROUPS)
    if extra:
        groups[f"rows_{extra[0]}_{extra[1]}"] = (extra[0], extra[1])
    summary = {}
    for name, (lo, hi) in groups.items():
        rows = [r for r in pairs_by_row if lo <= r <= hi]
        counts = {c: sum(1 for r in rows for p in pairs_by_row[r] if p["class"] == c) for c in PAIR_CLASSES}
        covered = {c: 0 for c in ("displacement", "standing", "short")}
        for r in rows:
            for run in runs_by_row[r]:
                covered[run["class"]] += run["pairs"] + 1
        n_supported = sum(1 for r in rows for s in record["steps"] if supported(_row_at(s, r), _bound(s), P))
        summary[name] = {"rows": [lo, hi], "pairs_by_class": counts, "supported_row_steps": n_supported, "row_steps_in_runs": covered}
    return {"params": P, "pairs_by_row": pairs_by_row, "runs_by_row": runs_by_row, "groups": summary, "displacement_set": displacement_set(record, P, runs_by_row)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("record"); ap.add_argument("--out", required=True); ap.add_argument("--text", required=True)
    ap.add_argument("--extra-rows", nargs=2, type=int, default=None, metavar=("LO", "HI"))
    a = ap.parse_args(argv)
    for p in (a.out, a.text):
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a record is never overwritten")
    record = json.load(open(a.record))
    sets = {"primary": dict(PRIMARY)}
    for name, change in SENSITIVITIES.items():
        sets[name] = {**PRIMARY, **change}
    results = {name: summarize(record, P, a.extra_rows) for name, P in sets.items()}
    n_e = sum(1 for s in record["steps"] for m in s["rows"] for x in m.get("wind_minima", []) if x["label"] == "E")
    out = {"generated_by": "scripts/pilot_entry_evolution_classes.py", "script_sha256": sha256_of(os.path.abspath(__file__)),
           "record": {"path": a.record, "sha256": sha256_of(a.record)}, "excluded_E_labeled_minima": n_e, "declared": "EASTERN_DISPLACEMENT_SWITCHING_2026-10-16.md, section 3",
           "label": "positions, not features; a classification of selected-position evolution under declared thresholds, not a track and not an identity", "results": results}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    lines = [f"# displacement against switching on {a.record} ({len(record['steps'])} steps); E-labeled minima excluded from the in-window set: {n_e}"]
    for name, res in results.items():
        P = res["params"]
        lines.append(f"\n## parameter set {name}: depth {P['depth_ms']:g} m/s, pair bound {P['pair_bound_deg']:g} deg, rate {P['rate_deg_per_day']:g} deg/day, min run {P['min_run_hours']} h, long run {P['long_run_hours']} h")
        for g, v in res["groups"].items():
            c = v["pairs_by_class"]; k = v["row_steps_in_runs"]
            lines.append(f"{g} ({v['rows'][0]}-{v['rows'][1]} N): pairs continuous {c['continuous']}, switching {c['switching']}, replacement {c['replacement']}, jump {c['jump']}, unassessable {c['unassessable']}; "
                         f"supported row-steps {v['supported_row_steps']}, in displacement runs {k['displacement']}, standing runs {k['standing']}, short runs {k['short']}")
        d = res["displacement_set"]
        lines.append(f"largest adjacent-row displacement set: {d['n_rows']} rows {d['rows']} meets criterion {d['meets_criterion']}" + ("" if not d["runs"] else "; runs " + "; ".join(f"{r['row']} N {r['start'][5:]}..{r['end'][5:]} {r['lon_start']:g}->{r['lon_end']:g} ({r['rate_deg_per_day']:+.1f}/day)" for r in d["runs"])))
        lines.append("runs of at least the minimum duration:")
        for row in sorted(res["runs_by_row"], reverse=True):
            for r in res["runs_by_row"][row]:
                if r["class"] != "short":
                    lines.append(f"  {row} N {r['start'][5:]}..{r['end'][5:]} {r['hours']} h {r['lon_start']:g}->{r['lon_end']:g} net {r['net_move']:+g} rate {r['rate_deg_per_day']:+.2f}/day {r['class']}")
        lines.append("switching pairs:")
        for row in sorted(res["pairs_by_row"], reverse=True):
            for p in res["pairs_by_row"][row]:
                if p["class"] == "switching":
                    lines.append(f"  {row} N {p['from'][5:]}->{p['to'][5:]} {p['lon_from']:g}->{p['lon_to']:g}: {p['detail']}")
    with open(a.text, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(l for l in lines if l.startswith("largest") or l.startswith("## ")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
