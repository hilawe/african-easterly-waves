#!/usr/bin/env python3
"""The developed evolution reader of the eleventh direction effort
(`EASTERN_DIRECTION_RECOMMENDATION_11_2026-10-16.md`, stage A). It reads an entry listing
record (`pilot_i1_entry_listing.py`) and computes nothing from the fields.

Per row and consecutive six-hourly pair: UNASSESSABLE when either step is not supported
(assigned, selected minimum at or under the depth threshold at full precision, selected
longitude at least the margin inside the window's western bound); CONTINUOUS_UNIQUE when
the selected position changes by at most the pair bound and no other deep in-window
minimum lies within the pair bound of the earlier selection at the later step or of the
later selection at the earlier step; CONTINUOUS_AMBIGUOUS when it changes by at most the
bound but such an alternative exists; otherwise SWITCHING (a deep minimum persists near
the old position and one pre-existed near the new), REPLACEMENT (exactly one of those) or
JUMP (neither). A CHAIN is a maximal sequence of continuous pairs. Along a chain each
step is classed by the selected position's change over a centered window of the declared
lag (clamped to the chain, with the rate threshold scaled to the window's length): WEST,
EAST or STANDING. A PHASE is a maximal sequence of steps of one moving direction, with its
span, net move, rate and the number of ambiguous pairs inside it. A STANDING interval is a
maximal sequence of standing steps. The adjacent-row displacement set (rows with westward
phases of at least the long-phase duration overlapping by at least the overlap duration)
is an operational statistic with no product consequence attached. Positions, not
features. Nothing here is a track, an identity or a physical statement.

    python3 scripts/pilot_entry_evolution_phases.py RECORD --out OUT.json --text OUT.txt
"""
import argparse
import hashlib
import json
import os

# The lag was 4 steps (a 24-hour centered window) at the end of stage A. It was changed to 8
# steps (48 hours) in stage B after the I2 development case's northern rows, which stand at
# 48 to 50 E for four days, were read as alternating 24-hour westward and eastward phases:
# with a 24-hour window and a 1 degree per day threshold, one grid column of change over the
# window meets the threshold, so a column shift held for days reads as movement. The harness
# challenge A8_8 reproduces that failure at lag 4 and passes at lag 8, under which the
# threshold is two columns over the window. The 24-hour lag is kept as the sensitivity.
# Development cases informed this change, so passing them is not validation.
PRIMARY = {"depth_ms": -8.0, "pair_bound_deg": 1.5, "margin_deg": 2.0, "lag_steps": 8, "rate_deg_per_day": 1.0,
           "min_standing_hours": 24, "long_phase_hours": 48, "overlap_hours": 24, "adjacent_rows": 4, "step_hours": 6}
SENSITIVITIES = {"depth_-6": {"depth_ms": -6.0}, "bound_2.5": {"pair_bound_deg": 2.5}, "rate_0.5": {"rate_deg_per_day": 0.5}, "lag_4": {"lag_steps": 4}}
GROUPS = {"rows_25_30": (25, 30), "rows_19_24": (19, 24), "rows_12_18": (12, 18)}
PAIR_CLASSES = ("continuous_unique", "continuous_ambiguous", "switching", "replacement", "jump", "unassessable")


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
    if m is None or m.get("status") != "assigned" or bound is None:
        return False
    return m["i1"]["v"] <= P["depth_ms"] and m["i1"]["lon"] >= bound + P["margin_deg"]


def deep_set(m, P):
    return [x["lon"] for x in m.get("wind_minima", []) if x["label"] == "I1" and x["v"] <= P["depth_ms"]]


def classify_pairs(record, P):
    """{row: [pair, ...]} in step order."""
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
            near = P["pair_bound_deg"]
            if abs(L1 - L0) <= near:
                later_alt = [x for x in deep_set(m1, P) if x != L1 and abs(x - L0) <= near]
                earlier_alt = [x for x in deep_set(m0, P) if x != L0 and abs(x - L1) <= near]
                if later_alt or earlier_alt:
                    pair["class"] = "continuous_ambiguous"
                    pair["detail"] = f"alternatives within the bound, later {sorted(later_alt)}, earlier {sorted(earlier_alt)}"
                else:
                    pair["class"] = "continuous_unique"
            else:
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


def chains(record, P, pairs_by_row):
    """{row: [chain, ...]}, each a maximal sequence of continuous pairs with the selected
    longitude at every step of the chain."""
    steps = record["steps"]
    out = {}
    for row, pairs in pairs_by_row.items():
        row_chains = []
        k = 0
        while k < len(pairs):
            if not pairs[k]["class"].startswith("continuous"):
                k += 1
                continue
            j = k
            while j + 1 < len(pairs) and pairs[j + 1]["class"].startswith("continuous"):
                j += 1
            lons = [pairs[i]["lon_from"] for i in range(k, j + 1)] + [pairs[j]["lon_to"]]
            row_chains.append({"row": row, "i_start": pairs[k]["k"], "i_end": pairs[j]["k"] + 1, "start": steps[pairs[k]["k"]]["time"], "end": steps[pairs[j]["k"] + 1]["time"],
                               "pairs": j - k + 1, "lons": lons, "classes": [pairs[i]["class"] for i in range(k, j + 1)]})
            k = j + 1
        out[row] = row_chains
    return out


def step_motion(chain, P):
    """Per step of the chain, WEST, EAST or STANDING by the selected position's change over
    a centered window of the lag, clamped to the chain, with the rate threshold scaled to
    the window's length in days."""
    half = P["lag_steps"] // 2
    n = len(chain["lons"])
    out = []
    for i in range(n):
        a, b = max(0, i - half), min(n - 1, i + half)
        if b <= a:
            out.append("standing")
            continue
        hours = (b - a) * P["step_hours"]
        d = chain["lons"][b] - chain["lons"][a]
        thr = P["rate_deg_per_day"] * hours / 24.0
        out.append("west" if d <= -thr else "east" if d >= thr else "standing")
    return out


def phases_and_standing(record, P, chains_by_row):
    """Phases, retained standing intervals (at least the minimum standing duration, elapsed)
    and the per-step motion labels of every chain, which the adequacy harness reads so that
    standing between switches shorter than the minimum duration is still checkable."""
    steps = record["steps"]
    phases, standing, motion_by_row = {}, {}, {}
    for row, row_chains in chains_by_row.items():
        ph, st, labels = [], [], []
        for ch in row_chains:
            motion = step_motion(ch, P)
            labels.extend((ch["i_start"] + k, m) for k, m in enumerate(motion))
            i = 0
            while i < len(motion):
                j = i
                while j + 1 < len(motion) and motion[j + 1] == motion[i]:
                    j += 1
                i0, i1 = ch["i_start"] + i, ch["i_start"] + j
                # Elapsed time between the first and last sample of the stretch. Until the correction
                # of 2026-10-16 (section 9 of the development record) this counted samples, (j - i + 1)
                # steps, so every span carried six hours too many, a one-sample stretch had six hours
                # and a rate, and the 48-hour filters admitted 42-hour spans. n_steps keeps the sample
                # count apart from the elapsed duration.
                hours = (j - i) * P["step_hours"]
                lons = ch["lons"][i:j + 1]
                if motion[i] == "standing":
                    st.append({"row": row, "i_start": i0, "i_end": i1, "start": steps[i0]["time"], "end": steps[i1]["time"], "hours": hours, "n_steps": j - i + 1,
                               "lon": lons[0], "lon_start": lons[0], "lon_end": lons[-1], "lon_min": min(lons), "lon_max": max(lons)})
                else:
                    net = lons[-1] - lons[0]
                    amb = sum(1 for c in ch["classes"][i:j] if c == "continuous_ambiguous")
                    ph.append({"row": row, "direction": motion[i], "i_start": i0, "i_end": i1, "start": steps[i0]["time"], "end": steps[i1]["time"], "hours": hours,
                               "n_steps": j - i + 1, "lon_start": lons[0], "lon_end": lons[-1], "lon_min": min(lons), "lon_max": max(lons), "net_move": net,
                               "rate_deg_per_day": (net / (hours / 24.0)) if hours > 0 else None, "ambiguous_pairs": amb})
                i = j + 1
        phases[row] = ph
        standing[row] = [s for s in st if s["hours"] >= P["min_standing_hours"]]
        motion_by_row[row] = labels
    return phases, standing, motion_by_row


def displacement_set(phases_by_row, P):
    """The largest set of adjacent rows each holding a westward phase of at least the
    long-phase duration, pairwise overlapping by at least the overlap duration. An
    operational statistic with no product consequence."""
    need = P["overlap_hours"] / P["step_hours"]
    long = {row: [p for p in pp if p["direction"] == "west" and p["hours"] >= P["long_phase_hours"]] for row, pp in phases_by_row.items()}
    best = {"rows": [], "n_rows": 0, "phases": [], "meets_criterion": False}

    def overlap(a, b):
        return min(a["i_end"], b["i_end"]) - max(a["i_start"], b["i_start"])

    def extend(row, chosen):
        nonlocal best
        if len(chosen) > best["n_rows"]:
            best = {"rows": [r for r, _ in chosen], "n_rows": len(chosen), "phases": [x for _, x in chosen], "meets_criterion": len(chosen) >= P["adjacent_rows"]}
        for p in long.get(row - 1, []):
            if all(overlap(p, x) >= need for _, x in chosen):
                extend(row - 1, chosen + [(row - 1, p)])

    for row in sorted(long, reverse=True):
        for p in long[row]:
            extend(row, [(row, p)])
    return best


def analyze(record, P):
    pairs_by_row = classify_pairs(record, P)
    chains_by_row = chains(record, P, pairs_by_row)
    phases, standing, motion_by_row = phases_and_standing(record, P, chains_by_row)
    groups = {}
    for name, (lo, hi) in GROUPS.items():
        rows = [r for r in pairs_by_row if lo <= r <= hi]
        groups[name] = {"rows": [lo, hi], "pairs_by_class": {c: sum(1 for r in rows for p in pairs_by_row[r] if p["class"] == c) for c in PAIR_CLASSES},
                        "supported_row_steps": sum(1 for r in rows for s in record["steps"] if supported(_row_at(s, r), _bound(s), P)),
                        "row_steps_in_west_phases": sum(p["n_steps"] for r in rows for p in phases[r] if p["direction"] == "west"),
                        "row_steps_in_east_phases": sum(p["n_steps"] for r in rows for p in phases[r] if p["direction"] == "east"),
                        "row_steps_in_standing": sum((s["i_end"] - s["i_start"] + 1) for r in rows for s in standing[r])}
    return {"params": P, "pairs": pairs_by_row, "chains": chains_by_row, "phases": phases, "standing": standing, "motion": motion_by_row, "groups": groups,
            "displacement_set": displacement_set(phases, P)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("record"); ap.add_argument("--out", required=True); ap.add_argument("--text", required=True)
    a = ap.parse_args(argv)
    for p in (a.out, a.text):
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a record is never overwritten")
    record = json.load(open(a.record))
    sets = {"primary": dict(PRIMARY)}
    for name, change in SENSITIVITIES.items():
        sets[name] = {**PRIMARY, **change}
    results = {name: analyze(record, P) for name, P in sets.items()}
    out = {"generated_by": "scripts/pilot_entry_evolution_phases.py", "script_sha256": sha256_of(os.path.abspath(__file__)), "record": {"path": a.record, "sha256": sha256_of(a.record)},
           "declared": "EASTERN_DIRECTION_RECOMMENDATION_11_2026-10-16.md, stage A", "label": "positions, not features; development evidence, not validation", "results": results}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    lines = [f"# evolution phases on {a.record} ({len(record['steps'])} steps)"]
    for name, res in results.items():
        P = res["params"]
        lines.append(f"\n## parameter set {name}: depth {P['depth_ms']:g} m/s, pair bound {P['pair_bound_deg']:g} deg, lag {P['lag_steps'] * P['step_hours']} h, rate {P['rate_deg_per_day']:g} deg/day")
        for g, v in res["groups"].items():
            c = v["pairs_by_class"]
            lines.append(f"{g}: pairs unique {c['continuous_unique']}, ambiguous {c['continuous_ambiguous']}, switching {c['switching']}, replacement {c['replacement']}, jump {c['jump']}, unassessable {c['unassessable']}; "
                         f"supported row-steps {v['supported_row_steps']}, in west phases {v['row_steps_in_west_phases']}, east phases {v['row_steps_in_east_phases']}, standing {v['row_steps_in_standing']}")
        d = res["displacement_set"]
        lines.append(f"largest adjacent-row set of westward phases >= {P['long_phase_hours']} h: {d['n_rows']} rows {d['rows']}" + ("" if not d["phases"] else "; " + "; ".join(f"{p['row']} N {p['start'][5:]}..{p['end'][5:]} {p['lon_start']:g}->{p['lon_end']:g}" for p in d["phases"])))
        lines.append("westward phases of 24 elapsed hours or more (hours are elapsed between the first and last sample):")
        for row in sorted(res["phases"], reverse=True):
            for p in res["phases"][row]:
                if p["direction"] == "west" and p["hours"] >= 24:
                    lines.append(f"  {row} N {p['start'][5:]}..{p['end'][5:]} {p['hours']} h ({p['n_steps']} samples) {p['lon_start']:g}->{p['lon_end']:g} net {p['net_move']:+g} rate {p['rate_deg_per_day']:+.2f}/day ambiguous pairs {p['ambiguous_pairs']}")
        lines.append("standing intervals of 48 elapsed hours or more (selected longitudes vary within the interval, the range is given):")
        for row in sorted(res["standing"], reverse=True):
            for s in res["standing"][row]:
                if s["hours"] >= 48:
                    lines.append(f"  {row} N {s['start'][5:]}..{s['end'][5:]} {s['hours']} h ({s['n_steps']} samples) selections {s['lon_min']:g} to {s['lon_max']:g} E")
        lines.append("switching pairs:")
        for row in sorted(res["pairs"], reverse=True):
            for p in res["pairs"][row]:
                if p["class"] == "switching":
                    lines.append(f"  {row} N {p['from'][5:]}->{p['to'][5:]} {p['lon_from']:g}->{p['lon_to']:g}: {p['detail']}")
    with open(a.text, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(l for l in lines if l.startswith("largest") or l.startswith("## ")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
