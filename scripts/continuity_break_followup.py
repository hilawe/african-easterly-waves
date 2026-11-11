#!/usr/bin/env python3
"""The follow-up to the break reading of the continuity effort, declared in section 29 of
`CONTINUITY_ACROSS_AFRICA_2026-11-09.md` after section 28 read the declared outcome and before
any of the three measures below was computed.

It reads the committed season records of section 27's reading, which must have been written by
the unchanged break-reading script, and replays nothing.

THE POSITION TEST replaces the sign test of section 27, whose two outcomes are not equally likely
when the two segments differ in length. Within each reference, its breaks are taken as equally
likely at any of its counted transitions, and the number of coastal breaks over all references is
compared with its exact distribution under that condition, each reference keeping its own counts.
The expected numbers of references with the higher share in each segment under the same condition
are given beside the observed ones, which is what shows the sign test's defect on these references.
The counts of coastal breaks from which each outcome would be read were added after the first run,
as a statement of what the test could have shown, and they change no outcome.

THE PRIMARY'S EXIT gives, for each coast-reaching reference whose primary history lacks the end
without sustained carriage, what the primary itself did at its last accompanied observation and
what became of the reference there.

THE RETURN gives, for each break with a holder that continued away, the time until a holder that
went away is beside the reference again.

Three tabulations of fields the season records already hold are added. Each break is grouped by
its main holder's attempts (the holder took the candidate nearest the reference, a candidate within
500 km was not taken, or no candidate lay within 500 km at an attempt read), the distance of the
nearest final candidate is summarized over the breaks classed as placed away, and the attempts with
a candidate outside the search area are counted by pass with the polygon's reading.

Everything is relative to QTrack and to two development seasons, and nothing here changes a
tracker setting or an earlier record.

    python3 scripts/continuity_break_followup.py --out-dir DIR
"""

import argparse
import gzip
import hashlib
import json
import os
import sys
from fractions import Fraction
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import continuity_break_reading as B  # noqa: E402

C, E, L, P = B.C, B.E, B.L, B.P

with open(os.path.abspath(__file__), "rb") as _fh:
    SCRIPT_SHA256_AT_LOAD = hashlib.sha256(_fh.read()).hexdigest()

STEM = "continuity_break_followup"
RETURN_HOURS = 24                                    # section 25's long gap begins above this
OUTCOMES = B.OUTCOMES
PRIMARY_DID = ("ended", "away", "past_the_window")
REFERENCE_THEN = ("kept_by_another_history", "unresolved") + B.NEXT
RETURNS = ("within_24_hours", "later", "never")
GROUPS = ("the_holder_took_the_nearest_candidate", "a_candidate_near_was_not_taken", "no_candidate_near_at_an_attempt_read")
PASSES = ("6hr", "12hr")


# --- the position test ------------------------------------------------------------------------------

def _range(nc, ne, b):
    return range(max(0, b - ne), min(nc, b) + 1)


def exact_positions(rows):
    """The exact distribution of the number of coastal breaks when each reference's breaks are
    equally likely at any of its counted transitions. `rows` are `per_reference` rows of the break
    reading. Returns the number of arrangements by total and the number of all arrangements."""
    dist, den = {0: 1}, 1
    for r in rows:
        (bc, nc), (be, ne) = r["coastal"], r["earlier"]
        b, new = bc + be, {}
        for total, w in dist.items():
            for x in _range(nc, ne, b):
                new[total + x] = new.get(total + x, 0) + w * comb(nc, x) * comb(ne, b - x)
        dist, den = new, den * comb(nc + ne, b)
    if sum(dist.values()) != den:
        raise SystemExit("REFUSED: the arrangements of the position test do not add up")
    return dist, den


def position_test(rows):
    """Section 29's reading, decided in whole numbers: concentrated near the coast when the
    observed coastal breaks or more have a probability of 0.05 or less, concentrated earlier when
    the observed or fewer have, cannot be scored when neither tail can reach 0.05."""
    dist, den = exact_positions(rows)
    observed = sum(r["coastal"][0] for r in rows)
    upper = sum(w for total, w in dist.items() if total >= observed)
    lower = sum(w for total, w in dist.items() if total <= observed)
    if not (P.at_most_alpha((dist[max(dist)], den)) or P.at_most_alpha((dist[min(dist)], den))):
        outcome = OUTCOMES[2]
    elif P.at_most_alpha((upper, den)):
        outcome = OUTCOMES[0]
    elif P.at_most_alpha((lower, den)):
        outcome = OUTCOMES[1]
    else:
        outcome = OUTCOMES[3]
    expected = sum(Fraction((r["coastal"][0] + r["earlier"][0]) * r["coastal"][1], r["coastal"][1] + r["earlier"][1]) for r in rows if r["coastal"][1] + r["earlier"][1])
    near = [c for c in dist if P.at_most_alpha((sum(w for total, w in dist.items() if total >= c), den))]
    early = [c for c in dist if P.at_most_alpha((sum(w for total, w in dist.items() if total <= c), den))]
    return {"coastal_breaks": observed, "references_with_a_coastal_break": sum(1 for r in rows if r["coastal"][0]), "expected_coastal_breaks": round(float(expected), 2),
            "least": min(dist), "most": max(dist), "at_least_observed": round(upper / den, 4), "at_most_observed": round(lower / den, 4),
            "near_the_coast_from": min(near, default=None), "earlier_up_to": max(early, default=None), "outcome": outcome}


def sign_null(rows):
    """Under the same condition, the expected numbers of scored references with the higher break
    share in the coastal segment, in the earlier one, and with equal shares."""
    plus = minus = ties = Fraction(0)
    for r in rows:
        (bc, nc), (be, ne) = r["coastal"], r["earlier"]
        if nc == 0 or ne == 0:
            continue
        b = bc + be
        for x in _range(nc, ne, b):
            w = Fraction(comb(nc, x) * comb(ne, b - x), comb(nc + ne, b))
            if x * ne > (b - x) * nc:
                plus += w
            elif x * ne < (b - x) * nc:
                minus += w
            else:
                ties += w
    return {"higher_share_coastal": round(float(plus), 2), "higher_share_earlier": round(float(minus), 2), "equal_shares": round(float(ties), 2)}


# --- the primary's exit and the return -----------------------------------------------------------------

def primary_exit(ref):
    """For a coast-reaching reference whose primary lacks the end without sustained carriage: what
    the primary did at its last accompanied observation and what became of the reference there.
    None for every other reference."""
    p = ref["primary"]
    if not ref["coast_reaching"] or p is None or ref["sustained"] or p["end"]:
        return None
    t = next((t for t in ref["transitions"] if t["k"] == p["last"]), None)
    if t is None or p["history"] not in t["holders"]:
        raise SystemExit(f"REFUSED: reference {ref['id']} has no transition at its primary's last accompanied observation")
    h = t["holders"][p["history"]]
    out = {"id": ref["id"], "stop": p["stop"], "primary_did": h["outcome"],
           "reference_then": REFERENCE_THEN[0] if t["class"] == "held" else (REFERENCE_THEN[1] if t["class"] == "unresolved" else t["next"])}
    if h["outcome"] == "beside":
        raise SystemExit(f"REFUSED: the primary of reference {ref['id']} is beside it after its last accompanied observation")
    if h["outcome"] == "away":
        out.update({"km": h["km"], "marginal": h["marginal"], "by_smoothing_alone": h["by_smoothing_alone"]})
    if t["class"] == "break" and t["main_holder"] == p["history"]:
        out["attempts"] = [a["class"] for a in t["attempts"]]
    return out


def return_of(t, by_history):
    """For a break with a holder that continued away: the hours until a holder that went away
    accompanies the reference again, and its class. None for a break whose holders all ended."""
    away = [h for h, v in t["holders"].items() if v["outcome"] == "away"]
    if not away:
        return None
    later = [k for h in away for k in by_history[h] if k > t["k"]]
    if not later:
        return {"hours": None, "return": RETURNS[2]}
    hours = 6 * (min(later) - t["k"])
    return {"hours": hours, "return": RETURNS[0] if hours <= RETURN_HOURS else RETURNS[1]}


def _inflated_sizes():
    """The area the port's inflation produces from each pass's minimum, the minimum times the
    square root of 8/9, less a hundredth for the rounding of the stored areas."""
    from aew.v1port import association as ASSOC
    return {"6hr": ASSOC.POLY_AREA_6HR * (8.0 / 9.0) ** 0.5 - 0.01, "12hr": ASSOC.POLY_AREA_12HR * (8.0 / 9.0) ** 0.5 - 0.01}


INFLATED = _inflated_sizes()


def group_of(t):
    """The group of a break by its main holder's attempts."""
    classes = [a["class"] for a in t["attempts"]]
    if B.ATTEMPTS[4] in classes:
        return GROUPS[0]
    return GROUPS[1] if any(c != B.ATTEMPTS[0] for c in classes) else GROUPS[2]


def check_transitions(fresh, recorded, name):
    """The references recomputed from the lifecycle rows must be those the season record holds."""
    if [r["id"] for r in fresh] != [r["id"] for r in recorded]:
        raise SystemExit(f"REFUSED: the {name} references recomputed from the lifecycle rows are not the season record's")
    for a, b in zip(fresh, recorded):
        if len(a["transitions"]) != len(b["transitions"]) or any(t2.get(k) != v for t, t2 in zip(a["transitions"], b["transitions"]) for k, v in t.items()):
            raise SystemExit(f"REFUSED: the {name} transitions of reference {a['id']} recomputed from the lifecycle rows are not the season record's")


def config_followup(refs, by_history):
    """The exits and the returns of one configuration of one season."""
    exits = [x for x in (primary_exit(r) for r in refs) if x is not None]
    returns, groups, placed, outside = [], [], [], []
    for r in refs:
        if not r["coast_reaching"]:
            continue
        for t in r["transitions"]:
            if t["class"] == "break":
                groups.append({"id": r["id"], "k": t["k"], "segment": t["segment"], "group": group_of(t), "by_smoothing_alone": bool(t.get("by_smoothing_alone"))})
                if t["next"] == "placed_away" and t["final_km"] is not None:
                    placed.append(t["final_km"])
                outside += [{"id": r["id"], "k": t["k"], **{k: a[k] for k in ("pass", "candidate_km", "prediction_km", "polygon_area", "polygon_at_or_below_minimum", "inside_if_inflated")}}
                            for a in t["attempts"] if a["class"] == B.ATTEMPTS[1]]
                got = return_of(t, by_history[r["id"]])
                if got is not None:
                    returns.append({"id": r["id"], "k": t["k"], "segment": t["segment"], "next": t["next"], "by_smoothing_alone": bool(t.get("by_smoothing_alone")),
                                    "marginal": bool(t.get("marginal")), **got})
    return {"exits": exits, "returns": returns, "groups": groups, "placed_away_final_km": placed, "outside": outside, "per_reference": B.per_reference(refs),
            "breaks": len(groups)}


def season_followup(year, artifacts=B.ARTIFACTS):
    import qtrack_pairing_pilot as M
    path = os.path.join(artifacts, f"{B.STEM}_{year}.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rec = json.load(fh)
    if rec["generated_by"]["sha256"] != B.SCRIPT_SHA256_AT_LOAD or rec["year"] != year:
        raise SystemExit(f"REFUSED: the {year} break-reading record was not written by the current break-reading script")
    life, place = B.load_records(year, artifacts)
    art = json.load(open(place["artifact"]["path"], encoding="utf-8"))
    if C.sha256_of(art["inputs"]["qtrack"]["path"]) != art["inputs"]["qtrack"]["sha256"]:
        raise SystemExit(f"REFUSED: {art['inputs']['qtrack']['path']} is not the file the {year} artifact names")
    q = {t["id"]: t for t in M.load_qtrack(art["inputs"]["qtrack"]["path"])[0]}
    out = {}
    for name in B.CONFIGS:
        stored = L.history_sets(life["configs"][name]["rows"])["STORED"]
        base = E.reference_measures(stored, q, art["sides"]["qtrack"]["tracks"])
        fresh = B.reference_reading(base, place["configs"][name]["rows"], q, {h["id"]: h for h in stored})
        refs = rec["configs"][name]["references"]
        check_transitions(fresh, refs, f"{year} {name}")
        out[name] = config_followup(refs, {r["id"]: r["by_history"] for r in base})
    return {"year": year, "record_sha256": C.sha256_of(path), "configs": out}


# --- pooling and tables ----------------------------------------------------------------------------------

def pooled(parts):
    """The tables of one configuration from its seasons' follow-ups."""
    rows = [r for p in parts for r in p["per_reference"]]
    exits = [x for p in parts for x in p["exits"]]
    returns = [x for p in parts for x in p["returns"]]
    groups = [x for p in parts for x in p["groups"]]
    placed = sorted(x for p in parts for x in p["placed_away_final_km"])
    outside = [x for p in parts for x in p["outside"]]
    declared = B.concentration(rows)
    return {"declared_sign_test": {k: declared[k] for k in ("scored", "higher_share_coastal", "higher_share_earlier", "equal_shares", "sign_test_value", "outcome")},
            "expected_under_equal_positions": sign_null(rows), "position_test": position_test(rows),
            "exits": {"count": len(exits),
                      "primary_did": {d: sum(1 for x in exits if x["primary_did"] == d) for d in PRIMARY_DID},
                      "reference_then": {c: sum(1 for x in exits if x["reference_then"] == c) for c in REFERENCE_THEN},
                      "by_stop": {s: {d: sum(1 for x in exits if x["stop"] == s and x["primary_did"] == d) for d in PRIMARY_DID} for s in B.STOPS},
                      "away_marginal": sum(1 for x in exits if x.get("marginal")), "away_by_smoothing_alone": sum(1 for x in exits if x.get("by_smoothing_alone"))},
            "groups": {"all": {g: sum(1 for x in groups if x["group"] == g) for g in GROUPS},
                       **{s: {g: sum(1 for x in groups if x["segment"] == s and x["group"] == g) for g in GROUPS} for s in B.SEGMENTS},
                       "by_smoothing_alone": {g: sum(1 for x in groups if x["group"] == g and x["by_smoothing_alone"]) for g in GROUPS}},
            "placed_away": {"count": len(placed), "within_750_km": sum(1 for x in placed if x <= B.MARGINAL_KM), "nearest_km": placed[0] if placed else None,
                            "median_km": B._median(placed), "farthest_km": placed[-1] if placed else None},
            "outside": {k: {"count": sum(1 for x in outside if x["pass"] == k), "inside_if_inflated": sum(1 for x in outside if x["pass"] == k and x["inside_if_inflated"]),
                            "polygon_under_the_inflated_size": sum(1 for x in outside if x["pass"] == k and x["polygon_area"] is not None and x["polygon_area"] < INFLATED[k])}
                        for k in PASSES},
            "returns": {"breaks": sum(p["breaks"] for p in parts), "with_a_holder_away": len(returns),
                        "return": {c: sum(1 for x in returns if x["return"] == c) for c in RETURNS},
                        "return_by_smoothing_alone": {c: sum(1 for x in returns if x["return"] == c and x["by_smoothing_alone"]) for c in RETURNS},
                        "return_by_next": {n: {c: sum(1 for x in returns if x["next"] == n and x["return"] == c) for c in RETURNS} for n in B.NEXT}}}


def tables_text(rec):
    T = ["# Follow-up to the break reading of the continuity effort (declared section 29)", "",
         f"{', '.join(map(str, B.YEARS))} pooled, from the season records of section 27's reading. Everything is relative to QTrack.", ""]
    for name, p in rec["pooled"].items():
        d, e, t, x, r, g, pa, o = (p[k] for k in ("declared_sign_test", "expected_under_equal_positions", "position_test", "exits", "returns", "groups", "placed_away", "outside"))
        T += [f"## {name}", "",
              "| references with the higher break share | observed | expected when a reference's breaks are equally likely at any of its counted transitions |", "| --- | --- | --- |",
              f"| in the coastal segment | {d['higher_share_coastal']} | {e['higher_share_coastal']:.2f} |", f"| in the earlier segment | {d['higher_share_earlier']} | {e['higher_share_earlier']:.2f} |",
              f"| equal | {d['equal_shares']} | {e['equal_shares']:.2f} |", "",
              f"Declared sign test {d['sign_test_value']:.4f}, declared outcome {d['outcome'].upper()}. Position test: {t['coastal_breaks']} coastal breaks where {t['expected_coastal_breaks']:.2f} "
              f"are expected (possible range {t['least']} to {t['most']}), probability of that many or more {t['at_least_observed']:.4f}, of that many or fewer {t['at_most_observed']:.4f}. "
              f"Outcome: {t['outcome'].upper()}. The coastal breaks lie on {t['references_with_a_coastal_break']} references. Concentrated near the coast would be read "
              + ("at no count" if t["near_the_coast_from"] is None else f"from {t['near_the_coast_from']} coastal breaks") + ", concentrated earlier "
              + ("at no count" if t["earlier_up_to"] is None else f"up to {t['earlier_up_to']}") + ".", "",
              f"Primary histories lacking the end, without sustained carriage: {x['count']}. At its last accompanied observation the primary "
              + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in x["primary_did"].items()) + f" (away and marginal {x['away_marginal']}, away by smoothing alone {x['away_by_smoothing_alone']}).", "",
              "| last accompanied observation of the primary | " + " | ".join(k.replace("_", " ") for k in PRIMARY_DID) + " |", "| --- |" + " --- |" * len(PRIMARY_DID)]
        for s in B.STOPS[1:]:
            T.append(f"| {s.replace('_', ' ')} | " + " | ".join(str(x["by_stop"][s][k]) for k in PRIMARY_DID) + " |")
        T += ["", "The reference at that observation: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in x["reference_then"].items() if v) + ".", "",
              f"Breaks: {r['breaks']}, with a holder that continued away {r['with_a_holder_away']}.", "",
              "| a holder that went away is beside the reference again | breaks | of which away by smoothing alone |", "| --- | --- | --- |"]
        for c in RETURNS:
            T.append(f"| {c.replace('_', ' ')} | {r['return'][c]} | {r['return_by_smoothing_alone'][c]} |")
        T += ["", "| breaks by the main holder's attempts | all | coastal | earlier | of which away by smoothing alone |", "| --- | --- | --- | --- | --- |"]
        for k in GROUPS:
            T.append(f"| {k.replace('_', ' ')} | {g['all'][k]} | {g['coastal'][k]} | {g['earlier'][k]} | {g['by_smoothing_alone'][k]} |")
        T += ["", f"Breaks classed as placed away: {pa['count']}, the nearest final candidate at {pa['nearest_km']} to {pa['farthest_km']} km from the reference's next position "
                  f"(median {pa['median_km']}), within {B.MARGINAL_KM:g} km for {pa['within_750_km']}.", "",
              "Attempts with a candidate outside the search area: " + ", ".join(
                  f"{k.replace('hr', '-hour')} {o[k]['count']} (polygon under its inflated size {o[k]['polygon_under_the_inflated_size']}, a trough point inside once inflated "
                  f"{o[k]['inside_if_inflated']})" for k in PASSES) + ".", ""]
    return "\n".join(T)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    pj, pm = (os.path.join(a.out_dir, f"{STEM}.{e}") for e in ("json", "md"))
    for p in (pj, pm):
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and records are never overwritten")
    combined_path = os.path.join(a.out_dir, f"{B.STEM}.json")
    combined = json.load(open(combined_path, encoding="utf-8"))
    if combined["generated_by"]["sha256"] != B.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the combined break-reading record was not written by the current break-reading script")
    seasons = [season_followup(y, a.out_dir) for y in B.YEARS]
    pool = {n: pooled([s["configs"][n] for s in seasons]) for n in B.CONFIGS}
    for n in B.CONFIGS:
        theirs = combined["pooled"][n]["concentration"]
        if any(pool[n]["declared_sign_test"][k] != theirs[k] for k in pool[n]["declared_sign_test"]) or pool[n]["returns"]["breaks"] != combined["pooled"][n]["breaks"]["count"]:
            raise SystemExit(f"REFUSED: the {n} season records do not give the combined break-reading record")
    rec = {"generated_by": {"script": "scripts/continuity_break_followup.py", "sha256": SCRIPT_SHA256_AT_LOAD, "break_reading_sha256": B.SCRIPT_SHA256_AT_LOAD,
                            "declared": "CONTINUITY_ACROSS_AFRICA_2026-11-09.md section 29", "combined_record_sha256": C.sha256_of(combined_path),
                            "seasons": {str(s["year"]): s["record_sha256"] for s in seasons}},
           "seasons": {str(s["year"]): s["configs"] for s in seasons}, "pooled": pool}
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    text = tables_text(rec)
    with open(pm, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
