#!/usr/bin/env python3
"""The prospective evaluation of the unchanged VCLIP placement, specified and frozen in section 25
of `CONTINUITY_ACROSS_AFRICA_2026-11-09.md` before any reserved season was read.

One season is replayed through the registered production loop under the released configuration and
under VCLIP (the 5-degree clip of the coarse merge at exhaustion, nothing else changed). The
released replay must reproduce the stored campaign tracks and the coast-crossing rows. Both sets of
finished tracks are then read against the same REFERENCE POPULATION, the QTrack cohort tracks that
start between 10 and 30 E, each to its first Atlantic-side observation or its last used one.

For every reference track and every stored history beside it the reading gives the PATH: how many
of the reference's observations the history accompanies, how far the reference had traveled when
the history first accompanied it, the longest time between successive accompanied observations (a
long gap when more than 24 hours), and whether the history is beside the reference near its start
and near its coastal end, within 500 km of its first position and of its first Atlantic-side
position. A history that is not beside it near its start is a late pickup. A coast-reaching reference has
SUSTAINED carriage when one history is beside it near both ends, accompanies at least half of its
observations and has no long gap, and UNMIXED sustained carriage when that history has no switch or
undetermined transition involving the reference. Section 3's endpoint class is kept as its own
measure. Each history beside a reference is clean, open (a relay of that history involves the
reference) or mixed (a switch or an undetermined transition does), and the PRIMARY history of a
reference is the one accompanying the most of its observations. Carriers are counted as histories
and as UNITS under the product's own duplicate grouping, and the tracks beyond one per unit are
the surplus tracks.

`combine` reports each season and the pool and applies section 25's rule, whose three parts
(benefit, adverse evidence, product) each end in pass, fail or insufficient evidence. The result is
operational evidence relative to QTrack. It does not establish physical continuity or the origin of
any wave.

A reserved season is read only with `--reserved`, only when the freeze record exists, matches the
files on disk and is committed, and only once.

    python3 scripts/continuity_placement_check.py season --year 1990 --out-dir DIR
    python3 scripts/continuity_placement_check.py combine --out-dir DIR
    python3 scripts/continuity_placement_check.py freeze --out-dir DIR
    python3 scripts/continuity_placement_check.py season --year 1984 --out-dir DIR --reserved
    python3 scripts/continuity_placement_check.py combine --out-dir DIR --reserved
"""

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import continuity_census as C  # noqa: E402
import continuity_evaluation as E  # noqa: E402
import continuity_join_test as J  # noqa: E402
import continuity_lifecycle as L  # noqa: E402
import continuity_merge_variants as VAR  # noqa: E402
import continuity_recovery as REC  # noqa: E402
import continuity_stage_sample as S  # noqa: E402
import continuity_variant_replay as R  # noqa: E402

with open(os.path.abspath(__file__), "rb") as _fh:
    SCRIPT_SHA256_AT_LOAD = hashlib.sha256(_fh.read()).hexdigest()

CANDIDATE = "VCLIP"
CONFIGS = {"released": {}, CANDIDATE: VAR.VARIANTS[CANDIDATE]}
DEVELOPMENT = S.YEARS
RESERVED = C.RESERVED
ARTIFACTS = L.JOIN_DIR
FREEZE = os.path.join(ARTIFACTS, "continuity_placement_freeze.json")
STEM = {"development": "continuity_placement_development", "reserved": "continuity_placement_reserved"}
RADIUS_KM = C.RADIUS_KM
GAP_HOURS = 24                                       # a longer time between accompanied observations is a long gap
BENEFIT_ONE_IN = 10                                  # a net gain of one coast-reaching reference in ten
ALPHA = (1, 20)                                      # 0.05, kept as a ratio so the exact tests are decided in whole numbers
SEASONS_WITH_A_GAIN = 3                              # of the seasons pooled
MIN_ASSESSED = 20                                    # references with a primary history, under each configuration
ADVERSE_FLOOR = (1, 20)                              # a mixed share this small passes whatever the released share
MARGIN = (5, 4)                                      # a share may reach 1.25 times the released one
OUTCOMES = ("pass", "fail", "insufficient evidence")
ADVERSE_MEASURES = ("mixed_with_a_history", "supported_false_link_with_a_history")
STATUSES = ("clean", "open", "mixed")
COUNT_KEYS = ("tracks", "tracks_in_sector", "units_in_sector", "surplus_tracks_in_sector", "stationary_tracks", "duplicate_groups", "never_accompanied_in_sector",
              "cohort_tracks", "cohort_tracks_atlantic_side", "cohort_units", "cohort_units_atlantic_side")


# --- the path of one history beside one reference ------------------------------------------------

def path(q, n, indices, radius_km=RADIUS_KM, gap_hours=GAP_HOURS):
    """How one history accompanies a reference read over its first `n` observations, from the
    indices at which it is within 500 km."""
    lat, lon, t = q["lat"], q["lon"], q["time"]
    k0 = next(k for k in range(n) if np.isfinite(lat[k]) and np.isfinite(lon[k]))
    from_start = [float(C.km(lat[k], lon[k], lat[k0], lon[k0])) for k in indices]
    from_end = [float(C.km(lat[k], lon[k], lat[n - 1], lon[n - 1])) for k in indices]
    gaps = [int(round(24.0 * (float(t[b]) - float(t[a])))) for a, b in zip(indices, indices[1:])]
    return {"count": len(indices), "first": int(indices[0]), "last": int(indices[-1]),
            "pickup_hours": int(round(24.0 * (float(t[indices[0]]) - float(t[k0])))), "pickup_km": round(from_start[0], 1),
            "start": any(d <= radius_km for d in from_start), "end": any(d <= radius_km for d in from_end),
            "majority": 2 * len(indices) >= n, "longest_gap_hours": max(gaps, default=0), "no_long_gap": all(g <= gap_hours for g in gaps)}


def involvement(history, q_index, q_spans):
    """The QTrack systems a history is mixed with (a switch or an undetermined transition names
    them as left or as new), open with (a relay does), and holds a supported false-link candidate
    with."""
    events, _ = E.events_of(history, q_index, q_spans)
    out = {"mixed": set(), "open": set(), "supported": set()}
    for e in events:
        ids = set(e["lost"]) | set(e["new"])
        if e["kind"] in E.RISK_KINDS:
            out["mixed"] |= ids
            if e["kind"] == "switch" and E.link_marks(e, history, q_spans)["supported"]:
                out["supported"] |= ids
        elif e["kind"] == "relay":
            out["open"] |= ids
    return out


def reference_rows(histories, group_of, q, q_index, q_spans, q_rows):
    """One row per reference track for one configuration's finished tracks, with the base rows of
    `continuity_evaluation.reference_measures` they were built from."""
    base = E.reference_measures(histories, q, q_rows)
    by_id = {h["id"]: h for h in histories}
    seen, out = {}, []

    def units(ids):
        return len({group_of.get(h, ("ungrouped", h)) for h in ids})

    for r in base:
        coast = r["outcome"] == "atlantic_side"
        per = {}
        for hid, idx in r["by_history"].items():
            if hid not in seen:
                seen[hid] = involvement(by_id[hid], q_index, q_spans)
            p = path(q[r["id"]], r["steps"], idx)
            p["status"] = "mixed" if r["id"] in seen[hid]["mixed"] else ("open" if r["id"] in seen[hid]["open"] else "clean")
            p["supported_false_link"] = r["id"] in seen[hid]["supported"]
            p["sustained"] = bool(p["start"] and p["end"] and p["majority"] and p["no_long_gap"])
            per[hid] = p
        primary = min(per, key=lambda h: (-per[h]["count"], per[h]["first"], h)) if per else None
        endpoint = sorted(set(r["start"]) & set(r["event"])) if coast else []
        if coast and bool(endpoint) != (r["class"] == "carried"):
            raise SystemExit(f"REFUSED: reference {r['id']} has endpoint carriers that disagree with its class")
        sustained = sorted(h for h in per if per[h]["sustained"]) if coast else []
        unmixed = [h for h in sustained if per[h]["status"] != "mixed"]
        clean = [h for h in sustained if per[h]["status"] == "clean"]
        out.append({"id": r["id"], "outcome": r["outcome"], "coast_reaching": coast, "steps": r["steps"], "accompanied": r["accompanied"], "best_count": r["longest"],
                    "histories": len(per), "endpoint_class": r.get("class"), "endpoint_carried": bool(endpoint), "endpoint_carriers": len(endpoint),
                    "endpoint_carrier_units": units(endpoint), "sustained": bool(sustained), "sustained_carriers": len(sustained),
                    "unmixed_sustained": bool(unmixed), "unmixed_sustained_carriers": len(unmixed), "unmixed_sustained_carrier_units": units(unmixed),
                    "clean_sustained": bool(clean), "mixed_with_a_history": any(p["status"] == "mixed" for p in per.values()),
                    "open_with_a_history": any(p["status"] == "open" for p in per.values()),
                    "supported_false_link_with_a_history": any(p["supported_false_link"] for p in per.values()),
                    "primary": None if primary is None else {"history": primary, **per[primary]}})
    return out, base


# --- exact tests and the three parts of the rule ----------------------------------------------------

def sign_test(gained, lost):
    """P(X >= gained) for X binomial(gained + lost, 1/2), as an exact fraction (numerator, denominator)."""
    n = gained + lost
    return sum(comb(n, k) for k in range(gained, n + 1)), 2 ** n


def fisher(m_c, a_c, m_r, a_r):
    """One-sided Fisher exact test that the candidate's share m_c of a_c exceeds the released
    m_r of a_r, P(X >= m_c) under the hypergeometric law, as an exact fraction."""
    total, marked = a_c + a_r, m_c + m_r
    num = sum(comb(marked, x) * comb(total - marked, a_c - x) for x in range(m_c, min(marked, a_c) + 1))
    return num, comb(total, a_c)


def at_most_alpha(fraction):
    return fraction[0] * ALPHA[1] <= ALPHA[0] * fraction[1]


def benefit_outcome(gained, lost, coast_reaching):
    """Pass on a net gain of unmixed sustained carriage of one coast-reaching reference in ten that a
    sign test puts beyond chance, fail on no net gain, insufficient evidence between."""
    if coast_reaching == 0:
        return "insufficient evidence"
    if (gained - lost) * BENEFIT_ONE_IN >= coast_reaching and at_most_alpha(sign_test(gained, lost)):
        return "pass"
    return "fail" if gained <= lost else "insufficient evidence"


def adverse_outcome(m_c, a_c, m_r, a_r):
    """On the references that have a history beside them, for a count of adverse references under
    each configuration: pass when the candidate's share is at most one in twenty or at most 1.25
    times the released share, fail when it is higher and a one-sided Fisher test puts the excess
    beyond chance, insufficient evidence otherwise or with too few references under either
    configuration."""
    if a_c < MIN_ASSESSED or a_r < MIN_ASSESSED:
        return "insufficient evidence"
    if m_c * ADVERSE_FLOOR[1] <= ADVERSE_FLOOR[0] * a_c or m_c * a_r * MARGIN[1] <= MARGIN[0] * m_r * a_c:
        return "pass"
    return "fail" if at_most_alpha(fisher(m_c, a_c, m_r, a_r)) else "insufficient evidence"


def product_outcome(unique, carried, c, r):
    """One carrier unit for a majority of the unmixed sustained references (pass or fail when a sign
    test puts the majority beyond chance either way), and the surplus tracks and the stationary
    tracks per track in the sector within 1.25 times the released record's. `c` and `r` are the
    candidate's and the released configuration's counts."""
    several = carried - unique
    checks = {"unique_carrier_unit": "pass" if unique > several and at_most_alpha(sign_test(unique, several)) else
              ("fail" if several > unique and at_most_alpha(sign_test(several, unique)) else "insufficient evidence")}
    for key in ("surplus_tracks_in_sector", "stationary_tracks"):
        if not r["tracks_in_sector"] or not c["tracks_in_sector"]:
            checks[key] = "insufficient evidence"
        else:
            checks[key] = "pass" if c[key] * r["tracks_in_sector"] * MARGIN[1] <= MARGIN[0] * r[key] * c["tracks_in_sector"] else "fail"
    return checks, worst(checks.values())


def worst(outcomes):
    outcomes = list(outcomes)
    return "fail" if "fail" in outcomes else ("insufficient evidence" if "insufficient evidence" in outcomes else "pass")


def tally(rows):
    """The counts of one configuration's reference rows."""
    coast = [r for r in rows if r["coast_reaching"]]
    prim = [r["primary"] for r in rows if r["primary"] is not None]
    carried = [r for r in coast if r["unmixed_sustained"]]
    return {"references": len(rows), "coast_reaching": len(coast), "observations": sum(r["steps"] for r in rows), "accompanied": sum(r["accompanied"] for r in rows),
            "best_single_history_observations": sum(r["best_count"] for r in rows), "histories_beside_references": sum(r["histories"] for r in rows),
            "endpoint_carried": sum(1 for r in coast if r["endpoint_carried"]), "sustained": sum(1 for r in coast if r["sustained"]),
            "unmixed_sustained": len(carried), "unmixed_sustained_with_one_carrier_unit": sum(1 for r in carried if r["unmixed_sustained_carrier_units"] == 1),
            "clean_sustained": sum(1 for r in coast if r["clean_sustained"]), "endpoint_carried_and_sustained": sum(1 for r in coast if r["endpoint_carried"] and r["sustained"]),
            "mixed_with_a_history": sum(1 for r in rows if r["mixed_with_a_history"]), "open_with_a_history": sum(1 for r in rows if r["open_with_a_history"]),
            "supported_false_link_with_a_history": sum(1 for r in rows if r["supported_false_link_with_a_history"]),
            "most_endpoint_carriers_of_one_reference": max((r["endpoint_carriers"] for r in coast), default=0),
            "most_unmixed_sustained_carrier_units_of_one_reference": max((r["unmixed_sustained_carrier_units"] for r in carried), default=0),
            "with_a_primary_history": len(prim), "without_any_history": len(rows) - len(prim),
            **{f"primary_{s}": sum(1 for p in prim if p["status"] == s) for s in STATUSES},
            "primary_late_pickup": sum(1 for p in prim if not p["start"]), "primary_long_gap": sum(1 for p in prim if not p["no_long_gap"])}


def paired(released, candidate, key):
    """Coast-reaching references by whether `key` holds under each configuration."""
    rel = {r["id"]: r for r in released if r["coast_reaching"]}
    cand = {r["id"]: r for r in candidate if r["coast_reaching"]}
    if sorted(rel, key=str) != sorted(cand, key=str):
        raise SystemExit("REFUSED: the two configurations were read against different reference tracks")
    return {"gained": sum(1 for i in cand if cand[i][key] and not rel[i][key]), "lost": sum(1 for i in cand if rel[i][key] and not cand[i][key]),
            "both": sum(1 for i in cand if cand[i][key] and rel[i][key]), "neither": sum(1 for i in cand if not cand[i][key] and not rel[i][key])}


def evaluate(released, candidate):
    """Section 25's three parts for one unit, a season or the pool. Each argument holds a
    configuration's reference `rows` and its output `counts`."""
    tr, tc = tally(released["rows"]), tally(candidate["rows"])
    if (tr["references"], tr["coast_reaching"], tr["observations"]) != (tc["references"], tc["coast_reaching"], tc["observations"]):
        raise SystemExit("REFUSED: the two configurations were read against different reference observations")
    sustained = paired(released["rows"], candidate["rows"], "unmixed_sustained")
    endpoint = paired(released["rows"], candidate["rows"], "endpoint_carried")
    p_sign = sign_test(sustained["gained"], sustained["lost"])
    a_c, a_r = tc["with_a_primary_history"], tr["with_a_primary_history"]
    adverse, p_fisher = {}, {}
    for key in ADVERSE_MEASURES:
        f = fisher(tc[key], a_c, tr[key], a_r)
        p_fisher[key] = round(f[0] / f[1], 6)
        adverse[key] = adverse_outcome(tc[key], a_c, tr[key], a_r)
    if 2 * a_c < tc["references"]:                        # most references have no history beside them, so nothing can be said of them
        adverse = {k: ("insufficient evidence" if v == "pass" else v) for k, v in adverse.items()}
    checks, product = product_outcome(tc["unmixed_sustained_with_one_carrier_unit"], tc["unmixed_sustained"], candidate["counts"], released["counts"])
    return {"released": tr, "candidate": tc, "counts": {"released": released["counts"], "candidate": candidate["counts"]},
            "paired_unmixed_sustained": sustained, "paired_endpoint_carried": endpoint, "paired_clean_sustained": paired(released["rows"], candidate["rows"], "clean_sustained"),
            "sign_test_p": round(p_sign[0] / p_sign[1], 6), "fisher_p": p_fisher,
            "benefit": benefit_outcome(sustained["gained"], sustained["lost"], tc["coast_reaching"]),
            "adverse_checks": adverse, "adverse": worst(adverse.values()),
            "product_checks": checks, "product": product}


def decide(seasons):
    """Each season and the pool. The pool's benefit passes only if no season has a net loss and at
    least `SEASONS_WITH_A_GAIN` have a net gain, and its adverse part only if no season fails."""
    per = {str(s["year"]): evaluate(s["configs"]["released"], s["configs"][CANDIDATE]) for s in seasons}

    def pooled(name):
        rows = [{**r, "id": f"{s['year']}:{r['id']}"} for s in seasons for r in s["configs"][name]["rows"]]
        return {"rows": rows, "counts": {k: sum(s["configs"][name]["counts"][k] for s in seasons) for k in COUNT_KEYS}}

    pool = evaluate(pooled("released"), pooled(CANDIDATE))
    nets = [e["paired_unmixed_sustained"]["gained"] - e["paired_unmixed_sustained"]["lost"] for e in per.values()]
    consistency = {"seasons_with_a_net_gain": sum(1 for n in nets if n > 0), "seasons_with_a_net_loss": sum(1 for n in nets if n < 0),
                   "seasons_failing_the_adverse_part": sum(1 for e in per.values() if e["adverse"] == "fail")}
    need = min(SEASONS_WITH_A_GAIN, len(seasons))
    if pool["benefit"] == "pass" and (consistency["seasons_with_a_net_loss"] or consistency["seasons_with_a_net_gain"] < need):
        pool["benefit"] = "insufficient evidence"
    if pool["adverse"] == "pass" and consistency["seasons_failing_the_adverse_part"]:
        pool["adverse"] = "insufficient evidence"
    return {"seasons": per, "pool": pool, "consistency": consistency, "outcome": worst([pool["benefit"], pool["adverse"], pool["product"]])}


# --- the freeze ---------------------------------------------------------------------------------------

def repo_modules():
    """The repository's own source files loaded in this process, relative to its root."""
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f:
            rel = os.path.relpath(os.path.realpath(f), os.path.realpath(ROOT))
            if rel.startswith(("scripts" + os.sep, "src" + os.sep)):
                out.add(rel)
    return sorted(out)


def committed(paths):
    """True when every path is tracked and has no staged or unstaged change."""
    if subprocess.run(["git", "ls-files", "--error-unmatch", *paths], cwd=ROOT, capture_output=True).returncode != 0:
        return False
    return subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=ROOT, capture_output=True, text=True).stdout.strip() == ""


def rule():
    return {"candidate": CONFIGS[CANDIDATE], "radius_km": RADIUS_KM, "gap_hours": GAP_HOURS, "benefit_one_in": BENEFIT_ONE_IN, "alpha": list(ALPHA),
            "seasons_with_a_gain": SEASONS_WITH_A_GAIN, "min_assessed": MIN_ASSESSED, "adverse_floor": list(ADVERSE_FLOOR), "margin": list(MARGIN),
            "adverse_measures": list(ADVERSE_MEASURES), "reserved": list(RESERVED), "development": list(DEVELOPMENT)}


def write_freeze(out_dir, path=FREEZE):
    """The digests of every source file the development replays loaded, with the rule's numbers."""
    if os.path.exists(path):
        raise SystemExit(f"REFUSED: {path} exists and a freeze is written once")
    files, records = set(), {}
    for y in DEVELOPMENT:
        p = os.path.join(out_dir, f"{STEM['development']}_{y}.json.gz")
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            rec = json.load(fh)
        if rec["generated_by"]["sha256"] != SCRIPT_SHA256_AT_LOAD:
            raise SystemExit(f"REFUSED: the {y} development record was not written by the current script")
        files |= set(rec["modules"]); records[str(y)] = C.sha256_of(p)
    files.add(os.path.relpath(os.path.realpath(__file__), os.path.realpath(ROOT)))
    rec = {"declared": "CONTINUITY_ACROSS_AFRICA_2026-11-09.md section 25", "rule": rule(), "development_records": records,
           "files": {f: C.sha256_of(os.path.join(ROOT, f)) for f in sorted(files)}}
    with open(path, "x", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def require_freeze(path=FREEZE, is_committed=committed):
    """The freeze must exist, name this rule, match every frozen file on disk and be committed with
    them. Returns its digest."""
    if not os.path.exists(path):
        raise SystemExit("REFUSED: no freeze record, so no reserved season is read")
    fr = json.load(open(path, encoding="utf-8"))
    if fr["rule"] != rule():
        raise SystemExit("REFUSED: the frozen rule is not the rule in this script")
    changed = [f for f, sha in fr["files"].items() if not os.path.exists(os.path.join(ROOT, f)) or C.sha256_of(os.path.join(ROOT, f)) != sha]
    if changed:
        raise SystemExit(f"REFUSED: files changed since the freeze: {', '.join(changed)}")
    if not is_committed([os.path.relpath(os.path.realpath(path), os.path.realpath(ROOT)), *fr["files"]]):
        raise SystemExit("REFUSED: the freeze record or a frozen file is not committed")
    return C.sha256_of(path), fr


def check_modules(frozen):
    extra = [f for f in repo_modules() if f not in frozen["files"]]
    if extra:
        raise SystemExit(f"REFUSED: source files outside the freeze were loaded: {', '.join(extra)}")


# --- one season ---------------------------------------------------------------------------------------

def output_counts(tracks, stored, membership, side, q_index):
    sector = R.in_sector_tracks(stored)
    effects = R.side_effects(tracks, q_index, membership)
    if effects["tracks_in_sector"] != len(sector):
        raise SystemExit("REFUSED: the stored histories and the finished tracks disagree on the tracks in the sector")
    import coast_crossing_measurement as CC

    def in_band(x):
        return C.START_LONS[0] <= x["first"]["lon"] < C.START_LONS[1]
    cohort = [r for r in side["tracks"] if in_band(r)]
    units = [u for u in CC.group_units(side["tracks"], membership) if in_band(u)]
    grouped = sum(1 for t in R.in_sector_tracks(tracks) if t["id"] in membership)
    units_in_sector = len(sector) - grouped + effects["duplicate_groups"]     # every group with a member in the sector is one unit there
    return {"tracks": len(tracks), "tracks_in_sector": len(sector), "units_in_sector": units_in_sector, "surplus_tracks_in_sector": len(sector) - units_in_sector,
            "stationary_tracks": effects["stationary_tracks"], "duplicate_groups": effects["duplicate_groups"],
            "never_accompanied_in_sector": sum(1 for h in sector if not L.ever_accompanied(h, q_index)),
            "cohort_tracks": len(cohort), "cohort_tracks_atlantic_side": sum(1 for r in cohort if r["outcome"] == "atlantic_side"),
            "cohort_units": len(units), "cohort_units_atlantic_side": sum(1 for u in units if u["outcome"] == "atlantic_side")}


def check_development(year, name, base, flags, costs, artifacts=ARTIFACTS):
    """On a development season the reading must give what the lifecycle census recorded."""
    with gzip.open(os.path.join(artifacts, f"continuity_lifecycle_{year}.json.gz"), "rt", encoding="utf-8") as fh:
        life = json.load(fh)
    if life["generated_by"]["sha256"] != L.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit(f"REFUSED: the {year} lifecycle record was not written by the current lifecycle script")
    stored = life["configs"][name]["census"]["STORED"]
    if E.pool_reference(base) != stored["reference"] or flags != stored["flags"] or costs != stored["costs"]:
        raise SystemExit(f"REFUSED: the {year} {name} reading does not reproduce the lifecycle census")


def run_season(year, mode, loader=REC.load_season, loop=L.lifecycle_loop, freeze_path=FREEZE, artifact_dir=C.ARTIFACT_DIR):
    if mode == "development" and year not in DEVELOPMENT:
        raise SystemExit(f"REFUSED: {year} is not a development season")
    if mode == "reserved" and year not in RESERVED:
        raise SystemExit(f"REFUSED: {year} is not a reserved season")
    frozen = require_freeze(freeze_path) if mode == "reserved" else None
    import coast_crossing_measurement as CC
    import pilot_margin_replay as MR
    import qtrack_pairing_pilot as M
    import season_metrics as SM
    art_path = os.path.join(artifact_dir, C.ARTIFACT.format(year=year))
    art = json.load(open(art_path, encoding="utf-8"))
    for key in ("this_record", "qtrack"):
        if C.sha256_of(art["inputs"][key]["path"]) != art["inputs"][key]["sha256"]:
            raise SystemExit(f"REFUSED: {art['inputs'][key]['path']} is not the file the {year} artifact names")
    run, case, ct, ft, flags, grids = loader(year)
    if run["tracks_sha256"] != art["inputs"]["this_record"]["sha256"]:
        raise SystemExit(f"REFUSED: the {year} campaign run and the artifact do not name one set of tracks")
    regions, _ = SM.load_regions(art["inputs"]["region_polygons"]["dir"])
    q_tracks = M.load_qtrack(art["inputs"]["qtrack"]["path"])[0]
    q = {t["id"]: t for t in q_tracks}
    q_index, q_spans, q_rows = C.index_by_time(q_tracks), J.spans(q_tracks), art["sides"]["qtrack"]["tracks"]
    out = {}
    for name, options in CONFIGS.items():
        finished, rows, checks = loop(case, grids, ct, ft, flags, options)
        if name == "released" and not MR.archive_gate(finished, run)["passed"]:
            raise SystemExit(f"REFUSED: the {year} released loop does not reproduce the stored campaign tracks")
        done = [r for r in rows if r["fate"] == "finished"]
        stored = L.finalized(done)
        if len(stored) != len(done):
            raise SystemExit(f"REFUSED: a finished history of the {year} {name} replay does not pass the final filter a second time")
        L.check_stored(f"{year} {name}", {"STORED": stored}, done, finished)
        tracks = R.as_tracks(finished)
        membership = CC.groups_ours(tracks)
        side = CC.measure(tracks, year, regions, M._day(year, 1, 1), membership)
        if name == "released":
            mine = sorted((r["id"], r["outcome"]) for r in side["tracks"])
            if mine != sorted((r["id"], r["outcome"]) for r in art["sides"]["this_record"]["tracks"]):
                raise SystemExit(f"REFUSED: the {year} released loop's coast-crossing rows differ from the artifact's")
        group_of = {r["birth"]: membership[r["finished_index"]] for r in done if r["finished_index"] in membership}
        ref, base = reference_rows(stored, group_of, q, q_index, q_spans, q_rows)
        corr = R.correspondence(tracks, side["tracks"], q, q_rows)
        pooled = E.pool_reference(base)
        if (pooled["accompanied"], pooled["steps"], pooled["classes"]) != (corr["accompanied_steps"], corr["qtrack_steps"], corr["qtrack_classes"]):
            raise SystemExit(f"REFUSED: the {year} {name} reference rows and the correspondence of section 14 disagree")
        flag, cost = E.flag_measures(stored, q_index, q_spans, base)
        if mode == "development":
            check_development(year, name, base, flag, cost)
        beside = {h for r in base for h in r["by_history"]}
        out[name] = {"rows": ref, "counts": output_counts(tracks, stored, membership, side, q_index), "flags": flag, "links": cost,
                     "fates": {k: sum(1 for r in rows if r["fate"] == k) for k in L.FATES}, "prune_checks": checks,
                     "histories": [{**r, "unit": group_of.get(r["birth"])} for r in done if r["birth"] in beside or L.in_box(r)]}
    modules = repo_modules()
    if frozen is not None:
        check_modules(frozen[1])
    return {"year": year, "mode": mode, "case_sha256": case["sha256"], "artifact": {"path": art_path, "sha256": C.sha256_of(art_path)},
            "freeze_sha256": None if frozen is None else frozen[0], "modules": modules, "configs": out}


# --- tables --------------------------------------------------------------------------------------------

def _row(label, t):
    return (f"| {label} | {t['accompanied']} of {t['observations']} | {t['best_single_history_observations']} | {t['endpoint_carried']} | {t['endpoint_carried_and_sustained']} | "
            f"{t['sustained']} | {t['unmixed_sustained']} | {t['clean_sustained']} | {t['unmixed_sustained_with_one_carrier_unit']} | {t['with_a_primary_history']} of {t['references']} | "
            f"{t['primary_clean']} | {t['primary_open']} | {t['primary_mixed']} | {t['primary_late_pickup']} | {t['primary_long_gap']} | {t['mixed_with_a_history']} | "
            f"{t['open_with_a_history']} | {t['supported_false_link_with_a_history']} |")


def tables_text(rec):
    D, mode = rec["decision"], rec["generated_by"]["mode"]
    units = list(D["seasons"].items()) + [("pool", D["pool"])]
    L_ = [f"# Prospective evaluation of the unchanged {CANDIDATE} placement, {mode} seasons (section 25)", "",
          "Finished tracks of each configuration against the QTrack cohort tracks starting at 10 to 30 E. Operational evidence relative to QTrack, "
          "not a statement about physical continuity or origin.", "",
          "| unit, configuration | reference observations accompanied | observations beside the best single history, summed | endpoint carried | endpoint carried and sustained | "
          "sustained | unmixed sustained | clean sustained | unmixed sustained with one carrier unit | references with a history | primary clean | primary open | primary mixed | "
          "primary late pickup | primary long gap | references mixed with any history | open with any history | with a supported false link |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for u, e in units:
        L_ += [_row(f"{u}, released", e["released"]), _row(f"{u}, {CANDIDATE}", e["candidate"])]
    L_ += ["", "| unit | coast-reaching references | unmixed sustained gained | lost | both | neither | sign test p | clean sustained gained | lost | endpoint carried gained | lost | "
               f"references with a history, released | mixed | with a supported false link | with a history, {CANDIDATE} | mixed | with a supported false link | Fisher p, mixed | "
               "benefit | adverse | product |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for u, e in units:
        s, n, k, r, c = e["paired_unmixed_sustained"], e["paired_endpoint_carried"], e["paired_clean_sustained"], e["released"], e["candidate"]
        L_.append(f"| {u} | {c['coast_reaching']} | {s['gained']} | {s['lost']} | {s['both']} | {s['neither']} | {e['sign_test_p']} | {k['gained']} | {k['lost']} | {n['gained']} | {n['lost']} | "
                  f"{r['with_a_primary_history']} | {r['mixed_with_a_history']} | {r['supported_false_link_with_a_history']} | {c['with_a_primary_history']} | {c['mixed_with_a_history']} | "
                  f"{c['supported_false_link_with_a_history']} | {e['fisher_p']['mixed_with_a_history']} | {e['benefit']} | {e['adverse']} | {e['product']} |")
    L_ += ["", "| unit, configuration | tracks | tracks in sector | units in sector | surplus tracks in sector | never beside a QTrack system | duplicate groups | stationary | "
               "cohort tracks reaching the Atlantic side | cohort units reaching the Atlantic side | most endpoint carriers of one reference | "
               "most unmixed sustained carrier units of one reference |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for u, e in units:
        for label, key in (("released", "released"), (CANDIDATE, "candidate")):
            k, t = e["counts"][key], e[key]
            L_.append(f"| {u}, {label} | {k['tracks']} | {k['tracks_in_sector']} | {k['units_in_sector']} | {k['surplus_tracks_in_sector']} | {k['never_accompanied_in_sector']} | "
                      f"{k['duplicate_groups']} | {k['stationary_tracks']} | "
                      f"{k['cohort_tracks_atlantic_side']} of {k['cohort_tracks']} | {k['cohort_units_atlantic_side']} of {k['cohort_units']} | "
                      f"{t['most_endpoint_carriers_of_one_reference']} | {t['most_unmixed_sustained_carrier_units_of_one_reference']} |")
    L_ += ["", "| unit, configuration | flagged tracks | switch transitions | relay transitions | undetermined | exposure pairs | QTrack systems left behind | "
               "reference tracks beside a flagged history | supported false-link candidates | links | jump links |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for y, s in rec["flags"].items():
        for name, f in s.items():
            L_.append(f"| {y}, {name} | {f['flagged_tracks']} | {f['events']['switch']} | {f['events']['relay']} | {f['events']['undetermined']} | {f['exposure_pairs']} | "
                      f"{f['systems_left_behind']} | {f['reference_tracks_with_a_flagged_history']} | {f['supported_false_link_candidates']} | {f['links']} | {f['jump_links']} |")
    L_ += ["", f"Coast-reaching references under {CANDIDATE}, one line each (season:reference, observations, then the primary history's accompanied observations, pickup in km "
               "and hours, longest gap in hours, status; endpoint carriers and their units; unmixed sustained carriers and their units; the kind of carriage; the same reference's released primary count):", ""]
    for line in rec["references"]:
        L_.append("- " + line)
    P_ = D["pool"]
    L_ += ["", "Adverse checks on the pool: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in P_["adverse_checks"].items()) + ".",
           "Product checks on the pool: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in P_["product_checks"].items()) + ".",
           f"Consistency across seasons: {D['consistency']['seasons_with_a_net_gain']} with a net gain of unmixed sustained carriage, {D['consistency']['seasons_with_a_net_loss']} "
           f"with a net loss, {D['consistency']['seasons_failing_the_adverse_part']} failing the adverse part.",
           f"Pool: benefit {P_['benefit']}, adverse {P_['adverse']}, product {P_['product']}. Outcome: {D['outcome']}.", ""]
    return "\n".join(L_)


def reference_lines(seasons):
    out = []
    for s in seasons:
        rel = {r["id"]: r for r in s["configs"]["released"]["rows"]}
        for r in s["configs"][CANDIDATE]["rows"]:
            if not r["coast_reaching"]:
                continue
            p, b = r["primary"], rel[r["id"]]["primary"]
            head = f"{s['year']}:{r['id']}, {r['steps']}"
            if p is None:
                out.append(f"{head}, no history; released primary {0 if b is None else b['count']}")
                continue
            kind = "clean sustained" if r["clean_sustained"] else ("unmixed sustained" if r["unmixed_sustained"] else ("sustained, mixed" if r["sustained"] else "not sustained"))
            out.append(f"{head}, {p['count']}, {p['pickup_km']} km and {p['pickup_hours']} h, {p['longest_gap_hours']} h, {p['status']}; "
                       f"{r['endpoint_carriers']} in {r['endpoint_carrier_units']}; {r['unmixed_sustained_carriers']} in {r['unmixed_sustained_carrier_units']}; "
                       f"{kind}; released primary {0 if b is None else b['count']}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("season", "combine", "freeze"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--year", type=int)
    ap.add_argument("--reserved", action="store_true")
    a = ap.parse_args(argv)
    mode = "reserved" if a.reserved else "development"
    stamp = {"script": "scripts/continuity_placement_check.py", "sha256": SCRIPT_SHA256_AT_LOAD, "lifecycle_sha256": L.SCRIPT_SHA256_AT_LOAD,
             "evaluation_sha256": E.SCRIPT_SHA256_AT_LOAD, "mode": mode, "declared": "CONTINUITY_ACROSS_AFRICA_2026-11-09.md section 25", "rule": rule()}
    if a.action == "freeze":
        rec = write_freeze(a.out_dir)
        print(json.dumps({"files": len(rec["files"]), "rule": rec["rule"]}, indent=1))
        return 0
    stem = STEM[mode]
    if a.action == "season":
        p = os.path.join(a.out_dir, f"{stem}_{a.year}.json.gz")
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists, and a season is read once and never overwritten")
        rec = {"generated_by": stamp, **run_season(a.year, mode)}
        with gzip.open(p, "xt", encoding="utf-8") as fh:
            json.dump(rec, fh)
        e = evaluate(rec["configs"]["released"], rec["configs"][CANDIDATE])
        print(json.dumps({k: e[k] for k in ("paired_unmixed_sustained", "paired_endpoint_carried", "benefit", "adverse_checks", "product_checks")}, indent=1))
        return 0
    pj, pm = (os.path.join(a.out_dir, f"{stem}.{e}") for e in ("json", "md"))
    for p in (pj, pm):
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and records are never overwritten")
    frozen = require_freeze() if mode == "reserved" else None
    seasons, digests = [], {}
    for y in (RESERVED if mode == "reserved" else DEVELOPMENT):
        path = os.path.join(a.out_dir, f"{stem}_{y}.json.gz")
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            r = json.load(fh)
        g = r["generated_by"]
        if any(g[k] != stamp[k] for k in ("sha256", "lifecycle_sha256", "evaluation_sha256", "mode", "rule")):
            raise SystemExit(f"REFUSED: the {y} season record was not written by the current scripts under this rule and mode")
        if frozen is not None and r["freeze_sha256"] != frozen[0]:
            raise SystemExit(f"REFUSED: the {y} season record was read under another freeze")
        seasons.append(r); digests[str(y)] = C.sha256_of(path)
    rec = {"generated_by": {**stamp, "seasons": digests, "freeze_sha256": None if frozen is None else frozen[0]},
           "decision": decide(seasons),
           "flags": {str(s["year"]): {n: {**c["flags"], "links": c["links"]["links"], "jump_links": c["links"]["jump_links"]} for n, c in s["configs"].items()} for s in seasons},
           "fates": {str(s["year"]): {n: c["fates"] for n, c in s["configs"].items()} for s in seasons},
           "references": reference_lines(seasons)}
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    text = tables_text(rec)
    with open(pm, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
