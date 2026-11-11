#!/usr/bin/env python3
"""The census of stored-track correspondence across Africa, declared in section 3 of
`CONTINUITY_ACROSS_AFRICA_2026-11-09.md` before any of its numbers was computed.

For the coast-crossing cohort tracks of both records that start between 10 and 30 E, it records
when each is ACCOMPANIED by the other record, meaning an observation of the other record lies
within 500 km at the same six-hourly time, and classes the QTrack tracks that reach the Atlantic
side as carried, relayed, picked up late, dropped or unaccompanied at both times. It also records
each AEWC v2 track's accompaniment, outcome and, where it ends inside Africa beside a QTrack track
that goes on to the Atlantic side, whether another AEWC v2 track accompanies that QTrack track
later. Accompaniment is proximity between stored positions and says nothing about identity.

Cohorts, outcomes and observation indices come from the retained coast-crossing artifacts, each
bound by the digests of the two tracks files it names. Reserved seasons and 2005 are refused.

    python3 scripts/continuity_census.py --out-dir DIR
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

with open(os.path.abspath(__file__), "rb") as _fh:
    SCRIPT_SHA256_AT_LOAD = hashlib.sha256(_fh.read()).hexdigest()

RADIUS_KM = 500.0
START_HOURS = 48.0                                   # inclusive, the earlier readings' window
START_LONS = (10.0, 30.0)                            # first longitude, [lo, hi)
RESERVED = (1984, 1992, 1997, 2003, 2009)
HELD_OUT = (2005,)
SEASONS = tuple(y for y in range(1981, 2011) if y not in RESERVED + HELD_OUT)
ARTIFACT_DIR = "docs/aewc_v2/artifacts/coast_crossing_30y"
ARTIFACT = "coast_crossing_{year}_2026-09-27.json"
Q_CLASSES = ("carried", "relayed", "picked_up_late", "dropped", "unaccompanied_at_both")


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def km(lat1, lon1, lat2, lon2):
    from aew.v1port.geometry import great_circle_distance
    return great_circle_distance(lat1, lon1, lat2, lon2, units="km")


def time_key(day):
    """The six-hourly step as an integer, refusing a time off the quarter-day grid."""
    k = float(day) * 4.0
    if abs(k - round(k)) > 1e-6:
        raise SystemExit(f"REFUSED: a time {day} is not on the six-hourly grid")
    return int(round(k))


def index_by_time(tracks):
    """The other record's observations grouped by step: {step: (lats, lons, ids)}."""
    acc = {}
    for t in tracks:
        for la, lo, d in zip(t["lat"], t["lon"], t["time"]):
            if np.isfinite(d) and np.isfinite(la) and np.isfinite(lo):
                acc.setdefault(time_key(d), []).append((float(la), float(lo), t["id"]))
    return {k: (np.array([v[0] for v in vs]), np.array([v[1] for v in vs]), [v[2] for v in vs]) for k, vs in acc.items()}


def companions(lat, lon, day, index):
    """The other record's tracks within RADIUS_KM at this step, nearest first, as (id, km)."""
    hit = index.get(time_key(day))
    if hit is None:
        return []
    la, lo, ids = hit
    d = km(lat, lon, la, lo)
    best = {}
    for tid, dk in zip(ids, np.atleast_1d(d)):
        if dk <= RADIUS_KM and (tid not in best or dk < best[tid]):
            best[tid] = float(dk)
    return sorted(best.items(), key=lambda x: (x[1], x[0]))


def start_set(track, index, upto):
    """The other record's tracks within RADIUS_KM at any step in the first START_HOURS, up to
    observation index `upto` inclusive."""
    t0 = float(track["time"][0])
    out = set()
    for k in range(upto + 1):
        if (float(track["time"][k]) - t0) * 24.0 > START_HOURS + 1e-9:
            break
        out.update(tid for tid, _ in companions(track["lat"][k], track["lon"][k], track["time"][k], index))
    return out


def accompanied_share(track, index, upto):
    n = upto + 1
    hits = sum(1 for k in range(n) if companions(track["lat"][k], track["lon"][k], track["time"][k], index))
    return hits, n


def classify_q(start, at_event):
    """Section 3's classes for a QTrack track that reaches the Atlantic side, from the AEWC v2
    tracks accompanying it in its first 48 hours and at its first Atlantic-side observation."""
    if start and at_event:
        return "carried" if start & at_event else "relayed"
    if at_event:
        return "picked_up_late"
    if start:
        return "dropped"
    return "unaccompanied_at_both"


def check_track(row, track, side, year):
    """The artifact's first position must be the loaded track's, so ids and indices agree."""
    f = row["first"]
    if abs(float(track["lat"][0]) - f["lat"]) > 1e-6 or abs(float(track["lon"][0]) - f["lon"]) > 1e-6:
        raise SystemExit(f"REFUSED: {side} track {row['id']} of {year} does not start where its artifact says")


def season(year, artifact_dir=ARTIFACT_DIR, loaders=None):
    if year in RESERVED or year in HELD_OUT:
        raise SystemExit(f"REFUSED: {year} is reserved or held out and is not read at track level")
    import qtrack_pairing_pilot as M
    load_ours, load_q = loaders or (M.load_ours, M.load_qtrack)
    path = os.path.join(artifact_dir, ARTIFACT.format(year=year))
    art = json.load(open(path, encoding="utf-8"))
    files = {}
    for side, key in (("this_record", "this_record"), ("qtrack", "qtrack")):
        p = art["inputs"][key]["path"]
        if sha256_of(p) != art["inputs"][key]["sha256"]:
            raise SystemExit(f"REFUSED: {p} is not the file the {year} artifact names")
        files[side] = p
    ours = {t["id"]: t for t in load_ours(files["this_record"])[0]}
    q = {t["id"]: t for t in load_q(files["qtrack"])[0]}
    idx_ours, idx_q = index_by_time(ours.values()), index_by_time(q.values())
    q_rows = {r["id"]: r for r in art["sides"]["qtrack"]["tracks"]}

    def in_band(r):
        return START_LONS[0] <= r["first"]["lon"] < START_LONS[1]

    q_out = []
    for r in art["sides"]["qtrack"]["tracks"]:
        if not in_band(r):
            continue
        t = q[r["id"]]; check_track(r, t, "qtrack", year)
        upto = r["first_atlantic_side"]["index"] if r["outcome"] == "atlantic_side" else r["last_used"]["index"]
        s = start_set(t, idx_ours, upto)
        hits, n = accompanied_share(t, idx_ours, upto)
        row = {"id": r["id"], "band": r["band"], "outcome": r["outcome"], "accompanied_start": bool(s),
               "accompanied_steps": hits, "steps": n}
        if r["outcome"] == "atlantic_side":
            k = upto
            ev = {tid for tid, _ in companions(t["lat"][k], t["lon"][k], t["time"][k], idx_ours)}
            row.update({"class": classify_q(s, ev), "start_companions": sorted(s), "event_companions": sorted(ev)})
        else:
            k = r["last_used"]["index"]
            row["accompanied_last"] = bool(companions(t["lat"][k], t["lon"][k], t["time"][k], idx_ours))
        q_out.append(row)

    a_out = []
    for r in art["sides"]["this_record"]["tracks"]:
        if not in_band(r):
            continue
        t = ours[r["id"]]; check_track(r, t, "this_record", year)
        k = r["last_used"]["index"]
        s = start_set(t, idx_q, k)
        last = companions(t["lat"][k], t["lon"][k], t["time"][k], idx_q)
        row = {"id": r["id"], "band": r["band"], "outcome": r["outcome"], "accompanied_start": bool(s),
               "accompanied_last": bool(last)}
        if r["outcome"] != "atlantic_side":
            goes_on = [tid for tid, _ in last if tid in q_rows and q_rows[tid]["outcome"] == "atlantic_side"]
            if goes_on:
                qid = goes_on[0]; qt = q[qid]; qe = q_rows[qid]["first_atlantic_side"]["index"]
                t_end = float(t["time"][k])
                later = sorted({tid for j in range(qe + 1) if float(qt["time"][j]) > t_end + 1e-9
                                for tid, _ in companions(qt["lat"][j], qt["lon"][j], qt["time"][j], idx_ours) if tid != r["id"]})
                row["beside_atlantic_qtrack"] = {"qtrack_id": qid, "later_aewc_tracks": later,
                                                 "kind": "candidate_discontinuity" if later else "not_followed"}
        a_out.append(row)
    return {"year": year, "artifact": {"path": path, "sha256": sha256_of(path)},
            "inputs": {k: {"path": v, "sha256": sha256_of(v)} for k, v in files.items()},
            "qtrack": q_out, "this_record": a_out}


def totals(seasons):
    q = [r for s in seasons for r in s["qtrack"]]
    a = [r for s in seasons for r in s["this_record"]]
    qa = [r for r in q if r["outcome"] == "atlantic_side"]
    qn = [r for r in q if r["outcome"] != "atlantic_side"]
    out = {"qtrack_n": len(q), "qtrack_atlantic_side": len(qa),
           "qtrack_atlantic_classes": {c: sum(1 for r in qa if r["class"] == c) for c in Q_CLASSES},
           "qtrack_other_by_start_last": {f"{s}_{l}": sum(1 for r in qn if r["accompanied_start"] == s and r["accompanied_last"] == l)
                                          for s in (True, False) for l in (True, False)},
           "qtrack_accompanied_share": round(sum(r["accompanied_steps"] for r in q) / max(1, sum(r["steps"] for r in q)), 4),
           "aewc_n": len(a), "aewc_atlantic_side": sum(1 for r in a if r["outcome"] == "atlantic_side")}
    out["aewc_by_start_and_outcome"] = {
        f"{'accompanied' if s else 'unaccompanied'}_start": {"n": sum(1 for r in a if r["accompanied_start"] == s),
                                                              "atlantic_side": sum(1 for r in a if r["accompanied_start"] == s and r["outcome"] == "atlantic_side")}
        for s in (True, False)}
    beside = [r["beside_atlantic_qtrack"] for r in a if "beside_atlantic_qtrack" in r]
    out["aewc_inland_end_beside_atlantic_qtrack"] = {"n": len(beside),
                                                    "candidate_discontinuity": sum(1 for b in beside if b["kind"] == "candidate_discontinuity"),
                                                    "not_followed": sum(1 for b in beside if b["kind"] == "not_followed")}
    return out


def tables_text(rec):
    T = rec["totals"]
    L = ["# Census of stored-track correspondence across Africa (declared section 3)", "",
         f"{len(rec['seasons'])} seasons, 1981 to 2010 without the reserved seasons {', '.join(map(str, RESERVED))} and 2005. "
         f"Cohort tracks starting at {START_LONS[0]:g} to {START_LONS[1]:g} E. Accompanied means an observation of the other record "
         f"within {RADIUS_KM:g} km at the same six-hourly time, proximity between stored positions and not identity.", "",
         f"QTrack: {T['qtrack_n']} tracks, {T['qtrack_atlantic_side']} on the Atlantic side. AEWC v2: {T['aewc_n']} tracks, "
         f"{T['aewc_atlantic_side']} on the Atlantic side.", "",
         "| QTrack Atlantic-side tracks, by class | count |", "| --- | --- |"]
    L += [f"| {c.replace('_', ' ')} | {n} |" for c, n in T["qtrack_atlantic_classes"].items()]
    L += ["", "| QTrack tracks not on the Atlantic side, accompanied at start / at last observation | count |", "| --- | --- |"]
    L += [f"| {k.replace('True', 'yes').replace('False', 'no').replace('_', ' / ')} | {n} |" for k, n in T["qtrack_other_by_start_last"].items()]
    L += ["", f"Share of QTrack observations (to the first Atlantic-side observation or the last used) accompanied: {T['qtrack_accompanied_share']}", "",
          "| AEWC v2 tracks, by accompaniment in the first 48 hours | n | on the Atlantic side |", "| --- | --- | --- |"]
    L += [f"| {k.replace('_', ' ')} | {v['n']} | {v['atlantic_side']} |" for k, v in T["aewc_by_start_and_outcome"].items()]
    b = T["aewc_inland_end_beside_atlantic_qtrack"]
    L += ["", f"AEWC v2 tracks ending inside Africa while accompanied by a QTrack track that reaches the Atlantic side: {b['n']}, "
              f"of which {b['candidate_discontinuity']} have that QTrack track accompanied later by another AEWC v2 track "
              f"(candidate discontinuities) and {b['not_followed']} do not.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    pj, pm = (os.path.join(a.out_dir, f"continuity_census{a.tag}.{e}") for e in ("json", "md"))
    for p in (pj, pm):
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and records are never overwritten")
    seasons = [season(y) for y in SEASONS]
    rec = {"generated_by": {"script": "scripts/continuity_census.py", "sha256": SCRIPT_SHA256_AT_LOAD,
                            "declared": "CONTINUITY_ACROSS_AFRICA_2026-11-09.md", "radius_km": RADIUS_KM, "start_hours": START_HOURS,
                            "start_lons": list(START_LONS), "reserved": list(RESERVED), "held_out": list(HELD_OUT)},
           "seasons": seasons, "totals": totals(seasons)}
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump(rec, fh)
    text = tables_text(rec)
    with open(pm, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
