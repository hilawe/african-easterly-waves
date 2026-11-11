#!/usr/bin/env python3
"""The repaired evaluation of the continuity effort, declared in section 18 of
`CONTINUITY_ACROSS_AFRICA_2026-11-09.md`. Pure measures over one set of histories and the QTrack
record of a season. Nothing here replays a season or decides an outcome.

REFERENCE MEASURES are taken on a population no method changes, the QTrack cohort tracks starting
between 10 and 30 E, each read to its first Atlantic-side observation or its last used one. They are
the accompanied observations, section 3's classes, the longest accompaniment of each reference
track by a single history, and the histories accompanying it.

FLAGS repair section 15's classifier in two places. A track is flagged when ANY of its transitions
is a switch or undetermined, whatever its display class, and the display class puts undetermined
ahead of relay. No flag is reported against the number of tracks, which unaccompanied tracks
inflate. They are counts of tracks and transitions, the QTrack systems left behind by a switch,
the reference tracks a flagged history accompanies, and the exposure (pairs of successive
accompanied observations within tracks). A relay is never cleared.

THE EXPOSURE IS PROTECTED AGAINST UNACCOMPANIED OUTPUT ONLY (narrowed in section 24 of the record).
A track that is never beside a QTrack system adds no pair. A history that is beside one adds pairs,
so copies of an unflagged accompanied history lower a rate per pair, one switch in 6 pairs becoming
one in 306 with 100 copies. A rate per pair is conditional on the output it is computed from and
cannot by itself show one configuration to be safer than another.

LINK SUPPORT comes from the tracker's own predictions. A link is a jump link when the claimed
position lies 500 km or more from the prediction it was matched under. A switch is a supported
false-link candidate when its interval holds a jump link and a system left behind lies within
500 km of that link's prediction at that time. That is agreement between two records of position
and not a physical judgment.

LINK STATUS reads one link against QTrack. A link whose two ends are both accompanied is consistent
when a QTrack system is beside both, and otherwise a switch, a relay or undetermined by the rule a
transition is classed with. A link with an unaccompanied end is unassessed, which is not a pass.

A history is a dict with `id`, `time`, `lat`, `lon` (the positions of the set, as arrays), `step`,
`raw_lat`, `raw_lon` (the unsmoothed positions) and `est` (for each observation after the first,
the prediction it was matched under, as [lat, lon, hours]).
"""

import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import continuity_census as C  # noqa: E402
import continuity_join_test as J  # noqa: E402
import continuity_merge_variants as V  # noqa: E402
import continuity_variant_replay as R  # noqa: E402

with open(os.path.abspath(__file__), "rb") as _fh:
    SCRIPT_SHA256_AT_LOAD = hashlib.sha256(_fh.read()).hexdigest()

RADIUS_KM = C.RADIUS_KM
RISK_KINDS = ("switch", "undetermined")
DISPLAY_ORDER = ("switch", "undetermined", "relay", "addition_only", "no_transition")
EVENT_KINDS = ("switch", "relay", "undetermined", "addition")
ELAPSED = ("to_12_hours", "18_to_24_hours", "over_24_hours")
LINK_STATUS = ("consistent", "switch", "relay", "undetermined", "unassessed")


def display_class(events):
    """The class a track is shown under, undetermined ahead of relay."""
    kinds = {e["kind"] for e in events}
    for k in ("switch", "undetermined", "relay"):
        if k in kinds:
            return k
    return "addition_only" if kinds else "no_transition"


def flagged(events):
    """True when any transition is a switch or undetermined, whatever else the track holds."""
    return any(e["kind"] in RISK_KINDS for e in events)


def events_of(track, q_index, q_spans):
    """Section 15's transitions, each with the observation indices its interval runs between, and
    the track's exposure, the number of pairs of successive accompanied observations."""
    acc = [i for i, (la, lo, d) in enumerate(zip(track["lat"], track["lon"], track["time"])) if C.companions(la, lo, d, q_index)]
    where = {C.time_key(track["time"][i]): k for k, i in enumerate(acc)}
    out = []
    for e in J.transitions(track, q_index, q_spans):
        k = where[C.time_key(e["time"])]
        if k == 0:
            raise SystemExit(f"REFUSED: a transition of track {track['id']} has no previous accompanied observation")
        out.append({**e, "from": acc[k - 1], "to": acc[k]})
    return out, max(0, len(acc) - 1)


def jump(history, j, radius_km=RADIUS_KM):
    """True when observation j was claimed `radius_km` or more from the prediction it was matched under."""
    est = history["est"][j - 1]
    return float(C.km(est[0], est[1], history["raw_lat"][j], history["raw_lon"][j])) >= radius_km


def link_marks(event, history, q_spans, radius_km=RADIUS_KM):
    """The jump links in a transition's interval, and whether a system left behind lies within
    `radius_km` of a jump link's prediction at that link's time."""
    jumps, supported = 0, False
    for j in range(event["from"] + 1, event["to"] + 1):
        if not jump(history, j, radius_km):
            continue
        jumps += 1
        est, key = history["est"][j - 1], C.time_key(history["time"][j])
        for a in event["lost"]:
            p = q_spans[a]["positions"].get(key)
            if p is not None and float(C.km(est[0], est[1], p[0], p[1])) <= radius_km:
                supported = True
    return {"jump_links": jumps, "supported": supported}


def link_status(history, j, q_index, q_spans):
    """How QTrack reads the link into observation j of a history."""
    a = {tid for tid, _ in C.companions(history["lat"][j - 1], history["lon"][j - 1], history["time"][j - 1], q_index)}
    b = {tid for tid, _ in C.companions(history["lat"][j], history["lon"][j], history["time"][j], q_index)}
    if not a or not b:
        return "unassessed"
    if a & b:
        return "consistent"
    d = float(history["time"][j])
    alive = [x for x in a if q_spans[x]["last"] >= d - J.TOL]
    if any(C.time_key(d) in q_spans[x]["positions"] for x in alive):
        return "switch"
    return "undetermined" if alive else "relay"


def link_census(histories, q_index, q_spans, sector=V.SECTOR):
    """The links of the histories in the sector, by the hours of the prediction each was matched
    under and by status."""
    out = {}
    for h in R.in_sector_tracks(histories, sector):
        for j in range(1, len(h["est"]) + 1):
            row = out.setdefault(str(int(h["est"][j - 1][2])), {k: 0 for k in LINK_STATUS})
            row[link_status(h, j, q_index, q_spans)] += 1
    return dict(sorted(out.items(), key=lambda x: int(x[0])))


def elapsed_class(event, history):
    hours = round(24.0 * (float(history["time"][event["to"]]) - float(history["time"][event["from"]])))
    return ELAPSED[0] if hours <= 12 else (ELAPSED[1] if hours <= 24 else ELAPSED[2])


def reference_measures(histories, q, q_rows, band=C.START_LONS):
    """One row per reference track, with the histories accompanying each used observation."""
    idx = C.index_by_time(histories)
    rows = []
    for r in q_rows:
        if not (band[0] <= r["first"]["lon"] < band[1]):
            continue
        t = q[r["id"]]
        upto = r["first_atlantic_side"]["index"] if r["outcome"] == "atlantic_side" else r["last_used"]["index"]
        comp = [[tid for tid, _ in C.companions(t["lat"][k], t["lon"][k], t["time"][k], idx)] for k in range(upto + 1)]
        by = {}
        for k, c in enumerate(comp):
            for tid in c:
                by.setdefault(tid, []).append(k)
        row = {"id": r["id"], "outcome": r["outcome"], "steps": upto + 1, "accompanied": sum(1 for c in comp if c),
               "longest": max((len(v) for v in by.values()), default=0), "by_history": by, "nearest": [c[0] if c else None for c in comp],
               "start": sorted(C.start_set(t, idx, upto)), "event": sorted(comp[upto])}
        if r["outcome"] == "atlantic_side":
            row["class"] = C.classify_q(set(row["start"]), set(row["event"]))
        rows.append(row)
    return rows


def pool_reference(rows):
    acc = [r for r in rows if r["accompanied"]]
    return {"tracks": len(rows), "steps": sum(r["steps"] for r in rows), "accompanied": sum(r["accompanied"] for r in rows),
            "longest_single_history": sum(r["longest"] for r in rows),
            "classes": {c: sum(1 for r in rows if r.get("class") == c) for c in C.Q_CLASSES},
            "tracks_accompanied": len(acc), "histories_on_accompanied": sum(len(r["by_history"]) for r in acc)}


def flag_measures(histories, q_index, q_spans, ref_rows, sector=V.SECTOR):
    """`flags`, which adding unaccompanied histories cannot change, and `costs`, which it can, over
    the histories with a position in the sector."""
    sec = R.in_sector_tracks(histories, sector)
    display = {k: 0 for k in DISPLAY_ORDER}
    events = {k: 0 for k in EVENT_KINDS}
    elapsed = {k: 0 for k in ELAPSED}
    flagged_ids, left = set(), set()
    exposure = links = jumps = with_jump = supported = relay_jump = 0
    for h in sec:
        links += len(h["est"])
        jumps += sum(1 for j in range(1, len(h["est"]) + 1) if jump(h, j))
        ev, pairs = events_of(h, q_index, q_spans)
        exposure += pairs
        if R.multi_system(h, q_index):
            display[display_class(ev)] += 1
        if flagged(ev):
            flagged_ids.add(h["id"])
        for e in ev:
            events[e["kind"]] += 1
            m = link_marks(e, h, q_spans)
            if e["kind"] == "switch":
                left.update(e["lost"])
                elapsed[elapsed_class(e, h)] += 1
                with_jump += int(m["jump_links"] > 0)
                supported += int(m["supported"])
            elif e["kind"] == "relay":
                relay_jump += int(m["jump_links"] > 0)
    touched = sum(1 for r in ref_rows if flagged_ids & set(r["by_history"]))
    flags = {"flagged_tracks": len(flagged_ids), "display_classes": display, "events": events, "exposure_pairs": exposure,
             "systems_left_behind": len(left), "reference_tracks_with_a_flagged_history": touched,
             "switch_events_with_a_jump_link": with_jump, "supported_false_link_candidates": supported,
             "relay_events_with_a_jump_link": relay_jump, "switch_events_by_elapsed": elapsed}
    costs = {"tracks_in_sector": len(sec), "links": links, "jump_links": jumps, "stationary_tracks": sum(1 for h in sec if R.stationary(h))}
    return flags, costs
