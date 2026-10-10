#!/usr/bin/env python3
"""The R1 measurement pilot: the design's measurements for R1 under the frozen configuration.

Reads the declared reference (`eval60_r1_pilot_reference_1990.json`), the traced replay of
the frozen candidate (`scripts/pilot_alledge_replay.py` run with that file as its trace
family and the retained record as `--reproduces`) and the retained finished record, and
writes the measurements of `EASTERN_EVALUATION_DESIGN_2026-10-08.md` section 5 for R1's
segments A and B at the inspected maps only: detection per map, active association in the
three coverage categories (sequential coverage, overlapping coverage, ambiguous proximity)
with a confirmed duplicate requiring the same well-formed candidate region digest, the
join reported with the positions and the lifecycle of the histories on its two maps, the
tail reported, the standing band's produced histories inspected with their per-step
displacements and a NUMERICAL displacement category at both stationary parameters (the
design's section 6 classes need the field reading and are not assigned here), and the
finished stage reported separately with raw and smoothed positions side by side. It
computes nothing the design does not name and labels no count as fragmentation.

    python3 scripts/pilot_r1_measurement.py --reference <json> --trace <trace.json.gz> \\
        --finished <retained.json.gz> --out <record.json>

THE BINDING IS RECORD CONSISTENCY, not proof of execution provenance. It refuses unless
the trace names the declared file as its family, claims reproduction, carries the retained
file's digest (the replay's convention, the digest of the decompressed bytes), holds the
retained finished histories complete and equal in every field, and carries the retained
record's configuration. THE INPUT CONTRACT is then checked once, before any measurement:
every logged step carries its candidate, live, seed and claim lists (an empty list is a
valid absence, a missing list is missing evidence), every candidate and raw history
carries a well-formed region digest and finite positions, every raw history has aligned
nonempty arrays with increasing steps, every logged observation of a retained history
agrees with that history, and every retained history that finished agrees with its bound
finished record. An inspected map of a segment or of the standing band missing from the
logged steps, or a produced history without its raw history, refuses. Missing or
malformed evidence never reads as an inspected absence.
"""
import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import re

EPOCH = dt.datetime(1900, 1, 1)
FINISHED_KEYS = ("birth", "time", "lat", "lon", "steps", "raw_lat", "raw_lon", "n_points", "region_sha256")
CONFIGURATION_KEYS = ("configuration", "preparation", "margin_edges", "tracker_flags", "year")
STATIONARY_DEG = (3.0, 5.0)      # the design's section 7, the stationary net displacement, default and alternative
ENTRY_LISTS = ("candidates", "live_before", "seeds", "claims")
HISTORY_LISTS = ("steps", "lon_claimed", "lat_claimed", "n_points", "region_sha256")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SECTION_6 = "undetermined pending the field reading"


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def sha256_decompressed(path):
    """The replay's convention for a retained .json.gz: the digest of the decompressed bytes."""
    with gzip.open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def is_digest(value):
    return isinstance(value, str) and bool(HEX64.fullmatch(value))


def finite(*values):
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)


def date_key(text):
    """'1990-09-08T00' and '1990-09-08T00:00Z' both become '1990-09-08T00'."""
    return text[:13]


def inspected_dates(window):
    """The 12-hourly dates (00Z and 12Z) from the window's first date to its last, inclusive,
    enumerated from the declaration and not from what a trace happens to hold."""
    t0 = dt.datetime.strptime(window[0], "%Y-%m-%dT%H")
    t1 = dt.datetime.strptime(window[1], "%Y-%m-%dT%H")
    out, t = [], t0
    while t <= t1:
        if t.hour in (0, 12):
            out.append(t.strftime("%Y-%m-%dT%H"))
        t += dt.timedelta(hours=6)
    return out


def axis_distance(lat, lon, axis_lon, lat_range):
    """Euclidean distance in coordinate degrees from a position to the axis segment at
    `axis_lon` over `lat_range`."""
    lo, hi = float(lat_range[0]), float(lat_range[1])
    dlat = max(lo - float(lat), 0.0, float(lat) - hi)
    return math.hypot(float(lon) - float(axis_lon), dlat)


def classify(distance, tol, alt):
    if distance <= tol:
        return "within"
    if distance <= alt:
        return "ambiguous"
    return "none"


def in_box(lat, lon, box):
    return bool(box["lat"][0] <= lat <= box["lat"][1] and box["lon"][0] <= lon <= box["lon"][1])


def validate_history(birth, h):
    """A raw history's arrays are present, aligned, nonempty, with increasing steps, finite
    positions and well-formed digests."""
    for key in HISTORY_LISTS:
        if not isinstance(h.get(key), list):
            raise SystemExit(f"REFUSED: the raw history of {birth} lacks {key}")
    n = len(h["steps"])
    if n == 0 or any(len(h[key]) != n for key in HISTORY_LISTS):
        raise SystemExit(f"REFUSED: the raw history of {birth} is empty or not aligned")
    if any(not isinstance(s, int) or isinstance(s, bool) for s in h["steps"]) or any(b <= a for a, b in zip(h["steps"][:-1], h["steps"][1:])):
        raise SystemExit(f"REFUSED: the raw history of {birth} has steps that are not increasing integers")
    if not finite(*h["lat_claimed"], *h["lon_claimed"]):
        raise SystemExit(f"REFUSED: the raw history of {birth} has a non-finite position")
    if not all(is_digest(d) for d in h["region_sha256"]):
        raise SystemExit(f"REFUSED: the raw history of {birth} has a malformed region digest")


def validate_trace(trace, finished_rec):
    """The input contract, checked once before any measurement."""
    tr = trace.get("trace")
    if not isinstance(tr, dict) or not isinstance(tr.get("steps"), list):
        raise SystemExit("REFUSED: the trace has no logged steps")
    for name in ("histories", "fates", "pruned_at", "finished_index"):
        if not isinstance(tr.get(name), dict):
            raise SystemExit(f"REFUSED: the trace lacks {name}")
    histories = tr["histories"]
    for birth, h in histories.items():
        if not isinstance(h, dict):
            raise SystemExit(f"REFUSED: the raw history of {birth} is not a record")
        validate_history(birth, h)
    dates = [date_key(e.get("date", "")) for e in tr["steps"]]
    steps = [e.get("step") for e in tr["steps"]]
    if len(set(dates)) != len(dates) or len(set(steps)) != len(steps):
        raise SystemExit("REFUSED: the trace holds a duplicate logged step or date")
    if steps != sorted(steps, key=lambda s: (s is None, s)) or dates != sorted(dates):
        raise SystemExit("REFUSED: the trace's logged steps are out of order")
    by_step = {e.get("step"): e for e in tr["steps"]}
    for birth, h in histories.items():
        for i, s in enumerate(h["steps"]):
            e = by_step.get(s)
            if e is None:
                continue
            found = [o for kind in ("seeds", "claims") for o in (e.get(kind) or []) if o.get("birth") == birth]
            if not found or found[0].get("lat") != h["lat_claimed"][i] or found[0].get("lon") != h["lon_claimed"][i]:
                raise SystemExit(f"REFUSED: the retained history {birth} observed at logged step {s} but the step holds no seed or claim for it at that position")
    for e in tr["steps"]:
        label = e.get("date", "?")
        for key in ("step", "date", "n_live_before") + ENTRY_LISTS:
            if key not in e:
                raise SystemExit(f"REFUSED: the logged step {label} lacks {key}")
        for key in ENTRY_LISTS:
            if not isinstance(e[key], list):
                raise SystemExit(f"REFUSED: the logged step {label} carries {key} that is not a list")
        indices = set()
        for c in e["candidates"]:
            for key in ("index", "lat_mean", "lon_mean", "n_points", "region_sha256"):
                if key not in c:
                    raise SystemExit(f"REFUSED: a candidate at {label} lacks {key}")
            if not finite(c["lat_mean"], c["lon_mean"]):
                raise SystemExit(f"REFUSED: a candidate at {label} has a non-finite position")
            if not is_digest(c["region_sha256"]):
                raise SystemExit(f"REFUSED: a candidate at {label} has a malformed region digest")
            indices.add(c["index"])
        for kind in ("seeds", "claims"):
            for o in e[kind]:
                for key in ("birth", "candidate", "lat", "lon"):
                    if key not in o:
                        raise SystemExit(f"REFUSED: a {kind[:-1]} at {label} lacks {key}")
                if o["candidate"] not in indices or not finite(o["lat"], o["lon"]):
                    raise SystemExit(f"REFUSED: a {kind[:-1]} at {label} names no logged candidate or has a non-finite position")
                h = histories.get(o["birth"])
                if h is not None:
                    try:
                        i = h["steps"].index(e["step"])
                    except ValueError:
                        raise SystemExit(f"REFUSED: the {kind[:-1]} of {o['birth']} at {label} is not in its raw history")
                    if h["lat_claimed"][i] != o["lat"] or h["lon_claimed"][i] != o["lon"]:
                        raise SystemExit(f"REFUSED: the {kind[:-1]} of {o['birth']} at {label} disagrees with its raw history")
                    c = next(c for c in e["candidates"] if c["index"] == o["candidate"])
                    if c["region_sha256"] != h["region_sha256"][i] or c["n_points"] != h["n_points"][i]:
                        raise SystemExit(f"REFUSED: the candidate claimed by {o['birth']} at {label} disagrees with its raw history in region or point count")
        for row in e["live_before"]:
            for key in ("birth", "last_lat", "last_lon", "last_step", "n_obs"):
                if key not in row:
                    raise SystemExit(f"REFUSED: a live row at {label} lacks {key}")
            if not finite(row["last_lat"], row["last_lon"]):
                raise SystemExit(f"REFUSED: a live row at {label} has a non-finite position")
    finished = finished_rec["finished"]
    for idx, f in enumerate(finished):
        if f["birth"] in histories and tr["finished_index"].get(f["birth"]) != idx:
            raise SystemExit(f"REFUSED: the finished history {f['birth']} is retained in the trace without its finished index")
    for birth, idx in tr["finished_index"].items():
        h = histories.get(birth)
        if h is None:
            raise SystemExit(f"REFUSED: the finished history {birth} has no raw history in the trace")
        if not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx < len(finished):
            raise SystemExit(f"REFUSED: the finished index of {birth} is out of range")
        f = finished[idx]
        if (f["birth"] != birth or f["steps"] != h["steps"] or f["raw_lat"] != h["lat_claimed"] or f["raw_lon"] != h["lon_claimed"]
                or f["region_sha256"] != h["region_sha256"] or f["n_points"] != h["n_points"]):
            raise SystemExit(f"REFUSED: the raw history of {birth} disagrees with its bound finished record")


def observations(entry):
    """What each live history observed at this step, seeds and claims alike, with the
    candidate's region digest so identity evidence is available."""
    cands = {c["index"]: c for c in entry["candidates"]}
    out = []
    for kind in ("seeds", "claims"):
        for o in entry[kind]:
            c = cands[o["candidate"]]
            out.append({"birth": o["birth"], "kind": kind[:-1], "candidate": int(o["candidate"]), "lat": float(o["lat"]), "lon": float(o["lon"]),
                        "region_sha256": c["region_sha256"], "n_points": int(c["n_points"])})
    return out


def measure_map(entry, m, ref, tol, alt):
    """Detection and active association at one inspected map, against its declared axis."""
    box = ref["case_box"]
    cands = []
    for c in entry["candidates"]:
        d = axis_distance(c["lat_mean"], c["lon_mean"], m["axis_lon"], m["lat_range"])
        cands.append({"index": c["index"], "lat": float(c["lat_mean"]), "lon": float(c["lon_mean"]), "n_points": int(c["n_points"]),
                      "region_sha256": c["region_sha256"], "distance_deg": round(d, 3), "class": classify(d, tol, alt),
                      "in_case_box": in_box(float(c["lat_mean"]), float(c["lon_mean"]), box)})
    nearest = min(cands, key=lambda c: c["distance_deg"]) if cands else None
    obs = []
    for o in observations(entry):
        d = axis_distance(o["lat"], o["lon"], m["axis_lon"], m["lat_range"])
        obs.append({**o, "distance_deg": round(d, 3), "class": classify(d, tol, alt)})
    observed = {o["birth"] for o in obs}
    alive_without = []
    for row in entry["live_before"]:
        if row["birth"] in observed:
            continue
        d = axis_distance(row["last_lat"], row["last_lon"], m["axis_lon"], m["lat_range"])
        if d <= tol:
            alive_without.append({"birth": row["birth"], "last_step": int(row["last_step"]), "last_lat": float(row["last_lat"]), "last_lon": float(row["last_lon"]),
                                  "distance_deg": round(d, 3), "n_obs": int(row["n_obs"]), "due": row.get("due")})
    return {"date": date_key(entry["date"]), "step": int(entry["step"]), "segment": m["segment"], "scored": bool(m["scored"]),
            "axis_lon": m["axis_lon"], "lat_range": m["lat_range"],
            "detection": {"candidates_in_case_box": sum(1 for c in cands if c["in_case_box"]), "candidates_total": len(cands),
                          "within": [c for c in cands if c["class"] == "within"], "ambiguous": [c for c in cands if c["class"] == "ambiguous"],
                          "nearest": nearest},
            "association": {"observations_within": [o for o in obs if o["class"] == "within"], "observations_ambiguous": [o for o in obs if o["class"] == "ambiguous"],
                            "alive_within_without_observation": alive_without, "n_live_before": int(entry["n_live_before"])}}


def segment_summary(results, segment, tol_name):
    """The three coverage categories for one segment, kept apart, with no count labeled
    fragmentation and an overlap a confirmed duplicate only with the same well-formed
    region digest claimed by both histories at that map."""
    maps = [r for r in results if r["segment"] == segment]
    per_birth, ambiguous_obs, alive_only, digests = {}, {}, {}, {}
    for r in maps:
        for o in r["association"]["observations_within"]:
            per_birth.setdefault(o["birth"], []).append(r["date"])
            digests[(o["birth"], r["date"])] = o["region_sha256"]
        for o in r["association"]["observations_ambiguous"]:
            ambiguous_obs.setdefault(o["birth"], []).append(r["date"])
        for a in r["association"]["alive_within_without_observation"]:
            alive_only.setdefault(a["birth"], []).append(r["date"])
    dates = [r["date"] for r in maps]
    overlaps, confirmed = [], []
    for r in maps:
        births = sorted({o["birth"] for o in r["association"]["observations_within"]})
        if len(births) >= 2:
            overlaps.append({"date": r["date"], "births": births})
            for i in range(len(births)):
                for j in range(i + 1, len(births)):
                    di, dj = digests[(births[i], r["date"])], digests[(births[j], r["date"])]
                    if is_digest(di) and di == dj:
                        confirmed.append({"date": r["date"], "births": [births[i], births[j]], "region_sha256": di})
    covered = sorted({d for ds in per_birth.values() for d in ds})
    return {"segment": segment, "tolerance": tol_name, "inspected_maps": dates,
            "sequential_coverage": {b: sorted(ds) for b, ds in sorted(per_birth.items())},
            "one_history_observes_every_map": [b for b, ds in per_birth.items() if set(ds) == set(dates)],
            "maps_covered": covered, "maps_uncovered": [d for d in dates if d not in covered],
            "coverage_share_any_history": round(len(covered) / len(dates), 3) if dates else None,
            "overlapping_coverage": overlaps, "confirmed_duplicates": confirmed,
            "ambiguous_proximity": {"observations_between_tolerance_and_alternative": {b: sorted(ds) for b, ds in sorted(ambiguous_obs.items())},
                                    "alive_within_tolerance_without_observation": {b: sorted(ds) for b, ds in sorted(alive_only.items())},
                                    "histories_in_either": len(set(ambiguous_obs) | set(alive_only))},
            "histories_touching_count": len(set(per_birth) | set(ambiguous_obs) | set(alive_only)),
            "note": "a count of histories is a count, not fragmentation; an overlap is a confirmed duplicate only when the same well-formed region digest was claimed at the same map"}


def lifecycle(birth, histories, fates, pruned_at, finished_index):
    h = histories.get(birth)
    return {"fate": fates.get(birth), "pruned_at": pruned_at.get(birth), "finished_index": finished_index.get(birth),
            "n_obs": len(h["steps"]) if h else None, "last_step": int(h["steps"][-1]) if h else None, "raw_history_retained": h is not None}


def join_report(results, histories, fates, pruned_at, finished_index):
    """The transition between the last A map and the first B map: the histories observing
    within tolerance on each, with their positions, their region and their lifecycle, and
    for each history on the last A map whether it continues, ends, or observes elsewhere,
    whether the B map's histories are newcomers, and a split where a continuing history
    and a newcomer claim the same region. Reported, never scored."""
    a_maps = [r for r in results if r["segment"] == "A"]
    b_maps = [r for r in results if r["segment"] == "B"]
    if not a_maps or not b_maps:
        raise SystemExit("REFUSED: the join needs a last A map and a first B map")
    a_last, b_first = a_maps[-1], b_maps[0]
    obs_a = a_last["association"]["observations_within"]
    obs_b = b_first["association"]["observations_within"]
    on_b = {o["birth"]: o for o in obs_b}
    rows_a = []
    for o in obs_a:
        birth = o["birth"]
        h = histories.get(birth)
        if birth in on_b:
            outcome = "continues"
        elif h is None:
            outcome = "unknown, raw history not retained"
        elif int(h["steps"][-1]) <= a_last["step"]:
            outcome = "ends"
        else:
            outcome = "observes elsewhere after the last A map"
        rows_a.append({"birth": birth, "position_on_last_A_map": [o["lat"], o["lon"]], "distance_deg": o["distance_deg"], "region_sha256": o["region_sha256"],
                       "outcome": outcome, "position_on_first_B_map": [on_b[birth]["lat"], on_b[birth]["lon"]] if birth in on_b else None,
                       **lifecycle(birth, histories, fates, pruned_at, finished_index)})
    a_births = {r["birth"] for r in rows_a}
    rows_b = [{"birth": o["birth"], "position_on_first_B_map": [o["lat"], o["lon"]], "distance_deg": o["distance_deg"], "region_sha256": o["region_sha256"],
               "newcomer": o["birth"] not in a_births, **lifecycle(o["birth"], histories, fates, pruned_at, finished_index)} for o in obs_b]
    continuing = [r["birth"] for r in rows_a if r["outcome"] == "continues"]
    newcomers = [r["birth"] for r in rows_b if r["newcomer"]]
    splits = []
    for c in continuing:
        for r in rows_b:
            if r["newcomer"] and is_digest(r["region_sha256"]) and r["region_sha256"] == on_b[c]["region_sha256"]:
                splits.append({"continuing": c, "newcomer": r["birth"], "region_sha256": r["region_sha256"]})
    if not rows_a and not rows_b:
        transition = "no history within tolerance on either map"
    elif continuing and splits:
        transition = "continues, with a split"
    elif continuing:
        transition = "continues"
    elif rows_a and newcomers:
        transition = "replaced"
    elif rows_a:
        transition = "ends or observes elsewhere, nothing within tolerance on the first B map"
    else:
        transition = "nothing within tolerance on the last A map, newcomers on the first B map"
    return {"last_A_map": a_last["date"], "first_B_map": b_first["date"], "histories_on_last_A_map": rows_a, "histories_on_first_B_map": rows_b,
            "continuing": continuing, "newcomers_on_first_B_map": newcomers, "splits": splits, "transition": transition,
            "note": "the identity across the join is unresolved in the fields; this is what the configuration did, not a score"}


def displacement_category(net, param):
    """A numerical category from the net longitude displacement against the parameter. It
    is not the design's section 6 class, which needs the field reading."""
    if abs(net) <= param:
        return "small_net_displacement"
    if net > param:
        return "east_moving"
    return "west_moving"


def standing_band_report(trace_steps, ref, histories, fates, pruned_at, finished_index, a_results, tol, a_results_alt=None, alt=None,
                         stationary=STATIONARY_DEG):
    """Produced histories observing inside the standing band at its inspected maps, which
    are enumerated from the declaration and must all be logged, with the band's candidate
    counts per map, each history's raw history, per-step displacements, net displacement,
    numerical displacement category at both stationary parameters, the design's section
    6 class left undetermined pending the field reading, fate, and whether it continues
    into A at the tolerance and at its alternative. Refuses a missing map or a missing or
    malformed raw history."""
    band = ref["standing_band"]
    expected = inspected_dates(band["window"])
    entries = {date_key(e["date"]): e for e in trace_steps}
    missing = [d for d in expected if d not in entries]
    if missing:
        raise SystemExit(f"REFUSED: the standing band's inspected maps {missing} are not among the trace's logged steps")
    box = {"lat": band["lat"], "lon": band["lon"]}
    candidates_per_map = {d: sum(1 for c in entries[d]["candidates"] if in_box(float(c["lat_mean"]), float(c["lon_mean"]), box)) for d in expected}
    seen = {}
    for d in expected:
        for o in observations(entries[d]):
            if in_box(o["lat"], o["lon"], box):
                seen.setdefault(o["birth"], []).append(d)
    a_births = {o["birth"] for r in a_results for o in r["association"]["observations_within"]}
    a_births_alt = {o["birth"] for r in (a_results_alt or []) for o in r["association"]["observations_within"]}
    out, counts = [], {str(p): {} for p in stationary}
    for birth, dates in sorted(seen.items()):
        h = histories.get(birth)
        if h is None:
            raise SystemExit(f"REFUSED: the raw history of produced history {birth} is not retained in the trace")
        validate_history(birth, h)
        lons = [float(x) for x in h["lon_claimed"]]
        net = round(lons[-1] - lons[0], 3)
        cats = {str(p): displacement_category(net, p) for p in stationary}
        for p, c in cats.items():
            counts[p][c] = counts[p].get(c, 0) + 1
        out.append({"birth": birth, "maps_observed_in_band": sorted(dates), "fate": fates.get(birth), "pruned_at": pruned_at.get(birth), "finished_index": finished_index.get(birth),
                    "continues_into_A_within_tolerance": birth in a_births,
                    "continues_into_A_within_alternative": (birth in a_births_alt) if a_results_alt is not None else None,
                    "raw_history": {"steps": h["steps"], "lon_claimed": h["lon_claimed"], "lat_claimed": h["lat_claimed"]},
                    "per_step_lon_displacement_deg": [round(b - a, 3) for a, b in zip(lons[:-1], lons[1:])],
                    "net_lon_displacement_deg": net, "n_obs": len(h["steps"]),
                    "displacement_category_by_stationary_parameter": cats, "section_6_class": SECTION_6})
    return {"band": band, "inspected_maps": expected, "candidates_in_band_per_map": candidates_per_map, "produced_histories": out,
            "displacement_category_counts_by_stationary_parameter": counts, "stationary_parameters_deg": list(stationary), "tolerance_deg": tol, "alternative_deg": alt,
            "note": "inspected, not scored; continuity from the band into A is reported and not labeled false; the displacement category is numerical and the design's section 6 class needs the field reading"}


def finished_report(finished, ref, step_of_date, tol, alt):
    """Finished histories with a raw position within the alternative tolerance at an
    inspected map, raw and smoothed positions side by side."""
    maps = [m for m in ref["maps"]]
    out = []
    for idx, h in enumerate(finished):
        pos = {int(s): i for i, s in enumerate(h["steps"])}
        rows = []
        for m in maps:
            s = step_of_date.get(m["date"])
            if s is None or s not in pos:
                continue
            i = pos[s]
            raw_d = axis_distance(h["raw_lat"][i], h["raw_lon"][i], m["axis_lon"], m["lat_range"])
            sm_d = axis_distance(h["lat"][i], h["lon"][i], m["axis_lon"], m["lat_range"])
            rows.append({"date": m["date"], "segment": m["segment"], "scored": bool(m["scored"]), "raw": [round(float(h["raw_lat"][i]), 3), round(float(h["raw_lon"][i]), 3)],
                         "raw_distance_deg": round(raw_d, 3), "raw_class": classify(raw_d, tol, alt),
                         "smoothed": [round(float(h["lat"][i]), 3), round(float(h["lon"][i]), 3)], "smoothed_distance_deg": round(sm_d, 3), "smoothed_class": classify(sm_d, tol, alt)})
        if not any(r["raw_class"] != "none" for r in rows):
            continue
        lons = [float(x) for x in h["raw_lon"]]
        crosses = any(lons[i] >= 40.0 and min(lons[i + 1:], default=99.0) < 40.0 for i in range(len(lons) - 1))
        cover = {}
        for seg in ("A", "B"):
            seg_maps = [m["date"] for m in maps if m["segment"] == seg]
            got = [r["date"] for r in rows if r["segment"] == seg and r["raw_class"] == "within"]
            cover[seg] = {"maps_within_raw": got, "share": round(len(got) / len(seg_maps), 3)}
        out.append({"finished_index": idx, "birth": h["birth"], "n_obs": len(h["steps"]), "first_step": int(h["steps"][0]), "last_step": int(h["steps"][-1]),
                    "first_raw": [round(float(h["raw_lat"][0]), 3), round(float(h["raw_lon"][0]), 3)], "last_raw": [round(float(h["raw_lat"][-1]), 3), round(float(h["raw_lon"][-1]), 3)],
                    "crosses_40E_raw": bool(crosses), "coverage": cover, "at_inspected_maps": rows})
    return out


def check_binding(trace, reference_path, finished_rec, finished_path):
    """Record consistency between the trace, the declaration and the retained record. It is
    not proof that the trace was executed from them."""
    fam = trace.get("trace") or {}
    if fam.get("families_sha256") != sha256_of(reference_path):
        raise SystemExit("REFUSED: the trace was not run with the declared reference file as its family")
    rep = trace.get("reproduces") or {}
    if not rep.get("finished_identical"):
        raise SystemExit("REFUSED: the trace did not reproduce a retained finished record")
    if rep.get("sha256") != sha256_decompressed(finished_path):
        raise SystemExit(f"REFUSED: the trace's reproduction digest is not {finished_path}'s")
    mine, theirs = trace.get("finished"), finished_rec.get("finished")
    if not isinstance(mine, list) or not isinstance(theirs, list) or len(mine) != len(theirs):
        raise SystemExit(f"REFUSED: the trace's finished histories are not {finished_path}'s")
    for a, b in zip(mine, theirs):
        for key in FINISHED_KEYS:
            if key not in a or key not in b or a[key] != b[key]:
                raise SystemExit(f"REFUSED: the trace's finished histories are not {finished_path}'s in every field ({key})")
    for key in CONFIGURATION_KEYS:
        if trace.get(key) != finished_rec.get(key):
            raise SystemExit(f"REFUSED: the trace's configuration differs from the retained record's ({key})")
    for key in ("coarse", "fine"):
        if (trace.get("thresholds") or {}).get(key) != (finished_rec.get("thresholds") or {}).get(key):
            raise SystemExit(f"REFUSED: the trace's {key} threshold differs from the retained record's")
    for run in ("B", "wide"):
        if ((trace.get("runs") or {}).get(run) or {}).get("case_sha256") != ((finished_rec.get("runs") or {}).get(run) or {}).get("case_sha256"):
            raise SystemExit(f"REFUSED: the trace's {run} case differs from the retained record's")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--reference", "--trace", "--finished", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--trace-digest", required=True,
                    help="the fingerprint the replay logged for the trace it wrote (the first twelve characters, at least, of the digest of its "
                         "decompressed bytes); the trace's digest must begin with it. This checks agreement with the logged fingerprint, not the "
                         "correctness or completeness of the replay's logging")
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    trace_digest = sha256_decompressed(args.trace)
    if len(args.trace_digest) < 12 or not trace_digest.startswith(args.trace_digest):
        raise SystemExit("REFUSED: the trace's digest does not begin with the one the replay logged")
    ref_all = json.load(open(args.reference))
    ref, tols = ref_all["reference"], ref_all["tolerances"]
    tol, alt = float(tols["positional_deg"]), float(tols["alternative_deg"])
    with gzip.open(args.trace, "rb") as fh:
        trace = json.loads(fh.read())
    with gzip.open(args.finished, "rb") as fh:
        finished_rec = json.loads(fh.read())
    check_binding(trace, args.reference, finished_rec, args.finished)
    validate_trace(trace, finished_rec)
    tr = trace["trace"]
    entries = {date_key(e["date"]): e for e in tr["steps"]}
    step_of_date = {d: int(e["step"]) for d, e in entries.items()}
    for m in ref["maps"]:
        if m["date"] not in entries:
            raise SystemExit(f"REFUSED: the inspected map {m['date']} is not among the trace's logged steps")
    results = {}
    for name, t in (("tolerance", tol), ("alternative", alt)):
        per_map = [measure_map(entries[m["date"]], m, ref, t, alt if name == "tolerance" else t) for m in ref["maps"]]
        seg_a, seg_b = segment_summary(per_map, "A", name), segment_summary(per_map, "B", name)
        results[name] = {"degrees": t, "maps": per_map, "segments": {"A": seg_a, "B": seg_b},
                         "join": join_report(per_map, tr["histories"], tr["fates"], tr["pruned_at"], tr["finished_index"]),
                         "tail": [r for r in per_map if r["segment"] == "tail"]}
    standing = standing_band_report(tr["steps"], ref, tr["histories"], tr["fates"], tr["pruned_at"], tr["finished_index"],
                                    [r for r in results["tolerance"]["maps"] if r["segment"] == "A"], tol,
                                    [r for r in results["alternative"]["maps"] if r["segment"] == "A"], alt)
    near_births = sorted({b for name in ("A", "B") for b in results["tolerance"]["segments"][name]["sequential_coverage"]})
    live_fates = {b: lifecycle(b, tr["histories"], tr["fates"], tr["pruned_at"], tr["finished_index"]) for b in near_births}
    fin = finished_report(finished_rec["finished"], ref, step_of_date, tol, alt)
    summary = {name: {"maps_with_a_candidate_within": [m["date"] for m in results[name]["maps"] if m["scored"] and m["detection"]["within"]],
                      "segment_coverage": {s: results[name]["segments"][s]["maps_covered"] for s in ("A", "B")},
                      "ambiguous_proximity": {s: {"observations_between_tolerance_and_alternative": len(results[name]["segments"][s]["ambiguous_proximity"]["observations_between_tolerance_and_alternative"]),
                                                  "alive_within_tolerance_without_observation": len(results[name]["segments"][s]["ambiguous_proximity"]["alive_within_tolerance_without_observation"]),
                                                  "histories_in_either": results[name]["segments"][s]["ambiguous_proximity"]["histories_in_either"]} for s in ("A", "B")},
                      "confirmed_duplicates": sum(len(results[name]["segments"][s]["confirmed_duplicates"]) for s in ("A", "B")),
                      "join_transition": results[name]["join"]["transition"]} for name in results}
    summary["standing_band_displacement_categories"] = standing["displacement_category_counts_by_stationary_parameter"]
    summary["standing_band_positive_net_displacements"] = sum(1 for p in standing["produced_histories"] if p["net_lon_displacement_deg"] > 0)
    summary["finished_histories_near_an_inspected_map"] = len(fin)
    out = {"generated_by": "scripts/pilot_r1_measurement.py", "script_sha256": sha256_of(os.path.abspath(__file__)),
           "inputs": {"reference": {"path": args.reference, "sha256": sha256_of(args.reference)},
                      "trace": {"path": args.trace, "sha256": sha256_of(args.trace), "sha256_decompressed": trace_digest, "declared_digest": args.trace_digest,
                                "bound_to_declared_digest": True,
                                "limit": "agreement with the fingerprint the replay logged, not the correctness or completeness of the replay's logging"},
                      "finished": {"path": args.finished, "sha256": sha256_of(args.finished), "sha256_decompressed": sha256_decompressed(args.finished)}},
           "binding": ("the trace is bound to the digest the replay logged when --trace-digest is given, and to the retained record and the declaration by "
                       "record consistency; the structural checks are a bounded list, not a guarantee against every damaged input"),
           "input_contract_checked": True,
           "reference_declared": ref_all, "scored_only_at_inspected_maps": True,
           "replay": {"reproduces": trace.get("reproduces"), "elapsed_seconds": trace.get("elapsed_seconds"), "logged_steps": len(tr["steps"]), "configuration": trace.get("configuration")},
           "summary": summary, "results": results, "standing_band": standing,
           "finished_stage": {"note": "reported separately; raw and smoothed positions side by side; not required for A or B",
                              "fates_of_histories_observing_within_tolerance": live_fates, "finished_histories_near_an_inspected_map": fin}}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for name in ("tolerance", "alternative"):
        r = results[name]
        print(f"[{name} {r['degrees']} deg]")
        for m in r["maps"]:
            d = m["detection"]; a = m["association"]
            print(f"  {m['date']} {m['segment']:4s} scored={m['scored']} candidates in box {d['candidates_in_case_box']}, within {len(d['within'])}, ambiguous {len(d['ambiguous'])}, "
                  f"nearest {d['nearest']['distance_deg'] if d['nearest'] else None} deg; observations within {len(a['observations_within'])}, ambiguous {len(a['observations_ambiguous'])}, "
                  f"alive-without {len(a['alive_within_without_observation'])}")
        for seg in ("A", "B"):
            s = r["segments"][seg]
            print(f"  segment {seg}: covered {s['maps_covered']} uncovered {s['maps_uncovered']} one-history-all {s['one_history_observes_every_map']} "
                  f"overlaps {len(s['overlapping_coverage'])} confirmed duplicates {len(s['confirmed_duplicates'])} histories touching {s['histories_touching_count']} "
                  f"(ambiguous observations {len(s['ambiguous_proximity']['observations_between_tolerance_and_alternative'])}, alive without observation {len(s['ambiguous_proximity']['alive_within_tolerance_without_observation'])})")
        print(f"  join: {r['join']['transition']} (on last A map {len(r['join']['histories_on_last_A_map'])}, on first B map {len(r['join']['histories_on_first_B_map'])})")
    print(f"standing band: {len(standing['produced_histories'])} produced histories; displacement categories {standing['displacement_category_counts_by_stationary_parameter']}; "
          f"positive net displacements {summary['standing_band_positive_net_displacements']}; candidates per map {standing['candidates_in_band_per_map']}")
    print(f"finished stage: {len(fin)} finished histories with a raw position within {alt} deg at an inspected map. Record {args.out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
