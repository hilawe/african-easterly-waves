#!/usr/bin/env python3
"""The row-feature graph, a working prototype of a representation of eastern evolution
with geometry and explicit uncertainty (`EASTERN_REPRESENTATION_PROTOTYPE_2026-10-15.md`,
sections 2 to 5, every rule and parameter declared there before this was written).

Observations. On each six-hourly step and each integer latitude row inside a box, three
kinds of feature are read from the raw 700 hPa fields under the field renderer's validity,
and kept apart: every reversal of the meridional wind from northerly to southerly going
east (interpolated between valid neighbors), every local minimum of the meridional wind,
and every local maximum of the relative vorticity. Features of one kind on adjacent rows
whose longitudes differ by at most the spatial tolerance are joined into a row-chain, a
set of features with a longitude on every row it spans, with the rows inside the span
where the feature is absent or invalid recorded, and rows holding several features of
the chain recorded as multiple.

Proposals. Between consecutive steps, chains of one kind are compared on their shared
rows. A pair qualifies when the per-row displacement is within the temporal bound on at
least the declared fraction of the shared rows, with at least the minimum number of
shared rows. A link is unique when it is the only qualifying link forward for one chain
and backward for the other, ambiguous otherwise. A pair with too few shared rows whose
missing rows are invalid cells is unassessable. Histories are runs of unique links only.
Branchings, joinings, starts, ends and candidate gaps are listed and never bridged.

    python3 scripts/pilot_row_feature_graph.py synthetic --out <json>
    python3 scripts/pilot_row_feature_graph.py field --year 1990 --start 1990-08-08T12 --end 1990-08-15T12 \
        --box 10 30 38 65 --out <json> --png-prefix <prefix>
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_field_sequence as F  # noqa: E402

DATA_DIR = "data/era5/region6h"
PARAMS = {"spatial_tol_deg": 2.5, "temporal_bound_deg_per_step": 3.0, "shared_fraction": 0.5, "min_shared_rows": 2,
          "min_chain_rows": 2, "gap_max_steps": 2, "sensitivity_spatial_tol_deg": 1.5}
KINDS = ("reversal", "wind_min", "vort_max")


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ----------------------------------------------------------------------------- observations

def local_extrema(values, valid, kind):
    """Interior local minima ('min') or maxima ('max') among valid cells with both
    neighbors valid, the first cell of a flat run counted once."""
    out = []
    for j in range(1, len(values) - 1):
        if not (valid[j] and valid[j - 1] and valid[j + 1]):
            continue
        a, b, c = values[j - 1], values[j], values[j + 1]
        if kind == "min" and b < a and b <= c and (b < c or not out or out[-1] != j - 1):
            out.append(j)
        if kind == "max" and b > a and b >= c and (b > c or not out or out[-1] != j - 1):
            out.append(j)
    return out


def row_features(lons, v, vvalid, zeta, zvalid):
    """The three kinds of feature on one row. Reversals are interpolated between adjacent
    valid cells where v goes from at most zero to above zero going east."""
    lons = np.asarray(lons, float); v = np.asarray(v, float); zeta = np.asarray(zeta, float)
    vvalid = np.asarray(vvalid, bool); zvalid = np.asarray(zvalid, bool)
    feats = {k: [] for k in KINDS}
    for j in range(len(lons) - 1):
        if vvalid[j] and vvalid[j + 1] and v[j] <= 0.0 < v[j + 1]:
            frac = (0.0 - v[j]) / (v[j + 1] - v[j])
            feats["reversal"].append({"lon": float(lons[j] + frac * (lons[j + 1] - lons[j])), "value": float(v[j + 1] - v[j])})
    for j in local_extrema(v, vvalid, "min"):
        feats["wind_min"].append({"lon": float(lons[j]), "value": float(v[j])})
    for j in local_extrema(zeta, zvalid, "max"):
        feats["vort_max"].append({"lon": float(lons[j]), "value": float(zeta[j] * 1e5)})
    return feats


def near_invalid(invalid_lons, lons_near, tol):
    """Whether any invalid cell on a row lies within tol of any of the given longitudes."""
    return any(abs(x - y) <= tol for x in invalid_lons for y in lons_near)


def chains_at_step(features_by_row, rows, invalid_lons_by_row, tol, min_rows):
    """Row-chains per kind, connected components of features on adjacent rows within the
    spatial tolerance. Each chain records its rows, the longitude(s) per row, rows inside
    its span with no feature (invalid when an invalid cell lies within the tolerance of
    the neighboring rows' longitudes, absent otherwise), and the rows just beyond its
    span where an invalid cell lies within the tolerance of its end rows' longitudes."""
    out = {}
    rows = sorted(int(r) for r in rows)
    for kind in KINDS:
        nodes = [(r, i, f) for r in rows for i, f in enumerate(features_by_row[r][kind])]
        index = {(r, i): n for n, (r, i, _) in enumerate(nodes)}
        parent = list(range(len(nodes)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x

        for r, i, f in nodes:
            if r + 1 not in features_by_row:
                continue
            for i2, f2 in enumerate(features_by_row[r + 1][kind]):
                if abs(f2["lon"] - f["lon"]) <= tol:
                    a, b = find(index[(r, i)]), find(index[(r + 1, i2)])
                    if a != b:
                        parent[a] = b
        groups = {}
        for n, (r, i, f) in enumerate(nodes):
            groups.setdefault(find(n), []).append((r, f))
        chains = []
        for members in groups.values():
            by_row = {}
            for r, f in members:
                by_row.setdefault(r, []).append(f)
            span = sorted(by_row)
            if len(span) < min_rows:
                continue
            missing = []
            for r in range(span[0], span[-1] + 1):
                if r not in by_row:
                    below = [x["lon"] for rr in range(r - 1, span[0] - 1, -1) if rr in by_row for x in by_row[rr]][:2]
                    above = [x["lon"] for rr in range(r + 1, span[-1] + 1) if rr in by_row for x in by_row[rr]][:2]
                    inv = near_invalid(invalid_lons_by_row.get(r, []), below + above, tol)
                    missing.append({"row": r, "reason": "invalid" if inv else "absent"})
            adjacent_invalid = []
            for r, edge in ((span[0] - 1, span[0]), (span[-1] + 1, span[-1])):
                if r in invalid_lons_by_row and near_invalid(invalid_lons_by_row[r], [x["lon"] for x in by_row[edge]], tol):
                    adjacent_invalid.append(r)
            chains.append({"kind": kind, "rows": span, "lon_by_row": {str(r): sorted(x["lon"] for x in by_row[r]) for r in span},
                           "value_by_row": {str(r): [x["value"] for x in by_row[r]] for r in span},
                           "rows_with_multiple": [r for r in span if len(by_row[r]) > 1], "missing_rows": missing, "adjacent_invalid_rows": adjacent_invalid,
                           "n_rows": len(span), "lon_span": [min(min(x["lon"] for x in by_row[r]) for r in span), max(max(x["lon"] for x in by_row[r]) for r in span)]})
        chains.sort(key=lambda c: (c["rows"][0], c["lon_span"][0]))
        for k, c in enumerate(chains):
            c["id"] = f"{kind}:{k}"
        out[kind] = chains
    return out


# ----------------------------------------------------------------------------- proposals

def pair_displacements(a, b):
    """Per shared row, the displacement from chain a to chain b, using the nearest pair of
    longitudes where a row holds several."""
    shared = sorted(set(a["rows"]) & set(b["rows"]))
    d = {}
    for r in shared:
        la, lb = a["lon_by_row"][str(r)], b["lon_by_row"][str(r)]
        d[r] = min((y - x for x in la for y in lb), key=abs)
    return d


def separated_by_invalid(c1, c2, invalid_lons_by_row, tol):
    """Whether two chains of one step lie on disjoint row spans whose every row in between
    holds an invalid cell within the tolerance of the chains' facing end longitudes, so
    that one structure cut by invalid cells cannot be told from two."""
    lo, hi = (c1, c2) if c1["rows"][-1] < c2["rows"][0] else (c2, c1)
    if lo["rows"][-1] >= hi["rows"][0]:
        return False
    between = range(lo["rows"][-1] + 1, hi["rows"][0])
    if not between:
        return False
    facing = lo["lon_by_row"][str(lo["rows"][-1])] + hi["lon_by_row"][str(hi["rows"][0])]
    return all(near_invalid(invalid_lons_by_row.get(r, []), facing, tol) for r in between)


def links_between(chains_t, chains_t1, invalid_t, invalid_t1, params):
    """Qualifying links per kind between two consecutive steps, with classes. A branching
    or joining whose several chains are separated only by rows with invalid cells is
    labeled unassessable, and so are its links, since the chains may be one structure cut
    by invalid cells. `invalid_t` and `invalid_t1` map rows to invalid longitudes."""
    bound, frac, min_shared, tol = params["temporal_bound_deg_per_step"], params["shared_fraction"], params["min_shared_rows"], params["spatial_tol_deg"]
    out = {}
    for kind in KINDS:
        cand = []
        unassessable = []
        for a in chains_t.get(kind, []):
            for b in chains_t1.get(kind, []):
                d = pair_displacements(a, b)
                n = len(d)
                if n < min_shared:
                    span_a, span_b = range(a["rows"][0], a["rows"][-1] + 1), range(b["rows"][0], b["rows"][-1] + 1)
                    near = set(span_a) & set(range(span_b.start - 1, span_b.stop + 1))
                    if any(near_invalid(invalid_t.get(r, []) + invalid_t1.get(r, []), a["lon_by_row"][str(a["rows"][0])] + a["lon_by_row"][str(a["rows"][-1])], tol) for r in near):
                        unassessable.append({"from": a["id"], "to": b["id"], "shared_rows": n, "reason": "shared rows below the minimum with invalid cells between the chains"})
                    continue
                within = [r for r, x in d.items() if abs(x) <= bound]
                if len(within) >= max(min_shared, math.ceil(frac * n)):
                    cand.append({"from": a["id"], "to": b["id"], "shared_rows": n, "rows_within_bound": len(within),
                                 "displacement_by_row": {str(r): round(x, 3) for r, x in sorted(d.items())},
                                 "median_displacement": float(np.median([x for x in d.values()]))})
        fwd, bwd = {}, {}
        for c in cand:
            fwd.setdefault(c["from"], []).append(c["to"]); bwd.setdefault(c["to"], []).append(c["from"])
        by_id_t = {c["id"]: c for c in chains_t.get(kind, [])}; by_id_t1 = {c["id"]: c for c in chains_t1.get(kind, [])}
        branchings, joinings = [], []
        for a_id, targets in fwd.items():
            if len(targets) > 1:
                cut = all(separated_by_invalid(by_id_t1[x], by_id_t1[y], invalid_t1, tol) for i, x in enumerate(targets) for y in targets[i + 1:])
                branchings.append({"chain": a_id, "to": targets, "unassessable": cut})
        for b_id, sources in bwd.items():
            if len(sources) > 1:
                cut = all(separated_by_invalid(by_id_t[x], by_id_t[y], invalid_t, tol) for i, x in enumerate(sources) for y in sources[i + 1:])
                joinings.append({"chain": b_id, "from": sources, "unassessable": cut})
        cut_fwd = {b["chain"] for b in branchings if b["unassessable"]}; cut_bwd = {j["chain"] for j in joinings if j["unassessable"]}
        for c in cand:
            if c["from"] in cut_fwd or c["to"] in cut_bwd:
                c["class"] = "unassessable"
            else:
                c["class"] = "unique" if len(fwd[c["from"]]) == 1 and len(bwd[c["to"]]) == 1 else "ambiguous"
            c["alternatives"] = {"forward_from_source": [x for x in fwd[c["from"]] if x != c["to"]], "backward_to_target": [x for x in bwd[c["to"]] if x != c["from"]]}
        out[kind] = {"links": cand, "unassessable": unassessable, "branchings": branchings, "joinings": joinings}
    return out


def histories_and_gaps(steps, link_sets, params):
    """Histories as runs of unique links, with starts and ends, and candidate gaps between
    an end and a start within the gap limit whose per-row longitudes lie within the bound
    per elapsed step on at least the minimum shared rows. Gaps are listed, not bridged."""
    out = {}
    for kind in KINDS:
        unique_fwd = {}
        has_qual_fwd, has_qual_bwd = set(), set()
        for k, ls in enumerate(link_sets):
            for c in ls[kind]["links"]:
                has_qual_fwd.add((k, c["from"])); has_qual_bwd.add((k + 1, c["to"]))
                if c["class"] == "unique":
                    unique_fwd[(k, c["from"])] = (k + 1, c["to"])
        unique_bwd = {v: k for k, v in unique_fwd.items()}
        unass_fwd = {(k, c["from"]) for k, ls in enumerate(link_sets) for c in ls[kind]["links"] if c["class"] == "unassessable"}
        unass_bwd = {(k + 1, c["to"]) for k, ls in enumerate(link_sets) for c in ls[kind]["links"] if c["class"] == "unassessable"}
        chain_ids = [(k, c["id"]) for k, s in enumerate(steps) for c in s["chains"].get(kind, [])]
        hists = []
        seen = set()
        for node in chain_ids:
            if node in seen or node in unique_bwd:
                continue
            run = [node]; seen.add(node)
            while run[-1] in unique_fwd:
                nxt = unique_fwd[run[-1]]
                if nxt in seen:
                    break
                run.append(nxt); seen.add(nxt)
            first, last = run[0], run[-1]
            hists.append({"id": f"{kind}:H{len(hists)}", "members": [{"step": k, "chain": cid} for k, cid in run], "n_steps": len(run),
                          "start_step": first[0], "end_step": last[0],
                          "start_reason": "no qualifying link backward" if first not in has_qual_bwd else ("backward link unassessable" if first in unass_bwd else "backward link not unique"),
                          "end_reason": "no qualifying link forward" if last not in has_qual_fwd else ("forward link unassessable" if last in unass_fwd else "forward link not unique")})
        by_node = {(k, c["id"]): c for k, s in enumerate(steps) for c in s["chains"].get(kind, [])}
        gaps = []
        for h in hists:
            if h["end_reason"] != "no qualifying link forward":
                continue
            ek = h["end_step"]; ec = by_node[(ek, h["members"][-1]["chain"])]
            for g in hists:
                if g is h or g["start_reason"] != "no qualifying link backward":
                    continue
                sk = g["start_step"]
                if not (ek < sk <= ek + params["gap_max_steps"] + 1) or sk - ek < 2:
                    continue
                sc = by_node[(sk, g["members"][0]["chain"])]
                d = pair_displacements(ec, sc)
                elapsed = sk - ek
                within = [r for r, x in d.items() if abs(x) <= params["temporal_bound_deg_per_step"] * elapsed]
                if len(within) >= params["min_shared_rows"]:
                    gaps.append({"end_history": h["id"], "start_history": g["id"], "elapsed_steps": elapsed, "shared_rows": len(d), "rows_within_bound": len(within), "bridged": False})
        out[kind] = {"histories": hists, "candidate_gaps": gaps}
    return out


def measure(times, lat_rows, lons, v, u, sp, level_hpa=700.0, params=PARAMS):
    """The whole measurement on arrays (time, row, lon). Returns the record body."""
    steps = []
    invalid_rows = []
    for k in range(len(times)):
        wcls = F.terrain_classes(sp[k], np.where(np.isfinite(u[k]) & np.isfinite(v[k]), v[k], np.nan), level_hpa)
        zeta, zcls = F.relative_vorticity(u[k], v[k], lat_rows, lons, wcls)
        feats, inv_lons = {}, {}
        for r_i, r in enumerate(lat_rows):
            vv = wcls[r_i] == "valid"; zv = zcls[r_i] == "valid"
            feats[int(r)] = row_features(lons, v[k][r_i], vv, zeta[r_i], zv)
            inv_lons[int(r)] = [float(x) for x in np.asarray(lons)[~vv]]
        invalid_rows.append(inv_lons)
        steps.append({"step": k, "time": times[k], "chains": chains_at_step(feats, [int(r) for r in lat_rows], inv_lons, params["spatial_tol_deg"], params["min_chain_rows"]),
                      "invalid_rows": [int(r) for r in lat_rows if len(inv_lons[int(r)]) == len(lons)], "rows_with_invalid_cells": [int(r) for r in lat_rows if inv_lons[int(r)]],
                      "n_features": {kd: sum(len(feats[int(r)][kd]) for r in lat_rows) for kd in KINDS}})
    link_sets = [links_between(steps[k]["chains"], steps[k + 1]["chains"], invalid_rows[k], invalid_rows[k + 1], params) for k in range(len(steps) - 1)]
    hg = histories_and_gaps(steps, link_sets, params)
    return {"params": params, "steps": steps, "links": [{"from_step": k, "to_step": k + 1, **ls} for k, ls in enumerate(link_sets)], "histories": hg}


# ----------------------------------------------------------------------------- synthetic cases

SYN_LAT = np.arange(10.0, 31.0, 1.0)
SYN_LON = np.arange(38.0, 66.0, 1.0)
SYN_STEPS = 24


def band(lat_lo, lat_hi, center_fn, amp=-10.0, width=2.0, southerly=None):
    """A Gaussian trough in v over rows lat_lo..lat_hi, with the center longitude given
    per (step, row) by center_fn, and an optional southerly ridge `southerly` degrees east
    with amplitude amp_s."""
    def field(k):
        out = np.zeros((SYN_LAT.size, SYN_LON.size))
        for r_i, lat in enumerate(SYN_LAT):
            if lat_lo <= lat <= lat_hi:
                c = center_fn(k, lat)
                if c is None:
                    continue
                out[r_i] += amp * np.exp(-((SYN_LON - c) ** 2) / (2 * width ** 2))
                if southerly is not None:
                    out[r_i] += -0.8 * amp * np.exp(-((SYN_LON - (c + southerly)) ** 2) / (2 * width ** 2))
        return out
    return field


def synthetic_case(name):
    """Fields (v, u, sp) over SYN_STEPS steps and the answer key, kept apart."""
    sp = np.full((SYN_STEPS, SYN_LAT.size, SYN_LON.size), 95000.0)
    u = np.zeros_like(sp)
    parts, key = [], {"name": name}
    if name == "S1":
        parts = [band(12, 18, lambda k, lat: 58.0 - 0.5 * k, southerly=6.0), band(22, 28, lambda k, lat: 63.0 - 1.0 * k, southerly=None)]
        key.update({"bands": [{"rows": (12, 18), "center": lambda k, lat: 58.0 - 0.5 * k}, {"rows": (22, 28), "center": lambda k, lat: 63.0 - 1.0 * k}],
                    "required": "two histories, never one chain, never linked"})
    elif name == "S2":
        parts = [band(10, 28, lambda k, lat: 43.5, amp=-8.0, width=3.0), band(12, 26, lambda k, lat: 60.0 - 1.0 * k, southerly=5.0)]
        key.update({"standing_center": 43.5, "moving": lambda k, lat: 60.0 - 1.0 * k, "required": "standing chain zero displacement; moving history while separated; joining then branching across the contact, no unique link through"})
    elif name == "S3":
        parts = [band(12, 28, lambda k, lat: (52.0 - 1.0 * k) if lat >= 22 else 52.0, southerly=6.0)]
        key.update({"north_rows": (22, 28), "south_rows": (12, 20), "required": "one chain with per-row displacements of -1 north and 0 south while within tolerance, then a branching, history ended at the branching"})
    elif name == "S4":
        parts = [band(14, 26, lambda k, lat: 42.0 + 1.0 * min(k, 7), southerly=None), band(14, 26, lambda k, lat: 60.0 - 1.0 * min(k, 7), southerly=None)]
        key.update({"required": "two histories, then a joining, neither continued through it"})
    elif name in ("U2a", "U2b"):
        def c(k, lat):
            if 8 <= k <= 9:
                return None
            return 55.0 if k < 8 else 53.0
        parts = [band(14, 26, c, southerly=6.0)]
        key.update({"required": "a candidate gap of two steps, never bridged; identical to the pair"})
    elif name in ("U3a", "U3b"):
        parts = [band(14, 26, lambda k, lat: 58.0 - 1.0 * k, southerly=None), band(14, 26, lambda k, lat: 40.0 + 1.0 * k, southerly=None)]
        key.update({"required": "a joining then a branching at the crossing, no unique link through; identical to the pair"})
    elif name == "T1":
        parts = [band(12, 26, lambda k, lat: 58.0 - 1.0 * k, southerly=6.0)]
        rows = (SYN_LAT >= 14) & (SYN_LAT <= 20); cols = (SYN_LON >= 48) & (SYN_LON <= 52)
        sp[:, np.ix_(rows, cols)[0], np.ix_(rows, cols)[1]] = 69000.0
        key.update({"patch": {"rows": (14, 20), "lons": (48, 52)}, "required": "no feature on invalid rows inside the patch, missing rows recorded, links depending on them unassessable, no break and no continuity claimed"})
    else:
        raise SystemExit(f"unknown case {name}")
    v = np.stack([sum(p(k) for p in parts) for k in range(SYN_STEPS)])
    return {"v": v, "u": u, "sp": sp, "lat": SYN_LAT, "lon": SYN_LON, "times": [f"T{k:02d}" for k in range(SYN_STEPS)]}, key


def add_noise(v, seed=0, sd=1.0, corr_deg=2.0):
    from scipy import ndimage
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(v.shape)
    sm = np.stack([ndimage.gaussian_filter(x, corr_deg / 2.0, mode="nearest") for x in w])
    sm *= sd / sm.std()
    return v + sm


# ----------------------------------------------------------------------------- scoring

def chain_center_match(chain, center_fn, k, tol=1.5):
    """Whether every row of a chain lies within tol of the given center function."""
    for r in chain["rows"]:
        c = center_fn(k, float(r))
        if c is None or min(abs(x - c) for x in chain["lon_by_row"][str(r)]) > tol:
            return False
    return True


def score(name, rec, key, kind="wind_min"):
    """The declared required result for each case, on the named kind (the wind minimum
    is the trough core every case holds; the reversal kind is scored where declared)."""
    H = rec["histories"][kind]["histories"]; L = rec["links"]; S = rec["steps"]
    findings = {"case": name, "kind": kind}
    if name == "S1":
        n_chains = [len(s["chains"][kind]) for s in S]
        findings["chains_per_step"] = sorted(set(n_chains))
        bands = key["bands"]
        pure = all(any(chain_center_match(c, b["center"], s["step"]) for b in bands) for s in S for c in s["chains"][kind])
        cross = any(abs(l["median_displacement"]) > 2.0 for ls in L for l in ls[kind]["links"])
        long_h = [h for h in H if h["n_steps"] >= SYN_STEPS - 1]
        findings.update({"every_chain_on_one_band": pure, "any_cross_link": cross, "full_length_histories": len(long_h)})
        findings["pass"] = pure and not cross and len(long_h) == 2 and set(n_chains) == {2}
    elif name == "S2":
        stand = [c for s in S for c in s["chains"][kind] if chain_center_match(c, lambda k, lat: key["standing_center"], s["step"], 1.0)]
        contact_steps = [k for k in range(SYN_STEPS) if abs(key["moving"](k, 15.0) - key["standing_center"]) <= rec["params"]["spatial_tol_deg"] + 2.0]
        unique_through = [l for ls in L for l in ls[kind]["links"] if l["class"] == "unique" and ls["from_step"] in contact_steps and ls["to_step"] in contact_steps
                          and any(j["chain"] == l["to"] for j in ls[kind]["joinings"]) ]
        joins = [ls["to_step"] for ls in L if ls[kind]["joinings"]]; branches = [ls["from_step"] for ls in L if ls[kind]["branchings"]]
        stand_disp = [abs(l["median_displacement"]) for ls in L for l in ls[kind]["links"] if any(c["id"] == l["from"] and chain_center_match(c, lambda k, lat: key["standing_center"], ls["from_step"], 1.0) for c in S[ls["from_step"]]["chains"][kind])
                      and not (ls["from_step"] in contact_steps)]
        moving_before = [h for h in H if h["end_step"] < min(contact_steps) and h["n_steps"] >= 3]
        findings.update({"standing_chains_seen": len(stand) > 0, "standing_max_displacement_outside_contact": max(stand_disp) if stand_disp else None,
                         "joining_steps": joins, "branching_steps": branches, "moving_history_before_contact": len(moving_before) >= 1, "unique_links_into_a_joining": len(unique_through)})
        findings["pass"] = bool(stand) and (not stand_disp or max(stand_disp) <= 0.5) and bool(joins) and bool(branches) and len(moving_before) >= 1 and len(unique_through) == 0
    elif name == "S3":
        first_links = [l for l in L[0][kind]["links"]]
        d = first_links[0]["displacement_by_row"] if first_links else {}
        north = [abs(d[str(r)] + 1.0) <= 0.3 for r in range(22, 29) if str(r) in d]; south = [abs(d[str(r)]) <= 0.3 for r in range(12, 21) if str(r) in d]
        branches = [ls["from_step"] for ls in L if ls[kind]["branchings"]]
        n_chains = [len(s["chains"][kind]) for s in S]
        first_split = next((k for k, n in enumerate(n_chains) if n >= 2), None)
        h0 = [h for h in H if h["start_step"] == 0]
        findings.update({"one_chain_at_start": n_chains[0] == 1, "north_rows_move_minus_one": all(north) and len(north) >= 5, "south_rows_zero": all(south) and len(south) >= 5,
                         "first_step_with_two_chains": first_split, "branching_steps": branches, "history_from_start_ends_at_or_before_split": bool(h0) and (first_split is None or h0[0]["end_step"] <= first_split)})
        findings["pass"] = all([findings["one_chain_at_start"], findings["north_rows_move_minus_one"], findings["south_rows_zero"], first_split is not None, findings["history_from_start_ends_at_or_before_split"], len(h0) == 1 and h0[0]["end_step"] < SYN_STEPS - 1])
    elif name == "S4":
        joins = [ls["to_step"] for ls in L if ls[kind]["joinings"]]
        early = [h for h in H if h["start_step"] == 0]
        n_chains = [len(s["chains"][kind]) for s in S]
        findings.update({"two_chains_at_start": n_chains[0] == 2, "joining_steps": joins, "histories_from_start": len(early), "ended_before_end": all(h["end_step"] < SYN_STEPS - 1 for h in early)})
        findings["pass"] = n_chains[0] == 2 and bool(joins) and len(early) == 2 and findings["ended_before_end"]
    elif name.startswith("U2"):
        gaps = rec["histories"][kind]["candidate_gaps"]
        findings.update({"candidate_gaps": gaps, "histories": len(H)})
        findings["pass"] = len(gaps) >= 1 and all(not g["bridged"] for g in gaps) and len(H) == 2
    elif name.startswith("U3"):
        joins = [ls["to_step"] for ls in L if ls[kind]["joinings"]]; branches = [ls["from_step"] for ls in L if ls[kind]["branchings"]]
        through = [l for ls in L for l in ls[kind]["links"] if l["class"] == "unique" and (any(j["chain"] == l["to"] for j in ls[kind]["joinings"]) or any(b["chain"] == l["from"] for b in ls[kind]["branchings"]))]
        findings.update({"joining_steps": joins, "branching_steps": branches, "unique_links_through": len(through), "histories": len(H)})
        findings["pass"] = bool(joins) and bool(branches) and len(through) == 0 and len(H) >= 4
    elif name == "T1":
        rows = range(key["patch"]["rows"][0], key["patch"]["rows"][1] + 1)
        on_patch = [k for k in range(SYN_STEPS) if key["patch"]["lons"][0] - 1 <= 58.0 - k <= key["patch"]["lons"][1] + 1]
        feats_on_invalid = sum(1 for k in on_patch for s in [S[k]] for c in s["chains"][kind] for r in c["rows"] if r in rows and c["kind"] == kind and any(48 <= x <= 52 for x in c["lon_by_row"][str(r)]))
        missing_recorded = any(m["reason"] == "invalid" for k in on_patch for c in S[k]["chains"][kind] for m in c["missing_rows"])
        adjacent_recorded = any(c["adjacent_invalid_rows"] for k in on_patch for c in S[k]["chains"][kind])
        unass_links = sum(1 for ls in L for l in ls[kind]["links"] if l["class"] == "unassessable" and (ls["from_step"] in on_patch or ls["to_step"] in on_patch))
        claimed_breaks = sum(1 for ls in L if (ls["from_step"] in on_patch or ls["to_step"] in on_patch) for b in ls[kind]["branchings"] + ls[kind]["joinings"] if not b["unassessable"])
        findings.update({"features_on_invalid_patch_cells": feats_on_invalid, "missing_or_adjacent_invalid_rows_recorded": missing_recorded or adjacent_recorded,
                         "unassessable_links_on_patch_steps": unass_links, "branchings_or_joinings_claimed_on_patch_steps": claimed_breaks, "steps_on_patch": on_patch, "histories": len(H)})
        findings["pass"] = feats_on_invalid == 0 and (missing_recorded or adjacent_recorded) and unass_links > 0 and claimed_breaks == 0
    return findings


def run_synthetic(out_path, noisy=True):
    names = ["S1", "S2", "S3", "S4", "U2a", "U2b", "U3a", "U3b", "T1"]
    results, records = {}, {}
    for name in names:
        f, key = synthetic_case(name)
        rec = measure(f["times"], f["lat"], f["lon"], f["v"], f["u"], f["sp"])
        records[name] = rec
        kind = "reversal" if name in ("S1",) else "wind_min"
        results[name] = score(name, rec, key, kind="wind_min")
        if name == "S1":
            results[name + "_reversal"] = score(name, rec, key, kind="reversal")
    for a, b in (("U2a", "U2b"), ("U3a", "U3b")):
        same = json.dumps(records[a], sort_keys=True) == json.dumps(records[b], sort_keys=True)
        results[f"{a}_{b}_identical_outputs"] = same
        results[a]["pass"] = results[a]["pass"] and same; results[b]["pass"] = results[b]["pass"] and same
    noisy_results = {}
    if noisy:
        for name in names:
            f, key = synthetic_case(name)
            rec = measure(f["times"], f["lat"], f["lon"], add_noise(f["v"], seed=1), f["u"], f["sp"])
            noisy_results[name] = score(name, rec, key, kind="wind_min")
    passed = all(results[n]["pass"] for n in names)
    out = {"generated_by": "scripts/pilot_row_feature_graph.py synthetic", "params": PARAMS, "cases": names, "results": results,
           "noisy_realization_reported_not_a_condition": {"sd_ms": 1.0, "corr_deg": 2.0, "seed": 1, "results": noisy_results},
           "pass_noise_free": passed}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    for n in names:
        print(f"{n}: {'PASS' if results[n]['pass'] else 'FAIL'}  {json.dumps({k: v for k, v in results[n].items() if k not in ('case', 'kind', 'pass', 'candidate_gaps')})[:220]}")
    for k in ("U2a_U2b_identical_outputs", "U3a_U3b_identical_outputs"):
        print(k, results[k])
    print("noise-free pass:", passed, "| noisy realization passes:", {n: r["pass"] for n, r in noisy_results.items()})
    return 0 if passed else 1


# ----------------------------------------------------------------------------- real fields

def run_field(a):
    level = 700.0
    files = {n: os.path.join(DATA_DIR, f"era5_{n}_{a.year}_6h_region.nc") for n in ("u700", "v700", "sp")}
    grid = F.grid_from_level_file(files["u700"], a.year)
    lat = grid["latgrid"][:, 0]; lon = grid["longrid"][0, :]
    la0, la1, lo0, lo1 = a.box
    rows = np.flatnonzero((lat >= la0 - 1e-6) & (lat <= la1 + 1e-6)); cols = np.flatnonzero((lon >= lo0 - 1e-6) & (lon <= lo1 + 1e-6))
    steps = F.sequence_steps(grid["time"], F.to_day(a.start), F.to_day(a.end), 6)
    step_days = [float(grid["time"][k]) for k in steps]
    lat_vals, lon_vals = lat[rows], lon[cols]
    u = F.field_on(files["u700"], "u", "m s**-1", lat_vals, lon_vals, step_days, level)
    v = F.field_on(files["v700"], "v", "m s**-1", lat_vals, lon_vals, step_days, level)
    sp = F.surface_pressure_on(files["sp"], lat_vals, lon_vals, step_days)
    times = [(F.EPOCH + dt.timedelta(days=d)).strftime("%Y-%m-%dT%H") for d in step_days]
    rec = measure(times, lat_vals, lon_vals, v, u, sp, level)
    sens = measure(times, lat_vals, lon_vals, v, u, sp, level, params={**PARAMS, "spatial_tol_deg": PARAMS["sensitivity_spatial_tol_deg"]})
    here = os.path.dirname(os.path.abspath(__file__))
    out = {"generated_by": "scripts/pilot_row_feature_graph.py field", "script_sha256": sha256_of(os.path.abspath(__file__)),
           "reuses": {"grid_from_level_file, field_on, surface_pressure_on, terrain_classes, relative_vorticity, sequence_steps": "scripts/pilot_field_sequence.py, unchanged", "renderer_sha256": sha256_of(os.path.join(here, "pilot_field_sequence.py"))},
           "year": a.year, "level_hpa": level, "sources": {n: {"path": p, "sha256": sha256_of(p)} for n, p in files.items()},
           "declared": {"window": [a.start, a.end], "box": a.box, "rows": [int(x) for x in lat_vals], "lons": lon_vals.tolist(), "params": PARAMS,
                        "label": "observations (features, row-chains) apart from proposals (links with classes); histories are runs of unique links; gaps, branchings and joinings are listed and never bridged; nothing here is physical identity"},
           "measurement": rec,
           "sensitivity_spatial_tol_1_5": {"n_histories": {k: len(sens["histories"][k]["histories"]) for k in KINDS}, "n_gaps": {k: len(sens["histories"][k]["candidate_gaps"]) for k in KINDS},
                                           "chains_per_step": {k: [len(s["chains"][k]) for s in sens["steps"]] for k in KINDS}}}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    figs = draw(out, a.png_prefix) if a.png_prefix else []
    out["figures"] = figs
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    for k in KINDS:
        H = rec["histories"][k]["histories"]
        print(f"{k}: chains per step {[len(s['chains'][k]) for s in rec['steps']]}; histories {len(H)} (lengths {sorted((h['n_steps'] for h in H), reverse=True)[:8]}); "
              f"candidate gaps {len(rec['histories'][k]['candidate_gaps'])}; branchings {sum(len(l[k]['branchings']) for l in rec['links'])}; joinings {sum(len(l[k]['joinings']) for l in rec['links'])}; "
              f"unassessable {sum(len(l[k]['unassessable']) for l in rec['links'])}")
    print(f"{len(figs)} figures. Record {a.out}.")
    return 0


def draw(out, prefix):
    """Two readable figures per kind for the primary kinds (reversal, wind minimum), drawn
    from a saved record. Maps at twelve-hourly steps draw every feature of a chain as a
    marker and join only adjacent single-valued rows with a segment, so no averaged
    position is drawn where a row holds several features (the first figures, kept, drew
    the mean of such rows into a continuous line). Chains in a history carry its color and
    its identifier at the first drawn step, since colors repeat, and gray chains are in no
    history. Links are not drawn, branchings and joinings are marked, and a time-longitude
    panel per declared row shows the same features. Corrected display of 2026-10-16."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rec = out["measurement"]; steps = rec["steps"]; figs = []
    for kind in ("reversal", "wind_min"):
        hist_of = {}
        for h in rec["histories"][kind]["histories"]:
            for m in h["members"]:
                hist_of[(m["step"], m["chain"])] = h["id"]
        hids = sorted({v for v in hist_of.values()}, key=lambda s: int(s.split("H")[1]))
        cmap = plt.get_cmap("tab20"); color = {h: cmap(i % 20) for i, h in enumerate(hids)}
        sel = list(range(0, len(steps), 2))
        ncol = 4; nrow = int(math.ceil(len(sel) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False, constrained_layout=True)
        branch_at = {(l["from_step"], b["chain"]) for l in rec["links"] for b in l[kind]["branchings"]}
        join_at = {(l["to_step"], j["chain"]) for l in rec["links"] for j in l[kind]["joinings"]}
        labeled = set()
        for ax_i, k in enumerate(sel):
            ax = axes[ax_i // ncol][ax_i % ncol]
            s = steps[k]
            for c in s["chains"][kind]:
                rows = c["rows"]
                hid = hist_of.get((k, c["id"]))
                col = color.get(hid, (0.6, 0.6, 0.6, 1.0)); lw = 2.0 if hid else 1.0
                for r in rows:
                    for xx in c["lon_by_row"][str(r)]:
                        ax.plot(xx, r, "x" if r in c["rows_with_multiple"] else ".", color=col, ms=5 if r in c["rows_with_multiple"] else 3)
                for r0, r1 in zip(rows, rows[1:]):
                    if r1 == r0 + 1 and r0 not in c["rows_with_multiple"] and r1 not in c["rows_with_multiple"]:
                        ax.plot([c["lon_by_row"][str(r0)][0], c["lon_by_row"][str(r1)][0]], [r0, r1], "-", color=col, lw=lw)
                x_first = c["lon_by_row"][str(rows[0])][0]; x_last = c["lon_by_row"][str(rows[-1])][0]
                if hid and hid not in labeled:
                    ax.annotate(hid.split(":")[1], (x_last, rows[-1]), fontsize=6, color=col, xytext=(2, 2), textcoords="offset points"); labeled.add(hid)
                if (k, c["id"]) in branch_at:
                    ax.plot(x_last, rows[-1], "v", color="k", ms=7)
                if (k, c["id"]) in join_at:
                    ax.plot(x_first, rows[0], "^", color="k", ms=7)
            for r in s["invalid_rows"]:
                ax.axhspan(r - 0.5, r + 0.5, color="0.85")
            lo_lon, hi_lon = min(out["declared"]["lons"]), max(out["declared"]["lons"]); lo_row, hi_row = min(out["declared"]["rows"]), max(out["declared"]["rows"])
            ax.set_xlim(lo_lon, hi_lon); ax.set_ylim(lo_row - 0.7, hi_row + 0.7)
            ax.axvline(40, color="k", ls="--", lw=0.7); ax.axvline(60, color="k", ls="-.", lw=0.7)
            ax.set_title(s["time"], fontsize=10); ax.grid(True, lw=0.3, ls=":")
        for ax_i in range(len(sel), nrow * ncol):
            axes[ax_i // ncol][ax_i % ncol].axis("off")
        fig.suptitle(f"Row-feature graph, {kind}, {out['year']}, 700 hPa, corrected display from the saved record: every feature a marker, segments only between adjacent single-valued rows "
                     "(no averaged position is drawn), x = several features on a row, chains colored and labeled by history identifier where in a history of unique component-to-component proposals (colors repeat, the label disambiguates), gray in none, "
                     "black triangles = a branching (down) or a joining (up), gray rows invalid. Links are not drawn.", fontsize=8, wrap=True)
        path = f"{prefix}_{kind}_maps.png"; fig.savefig(path, dpi=110); plt.close(fig)
        figs.append({"kind": kind, "path": path, "sha256": sha256_of(path), "steps_drawn": [steps[k]["time"] for k in sel]})
        # time-longitude per declared row
        rows_draw = [r for r in (26, 22, 18, 14) if r in out["declared"]["rows"]]
        fig, axes = plt.subplots(1, len(rows_draw), figsize=(4.0 * len(rows_draw), 6.0), squeeze=False, constrained_layout=True)
        for ax, r in zip(axes[0], rows_draw):
            for k, s in enumerate(steps):
                for c in s["chains"][kind]:
                    if r in c["rows"]:
                        hid = hist_of.get((k, c["id"])); col = color.get(hid, (0.6, 0.6, 0.6, 1.0))
                        for xx in c["lon_by_row"][str(r)]:
                            ax.plot(xx, k, "o", color=col, ms=4 if hid else 3)
                if r in s["invalid_rows"]:
                    ax.axhspan(k - 0.5, k + 0.5, color="0.85")
            ax.set_title(f"{r} N", fontsize=10); ax.set_xlim(min(out["declared"]["lons"]), max(out["declared"]["lons"])); ax.invert_yaxis()
            ax.set_yticks(range(0, len(steps), 4)); ax.set_yticklabels([steps[k]["time"][5:13] for k in range(0, len(steps), 4)], fontsize=7)
            ax.axvline(40, color="k", ls="--", lw=0.7); ax.axvline(60, color="k", ls="-.", lw=0.7); ax.grid(True, lw=0.3, ls=":")
        fig.suptitle(f"Row-feature graph, {kind}, corrected display from the saved record: features on four rows by time (down) and longitude, colored by history (colors repeat across the many short histories), gray where in no history. No link is drawn.", fontsize=9, wrap=True)
        path = f"{prefix}_{kind}_rows.png"; fig.savefig(path, dpi=110); plt.close(fig)
        figs.append({"kind": kind, "path": path, "sha256": sha256_of(path), "rows_drawn": rows_draw})
    return figs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    ps = sub.add_parser("synthetic"); ps.add_argument("--out", required=True); ps.add_argument("--no-noise", action="store_true")
    pf = sub.add_parser("field"); pf.add_argument("--year", type=int, required=True); pf.add_argument("--start", required=True); pf.add_argument("--end", required=True)
    pf.add_argument("--box", nargs=4, type=float, required=True, metavar=("LAT0", "LAT1", "LON0", "LON1")); pf.add_argument("--out", required=True); pf.add_argument("--png-prefix")
    pr = sub.add_parser("redraw", help="draw the corrected figures from a saved field record, with no field run and the saved record left unmodified")
    pr.add_argument("--record", required=True); pr.add_argument("--png-prefix", required=True); pr.add_argument("--out", required=True, help="a sidecar JSON naming the figures and the record they were drawn from")
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"REFUSED: {a.out} exists and a record is never overwritten")
    if a.mode == "synthetic":
        return run_synthetic(a.out, noisy=not a.no_noise)
    if a.mode == "redraw":
        rec = json.load(open(a.record))
        figs = draw(rec, a.png_prefix)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"generated_by": "scripts/pilot_row_feature_graph.py redraw", "script_sha256": sha256_of(os.path.abspath(__file__)), "record": {"path": a.record, "sha256": sha256_of(a.record)}, "figures": figs,
                       "note": "display correction only, with the measurement, its parameters and its first figures unchanged"}, fh)
        print(f"{len(figs)} figures redrawn from {a.record}.")
        return 0
    return run_field(a)


if __name__ == "__main__":
    raise SystemExit(main())
