#!/usr/bin/env python3
"""The builder of the eastern descriptive layer, version 1
(`EASTERN_DESCRIPTIVE_LAYER_2026-10-17.md`, sections 4 and 6). It reads one saved listing and
its saved reader output, refuses a reader output whose recorded listing digest does not
match the listing, joins the two by row and step into one flat table, and writes the
tables and a manifest. It computes nothing from the fields and changes no saved file.

    python3 scripts/build_eastern_descriptive_layer.py --manifest docs/aewc_v2/eastern_descriptive_layer_v1/manifest.json
"""
import argparse
import csv
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from pilot_entry_evolution_phases import supported as reader_supported  # noqa: E402

# The frozen reader's digest, so that the support helper imported above is the frozen one
# (the standing rule of the handoff: a producer reuses the checker's function rather than
# re-deriving its facts; added at the review of 2026-10-17, F6).
FROZEN_READER_SHA256_PREFIX = "7e33818acdc468a5"
EXT = "docs/aewc_v2/evidence/validation/eastern_pilot_2026-10-05/reports/extent"
INTERVALS = [
    {"name": "1983_june", "role": "development (stage D's interval, then native-grid development evidence)", "annotation": f"{EXT}/eval60_stageD_entry_annotation_1983.json", "listing": f"{EXT}/eval60_native_entry_listing_1983.json", "phases": f"{EXT}/eval60_native_phases_1983.json"},
    {"name": "1987_september", "role": "the stage E fresh evaluation of the re-frozen configuration", "annotation": f"{EXT}/eval60_stageE_entry_annotation_1987.json", "listing": f"{EXT}/eval60_stageE_entry_listing_1987.json", "phases": f"{EXT}/eval60_stageE_phases_1987.json"},
    {"name": "1990_august", "role": "development (I1, the broad window)", "annotation": f"{EXT}/eval60_i1wide_entry_annotation.json", "listing": f"{EXT}/eval60_native_entry_listing_i1wide_1990.json", "phases": f"{EXT}/eval60_native_phases_i1wide_1990.json"},
    {"name": "2006_june", "role": "development (I2)", "annotation": f"{EXT}/eval60_i2_entry_annotation.json", "listing": f"{EXT}/eval60_native_entry_listing_i2_2006.json", "phases": f"{EXT}/eval60_native_phases_i2_2006.json"},
]
COLUMNS = ["interval", "time", "row_lat", "status", "window_lon_lo", "window_lon_hi", "selected_lon", "selected_v_ms", "supported_primary", "alternatives", "all_minima",
           "reversal_lon", "reversal_status", "edge_primary_lon", "edge_sensitivity_lon", "zeta_maxima", "pair_class_to_next", "pair_move_to_next", "motion_label",
           "chain_index", "phase_index", "phase_direction", "standing_index"]
CLASSES = ("continuous_unique", "continuous_ambiguous", "switching", "replacement", "jump", "unassessable")


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def g(x):
    return "" if x is None else f"{x:g}"


def _load_pair(listing_path, phases_path):
    rec = json.load(open(listing_path)); ph = json.load(open(phases_path))
    if ph["record"]["sha256"] != sha256_of(listing_path):
        raise SystemExit(f"REFUSED: {phases_path} records a listing digest that does not match {listing_path}")
    reader_path = os.path.join(HERE, "pilot_entry_evolution_phases.py")
    if not sha256_of(reader_path).startswith(FROZEN_READER_SHA256_PREFIX):
        raise SystemExit(f"REFUSED: {reader_path} is not the frozen reader ({FROZEN_READER_SHA256_PREFIX}); its support helper cannot be reused")
    return rec, ph


def _index(res, row):
    r = str(row)
    def spans(items):
        return [(k, it["i_start"], it["i_end"]) for k, it in enumerate(items)]
    return {"pairs": {p["k"]: p for p in res["pairs"].get(r, [])}, "chains": spans(res["chains"].get(r, [])), "phases": [(k, it["i_start"], it["i_end"], it["direction"]) for k, it in enumerate(res["phases"].get(r, []))],
            "standing": spans(res["standing"].get(r, [])), "motion": {i: m for i, m in res["motion"].get(r, [])}}


def _which(spans, i):
    for sp in spans:
        if sp[1] <= i <= sp[2]:
            return sp
    return None


def build_table(listing_path, phases_path, interval):
    rec, ph = _load_pair(listing_path, phases_path)
    res = ph["results"]["primary"]; P = res["params"]
    steps = rec["steps"]; n = len(steps)
    rows_out = []
    idx = {}
    for i, s in enumerate(steps):
        win = (s.get("annotation") or {}).get("i1", {}).get("lon", [None, None])
        for m in s["rows"]:
            row = int(m["row_lat"])
            if row not in idx:
                idx[row] = _index(res, row)
            ix = idx[row]
            assigned = m.get("status") == "assigned" and m.get("i1") is not None
            sel_lon = m["i1"]["lon"] if assigned else None; sel_v = m["i1"]["v"] if assigned else None
            supported = bool(reader_supported(m, win[0], P))
            pair = ix["pairs"].get(i) if i < n - 1 else None
            ch = _which(ix["chains"], i); phs = _which(ix["phases"], i); st = _which(ix["standing"], i)
            alts = sorted(((a["lon"], a["v"]) for a in m.get("i1_alternatives", [])), key=lambda t: t[1])
            rows_out.append({
                "interval": interval, "time": s["time"], "row_lat": row, "status": m.get("status", ""),
                "window_lon_lo": win[0] if win[0] is not None else "", "window_lon_hi": win[1] if win[1] is not None else "",
                "selected_lon": sel_lon if assigned else "", "selected_v_ms": sel_v if assigned else "", "supported_primary": supported,
                "alternatives": ";".join(f"{g(a)}:{g(b)}" for a, b in alts),
                "all_minima": ";".join(f"{g(x['lon'])}:{g(x['v'])}:{x['label']}" for x in m.get("wind_minima", [])),
                "reversal_lon": g((m.get("reversal") or {}).get("lon")), "reversal_status": (m.get("reversal") or {}).get("status", ""),
                "edge_primary_lon": g(((m.get("edge") or {}).get("primary") or {}).get("lon")), "edge_sensitivity_lon": g(((m.get("edge") or {}).get("sensitivity") or {}).get("lon")),
                "zeta_maxima": ";".join(f"{g(z['lon'])}:{g(z['zeta_1e5'])}" for z in m.get("zeta_maxima", [])),
                "pair_class_to_next": pair["class"] if pair else "", "pair_move_to_next": pair["move"] if pair and pair["move"] is not None else "",
                "motion_label": ix["motion"].get(i, ""), "chain_index": ch[0] if ch else "", "phase_index": phs[0] if phs else "", "phase_direction": phs[3] if phs else "",
                "standing_index": st[0] if st else ""})
    return rows_out


def manifest_entry(listing_path, phases_path, name, role, rows):
    rec, ph = _load_pair(listing_path, phases_path)
    res = ph["results"]["primary"]; P = res["params"]
    by_status = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    pairs = [p for ps in res["pairs"].values() for p in ps]
    d = res["displacement_set"]
    overlap = (min(p["i_end"] for p in d["phases"]) - max(p["i_start"] for p in d["phases"])) * P["step_hours"] if d["phases"] else None
    return {"name": name, "role": role, "window": {"start": rec["steps"][0]["time"], "end": rec["steps"][-1]["time"], "steps": len(rec["steps"]), "every_hours": rec["declared"].get("every_hours")},
            "grid": rec["declared"].get("grid"), "lon_spacing_deg": rec["declared"].get("lon_spacing_deg"), "rows": rec["declared"].get("rows"),
            "listing": {"path": listing_path, "sha256": sha256_of(listing_path), "script_sha256": rec.get("script_sha256")},
            "reader_output": {"path": phases_path, "sha256": sha256_of(phases_path), "script_sha256": ph.get("script_sha256"), "params": P},
            "window_bounds_by_step": [{"time": s["time"], "lon_lo": (s.get("annotation") or {}).get("i1", {}).get("lon", [None, None])[0], "lon_hi": (s.get("annotation") or {}).get("i1", {}).get("lon", [None, None])[1]} for s in rec["steps"]],
            "row_steps": len(rows), "by_status": by_status, "supported_primary": sum(1 for r in rows if r["supported_primary"]),
            "pairs_by_class": {c: sum(1 for p in pairs if p["class"] == c) for c in CLASSES},
            "chains": sum(len(v) for v in res["chains"].values()), "phases": sum(len(v) for v in res["phases"].values()), "standing_intervals": sum(len(v) for v in res["standing"].values()),
            "largest_adjacent_row_set_operational_statistic": {"rows": d["rows"], "n_rows": d["n_rows"], "common_overlap_hours": overlap, "note": "an operational statistic of the reader with no product consequence; positions, not features"}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--build-note", default="", help="a note carried in the manifest, for a corrected build")
    a = ap.parse_args(argv)
    out_dir = os.path.dirname(os.path.abspath(a.manifest))
    os.makedirs(out_dir, exist_ok=True)
    targets = [a.manifest] + [os.path.join(out_dir, f"{iv['name']}.csv") for iv in INTERVALS]
    for t in targets:
        if os.path.exists(t):
            raise SystemExit(f"REFUSED: {t} exists and a product file is never overwritten")
    entries = []
    for iv in INTERVALS:
        lp, pp = os.path.join(ROOT, iv["listing"]), os.path.join(ROOT, iv["phases"])
        rows = build_table(lp, pp, iv["name"])
        csv_path = os.path.join(out_dir, f"{iv['name']}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS); w.writeheader()
            for r in rows:
                w.writerow(r)
        e = manifest_entry(lp, pp, iv["name"], iv["role"], rows)
        e["annotation"] = {"path": iv["annotation"], "sha256": sha256_of(os.path.join(ROOT, iv["annotation"]))}
        e["listing"]["path"] = iv["listing"]; e["reader_output"]["path"] = iv["phases"]
        e["table"] = {"path": os.path.relpath(csv_path, ROOT), "sha256": sha256_of(csv_path), "rows": len(rows)}
        entries.append(e)
        print(f"{iv['name']}: {len(rows)} row-steps, {e['by_status']}, supported {e['supported_primary']}, pairs {e['pairs_by_class']}, set {e['largest_adjacent_row_set_operational_statistic']['n_rows']} rows")
    manifest = {"product": "eastern descriptive layer, version 1", "declared": "EASTERN_DESCRIPTIVE_LAYER_2026-10-17.md", "built": "2026-10-17", "build_note": a.build_note,
                "frozen_reader_sha256_prefix": FROZEN_READER_SHA256_PREFIX,
                "builder": {"path": "scripts/build_eastern_descriptive_layer.py", "sha256": sha256_of(os.path.abspath(__file__))},
                "columns": COLUMNS, "label": "positions and their labels, no identity, no history, no track and no physical classification; the adjacent-row set is an operational statistic",
                "intervals": entries}
    with open(a.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print("manifest", a.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
