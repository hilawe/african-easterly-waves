"""IBTrACS developer-association prototype, season 2004.

A working prototype of the R6 audit record: for each 2004 Atlantic tropical
cyclone in IBTrACS, find the AEW track whose position at genesis time is
nearest, in each of three wave records, and emit the auditable association
row the requirements draft specifies (storm identifier, name, genesis time
and place, matched wave, distance, time offset used, competing waves within
the radius). The DERIVED developer flag is whatever survives this record,
not the record itself.

PROTOTYPE CRITERIA, placeholders for the scientific director to react to,
not proposals: genesis is the storm's first best-track point; a wave matches
if it has a position within 6 hours of genesis time and within 500 km of the
genesis point; competing waves are all others inside that radius. IBTrACS
mixes agencies and revisions, and best tracks begin near recognized storm
stages rather than at the wave precursor, so real criteria need lag windows,
staging, and case review; this prototype measures only how far the naive
rule gets.

Wave records: v1 (deduplicated, ERA-Interim), the local QTrack extended run
(from scripts/aewc_v2_qtrack_probe.py; run it first if its output is
missing), and the published AEWDAT wts2004 file. Inputs pinned by SHA-256.

Run from the repo root: .venv/bin/python scripts/aewc_v2_ibtracs_prototype.py
  --mutate=timeshift10d   genesis times shifted ten days; the match-count
                          checks must fail, proving the association reads
                          time and not just geography.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data" / "aewc_v2_pilot"

INPUT_SHA256 = {
    "data/aewc_v2_pilot/ibtracs.NA.list.v04r01.csv":
        "1b67895f53fdfd78742bd90a585f871b5d735638aed17b6b8e91ec0befb3fff7",
    "data/aewc/ERA-Int_ew_700hPa_2004_AFR.nc":
        "5564ea1cdbb9e03f7db4c5fb42663f773b34f881fb373c7b6cc21e52a845db09",
    "data/aewc_v2_pilot/wts2004.json":
        "31beef8a888d06766593a742ac83c50b8882a11927c62091660d81b7c56f2702",
    "data/aewc_v2_pilot/qtrack_run/AEW_pp_east.nc":
        "07530ba423f0c017de712939a80fc7189153ebe3e19ba4c35dae2ee09be95d94",
}

RADIUS_KM = 500.0
TIME_TOL = np.timedelta64(6, "h")

# measured on first run and pinned; the audit table the note shows derives
# from these. EXPECTED_ASSOC pins each storm's matched wave per record (None
# for no match), so the per-storm claims are bound, not only the aggregates.
EXPECTED = {
    "n_storms": 15,
    "matched": {"v1": 2, "qtrack_ext": 8, "aewdat": 6},
}
EXPECTED_SIDS = ['2004214N30282', '2004217N13306', '2004223N11301', '2004227N09314', '2004227N12338', '2004238N11325', '2004241N29295', '2004241N32282', '2004247N10332', '2004252N32320', '2004258N16300', '2004260N11331', '2004264N13328', '2004283N24265', '2004284N30295']
EXPECTED_ASSOC = {'2004214N30282': {'v1': None, 'qtrack_ext': None, 'aewdat': None}, '2004217N13306': {'v1': None, 'qtrack_ext': 28, 'aewdat': 28}, '2004223N11301': {'v1': 282, 'qtrack_ext': 35, 'aewdat': 29}, '2004227N12338': {'v1': 296, 'qtrack_ext': 42, 'aewdat': 30}, '2004227N09314': {'v1': None, 'qtrack_ext': 40, 'aewdat': 33}, '2004238N11325': {'v1': None, 'qtrack_ext': 45, 'aewdat': 39}, '2004241N32282': {'v1': None, 'qtrack_ext': None, 'aewdat': None}, '2004241N29295': {'v1': None, 'qtrack_ext': None, 'aewdat': None}, '2004247N10332': {'v1': None, 'qtrack_ext': 52, 'aewdat': None}, '2004252N32320': {'v1': None, 'qtrack_ext': None, 'aewdat': None}, '2004258N16300': {'v1': None, 'qtrack_ext': 55, 'aewdat': None}, '2004260N11331': {'v1': None, 'qtrack_ext': 56, 'aewdat': 49}, '2004264N13328': {'v1': None, 'qtrack_ext': None, 'aewdat': None}, '2004283N24265': {'v1': None, 'qtrack_ext': None, 'aewdat': None}, '2004284N30295': {'v1': None, 'qtrack_ext': None, 'aewdat': None}}

MUTATION = None
for a in sys.argv[1:]:
    if a.startswith("--mutate="):
        MUTATION = a.split("=", 1)[1]
if MUTATION is not None and MUTATION != "timeshift10d":
    sys.exit(f"unknown mutation {MUTATION!r}")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_storms():
    df = pd.read_csv(PILOT / "ibtracs.NA.list.v04r01.csv",
                     skiprows=[1], low_memory=False,
                     usecols=["SID", "SEASON", "NAME", "ISO_TIME", "NATURE",
                              "LAT", "LON", "USA_STATUS"])
    df = df[df["SEASON"].astype(str) == "2004"]
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    storms = []
    for sid, g in df.groupby("SID"):
        g = g.sort_values("ISO_TIME")
        first = g.iloc[0]
        t0 = first["ISO_TIME"]
        if not (np.datetime64("2004-06-01") <= np.datetime64(t0)
                < np.datetime64("2004-11-01")):
            continue
        if MUTATION == "timeshift10d":
            t0 = t0 + pd.Timedelta(days=10)
        storms.append({"sid": sid, "name": first["NAME"],
                       "genesis_time": np.datetime64(t0),
                       "genesis_lat": float(first["LAT"]),
                       "genesis_lon": float(first["LON"]),
                       "first_status": str(first["NATURE"])})
    storms.sort(key=lambda s: s["genesis_time"])
    return storms


def _gc_km(lat1, lon1, lat2, lon2):
    r = np.pi / 180
    a = (np.sin((lat2 - lat1) * r / 2) ** 2
         + np.cos(lat1 * r) * np.cos(lat2 * r)
         * np.sin((lon2 - lon1) * r / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_v1():
    sys.path.insert(0, str(ROOT / "src"))
    from aew.data.aewc import load_aewc_troughs
    tr = load_aewc_troughs(str(ROOT / "data/aewc/ERA-Int_ew_700hPa_2004_AFR.nc"),
                           dedup=True)
    out = defaultdict(list)
    t = pd.DatetimeIndex(tr.time)
    for i in range(len(tr.time)):
        out[int(tr.variables["traj_id"][i])].append(
            (np.datetime64(t[i]), float(tr.lat[i]), float(tr.lon[i])))
    return {k: sorted(v) for k, v in out.items()}


def load_qtrack_ext():
    import xarray as xr
    p = PILOT / "qtrack_run" / "AEW_pp_east.nc"
    assert p.exists(), "run scripts/aewc_v2_qtrack_probe.py first"
    ds = xr.open_dataset(p)
    lon, lat = ds["AEW_lon"].values, ds["AEW_lat"].values
    systems = np.asarray(ds["system"].values, dtype=int)
    times = pd.DatetimeIndex(ds["time"].values)
    out = {}
    for s in range(lon.shape[0]):
        g = np.isfinite(lon[s]) & np.isfinite(lat[s])
        pts = [(np.datetime64(times[j]), float(lat[s, j]), float(lon[s, j]))
               for j in np.where(g)[0]]
        if pts:
            out[int(systems[s])] = pts
    ds.close()
    return out


def load_aewdat():
    d = json.loads((PILOT / "wts2004.json").read_text())
    s = d["sets"][0]
    out = {}
    for k, trk in enumerate(s["tracks"]):
        seen = {}
        for e in trk["edges"]:
            for n in [e["parent"]] + list(e.get("children", [])):
                o = n.get("object") or n["feature"]
                key = (n["time"], o["id"])
                if key in seen:
                    continue
                pts = o["properties"]["linePts"]
                seen[key] = (float(np.mean([p["lat"] for p in pts])),
                             float(np.mean([p["lon"] for p in pts])))
        by = defaultdict(list)
        for (t, _), c in seen.items():
            by[t].append(c)
        pts = sorted((np.datetime64(t),
                      float(np.mean([c[0] for c in v])),
                      float(np.mean([c[1] for c in v])))
                     for t, v in by.items())
        if pts:
            out[k] = pts
    return out


def associate(storm, tracks):
    """The prototype audit row for one storm against one wave record."""
    cands = []
    for wid, pts in tracks.items():
        best = None
        for t, la, lo in pts:
            dt_signed = (t - storm["genesis_time"]) / np.timedelta64(1, "h")
            if abs(dt_signed) > TIME_TOL / np.timedelta64(1, "h"):
                continue
            d = float(_gc_km(la, lo, storm["genesis_lat"],
                             storm["genesis_lon"]))
            if d <= RADIUS_KM and (best is None or d < best[0]):
                best = (d, float(dt_signed))
        if best:
            cands.append((best[0], best[1], wid))
    cands.sort()
    if not cands:
        return {"matched": False, "competing": 0}
    d, dt, wid = cands[0]
    return {"matched": True, "wave_id": int(wid), "distance_km": int(round(d)),
            "time_offset_h": int(round(dt)), "competing": len(cands) - 1}


def main():
    for rel, want in INPUT_SHA256.items():
        got = _sha256(ROOT / rel)
        assert got == want, f"{rel}: sha256 {got[:12]}... != pinned"

    storms = load_storms()
    records = {"v1": load_v1(), "qtrack_ext": load_qtrack_ext(),
               "aewdat": load_aewdat()}

    rows = []
    counts = {k: 0 for k in records}
    assoc_map = {}
    for st in storms:
        row = {"storm": f"{st['name']} ({st['sid']})",
               "genesis": f"{str(st['genesis_time'])[:13]}Z "
                          f"{st['genesis_lat']:.1f}N "
                          f"{abs(st['genesis_lon']):.1f}"
                          f"{'W' if st['genesis_lon'] < 0 else 'E'}",
               "nature": st["first_status"], "associations": {}}
        assoc_map[st["sid"]] = {}
        for name, tracks in records.items():
            a = associate(st, tracks)
            row["associations"][name] = a
            assoc_map[st["sid"]][name] = a["wave_id"] if a["matched"] else None
            if a["matched"]:
                counts[name] += 1
                row[name] = (f"wave {a['wave_id']} at {a['distance_km']} km, "
                             f"{a['time_offset_h']:+d} h, "
                             f"{a['competing']} competing")
            else:
                row[name] = "no match"
        rows.append(row)

    if MUTATION == "timeshift10d":
        bad = [k for k, v in counts.items() if v == EXPECTED["matched"][k]]
        if len(bad) == len(counts):
            print("MUTATION timeshift10d: every match count unchanged; the "
                  "association does not observe time (a finding)")
            sys.exit(2)
        print(f"MUTATION timeshift10d: match counts changed as required "
              f"(measured {counts} against pinned {EXPECTED['matched']})")
        return

    assert len(storms) == EXPECTED["n_storms"], len(storms)
    assert counts == EXPECTED["matched"], counts
    got_sids = sorted(st["sid"] for st in storms)
    assert got_sids == EXPECTED_SIDS, got_sids
    assert assoc_map == EXPECTED_ASSOC, assoc_map

    out = PILOT / "results" / "ibtracs_prototype_2004.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"criteria": {
        "genesis": "first best-track point",
        "match": "wave position within 6 h and 500 km of genesis",
        "records": {"v1": "deduplicated ERA-Interim",
                    "qtrack_ext": "local extended run",
                    "aewdat": "published wts2004"}},
        "match_counts": counts, "rows": rows}, indent=2, default=str))

    print(f"{len(storms)} storms with June-October 2004 genesis")
    for r in rows:
        print(f"  {r['storm']:32s} {r['genesis']:24s} "
              f"v1[{r['v1']}] qtrack[{r['qtrack_ext']}] aewdat[{r['aewdat']}]")
    print(f"matched: {counts}")
    print(f"results written to {out}")
    print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
