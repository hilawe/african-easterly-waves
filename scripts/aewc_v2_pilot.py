"""One-season archive pilot for the AEWC v2 tracker decision (season 2005).

Compares three AEW track records over a common window and domain:
  v1     AEWC version 1 (Belanger), data/aewc/ERA-Int_ew_700hPa_2005_AFR.nc
  aewdat Fischer et al. (2024) trough-line archive, data/aewc_v2_pilot/wts2005.json
  qtrack Lawton et al. (2022) QTrack ERA5 archive,
         data/aewc_v2_pilot/AEW_tracks_post_processed_year_2005.nc

Window: 2005-06-01 00Z to 2005-10-31 18Z, 6-hourly (612 steps, the native window of
both external archives). Analysis domain: lon [-75, 40], lat [0, 35], the
intersection of the three records' stated or observed domains. East-of-40E counts
are measured OUTSIDE that common domain and say so. The script is specific to
season 2005; the input files are pinned by SHA-256.

Every number published in docs/aewc_v2/ARCHIVE_PILOT_2005.md is guarded by a named
check in the EXPECTED block or by a named relational check. Checks run through a
harness that records every outcome, so a mutation reports every failed check
rather than only the first assertion raised.

Centers: v1 ships trough centroids. AEWDAT ships trough polylines; the center used
here is the unweighted mean of all polyline vertices of the track's objects at that
time (a schema choice, stated in the results). QTrack ships point centers (the
unsmoothed AEW_lon/AEW_lat), labeled by the file's 1-based `system` coordinate.

Run from the repo root:  .venv/bin/python scripts/aewc_v2_pilot.py
Mutations (each must fail the named check; the harness verifies that and exits
nonzero if the named check did NOT fail):
  --mutate=radius90    -> null_sanity_v1_aewdat
  --mutate=noshuffle   -> null_sanity_v1_aewdat
  --mutate=minshared1  -> binding_representatives
  --mutate=timeshift6h -> dlon_bias_v1_qtrack
  --mutate=lonflip     -> null_sanity_v1_aewdat
  --mutate=eastdrop    -> east_qtrack_any (QTrack lons east of 40E moved to 100W;
                          added after an external review constructed exactly this
                          mutation and every then-existing check survived it)
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aew.data.aewc import load_aewc_troughs  # noqa: E402

PILOT = ROOT / "data" / "aewc_v2_pilot"
RESULTS = PILOT / "results"

T0 = np.datetime64("2005-06-01T00:00:00")
T1 = np.datetime64("2005-10-31T18:00:00")
N_STEPS = 612

LON_MIN, LON_MAX = -75.0, 40.0
LAT_MIN, LAT_MAX = 0.0, 35.0

MIN_OBS_MATCH = 8          # >= 8 in-window observations to be matchable
MIN_SHARED_TIMES = 4       # >= 4 shared timestamps to count as a match
RADIUS_KM = {"r300": 300.0, "r500": 500.0, "r800": 800.0}
PRIMARY_R = "r500"
N_NULL = 20
NULL_SEED = 20260826

INPUT_SHA256 = {
    "data/aewc/ERA-Int_ew_700hPa_2005_AFR.nc":
        "82233f45ff766fb0de4d599242a6d7154d72404726a6dfddd328b7106171137c",
    "data/aewc_v2_pilot/wts2005.json":
        "8f5eb49ee1b076d740c4941db9ce5987ffda63a6e32ed29ba8e1a35fa253e272",
    "data/aewc_v2_pilot/AEW_tracks_post_processed_year_2005.nc":
        "11a51cc4a989e45a2c287aac483183276a0718e040fa2f6f7cebe8570668f9e7",
}

# Every published number. Floats carry the tolerance used to compare them.
EXPECTED = {
    # full-year v1 file (binding scope)
    "v1_full_n_tracks": 485, "v1_full_n_obs": 7470,
    "v1_full_excess": 2593, "v1_full_merged": 113,
    "v1_full_excess_frac": (0.34712, 5e-5),
    # window+domain gate
    "v1_win_n_tracks": 149, "v1_win_n_obs": 2344,
    "v1_win_excess": 763, "v1_win_merged": 32,
    "v1_win_excess_frac": (0.32551, 5e-5),
    "aewdat_excess": 0, "aewdat_merged": 0, "aewdat_n_tracks": 64,
    "aewdat_n_obs": 1285,
    "qtrack_excess": 28, "qtrack_merged": 2, "qtrack_n_tracks": 64,
    "qtrack_n_obs": 1936,
    "qtrack_dup_pairs": {(32, 33): 16, (56, 57): 12},   # 1-based system labels
    "aewdat_multi_object_track_times": 0,
    "aewdat_timesteps_in_window": 601, "qtrack_timesteps_in_window": 606,
    # matchable tracks and their sizes
    "matchable": {"v1": 86, "aewdat": 64, "qtrack": 64},
    "median_obs": {"v1": 12.0, "aewdat": 17.0, "qtrack": 31.0},
    "median_elapsed_h": {"v1": 93.0, "aewdat": 96.0, "qtrack": 180.0},
    # matched pairs per radius, and the symmetric score 2m/(nx+ny)
    "n_pairs": {"v1_aewdat": {"r300": 10, "r500": 24, "r800": 27},
                "v1_qtrack": {"r300": 15, "r500": 25, "r800": 36},
                "aewdat_qtrack": {"r300": 16, "r500": 37, "r800": 41}},
    "sym": {"v1_aewdat": {"r300": (0.1333, 1e-3), "r500": (0.3200, 1e-3),
                          "r800": (0.3600, 1e-3)},
            "v1_qtrack": {"r300": (0.2000, 1e-3), "r500": (0.3333, 1e-3),
                          "r800": (0.4800, 1e-3)},
            "aewdat_qtrack": {"r300": (0.2500, 1e-3), "r500": (0.5781, 1e-3),
                              "r800": (0.6406, 1e-3)}},
    # primary-radius offsets, biases, matched-pair medians
    "offset_median_km": {"v1_aewdat": (260.9, 1.0), "v1_qtrack": (254.5, 1.0),
                         "aewdat_qtrack": (275.5, 1.0)},
    "offset_p25_km": {"v1_aewdat": (169.3, 1.0), "v1_qtrack": (163.9, 1.0),
                      "aewdat_qtrack": (161.7, 1.0)},
    "offset_p75_km": {"v1_aewdat": (411.8, 1.0), "v1_qtrack": (360.7, 1.0),
                      "aewdat_qtrack": (435.9, 1.0)},
    "dlon_median": {"v1_aewdat": (0.097, 5e-3), "v1_qtrack": (0.182, 5e-3),
                    "aewdat_qtrack": (0.092, 5e-3)},
    "null_mean": {"v1_aewdat": (0.0453, 1e-3), "v1_qtrack": (0.0599, 1e-3),
                  "aewdat_qtrack": (0.1086, 1e-3)},
    "lifetime_h": {"v1_aewdat": {"v1": 123.0, "aewdat": 111.0},
                   "v1_qtrack": {"v1": 120.0, "qtrack": 240.0},
                   "aewdat_qtrack": {"aewdat": 126.0, "qtrack": 228.0}},
    "lon_extent": {"v1_aewdat": {"v1": (24.8, 0.1), "aewdat": (31.0, 0.1)},
                   "v1_qtrack": {"v1": (18.2, 0.1), "qtrack": (56.9, 0.1)},
                   "aewdat_qtrack": {"aewdat": (32.2, 0.1), "qtrack": (57.2, 0.1)}},
    # east of 40E (outside the common domain)
    "east": {"aewdat": {"any": 0, "first": 0, "max_lon": (36.440, 5e-3)},
             "qtrack": {"any": 4, "first": 4, "max_lon": (41.903, 5e-3)}},
    # derived quantities published in the note
    "qtrack_excess_frac": (0.01446, 5e-5),
    "ratio_modern_v1aewdat_r500": (1.807, 5e-3),
    "ratio_modern_v1qtrack_r500": (1.734, 5e-3),
    "ratio_modern_v1qtrack_r800": (1.335, 5e-3),
}

EXPECTED_N_CHECKS = 103   # asserted after the run; a changed count means checks
                          # were added or lost without this constant moving


DLON_BIAS_CAP = 0.5   # deg; clean medians are <= 0.19, a 6 h shift gives ~1.0-1.5

MUTATION = None
for a in sys.argv[1:]:
    if a.startswith("--mutate="):
        MUTATION = a.split("=", 1)[1]

MUTATION_MUST_FAIL = {
    "radius90": "null_sanity_v1_aewdat",
    "noshuffle": "null_sanity_v1_aewdat",
    "minshared1": "binding_representatives",
    "timeshift6h": "dlon_bias_v1_qtrack",
    "lonflip": "null_sanity_v1_aewdat",
    "eastdrop": "east_qtrack_any",
}

# ------------------------------------------------------------- check harness

FAILED: list[tuple[str, str]] = []
_N_CHECKS = 0


def check(name, cond, msg):
    """Record a named check; raise immediately only on a clean (unmutated) run."""
    global _N_CHECKS
    _N_CHECKS += 1
    if not cond:
        FAILED.append((name, str(msg)))
        if MUTATION is None:
            raise AssertionError(f"{name}: {msg}")


def check_close(name, value, expected_tol, what):
    exp, tol = expected_tol
    check(name, value is not None and abs(value - exp) <= tol,
          f"{what}: {value} not within {tol} of {exp}")


# ---------------------------------------------------------------- loading

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_v1(dedup):
    tr = load_aewc_troughs(str(ROOT / "data/aewc/ERA-Int_ew_700hPa_2005_AFR.nc"),
                           dedup=dedup)
    df = pd.DataFrame({"time": pd.DatetimeIndex(tr.time),
                       "lat": tr.lat, "lon": tr.lon,
                       "track": tr.variables["traj_id"]})
    return df, tr


def load_aewdat():
    d = json.loads((PILOT / "wts2005.json").read_text())
    sets = d["sets"]
    check("aewdat_schema", len(sets) == 1 and float(sets[0]["level"]) == 700.0,
          "expected a single 700 hPa set")
    rows = []
    multi_object_track_times = 0
    for k, trk in enumerate(sets[0]["tracks"]):
        seen = {}
        for e in trk["edges"]:
            for n in [e["parent"]] + list(e.get("children", [])):
                t = np.datetime64(n["time"])
                oid = n["object"]["id"]
                if (t, oid) in seen:
                    continue
                pts = n["object"]["properties"]["linePts"]
                seen[(t, oid)] = (float(np.mean([p["lat"] for p in pts])),
                                  float(np.mean([p["lon"] for p in pts])))
        by_time = defaultdict(list)
        for (t, _oid), (la, lo) in seen.items():
            by_time[t].append((la, lo))
        for t, cents in by_time.items():
            if len(cents) > 1:
                multi_object_track_times += 1
            la = float(np.mean([c[0] for c in cents]))
            lo = float(np.mean([c[1] for c in cents]))
            if MUTATION == "lonflip":
                lo = -lo
            rows.append((t, la, lo, k))
    df = pd.DataFrame(rows, columns=["time", "lat", "lon", "track"])
    # normalize to ns so integer timestamp keys are comparable across records
    # (the JSON parse otherwise yields datetime64[s] and every key comparison
    # against the ns-based records silently fails)
    df["time"] = pd.DatetimeIndex(df["time"]).astype("datetime64[ns]")
    return df, multi_object_track_times


def load_qtrack():
    ds = xr.open_dataset(PILOT / "AEW_tracks_post_processed_year_2005.nc")
    lon = ds["AEW_lon"].values
    lat = ds["AEW_lat"].values
    systems = np.asarray(ds["system"].values, dtype=int)   # 1-based labels
    times = pd.DatetimeIndex(ds["time"].values)
    if MUTATION == "timeshift6h":
        times = times + pd.Timedelta(hours=6)
    rows = []
    for s in range(lon.shape[0]):
        good = np.isfinite(lon[s]) & np.isfinite(lat[s])
        for j in np.where(good)[0]:
            lo = float(lon[s, j])
            if MUTATION == "eastdrop" and lo > 40.0:
                lo = -100.0
            rows.append((times[j], float(lat[s, j]), lo, int(systems[s])))
    ds.close()
    return pd.DataFrame(rows, columns=["time", "lat", "lon", "track"])


def window_domain(df):
    m = (df["time"] >= T0) & (df["time"] <= T1)
    m &= (df["lon"] >= LON_MIN) & (df["lon"] <= LON_MAX)
    m &= (df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX)
    return df.loc[m].reset_index(drop=True)


# ------------------------------------------------- the dedup gate, ported

def dedup_gate(df, min_shared=3):
    """The deduplicate() algorithm's measurement half, on a generic track table.

    Port of the key construction, pair counting, and threshold union of
    src/aew/data/aewc.py deduplicate(). Bound to the repo function by
    binding_checks(), which compares the chosen representative-track set and the
    kept-row count, not only the component count.
    """
    if MUTATION == "minshared1":
        min_shared = 1
    t = pd.DatetimeIndex(df["time"]).asi8
    keys = list(zip(t.tolist(),
                    np.round(df["lat"].to_numpy(), 3).tolist(),
                    np.round(df["lon"].to_numpy(), 3).tolist()))
    tid = df["track"].to_numpy()
    groups = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    multi = excess = 0
    pair = defaultdict(int)
    for idxs in groups.values():
        tset = sorted(set(int(tid[i]) for i in idxs))
        if len(tset) > 1:
            multi += 1
            excess += len(tset) - 1
        for a in range(len(tset)):
            for b in range(a + 1, len(tset)):
                pair[(tset[a], tset[b])] += 1
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    for u in np.unique(tid):
        find(int(u))
    for (a, b), n in pair.items():
        if n >= min_shared:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    comp = {int(u): find(int(u)) for u in np.unique(tid)}
    members = defaultdict(list)
    for u, r in comp.items():
        members[r].append(u)
    # representative per component: longest member, tie broken by smallest id
    # (the repo rule with tie="smallest")
    counts = pd.Series(tid).value_counts()
    reps = set()
    for r, us in members.items():
        best = max(us, key=lambda u: (int(counts.get(u, 0)), -u))
        reps.add(int(best))
    kept_rows = int(np.isin(tid, list(reps)).sum())
    return {
        "n_tracks": int(len(np.unique(tid))),
        "n_obs": int(len(df)),
        "multi_track_keys": int(multi),
        "excess_duplicate_obs": int(excess),
        "excess_fraction": float(excess / max(len(df), 1)),
        "n_components": int(len(members)),
        "n_merged_components": int(sum(1 for us in members.values() if len(us) > 1)),
        "_pair_counts": dict(pair),
        "_representatives": reps,
        "_kept_rows": kept_rows,
    }


def binding_checks(gate_full, v1_dd_full_df, tr_dd):
    """Bind the port to the repo deduplicate() on the full v1 2005 record.

    Compares the component count, the SET of representative trajectories the
    longest/tie-smallest rule selects, and the kept-row count. This does NOT
    prove full partition equality: the repo function's output does not expose
    the component membership of dropped trajectories, so a partition that
    differed only in which component a dropped trajectory belongs to would
    pass. The port reproduces the repo algorithm line for line, and the
    minshared1 mutation shows the representative-set comparison detects a
    changed partition that the component count alone might not.
    """
    repo_reps = set(int(x) for x in np.unique(tr_dd.variables["orig_traj_id"]))
    repo_waves = len(np.unique(tr_dd.variables["wave_id"]))
    check("binding_components", gate_full["n_components"] == repo_waves,
          f"port components {gate_full['n_components']} != repo waves {repo_waves}")
    check("binding_representatives", gate_full["_representatives"] == repo_reps,
          f"port representative set differs from repo "
          f"({len(gate_full['_representatives'] ^ repo_reps)} symmetric difference)")
    check("binding_kept_rows", gate_full["_kept_rows"] == len(v1_dd_full_df),
          f"port kept rows {gate_full['_kept_rows']} != repo {len(v1_dd_full_df)}")


# ------------------------------------------------------------- matching

def _tracks(df):
    out = {}
    for k, g in df.groupby("track"):
        if len(g) >= MIN_OBS_MATCH:
            g = g.sort_values("time")
            out[k] = (pd.DatetimeIndex(g["time"]).asi8,
                      g["lat"].to_numpy(), g["lon"].to_numpy())
    return out


def _gc_km(lat1, lon1, lat2, lon2):
    r = np.pi / 180.0
    a = (np.sin((lat2 - lat1) * r / 2) ** 2
         + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin((lon2 - lon1) * r / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def match_pair(tx, ty, radius_km):
    if MUTATION == "radius90":
        radius_km = 10000.0
    cands = []
    for kx, (t1, la1, lo1) in tx.items():
        ix = {t: i for i, t in enumerate(t1)}
        for ky, (t2, la2, lo2) in ty.items():
            shared = [(ix[t], j) for j, t in enumerate(t2) if t in ix]
            if len(shared) < MIN_SHARED_TIMES:
                continue
            i1 = np.array([s[0] for s in shared])
            i2 = np.array([s[1] for s in shared])
            d = _gc_km(la1[i1], lo1[i1], la2[i2], lo2[i2])
            if np.mean(d) <= radius_km:
                cands.append((float(np.mean(d)), kx, ky, d, lo1[i1] - lo2[i2]))
    cands.sort(key=lambda c: c[0])
    used_x, used_y, pairs = set(), set(), []
    for meand, kx, ky, d, dlon in cands:
        if kx in used_x or ky in used_y:
            continue
        used_x.add(kx); used_y.add(ky)
        pairs.append({"x": int(kx), "y": int(ky), "mean_km": meand,
                      "dists_km": d, "dlon": dlon})
    return pairs


def null_match_rate(tx, ty, radius_km, rng):
    """Match rate under circular time shifts of ty (10 to 40 days, both signs).

    A seeded 20-draw sanity reference, not a false-positive-controlled test;
    used only in the ratio check named null_sanity_*.
    """
    rates = []
    span = (T1 - T0).astype("timedelta64[h]").astype(int)
    t0_ns = T0.astype("datetime64[ns]").astype(np.int64)
    for _ in range(N_NULL):
        if MUTATION == "noshuffle":
            shift_h = 0
        else:
            days = rng.integers(10, 41)
            sign = rng.choice([-1, 1])
            shift_h = int(sign * days * 24)
        ty_s = {}
        for k, (t, la, lo) in ty.items():
            th = ((t - t0_ns) // 3_600_000_000_000 + shift_h) % (span + 6)
            ty_s[k] = (t0_ns + th * 3_600_000_000_000, la, lo)
        pairs = match_pair(tx, ty_s, radius_km)
        rates.append(len(pairs) / max(len(tx), 1))
    return float(np.mean(rates)), float(np.max(rates))


def median_elapsed_h(tracks):
    return float(np.median([(t[0].max() - t[0].min()) / 3_600_000_000_000
                            for t in tracks.values()]))


# ------------------------------------------------------------------ main

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(NULL_SEED)
    res = {"season": 2005, "window": [str(T0), str(T1)],
           "domain": {"lon": [LON_MIN, LON_MAX], "lat": [LAT_MIN, LAT_MAX]},
           "center_defs": {
               "v1": "shipped trough centroid",
               "aewdat": "unweighted mean of polyline vertices of the track's "
                         "objects at that time",
               "qtrack": "shipped AEW_lon/AEW_lat (unsmoothed), 1-based system "
                         "labels"},
           "mutation": MUTATION}

    # input identity
    for rel, want in INPUT_SHA256.items():
        got = _sha256(ROOT / rel)
        check(f"sha256_{Path(rel).name}", got == want,
              f"{rel}: sha256 {got[:12]}... != pinned {want[:12]}...")

    v1_raw_full, _ = load_v1(dedup=False)
    v1_dd_full, tr_dd = load_v1(dedup=True)
    aewdat_full, multi_obj = load_aewdat()
    qtrack_full = load_qtrack()

    check("v1_max_lon", float(v1_raw_full["lon"].max()) <= 40.0,
          "v1 has data east of 40E")

    for name, df in (("aewdat", aewdat_full), ("qtrack", qtrack_full)):
        n_t = int(df.loc[(df["time"] >= T0) & (df["time"] <= T1), "time"].nunique())
        res[f"{name}_timesteps_in_window"] = n_t
        check(f"{name}_timesteps", n_t == EXPECTED[f"{name}_timesteps_in_window"],
              f"{name}: {n_t} in-window timesteps != expected")

    # ---- the dedup gate ------------------------------------------------------
    gate_full = dedup_gate(v1_raw_full)
    binding_checks(gate_full, v1_dd_full, tr_dd)
    check("v1_full_n_tracks", gate_full["n_tracks"] == EXPECTED["v1_full_n_tracks"],
          gate_full["n_tracks"])
    check("v1_full_n_obs", gate_full["n_obs"] == EXPECTED["v1_full_n_obs"],
          gate_full["n_obs"])
    check("v1_full_excess",
          gate_full["excess_duplicate_obs"] == EXPECTED["v1_full_excess"],
          gate_full["excess_duplicate_obs"])
    check("v1_full_merged",
          gate_full["n_merged_components"] == EXPECTED["v1_full_merged"],
          gate_full["n_merged_components"])
    check_close("v1_full_excess_frac", gate_full["excess_fraction"],
                EXPECTED["v1_full_excess_frac"], "full-year excess fraction")

    v1_raw = window_domain(v1_raw_full)
    v1_dd = window_domain(v1_dd_full)
    aewdat = window_domain(aewdat_full)
    qtrack = window_domain(qtrack_full)

    g = {"v1_raw": dedup_gate(v1_raw), "aewdat": dedup_gate(aewdat),
         "qtrack": dedup_gate(qtrack)}
    check("v1_win_n_tracks", g["v1_raw"]["n_tracks"] == EXPECTED["v1_win_n_tracks"],
          g["v1_raw"]["n_tracks"])
    check("v1_win_n_obs", g["v1_raw"]["n_obs"] == EXPECTED["v1_win_n_obs"],
          g["v1_raw"]["n_obs"])
    check("v1_win_excess",
          g["v1_raw"]["excess_duplicate_obs"] == EXPECTED["v1_win_excess"],
          g["v1_raw"]["excess_duplicate_obs"])
    check("v1_win_merged",
          g["v1_raw"]["n_merged_components"] == EXPECTED["v1_win_merged"],
          g["v1_raw"]["n_merged_components"])
    check_close("v1_win_excess_frac", g["v1_raw"]["excess_fraction"],
                EXPECTED["v1_win_excess_frac"], "window excess fraction")
    for name in ("aewdat", "qtrack"):
        check(f"{name}_n_tracks", g[name]["n_tracks"] == EXPECTED[f"{name}_n_tracks"],
              g[name]["n_tracks"])
        check(f"{name}_n_obs", g[name]["n_obs"] == EXPECTED[f"{name}_n_obs"],
              g[name]["n_obs"])
        check(f"{name}_excess",
              g[name]["excess_duplicate_obs"] == EXPECTED[f"{name}_excess"],
              g[name]["excess_duplicate_obs"])
        check(f"{name}_merged",
              g[name]["n_merged_components"] == EXPECTED[f"{name}_merged"],
              g[name]["n_merged_components"])
    check_close("qtrack_excess_frac", g["qtrack"]["excess_fraction"],
                EXPECTED["qtrack_excess_frac"], "qtrack excess fraction")
    qt_pairs = {p: n for p, n in g["qtrack"]["_pair_counts"].items() if n >= 3}
    check("qtrack_dup_pairs", qt_pairs == EXPECTED["qtrack_dup_pairs"],
          f"duplicate pair map {qt_pairs} != expected")
    check("aewdat_multi_object_track_times",
          multi_obj == EXPECTED["aewdat_multi_object_track_times"], multi_obj)
    res["gate"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                   for k, v in g.items()}
    res["gate"]["v1_full"] = {kk: vv for kk, vv in gate_full.items()
                              if not kk.startswith("_")}
    res["qtrack_dup_pairs"] = {f"{a}_{b}": n for (a, b), n in qt_pairs.items()}
    res["aewdat_multi_object_track_times"] = int(multi_obj)

    # ---- matching ------------------------------------------------------------
    tracks = {"v1": _tracks(v1_dd), "aewdat": _tracks(aewdat),
              "qtrack": _tracks(qtrack)}
    res["n_matchable_tracks"] = {k: len(v) for k, v in tracks.items()}
    res["median_matchable_obs"] = {
        k: float(np.median([len(t[0]) for t in v.values()]))
        for k, v in tracks.items()}
    res["median_elapsed_h"] = {k: median_elapsed_h(v) for k, v in tracks.items()}
    for k in tracks:
        check(f"matchable_{k}",
              res["n_matchable_tracks"][k] == EXPECTED["matchable"][k],
              res["n_matchable_tracks"][k])
        check(f"median_obs_{k}",
              res["median_matchable_obs"][k] == EXPECTED["median_obs"][k],
              res["median_matchable_obs"][k])
        check(f"median_elapsed_{k}",
              res["median_elapsed_h"][k] == EXPECTED["median_elapsed_h"][k],
              res["median_elapsed_h"][k])

    res["matching"] = {}
    sym_primary = {}
    for x, y in (("v1", "aewdat"), ("v1", "qtrack"), ("aewdat", "qtrack")):
        pk = f"{x}_{y}"
        entry = {}
        for rk, rkm in RADIUS_KM.items():
            pairs = match_pair(tracks[x], tracks[y], rkm)
            sym = 2 * len(pairs) / (len(tracks[x]) + len(tracks[y]))
            e = {"n_pairs": len(pairs), "sym_score": sym,
                 "frac_x_matched": len(pairs) / max(len(tracks[x]), 1),
                 "frac_y_matched": len(pairs) / max(len(tracks[y]), 1)}
            check(f"n_pairs_{pk}_{rk}",
                  len(pairs) == EXPECTED["n_pairs"][pk][rk], len(pairs))
            check_close(f"sym_{pk}_{rk}", sym, EXPECTED["sym"][pk][rk],
                        f"{pk} {rk} symmetric score")
            if rk == PRIMARY_R:
                sym_primary[pk] = sym
                dists = (np.concatenate([p["dists_km"] for p in pairs])
                         if pairs else np.array([]))
                dlon = (np.concatenate([p["dlon"] for p in pairs])
                        if pairs else np.array([]))
                e["center_offset_km"] = {
                    "median": float(np.median(dists)) if dists.size else None,
                    "p25": float(np.percentile(dists, 25)) if dists.size else None,
                    "p75": float(np.percentile(dists, 75)) if dists.size else None}
                e["dlon_median_deg"] = float(np.median(dlon)) if dlon.size else None
                check_close(f"offset_median_{pk}", e["center_offset_km"]["median"],
                            EXPECTED["offset_median_km"][pk], f"{pk} median offset")
                check_close(f"offset_p25_{pk}", e["center_offset_km"]["p25"],
                            EXPECTED["offset_p25_km"][pk], f"{pk} p25 offset")
                check_close(f"offset_p75_{pk}", e["center_offset_km"]["p75"],
                            EXPECTED["offset_p75_km"][pk], f"{pk} p75 offset")
                check_close(f"dlon_median_{pk}", e["dlon_median_deg"],
                            EXPECTED["dlon_median"][pk], f"{pk} median dlon")
                check(f"dlon_bias_{pk}",
                      e["dlon_median_deg"] is not None
                      and abs(e["dlon_median_deg"]) < DLON_BIAS_CAP,
                      f"systematic longitude offset {e['dlon_median_deg']} deg")
                lt = {x: float(np.median(
                          [(tracks[x][p["x"]][0].max() - tracks[x][p["x"]][0].min())
                           / 3_600_000_000_000 for p in pairs])),
                      y: float(np.median(
                          [(tracks[y][p["y"]][0].max() - tracks[y][p["y"]][0].min())
                           / 3_600_000_000_000 for p in pairs]))}
                ex = {x: float(np.median([tracks[x][p["x"]][2].max()
                                          - tracks[x][p["x"]][2].min()
                                          for p in pairs])),
                      y: float(np.median([tracks[y][p["y"]][2].max()
                                          - tracks[y][p["y"]][2].min()
                                          for p in pairs]))}
                e["matched_lifetime_h_median"] = lt
                e["matched_lon_extent_deg_median"] = ex
                for side in (x, y):
                    check(f"lifetime_{pk}_{side}",
                          lt[side] == EXPECTED["lifetime_h"][pk][side], lt[side])
                    check_close(f"extent_{pk}_{side}", ex[side],
                                EXPECTED["lon_extent"][pk][side],
                                f"{pk} {side} extent")
                nm, nx_ = null_match_rate(tracks[x], tracks[y], rkm, rng)
                e["null_frac_x_mean"] = nm
                e["null_frac_x_max"] = nx_
                check_close(f"null_mean_{pk}", nm, EXPECTED["null_mean"][pk],
                            f"{pk} null mean")
                # sanity ratio (not a false-positive-controlled test): the real
                # directional rate sits well above the shifted-time reference
                check(f"null_sanity_{pk}",
                      e["frac_x_matched"] > 3 * max(nm, 1e-9),
                      f"real {e['frac_x_matched']:.3f} not above 3x null {nm:.3f}")
            entry[rk] = e
        res["matching"][pk] = entry

    # comparative claims on the SYMMETRIC score at the primary radius
    check("modern_pair_over_half", sym_primary["aewdat_qtrack"] > 0.5,
          f"{sym_primary['aewdat_qtrack']:.3f}")
    check("modern_pair_1p5x_v1",
          sym_primary["aewdat_qtrack"] > 1.5 * sym_primary["v1_aewdat"]
          and sym_primary["aewdat_qtrack"] > 1.5 * sym_primary["v1_qtrack"],
          f"modern {sym_primary['aewdat_qtrack']:.3f} vs "
          f"v1 {sym_primary['v1_aewdat']:.3f}/{sym_primary['v1_qtrack']:.3f}")
    check("v1_median_obs_below_qtrack",
          res["median_matchable_obs"]["v1"] < res["median_matchable_obs"]["qtrack"],
          "v1 median matchable track length no longer below qtrack's")
    check_close("ratio_modern_v1aewdat_r500",
                sym_primary["aewdat_qtrack"] / sym_primary["v1_aewdat"],
                EXPECTED["ratio_modern_v1aewdat_r500"], "500km ratio vs v1-aewdat")
    check_close("ratio_modern_v1qtrack_r500",
                sym_primary["aewdat_qtrack"] / sym_primary["v1_qtrack"],
                EXPECTED["ratio_modern_v1qtrack_r500"], "500km ratio vs v1-qtrack")
    r800 = {k: res["matching"][k]["r800"]["sym_score"]
            for k in ("aewdat_qtrack", "v1_qtrack")}
    check_close("ratio_modern_v1qtrack_r800",
                r800["aewdat_qtrack"] / r800["v1_qtrack"],
                EXPECTED["ratio_modern_v1qtrack_r800"], "800km ratio vs v1-qtrack")

    # ---- east of 40E, outside the common domain ------------------------------
    east = {}
    for name, df_full in (("aewdat", aewdat_full), ("qtrack", qtrack_full)):
        m = (df_full["time"] >= T0) & (df_full["time"] <= T1)
        d = df_full.loc[m]
        by = d.groupby("track")
        any_e = by.apply(lambda t: bool((t["lon"] > 40.0).any()),
                         include_groups=False)
        first_e = by.apply(
            lambda t: bool(t.sort_values("time")["lon"].iloc[0] > 40.0),
            include_groups=False)
        east[name] = {"tracks_any_east_of_40E": int(any_e.sum()),
                      "tracks_first_point_east_of_40E": int(first_e.sum()),
                      "max_lon": float(d["lon"].max())}
        check(f"east_{name}_any",
              east[name]["tracks_any_east_of_40E"] == EXPECTED["east"][name]["any"],
              east[name]["tracks_any_east_of_40E"])
        check(f"east_{name}_first",
              east[name]["tracks_first_point_east_of_40E"]
              == EXPECTED["east"][name]["first"],
              east[name]["tracks_first_point_east_of_40E"])
        check_close(f"east_{name}_max_lon", east[name]["max_lon"],
                    EXPECTED["east"][name]["max_lon"], f"{name} max lon")
    res["east_of_40E"] = east

    # ---- write and report ----------------------------------------------------
    out = RESULTS / ("pilot_2005.json" if MUTATION is None
                     else f"pilot_2005_mut_{MUTATION}.json")

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()
                    if k not in ("dists_km", "dlon")}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        return o

    res["failed_checks"] = sorted(set(n for n, _ in FAILED))
    out.write_text(json.dumps(_clean(res), indent=2, default=str))
    print(json.dumps(_clean(res), indent=2, default=str))
    print(f"\nresults written to {out}")

    if MUTATION is None:
        assert _N_CHECKS == EXPECTED_N_CHECKS, (
            f"named-check count {_N_CHECKS} != expected {EXPECTED_N_CHECKS}; "
            f"update EXPECTED_N_CHECKS and the note together")
        print(f"ALL {_N_CHECKS} NAMED CHECKS PASSED")
    else:
        target = MUTATION_MUST_FAIL[MUTATION]
        names = [n for n, _ in FAILED]
        print(f"MUTATION {MUTATION}: {len(set(names))} checks failed")
        if target in names:
            print(f"MUTATION {MUTATION}: named check '{target}' FAILED as required")
        else:
            print(f"MUTATION {MUTATION}: named check '{target}' DID NOT FAIL; "
                  f"the mutation is not observed by its named check (a finding)")
            sys.exit(2)


if __name__ == "__main__":
    main()
