#!/usr/bin/env python3
"""Two probes from the review of the matched phase and envelope reading
(`EASTERN_PHASE_ENVELOPE_MATCHED_2026-10-27.md`, section 12), both run after the reading and
neither declared before it. The reference probe repeats the two primary readings with the
paper's kind of reference, the meridional wind at the single point 10 N at the base longitude,
in place of the 8 to 15 N band mean, everything else held fixed. The exposure probe counts, for
every reading of the phase and envelope effort at 700 and 600 hPa, whether the pooled path and
how many of the 1,000 resampled paths try a following window that reaches an end column of the
grid (10 W or 75 E), where the density is undefined. Exposure says that an edge can affect a
reading. It does not say by how much or in which direction.

    python3 scripts/eastern_phase_envelope_probes.py reference --matched PATH --out-dir DIR
    python3 scripts/eastern_phase_envelope_probes.py exposure --xband DIR --stored700 PATH --stored600 PATH --out-dir DIR
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eastern_lag_regression as LR  # noqa: E402
import eastern_phase_envelope as PE  # noqa: E402
import eastern_phase_envelope_matched as M  # noqa: E402
import pilot_eastern_propagation as P  # noqa: E402

SCRIPT_SHA256_AT_LOAD = PE.sha256_of(os.path.abspath(__file__))
POINT_LAT = 10.0
MAX_LAG = int(max(abs(PE.LAGS)))


# ------------------------------------------------------------------ the reference probe

def point_series(year, level, lon, lat=POINT_LAT):
    """One season's meridional wind at the grid point (lat, lon) and `level`, refused unless
    the surface pressure exceeds the level strictly at every time, stored to two decimals as
    the band means are."""
    import netCDF4
    if year == P.HELD_OUT:
        raise SystemExit("REFUSED: the held-out season is never opened")
    d = netCDF4.Dataset(os.path.join(P.DATA_DIR, f"era5_v{level}_{year}_6h_region.nc"))
    try:
        la = np.asarray(d.variables["latitude"][:], float); lo = np.asarray(d.variables["longitude"][:], float)
    finally:
        d.close()
    r, c = np.flatnonzero(np.isclose(la, lat)), np.flatnonzero(np.isclose(lo, lon))
    if r.size != 1 or c.size != 1:
        raise SystemExit(f"REFUSED: no single grid point at {lat:g} N, {lon:g} E")
    blk = P.read_season(year, level, r, c)
    v, sp = blk["v"][:, 0, 0], blk["sp"][:, 0, 0]
    if not (np.isfinite(v).all() and (sp > level * 100.0).all()):
        raise SystemExit(f"REFUSED: the point {lat:g} N, {lon:g} E is not valid at {level} hPa throughout {year}")
    return np.round(v, 2)


def sums_with_reference(S, ref):
    """One season's frozen sums for the band means S with the reference series `ref` in place
    of a band-mean column: the reference is appended as a last column, the frozen sums are
    formed against it, and the appended column is removed from the field sums again."""
    Z = PE.analytic(np.column_stack([S, ref]))
    s = PE.season_sums(Z, Z.shape[1] - 1)
    for k in ("sb", "sab", "sbb"):
        s[k] = s[k][:, :-1]
    return s


def reference_results(seasons, refs, base, lons=M.MLONS, n_boot=PE.N_BOOT):
    col = int(np.flatnonzero(np.isclose(lons, base))[0])
    st = PE.stack([sums_with_reference(S, r) for S, r in zip(seasons, refs)])
    return PE.base_result(st, col, PE.boot_weights(len(seasons), n_boot), lons)


def run_reference(matched_path, n_boot=PE.N_BOOT):
    """Both primary configurations with the band-mean reference (checked against the stored
    matched record) and with the 10 N point reference."""
    with open(matched_path, encoding="utf-8") as fh:
        stored = json.load(fh)["results"]["primary and transect"]
    out = {}
    for level, base in M.PRIMARY:
        seasons = [M.band_means(y, level, M.MATCHED_BAND) for y in M.SHARED_YEARS]
        refs = [point_series(y, level, base) for y in M.SHARED_YEARS]
        band = json.loads(json.dumps(M.results(seasons, (base,), M.MLONS, n_boot)))[f"{base:g}"]
        diffs = M._same(band, stored[f"{level}"][f"{base:g}"])
        if diffs:
            raise SystemExit("REFUSED: the band-mean reading does not reproduce the stored matched record, " + "; ".join(diffs[:5]))
        point = json.loads(json.dumps(reference_results(seasons, refs, base, n_boot=n_boot)))
        col = int(np.flatnonzero(np.isclose(M.MLONS, base))[0])
        corr = float(np.corrcoef(np.concatenate([S[:, col] for S in seasons]), np.concatenate(refs))[0, 1])
        out[f"{level} hPa, {base:g} E"] = {"band_mean_reference": band, "point_reference": point, "reference_correlation": corr}
    return out


# ------------------------------------------------------------------ the exposure probe

def touches(r, base, lons):
    """Whether any following window the reading tried reaches an end column: the window at lag
    0 around the base and, for every followed position short of the largest lag, the window
    around it that the next lag tried."""
    centers = [float(base)] + [x for L, x in zip(r["lags"], r["positions"]) if abs(L) < MAX_LAG]
    return any(x - PE.TRACK_HALF_DEG <= lons[0] + 1e-9 or x + PE.TRACK_HALF_DEG >= lons[-1] - 1e-9 for x in centers)


def exposure_one(seasons_Z, base, stored, lons=PE.LONS, n_boot=PE.N_BOOT):
    """The pooled reading (checked against the stored one) and the count of resamples whose
    paths touch an edge, each resample read in the pooled reading's direction as the frozen
    intervals are."""
    col = int(np.flatnonzero(np.isclose(lons, base))[0])
    st = PE.stack([PE.season_sums(Z, col) for Z in seasons_Z])
    C, E, N = PE.maps(st)
    r = PE.readings(C, N, col, None, lons)
    for k in ("envelope_speed", "positions", "lags", "sign"):
        if M._same(json.loads(json.dumps(r[k])), stored[k]):
            raise SystemExit(f"REFUSED: the pooled reading at {base:g} E does not reproduce the stored {k}")
    n = 0
    for w in PE.boot_weights(len(seasons_Z), n_boot):
        Cb, Eb, Nb = PE.maps(st, w)
        n += touches(PE.readings(Cb, Nb, col, r["sign"], lons), base, lons)
    return {"pooled_touches": touches(r, base, lons), "resamples_touching": n, "n_boot": n_boot,
            "envelope_class": stored["envelope_class"], "envelope_speed": stored["envelope_speed"]}


def run_exposure(xband, stored700, stored600, n_boot=PE.N_BOOT):
    files, by = PE.load_xband(xband)
    PE.check_level_builder(by)
    seasons = {700: {b: [LR.season_samples(by[y], b) for y in PE.YEARS] for b in PE.BAND_LAT}}
    built = {y: PE.level_band_means(y, 600) for y in PE.YEARS}
    seasons[600] = {b: [built[y][b] for y in PE.YEARS] for b in PE.BAND_LAT}
    out = {}
    for level, path in ((700, stored700), (600, stored600)):
        with open(path, encoding="utf-8") as fh:
            stored = json.load(fh)["results"]
        out[f"{level}"] = {}
        for band in PE.BAND_LAT:
            Zs = [PE.analytic(S) for S in seasons[level][band]]
            out[f"{level}"][band] = {f"{b:g}": exposure_one(Zs, b, stored[band][f"{b:g}"], n_boot=n_boot) for b in PE.BASES}
    return files, out


# ------------------------------------------------------------------ text and main

def _f(v):
    return "" if v is None else f"{v:+.2f}"


def reference_text(res):
    L = ["# the matched reading's primary configurations with a 10 N point reference (a probe after the reading)", "",
         "| configuration | reference | envelope class | envelope speed (interval) | undefined | phase speed (interval) |",
         "| --- | --- | --- | --- | --- | --- |"]
    for k, v in res.items():
        for name, key in (("band mean, 8 to 15 N", "band_mean_reference"), ("point, 10 N", "point_reference")):
            r = v[key]
            L.append(f"| {k} | {name} | {r['envelope_class']} | {_f(r['envelope_speed'])} ({_f(r['envelope_interval'][0])} to {_f(r['envelope_interval'][1])}) | "
                     f"{r['envelope_undefined']:.3f} | {_f(r['phase_speed'])} ({_f(r['phase_interval'][0])} to {_f(r['phase_interval'][1])}) |")
    L += [""] + [f"- {k}, correlation of the two raw reference series over the seasons {v['reference_correlation']:.3f}" for k, v in res.items()]
    return "\n".join(L) + "\n"


def exposure_text(res):
    L = ["# edge exposure of the phase and envelope effort's readings (a probe after the matched reading)", "",
         "Each cell gives the stored envelope class, then whether the pooled path tries a window at an end column (yes or no), "
         "then how many of the 1,000 resampled paths do.", ""]
    bases = [f"{b:g}" for b in PE.BASES]
    for level, v in res.items():
        L += [f"## {level} hPa", "", "| band | " + " | ".join(f"{b} E" for b in bases) + " |", "| --- |" + " --- |" * len(bases)]
        for band, row in v.items():
            L.append(f"| {band} | " + " | ".join(f"{row[b]['envelope_class']}, {'yes' if row[b]['pooled_touches'] else 'no'}, {row[b]['resamples_touching']}" for b in bases) + " |")
        L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("reference", "exposure"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--matched", default=None)
    ap.add_argument("--xband", default=None)
    ap.add_argument("--stored700", default=None)
    ap.add_argument("--stored600", default=None)
    a = ap.parse_args(argv)
    M.check_frozen()
    pj, pm = (os.path.join(a.out_dir, f"eval60_phase_env_{a.mode}.{e}") for e in ("json", "md"))
    PE.refuse_existing([pj, pm])
    gen = {"script": "scripts/eastern_phase_envelope_probes.py", "sha256": SCRIPT_SHA256_AT_LOAD, "mode": a.mode,
           "matched_sha256": M.SCRIPT_SHA256_AT_LOAD, "frozen_sha256": PE.SCRIPT_SHA256_AT_LOAD,
           "declared": "EASTERN_PHASE_ENVELOPE_MATCHED_2026-10-27.md, section 12 (after the reading)"}
    if a.mode == "reference":
        if not a.matched:
            raise SystemExit("REFUSED: --matched is required for the reference probe")
        res = run_reference(a.matched)
        gen.update({"matched": {"path": a.matched, "sha256": PE.sha256_of(a.matched)}, "point_lat": POINT_LAT,
                    "inputs": {os.path.basename(p): PE.sha256_of(p) for p in M.input_files(M.SHARED_YEARS)}})
        text = reference_text(res)
    else:
        if not (a.xband and a.stored700 and a.stored600):
            raise SystemExit("REFUSED: --xband, --stored700 and --stored600 are required for the exposure probe")
        files, res = run_exposure(a.xband, a.stored700, a.stored600)
        gen.update({"xband": {os.path.basename(p): PE.sha256_of(p) for p in files},
                    "stored": {p: PE.sha256_of(p) for p in (a.stored700, a.stored600)}})
        text = exposure_text(res)
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "results": res}, fh)
    with open(pm, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
