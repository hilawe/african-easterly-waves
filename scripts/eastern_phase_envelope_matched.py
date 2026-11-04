#!/usr/bin/env python3
"""Phase and envelope readings at the published bases of Diaz and Aiyyer (2013)
(`EASTERN_PHASE_ENVELOPE_MATCHED_2026-10-27.md`). The frozen estimator of
scripts/eastern_phase_envelope.py, imported unchanged and refused when its digest differs from
the declared one, reads ERA5 meridional-wind band means on a 45 W to 35 E grid. The primary
readings are the two configurations the paper publishes (850 hPa with the base at 15 W, 700 hPa
with the base at 30 W, the 8 to 15 N band, the 17 seasons the two periods share). A transect of
bases from 30 W to 10 E along the same band at both levels and two sensitivities (the effort's
10 to 14.5 N band, and all 24 retained seasons) follow. Before any of it, a new band-mean
builder must reproduce the stored 700 hPa extended band means, and the new reading path must
reproduce the stored 700 hPa readings on the frozen grid. The readings say what moves and how
fast for the population, not why. An envelope class is not a group velocity, and the
published direction rests on flux and budget terms that are not computed here.

    python3 scripts/eastern_phase_envelope_matched.py gate --xband DIR --stored PATH
    python3 scripts/eastern_phase_envelope_matched.py read --xband DIR --stored PATH --out-dir DIR
"""

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import eastern_anomaly_minima as A  # noqa: E402
import eastern_lag_regression as LR  # noqa: E402
import eastern_phase_envelope as PE  # noqa: E402
import pilot_eastern_propagation as P  # noqa: E402

SCRIPT_SHA256_AT_LOAD = PE.sha256_of(os.path.abspath(__file__))
FROZEN_SHA256 = "49469e14779b69b27bf6861c0e652f34f754ceb09470f3816202d10dd83f9b6a"   # the frozen script at 287e56f (section 2)

MSPAN = (-45.0, 35.0)
MLONS = np.arange(MSPAN[0], MSPAN[1] + 1e-9, 0.5)
MATCHED_BAND = "8 to 15 N"
RECORD_BAND = "10 to 14.5 N"
BANDS = {MATCHED_BAND: (8.0, 15.0, 15), RECORD_BAND: (10.0, 14.5, 10)}      # south and north edges, half-degree rows
ALL_YEARS = tuple(PE.YEARS)
SHARED_YEARS = tuple(y for y in ALL_YEARS if 1990 <= y <= 2010)           # the paper's 1990 to 2010 within the retained years
LEVELS = (850, 700)
TRANSECT = (-30.0, -20.0, -15.0, -10.0, 0.0, 10.0)
PRIMARY = ((850, -15.0), (700, -30.0))
PUBLISHED_LAT = 11.5
PUBLISHED = {850: {"base": -15.0, "maxima_ms": 3.3, "phase_ms": -10.4, "figure": "Fig. 2"},     # the drift of the regressed wind maxima,
             700: {"base": -30.0, "maxima_ms": 3.6, "phase_ms": -10.1, "figure": "Fig. 3"}}     # which the paper reads as its group velocity
READINGS = (("primary and transect", MATCHED_BAND, SHARED_YEARS),
            ("band sensitivity", RECORD_BAND, SHARED_YEARS),
            ("season sensitivity", MATCHED_BAND, ALL_YEARS))
MATCH_TOL = 1e-12


def check_frozen(digest=None):
    """Gate 1. The imported frozen script is the declared one."""
    got = PE.SCRIPT_SHA256_AT_LOAD if digest is None else digest
    if got != FROZEN_SHA256:
        raise SystemExit(f"REFUSED: the frozen script's digest is {got}, not the declared {FROZEN_SHA256}")
    return True


def published_deg_per_day(ms, lat=PUBLISHED_LAT):
    """A published speed in m/s as degrees of longitude per day at `lat`, on the sphere of
    LR.speed_ms (111,195 m per degree at the equator)."""
    return ms * 86400.0 / (111195.0 * math.cos(math.radians(lat)))


# ------------------------------------------------------------------ band means

def band_means(year, level, band, span=MSPAN):
    """One season's band means of the meridional wind at `level` over `band` and the
    longitudes of `span` (inclusive, half a degree), by the effort's band-mean rule (a cell
    valid where the surface pressure exceeds the level strictly, a sample kept where at least
    80 percent of the band's rows are valid), stored to two decimals as (samples,
    longitudes). Refused for the held-out season, a band with the wrong number of rows, a grid
    other than the span, a season of the wrong length, or any missing sample (gate 4)."""
    import netCDF4
    if year == P.HELD_OUT:
        raise SystemExit("REFUSED: the held-out season is never opened")
    south, north, n_rows = BANDS[band]
    d = netCDF4.Dataset(os.path.join(P.DATA_DIR, f"era5_v{level}_{year}_6h_region.nc"))
    try:
        lat = np.asarray(d.variables["latitude"][:], float); lon = np.asarray(d.variables["longitude"][:], float)
    finally:
        d.close()
    rows = np.flatnonzero((lat >= south - 1e-6) & (lat <= north + 1e-6))
    if rows.size != n_rows or np.ptp(rows) != n_rows - 1:
        raise SystemExit(f"REFUSED: band {band} holds {rows.size} rows, not {n_rows} contiguous ones")
    cols = np.flatnonzero((lon >= span[0] - 1e-6) & (lon <= span[1] + 1e-6))
    want = np.arange(span[0], span[1] + 1e-9, 0.5)
    blon = lon[cols.min():cols.max() + 1]
    if blon.shape != want.shape or not np.allclose(blon, want):
        raise SystemExit(f"REFUSED: the longitudes are not {span[0]:g} to {span[1]:g} at half a degree")
    blk = P.read_season(year, level, rows, cols)
    series, _ = P.band_series(blk["v"], blk["sp"], level)
    S = np.round(series, 2)
    if S.shape != (A.SEASON_DAYS * PE.OBS_PER_DAY, want.size):
        raise SystemExit(f"REFUSED: {year} {band} at {level} hPa has shape {S.shape}")
    if not np.isfinite(S).all():
        raise SystemExit(f"REFUSED: {year} {band} at {level} hPa holds a missing sample")
    return S


def input_files(years, levels=LEVELS):
    return [os.path.join(P.DATA_DIR, f"era5_{v}_{y}_6h_region.nc") for y in years for v in [f"v{lv}" for lv in levels] + ["sp"]]


# ------------------------------------------------------------------ readings

def results(seasons, bases, lons, n_boot=PE.N_BOOT):
    """The frozen readings at each base from per-season band means on the grid `lons`, as the
    frozen summary forms them (one set of resample weights for all bases)."""
    Zs = [PE.analytic(S) for S in seasons]
    W = PE.boot_weights(len(Zs), n_boot)
    out = {}
    for b in bases:
        col = int(np.flatnonzero(np.isclose(lons, b))[0])
        st = PE.stack([PE.season_sums(Z, col) for Z in Zs])
        out[f"{b:g}"] = PE.base_result(st, col, W, lons)
    return out


def _same(a, b, path=""):
    """Differences between two JSON-like values, numbers within MATCH_TOL, all else exact."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: keys {sorted(set(a) ^ set(b))}"]
        return [d for k in a for d in _same(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} against {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _same(x, y, f"{path}[{i}]")]
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str) or isinstance(b, str):
        return [] if a == b else [f"{path}: {a!r} against {b!r}"]
    return [] if abs(float(a) - float(b)) <= MATCH_TOL else [f"{path}: {a!r} against {b!r}"]


def check_builder(by, years=ALL_YEARS):
    """Gate 2. The builder reproduces the stored 700 hPa extended band means for 10 to 14.5 N
    on 10 W to 75 E exactly in every season, and returns them."""
    out = []
    for y in years:
        got = band_means(y, 700, RECORD_BAND, span=(float(PE.LONS[0]), float(PE.LONS[-1])))
        want = LR.season_samples(by[y], RECORD_BAND)
        if not np.array_equal(got, want):
            raise SystemExit(f"REFUSED: the band-mean builder does not reproduce the stored 700 hPa record for {y}")
        out.append(got)
    return out


def check_reproduction(seasons, stored_path, n_boot=PE.N_BOOT):
    """Gate 3. The new reading path on the frozen grid reproduces the stored 700 hPa readings
    for 10 to 14.5 N at every base, classes and generated readings identically and numbers
    within MATCH_TOL."""
    with open(stored_path, encoding="utf-8") as fh:
        stored = json.load(fh)
    got = json.loads(json.dumps(results(seasons, PE.BASES, PE.LONS, n_boot)))
    diffs = _same(got, stored["results"][RECORD_BAND], RECORD_BAND)
    texts = {b: PE.reading_text(r) for b, r in got.items()}
    diffs += [f"reading {b}" for b in texts if texts[b] != stored["readings"][RECORD_BAND].get(b)]
    if diffs:
        raise SystemExit("REFUSED: the reading path does not reproduce the stored 700 hPa readings, " + "; ".join(diffs[:5]))
    return True


def direction_agreement(envelope_class):
    """Section 4, rule 2, at one primary configuration."""
    return {"eastward": "agrees in direction", "westward": "disagrees in direction",
            "standing": "disagrees in direction"}.get(envelope_class, "decides nothing")


def outcome(primary_classes):
    """Section 5's row for the envelope classes at the two primary configurations."""
    c = list(primary_classes)
    if len(c) != len(PRIMARY):
        raise SystemExit("REFUSED: the outcome needs both primary configurations")
    if all(x == "eastward" for x in c):
        return "both eastward"
    if all(x in ("westward", "standing") for x in c):
        return "both westward, both standing, or one of each"
    return "one eastward and the other not, or either unresolved"


def dependence(res):
    """Section 4, rule 4. For each primary configuration, the sensitivities under which its
    envelope or phase class changes."""
    out = {}
    for level, base in PRIMARY:
        key, p = f"{level}", res["primary and transect"][f"{level}"][f"{base:g}"]
        out[f"{level} hPa, {base:g} E"] = [name for name in ("band sensitivity", "season sensitivity")
                                          if (res[name][key][f"{base:g}"]["envelope_class"], res[name][key][f"{base:g}"]["phase_class"])
                                          != (p["envelope_class"], p["phase_class"])]
    return out


def summary(res):
    prim = [res["primary and transect"][f"{lv}"][f"{b:g}"]["envelope_class"] for lv, b in PRIMARY]
    return {"primary_envelope_classes": {f"{lv} hPa, {b:g} E": c for (lv, b), c in zip(PRIMARY, prim)},
            "direction": {f"{lv} hPa, {b:g} E": direction_agreement(c) for (lv, b), c in zip(PRIMARY, prim)},
            "outcome": outcome(prim), "dependence": dependence(res),
            "published_deg_per_day": {f"{lv} hPa, {v['base']:g} E": {"maxima": published_deg_per_day(v["maxima_ms"]), "phase": published_deg_per_day(v["phase_ms"])}
                                      for lv, v in PUBLISHED.items()}}


# ------------------------------------------------------------------ text, figure and main

def matched_text(res, summ):
    L = ["# phase and envelope at the published bases of Diaz and Aiyyer (2013)", "",
         "The envelope classes at the two primary configurations are "
         + " and ".join(f"{v} at {k} ({summ['direction'][k]})" for k, v in summ["primary_envelope_classes"].items())
         + f", so the outcome row of section 5 is {summ['outcome']}.", "",
         "The paper's figures give, at 11.5 N, "
         + " and ".join(f"regressed wind maxima moving at {v['maxima']:+.2f} and troughs at {v['phase']:+.2f} deg/day at {k}"
                        for k, v in summ["published_deg_per_day"].items()) + ".", "",
         "Under the sensitivities the class changes at "
         + " and ".join(f"{k} in {', '.join(v) if v else 'neither'}" for k, v in summ["dependence"].items()) + ".", ""]
    for name, band, years in READINGS:
        L += [f"## {name}, {band}, {len(years)} seasons", "", PE.HEAD]
        for level in LEVELS:
            L += PE.rows(f"{level} hPa", res[name][f"{level}"])
        L.append("")
    L += ["## generated readings", ""]
    for name, band, years in READINGS:
        for level in LEVELS:
            for b, r in res[name][f"{level}"].items():
                L.append(f"- {name}, {band}, {level} hPa, {b} E, {PE.reading_text(r)}")
    return "\n".join(L) + "\n"


def draw(path, seasons, res):
    """The transect's covariance maps at both levels, as the paper's Figs. 2 and 3 show theirs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(LEVELS), len(TRANSECT), figsize=(3.0 * len(TRANSECT), 2.9 * len(LEVELS)), sharex=True, sharey=True, constrained_layout=True)
    lagd = PE.LAGS * PE.DT_DAYS
    zero = int(np.flatnonzero(PE.LAGS == 0)[0])
    for r_i, level in enumerate(LEVELS):
        Zs = [PE.analytic(S) for S in seasons[level]]
        for c_i, b in enumerate(TRANSECT):
            col = int(np.flatnonzero(np.isclose(MLONS, b))[0])
            C, E, N = PE.maps(PE.stack([PE.season_sums(Z, col) for Z in Zs]))
            ax = axes[r_i, c_i]
            v0 = float(np.abs(C[zero, col]))
            cs = ax.contourf(MLONS, lagd, C.real / v0, levels=np.arange(-1.0, 1.05, 0.1), cmap="RdBu_r", extend="both")
            ax.contour(MLONS, lagd, N, levels=[PE.N_MIN], colors="k", linewidths=0.6)
            rr = res["primary and transect"][f"{level}"][f"{b:g}"]
            ax.plot(rr["positions"], np.array(rr["lags"]) * PE.DT_DAYS, "k.", ms=2.5)
            if rr["phase_speed"] is not None and rr["positions"]:
                x0 = rr["positions"][rr["lags"].index(0)]
                ax.plot([x0 - 4 * rr["phase_speed"], x0 + 4 * rr["phase_speed"]], [-4, 4], color="m", lw=0.8)
            prim = (level, b) in PRIMARY
            ax.set_title(f"{level} hPa, base {b:g} E{' (published)' if prim else ''}", fontsize=8, fontweight="bold" if prim else "normal")
            ax.set_xlim(MLONS[0], MLONS[-1])
            if c_i == 0:
                ax.set_ylabel("lag (days)")
            if r_i == len(LEVELS) - 1:
                ax.set_xlabel("longitude (deg E)")
    fig.colorbar(cs, ax=axes, shrink=0.6, label="real part of the lagged covariance over the base variance")
    fig.suptitle(f"{MATCHED_BAND}, {len(SHARED_YEARS)} seasons. Black dots: the activity envelope followed. Magenta: the phase speed through it. "
                 f"Black contour: coherence {PE.N_MIN}.", fontsize=8)
    fig.savefig(path, dpi=90); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("gate", "read"))
    ap.add_argument("--xband", required=True)
    ap.add_argument("--stored", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    check_frozen()
    paths = None
    if a.mode == "read":
        if not a.out_dir:
            raise SystemExit("REFUSED: --out-dir is required for the reading")
        paths = {k: os.path.join(a.out_dir, f"eval60_phase_env_matched{s}{a.tag}.{e}") for k, s, e in
                 (("record", "", "json"), ("tables", "_tables", "md"), ("maps", "_maps", "png"))}
        PE.refuse_existing(paths.values())
    files, by = PE.load_xband(a.xband)
    record_seasons = check_builder(by)
    check_reproduction(record_seasons, a.stored)
    built = {(level, band): {y: band_means(y, level, band) for y in ALL_YEARS} for level in LEVELS for band in BANDS}
    print("gates 1 to 4 passed")
    if a.mode == "gate":
        return 0
    res = {name: {f"{level}": results([built[(level, band)][y] for y in years], TRANSECT, MLONS) for level in LEVELS}
           for name, band, years in READINGS}
    res = json.loads(json.dumps(res))
    summ = summary(res)
    draw(paths["maps"], {level: [built[(level, MATCHED_BAND)][y] for y in SHARED_YEARS] for level in LEVELS}, res)
    gen = {"script": "scripts/eastern_phase_envelope_matched.py", "sha256": SCRIPT_SHA256_AT_LOAD,
           "declared": "EASTERN_PHASE_ENVELOPE_MATCHED_2026-10-27.md", "frozen_script": "scripts/eastern_phase_envelope.py",
           "frozen_sha256": PE.SCRIPT_SHA256_AT_LOAD, "lag_regression_sha256": LR.SCRIPT_SHA256_AT_LOAD,
           "filtering_sha256": PE.sha256_of(os.path.join(os.path.dirname(HERE), "src", "aew", "filtering.py")),
           "grid": [MSPAN[0], MSPAN[1], 0.5], "bands": {k: list(v) for k, v in BANDS.items()},
           "years": {"shared": list(SHARED_YEARS), "all": list(ALL_YEARS)},
           "inputs": {os.path.basename(p): PE.sha256_of(p) for p in input_files(ALL_YEARS)},
           "xband": {os.path.basename(p): PE.sha256_of(p) for p in files},
           "stored": {"path": a.stored, "sha256": PE.sha256_of(a.stored)},
           "gates": "1 frozen digest, 2 builder against the stored band means, 3 reading path against the stored readings, 4 complete seasons"}
    with open(paths["record"], "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "results": res, "summary": summ,
                   "readings": {name: {lv: {b: PE.reading_text(r) for b, r in v.items()} for lv, v in res[name].items()} for name in res},
                   "figure": {"path": paths["maps"], "sha256": PE.sha256_of(paths["maps"])}}, fh)
    text = matched_text(res, summ)
    with open(paths["tables"], "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
