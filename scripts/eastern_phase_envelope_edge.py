#!/usr/bin/env python3
"""The western grid edge in the phase and envelope effort's 700 hPa readings
(`EASTERN_PHASE_ENVELOPE_MATCHED_2026-10-27.md`, section 10). A diagnostic found after the
matched reading and not declared before it. The effort's grid starts at 10 W, so as an
envelope followed from 0 or 10 E moves west its 15 degree window is cut at the grid's first
column. This script reads the effort's six western readings (700 hPa, bases 0 and 10 E, the
three southern bands, 24 seasons) twice with the frozen estimator. First it reads them on the
frozen grid, where they must reproduce the stored readings exactly. Then it reads them on
45 W to 35 E, where no window is cut. Only the grid differs between the two, so the
difference is the edge's effect on these readings. It says nothing about the eastern bases,
whose edge at 75 E is the end of the retained files.

    python3 scripts/eastern_phase_envelope_edge.py --xband DIR --stored PATH --out-dir DIR
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

SCRIPT_SHA256_AT_LOAD = PE.sha256_of(os.path.abspath(__file__))
EDGE_BANDS = {"10 to 14.5 N": (10.0, 14.5, 10), "15 to 19.5 N": (15.0, 19.5, 10), "20 to 24.5 N": (20.0, 24.5, 10)}
EDGE_BASES = (0.0, 10.0)
LEVEL = 700


def season_means(band, span):
    """The band means of every retained season over `span`, by the matched script's builder
    with the effort's two further southern bands added to its band table."""
    M.BANDS = {**M.BANDS, **EDGE_BANDS}
    return [M.band_means(y, LEVEL, band, span=span) for y in M.ALL_YEARS]


def frozen_span():
    return (float(PE.LONS[0]), float(PE.LONS[-1]))


def check_stored(by, band, frozen_seasons, stored, n_boot=PE.N_BOOT):
    """The builder reproduces the stored band means of `band`, and the frozen-grid readings at
    the two western bases reproduce the stored ones, before the wide grid is read."""
    for y, S in zip(M.ALL_YEARS, frozen_seasons):
        if not np.array_equal(S, LR.season_samples(by[y], band)):
            raise SystemExit(f"REFUSED: the builder does not reproduce the stored band means for {y} {band}")
    got = json.loads(json.dumps(M.results(frozen_seasons, EDGE_BASES, PE.LONS, n_boot)))
    diffs = M._same(got, {f"{b:g}": stored[band][f"{b:g}"] for b in EDGE_BASES}, band)
    if diffs:
        raise SystemExit("REFUSED: the frozen-grid readings do not reproduce the stored ones, " + "; ".join(diffs[:5]))
    return got


def compare(frozen, wide):
    """Per base, the two readings' envelope classes, speeds, intervals and westernmost
    positions, and the change in envelope speed from the frozen grid to the wide one."""
    out = {}
    for b in frozen:
        f, w = frozen[b], wide[b]
        out[b] = {grid: {"envelope_class": r["envelope_class"], "envelope_speed": r["envelope_speed"],
                         "envelope_interval": r["envelope_interval"], "phase_speed": r["phase_speed"],
                         "westmost": min(r["positions"]) if r["positions"] else None, "pos_lag_days": r["pos_lag_days"]}
                  for grid, r in (("frozen", f), ("wide", w))}
        out[b]["class_changed"] = f["envelope_class"] != w["envelope_class"]
        out[b]["speed_change"] = None if f["envelope_speed"] is None or w["envelope_speed"] is None else w["envelope_speed"] - f["envelope_speed"]
    return out


def edge_text(res):
    f = lambda v: "" if v is None else f"{v:+.2f}"
    L = ["# the western grid edge in the effort's 700 hPa readings (a diagnostic after the matched reading)", "",
         "| band | base (deg E) | frozen grid, envelope (interval) | westernmost (deg E) | wide grid, envelope (interval) | westernmost (deg E) | class changed | speed change |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for band, v in res.items():
        for b, r in v.items():
            fr, wd = r["frozen"], r["wide"]
            L.append(f"| {band} | {b} | {fr['envelope_class']} {f(fr['envelope_speed'])} ({f(fr['envelope_interval'][0])} to {f(fr['envelope_interval'][1])}) | {f(fr['westmost'])} | "
                     f"{wd['envelope_class']} {f(wd['envelope_speed'])} ({f(wd['envelope_interval'][0])} to {f(wd['envelope_interval'][1])}) | {f(wd['westmost'])} | "
                     f"{'yes' if r['class_changed'] else 'no'} | {f(r['speed_change'])} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xband", required=True)
    ap.add_argument("--stored", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    M.check_frozen()
    pj, pm = (os.path.join(a.out_dir, f"eval60_phase_env_edge.{e}") for e in ("json", "md"))
    PE.refuse_existing([pj, pm])
    files, by = PE.load_xband(a.xband)
    with open(a.stored, encoding="utf-8") as fh:
        stored = json.load(fh)["results"]
    res = {}
    for band in EDGE_BANDS:
        frozen = check_stored(by, band, season_means(band, frozen_span()), stored)
        wide = json.loads(json.dumps(M.results(season_means(band, M.MSPAN), EDGE_BASES, M.MLONS)))
        res[band] = compare(frozen, wide)
    gen = {"script": "scripts/eastern_phase_envelope_edge.py", "sha256": SCRIPT_SHA256_AT_LOAD, "matched_sha256": M.SCRIPT_SHA256_AT_LOAD,
           "frozen_sha256": PE.SCRIPT_SHA256_AT_LOAD, "declared": "EASTERN_PHASE_ENVELOPE_MATCHED_2026-10-27.md, section 10 (after the reading)",
           "stored": {"path": a.stored, "sha256": PE.sha256_of(a.stored)}, "years": list(M.ALL_YEARS), "level": LEVEL,
           "grids": {"frozen": [frozen_span()[0], frozen_span()[1], 0.5], "wide": [M.MSPAN[0], M.MSPAN[1], 0.5]},
           "inputs": {os.path.basename(p): PE.sha256_of(p) for p in M.input_files(M.ALL_YEARS, levels=(LEVEL,))}}
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "results": res}, fh)
    text = edge_text(res)
    with open(pm, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
