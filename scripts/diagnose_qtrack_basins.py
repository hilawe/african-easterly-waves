#!/usr/bin/env python3
"""Derive what QTrack v2's undocumented basin codes mean, from the tracks themselves.

WHY THIS EXISTS. The ERA5_WITH_EPAC files carry `basin_des` (system x time) and
`first_basin_des` (system) with integer codes and no attributes, and the codes are defined
nowhere in the public QTrack repository (main and the four other branches were searched,
and the Zenodo description says only that 2024 includes "the ATL and EPAC basins"). The
codes can still be read off the data: a per-timestep basin field is a set of regions, so
the longitudes at which tracks CHANGE code show the region edges, and the positions of one
code show whether it is a band or a land mask.

WHAT IT MEASURES, per code: the number of track steps, the longitude and latitude
extent (minimum, 1st, 50th and 99th percentiles, maximum), and the number of systems
whose first basin is that code. Per transition between codes: the count and the
longitude and latitude at the step of entry. And a fixed random sample of positions for
the one code (1) whose extent overlaps others, so a reader can place them on a map.

WHAT IT DOES NOT ESTABLISH. The labels. Those are inferred by a person from the numbers
here and recorded in the handoff as inferences, to be confirmed by the dataset's authors.

    .venv/bin/python scripts/diagnose_qtrack_basins.py \\
        --directory data/aewc_v2_pilot/ERA5_WITH_EPAC --out <artifact.json>
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import sys

import numpy as np

try:
    import netCDF4 as nc
except ImportError:                                            # pragma: no cover
    sys.exit("netCDF4 is required")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def summarize(lon, lat):
    q = lambda a, p: float(np.percentile(a, p))  # noqa: E731
    return {"n_steps": int(lon.size),
            "lon": {"min": float(lon.min()), "p1": q(lon, 1), "p50": q(lon, 50),
                    "p99": q(lon, 99), "max": float(lon.max())},
            "lat": {"min": float(lat.min()), "p1": q(lat, 1), "p50": q(lat, 50),
                    "p99": q(lat, 99), "max": float(lat.max())}}


def derive(files, sample_seed=0, sample_size=15):
    """Extents, first-basin counts, transitions and a code-1 sample over the files."""
    lon_by, lat_by = collections.defaultdict(list), collections.defaultdict(list)
    first = collections.Counter()
    trans = collections.defaultdict(list)
    code1_points = []
    for path in files:
        d = nc.Dataset(path)
        bd = np.ma.filled(d["basin_des"][:], np.nan)
        lon = np.ma.filled(d["AEW_lon"][:], np.nan)
        lat = np.ma.filled(d["AEW_lat"][:], np.nan)
        fb = np.ma.filled(d["first_basin_des"][:], np.nan)
        d.close()
        ok = ~np.isnan(bd) & ~np.isnan(lon) & ~np.isnan(lat)
        for c in np.unique(bd[ok]):
            m = ok & (bd == c)
            lon_by[int(c)].append(lon[m])
            lat_by[int(c)].append(lat[m])
        for c in fb[~np.isnan(fb)]:
            first[int(c)] += 1
        for i in range(bd.shape[0]):
            idx = np.where(ok[i])[0]
            for a, b in zip(idx[:-1], idx[1:]):
                if bd[i, a] != bd[i, b]:
                    trans[(int(bd[i, a]), int(bd[i, b]))].append(
                        (float(lon[i, b]), float(lat[i, b])))
        m1 = ok & (bd == 1)
        code1_points.extend(zip(lon[m1].tolist(), lat[m1].tolist()))
    codes = {}
    for c in sorted(lon_by):
        codes[str(c)] = summarize(np.concatenate(lon_by[c]), np.concatenate(lat_by[c]))
        codes[str(c)]["first_basin_systems"] = int(first.get(c, 0))
    transitions = {}
    for (a, b), pts in sorted(trans.items(), key=lambda kv: -len(kv[1])):
        arr = np.array(pts)
        transitions[f"{a}->{b}"] = {
            "count": int(len(pts)),
            "entry_lon": {"p5": float(np.percentile(arr[:, 0], 5)),
                          "p50": float(np.percentile(arr[:, 0], 50)),
                          "p95": float(np.percentile(arr[:, 0], 95))},
            "entry_lat": {"p5": float(np.percentile(arr[:, 1], 5)),
                          "p50": float(np.percentile(arr[:, 1], 50)),
                          "p95": float(np.percentile(arr[:, 1], 95))}}
    rng = np.random.default_rng(sample_seed)
    pts = np.array(code1_points)
    pick = rng.choice(len(pts), size=min(sample_size, len(pts)), replace=False)
    sample = [[round(float(x), 1), round(float(y), 1)] for x, y in pts[pick]]
    return {"codes": codes, "transitions": transitions,
            "code1_sample_lon_lat": sample,
            "code1_sample": {"seed": sample_seed, "generator": "PCG64", "size": len(sample)}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--directory", default="data/aewc_v2_pilot/ERA5_WITH_EPAC")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    files = sorted(glob.glob(os.path.join(args.directory, "*.nc")))
    if not files:
        print(f"REFUSED: no .nc files under {args.directory}", flush=True)
        return 2
    result = derive(files)
    result["generated_by"] = "scripts/diagnose_qtrack_basins.py"
    result["what_this_establishes"] = (
        "The geographic extent of each basin_des code and the positions at which tracks "
        "change code, over every file read. It does not establish the labels, which are "
        "inferred from these numbers and recorded elsewhere as inferences.")
    result["input_file_sha256"] = {os.path.basename(p): _sha256(p) for p in files}
    here = os.path.abspath(__file__)
    result["source_sha256"] = {"scripts/diagnose_qtrack_basins.py": _sha256(here)}
    print("code  steps    lon min   p1    p50    p99    max | lat p1   p50   p99 | first")
    for c, s in result["codes"].items():
        lo, la = s["lon"], s["lat"]
        print(f"{c:>4} {s['n_steps']:7d} {lo['min']:8.1f} {lo['p1']:6.1f} {lo['p50']:6.1f} "
              f"{lo['p99']:6.1f} {lo['max']:6.1f} | {la['p1']:5.1f} {la['p50']:5.1f} "
              f"{la['p99']:5.1f} | {s['first_basin_systems']}")
    print("transition count  entry lon p5/p50/p95")
    for k, t in list(result["transitions"].items())[:12]:
        e = t["entry_lon"]
        print(f"  {k:7s} {t['count']:6d}  {e['p5']:7.1f} {e['p50']:7.1f} {e['p95']:7.1f}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1, sort_keys=True)
        print(f"written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
