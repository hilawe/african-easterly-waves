#!/usr/bin/env python3
"""Export a window of raw fields so version 1's whole tracker can run on them.

find_ews_f.m does its own smoothing, masking, contouring, merging, association and
filtering from RAW coarse and fine fields, which is exactly what the port's
`pipeline.track_year` has in hand. So the two can be handed identical input and their
finished tracks compared, which covers the association and prune stages that no oracle
has ever reached.

A CONTIGUOUS WINDOW, because association carries state across timesteps. Tracks that
began before the window are absent from both sides equally, and version 1's own lifetime
and speed filters only engage once the run is at least two days long, so the window has
to be comfortably longer than that.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
from scipy.io import savemat

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from aew.v1port import climatology as clim  # noqa: E402
from aew.v1port import load as L  # noqa: E402
from aew.v1port import pipeline as P  # noqa: E402
from aew.v1port.climo_cache import load_or_build  # noqa: E402


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def port_producer_record(case, args, climo_cache):
    """What produced the port's tracks: the source that executed, hashed now, the
    checkout it came from and whether it was dirty, the settings, and the climatology
    cache. Written into the output so the record travels with the tracks."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(repo, "src", "aew")
    sources = {}
    for root, _dirs, files in os.walk(src_dir):
        for name in sorted(files):
            if name.endswith(".py"):
                full = os.path.join(root, name)
                sources[os.path.relpath(full, repo)] = _sha256(full)
    sources["scripts/export_tracker_case.py"] = _sha256(os.path.abspath(__file__))

    def git(*argv):
        try:
            return subprocess.run(["git", "-C", repo, *argv], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--", "src", "scripts")
    return {"producer": "scripts/export_tracker_case.py", "case_id": case,
            "git_head": head, "git_dirty": None if dirty is None else bool(dirty),
            "source_sha256": sources,
            "settings": {"year": args.year, "start": args.start, "steps": args.steps,
                         "exclusive": bool(args.exclusive), "absorb": False,
                         "directory": os.path.basename(os.path.normpath(args.directory))},
            "climo_cache_sha256": _sha256(climo_cache) if os.path.exists(climo_cache) else None}


def case_id(payload):
    """A digest of the exported fields, stamped into every file of one comparison.

    THE THREE FILES MUST BE THE SAME CASE AND NOTHING ELSE ESTABLISHED THAT. The Octave
    side writes its tracks to a fixed path, so a run that dies part-way leaves the
    PREVIOUS run's output in place, and the comparison would then read version 1's tracks
    for one window against the port's for another and report the disagreement as a
    finding. That is not hypothetical here: two runs of this harness died silently
    mid-window, and a 40-timestep export sat in the same directory as a 60-timestep one.
    """
    h = hashlib.sha256()
    for name in sorted(payload):
        value = payload[name]
        h.update(name.encode())
        if isinstance(value, str):
            h.update(value.encode())
        else:
            array = np.ascontiguousarray(np.asarray(value, dtype=float))
            h.update(repr(array.shape).encode())
            h.update(array.tobytes())
    return h.hexdigest()[:32]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--year", type=int, default=1990)
    ap.add_argument("--start", type=int, default=600, help="first timestep index")
    ap.add_argument("--steps", type=int, default=60, help="15 days at six-hourly")
    # THE REPAIRED RUN HAS TO BE REPRODUCIBLE TOO. The wave-level comparison needs the
    # port's output with the duplication repair on, and the first version of that analysis
    # produced it from an ad-hoc command, so its central row could not be regenerated from
    # anything committed. It writes a separate file and records the setting inside it, so
    # a reader can tell which run they have rather than trusting a filename.
    ap.add_argument("--exclusive", action="store_true",
                    help="run the port with the duplication repair on, writing "
                         "tracker_port_exclusive.mat instead")
    # ANCHORED TO THE REPOSITORY, not the working directory, so invoking this by absolute
    # path from elsewhere finds the data instead of failing on a relative default.
    ap.add_argument("--directory", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "eraint", "v1port_buffered"))
    args = ap.parse_args(argv)

    # REJECT A NONSENSE WINDOW BEFORE WRITING ANYTHING, so a bad argument does not leave
    # a truncated or empty case file for the next stage to run on.
    if args.steps < 1 or args.start < 0:
        ap.error("--start must be at least 0 and --steps at least 1")

    out_dir = os.environ["AEW_ORACLE_DIR"]
    # THROUGH THE GUARDED LOADER, not a bare np.load. An earlier version of this script
    # read the cache file directly, so the oracle comparison it feeds rested on a
    # climatology whose provenance nothing checked, while the worked-timestep diagnostic
    # refused that same file for exactly that reason.
    climo = load_or_build(L.available_years(args.directory, "eraint"),
                          args.directory, "eraint", os.environ["AEW_CLIMO_CACHE"])

    times, latgrid, longrid, u, v = L.load_year(args.year, args.directory, "eraint")
    curvature = P.curvature_from_winds(latgrid, longrid, u, v)
    anomaly = P.anomaly_from_climatology(curvature, times, climo)
    del curvature
    advection = P._advection(latgrid, longrid, u, v, anomaly)
    native = abs(float(latgrid[1, 0] - latgrid[0, 0]))
    coarse = {n: clim.gaussian_decimate(f, native, P.COARSE_RESOLUTION_DEG)
              for n, f in (("u", u), ("v", v), ("anomaly", anomaly),
                           ("advection", advection))}
    del advection
    lat_c, lon_c = P.coarse_grid(latgrid, longrid, native, P.COARSE_RESOLUTION_DEG)
    rows_f, cols_f = P._subset(latgrid, longrid, P.DOMAIN_LAT, P.DOMAIN_LON)
    rows_c, cols_c = P._subset(lat_c, lon_c, P.DOMAIN_LAT, P.DOMAIN_LON)
    if args.start + args.steps > times.size:
        ap.error(f"steps {args.start} to {args.start + args.steps - 1} run past the "
                 f"{times.size} timesteps in {args.year}")
    window = slice(args.start, args.start + args.steps)

    fine_lat = latgrid[np.ix_(rows_f, cols_f)]
    fine_lon = longrid[np.ix_(rows_f, cols_f)]
    coarse_lat = lat_c[np.ix_(rows_c, cols_c)]
    coarse_lon = lon_c[np.ix_(rows_c, cols_c)]

    def cut_f(field):
        return field[window][:, rows_f, :][:, :, cols_f]

    def cut_c(field):
        return field[window][:, rows_c, :][:, :, cols_c]

    payload = {
        # find_ews_f applies meshgrid to lat_c/lon_c itself, so they go as VECTORS while
        # the fine grids go as the meshes it expects.
        "lat_c": coarse_lat[:, 0], "lon_c": coarse_lon[0, :],
        "latgrid": fine_lat, "longrid": fine_lon,
        "time": np.asarray(times[window], dtype=float),
        "u_c": cut_c(coarse["u"]), "v_c": cut_c(coarse["v"]),
        "currv_anom_c": cut_c(coarse["anomaly"]),
        "advcurrv_anom_c": cut_c(coarse["advection"]),
        "u": cut_f(u), "v": cut_f(v), "currv_anom": cut_f(anomaly),
        "rean": "ERA-Int", "level": float(700),
    }
    payload["case_id"] = case_id(payload)
    path = os.path.join(out_dir, "tracker_case.mat")
    savemat(path, payload, do_compression=True)
    print(f"wrote {path}")
    print(f"  window: steps {args.start} to {args.start + args.steps - 1}, "
          f"times {times[args.start]:.2f} to {times[args.start + args.steps - 1]:.2f}")
    print(f"  coarse {coarse_lat.shape}, fine {fine_lat.shape}")

    port_name = ("tracker_port_exclusive.mat" if args.exclusive
                 else "tracker_port.mat")
    # REMOVE THE PREVIOUS PORT ANSWER BEFORE COMPUTING A NEW ONE, for the same reason the
    # Octave script does. The case id binds the INPUT, not the code that consumed it, so
    # re-exporting an identical window after a change to the port and then dying before
    # the save would leave a stale file carrying a MATCHING id, which the comparison would
    # accept. Deleting first turns that into an absent file and a loud refusal.
    stale = os.path.join(out_dir, port_name)
    if os.path.exists(stale):
        os.remove(stale)

    # the port's own answer on the identical window, so nothing is recomputed later
    ct, ft = P.thresholds_for("ERA-Int", 700)
    tracks = P.track_year(np.asarray(times[window], dtype=float), latgrid, longrid,
                          u[window], v[window], anomaly[window],
                          coarse_threshold=ct, fine_threshold=ft,
                          exclusive=args.exclusive)
    port = {"n": float(len(tracks)), "case_id": payload["case_id"],
            "exclusive": float(args.exclusive), "absorb": 0.0,
            # PRODUCING PROVENANCE, recorded HERE at run time and carried inside the
            # output, so a comparison can say which port source produced these tracks
            # rather than reading the checkout it happens to run in. A review reused
            # unchanged exchange files and watched the recorded head change with the
            # comparison's own checkout, which proved the earlier field named nothing.
            "producer_json": json.dumps(port_producer_record(
                payload["case_id"], args, os.environ["AEW_CLIMO_CACHE"]), sort_keys=True)}
    for i, t in enumerate(tracks):
        port[f"lat{i}"] = np.asarray(t["meanlat"], dtype=float)
        port[f"lon{i}"] = np.asarray(t["meanlon"], dtype=float)
        port[f"time{i}"] = np.asarray(t["time"], dtype=float)
    savemat(os.path.join(out_dir, port_name), port, do_compression=True)
    print(f"  the port produces {len(tracks)} tracks on this window "
          f"(exclusive={args.exclusive}, absorb=False)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
