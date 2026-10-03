#!/usr/bin/env python3
"""Before any tracking batch, run real detection on a short window and require the
retained answer.

The protocol entry point's own functions are called on one window of a reference year:
input validation, the inputs stage, and the tracking stage, which validates every
climatology input, checks the calibration, loads the climatology cache, builds the anomaly
and advection fields, runs detection (contouring with contourpy) and association, and
writes a record. The window's tracks, compared exactly, must equal the committed
reference, and so must the manifest, calibration, climatology-input and year-input
digests. The record must name a commit with a checked CLEAN tree, and its tracking source,
every module of the port and the entry point by digest, must be the reference's. The
commit itself may differ, since a commit that changes no tracking source changes nothing
this compares, but a changed tracker is never passed on one matching window: it needs a
new reference, written deliberately from its own clean commit.

WHY. The suite stands in for the tracker, so it cannot see a missing contouring library or
a git too old to record provenance, and a batch on another machine met both on 2026-10-03:
the first run stopped at tracking after 17 minutes, and records written without a working
git called an unchecked tree clean. This takes about a minute and a half, most of it
digesting the 64 climatology inputs.

WHAT IT DOES NOT SHOW. One window of one year agreeing does not establish that every year
will. The 2026-10-03 extension also tracked 2010 in full beside its batch and compared it
with the retained record; this preflight is a gate before launch, not that check.

    python3 scripts/preflight_tracking.py --climo-cache <the batch's cache .npz>
    python3 scripts/preflight_tracking.py --climo-cache <npz> --write-reference <new json>   # once, on the reference machine
"""
import argparse
import datetime as dt
import importlib
import json
import os
import platform
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402
import export_protocol_case as E  # noqa: E402
import season_metrics as S  # noqa: E402

REFERENCE = "docs/aewc_v2/protocol/preflight_reference_era5_2010_08_01_24steps_v2_2026-10-03.json"
BASE_MANIFEST = "docs/aewc_v2/protocol/manifest_2026-09-25.json"
MODULES = ("numpy", "scipy", "contourpy", "netCDF4")
TIME_MAJOR = ("times", "u", "v", "curvature")


def environment():
    """The versions of the libraries the tracking chain imports, or a refusal naming the
    one that will not import."""
    out = {"python": platform.python_version(), "machine": platform.machine(), "system": platform.system()}
    for name in MODULES:
        try:
            out[name] = importlib.import_module(name).__version__
        except ImportError as exc:
            raise SystemExit(f"REFUSED: {name} does not import here ({exc}), and the tracker needs it")
    return out


def window_start(year, month, day):
    return (dt.date(year, month, day) - dt.date(year, 1, 1)).days * 4


def sliced(arrays, start, steps):
    """The inputs stage's arrays cut to one window along time, everything else unchanged."""
    out = dict(arrays)
    if not 0 <= start < start + steps <= arrays["times"].size:
        raise SystemExit(f"REFUSED: the window {start} to {start + steps} is not inside the year's {arrays['times'].size} steps")
    for key in TIME_MAJOR:
        out[key] = arrays[key][start:start + steps]
    out["coarse"] = {k: v[start:start + steps] for k, v in arrays["coarse"].items()}
    return out


def run_window(manifest_path, dataset_name, year, start, steps, climo_cache, calibration):
    """The window's tracks in canonical form and the record of the run that made them."""
    manifest, manifest_sha256 = E.load_manifest(manifest_path)
    dataset = manifest["datasets"][dataset_name]
    preflight, inputs_sha256 = E.validate_inputs(manifest, dataset, [year])
    readiness, arrays = E.inputs_stage(manifest, dataset, year, preflight)
    with tempfile.TemporaryDirectory(prefix="preflight-") as run_dir:
        record, _n = E.tracking_stage(manifest, manifest_sha256, dataset_name, dataset, year,
                                      sliced(arrays, start, steps), climo_cache, calibration, run_dir,
                                      inputs_sha256, readiness)
        with open(os.path.join(run_dir, "tracker_port.mat"), "rb") as fh:
            tracks, _case = S.read_mat_tracks(fh.read())
    canon = X.canonical([{"time": t["time"], "meanlat": t["lat"], "meanlon": t["lon"]} for t in tracks])
    return [[list(t), list(la), list(lo)] for t, la, lo in canon], record


def bindings(record):
    d = record["dataset_specific"]
    return {"manifest_sha256": record["protocol_settings"]["manifest_sha256"], "calibration_sha256": d["calibration_sha256"],
            "climatology_inputs_sha256": d["climatology_inputs_sha256"], "inputs_sha256": d["inputs_sha256"],
            "source_sha256": record.get("source_sha256")}


def problems(tracks, record, reference, window):
    """Why this machine's window is not the reference's, or an empty list."""
    out = []
    head, dirty = record.get("git_head"), record.get("git_dirty")
    if not (isinstance(head, str) and re.fullmatch(r"[0-9a-f]{40}", head)) or not isinstance(dirty, bool):
        out.append(f"git cannot record provenance here (head {head!r}, tree state {dirty!r})")
    elif dirty:
        out.append(f"the tree at {head[:12]} has uncommitted tracking source, so the record names no commit that holds it")
    if reference.get("window") != window:
        out.append(f"the window {window} is not the reference's {reference.get('window')}")
    mine = bindings(record)
    for key in mine:
        if mine[key] != reference.get(key):
            out.append(f"{key} is not the reference's, so this is not the reference run"
                       + (". A changed tracker needs a new reference, written from its own clean commit" if key == "source_sha256" else ""))
    if len(tracks) != len(reference.get("tracks", [])):
        out.append(f"{len(tracks)} tracks where the reference has {len(reference.get('tracks', []))}")
    elif tracks != reference["tracks"]:
        moved = sum(a != b for a, b in zip(tracks, reference["tracks"]))
        out.append(f"{moved} of {len(tracks)} tracks differ from the reference in time, latitude or longitude")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--climo-cache", required=True, help="the climatology cache the batch will use")
    ap.add_argument("--reference", default=REFERENCE)
    ap.add_argument("--write-reference", default=None, help="publish this machine's window as a new reference, never over a file")
    ap.add_argument("--manifest", default=BASE_MANIFEST)
    ap.add_argument("--dataset", default="era5")
    ap.add_argument("--year", type=int, default=2010)
    ap.add_argument("--start", type=int, default=window_start(2010, 8, 1), help="first timestep index of the window")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--calibration", default="docs/aewc_v2/artifacts/thresholds_protocol_era5_1979_2010.json")
    args = ap.parse_args(argv)
    env = environment()
    window = {"dataset": args.dataset, "year": args.year, "start": args.start, "steps": args.steps}
    tracks, record = run_window(args.manifest, args.dataset, args.year, args.start, args.steps, args.climo_cache, args.calibration)
    if args.write_reference:
        if os.path.exists(args.write_reference):
            raise SystemExit(f"REFUSED: {args.write_reference} exists and a reference is never overwritten")
        head, dirty = record.get("git_head"), record.get("git_dirty")
        if not (isinstance(head, str) and re.fullmatch(r"[0-9a-f]{40}", head)) or dirty is not False:
            raise SystemExit("REFUSED: a reference is written only from a named commit with a checked clean tree")
        X.publish_json(args.write_reference, {"generated_by": "scripts/preflight_tracking.py", "window": window,
                                              **bindings(record), "git_head": head, "environment": env,
                                              "tracks": tracks}, exclusive=True)
        print(f"wrote {args.write_reference}: {len(tracks)} tracks in the window")
        return 0
    with open(args.reference) as fh:
        reference = json.load(fh)
    found = problems(tracks, record, reference, window)
    print(f"environment {env}")
    print(f"reference made on {reference.get('environment')}")
    for line in found:
        print("PROBLEM:", line)
    print(f"{len(tracks)} tracks in the window; " + ("PREFLIGHT PASSED" if not found else "PREFLIGHT REFUSED"))
    return 0 if not found else 1


if __name__ == "__main__":
    raise SystemExit(main())
