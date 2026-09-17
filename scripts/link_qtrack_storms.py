#!/usr/bin/env python3
"""Link every QTrack storm tag in the Atlantic-side population to IBTrACS, and tabulate
the wave track's basin code at the genesis event against the best track's subbasin at
that event, from VALIDATED links only.

WHAT IT WRITES. Per year and per distinct storm key, the link in one of the states the
module keeps apart (matched, ambiguous, rejected_distance, outside_window, no_candidate,
shared_identifier), with the candidate's identifier, its genesis event and first
observation, the offset, the wave-to-event distance, and the wave observation at
genesis or the reason it is unavailable. Then the cross-tab of the wave track's basin
code against the best track's subbasin AT THE GENESIS EVENT, over matched links whose
wave observation is available and nothing else; every other state is counted
separately. And an exception table listing every key that is not a validated match,
with its reason, so the residual is inspectable and a later eastern Pacific catalog can
extend the candidates without hiding earlier contradictions.

WHAT IT DOES NOT ESTABLISH. The identity of tags with no candidate in the North
Atlantic file, or of ambiguous, distance-rejected and shared-identifier links; the
authors' basin definitions (the cross-tab is consistency evidence for the inferred key,
and a storm centre and its wave centre need not share a subbasin); and anything about
the archive's producing revision.

    .venv/bin/python scripts/link_qtrack_storms.py \\
        --tracks data/aewc_v2_pilot/ERA5_WITH_EPAC \\
        --ibtracs data/aewc_v2_pilot/ibtracs.NA.list.v04r01.csv --out <artifact.json>
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from aew import qtrack as Q  # noqa: E402
from aew import qtrack_storms as QS  # noqa: E402

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


def decode_genesis_list(values, units):
    """Genesis datetimes decoded through the declared units, None for the sentinels."""
    import datetime as dt

    out = []
    head = str(units).split(" since ")[0].strip().lower()
    for v in np.asarray(values, dtype=float):
        if v != v or not np.isfinite(v) or v <= 0:
            out.append(None)
            continue
        if head in ("nanoseconds", "nanosecond", "ns"):
            g = nc.num2date(v / 1e9, "seconds since " + str(units).split(" since ", 1)[1],
                            only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        else:
            g = nc.num2date(v, str(units), only_use_cftime_datetimes=False,
                            only_use_python_datetimes=True)
        out.append(g.replace(tzinfo=None) if isinstance(g, dt.datetime) else g)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--tracks", default="data/aewc_v2_pilot/ERA5_WITH_EPAC")
    ap.add_argument("--ibtracs", default="data/aewc_v2_pilot/ibtracs.NA.list.v04r01.csv")
    ap.add_argument("--max-offset-hours", type=float, default=QS.DEFAULT_MAX_OFFSET_HOURS)
    ap.add_argument("--max-distance-km", type=float, default=QS.DEFAULT_MAX_DISTANCE_KM)
    ap.add_argument("--max-step-hours", type=float, default=QS.DEFAULT_MAX_STEP_HOURS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    files = sorted(glob.glob(os.path.join(args.tracks, "*.nc")))
    if not files:
        print(f"REFUSED: no .nc files under {args.tracks}", flush=True)
        return 2
    if not os.path.exists(args.ibtracs):
        print(f"REFUSED: no IBTrACS file at {args.ibtracs}", flush=True)
        return 2
    events = QS.storm_events(QS.load_ibtracs_records(args.ibtracs))
    years, status_counts = {}, collections.Counter()
    wave_states = collections.Counter()
    crosstab = collections.defaultdict(collections.Counter)
    exceptions = []
    exact_offsets, validated = 0, 0
    for path in files:
        year = os.path.basename(path)[-7:-3]
        d = nc.Dataset(path)
        tv = d["time"]
        times = nc.num2date(tv[:], tv.units, only_use_cftime_datetimes=False,
                            only_use_python_datetimes=True)
        gen_units = getattr(d["TC_gen_time"], "units", None)
        if gen_units is None:
            print(f"REFUSED: {os.path.basename(path)} declares no units on TC_gen_time",
                  flush=True)
            d.close()
            return 2
        d.close()
        data = Q.read_year(path)
        r = Q.filter_year(data, repeat_policy="keep")
        genesis = decode_genesis_list(data["gen_time"], gen_units)
        rows = QS.link_year(data, times, genesis, events, r["keep"], args.max_offset_hours,
                            args.max_distance_km, args.max_step_hours)
        for row in rows:
            status_counts[row["status"]] += 1
            wave = row["wave_at_genesis"]
            wave_states["available" if wave["code"] is not None else wave["reason"]] += 1
            if row["status"] == "matched" and wave["code"] is not None:
                crosstab[row["event_subbasin"]][str(wave["code"])] += 1
                validated += 1
                if row["offset_hours"] == 0.0:
                    exact_offsets += 1
            else:
                exceptions.append({"year": year, "name": row["name"],
                                   "qtrack_genesis": row["qtrack_genesis"],
                                   "systems": row["systems"], "status": row["status"],
                                   "reason": row.get("reason", row.get("wave_at_genesis", {}).get("reason")),
                                   "nearest_sid": row.get("nearest_sid"),
                                   "offset_hours": row.get("offset_hours"),
                                   "wave_to_event_km": row.get("wave_to_event_km"),
                                   "wave_code": wave["code"]})
        years[year] = {"storm_keys": len(rows), "rows": rows, "input_sha256": _sha256(path)}
        print(f"  {year}: {len(rows):3d} keys, "
              f"{sum(1 for x in rows if x['status'] == 'matched'):3d} matched, "
              f"{sum(1 for x in rows if x['status'] not in ('matched', 'no_candidate')):2d} "
              f"other, {sum(1 for x in rows if x['status'] == 'no_candidate'):3d} no candidate",
              flush=True)
    out = {"generated_by": "scripts/link_qtrack_storms.py",
           "rule": {"genesis_event": "first best-track record whose USA_STATUS is not DB, LO "
                                     "or WV, the installed QTrack's own exclusion; the first "
                                     "observation of any stage is kept as a separate field",
                    "candidates": "IBTrACS main-track storms of the same SEASON whose NAME is "
                                  "among candidate_names(tag), with a genesis event",
                    "choice": "nearest genesis event to the QTrack genesis time",
                    "max_offset_hours": args.max_offset_hours,
                    "max_distance_km": args.max_distance_km,
                    "distance_rule": "QTrack's own genesis-association distance "
                                     "(TC_merge_dist default), between the wave track's "
                                     "position at the genesis step and the event position",
                    "max_step_hours": args.max_step_hours,
                    "states": ["matched", "ambiguous", "rejected_distance", "outside_window",
                               "no_candidate", "shared_identifier"],
                    "crosstab_population": "matched links whose wave observation at genesis "
                                           "is available; every other state counted separately",
                    "event_subbasin": "IBTrACS SUBBASIN at the genesis event",
                    "wave_code": "basin_des at the valid step nearest the genesis time on the "
                                 "first listed track, under the inferred key, or unavailable"},
           "status_counts": dict(status_counts),
           "wave_observation_states": dict(wave_states),
           "validated_links": validated,
           "validated_links_with_zero_offset": exact_offsets,
           "crosstab_event_subbasin_vs_wave_code": {k: dict(v) for k, v in crosstab.items()},
           "exceptions": exceptions,
           "years": years,
           "ibtracs_sha256": _sha256(args.ibtracs),
           "source_sha256": {"src/aew/qtrack_storms.py": _sha256(os.path.join(
               os.path.dirname(os.path.abspath(__file__)), "..", "src", "aew", "qtrack_storms.py")),
               "src/aew/qtrack.py": _sha256(os.path.join(
                   os.path.dirname(os.path.abspath(__file__)), "..", "src", "aew", "qtrack.py")),
               "scripts/link_qtrack_storms.py": _sha256(os.path.abspath(__file__))},
           "what_this_does_not_establish": "the identity of keys in any state but matched; "
                                           "the authors' basin definitions; the archive's "
                                           "producing revision"}
    print("status:", dict(status_counts))
    print("wave observation:", dict(wave_states))
    print(f"validated links {validated}, of which zero offset {exact_offsets}")
    print("crosstab (event subbasin -> wave code at genesis, validated links only):")
    for k, v in crosstab.items():
        print(f"  {k:6s} {dict(v)}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=True, default=str)
        print(f"written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
