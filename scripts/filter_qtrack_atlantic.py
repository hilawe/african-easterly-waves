#!/usr/bin/env python3
"""Write the Atlantic subset of QTrack v2 (ERA5_WITH_EPAC), year by year, removing nothing
for shared positions and MEASURING them instead.

The rules live in `aew.qtrack` and are decisions recorded there: Atlantic means any step
in basin codes 2, 6 or 7 with a first valid code that is not Pacific (an inferred key,
see that module). Tracks sharing at least `--min-shared` identical positions are NOT
removed; the installed QTrack post-processing can impose that shape by copying one
track's coordinates into another, the archive's producing revision is unconfirmed, and
the differing heads are separate detections whose physical relation is not settled. The summary records every such pair per year, the distinct and copied
coordinate-and-time record counts among the kept tracks, the distinct storm keys (name
plus genesis time), and, as sensitivity only, what the retired drop-the-shorter rule
would have removed. Every input and the module are hashed into the summary. Output files
keep the published schema and the original system numbers.

    .venv/bin/python scripts/filter_qtrack_atlantic.py \\
        --directory data/aewc_v2_pilot/ERA5_WITH_EPAC \\
        --out-directory data/aewc_v2_pilot/ERA5_ATLANTIC \\
        --summary <summary.json>
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from aew import qtrack as Q  # noqa: E402

SENSITIVITY_THRESHOLDS = (2, 4, 8)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--directory", default="data/aewc_v2_pilot/ERA5_WITH_EPAC")
    ap.add_argument("--out-directory", default="data/aewc_v2_pilot/ERA5_ATLANTIC")
    ap.add_argument("--min-shared", type=int, default=Q.DEFAULT_MIN_SHARED)
    ap.add_argument("--summary", default=None)
    args = ap.parse_args(argv)
    files = sorted(glob.glob(os.path.join(args.directory, "*.nc")))
    if not files:
        print(f"REFUSED: no .nc files under {args.directory}", flush=True)
        return 2
    os.makedirs(args.out_directory, exist_ok=True)
    years, totals = {}, {}
    for path in files:
        m = re.search(r"(\d{4})\.nc$", os.path.basename(path))
        if not m:
            print(f"REFUSED: cannot read a year from {os.path.basename(path)}", flush=True)
            return 2
        year = m.group(1)
        data = Q.read_year(path)
        r = Q.filter_year(data, args.min_shared, repeat_policy="keep")
        retired = Q.filter_year(data, args.min_shared, repeat_policy="drop_shorter")
        sysno = lambda i: int(data["system"][i])  # noqa: E731
        dst = os.path.join(args.out_directory, f"ERA5_AEW_tracks_atlantic_{year}.nc")
        tmp = dst + ".partial"
        Q.write_subset(path, tmp, r["keep"], args.min_shared)
        os.replace(tmp, dst)
        years[year] = {k: r[k] for k in ("n_systems", "n_atlantic", "n_kept",
                                          "n_developer_tags_kept",
                                          "n_distinct_storms_kept",
                                          "n_unresolved_storm_identities_kept",
                                          "n_shared_tail_pairs")}
        years[year]["observations_kept"] = r["observations"]
        # SYSTEM NUMBERS, the published one-based coordinate, never array indices
        years[year]["shared_tail_pairs"] = [
            {"systems": [sysno(q["a"]), sysno(q["b"])],
             "names": [str(data["name"][q["a"]]), str(data["name"][q["b"]])],
             "first_shared_step": q["first_shared"], "last_shared_step": q["last_shared"],
             "n_shared": q["n_shared"]} for q in r["shared_tail_pairs"]]
        years[year]["shared_tail_groups"] = [[sysno(i) for i in g]
                                             for g in r["clusters"] if len(g) > 1]
        # what the retired rule would have done, for the record and nothing else
        years[year]["retired_drop_shorter_rule"] = {
            "would_remove": [sysno(i) for i in np.where(retired["duplicate"])[0]],
            "observations_if_applied": retired["observations"]}
        years[year]["input_sha256"] = _sha256(path)
        years[year]["output_sha256"] = _sha256(dst)
        for k, v in years[year].items():
            if isinstance(v, int):
                totals[k] = totals.get(k, 0) + v
        for k, v in years[year]["observations_kept"].items():
            totals["observations_" + k] = totals.get("observations_" + k, 0) + v
        totals["retired_rule_would_remove"] = totals.get("retired_rule_would_remove", 0) \
            + len(years[year]["retired_drop_shorter_rule"]["would_remove"])
        o = r["observations"]
        print(f"  {year}: {r['n_systems']:4d} systems, {r['n_atlantic']:4d} Atlantic kept, "
              f"{r['n_shared_tail_pairs']:2d} shared-tail pairs, {o['copied_records']:4d} "
              f"copied of {o['valid_records']:5d} records, {r['n_distinct_storms_kept']} "
              f"storm keys", flush=True)
    summary = {
        "generated_by": "scripts/filter_qtrack_atlantic.py",
        "rules": {"atlantic": "any valid step with basin_des in " + str(sorted(Q.ATLANTIC_CODES))
                              + " and first valid code not in " + str(sorted(Q.PACIFIC_CODES)),
                  "identifiers": "every system in this summary is the published one-based "
                                 "`system` coordinate, not an array index",
                  "basin_key": {str(k): v for k, v in Q.BASIN_KEY.items()},
                  "basin_key_status": "inferred from the tracks "
                                      "(docs/aewc_v2/artifacts/qtrack_basin_codes.json), "
                                      "awaiting the authors' confirmation",
                  "shared_tails": f"tracks sharing >= {args.min_shared} identical (lon, lat) "
                                  "positions at the same time step are KEPT and listed; the "
                                  "installed QTrack post-processing can impose the shape by "
                                  "copying one track's coordinates into another, the "
                                  "archive's producing revision is unconfirmed, and the heads "
                                  "are separate detections whose physical relation is not "
                                  "settled",
                  "observation_counting": "distinct = exact (time step, lon, lat) within a "
                                          "file among kept tracks; copied = valid minus "
                                          "distinct; a record on three tracks is one distinct "
                                          "and two copies",
                  "storm_identity": "(TC_name, TC_gen_time in nanoseconds since 1970); a "
                                    "storm tagged on both heads of a pair counts once; a "
                                    "tagged track with missing, zero or non-finite genesis "
                                    "is an unresolved identity counted separately, never "
                                    "merged by name; a count of tags, not of verified "
                                    "storms or their basins",
                  "retired_rule": "drop_shorter, reported per year for the record only"},
        "totals": totals,
        "years": years,
        "source_sha256": {"src/aew/qtrack.py": _sha256(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "src", "aew", "qtrack.py")),
            "scripts/filter_qtrack_atlantic.py": _sha256(os.path.abspath(__file__))},
        "what_this_does_not_settle": "whether the basin key is the authors' own; whether "
                                     "the heads of a shared-tail pair are distinct physical "
                                     "waves, one wave detected twice, or a tracking error; "
                                     "and which copy of a shared record an analysis should "
                                     "count. The first two are questions for the archive's "
                                     "authors, the third is each analysis's declared choice."}
    print(f"totals: {totals}", flush=True)
    if args.summary:
        with open(args.summary, "w") as fh:
            json.dump(summary, fh, indent=1, sort_keys=True)
        print(f"written to {args.summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
