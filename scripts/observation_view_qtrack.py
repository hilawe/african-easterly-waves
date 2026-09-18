#!/usr/bin/env python3
"""Write the observation view for the QTrack Atlantic population: one owning track per
shared record, so every density or longitude-time analysis counts each stored location
once by the same tested rule.

WHAT IT WRITES. Per year, a companion netCDF beside the filtered file
(`ERA5_AEW_tracks_atlantic_<year>_owned.nc`) with `system` (the original numbers),
`time`, and `owned` (system x time, 1 where a kept track's record is the one to count,
0 elsewhere), plus attributes stating the rule. And a summary artifact with, per year,
the valid, distinct and owned record counts (owned must equal distinct, and the driver
refuses to write a year where it does not), the number of records with more than one
holder, and the input and source hashes.

THE RULE, from `aew.qtrack.observation_owners`: a record held by several kept tracks is
owned by the track with the most valid records, ties to the lower system number.
Ownership decides counting only; coordinates are untouched.

    .venv/bin/python scripts/observation_view_qtrack.py \\
        --tracks data/aewc_v2_pilot/ERA5_ATLANTIC --summary <artifact.json>
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


def write_owned(src, dst, owned, rule_text):
    s = nc.Dataset(src)
    d = nc.Dataset(dst, "w")
    try:
        d.setncattr("aew_observation_rule", rule_text)
        d.setncattr("source_file", os.path.basename(src))
        d.createDimension("system", len(s.dimensions["system"]))
        d.createDimension("time", None)
        # Both coordinate variables are copied with EVERY attribute, because a time axis
        # is only the same axis under the same units AND calendar: a review wrote a
        # 360_day source and watched a companion carrying units alone decode the same
        # number to a different date. A consumer must be able to decode the two files
        # to identical coordinates, and a test reopens both and checks that it can.
        for name in ("system", "time"):
            src = s[name]
            attrs = {k: src.getncattr(k) for k in src.ncattrs()}
            fill = attrs.pop("_FillValue", None)
            v = d.createVariable(name, src.dtype, src.dimensions, fill_value=fill)
            for k, val in attrs.items():
                v.setncattr(k, val)
            v[:] = src[:]
        ov = d.createVariable("owned", "i1", ("system", "time"))
        ov.long_name = "1 where this track's record is the one to count at this step"
        ov[:] = owned.astype(np.int8)
    finally:
        d.close()
        s.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--tracks", default="data/aewc_v2_pilot/ERA5_ATLANTIC")
    ap.add_argument("--summary", default=None)
    args = ap.parse_args(argv)
    # The directory IS the population, so every netCDF file in it must be either a
    # filtered year in the declared name or that year's companion. A stray file with a
    # year in its name once made the driver count a year twice while recording it once
    # (a review's planted backup_1999.nc), so anything else is a refusal, and the
    # declared name makes two files for one year impossible in one directory.
    pattern = re.compile(r"^ERA5_AEW_tracks_atlantic_(\d{4})\.nc$")
    files, years_seen = [], set()
    for p in sorted(glob.glob(os.path.join(args.tracks, "*.nc"))):
        base = os.path.basename(p)
        if base.endswith("_owned.nc"):
            continue
        m = pattern.match(base)
        if not m:
            print(f"REFUSED: {base} is not a filtered year file; the directory must hold "
                  f"only ERA5_AEW_tracks_atlantic_<year>.nc and their companions",
                  flush=True)
            return 2
        if m.group(1) in years_seen:
            print(f"REFUSED: year {m.group(1)} appears twice", flush=True)
            return 2
        years_seen.add(m.group(1))
        files.append(p)
    if not files:
        print(f"REFUSED: no track files under {args.tracks}", flush=True)
        return 2
    rule = ("a record held by several tracks is owned by the track with the most valid "
            "records, ties to the lower system number; owned records equal distinct "
            "(time step, lon, lat) records")
    years, totals = {}, {"valid": 0, "distinct": 0, "owned": 0, "multi_holder": 0}
    worked_example = None
    for path in files:
        year = pattern.match(os.path.basename(path)).group(1)
        data = Q.read_year(path)
        keep = np.ones(len(data["system"]), dtype=bool)      # the file IS the population
        counts = Q.observation_counts(data["lon"], data["lat"], keep)
        view = Q.observation_owners(data["lon"], data["lat"], keep, data["system"])
        owned = int(view["owned"].sum())
        if owned != counts["distinct_records"]:
            print(f"REFUSED: {year} owned {owned} != distinct {counts['distinct_records']}",
                  flush=True)
            return 2
        holders = {}
        valid = ~np.isnan(data["lon"]) & ~np.isnan(data["lat"])
        for i in range(len(keep)):
            for t in np.where(valid[i])[0]:
                holders[(int(t), float(data["lon"][i, t]), float(data["lat"][i, t]))] = \
                    holders.get((int(t), float(data["lon"][i, t]), float(data["lat"][i, t])), 0) + 1
        multi = sum(1 for v in holders.values() if v > 1)
        if worked_example is None and multi:
            # One shared record written out in full, so a reader can check the rule on
            # a real case rather than on aggregate agreement: the first shared record of
            # the first year that has one, its holders with their valid-record counts,
            # and the owner the rule chose.
            key = min(k for k, v in holders.items() if v > 1)
            t = key[0]
            held_by = [i for i in range(len(keep)) if valid[i, t]
                       and (int(t), float(data["lon"][i, t]), float(data["lat"][i, t])) == key]
            worked_example = {
                "year": year, "time_index": t, "lon": key[1], "lat": key[2],
                "holders": [{"system": int(data["system"][i]),
                             "valid_records": int(valid[i].sum())} for i in held_by],
                "owner_system": view["owner_system"][key]}
        dst = path[:-3] + "_owned.nc"
        tmp = dst + ".partial"
        write_owned(path, tmp, view["owned"], rule)
        os.replace(tmp, dst)
        years[year] = {"valid_records": counts["valid_records"],
                       "distinct_records": counts["distinct_records"],
                       "owned_records": owned, "records_with_several_holders": multi,
                       "input_sha256": _sha256(path), "output_sha256": _sha256(dst)}
        totals["valid"] += counts["valid_records"]
        totals["distinct"] += counts["distinct_records"]
        totals["owned"] += owned
        totals["multi_holder"] += multi
        print(f"  {year}: {counts['valid_records']:6d} valid, {counts['distinct_records']:6d} "
              f"distinct, {owned:6d} owned, {multi:4d} shared", flush=True)
    # Totals are accumulated beside the per-year entries; the two are reconciled here so
    # a summary can never report a total its own entries do not add up to.
    for total_key, year_key in (("valid", "valid_records"), ("distinct", "distinct_records"),
                                ("owned", "owned_records"),
                                ("multi_holder", "records_with_several_holders")):
        if totals[total_key] != sum(y[year_key] for y in years.values()):
            print(f"REFUSED: total {total_key} does not equal the sum over years", flush=True)
            return 2
    summary = {"generated_by": "scripts/observation_view_qtrack.py", "rule": rule,
               "totals": totals, "years": years, "worked_example": worked_example,
               "source_sha256": {"src/aew/qtrack.py": _sha256(os.path.join(
                   os.path.dirname(os.path.abspath(__file__)), "..", "src", "aew", "qtrack.py")),
                   "scripts/observation_view_qtrack.py": _sha256(os.path.abspath(__file__))},
               "what_this_does_not_settle": "which track's intensity or other field to "
                                            "read at a shared position; ownership decides "
                                            "counting only"}
    print("totals:", totals, flush=True)
    if args.summary:
        with open(args.summary, "w") as fh:
            json.dump(summary, fh, indent=1, sort_keys=True)
        print(f"written to {args.summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
