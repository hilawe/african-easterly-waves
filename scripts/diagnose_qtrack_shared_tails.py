#!/usr/bin/env python3
"""Record the evidence for the shared-tail pairs in QTrack v2's Atlantic-side tracks.

WHAT IT WRITES. For every pair of Atlantic-side systems sharing at least four identical
positions, its geometry class; for the six pairs that carry two storm names or a storm
name on the shorter member only, the two tracks in full (dates decoded through each
variable's own units, extent, genesis time, steps before the shared span, the shared
span); and the time units of every file. Dates are decoded through each variable's own
units because 43 files store the time axis as hours since 1900, 2024 as seconds since
1970, and TC_gen_time as nanoseconds since 1970 in all 44.

WHAT IT DOES NOT ESTABLISH. Whether the two heads of a pair are distinct physical waves,
and how the archive's files were produced. The INSTALLED QTrack post-processing can
impose a shared tail by copying one track's coordinates into another from the step at
which they come within its merge distance (it compares the last finite longitudes and
copies the westward-ending track's positions into the other); the archive's producing
revision and settings are unconfirmed, so that is an inference from a capable
implementation. The geometry class is what the stored tracks show, and no more.

    .venv/bin/python scripts/diagnose_qtrack_shared_tails.py --out <artifact.json>
"""
import argparse
import collections
import datetime as dt
import glob
import hashlib
import json
import os
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


def geometry_class(valid_a, valid_b, same):
    """shared tail, contained, split, or partial, from validity masks and shared steps.

    The shared-tail class requires the whole SUFFIX from the first shared step to agree:
    equal validity in both tracks and equal positions at every valid step. A first
    version checked only the steps before the first match and after the last, so two
    tracks that diverged in the interior of that span were classed as identical to a
    common end. The contained class likewise requires every valid step of the contained
    track to be a shared step.
    """
    valid_a, valid_b, same = (np.asarray(x, dtype=bool) for x in (valid_a, valid_b, same))
    where = np.where(same)[0]
    f, l = where[0], where[-1]
    a_before, b_before = int(valid_a[:f].sum()), int(valid_b[:f].sum())
    a_after, b_after = int(valid_a[l + 1:].sum()), int(valid_b[l + 1:].sum())
    suffix_agrees = (np.array_equal(valid_a[f:], valid_b[f:])
                     and np.array_equal(same[f:], valid_a[f:]))
    if np.array_equal(same, valid_a) or np.array_equal(same, valid_b):
        return "contained"
    if a_before > 0 and b_before > 0 and a_after == 0 and b_after == 0 and suffix_agrees:
        return "shared tail: different heads, identical positions to a common end"
    if a_before == 0 and b_before == 0 and np.array_equal(same[:l + 1], valid_a[:l + 1]) \
            and np.array_equal(same[:l + 1], valid_b[:l + 1]):
        return "split: shared head, different tails"
    return "partial overlap"


def decode_genesis(value, units):
    """A genesis value decoded through ITS OWN declared units, or None for no genesis.

    Zero, NaN and non-finite are the archive's no-genesis sentinels. Nanoseconds are
    handled by conversion to seconds (netCDF4's calendar decoding has no nanosecond
    unit); any other declared unit goes through netCDF4's decoder; an unreadable unit
    string is refused rather than guessed. A first version always divided by one
    billion, which turned a genesis declared in seconds into 1970.
    """
    if value != value or not np.isfinite(value) or value <= 0:
        return None
    units = str(units)
    head = units.split(" since ")[0].strip().lower()
    if head in ("nanoseconds", "nanosecond", "ns"):
        units = "seconds since " + units.split(" since ", 1)[1]
        value = value / 1e9
    try:
        g = nc.num2date(value, units, only_use_cftime_datetimes=False,
                        only_use_python_datetimes=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"cannot decode genesis with units {units!r}: {exc}") from exc
    return g.replace(tzinfo=dt.timezone.utc)


def relative_genesis(g, first_shared_time):
    """'before' or 'at or after' the first shared step, None for no genesis."""
    if g is None:
        return None
    return "before" if g < first_shared_time.replace(tzinfo=dt.timezone.utc) else "at or after"


def gen_date(g):
    return None if g is None else g.strftime("%Y-%m-%d %HZ")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--directory", default="data/aewc_v2_pilot/ERA5_WITH_EPAC")
    ap.add_argument("--min-shared", type=int, default=Q.DEFAULT_MIN_SHARED)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    files = sorted(glob.glob(os.path.join(args.directory, "*.nc")))
    if not files:
        print(f"REFUSED: no .nc files under {args.directory}", flush=True)
        return 2
    classes = collections.Counter()
    named, units = [], {}
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
        units[year] = {"time": tv.units, "TC_gen_time": gen_units}
        d.close()
        data = Q.read_year(path)
        r = Q.filter_year(data, args.min_shared, repeat_policy="keep")
        lon, lat, names, gen = data["lon"], data["lat"], data["name"], data["gen_time"]
        sysno = data["system"].astype(int)
        valid = ~np.isnan(lon) & ~np.isnan(lat)
        for q in r["shared_tail_pairs"]:
            a, b = q["a"], q["b"]
            same = valid[a] & valid[b] & (lon[a] == lon[b]) & (lat[a] == lat[b])
            classes[geometry_class(valid[a], valid[b], same)] += 1
            tagged = [i for i in (a, b) if names[i] != "N/A"]
            two_names = len({names[i] for i in tagged}) == 2
            name_on_shorter_only = (len(tagged) == 1 and
                                    valid[tagged[0]].sum() < valid[a if tagged[0] == b else b].sum())
            if not (two_names or name_on_shorter_only):
                continue
            where = np.where(same)[0]

            def track(i):
                ok = np.where(valid[i])[0]
                return {"system": int(sysno[i]), "name": str(names[i]),
                        "steps": int(ok.size),
                        "first": times[ok[0]].strftime("%Y-%m-%d %HZ"),
                        "last": times[ok[-1]].strftime("%Y-%m-%d %HZ"),
                        "start_lon_lat": [round(float(lon[i, ok[0]]), 1),
                                          round(float(lat[i, ok[0]]), 1)],
                        "tc_genesis": gen_date(decode_genesis(gen[i], gen_units)),
                        "steps_before_shared_span": int(valid[i][:where[0]].sum())}
            named.append({"year": year, "tracks": [track(a), track(b)],
                          "shared_steps": int(same.sum()),
                          "shared_span": {"first": times[where[0]].strftime("%Y-%m-%d %HZ"),
                                          "last": times[where[-1]].strftime("%Y-%m-%d %HZ"),
                                          "lon_lat_at_first": [round(float(lon[a, where[0]]), 1),
                                                               round(float(lat[a, where[0]]), 1)]},
                          "genesis_relative_to_shared_span": {
                              str(sysno[i]): relative_genesis(decode_genesis(gen[i], gen_units),
                                                              times[where[0]])
                              for i in (a, b)}})
    out = {"what_this_establishes": __doc__.split("WHAT IT DOES NOT ESTABLISH.")[0].strip(),
           "what_this_does_not_establish": "WHAT IT DOES NOT ESTABLISH."
           + __doc__.split("WHAT IT DOES NOT ESTABLISH.")[1].split("\n\n")[0],
           "pair_geometry_classes": dict(classes),
           "pairs_with_two_names_or_a_name_on_the_shorter_member_only": named,
           "time_units_by_year": units,
           "generated_by": "scripts/diagnose_qtrack_shared_tails.py",
           "input_file_sha256": {os.path.basename(p): _sha256(p) for p in files},
           "source_sha256": {"scripts/diagnose_qtrack_shared_tails.py": _sha256(os.path.abspath(__file__)),
                             "src/aew/qtrack.py": _sha256(os.path.join(
                                 os.path.dirname(os.path.abspath(__file__)), "..", "src", "aew", "qtrack.py"))}}
    print("pair geometry classes:", dict(classes))
    for c in named:
        a, b = c["tracks"]
        print(f"  {c['year']}: {a['system']} {a['name']} + {b['system']} {b['name']}, shared "
              f"{c['shared_steps']} from {c['shared_span']['first']}; genesis "
              f"{c['genesis_relative_to_shared_span']}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
        print(f"written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
