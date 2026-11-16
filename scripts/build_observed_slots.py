#!/usr/bin/env python
"""The calendar of time slots that one satellite position observed, from the source's slot counts.

    PYTHONPATH=src .venv/bin/python scripts/build_observed_slots.py \\
        --counts data/original/csct/allcs_slot_counts_jas_1983_2007_by_position.txt.gz

The input is a read-only pull of slot counts from the source record, which is not redistributed. It
holds, for every July to September slot and satellite position with at least one source row, the
number of rows of any size. A slot in which the position has no row anywhere in its view was not
observed, and a slot with under a quarter of the median row count of its hour of day is taken as
a partial image. Writes the observed slots to scripts/met_observed_slots.csv, which the pipeline
reads, and a record of what was found beside it.
"""
import argparse
import gzip
import hashlib
import json
import os

import numpy as np
import pandas as pd

from aew.observed import PARTIAL_FRACTION, observed_from_counts

HERE = os.path.dirname(os.path.abspath(__file__))
YEARS = range(1983, 2008)


def read_counts(path, position):
    """Slot times and row counts of one position, with the years the pull covered."""
    rows, years = [], []
    with gzip.open(path, "rt", errors="replace") as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] == "E":
                raise SystemExit(f"REFUSED: the pull reports a missing year file: {line.strip()}")
            if p[0] == "Y":
                years.append(int(p[1]))
            elif p[0] == "S" and p[1] == position:
                if len(p) != 9:
                    raise SystemExit(f"REFUSED: a slot line does not hold nine fields: {line.strip()}")
                rows.append((pd.Timestamp(year=int(p[2]), month=int(p[3]), day=int(p[4]), hour=int(p[5])), int(p[6])))
    if not rows:
        raise SystemExit(f"REFUSED: the pull holds no slot of position {position}")
    t = pd.DatetimeIndex([r[0] for r in rows])
    return t, np.array([r[1] for r in rows]), sorted(years)


def season_slots(years):
    return pd.DatetimeIndex(np.concatenate([pd.date_range(f"{y}-07-01", f"{y}-09-30 21:00", freq="3h").values for y in years]))


def runs_of(missing):
    """Runs of consecutive missing slots as (first, last, slots)."""
    m = pd.Series(pd.DatetimeIndex(sorted(missing)))
    if m.empty:
        return []
    block = (m.diff() != pd.Timedelta("3h")).cumsum()
    return [(str(g.iloc[0]), str(g.iloc[-1]), int(len(g))) for _, g in m.groupby(block)]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--position", default="MET")
    ap.add_argument("--out", default=os.path.join(HERE, "met_observed_slots.csv"))
    a = ap.parse_args()
    t, n, years = read_counts(a.counts, a.position)
    if years != list(YEARS):
        raise SystemExit(f"REFUSED: the pull covers {years[:1]} to {years[-1:]} and not every year of {YEARS[0]} to {YEARS[-1]}")
    slots = season_slots(YEARS)
    if not t.isin(slots).all():
        raise SystemExit("REFUSED: a slot of the pull lies outside the 3-hourly July to September lattice")
    observed, partial = observed_from_counts(t, n)
    order = np.argsort(t.values)
    count_of = pd.Series(n[order], index=t.values[order])
    pd.DataFrame({"time": pd.DatetimeIndex(observed).strftime("%Y-%m-%d %H:%M"), "rows": count_of.reindex(observed).values.astype(int)}).to_csv(a.out, index=False)
    absent = slots[~slots.isin(t)]
    unobserved = slots[~slots.isin(pd.DatetimeIndex(observed))]
    runs = runs_of(unobserved)
    by_year = pd.Series(unobserved.year).value_counts().reindex(list(YEARS), fill_value=0)
    midnight_sept = pd.Series(unobserved[(unobserved.month == 9) & (unobserved.hour == 0)].year).value_counts().reindex(list(YEARS), fill_value=0)
    record = dict(counts=a.counts, counts_sha256=sha256(a.counts), position=a.position, partial_fraction=PARTIAL_FRACTION,
                  slots_of_the_seasons=int(len(slots)), slots_with_a_row=int(len(t)), slots_without_a_row=int(len(absent)),
                  slots_taken_as_partial=[dict(time=str(pd.Timestamp(x)), rows=int(count_of[x])) for x in partial],
                  slots_observed=int(len(observed)), slots_not_observed=int(len(unobserved)),
                  rows_per_slot_percentiles={str(q): float(np.percentile(n, q)) for q in (0, 1, 5, 25, 50, 75, 100)},
                  median_rows_by_hour={str(int(h)): float(v) for h, v in pd.Series(n).groupby(t.hour.values).median().items()},
                  lowest_row_count_taken_as_observed=int(count_of.reindex(observed).min()),
                  not_observed_by_year={str(y): int(v) for y, v in by_year.items()},
                  not_observed_by_hour={str(h): int((unobserved.hour == h).sum()) for h in range(0, 24, 3)},
                  not_observed_at_00_utc_in_september_by_year={str(y): int(v) for y, v in midnight_sept.items()},
                  runs_not_observed=len(runs), runs_by_length={str(k): int(v) for k, v in pd.Series([r[2] for r in runs]).value_counts().sort_index().items()},
                  runs_of_a_day_or_more=[dict(first=r[0], last=r[1], slots=r[2]) for r in runs if r[2] >= 8],
                  out=os.path.relpath(a.out), out_sha256=sha256(a.out))
    with open(os.path.splitext(a.out)[0] + "_record.json", "w") as f:
        json.dump(record, f, indent=1)
    print(f"{a.position}: {len(observed)} of {len(slots)} slots observed, {len(absent)} without a row, {len(partial)} taken as partial; wrote {os.path.relpath(a.out)}")


if __name__ == "__main__":
    main()
