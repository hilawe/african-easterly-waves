#!/usr/bin/env python
"""Trough observations and trajectories of the corridor before and after deduplication.

The climatology represents some waves with more than one trajectory, and the loader
removes the repeats before anything is composited. The paper states what that removal
does in words ("close to two in five", "about half"), and this script counts it, so the
words can be read against numbers. Two stages are counted over the corridor of the paper
(5 to 20 N, 30 W to 40 E, July to September): the record as archived and the record after
deduplication. For each stage the script gives the observations, the trajectories or
waves that carry them, and the distinct pairs of time and position.

A pure consumer. It reads the archived trajectories twice through the loader, integrates
nothing and draws no random number. Before it writes, it requires the deduplicated count
to equal the corridor count that the deposit states, and stops if it does not.

Writes deposit/dedup_counts.csv.
"""

import argparse
import os

import numpy as np
import pandas as pd

from aew.data.aewc import load_aewc_troughs

CORRIDOR = dict(min_lat=5, max_lat=20, min_lon=-30, max_lon=40)
MONTHS = (7, 8, 9)


def count_stage(time, lat, lon, traj_id):
    """Observations, trajectories and distinct time-position pairs of one stage."""
    time = np.asarray(time).astype("datetime64[ns]")
    key = pd.DataFrame(dict(t=time, lat=np.asarray(lat, float), lon=np.asarray(lon, float)))
    return dict(observations=int(len(key)), trajectories=int(np.unique(np.asarray(traj_id)).size),
                distinct_time_positions=int(len(key.drop_duplicates())))


def derived(raw, dedup):
    """The shares the text states in words, from the two stages."""
    if raw["observations"] <= 0 or raw["trajectories"] <= 0:
        raise SystemExit("REFUSED: the archived record holds no observation or no trajectory")
    repeats = raw["observations"] - raw["distinct_time_positions"]
    return dict(pct_archived_repeats=100.0 * repeats / raw["observations"],
                pct_waves_of_trajectories=100.0 * dedup["trajectories"] / raw["trajectories"])


def deposit_count(dep):
    r = pd.read_csv(os.path.join(dep, "canonical_numbers.csv"))
    m = (r.tier == "pooled") & (r.statistic == "n_corridor_troughs_all_season")
    if int(m.sum()) != 1:
        raise SystemExit(f"REFUSED: the deposit states the corridor count {int(m.sum())} times")
    return int(round(float(r["diff"][m].iloc[0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aewc-glob", default="data/aewc/ERA-Int_ew_700hPa_*_AFR.nc")
    ap.add_argument("--deposit", default="deposit")
    a = ap.parse_args()
    stages = {}
    for name, dd in (("as_archived", False), ("deduplicated", True)):
        tr = load_aewc_troughs(a.aewc_glob, dedup=dd).filter_region(**CORRIDOR).filter_months(MONTHS)
        stages[name] = count_stage(tr.time, tr.lat, tr.lon, tr.variables["traj_id"])
    want = deposit_count(a.deposit)
    if stages["deduplicated"]["observations"] != want:
        raise SystemExit(f"REFUSED: {stages['deduplicated']['observations']} observations after deduplication, "
                         f"and the deposit states {want}")
    d = derived(stages["as_archived"], stages["deduplicated"])
    rows = [dict(stage=k, **v) for k, v in stages.items()]
    out = pd.DataFrame(rows)
    for k, v in d.items():
        out[k] = v
    path = os.path.join(a.deposit, "dedup_counts.csv")
    out.to_csv(path, index=False, float_format="%.6f")
    for row in rows:
        print(f"{row['stage']:13s} {row['observations']:7,d} observations on {row['trajectories']:5,d} trajectories, "
              f"{row['distinct_time_positions']:7,d} distinct time-position pairs")
    print(f"repeats among the archived observations {d['pct_archived_repeats']:.1f} percent, "
          f"waves per archived trajectory {d['pct_waves_of_trajectories']:.1f} percent")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
