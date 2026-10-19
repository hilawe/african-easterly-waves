#!/usr/bin/env python3
"""A reading added after the QTrack look-back's declared reading (`EASTERN_QTRACK_LOOKBACK_2026-10-19.md`,
section 10): per lag, the composite's maximum (a southerly anomaly) over the same relative
longitudes, its location, and how many of the 23 season-shifted composites reach it, read
from the retained archive record. No construction tested maxima, so this is a description of
the record, not a test.

    python3 scripts/eastern_qtrack_lookback_maxima.py --archive PATH --out PATH
"""

import argparse
import json
import os

import numpy as np

LAGS = tuple(range(-5, 3))
DREL = np.arange(-15.0, 30.0 + 1e-9, 0.5)
TIE_EPS = 1e-9


def as_array(rows):
    return np.array([[np.nan if v is None else v for v in row] for row in rows], float)


def maxima_table(archive):
    """{rep: [(lag, maximum, location, shifts at or above, largest shift maximum)]}, the
    westernmost location on a tie, a shift at or above when within TIE_EPS, and the count
    unavailable when any shift has no value at that lag."""
    out = {}
    for rep, r in archive["results"].items():
        C = as_array(r["composite"]); S = [as_array(c) for c in r["shift_composites"]]
        rows = []
        for i, k in enumerate(LAGS):
            if not np.isfinite(C[i]).any():
                rows.append((k, None, None, None, None))
                continue
            j = int(np.nanargmax(C[i])); m = float(C[i, j])
            smax = [float(np.nanmax(s[i])) for s in S if np.isfinite(s[i]).any()]
            complete = len(smax) == len(S)      # a shift without a value leaves the count unavailable (review fold, section 12)
            rows.append((k, m, float(DREL[j]), (int(sum(1 for v in smax if v >= m - TIE_EPS)) if complete else None), (max(smax) if smax else None)))
        out[rep] = rows
    return out


def maxima_text(table, source):
    f = lambda x: "" if x is None else f"{x:.2f}"
    L = ["# QTrack look-back, the composite maxima, read after the declared reading (section 10)", "",
         f"from {source}, maximum in m/s at its relative longitude D in degrees, shifts of 23 at or above it, no construction tested maxima", "",
         "| representation | lag | maximum | D | shifts at or above | largest shift maximum |", "| --- | --- | --- | --- | --- | --- |"]
    for rep, rows in table.items():
        for k, m, d, n, smax in rows:
            L.append(f"| {rep} | {k} | {f(m)} | {'' if d is None else f'{d:g}'} | {'' if n is None else n} | {f(smax)} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"REFUSED: {a.out} exists and a record is never overwritten")
    with open(a.archive, encoding="utf-8") as fh:
        archive = json.load(fh)
    text = maxima_text(maxima_table(archive), os.path.basename(a.archive))
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
