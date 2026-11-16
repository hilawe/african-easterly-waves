"""Which 3-hourly time slots a satellite position observed, and what follows for counts.

The cloud-system catalog is a list of detections, so a slot without a row can be an empty sky
or a missing image. The source rows carry a satellite-position code, and a slot in which one
position has no row of any size anywhere in its view is a slot that position did not observe.
This module holds the calendar of observed slots and the rules that keep an unobserved slot
from being read as a count of zero:

- ``window_observed_mask``: a trough is eligible only when every slot of its forward window and
  of its antecedent window was observed,
- ``slots_observed``: the same test for any set of offsets from a time,
- ``exposure_series`` and ``fill_short_gaps``: a count series on a coarser grid, corrected for
  the slots it could not see,
- ``support_check``: every relative-longitude bin of every placed trough inside the longitude
  reach of a regional catalog, bin edges included and with no allowance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SLOT_H = 3.0                      # the catalog's cadence
PARTIAL_FRACTION = 0.25           # a slot under this share of its hour's median row count is a partial image


def _ns(times):
    return pd.DatetimeIndex(np.asarray(times)).values.astype("datetime64[ns]").astype("int64")


def observed_from_counts(times, rows, fraction=PARTIAL_FRACTION):
    """Slots a position observed, from its row count of any size at every slot that has a row.

    A slot with no row is absent from ``times`` and so is not observed. A slot whose count is
    under ``fraction`` of the median count of its hour of day is taken as a partial image and
    is not observed either. Returns the observed slot times, sorted, and the flagged ones.
    """
    t = pd.DatetimeIndex(np.asarray(times))
    n = np.asarray(rows, dtype=float)
    if t.has_duplicates:
        raise ValueError("a slot is listed twice")
    if n.size == 0 or (n <= 0).any():
        raise ValueError("every listed slot must hold at least one row")
    med = pd.Series(n).groupby(t.hour.values).median()
    partial = n < fraction * med.reindex(t.hour.values).values
    order = np.argsort(t.values)
    keep = ~partial[order]
    return t.values[order][keep], np.sort(t.values[partial])


def load_observed_slots(path):
    """The observed slots of a calendar file written by scripts/build_observed_slots.py, sorted."""
    d = pd.read_csv(path, parse_dates=["time"])
    if "time" not in d.columns or d.time.duplicated().any() or len(d) == 0:
        raise ValueError(f"{path} is not a calendar of distinct observed slots")
    return np.sort(d.time.values.astype("datetime64[ns]"))


def slots_observed(times, observed, offsets_h):
    """True where every slot at ``times + offset`` hours, for each offset, is an observed slot."""
    obs = np.sort(_ns(observed))
    t = _ns(times)
    out = np.ones(t.size, dtype=bool)
    for off in offsets_h:
        q = t + int(round(off * 3.6e12))
        i = np.searchsorted(obs, q)
        hit = (i < obs.size) & (obs[np.minimum(i, obs.size - 1)] == q)
        out &= hit
    return out


def window_offsets(resp_h=24.0, antecedent=True):
    """Slot offsets in hours of the forward window [t, t + resp_h) and the antecedent window [t - 48, t - 24)."""
    fwd = list(np.arange(0.0, resp_h, SLOT_H))
    return fwd + (list(np.arange(-48.0, -24.0, SLOT_H)) if antecedent else [])


def window_observed_mask(times, observed, resp_h=24.0, antecedent=True):
    """True where both counting windows of a trough observation were observed in full.

    The windows are those of ``aew.environment.forward_response``: the response window
    [t, t + resp_h) and the antecedent window [t - 48 h, t - 24 h). A window with one
    unobserved slot has an unknown count, so the trough is not eligible.
    """
    return slots_observed(times, observed, window_offsets(resp_h, antecedent))


def exposure_series(grid_times, observed):
    """The weight of observed slots at each grid time under ``aew.leadlag.box_count_series``.

    That function splits each 3-hourly observation between its two bracketing grid times in
    proportion to distance. On a 6-hourly grid the slot at a grid time counts in full and the
    slots three hours either side count by half, so a fully observed grid time has weight 2.
    Returns the weight per grid time.
    """
    g = _ns(grid_times)
    if g.size < 2:
        raise ValueError("a grid needs two times at least")
    step = np.diff(g)
    if not (step == step[0]).all() or step[0] <= 0:
        raise ValueError("the grid is not uniform and ascending")
    obs = _ns(observed)
    pos = (obs - g[0]) / float(step[0])
    lo = np.floor(pos).astype(int)
    w_hi = pos - lo
    out = np.zeros(g.size)
    ok = (lo >= 0) & (lo < g.size)
    np.add.at(out, lo[ok], 1.0 - w_hi[ok])
    ok = (lo + 1 >= 0) & (lo + 1 < g.size) & (w_hi > 0)
    np.add.at(out, lo[ok] + 1, w_hi[ok])
    return out


def fill_short_gaps(values, exposure, full, max_gap):
    """A count series scaled to full exposure, with short unobserved runs filled linearly.

    ``values`` are counts at grid times, ``exposure`` their observed weight and ``full`` the
    weight of a fully observed grid time. A grid time with no exposure is a gap. Runs of up to
    ``max_gap`` gaps between two observed times are filled by linear interpolation, and longer
    runs, and runs at either end, stay NaN. Returns the series and the longest run of gaps.
    """
    v = np.asarray(values, dtype=float)
    e = np.asarray(exposure, dtype=float)
    out = np.where(e > 0, v * full / np.where(e > 0, e, 1.0), np.nan)
    gap = ~(e > 0)
    longest, i, n = 0, 0, v.size
    while i < n:
        if not gap[i]:
            i += 1
            continue
        j = i
        while j < n and gap[j]:
            j += 1
        longest = max(longest, j - i)
        if i > 0 and j < n and (j - i) <= max_gap:
            out[i:j] = np.interp(np.arange(i, j), [i - 1, j], [out[i - 1], out[j]])
        i = j
    return out, int(longest)


def support_check(lon_placed, rel_centers, cat_lon_min, cat_lon_max):
    """Stop unless every relative-longitude bin of every placed trough lies inside the catalog's reach.

    ``rel_centers`` are the bin centers of the composite, uniformly spaced, so the outer bin
    edges lie half a bin beyond the outer centers. The reach is the catalog's extreme
    detection longitudes, with no allowance. This is the support assertion of specification
    R1 for a regional catalog, where the guard in ``aew.composites`` passes any catalog that
    spans the globe in longitude and allows one degree beyond the extremes elsewhere.
    """
    rel = np.asarray(rel_centers, dtype=float)
    half = 0.5 * float(rel[1] - rel[0])
    lon = np.asarray(lon_placed, dtype=float)
    west, east = float(lon.min() + rel.min() - half), float(lon.max() + rel.max() + half)
    if west < cat_lon_min or east > cat_lon_max:
        raise RuntimeError(f"support assertion failed: placed troughs need {west:.2f} to {east:.2f} "
                           f"and the catalog reaches {cat_lon_min:.2f} to {cat_lon_max:.2f}")
    return west, east
