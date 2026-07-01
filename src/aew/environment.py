"""Developing vs non-developing trough environment (the conditioned causality test).

The amplitude-preconditioning test (does a stronger wave 48 h earlier carry more convection)
was near-null, which is expected where convection organizes the wave rather than the reverse.
Instead of asking whether the wave forces the convection linearly, this conditions on the
thermodynamic environment: split troughs by the convection that DID develop, then ask whether
the developing troughs sat in a moister environment BEFORE that convection existed. A moisture
precondition set ahead of the convection is a cleaner causal handle than the wave-convection
covariation, which cannot separate the two.

Two pure pieces:

- ``forward_response`` -- per-trough convective response from an independent cloud-system
  record: the count of systems in a trough-relative box over a forward time window. This is
  the variable troughs are split on (developing vs non-developing).
- ``lead_value`` -- the pre-trough environment sampler: for each trough it takes a
  per-trajectory scalar (e.g. trough-mean TCWV) from the nearest real earlier sample about
  ``lead_h`` before the observation, within a tolerance. It never interpolates across a gap,
  so a sparse satellite field (AEWC SSM/I TCWV is ~40% populated) contributes only where a
  genuine earlier sample exists, rather than a fabricated one.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

__all__ = ["forward_response", "lead_value", "lead_field_box", "terciles",
           "cluster_bootstrap_diff"]


def _ns_hours(times):
    return pd.DatetimeIndex(times).values.astype("datetime64[ns]").astype("int64") / 3.6e12


def forward_response(trough_times, trough_lons, cs_time, cs_lon, cs_lat,
                     win_h=24.0, dlon=8.0, lat_lo=5.0, lat_hi=15.0):
    """Per-trough convective response: cloud systems in a trough-relative box ahead in time.

    For each trough observation ``(t, L)`` this counts cloud systems whose time is in
    ``[t, t + win_h)`` and whose longitude relative to the trough (wrapped to [-180, 180))
    is within ``+/- dlon`` and whose latitude is in ``[lat_lo, lat_hi]``. The forward window
    measures the convection that develops with and after the trough, which is what separates
    developing from non-developing troughs.

    Returns an array of counts, one per trough observation (in input order).
    """
    tt = _ns_hours(trough_times)
    L = np.asarray(trough_lons, dtype=float)
    ct = _ns_hours(cs_time)
    clon = np.asarray(cs_lon, dtype=float)
    clat = np.asarray(cs_lat, dtype=float)
    o = np.argsort(ct)
    ct, clon, clat = ct[o], clon[o], clat[o]

    out = np.zeros(tt.size)
    for i in range(tt.size):
        lo = np.searchsorted(ct, tt[i], side="left")
        hi = np.searchsorted(ct, tt[i] + win_h, side="left")
        if hi <= lo:
            continue
        rel = (clon[lo:hi] - L[i] + 180.0) % 360.0 - 180.0
        m = (np.abs(rel) <= dlon) & (clat[lo:hi] >= lat_lo) & (clat[lo:hi] <= lat_hi)
        out[i] = float(m.sum())
    return out


def lead_value(trough_times, traj_id, values, lead_h=24.0, tol_h=6.0):
    """Pre-trough environment: a per-trajectory scalar taken ~``lead_h`` before each trough.

    For each trough observation, this looks along its own trajectory for the sample nearest
    to ``t - lead_h`` that is strictly earlier than ``t`` and has a finite value, and returns
    that value only if it lies within ``tol_h`` of the target lead time. Otherwise NaN. No
    interpolation across gaps, so a sparse ``values`` field contributes only real samples.

    Parameters
    ----------
    trough_times : array of datetime64
    traj_id : array of int    trajectory id per observation (groups a wave's samples)
    values : array of float   the per-observation scalar (NaN where unobserved)
    lead_h : float            target lead in hours before the observation
    tol_h : float             accept a sample within this many hours of the target

    Returns
    -------
    lead : ndarray, one value per trough observation (NaN where no real earlier sample)
    """
    th = _ns_hours(trough_times)
    tid = np.asarray(traj_id)
    vals = np.asarray(values, dtype=float)
    out = np.full(th.size, np.nan)

    groups = defaultdict(list)
    for i in range(th.size):
        groups[tid[i]].append(i)

    for idx in groups.values():
        idx = np.array(idx)
        order = np.argsort(th[idx])
        idx = idx[order]
        tt = th[idx]
        vv = vals[idx]
        finite = np.isfinite(vv)
        if not finite.any():
            continue
        ft = tt[finite]
        fv = vv[finite]
        for k in range(idx.size):
            target = tt[k] - lead_h
            earlier = ft < tt[k]
            if not earlier.any():
                continue
            cand_t = ft[earlier]
            cand_v = fv[earlier]
            j = int(np.argmin(np.abs(cand_t - target)))
            if abs(cand_t[j] - target) <= tol_h:
                out[idx[k]] = cand_v[j]
    return out


def lead_field_box(trough_times, trough_lons, f_time, f_lat, f_lon, field,
                   lead_h=24.0, tol_h=3.0, dlon=5.0, lat_lo=5.0, lat_hi=15.0):
    """Pre-arrival environment from a gridded field: box mean at the trough's meridian,
    ``lead_h`` hours BEFORE the trough observation.

    For a trough observed at ``(t, L)``, the sampled air is the box ``L +/- dlon``,
    ``[lat_lo, lat_hi]`` at time ``t - lead_h``, taken from the nearest field time within
    ``tol_h``. Because an African easterly wave moves westward (~7 deg/day), at ``t - 24 h``
    the trough sits ~7 deg east of ``L``, so with the default 5-deg half-width its prior-day
    convective envelope can reach the eastern box edge; the separation grows with a westward
    box offset (pass a shifted ``trough_lons``) or a longer lead, and the caller should report
    those sensitivities rather than treat a single box as fully upstream. This is the
    full-coverage (e.g. ERA5) replacement for the sparse along-track satellite sampler
    ``lead_value``.

    Parameters
    ----------
    trough_times : array of datetime64
    trough_lons : array of float
    f_time : array of datetime64      field times (sorted ascending)
    f_lat, f_lon : 1-D arrays          field coordinates (degrees)
    field : ndarray (ntime, nlat, nlon)
    lead_h, tol_h : float              lead and time-match tolerance in hours
    dlon : float                       box half-width in longitude
    lat_lo, lat_hi : float             latitude band

    Returns
    -------
    out : ndarray, one box-mean value per trough observation (NaN where no field time is
          within ``tol_h`` of the target, or the box has no finite cells)
    """
    tt = _ns_hours(trough_times)
    L = np.asarray(trough_lons, dtype=float)
    ft = _ns_hours(f_time)
    if ft.size > 1 and not np.all(np.diff(ft) > 0):
        raise ValueError("f_time must be sorted ascending")
    f_lat = np.asarray(f_lat, dtype=float)
    f_lon = np.asarray(f_lon, dtype=float)
    field = np.asarray(field, dtype=float)
    band = (f_lat >= lat_lo) & (f_lat <= lat_hi)
    if not band.any():
        raise ValueError(f"no field latitudes in [{lat_lo}, {lat_hi}]")

    out = np.full(tt.size, np.nan)
    for i in range(tt.size):
        target = tt[i] - lead_h
        j = int(np.searchsorted(ft, target))
        # nearest of the two bracketing field times
        cand = [k for k in (j - 1, j) if 0 <= k < ft.size]
        if not cand:
            continue
        k = min(cand, key=lambda k: abs(ft[k] - target))
        if abs(ft[k] - target) > tol_h:
            continue
        rel = (f_lon - L[i] + 180.0) % 360.0 - 180.0
        box = np.abs(rel) <= dlon
        if not box.any():
            continue
        cell = field[k][np.ix_(band, box)]
        if np.isfinite(cell).any():
            out[i] = np.nanmean(cell)
    return out


def cluster_bootstrap_diff(gid_a, val_a, gid_b, val_b, rng, n_boot=2000, pct=(2.5, 97.5)):
    """Cluster bootstrap CI for mean(b) - mean(a), resampling CLUSTERS not observations.

    Trough observations from one wave are autocorrelated (a wave contributes many 6-hourly
    samples), so treating each observation as independent overstates the sample size. This
    resamples whole trajectories (clusters) with replacement and recomputes the difference of
    the two pooled means, so the interval reflects the number of independent waves.

    Estimand note: the pooled means are OBSERVATION-weighted (a long-lived wave contributes
    more observations), which matches a per-trough-observation contrast ("is development at a
    given time and place associated with a moister environment"). For a WAVE-level contrast
    ("do developing waves live in moister air"), collapse to per-trajectory means first and
    pass one value per trajectory; the two estimands can differ and both should be reported.

    A wave can straddle both groups (some of its observations high-response, some low), so the
    resample draws from the UNION of trajectory ids once per iteration; a drawn wave
    contributes its group-a observations to arm a and its group-b observations to arm b. This
    keeps a wave's presence in the two arms tied together rather than resampling the arms
    independently. Iterations that leave an arm empty (possible only with tiny cluster counts)
    are skipped. Returns ``(diff, lo, hi, n_clusters_a, n_clusters_b)``.
    """
    gid_a = np.asarray(gid_a); val_a = np.asarray(val_a, dtype=float)
    gid_b = np.asarray(gid_b); val_b = np.asarray(val_b, dtype=float)

    def _by_cluster(gid):
        idx = defaultdict(list)
        for i, g in enumerate(gid):
            idx[g].append(i)
        return {k: np.asarray(v) for k, v in idx.items()}

    ma = _by_cluster(gid_a)
    mb = _by_cluster(gid_b)
    keys = np.array(sorted(set(ma) | set(mb)))

    diff = val_b.mean() - val_a.mean()
    boot = []
    for _ in range(n_boot):
        pick = keys[rng.integers(0, keys.size, keys.size)]
        a_sel = [ma[k] for k in pick if k in ma]
        b_sel = [mb[k] for k in pick if k in mb]
        if not a_sel or not b_sel:
            continue
        boot.append(val_b[np.concatenate(b_sel)].mean() - val_a[np.concatenate(a_sel)].mean())
    lo, hi = np.percentile(boot, list(pct))
    return float(diff), float(lo), float(hi), len(ma), len(mb)


def terciles(x):
    """Boolean (low, high) masks for the bottom and top terciles of the finite part of x.

    Non-finite entries are False in both masks. Used to split troughs into non-developing
    (low response) and developing (high response).
    """
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    lo_t, hi_t = np.nanpercentile(x[finite], [33.3, 66.7]) if finite.any() else (np.nan, np.nan)
    low = finite & (x <= lo_t)
    high = finite & (x >= hi_t)
    return low, high
