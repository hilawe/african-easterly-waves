#!/usr/bin/env python
"""Render a demo Fig-2-style Hovmoller from SYNTHETIC data using the real engine.

This is a style/plumbing preview while the 24-year ERA5 download finishes. It builds a
westward-propagating wave, filters it, selects composite dates at a basepoint, composites
the unfiltered field for the contours, and bins phase-locked synthetic cloud systems for
the shading -- then draws it with aew.plotting.hovmoller. The real figure swaps the
synthetic arrays for ERA5 v700 + the CS/CT tracks; the plotting call is identical.
"""

import argparse

import numpy as np
import pandas as pd

from aew.composites import (anomaly, composite_xt_preread, hovmoller_event_counts,
                            lag_axis)
from aew.events import composite_dates, std_threshold
from aew.filtering import lanczos_bandpass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hovmoller_demo.png")
    args = ap.parse_args()

    rng = np.random.default_rng(3)
    ndays = 24 * 92  # ~24 JAS seasons worth of days
    lons = np.arange(-40.0, 80.0, 2.0)
    t = np.arange(ndays, dtype=float)
    # westward wave: phase advances with +t and +lon -> crests move toward -lon (west)
    period, wlen = 5.0, 35.0
    wave = np.sin(2 * np.pi * (t[:, None] / period + lons[None, :] / wlen))
    v = 3.0 * wave + 1.2 * rng.standard_normal((ndays, lons.size))
    time = pd.date_range("1984-06-01", periods=ndays, freq="D")

    base_lon = 0.0
    bi = int(np.argmin(np.abs(lons - base_lon)))
    filt = lanczos_bandpass(v, 2, 10, obs_per_day=1, axis=0)
    base_series = filt[:, bi]
    valid = np.isfinite(base_series)
    thr = std_threshold(base_series[valid], 1.5)
    cd = composite_dates(base_series, thr, time=time.values)
    print(f"composite dates: {len(cd)}  threshold: {thr:.3f}")

    # contour field: composite of the UNFILTERED wave about those dates
    comp = composite_xt_preread(v, time.values, cd.time, -6, 6, 1)

    # shaded: cloud systems enhanced in the trough (v<0), so counts track the wave phase
    cs_time, cs_lon, cs_lat = [], [], []
    for it in range(ndays):
        # probability of a CS at each lon proportional to trough strength
        p = np.clip(-filt[it], 0, None)
        if not np.isfinite(p).any() or p.sum() == 0:
            continue
        p = p / p.sum()
        n = rng.poisson(6)
        if n:
            idx = rng.choice(lons.size, size=n, p=p)
            cs_lon.extend(lons[idx])
            cs_time.extend([time.values[it]] * n)
            cs_lat.extend(rng.uniform(6, 14, n))
    cs_time = np.array(cs_time, dtype="datetime64[ns]")
    cs_lon = np.array(cs_lon)
    cs_lat = np.array(cs_lat)

    lag = lag_axis(-6, 6, 1).astype(float)
    counts = hovmoller_event_counts(cd.time, cs_time, cs_lon, lons, lag,
                                    cs_lat=cs_lat, min_lat=5.0, max_lat=15.0)
    shaded = anomaly(counts, "anomaly")

    from aew.plotting import hovmoller, save
    prov = (f"Number of dates: {len(cd)}\nStd. Dev. threshold: 1.5\n"
            f"Bin degree scale: 2\n(SYNTHETIC demo)")
    fig, ax = hovmoller(
        shaded, lons, lag, contour=comp.values, contour_lon=lons, contour_lag=comp.lag,
        base_lon=base_lon, title="Averaged 5N-15N  (demo)",
        shaded_label="CS count anomaly", lon_range=(-40, 80), provenance=prov,
    )
    out = save(fig, args.out)
    print("wrote", out)


if __name__ == "__main__":
    main()
