#!/usr/bin/env python
"""Render a demo Fig-4-style basepoint MAP from synthetic data using the real engine.

Builds a synthetic (time, lat, lon) wave field, selects composite dates at a basepoint,
composites the field for the lon-lat v700 contours (composite_xy), bins phase-locked
synthetic cloud systems into a lat-lon CS-anomaly map (map_event_counts), and draws it
with aew.plotting.basepoint_map. Style preview while real data downloads.
"""

import argparse

import numpy as np
import pandas as pd

from aew.composites import composite_xy_preread, map_event_counts
from aew.events import composite_dates, std_threshold
from aew.filtering import lanczos_bandpass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="basepoint_map_demo.png")
    args = ap.parse_args()
    rng = np.random.default_rng(5)

    ndays = 12 * 92
    lats = np.arange(0.0, 25.0, 1.0)
    lons = np.arange(-40.0, 80.0, 1.0)
    t = np.arange(ndays, dtype=float)
    # westward wave, meridionally centered near 12N
    latw = np.exp(-((lats - 12.0) ** 2) / (2 * 6.0 ** 2))
    phase = 2 * np.pi * (t[:, None] / 5.0 + lons[None, :] / 35.0)
    wave = np.sin(phase)[:, None, :] * latw[None, :, None]  # (time, lat, lon)
    v = 3.0 * wave + 0.8 * rng.standard_normal((ndays, lats.size, lons.size))
    time = pd.date_range("1990-06-01", periods=ndays, freq="D")

    base_lat, base_lon = 12.0, 10.0
    bj = int(np.argmin(np.abs(lats - base_lat)))
    bi = int(np.argmin(np.abs(lons - base_lon)))
    filt = lanczos_bandpass(v, 2, 10, obs_per_day=1, axis=0)
    base_series = filt[:, bj, bi]
    valid = np.isfinite(base_series)
    thr = std_threshold(base_series[valid], 1.5)
    cd = composite_dates(base_series, thr, time=time.values)
    print(f"composite dates: {len(cd)}  threshold: {thr:.3f}")

    # v700 contours: lon-lat composite of the unfiltered field at lag 0
    comp = composite_xy_preread(v, time.values, cd.time, lags=[0])
    contour = comp.values[0]  # (lat, lon)

    # CS shading: cloud systems enhanced in the trough, binned to a lat-lon map at lag 0
    cs_time, cs_lon, cs_lat = [], [], []
    for it in range(ndays):
        field = np.clip(-filt[it], 0, None)  # trough strength (lat, lon)
        if field.sum() == 0 or not np.isfinite(field).any():
            continue
        flat = field.ravel() / field.sum()
        n = rng.poisson(10)
        if n:
            idx = rng.choice(flat.size, size=n, p=flat)
            jj, ii = np.unravel_index(idx, field.shape)
            cs_lat.extend(lats[jj]); cs_lon.extend(lons[ii])
            cs_time.extend([time.values[it]] * n)
    cs_time = np.array(cs_time, dtype="datetime64[ns]")
    cs_lat = np.array(cs_lat); cs_lon = np.array(cs_lon)

    counts = map_event_counts(cd.time, cs_time, cs_lon, cs_lat, lons, lats, lag=0.0,
                              half_window=0.5)
    shaded = counts - np.nanmean(counts)  # simple anomaly for the demo

    from aew.plotting import basepoint_map, save
    prov = f"basepoint {base_lat}N/{base_lon}E\nNumber of dates: {len(cd)}\n(SYNTHETIC demo)"
    fig, ax = basepoint_map(
        shaded, lons, lats, contour=contour, base_lon=base_lon, base_lat=base_lat,
        extent=(-40, 80, 0, 24), title="CS anomaly + v700 (demo)",
        shaded_label="CS count anomaly", provenance=prov,
    )
    print("wrote", save(fig, args.out))


if __name__ == "__main__":
    main()
