"""End-to-end smoke test of the Figure-2 compute chain on synthetic data.

filter (Lanczos band-pass) -> composite_dates (peaks of basepoint series) ->
composite_xt_preread (lag-lon composite for the v700 contours) +
hovmoller_event_counts (CS-count shading) -> anomaly.

No real data; this checks the pieces connect and produce physically sensible structure
(the lag-0 composite peaks at the basepoint; CS counts are enhanced at the basepoint).
"""

import numpy as np
import pandas as pd

from aew.composites import (
    anomaly,
    composite_xt_preread,
    hovmoller_event_counts,
    lag_axis,
)
from aew.events import composite_dates, std_threshold
from aew.filtering import lanczos_bandpass


def _synthetic_wave(ndays=900, lons=None, period=5.0):
    if lons is None:
        lons = np.arange(0.0, 120.0, 4.0)
    t = np.arange(ndays, dtype=float)
    # westward-propagating wave v(t, lon): phase increases with t and lon
    phase = 2 * np.pi * (t[:, None] / period + lons[None, :] / 40.0)
    wave = np.sin(phase)
    rng = np.random.default_rng(7)
    noise = 0.3 * rng.standard_normal((ndays, lons.size))
    return wave + noise, lons, t


def test_full_fig2_chain_runs_and_is_sensible():
    data2d, lons, t = _synthetic_wave()
    time = pd.date_range("2000-01-01", periods=data2d.shape[0], freq="D")

    # band-pass filter along time (axis 0), then take the basepoint column
    filt = lanczos_bandpass(data2d, period_low=2, period_high=10, obs_per_day=1, axis=0)
    base_lon_idx = 5  # basepoint at lons[5] = 20 E
    base_series = filt[:, base_lon_idx]

    valid = np.isfinite(base_series)
    thr = std_threshold(base_series[valid], 1.5)
    cd = composite_dates(base_series, thr, time=time.values)
    assert len(cd) > 10  # peaks above 1.5 sigma over 900 days

    # lag-lon composite of the (unfiltered) field about those dates
    comp = composite_xt_preread(data2d, time.values, cd.time, -6, 6, 1)
    assert comp.values.shape == (13, lons.size)
    lag0 = np.where(comp.lag == 0)[0][0]
    # at lag 0 the composite should peak at (or adjacent to) the basepoint longitude
    peak_lon = np.nanargmax(comp.values[lag0])
    assert abs(peak_lon - base_lon_idx) <= 1

    # CS field: a uniform daily background at the basepoint (flat in lag -> zero
    # anomaly) PLUS an extra CS on each event day (phase-locked -> a lag-0 spike).
    bg_time = time.values
    bg_lon = np.full(time.size, lons[base_lon_idx])
    cs_time = np.concatenate([bg_time, cd.time])
    cs_lon = np.concatenate([bg_lon, np.full(cd.time.size, lons[base_lon_idx])])
    cs_lat = np.full(cs_time.size, 10.0)
    lag_centers = lag_axis(-6, 6, 1).astype(float)
    counts = hovmoller_event_counts(
        cd.time, cs_time, cs_lon, lons, lag_centers,
        cs_lat=cs_lat, min_lat=5.0, max_lat=15.0,
    )
    assert counts.shape == (13, lons.size)

    anom = anomaly(counts, "anomaly")
    assert anom.shape == counts.shape
    # the phase-locked extra CS makes lag 0 at the basepoint a positive anomaly
    i0 = np.where(lag_centers == 0)[0][0]
    assert anom[i0, base_lon_idx] > 0
    # and lag 0 is the maximum-anomaly lag at the basepoint
    assert np.nanargmax(anom[:, base_lon_idx]) == i0
