"""Lanczos bandpass filtering for AEW extraction.

Ports the two NCL calls used everywhere in the legacy filter scripts
(e.g. legacy_ncl/filter/v_simple_bandpass.ncl):

    weights = filwgts_lanczos( nWgts, 2, freq(1), freq(0), 1 )   ; band-pass
    outData = wgt_runave_n_Wrap( inData, weights, 0, 0 )         ; apply, ends -> missing

For the AEW work the base field is daily-averaged v700 (obsPerDay = 1), period band
2-10 days, bufferDays = 30 so nWgts = 2*bufferDays*obsPerDay + 1 = 61. The exact
weights are validated against NCL's own output in the test suite
(tests/data/ncl_lanczos_weights_61_0.1_0.5.txt).
"""

from __future__ import annotations

import numpy as np

__all__ = ["filwgts_lanczos", "wgt_runave", "lanczos_bandpass", "bandpass_weights", "wk_bandpass"]


def filwgts_lanczos(nwt, ihp, fca, fcb, nsigma=1):
    """Lanczos filter weights, matching NCL ``filwgts_lanczos``.

    Parameters
    ----------
    nwt : int
        Number of weights (odd). nwt = 2*order + 1.
    ihp : {0, 1, 2}
        0 low-pass (cutoff fca), 1 high-pass (cutoff fca), 2 band-pass (fca..fcb).
    fca, fcb : float
        Cutoff frequencies in cycles per time step. For band-pass, fca < fcb.
        In the legacy call ``filwgts_lanczos(61, 2, freq(1), freq(0), 1)`` with
        freq = 1/period = (0.5, 0.1), so fca=0.1 (10-day), fcb=0.5 (2-day).
    nsigma : int
        Power of the Lanczos (sigma) taper. Legacy uses 1.

    Returns
    -------
    ndarray, shape (nwt,)
        Symmetric weight array centered on the middle element.
    """
    nwt = int(nwt)
    if nwt % 2 == 0:
        raise ValueError("nwt should be odd")
    order = (nwt - 1) // 2
    k = np.arange(1, order + 1, dtype=float)
    # Lanczos sigma taper; sin(pi)=0 at k=order so endpoint weights vanish.
    sigma = (np.sin(np.pi * k / order) * order / (np.pi * k)) ** nsigma

    def lowpass(fc):
        c = np.empty(order + 1)
        c[0] = 2.0 * fc
        c[1:] = (np.sin(2.0 * np.pi * fc * k) / (np.pi * k)) * sigma
        return c

    if ihp == 0:
        c = lowpass(fca)
    elif ihp == 1:
        c = -lowpass(fca)
        c[0] += 1.0
    elif ihp == 2:
        c = lowpass(fcb) - lowpass(fca)
    else:
        raise ValueError("ihp must be 0, 1, or 2")

    return np.concatenate([c[:0:-1], c])


def bandpass_weights(period_low, period_high, obs_per_day=1, buffer_days=30, nsigma=1):
    """Convenience wrapper reproducing the legacy band-pass setup.

    period_low/period_high are in DAYS (e.g. 2 and 10). Returns the weight array.
    """
    nwt = 2 * buffer_days * obs_per_day + 1
    period = np.array([period_low, period_high], dtype=float)
    freq = 1.0 / (period * obs_per_day)  # (0.5, 0.1) for (2,10) day at obs_per_day=1
    # legacy passes (freq(1), freq(0)) = (low-freq cutoff, high-freq cutoff)
    return filwgts_lanczos(nwt, 2, freq[1], freq[0], nsigma)


def wgt_runave(x, wgt, axis=0, fill_value=None):
    """Weighted running average, matching NCL ``wgt_runave_n_Wrap(x, wgt, 0, n)``.

    Weights are applied AS GIVEN (no normalization) -- correct for digital filter
    weights, which sum to ~0 for a band-pass. End points that lack a full window
    (the first and last ``len(wgt)//2`` along ``axis``) are set to NaN, matching the
    NCL ``kopt=0`` boundary option. Any window containing a missing value yields a
    missing (NaN) result, as in NCL.

    ``axis`` defaults to 0 because the legacy scripts call
    ``wgt_runave_n_Wrap(inData, weights, 0, 0)`` (filter along the leading TIME
    dimension of a (time, lat, lon) field).

    Missing values: NaN inputs propagate to NaN. A ``numpy.ma.MaskedArray`` mask is
    honored, and ``fill_value`` (e.g. an NCL ``_FillValue`` sentinel like -999) is
    converted to NaN before filtering.
    """
    if isinstance(x, np.ma.MaskedArray):
        x = np.ma.asarray(x, dtype=float).filled(np.nan)  # float first: int+nan errors
    x = np.asarray(x, dtype=float)
    if fill_value is not None:
        x = np.where(x == fill_value, np.nan, x)
    wgt = np.asarray(wgt, dtype=float)
    nwt = wgt.size
    half = nwt // 2

    x = np.moveaxis(x, axis, -1)
    out = np.full_like(x, np.nan)
    n = x.shape[-1]
    if n >= nwt:
        # Sliding-window dot product with the (reversed-symmetric) weights.
        # weights are symmetric so orientation does not matter, but use as-is.
        windows = np.lib.stride_tricks.sliding_window_view(x, nwt, axis=-1)
        conv = np.tensordot(windows, wgt, axes=([-1], [0]))
        out[..., half : n - half] = conv
    return np.moveaxis(out, -1, axis)


def lanczos_bandpass(
    x, period_low=2.0, period_high=10.0, obs_per_day=1, buffer_days=30, nsigma=1,
    axis=0, fill_value=None
):
    """Full band-pass filter: build Lanczos weights and apply along ``axis``.

    ``axis`` defaults to 0 (the time dimension), matching the legacy NCL call.
    Returns the filtered array with NaN at the unsmoothed ends (buffer_days*obs_per_day
    points at each end).
    """
    wgt = bandpass_weights(period_low, period_high, obs_per_day, buffer_days, nsigma)
    return wgt_runave(x, wgt, axis=axis, fill_value=fill_value)


def wk_bandpass(
    data, obs_per_day, period_min, period_max, wavenum_min, wavenum_max,
    time_axis=0, lon_axis=-1, taper_frac=0.05, detrend=True,
    lon=None, allow_regional=False,
):
    """Wavenumber-frequency (space-time) band-pass filter, Wheeler-Kiladis / Frank-Roundy
    style. Isolates zonally propagating waves in a (wavenumber, period) window, keeping
    propagation direction via the sign convention below.

    This is the filter behind the published AEW base series (v700.anom.waves.nc `td`:
    westward wavenumbers -20..0, period 2.5-10 day). NCL did this with kf_filter; this is
    the clean-room equivalent for running from ERA5 without the original wave file.

    Sign convention (verified by the synthetic tests): a SIGNED zonal wavenumber is defined
    so that WESTWARD-propagating waves have NEGATIVE wavenumber (Frank-Roundy "-20..0").
    Pass wavenum_min/max as that signed wavenumber, e.g. (-20, 0) for westward TD/MRG, or
    (1, 14) for eastward Kelvin.

    Parameters
    ----------
    data : ndarray with a time axis and a (cyclic, full 360 deg) longitude axis.
    obs_per_day : samples per day along the time axis (e.g. 4 for 6-hourly).
    period_min, period_max : period band in days (e.g. 2.5, 10).
    wavenum_min, wavenum_max : signed zonal-wavenumber band (westward negative).
    time_axis, lon_axis : axis indices.
    taper_frac : cosine-taper fraction at each end of the time series. Default 0.05 to
        match NCL kf_filter (which always tapers 5%).
    detrend : remove a linear trend (not just the mean) along time before the FFT, as NCL
        kf_filter does. Default True.
    lon : 1-D longitude coordinate (optional). If given, the function verifies the grid is
        a full cyclic ~360 deg circle; a regional subset makes the zonal FFT a regional
        (not global zonal) wavenumber decomposition, which is usually a mistake.
    allow_regional : set True to bypass the global-longitude check on purpose.

    Returns
    -------
    filtered array, same shape as ``data`` (real).
    """
    from scipy.signal import detrend as _detrend

    data = np.asarray(data, dtype=float)
    x = np.moveaxis(data, (time_axis, lon_axis), (0, -1))
    nt, nlon = x.shape[0], x.shape[-1]

    if lon is not None and not allow_regional:
        lon = np.asarray(lon, dtype=float)
        dx = np.median(np.abs(np.diff(lon)))
        span = nlon * dx  # full circle if the grid wraps
        if abs(span - 360.0) > 1.5 * dx:
            raise ValueError(
                f"wk_bandpass needs a global cyclic longitude grid for a true zonal "
                f"wavenumber FFT; got span ~{span:.0f} deg over {nlon} points. Pass a "
                f"global field, or allow_regional=True to override.")

    # NCL kf_filter preprocessing: linear detrend along time, then 5% cosine taper.
    xw = _detrend(x, axis=0, type="linear") if detrend else x - x.mean(axis=0, keepdims=True)

    if taper_frac and taper_frac > 0:
        ntap = max(1, int(taper_frac * nt))
        w = np.ones(nt)
        ramp = 0.5 * (1 - np.cos(np.pi * (np.arange(ntap) + 1) / (ntap + 1)))
        w[:ntap] = ramp
        w[-ntap:] = ramp[::-1]
        xw = xw * w.reshape((nt,) + (1,) * (xw.ndim - 1))

    # 2D FFT over time (axis 0) and longitude (last axis)
    F = np.fft.fft2(xw, axes=(0, -1))
    freq = np.fft.fftfreq(nt, d=1.0 / obs_per_day)          # cycles/day, signed
    wn = np.fft.fftfreq(nlon, d=1.0 / nlon)                 # integer wavenumbers, signed
    wn = np.rint(wn).astype(int)

    fgrid = freq.reshape((nt,) + (1,) * (xw.ndim - 1))      # (nt,1,...,1)
    # broadcast wavenumber over the longitude (last) axis
    wshape = (1,) * (xw.ndim - 1) + (nlon,)
    wgrid = wn.reshape(wshape)

    # period band on |freq|
    with np.errstate(divide="ignore"):
        period = 1.0 / np.abs(fgrid)
    inband_p = (period >= period_min) & (period <= period_max)

    # Signed wavenumber with westward NEGATIVE. In numpy's inverse convention
    # x ~ sum X exp(+2pi i(m x/N + f t)), a constant-phase line moves westward (dx/dt<0)
    # when f and m have the SAME sign (m*f > 0). Both members of a real wave's conjugate
    # pair share that sign product, so this keeps/drops whole pairs (output stays real).
    # swn = -|m| for westward (m*f>0), +|m| for eastward (m*f<0), 0 for the zonal mean.
    swn = -np.sign(fgrid * wgrid) * np.abs(wgrid)
    inband_k = (swn >= wavenum_min) & (swn <= wavenum_max)

    mask = inband_p & inband_k
    Ffilt = F * mask
    out = np.fft.ifft2(Ffilt, axes=(0, -1)).real
    out = np.moveaxis(out, (0, -1), (time_axis, lon_axis))
    return out
