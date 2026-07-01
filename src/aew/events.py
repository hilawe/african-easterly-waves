"""Composite event-date selection.

Faithful port of Carl Schreck's ``composite_dates`` (NCL), confirmed from the
surviving source on the host:

    if thresh >= 0:
        max_ind   = local_max_1d(baseData, False, 0, 1)
        threshInd = max_ind[ baseData[max_ind] >= thresh ]
    else:
        max_ind   = local_min_1d(baseData, False, 0, 1)
        threshInd = max_ind[ baseData[max_ind] <= thresh ]
    return baseData.time[threshInd]   (with amplitude attached)

So composite dates are the LOCAL MAXIMA (peaks) of the filtered base series that
exceed the threshold, NOT every day above threshold. This is what makes the event
counts modest (e.g. 272 peaks above 2*sigma over 24 JAS seasons at 10N/0E); see
docs/VALIDATION_TARGETS.md. The usual threshold is ``thresh = n_sigma * std(baseData)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import local_max_1d, local_min_1d

__all__ = ["composite_dates", "CompositeDates"]


@dataclass
class CompositeDates:
    """Result of composite_dates."""

    index: np.ndarray  # integer indices into the base series
    time: np.ndarray  # the times at those indices (same dtype as input time)
    amp: np.ndarray  # base-series value (amplitude) at each selected peak

    def __len__(self):
        return self.index.size


def composite_dates(base_data, thresh, time=None):
    """Select composite event dates from a 1-D base time series.

    Parameters
    ----------
    base_data : 1-D array_like
        The base series, typically 2-10 day bandpass filtered v700 averaged at a
        basepoint.
    thresh : float
        Threshold. If >= 0, positive peaks (local maxima) with value >= thresh are
        selected; if < 0, negative peaks (local minima) with value <= thresh.
        Commonly ``n_sigma * np.std(base_data)``.
    time : 1-D array_like, optional
        Times aligned with base_data. If omitted, integer indices are returned as
        the time.

    Returns
    -------
    CompositeDates
    """
    base = np.asarray(base_data, dtype=float)
    if time is None:
        time = np.arange(base.size)
    time = np.asarray(time)
    if time.shape[0] != base.shape[0]:
        raise ValueError("time and base_data must have the same length")

    if thresh >= 0:
        peaks = local_max_1d(base)
        keep = peaks[base[peaks] >= thresh]
    else:
        peaks = local_min_1d(base)
        keep = peaks[base[peaks] <= thresh]

    return CompositeDates(index=keep, time=time[keep], amp=base[keep])


def std_threshold(base_data, n_sigma):
    """Convenience: ``n_sigma * np.std(base_data)`` matching NCL ``stddev`` (population, ddof=0).

    NOTE: NCL's ``stddev`` ignores missing values and uses N-1 (sample) normalization.
    This is a PARITY HOTSPOT (docs/VALIDATION_TARGETS.md): verify ddof against the
    target thresholds before trusting counts. Default here is ddof=1 to match NCL.
    """
    base = np.asarray(base_data, dtype=float)
    base = base[np.isfinite(base)]
    return n_sigma * np.std(base, ddof=1)
