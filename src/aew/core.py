"""Small NCL-compatible helpers shared across the library.

These reproduce the specific NCL built-ins the legacy scripts relied on, with the
same edge-case behavior, so the ported analysis matches the originals.
"""

from __future__ import annotations

import numpy as np

__all__ = ["local_max_1d", "local_min_1d"]


def local_max_1d(x):
    """Indices of strict interior local maxima of a 1-D array.

    Mirrors NCL ``local_max_1d(x, False, 0, 1)`` as used by composite_dates:
    non-cyclic, delta=0, so index ``i`` (1 <= i <= n-2) qualifies when
    ``x[i] > x[i-1]`` and ``x[i] > x[i+1]``. Missing values (NaN) break a run and
    are never selected. Endpoints are never local maxima (cyclic=False).
    """
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return np.empty(0, dtype=int)
    left = x[1:-1] > x[:-2]
    right = x[1:-1] > x[2:]
    interior = left & right & np.isfinite(x[1:-1]) & np.isfinite(x[:-2]) & np.isfinite(x[2:])
    return np.nonzero(interior)[0] + 1


def local_min_1d(x):
    """Indices of strict interior local minima. NCL ``local_min_1d(x, False, 0, 1)``."""
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return np.empty(0, dtype=int)
    left = x[1:-1] < x[:-2]
    right = x[1:-1] < x[2:]
    interior = left & right & np.isfinite(x[1:-1]) & np.isfinite(x[:-2]) & np.isfinite(x[2:])
    return np.nonzero(interior)[0] + 1
