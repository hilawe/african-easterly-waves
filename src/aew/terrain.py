"""Terrain-validity masking for pressure-level fields (REPAIR_SPEC.md R2).

A grid point is valid for pressure level p (hPa) at time t when the instantaneous
surface pressure at that point and time exceeds p + delta. Masking happens at the data
level, invalid points become NaN, so every downstream consumer inherits the rule by NaN
propagation: box means see missing points, ``Gridded`` bilinear sampling returns NaN
whenever any stencil corner is invalid (the spec's stencil rule), and a trajectory wind
evaluation at a masked corner makes the parcel NaN from that step onward into earlier
times (``back_trajectories`` holds NaN once a position is NaN), which is the required
invalid-from-first-bad-evaluation behavior including intermediate Runge-Kutta stages.
"""

import numpy as np

# primary buffer (hPa) above the level a surface must clear; roughly one model
# half-level (~200 m), guarding near-surface extrapolation artifacts. Sensitivities at
# 0 and 50 hPa are part of the spec.
DELTA_HPA = 25.0


def apply_terrain_mask(values, sp_pa, level_hpa, delta_hpa=DELTA_HPA):
    """Return ``values`` with points failing the validity rule set to NaN.

    ``values`` and ``sp_pa`` must be aligned elementwise on the same (time, lat, lon)
    axes; ``sp_pa`` is instantaneous surface pressure in Pa (the ERA5 unit).
    """
    v = np.asarray(values, dtype=float)
    sp = np.asarray(sp_pa, dtype=float)
    if v.shape != sp.shape:
        raise ValueError(f"values {v.shape} and surface pressure {sp.shape} are not "
                         "aligned; both must be (time, lat, lon) on one grid")
    return np.where(sp > (level_hpa + delta_hpa) * 100.0, v, np.nan)


def valid_fraction(sp_pa, level_hpa, delta_hpa=DELTA_HPA):
    """Fraction of points valid for ``level_hpa``, per time step (attrition report)."""
    sp = np.asarray(sp_pa, dtype=float)
    ok = sp > (level_hpa + delta_hpa) * 100.0
    return ok.reshape(ok.shape[0], -1).mean(axis=1)


def masked_box_mean(field, keep_min=0.5):
    """Mean over the last axes of a (..., ny, nx) box with the at-least-half rule.

    Returns NaN where fewer than ``keep_min`` of the box's points are finite, so a box
    dominated by masked terrain reports missing rather than a shrunken-footprint mean.
    """
    f = np.asarray(field, dtype=float)
    flat = f.reshape(f.shape[0], -1) if f.ndim > 1 else f.reshape(1, -1)
    finite = np.isfinite(flat)
    frac = finite.mean(axis=1)
    with np.errstate(invalid="ignore"):
        means = np.where(frac >= keep_min, np.nanmean(np.where(finite, flat, np.nan),
                                                      axis=1), np.nan)
    return means if f.ndim > 1 else float(means[0])


def mask_level_inplace(field, sp_pa, level_hpa, delta_hpa=DELTA_HPA):
    """NaN out invalid points of ``field`` in place (the memory-lean driver path).

    Same rule as :func:`apply_terrain_mask` without allocating a masked copy; the
    driver holds multi-gigabyte season stacks, so masking mutates the loaded array.
    """
    f = np.asarray(field)
    sp = np.asarray(sp_pa, dtype=float)
    if f.shape != sp.shape:
        raise ValueError(f"field {f.shape} and surface pressure {sp.shape} are not "
                         "aligned; both must be (time, lat, lon) on one grid")
    # ~(sp > thr) rejects low, equal, AND nonfinite surface pressure (a bare
    # sp <= thr comparison is False for NaN and would leave the point valid)
    f[~(sp > (level_hpa + delta_hpa) * 100.0)] = np.nan
    return f
