"""2D binning of scattered points into a (glon x glat) grid.

Faithful port of the two NCL/Fortran routines the PhD scripts relied on. In the
AEW Hovmoller analysis the "points" are individual convective systems located at
(longitude, lag) and the grid axes are (longitude center array, lag center array);
``z`` is usually an array of ones so the binned sum is a count.

Two variants, because the published figures and the legacy code use different ones
and we need to compare them (see docs/VALIDATION_TARGETS.md, the bin_sum hazard):

- ``variant="fixed"``  -> DATABINSUM3, the corrected Fortran routine (bindata_ncl.f)
  that the Hovmoller scripts actually call via ``bin_sum_so``. It bounds-checks each
  point against the grid edges BEFORE computing its bin index. This produced the
  published Hovmollers, so DATABINSUM3 (not the NCL ``bin_sum_other`` variant, which
  adds an extra exact-upper-edge rescue) is the authoritative "fixed" reference here.
- ``variant="buggy"``  -> the original ``bin_sum`` (emulated by the legacy
  bin_sum_ncl.ncl with APPLY_THE_FIX=False). It computes the index first and only
  checks that the index is in range, so points just outside the domain fold into the
  edge bins. Implemented for comparison only.

The two differ only for points at or outside the domain edges; interior points bin
identically.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bin_sum"]


def bin_sum(glon, glat, x, y, z=None, variant="fixed", fill_value=None):
    """Bin scattered points (x, y, z) onto grid centers (glon, glat).

    Parameters
    ----------
    glon, glat : 1-D array_like
        Bin-CENTER coordinates along each axis, assumed evenly spaced. The AEW
        workflow always uses ascending centers (glon = lon, glat = lag = -6..6).
        Descending ``glat`` is also handled correctly here; note this is a deliberate
        correction, because the literal DATABINSUM3 Fortran has a broken descending
        path (its ordered bound test would reject all points). Results are identical
        to the Fortran for the ascending axes used in practice.
    x, y : 1-D array_like
        Coordinates of each point (x paired with glon, y paired with glat).
    z : 1-D array_like, optional
        Value to accumulate per point. Defaults to ones (so gbin == count).
    variant : {"fixed", "buggy"}
        Which historical routine to reproduce. "fixed" is the published one.
    fill_value : float, optional
        Points with z == fill_value are skipped (matches the Fortran ZMSG test).

    Returns
    -------
    gbin : ndarray, shape (nlat, nlon)
        Summed z per bin. Indexed [lat, lon] to match the NCL arrays used by the
        scripts (e.g. ans(lag, lon)).
    gcnt : ndarray, shape (nlat, nlon), int
        Count of points per bin.
    """
    glon = np.asarray(glon, dtype=float)
    glat = np.asarray(glat, dtype=float)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if z is None:
        z = np.ones(x.shape, dtype=float)
    else:
        if isinstance(z, np.ma.MaskedArray):
            z = np.ma.asarray(z, dtype=float).filled(np.nan)  # float first: int+nan errors
        z = np.asarray(z, dtype=float).ravel()
    if not (x.shape == y.shape == z.shape):
        raise ValueError("x, y, z must have the same length")

    nlon = glon.size
    nlat = glat.size
    gbin = np.zeros((nlat, nlon), dtype=float)
    gcnt = np.zeros((nlat, nlon), dtype=int)

    dlon = abs(glon[1] - glon[0])
    dlat = abs(glat[1] - glat[0])

    if variant == "fixed":
        _bin_fixed(gbin, gcnt, glon, glat, x, y, z, dlon, dlat, fill_value)
    elif variant == "buggy":
        _bin_buggy(gbin, gcnt, glon, glat, x, y, z, dlon, dlat, fill_value)
    else:
        raise ValueError("variant must be 'fixed' or 'buggy'")
    return gbin, gcnt


def _valid(z, fill_value):
    good = np.isfinite(z)
    if fill_value is not None:
        good &= z != fill_value
    return good


def _accumulate(gbin, gcnt, nl, ml, z, nlat, nlon, sel):
    """Vectorized scatter-add into the (nlat, nlon) grid for the selected points."""
    nl = nl[sel]
    ml = ml[sel]
    zz = z[sel]
    inrange = (nl >= 0) & (nl < nlat) & (ml >= 0) & (ml < nlon)
    nl = nl[inrange]
    ml = ml[inrange]
    zz = zz[inrange]
    np.add.at(gbin, (nl, ml), zz)
    np.add.at(gcnt, (nl, ml), 1)


def _bin_fixed(gbin, gcnt, glon, glat, x, y, z, dlon, dlat, fill_value):
    """DATABINSUM3 (bindata_ncl.f): bounds-check then floor index. Vectorized.

    Fortran (1-based):
        iflag = sign(glat[2]-glat[1])
        glatbnd1 = glat[1] - iflag*dlat/2 ; glatbnd2 = glat[NLAT] + iflag*dlat/2
        glonbnd1 = glon[1] - dlon/2       ; glonbnd2 = glon[MLON] + dlon/2
        if glatbnd1 <= y <= glatbnd2 and glonbnd1 <= x <= glonbnd2:
            nl = int(abs((y-glatbnd1)/dlat)) + 1   ; ml = int(abs((x-glonbnd1)/dlon)) + 1
            if in range: accumulate
    int(abs(...)) truncates toward zero; for the non-negative offsets here that equals
    floor, so np.floor(...).astype(int) reproduces it. 0-based here.
    """
    nlat = glat.size
    nlon = glon.size
    iflag = 1.0 if (glat[1] - glat[0]) > 0 else -1.0
    glatbnd1 = glat[0] - iflag * dlat / 2.0
    glatbnd2 = glat[nlat - 1] + iflag * dlat / 2.0
    glonbnd1 = glon[0] - dlon / 2.0
    glonbnd2 = glon[nlon - 1] + dlon / 2.0
    # Normalize so the comparison works regardless of lat orientation (deliberate
    # correction; the literal Fortran descending path is broken -- see module docstring).
    lat_lo, lat_hi = min(glatbnd1, glatbnd2), max(glatbnd1, glatbnd2)

    sel = _valid(z, fill_value)
    sel &= (y >= lat_lo) & (y <= lat_hi) & (x >= glonbnd1) & (x <= glonbnd2)

    nl = np.floor(np.abs((y - glatbnd1) / dlat)).astype(np.int64)
    ml = np.floor(np.abs((x - glonbnd1) / dlon)).astype(np.int64)
    _accumulate(gbin, gcnt, nl, ml, z, nlat, nlon, sel)


def _bin_buggy(gbin, gcnt, glon, glat, x, y, z, dlon, dlat, fill_value):
    """Original bin_sum (bin_sum_ncl.ncl with APPLY_THE_FIX=False). Vectorized.

    Index is computed first from the lower-edge offset using abs()+truncation; only
    the index range is checked. Points just outside the domain fold onto edge bins
    because abs() maps small negative offsets to small positive indices.
    """
    nlat = glat.size
    nlon = glon.size
    glatbnd = glat[0] - dlat / 2.0
    glonbnd = glon[0] - dlon / 2.0

    sel = _valid(z, fill_value)
    nl = np.floor(np.abs((y - glatbnd) / dlat)).astype(np.int64)
    ml = np.floor(np.abs((x - glonbnd) / dlon)).astype(np.int64)
    _accumulate(gbin, gcnt, nl, ml, z, nlat, nlon, sel)
