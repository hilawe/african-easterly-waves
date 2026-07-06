"""Plotting for the AEW analysis (Hovmoller now; maps later via cartopy).

Reproduces the published Hovmoller style (Semunegus et al. 2017, Figs 2-3; dissertation
Ch. 4-5): shaded CS/CT count anomaly with the unfiltered-v700 composite overlaid as
red (southerly/positive) and blue (northerly/negative) contours, longitude on x, lag in
days on a REVERSED y-axis (lag -6 at top, +6 at bottom, as in the NCL trYReverse plots),
a green basepoint marker at lag 0, and a horizontal colorbar.

Pure rendering: it takes already-computed arrays (from aew.composites) and draws them,
so computation and plotting stay separate (intermediate NetCDF can be re-plotted).
"""

from __future__ import annotations

import numpy as np

__all__ = ["hovmoller", "basepoint_map", "save", "panel_label"]


def panel_label(ax, letter, size=13):
    """Bold (a)/(b)/(c) marker in the panel's upper-left corner, on a semi-transparent
    white rounded box, the AMS published-journal convention. The label sits just inside
    the axes (1.5 percent from the left, 4 percent down) so it reads as a boundary marker
    rather than title text; put the panel's descriptive title separately, without an
    (a)/(b) prefix. ``size`` is the matplotlib point size; scale it up for wide figures
    that render reduced on the page (about target_rendered * figure_width_in / 6.5)."""
    ax.text(0.015, 0.96, f"({letter})", transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75,
                      edgecolor="none"))


def hovmoller(
    shaded,
    lon,
    lag,
    contour=None,
    contour_lon=None,
    contour_lag=None,
    base_lon=None,
    shaded_levels=None,
    contour_levels=None,
    cmap="RdBu_r",
    title=None,
    shaded_label="CS count anomaly",
    lon_range=None,
    provenance=None,
    ax=None,
):
    """Draw a longitude-lag Hovmoller.

    Parameters
    ----------
    shaded : 2-D array (nlag, nlon)
        The shaded field (e.g. CS/CT count anomaly from aew.composites.anomaly).
    lon, lag : 1-D arrays
        Coordinates for ``shaded``. lag in days.
    contour : 2-D array, optional
        Overlaid contour field (e.g. unfiltered-v700 lag composite). If its grid differs
        from ``shaded``, pass contour_lon/contour_lag.
    base_lon : float, optional
        Basepoint longitude; a green star is drawn at (base_lon, lag=0).
    shaded_levels, contour_levels : array-like, optional
        Explicit contour levels. Defaults are derived from the data.
    cmap : str
        Diverging colormap for the shading (RdBu_r: blue negative, red positive).
    lon_range : (lo, hi), optional
        x-axis limits (e.g. (-40, 80)).
    provenance : str, optional
        Text drawn in the lower-right (n dates, threshold, bin scale, ...).

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt

    lon = np.asarray(lon, dtype=float)
    lag = np.asarray(lag, dtype=float)
    shaded = np.asarray(shaded, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 7.0))
    else:
        fig = ax.figure

    if shaded_levels is None:
        amax = np.nanmax(np.abs(shaded))
        amax = 1.0 if (not np.isfinite(amax) or amax == 0) else amax
        shaded_levels = np.linspace(-amax, amax, 21)

    cf = ax.contourf(
        lon, lag, shaded, levels=shaded_levels, cmap=cmap, extend="both"
    )

    if contour is not None:
        clon = lon if contour_lon is None else np.asarray(contour_lon, dtype=float)
        clag = lag if contour_lag is None else np.asarray(contour_lag, dtype=float)
        contour = np.asarray(contour, dtype=float)
        if contour_levels is None:
            cmax = np.nanmax(np.abs(contour))
            cmax = 1.0 if (not np.isfinite(cmax) or cmax == 0) else cmax
            step = cmax / 6.0
            contour_levels = np.arange(-6, 7) * step
        contour_levels = np.asarray(contour_levels, dtype=float)
        neg = contour_levels[contour_levels < 0]
        pos = contour_levels[contour_levels > 0]
        if neg.size:
            ax.contour(clon, clag, contour, levels=neg, colors="blue",
                       linewidths=0.7, linestyles="dashed")
        if pos.size:
            ax.contour(clon, clag, contour, levels=pos, colors="red",
                       linewidths=0.7, linestyles="solid")

    if base_lon is not None:
        ax.plot([base_lon], [0.0], marker="*", markersize=16, color="lime",
                markeredgecolor="black", markeredgewidth=0.6, zorder=5)

    ax.invert_yaxis()  # lag -6 at top, +6 at bottom (NCL trYReverse)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Lag (days)")
    if lon_range is not None:
        ax.set_xlim(lon_range)
    if title:
        ax.set_title(title, loc="left", fontsize=11)

    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.09,
                      fraction=0.05, aspect=40)
    cb.set_label(shaded_label)

    if provenance:
        ax.text(0.98, 0.02, provenance, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=6.5,
                bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))

    fig.tight_layout()
    return fig, ax


def basepoint_map(
    shaded,
    lon,
    lat,
    contour=None,
    contour_lon=None,
    contour_lat=None,
    base_lon=None,
    base_lat=None,
    shaded_levels=None,
    contour_levels=None,
    cmap="RdBu_r",
    title=None,
    shaded_label="CS count anomaly",
    extent=None,
    provenance=None,
    ax=None,
):
    """Draw a longitude-latitude basepoint composite map (Fig-4 style).

    Shaded CS/CT anomaly on a cartopy map with coastlines, the unfiltered-v700 composite
    overlaid as red/blue contours, and a green basepoint star. Requires cartopy.

    Parameters mirror ``hovmoller`` but on a geographic (lon, lat) grid.
    ``extent`` is (west, east, south, north), e.g. (-40, 80, 0, 25).

    Returns (fig, ax).
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    shaded = np.asarray(shaded, dtype=float)
    proj = ccrs.PlateCarree()

    if ax is None:
        fig = plt.figure(figsize=(8.0, 4.0))
        ax = fig.add_subplot(1, 1, 1, projection=proj)
    else:
        fig = ax.figure

    if shaded_levels is None:
        amax = np.nanmax(np.abs(shaded))
        amax = 1.0 if (not np.isfinite(amax) or amax == 0) else amax
        shaded_levels = np.linspace(-amax, amax, 21)

    cf = ax.contourf(lon, lat, shaded, levels=shaded_levels, cmap=cmap,
                     extend="both", transform=proj)

    if contour is not None:
        clon = lon if contour_lon is None else np.asarray(contour_lon, dtype=float)
        clat = lat if contour_lat is None else np.asarray(contour_lat, dtype=float)
        contour = np.asarray(contour, dtype=float)
        if contour_levels is None:
            cmax = np.nanmax(np.abs(contour))
            cmax = 1.0 if (not np.isfinite(cmax) or cmax == 0) else cmax
            contour_levels = np.arange(-6, 7) * (cmax / 6.0)
        contour_levels = np.asarray(contour_levels, dtype=float)
        neg = contour_levels[contour_levels < 0]
        pos = contour_levels[contour_levels > 0]
        if neg.size:
            ax.contour(clon, clat, contour, levels=neg, colors="blue",
                       linewidths=0.7, linestyles="dashed", transform=proj)
        if pos.size:
            ax.contour(clon, clat, contour, levels=pos, colors="red",
                       linewidths=0.7, linestyles="solid", transform=proj)

    # ax.coastlines + tick labels are more robust than add_feature(COASTLINE) +
    # gridlines(draw_labels=True), which can raise a shapely LinearRing error on some
    # cartopy/shapely combinations.
    ax.coastlines(resolution="110m", linewidth=0.5)
    try:
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="0.5")
    except Exception:
        pass
    if extent is not None:
        ax.set_extent(extent, crs=proj)
        w, e, s, n = extent
        ax.set_xticks(np.arange(np.ceil(w / 20) * 20, e + 1, 20), crs=proj)
        ax.set_yticks(np.arange(np.ceil(s / 10) * 10, n + 1, 10), crs=proj)
        ax.tick_params(labelsize=8)

    if base_lon is not None and base_lat is not None:
        ax.plot([base_lon], [base_lat], marker="*", markersize=15, color="lime",
                markeredgecolor="black", markeredgewidth=0.6, transform=proj, zorder=5)

    if title:
        ax.set_title(title, loc="left", fontsize=11)
    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.08,
                      fraction=0.05, aspect=45)
    cb.set_label(shaded_label)
    if provenance:
        ax.text(0.99, 0.02, provenance, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=6.5,
                bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))
    return fig, ax


def save(fig, path, dpi=150):
    """Save a figure to ``path``.

    Uses a tight bounding box for normal axes, but NOT when a cartopy GeoAxes is present:
    bbox_inches="tight" collapses GeoAxes (it crops the map out, leaving only the
    colorbar). Detected by class name to avoid importing cartopy here.
    """
    has_geo = any(type(ax).__name__ == "GeoAxes" for ax in fig.axes)
    if has_geo:
        fig.savefig(path, dpi=dpi)
    else:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
