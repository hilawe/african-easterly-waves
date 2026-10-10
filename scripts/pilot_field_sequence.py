#!/usr/bin/env python3
"""Field-only map sequences from a retained wide case, with no tracker output drawn.

A reading aid for the field-first reference work (`EASTERN_REFERENCE_FEASIBILITY_2026-10-08.md`,
section 9). For a declared time interval and cadence it draws a raw 700 hPa wind
component on the fine grid, shaded, with the wind vectors over it, in a declared box, as
figures of up to eight maps each, and writes a record naming the wide case, the steps, the
dates and the figures. It draws no axis, no candidate, no history and no prepared
field, so a reader sees the fields and nothing the detector made of them.

    python3 scripts/pilot_field_sequence.py --wide-dir <dir> --year 1990 --start 1990-09-03T00 \
        --end 1990-09-14T12 --every-hours 12 --box 0 30 20 62 --out <record.json> --png-prefix <path>

THE DEFAULT is the meridional wind, drawn as before except that, from 2026-10-09, every
figure drawn without `--surface-pressure` is labeled UNMASKED in its title, and its record
says no terrain mask was applied. `--shade u` draws the zonal
wind instead, on a fixed color scale with out-of-range cells in their own colors and a
labeled reference arrow, so every map of a sequence reads on one scale.

THE LAYOUT, from 2026-10-10. Each figure holds exactly its maps, in at most four columns,
and is sized to them, with longitude labels on the lowest map of each column and the long
titles broken into lines that fit. Before, every figure was an eight-slot grid, so a
figure of fewer maps left empty space and hid its longitude labels. The maps themselves,
their scales and the records are drawn as before.

`--surface-pressure` masks the shading and the vectors below model ground. A cell is
kept where the instantaneous ERA5 surface pressure exceeds the plotted pressure level
(strictly), and masked where it is equal, lower or not finite. Surface pressure is read
in pascals from a retained file whose grid must contain every plotted cell at the same
coordinates and whose times must match every plotted step, or the run refuses. The mask
is a display and record mask. The winds were interpolated to the plotted grid before it,
so any influence of below-ground source values on above-ground cells is not removed by it.

`--level-files U V` with `--level-hpa` draws another retained single-level ERA5 wind pair
instead of the wide case's 700 hPa winds. With a wide case they are read at its grid points
and times exactly, so two levels are drawn on identical points. Without one, the grid and
times are the U file's own integer-degree points and times. The terrain mask then uses
that level.

`--shade zeta` draws relative vorticity from the drawn winds, by centered differences on
the sphere between neighboring plotted points. A value is kept only where the cell and its
four stencil neighbors are all above model ground at the drawn level with finite winds.
Cells on the box edge have no centered stencil and are left blank.

`--markers` draws declared positions from a marker file, each only on the map at its own
time, never joined across times, with the labels and symbols the file gives.

`--reference` draws declared reference segments from a reference file (the R1 pilot's
declaration), only for segments A and B and only on the maps at their declared times,
labeled as the reader's interpretation. It never draws the tail or any other entry.
"""
import argparse
import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import exact_tracks as X  # noqa: E402
import pilot_alledge_replay as A  # noqa: E402

EPOCH = dt.datetime(1900, 1, 1)
PANELS = 8
MAX_COLS = 4
PANEL_IN = (4.4, 3.7)       # inches given to each map with its title and labels
MARGIN_IN = (1.3, 1.5)      # the colorbar, and the title lines with the legend
DAYS_1900_TO_1970 = 25567.0
LEVEL_HPA = 700.0
VECTOR_EVERY = 2
VECTOR_SCALE = 350
VECTOR_KEY_MS = 10.0
SHADES = {
    "v": {"component": "meridional", "symbol": "v", "vmin": -10.0, "vmax": 10.0,
          "sign": "positive from the south (southerly), negative from the north (northerly)"},
    "u": {"component": "zonal", "symbol": "u", "vmin": -20.0, "vmax": 20.0,
          "sign": "positive from the west (westerly), negative from the east (easterly)"},
    "zeta": {"component": "relative vorticity", "symbol": "zeta", "vmin": -5.0, "vmax": 5.0, "units": "1e-5 s-1",
             "sign": "positive cyclonic in the Northern Hemisphere, in units of 1e-5 per second"},
}
EARTH_RADIUS_M = 6.371e6
NEIGHBOR_COLOR = "0.75"
ZETA_CLASSES = ("valid", "terrain", "sp_missing", "wind_missing", "neighbor_invalid", "edge")
UNDER_COLOR, OVER_COLOR = "#ff00ff", "#000000"
TERRAIN_COLOR, MISSING_COLOR = "0.55", "#f2e6a0"
UNMASKED_LABEL = "UNMASKED, no terrain-validity mask applied"
SEGMENTS = ("A", "B")


def to_day(stamp):
    return (dt.datetime.strptime(stamp, "%Y-%m-%dT%H") - EPOCH).total_seconds() / 86400.0


def sequence_steps(times, start_day, end_day, every_hours):
    """The step indices whose times lie in [start, end] at the cadence, measured from the
    start, with a tolerance of one minute."""
    times = np.asarray(times, float).ravel()
    step_days = every_hours / 24.0
    out = []
    for k, t in enumerate(times):
        if t < start_day - 1e-6 or t > end_day + 1e-6:
            continue
        n = (t - start_day) / step_days
        if abs(n - round(n)) * step_days * 1440.0 <= 1.0:
            out.append(int(k))
    return out


def date_of(t):
    return f"{(EPOCH + dt.timedelta(days=float(t))):%Y-%m-%d %HZ}"


def surface_pressure_on(path, lat_vals, lon_vals, step_days):
    """The surface pressure (Pa) at the plotted cells and steps, from a retained ERA5 file.

    Every plotted latitude and longitude must be one of the file's coordinates within
    1e-6 degrees, and every step's time one of the file's times within one minute, or
    the run refuses. No interpolation is done."""
    return field_on(path, "sp", "Pa", lat_vals, lon_vals, step_days)


def field_on(path, name, units, lat_vals, lon_vals, step_days, level_hpa=None):
    """A retained ERA5 field at the plotted cells and steps, exact coordinates and times
    or refused, no interpolation. With `level_hpa`, the file must hold that single
    pressure level, which is then dropped from the array's axes."""
    import netCDF4
    d = netCDF4.Dataset(path)
    try:
        if name not in d.variables:
            raise SystemExit(f"REFUSED: {path} holds no variable {name!r}")
        var = d.variables[name]
        if getattr(var, "units", None) != units:
            what = "surface pressure" if name == "sp" else name
            raise SystemExit(f"REFUSED: {path} holds {what} in {getattr(var, 'units', None)!r}, not {units}")
        if level_hpa is not None:
            levels = np.asarray(d.variables["pressure_level"][:], float).ravel() if "pressure_level" in d.variables else np.array([])
            if levels.size != 1 or abs(levels[0] - float(level_hpa)) > 1e-6:
                raise SystemExit(f"REFUSED: {path} holds levels {levels.tolist()}, not the single level {level_hpa:g} hPa")
        flat = np.asarray(d.variables["latitude"][:], float)
        flon = np.asarray(d.variables["longitude"][:], float)
        tname = "valid_time" if "valid_time" in d.variables else "time"
        tvar = d.variables[tname]
        if not str(tvar.units).startswith("seconds since 1970-01-01"):
            raise SystemExit(f"REFUSED: {path} has times in {tvar.units!r}, not seconds since 1970-01-01")
        fdays = np.asarray(tvar[:], float) / 86400.0 + DAYS_1900_TO_1970

        def index(wanted, have, what, tol):
            out = []
            for w in wanted:
                j = int(np.argmin(np.abs(have - w)))
                if abs(have[j] - w) > tol:
                    raise SystemExit(f"REFUSED: the plotted {what} {w} is not a coordinate of {path}")
                out.append(j)
            return out
        ri = index(np.asarray(lat_vals, float), flat, "latitude", 1e-6)
        ci = index(np.asarray(lon_vals, float), flon, "longitude", 1e-6)
        ti = index(np.asarray(step_days, float), fdays, "time", 1.0 / 1440.0)
        arr = np.asarray(var[:], float)
        if level_hpa is not None:
            arr = arr[:, 0]
        return arr[np.ix_(ti, ri, ci)]
    finally:
        d.close()


def terrain_classes(sp_pa, wind, level_hpa=LEVEL_HPA):
    """Per cell: 'valid' (surface pressure above the level, wind finite), 'terrain'
    (surface pressure equal to or below the level), 'sp_missing' (surface pressure not
    finite) or 'wind_missing' (above ground with no finite wind). Without surface
    pressure every finite wind is valid and nothing is terrain."""
    wind = np.asarray(wind, float)
    if sp_pa is None:
        cls = np.where(np.isfinite(wind), "valid", "wind_missing")
        return cls
    sp = np.asarray(sp_pa, float)
    above = sp > level_hpa * 100.0
    cls = np.full(wind.shape, "valid", dtype=object)
    cls[~np.isfinite(sp)] = "sp_missing"
    cls[np.isfinite(sp) & ~above] = "terrain"
    cls[above & ~np.isfinite(wind)] = "wind_missing"
    return cls


def segment_cells(lat, lon, m):
    """The plotted cells on a declared segment, the column nearest its axis longitude
    over its latitude range."""
    c = int(np.argmin(np.abs(lon - float(m["axis_lon"]))))
    rows = [r for r in range(lat.size) if float(m["lat_range"][0]) <= lat[r] <= float(m["lat_range"][1])]
    return rows, c


def relative_vorticity(u, v, lat1, lon1, wind_classes):
    """Relative vorticity (1/s) on the sphere at interior cells, by centered differences
    between neighboring plotted points, dv/dx - du/dy + u tan(lat)/a, and its classes. A
    cell is 'valid' only when it and its four stencil neighbors are valid wind cells. A
    valid wind cell with an invalid neighbor is 'neighbor_invalid', one on the box edge is
    'edge', and a cell that is not a valid wind cell keeps its own class."""
    u = np.asarray(u, float); v = np.asarray(v, float)
    base = np.asarray(wind_classes, dtype=object)
    ok = base == "valid"
    ny, nx = u.shape
    cls = base.copy()
    zeta = np.full((ny, nx), np.nan)
    edge = np.zeros((ny, nx), bool); edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    cls[ok & edge] = "edge"
    if ny < 3 or nx < 3:
        return zeta, cls
    phi = np.deg2rad(np.asarray(lat1, float)); lam = np.deg2rad(np.asarray(lon1, float))
    inner = ok[1:-1, 1:-1]
    stencil = ok[:-2, 1:-1] & ok[2:, 1:-1] & ok[1:-1, :-2] & ok[1:-1, 2:]
    keep = inner & stencil
    cls_in = cls[1:-1, 1:-1]
    cls_in[inner & ~stencil] = "neighbor_invalid"
    dvdx = (v[1:-1, 2:] - v[1:-1, :-2]) / (EARTH_RADIUS_M * np.cos(phi[1:-1])[:, None] * (lam[2:] - lam[:-2])[None, :])
    dudy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (EARTH_RADIUS_M * (phi[2:] - phi[:-2])[:, None])
    z = dvdx - dudy + u[1:-1, 1:-1] * np.tan(phi[1:-1])[:, None] / EARTH_RADIUS_M
    zeta[1:-1, 1:-1] = np.where(keep, z, np.nan)
    return zeta, cls


def panel_arrays(wide, k, rows, cols, shade, sp_k=None, level_hpa=LEVEL_HPA):
    """The shaded component and the vectors for one map, with every cell that is not
    valid set to NaN, and the cell classes."""
    u = np.asarray(wide["u"][k], float)[rows, :][:, cols]
    v = np.asarray(wide["v"][k], float)[rows, :][:, cols]
    if shade == "zeta":
        wcls = terrain_classes(sp_k, np.where(np.isfinite(u) & np.isfinite(v), u, np.nan), level_hpa)
        lat1 = np.asarray(wide["latgrid"], float)[rows, 0]; lon1 = np.asarray(wide["longrid"], float)[0, cols]
        zeta, zcls = relative_vorticity(u, v, lat1, lon1, wcls)
        wok = wcls == "valid"
        return {"shaded": np.where(zcls == "valid", zeta * 1e5, np.nan), "u": np.where(wok, u, np.nan), "v": np.where(wok, v, np.nan), "classes": zcls}
    field = u if shade == "u" else v
    cls = terrain_classes(sp_k, np.where(np.isfinite(u) & np.isfinite(v), field, np.nan), level_hpa)
    ok = cls == "valid"
    return {"shaded": np.where(ok, field, np.nan), "u": np.where(ok, u, np.nan), "v": np.where(ok, v, np.nan), "classes": cls}


def draw_panel(ax, lon, lat, arr, shade, extended):
    """One map. The default (meridional, not extended) is drawn exactly as before."""
    spec = SHADES[shade]
    if not extended:
        m = ax.pcolormesh(lon, lat, arr["shaded"], cmap="RdBu_r", vmin=-10, vmax=10, shading="auto")
        ax.quiver(np.meshgrid(lon, lat)[0][::2, ::2], np.meshgrid(lon, lat)[1][::2, ::2], arr["u"][::2, ::2], arr["v"][::2, ::2], color="0.2", scale=350, width=0.0018)
        return m
    import matplotlib
    from matplotlib.colors import ListedColormap
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_under(UNDER_COLOR); cmap.set_over(OVER_COLOR); cmap.set_bad((0, 0, 0, 0))
    m = ax.pcolormesh(lon, lat, np.ma.masked_invalid(arr["shaded"]), cmap=cmap, vmin=spec["vmin"], vmax=spec["vmax"], shading="auto")
    cls = arr["classes"]
    for name, color in (("terrain", TERRAIN_COLOR), ("neighbor_invalid", NEIGHBOR_COLOR), ("wind_missing", MISSING_COLOR), ("sp_missing", MISSING_COLOR)):
        hit = (cls == name).astype(float)
        if hit.any():
            ax.pcolormesh(lon, lat, np.ma.masked_equal(hit, 0.0), cmap=ListedColormap([color]), vmin=0, vmax=1, shading="auto")
    LO, LA = np.meshgrid(lon, lat)
    q = ax.quiver(LO[::VECTOR_EVERY, ::VECTOR_EVERY], LA[::VECTOR_EVERY, ::VECTOR_EVERY],
                  np.ma.masked_invalid(arr["u"][::VECTOR_EVERY, ::VECTOR_EVERY]), np.ma.masked_invalid(arr["v"][::VECTOR_EVERY, ::VECTOR_EVERY]),
                  color="0.2", scale=VECTOR_SCALE, width=0.0018)
    ax.quiverkey(q, 0.86, 1.02, VECTOR_KEY_MS, f"{VECTOR_KEY_MS:g} m/s", labelpos="E", fontproperties={"size": 7})
    return m


def panel_grid(n, max_cols=MAX_COLS):
    """Rows and columns for n maps, one row up to `max_cols`, never an empty row or column."""
    cols = min(n, max_cols)
    return -(-n // cols), cols


def make_figure(plt, n):
    """A figure holding exactly n map axes and sized to them. Longitude labels go on the
    lowest map of each column, latitude labels on the first column, and no hidden axis is
    left in the grid."""
    nrows, ncols = panel_grid(n)
    fig = plt.figure(figsize=(PANEL_IN[0] * ncols + MARGIN_IN[0], PANEL_IN[1] * nrows + MARGIN_IN[1]), layout="constrained")
    grid = fig.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False).ravel()
    for ax in grid[n:]:
        fig.delaxes(ax)
    used = list(grid[:n])
    for i, ax in enumerate(used):
        if i + ncols >= n:                                            # no map below it
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("longitude (degrees E)", fontsize=9)
        if i % ncols == 0:
            ax.set_ylabel("latitude (degrees N)", fontsize=9)
    return fig, used


def wrap_title(fig, text, fontsize):
    """Line breaks so a title fits the figure's width. The words are unchanged."""
    import textwrap
    width = max(40, int(fig.get_figwidth() * 72.0 / (fontsize * 0.56)))
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)


def draw_sequence(wide, steps, box, out_prefix, label, panels=PANELS, shade="v", sp=None, reference=None, level_hpa=LEVEL_HPA, markers=None):
    """Figures of `panels` maps each, a wind component shaded with (u, v) vectors, the 40 E
    line and the 5, 15 and 25 N lines for orientation. With the default shade and no
    surface pressure or reference, the maps are drawn as before, in the layout of
    `make_figure`. Returns the figure
    paths, the panel dates and, per panel, the cell classes, the out-of-range counts and
    the reference overlays drawn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    lat = np.asarray(wide["latgrid"], float); lon = np.asarray(wide["longrid"], float)
    rows = np.flatnonzero((lat[:, 0] >= box[0]) & (lat[:, 0] <= box[1]))
    cols = np.flatnonzero((lon[0, :] >= box[2]) & (lon[0, :] <= box[3]))
    if rows.size == 0 or cols.size == 0:
        raise SystemExit("REFUSED: the box lies outside the wide case's grid")
    la, lo = lat[rows, :][:, cols], lon[rows, :][:, cols]
    lat1, lon1 = la[:, 0], lo[0, :]
    times = np.asarray(wide["time"], float).ravel()
    extended = shade != "v" or sp is not None or reference is not None or level_hpa != LEVEL_HPA or markers is not None
    spec = SHADES[shade]
    ref_by_date = {}
    if reference is not None:
        for m in reference["maps"]:
            if m.get("segment") in SEGMENTS:
                ref_by_date[f"{dt.datetime.strptime(m['date'], '%Y-%m-%dT%H'):%Y-%m-%d %HZ}"] = m
    figures = []
    for f0 in range(0, len(steps), panels):
        chunk = steps[f0:f0 + panels]
        fig, used = make_figure(plt, len(chunk))
        panel_records = []
        for ax, k in zip(used, chunk):
            sp_k = None if sp is None else sp[steps.index(k)]
            arr = panel_arrays(wide, k, rows, cols, shade, sp_k, level_hpa)
            m = draw_panel(ax, lon1, lat1, arr, shade, extended)
            ax.axvline(40.0, color="k", lw=0.8, ls="--")
            for y in (5.0, 15.0, 25.0):
                ax.axhline(y, color="0.4", lw=0.5, ls=":")
            date = date_of(times[k])
            ax.set_title(date, fontsize=10)
            ax.set_xlim(box[2], box[3]); ax.set_ylim(box[0], box[1])
            if not extended:
                continue
            cls = arr["classes"]
            valid = cls == "valid"
            vals = arr["shaded"][valid]
            rec = {"step": int(k), "date": date, "cells": int(cls.size),
                   "classes": {name: int((cls == name).sum()) for name in (ZETA_CLASSES if shade == "zeta" else ("valid", "terrain", "sp_missing", "wind_missing"))},
                   "out_of_range": {"below": int((vals < spec["vmin"]).sum()), "above": int((vals > spec["vmax"]).sum())}}
            if sp_k is not None:
                rec["terrain_with_25hPa_buffer"] = int((np.isfinite(sp_k) & ~(sp_k > (level_hpa + 25.0) * 100.0)).sum())
            if markers is not None:
                drawn = []
                for mset in markers["sets"]:
                    for pos in mset["positions"]:
                        if f"{dt.datetime.strptime(pos['time'], '%Y-%m-%dT%H'):%Y-%m-%d %HZ}" != date:
                            continue
                        if "lat_range" in pos:
                            ax.plot([float(pos["lon"])] * 2, [float(x) for x in pos["lat_range"]], color=mset["color"], lw=2.5, zorder=5)
                        else:
                            ax.scatter([float(pos["lon"])], [float(pos["lat"])], marker=mset["marker"], s=70, facecolors="none", edgecolors=mset["color"], linewidths=2.0, zorder=6)
                        drawn.append({"set": mset["label"], "lon": float(pos["lon"]), **({"lat_range": [float(x) for x in pos["lat_range"]]} if "lat_range" in pos else {"lat": float(pos["lat"])})})
                rec["markers_drawn"] = drawn
            mref = ref_by_date.get(date)
            if mref is not None:
                r_rows, c = segment_cells(lat1, lon1, mref)
                ax.plot([float(mref["axis_lon"])] * 2, [float(mref["lat_range"][0]), float(mref["lat_range"][1])], color="#00a000", lw=3.0, solid_capstyle="butt")
                ax.text(float(mref["axis_lon"]) + 0.6, float(mref["lat_range"][1]) + 0.4, f"R1 {mref['segment']} (declared reading)", color="#006400", fontsize=8)
                seg_cls = [str(cls[r, c]) for r in r_rows]
                seg_u = [float(arr["u"][r, c]) for r in r_rows if cls[r, c] == "valid"]
                near = (np.abs(lo - float(mref["axis_lon"])) <= 3.0) & (la >= float(mref["lat_range"][0]) - 3.0) & (la <= float(mref["lat_range"][1]) + 3.0)
                rec["reference_overlay"] = {"segment": mref["segment"], "axis_lon": float(mref["axis_lon"]), "lat_range": [float(x) for x in mref["lat_range"]],
                                            "segment_cells": len(r_rows), "segment_classes": {n: seg_cls.count(n) for n in sorted(set(seg_cls))},
                                            "u_on_segment_valid": None if not seg_u else {"min": round(min(seg_u), 2), "mean": round(float(np.mean(seg_u)), 2), "max": round(max(seg_u), 2), "n": len(seg_u)},
                                            "within_3deg": {"cells": int(near.sum()), "terrain": int((near & (cls == "terrain")).sum())}}
            panel_records.append(rec)
        if not extended:
            fig.suptitle(wrap_title(fig, f"{label}, raw ERA5 700 hPa meridional wind (shaded) and wind vectors, fine grid, {UNMASKED_LABEL}, no tracker output drawn", 12), fontsize=12)
            fig.colorbar(m, ax=used, shrink=0.8, label="700 hPa meridional wind (m/s)")
        else:
            masked = f"masked below model ground (surface pressure not above {level_hpa:g} hPa)" if sp is not None else UNMASKED_LABEL
            if shade == "zeta":
                masked = (f"masked where the cell or a derivative neighbor is below model ground (surface pressure not above {level_hpa:g} hPa)"
                          if sp is not None else UNMASKED_LABEL)
            title = (f"{label}, ERA5 {level_hpa:g} hPa relative vorticity (shaded, centered differences between plotted points) and wind vectors, {masked}, no tracker output drawn" if shade == "zeta"
                     else f"{label}, raw ERA5 {level_hpa:g} hPa {spec['component']} wind {spec['symbol']} (shaded) and wind vectors, fine grid, {masked}, no tracker output drawn")
            units = spec.get("units", "m/s")
            note = (f"{spec['symbol']} {spec['sign']}. Below {spec['vmin']:g} {units} drawn magenta, above {spec['vmax']:g} {units} drawn black. Vectors on one scale (key {VECTOR_KEY_MS:g} m/s)."
                    + (" Open symbols and bars are the marker file's declared positions, each drawn only at its own time and never joined." if markers is not None else "")
                    + (f" Green bar the declared R1 reading (segments A and B, read at {LEVEL_HPA:g} hPa) at its declared times, the reader's interpretation." if reference is not None else ""))
            fig.suptitle(wrap_title(fig, title, 10) + "\n" + wrap_title(fig, note, 10), fontsize=10)
            fig.colorbar(m, ax=used, shrink=0.8, extend="both",
                         label=f"{level_hpa:g} hPa relative vorticity ({units})" if shade == "zeta" else f"{level_hpa:g} hPa {spec['component']} wind {spec['symbol']} (m/s)")
            handles = [Patch(color=TERRAIN_COLOR, label="below model ground at this level, masked"), Patch(color=MISSING_COLOR, label="missing in source")]
            if shade == "zeta":
                handles.insert(1, Patch(color=NEIGHBOR_COLOR, label="above ground, a derivative neighbor is not, masked"))
            if markers is not None:
                from matplotlib.lines import Line2D
                for mset in markers["sets"]:
                    bar = any("lat_range" in pos for pos in mset["positions"])
                    handles.append(Line2D([], [], color=mset["color"], lw=2.5 if bar else 0, marker=None if bar else mset["marker"],
                                          markerfacecolor="none", markeredgecolor=mset["color"], markeredgewidth=2.0, label=mset["label"]))
            fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=9)
        path = f"{out_prefix}_p{f0 // panels + 1}.png"
        fig.savefig(path, dpi=100); plt.close(fig)
        entry = {"path": path, "sha256": X.digest(path), "steps": [int(k) for k in chunk], "dates": [date_of(times[k]) for k in chunk]}
        if extended:
            entry["panels"] = panel_records
        figures.append(entry)
    return figures


def grid_from_level_file(path, year):
    """A grid and time axis taken from a retained level file, for a year with no wide case:
    the file's integer-degree latitudes and longitudes, in the file's order, and its times
    as days since 1900. Refuses a file whose times are not all in `year`."""
    import netCDF4
    d = netCDF4.Dataset(path)
    try:
        lat = np.asarray(d.variables["latitude"][:], float)
        lon = np.asarray(d.variables["longitude"][:], float)
        tvar = d.variables["valid_time" if "valid_time" in d.variables else "time"]
        if not str(tvar.units).startswith("seconds since 1970-01-01"):
            raise SystemExit(f"REFUSED: {path} has times in {tvar.units!r}, not seconds since 1970-01-01")
        days = np.asarray(tvar[:], float) / 86400.0 + DAYS_1900_TO_1970
    finally:
        d.close()
    years = {(EPOCH + dt.timedelta(days=float(x))).year for x in days}
    if years != {year}:
        raise SystemExit(f"REFUSED: {path} holds years {sorted(years)}, not {year} alone")
    la = lat[np.abs(lat - np.round(lat)) < 1e-6]
    lo = lon[np.abs(lon - np.round(lon)) < 1e-6]
    latgrid, longrid = np.meshgrid(la, lo, indexing="ij")
    return {"latgrid": latgrid, "longrid": longrid, "time": days, "sha256": X.digest(path), "record_sha256": None}


def remap_steps(figures, steps):
    """Figure records drawn on positions 0..n-1 of `steps`, renumbered to the wide case's
    own step indices."""
    out = []
    for f in figures:
        g = dict(f)
        g["steps"] = [int(steps[k]) for k in f["steps"]]
        if "panels" in f:
            g["panels"] = [dict(p, step=int(steps[p["step"]])) for p in f["panels"]]
        out.append(g)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--start", "--end", "--out", "--png-prefix", "--label"):
        ap.add_argument(name, required=True)
    ap.add_argument("--wide-dir", help="the retained wide case giving the grid and times; without it, --level-files must be given and the grid is the U file's integer-degree points")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--every-hours", type=int, default=12)
    ap.add_argument("--box", nargs=4, type=float, required=True, metavar=("LAT0", "LAT1", "LON0", "LON1"))
    ap.add_argument("--shade", choices=sorted(SHADES), default="v", help="the shaded component, meridional (default, as before) or zonal")
    ap.add_argument("--surface-pressure", help="a retained ERA5 surface-pressure file (Pa) covering the plotted cells and steps; masks below model ground")
    ap.add_argument("--reference", help="a reference declaration whose segments A and B are drawn at their declared times")
    ap.add_argument("--level-files", nargs=2, metavar=("U_FILE", "V_FILE"),
                    help="retained ERA5 single-level u and v files to draw instead of the wide case's 700 hPa winds, read at the wide case's points and times exactly, or without a wide case at the U file's integer-degree points and times")
    ap.add_argument("--markers", help="a marker file of declared positions, each drawn only on the map at its own time, never joined across times")
    ap.add_argument("--level-hpa", type=float, default=LEVEL_HPA, help="the pressure level drawn; must be the level the files hold, and 700 without --level-files")
    args = ap.parse_args(argv)
    if args.level_files is None and args.level_hpa != LEVEL_HPA:
        raise SystemExit(f"REFUSED: the wide case holds {LEVEL_HPA:g} hPa only; another level needs --level-files")
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    if args.wide_dir is None and args.level_files is None:
        raise SystemExit("REFUSED: without a wide case the grid and times must come from --level-files")
    wide = A.load_wide(args.wide_dir, args.year) if args.wide_dir else grid_from_level_file(args.level_files[0], args.year)
    steps = sequence_steps(wide["time"], to_day(args.start), to_day(args.end), args.every_hours)
    if not steps:
        raise SystemExit("REFUSED: no retained step lies in the declared interval")
    lat = np.asarray(wide["latgrid"], float); lon = np.asarray(wide["longrid"], float)
    rows = np.flatnonzero((lat[:, 0] >= args.box[0]) & (lat[:, 0] <= args.box[1]))
    cols = np.flatnonzero((lon[0, :] >= args.box[2]) & (lon[0, :] <= args.box[3]))
    times = np.asarray(wide["time"], float).ravel()
    step_days = [times[k] for k in steps]
    sp = None
    if args.surface_pressure:
        sp = surface_pressure_on(args.surface_pressure, lat[rows, 0], lon[0, cols], step_days)
    markers = None
    if args.markers:
        import json
        markers = json.load(open(args.markers))
    reference = None
    if args.reference:
        import json
        reference = json.load(open(args.reference))["reference"]
    if args.level_files:
        level_winds = {"latgrid": lat[rows, :][:, cols], "longrid": lon[rows, :][:, cols], "time": np.asarray(step_days, float),
                       "u": field_on(args.level_files[0], "u", "m s**-1", lat[rows, 0], lon[0, cols], step_days, args.level_hpa),
                       "v": field_on(args.level_files[1], "v", "m s**-1", lat[rows, 0], lon[0, cols], step_days, args.level_hpa)}
        figures = draw_sequence(level_winds, list(range(len(steps))), args.box, args.png_prefix, args.label, shade=args.shade, sp=sp, reference=reference, level_hpa=args.level_hpa, markers=markers)
        figures = remap_steps(figures, steps)
    else:
        figures = draw_sequence(wide, steps, args.box, args.png_prefix, args.label, shade=args.shade, sp=sp, reference=reference, markers=markers)
    spec = SHADES[args.shade]
    out = {"generated_by": "scripts/pilot_field_sequence.py", "script_sha256": X.digest(os.path.abspath(__file__)), "year": args.year,
           "wide": ({"dir": args.wide_dir, "case_sha256": wide["sha256"], "record_sha256": wide["record_sha256"]} if args.wide_dir
                    else {"dir": None, "grid_from": args.level_files[0], "sha256": wide["sha256"], "points": "the file's integer-degree latitudes and longitudes"}),
           "declared": {"start": args.start, "end": args.end, "every_hours": args.every_hours, "box": [float(x) for x in args.box], "label": args.label,
                        "drawn": f"{spec['symbol']}{args.level_hpa:g} shaded, (u, v) vectors, the 40 E line, 5, 15 and 25 N lines; no axis, candidate, history or prepared field"},
           "steps": steps, "figures": figures,
           "terrain_mask": "applied, see terrain" if sp is not None else "none applied, the figures are labeled UNMASKED"}
    if args.level_files:
        out["level_source"] = {"level_hpa": args.level_hpa, "u": {"path": args.level_files[0], "sha256": X.digest(args.level_files[0])},
                               "v": {"path": args.level_files[1], "sha256": X.digest(args.level_files[1])},
                               "read": ("at the wide case's grid points and times" if args.wide_dir else "at the U file's own integer-degree points and times")
                                       + " inside the box, exact coordinates and times or refused, no interpolation"}
    if args.shade != "v" or sp is not None or reference is not None or args.level_files or markers is not None:
        out["shaded"] = {"component": spec["component"], "symbol": spec["symbol"], "level_hpa": args.level_hpa, "units": spec.get("units", "m/s"), "sign_convention": spec["sign"],
                         "color_scale": [spec["vmin"], spec["vmax"]], "out_of_range_colors": {"below": UNDER_COLOR, "above": OVER_COLOR}}
        out["vectors"] = {"every": VECTOR_EVERY, "scale": VECTOR_SCALE, "key_ms": VECTOR_KEY_MS, "masked_like_the_shading": True}
        out["terrain"] = None if sp is None else {
            "surface_pressure": {"path": args.surface_pressure, "sha256": X.digest(args.surface_pressure), "units": "Pa"},
            "rule": f"kept where surface pressure > {args.level_hpa:g} hPa, strictly; equal or lower is below model ground and masked; non-finite surface pressure is masked and counted apart",
            "alignment": "every plotted cell a coordinate of the surface-pressure grid within 1e-6 degrees, every step one of its times within one minute, no interpolation",
            "sensitivity": "terrain_with_25hPa_buffer counts the cells the project's terrain module would mask with its default 25 hPa buffer",
            "not_removed": "the winds were interpolated to the plotted grid before this mask, so influence of below-ground source values on kept cells is not removed"}
        if args.shade == "zeta":
            out["shaded"]["method"] = ("dv/dx - du/dy + u tan(lat)/a on the sphere, a = 6.371e6 m, centered differences between the neighboring plotted points; "
                                       "kept only where the cell and its four stencil neighbors are above model ground with finite winds; box-edge cells left blank")
        out["markers"] = None if markers is None else {"path": args.markers, "sha256": X.digest(args.markers), "drawn": "each position only at its own time, never joined across times"}
        out["reference"] = None if reference is None else {"path": args.reference, "sha256": X.digest(args.reference), "drawn": "segments A and B only, at their declared times, as the reader's interpretation"}
    X.publish_json(args.out, out, exclusive=True)
    print(f"{args.label}: {len(steps)} maps in {len(figures)} figures, {args.start} to {args.end} every {args.every_hours} h, shade {args.shade}. Record {args.out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
