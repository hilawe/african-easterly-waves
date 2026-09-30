#!/usr/bin/env python3
"""Map of the recorded Atlantic-side fraction by first recorded position, 1981 to 2010, for
this record and QTrack's ERA5 record, drawn from the retained thirty-season artifacts only.

Each cohort track is placed in a fixed 5-degree box by its first recorded longitude and
latitude, as stored, with no smoothing, interpolation or rounding. Boxes are [lower, upper)
in both coordinates, except that the northern row is closed at 25 N, the cohort's inclusive
northern edge, so the binning matches the cohort's own latitude bands. The grid spans 20 W
to 55 E and 0 to 25 N and is identical in both panels. Each box is shaded by the number of
its cohort tracks recorded on the Atlantic side divided by its cohort tracks, on one 0 to
100 percent scale, and labeled with that numerator over that denominator. A box with no
cohort start is left unfilled and unlabeled. A box with a start but no Atlantic-side track
is filled at 0 percent. Boxes with fewer than SPARSE cohort starts are hatched.

Every number is checked before anything is drawn. Each season artifact must match the
digest the origin re-tabulation recorded for it. Each season's recomputed cohort and
Atlantic-side counts must equal that season's own totals. The map's grand totals and its
counts by longitude band, and by longitude band and latitude band together, must equal the
origin re-tabulation's pooled values. The Africa outline must match the digest the season artifacts recorded. Any
disagreement refuses the figure.

    .venv/bin/python3 scripts/fig_start_position_map.py --origin <origin json> \
        --regions-dir <dir holding africa.mat> --out-stem <path without extension>

writes <stem>.pdf (vector) and <stem>.png (preview), each by exclusive creation, and prints
the verified totals.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

LON_EDGES = np.arange(-20.0, 55.0 + 1e-9, 5.0)            # 15 columns, 20 W to 55 E
LAT_EDGES = np.arange(0.0, 25.0 + 1e-9, 5.0)              # 5 rows, 0 to 25 N
SPARSE = 10                                              # boxes with fewer cohort starts are hatched
SIDES = (("this_record", "Version 2"), ("qtrack", "QTrack"))
LON_BANDS = (("west of 10 W", -np.inf, -10.0), ("10 W to 10 E", -10.0, 10.0),
             ("10 E to 30 E", 10.0, 30.0), ("east of 30 E", 30.0, np.inf))
LAT_LABELS = ("0 to 5", "5 to 10", "10 to 15", "15 to 20", "20 to 25")
ATLANTIC_CUTOFF = -7.5
BAND_EDGES = (-10.0, 10.0, 30.0)


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def box_index(lon, lat):
    """(row, column) of a first position. Columns and rows are [lower, upper), and the
    northern row also takes a latitude exactly at 25 N. Refuses a position off the grid."""
    col = int(np.searchsorted(LON_EDGES, lon, side="right")) - 1
    row = int(np.searchsorted(LAT_EDGES, lat, side="right")) - 1
    if lat == LAT_EDGES[-1]:
        row = len(LAT_EDGES) - 2
    if not (0 <= col < len(LON_EDGES) - 1 and 0 <= row < len(LAT_EDGES) - 1):
        raise SystemExit(f"REFUSED: first position {lon}, {lat} lies outside the fixed grid")
    return row, col


def lon_band(lon):
    for name, lo, hi in LON_BANDS:
        if lo <= lon < hi:
            return name
    raise SystemExit(f"REFUSED: longitude {lon} fits no band")


def lat_band(lat):
    for k, lo in enumerate(LAT_EDGES[:-1]):
        if lo <= lat < LAT_EDGES[k + 1] or (k == len(LAT_EDGES) - 2 and lat == LAT_EDGES[-1]):
            return LAT_LABELS[k]
    raise SystemExit(f"REFUSED: latitude {lat} fits no band")


def tabulate(origin_path, root="."):
    """Counts per box for both sides, after every check. Returns (grids, totals, meta)."""
    origin = json.load(open(origin_path))
    years = [str(y) for y in origin["years"]]
    shape = (len(LAT_EDGES) - 1, len(LON_EDGES) - 1)
    grids = {s: {"cohort": np.zeros(shape, int), "atlantic_side": np.zeros(shape, int)} for s, _ in SIDES}
    bands = {s: {} for s, _ in SIDES}
    africa_digests = set()
    for y in years:
        rec = origin["inputs"][y]
        path = os.path.join(root, rec["path"])
        if _sha256(path) != rec["sha256"]:
            raise SystemExit(f"REFUSED: {path} does not match the digest the origin re-tabulation recorded")
        season = json.load(open(path))
        africa_digests.add(season["inputs"]["region_polygons"]["sha256"]["africa"])
        for side, _ in SIDES:
            S = season["sides"][side]
            n = a = 0
            for t in S["tracks"]:
                lon, lat = float(t["first"]["lon"]), float(t["first"]["lat"])
                hit = t["first_atlantic_side"] is not None
                r, c = box_index(lon, lat)
                grids[side]["cohort"][r, c] += 1
                grids[side]["atlantic_side"][r, c] += int(hit)
                key = (lon_band(lon), lat_band(lat))
                cell = bands[side].setdefault(key, [0, 0])
                cell[0] += int(hit)
                cell[1] += 1
                n += 1
                a += int(hit)
            total = S["by_band"]["total"]
            if n != total["cohort"] or a != total["atlantic_side"]:
                raise SystemExit(f"REFUSED: {y} {side} recomputes {a} of {n}, the season records "
                                 f"{total['atlantic_side']} of {total['cohort']}")
    totals = {}
    for side, _ in SIDES:
        pooled = origin["pooled_all_years"][side]["by_longitude_band"]
        a = int(grids[side]["atlantic_side"].sum())
        n = int(grids[side]["cohort"].sum())
        want_a = sum(pooled[b]["atlantic_side"] for b, _, _ in LON_BANDS)
        want_n = sum(pooled[b]["cohort"] for b, _, _ in LON_BANDS)
        if (a, n) != (want_a, want_n):
            raise SystemExit(f"REFUSED: {side} map totals {a} of {n} disagree with the origin "
                             f"re-tabulation's {want_a} of {want_n}")
        for b, _, _ in LON_BANDS:
            got_band = [sum(v[0] for (bb, _), v in bands[side].items() if bb == b),
                        sum(v[1] for (bb, _), v in bands[side].items() if bb == b)]
            if got_band != [pooled[b]["atlantic_side"], pooled[b]["cohort"]]:
                raise SystemExit(f"REFUSED: {side} {b} recomputes {got_band[0]} of {got_band[1]}, the origin "
                                 f"re-tabulation records {pooled[b]['atlantic_side']} of {pooled[b]['cohort']}")
            for lab in LAT_LABELS:
                got = bands[side].get((b, lab), [0, 0])
                comp = pooled[b]["latitude_composition"][lab]
                if got != [comp["atlantic_side"], comp["cohort"]]:
                    raise SystemExit(f"REFUSED: {side} {b} {lab} recomputes {got}, the origin "
                                     f"re-tabulation records {comp}")
        totals[side] = (a, n)
    if len(africa_digests) != 1:
        raise SystemExit("REFUSED: the season artifacts record different Africa outlines")
    meta = {"years": (int(years[0]), int(years[-1])), "n_years": len(years),
            "africa_sha256": africa_digests.pop(), "origin_sha256": _sha256(origin_path)}
    return grids, totals, meta


def box_style(n, a):
    """How a box is drawn: None for a box with no cohort start (no fill, no label), else
    (percent recorded on the Atlantic side, hatched), where hatched marks fewer than SPARSE
    cohort starts. A box with starts and no Atlantic-side track is (0.0, ...), not None."""
    if n == 0:
        return None
    return 100.0 * a / n, n < SPARSE


def load_africa(regions_dir, expected_sha256):
    from scipy.io import loadmat
    path = os.path.join(regions_dir, "africa.mat")
    if _sha256(path) != expected_sha256:
        raise SystemExit(f"REFUSED: {path} does not match the Africa outline the artifacts recorded")
    arr = loadmat(path)["africa"]
    return np.asarray(arr[0], float), np.asarray(arr[1], float)


def _lon_label(v, _pos=None):
    v = int(round(v))
    return "0" if v == 0 else (f"{abs(v)}W" if v < 0 else f"{v}E")


def _lat_label(v, _pos=None):
    v = int(round(v))
    return "0" if v == 0 else f"{v}N"


def render(grids, totals, meta, africa, stem, dpi=300):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    from matplotlib import patheffects as pe
    from matplotlib.ticker import FixedLocator, FuncFormatter
    from aew.plotting import panel_label
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 8, "pdf.fonttype": 42})
    cmap = plt.get_cmap("viridis")
    norm = Normalize(0.0, 100.0)
    fig = plt.figure(figsize=(6.5, 5.9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.07], hspace=0.30,
                          left=0.08, right=0.985, top=0.955, bottom=0.085)
    y0, y1 = meta["years"]
    for k, (side, name) in enumerate(SIDES):
        ax = fig.add_subplot(gs[k])
        coh, atl = grids[side]["cohort"], grids[side]["atlantic_side"]
        for r in range(coh.shape[0]):
            for c in range(coh.shape[1]):
                n, a = int(coh[r, c]), int(atl[r, c])
                style = box_style(n, a)
                if style is None:
                    continue                                   # an empty box gets no fill and no label
                frac, hatched = style
                x, y = LON_EDGES[c], LAT_EDGES[r]
                fill_color = cmap(norm(frac))
                ax.add_patch(Rectangle((x, y), 5, 5, facecolor=fill_color, edgecolor="white", lw=0.6, zorder=1))
                if hatched:
                    ax.add_patch(Rectangle((x, y), 5, 5, facecolor="none", edgecolor=(1, 1, 1, 0.55),
                                           hatch="///", lw=0, zorder=2))
                lum = 0.299 * fill_color[0] + 0.587 * fill_color[1] + 0.114 * fill_color[2]
                ink = "black" if lum > 0.5 else "white"
                ax.text(x + 2.5, y + 2.5, rf"$\dfrac{{{a}}}{{{n}}}$", ha="center", va="center",
                        fontsize=6.5, color=ink, zorder=3,
                        path_effects=[pe.withStroke(linewidth=2.4, foreground=fill_color)])
        # boundaries sit above the fills and below the labels, whose outline masks them
        ax.plot(africa[0], africa[1], color="0.35", lw=0.7, zorder=2.5)
        for e in BAND_EDGES:
            ax.axvline(e, color="black", lw=1.0, ls=(0, (1, 1.5)), zorder=2.6)
        ax.axvline(ATLANTIC_CUTOFF, color="#D55E00", lw=1.1, ls=(0, (4, 2)), zorder=2.6)
        ax.set_xlim(LON_EDGES[0], LON_EDGES[-1])
        ax.set_ylim(-3.0, 31.5)                         # clear space above 25 N for the panel label
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(FixedLocator(np.arange(-20, 56, 10)))
        ax.yaxis.set_major_locator(FixedLocator(np.arange(0, 26, 5)))
        ax.xaxis.set_major_formatter(FuncFormatter(_lon_label))
        ax.yaxis.set_major_formatter(FuncFormatter(_lat_label))
        ax.tick_params(labelsize=7.5)
        for e in LAT_EDGES:
            ax.axhline(e, color="0.8", lw=0.4, zorder=0)
        for e in LON_EDGES:
            ax.axvline(e, color="0.8", lw=0.4, zorder=0)
        a_tot, n_tot = totals[side]
        ax.set_title(f"{name}, ERA5, {y0} to {y1}", fontsize=9.5, pad=3)
        ax.text(0.99, 0.965, f"all boxes: {a_tot} of {n_tot:,} ({100.0 * a_tot / n_tot:.1f}%)",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5, zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, edgecolor="none"))
        ax.text(ATLANTIC_CUTOFF + 0.5, 30.8, "7.5W", color="#D55E00", ha="left", va="top", fontsize=7)
        panel_label(ax, "ab"[k], size=11)
    cax = fig.add_subplot(gs[2])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, 20, 40, 60, 80, 100])
    cb.set_label("recorded on the Atlantic side, percent of the box's cohort starts", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    written = []
    for ext in ("pdf", "png"):
        path = f"{stem}.{ext}"
        with open(path, "xb") as fh:                   # exclusive creation, never overwritten
            fig.savefig(fh, format=ext, dpi=dpi)
        written.append(path)
    plt.close(fig)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--origin", required=True)
    ap.add_argument("--regions-dir", required=True)
    ap.add_argument("--out-stem", required=True)
    ap.add_argument("--root", default=".", help="directory the origin artifact's paths are relative to")
    args = ap.parse_args(argv)
    grids, totals, meta = tabulate(args.origin, args.root)
    africa = load_africa(args.regions_dir, meta["africa_sha256"])
    try:
        written = render(grids, totals, meta, africa, args.out_stem)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {args.out_stem}.* exists and figures beside artifacts are never overwritten")
    for side, name in SIDES:
        coh = grids[side]["cohort"]
        print(f"{name}: {totals[side][0]} of {totals[side][1]}, boxes with starts {int((coh > 0).sum())}, "
              f"of which sparse (under {SPARSE}) {int(((coh > 0) & (coh < SPARSE)).sum())}")
    print(f"verified against origin {meta['origin_sha256'][:8]}, {meta['n_years']} seasons "
          f"{meta['years'][0]} to {meta['years'][1]}; wrote {', '.join(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
