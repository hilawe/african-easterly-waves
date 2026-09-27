#!/usr/bin/env python3
"""The map of first recorded positions by record and outcome for the origin-dependent
comparison: one panel per season, this record as circles and QTrack as squares, colored
by outcome, with the predeclared longitude band edges drawn and the archive's Africa and
North Atlantic polygon outlines behind them. Positions come from the origin artifact,
whose season-artifact digests are verified first, and the polygons come from the files the
season artifacts name, verified by digest. No projection is drawn.

    .venv/bin/python3 scripts/fig_coast_crossing_origin.py --origin <json> --artifacts <dir> --out <png>
"""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

COLORS = {"atlantic_side": "#1f5fa8", "gulf_only": "#e08a1e", "none": "#8a8a8a", "follow_up_incomplete": "#c8552d"}
LABELS = {"atlantic_side": "Atlantic side", "gulf_only": "Gulf only", "none": "none", "follow_up_incomplete": "follow-up incomplete"}


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def verified_season_artifacts(origin, artifacts_dir):
    import coast_crossing_origin as O
    arts = {}
    for y, rec in origin["inputs"].items():
        path = os.path.join(artifacts_dir, O.ARTIFACT.format(year=y))
        if _sha256(path) != rec["sha256"]:
            raise SystemExit(f"REFUSED: {path} is not the season artifact the origin artifact names")
        arts[y] = json.load(open(path))
    return arts


def bound_positions(origin, arts):
    """The points to plot, rebuilt from the VERIFIED season artifacts and required to equal
    the origin artifact's first_positions exactly, so nothing plotted can come from an
    edited origin payload."""
    import coast_crossing_origin as O
    out = {}
    for y, art in arts.items():
        out[y] = {}
        for name, side in art["sides"].items():
            rebuilt = [{"id": t["id"], "lon": t["first"]["lon"], "lat": t["first"]["lat"], "status": O.status(t)} for t in side["tracks"]]
            if rebuilt != origin["seasons"][y][name]["first_positions"]:
                raise SystemExit(f"REFUSED: the origin artifact's first positions for {y} {name} are not the season artifact's")
            out[y][name] = rebuilt
    return out


def polygons(art):
    """The Africa and North Atlantic polygons from the files the season artifact names,
    refused unless their digests are the artifact's."""
    import season_metrics as S
    rdir = art["inputs"]["region_polygons"]["dir"]
    recorded = art["inputs"]["region_polygons"]["sha256"]
    if {name: _sha256(os.path.join(rdir, f"{name}.mat")) for name in recorded} != recorded:
        raise SystemExit("REFUSED: the region polygons are not the ones the season artifact names")
    regions, _ = S.load_regions(rdir)
    return {code: path.vertices for code, path in regions if code in ("AFR", "NAL")}


def render(origin, arts, out, dpi=150):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aew.plotting import panel_label
    import coast_crossing_origin as O
    years = [str(y) for y in origin["years"]]
    fig, axes = plt.subplots(len(years), 1, figsize=(11, 3.4 * len(years)), squeeze=False)
    drawn = {}
    points = bound_positions(origin, arts)
    polys = polygons(arts[years[0]])
    for ax, y in zip(axes[:, 0], years):
        for code, v in polys.items():
            ax.plot(v[:, 0], v[:, 1], color="#bbbbbb", lw=0.5, zorder=0)
        for edge in O.LON_EDGES:
            ax.axvline(edge, color="k", lw=0.6, ls=":", zorder=1)
        n = 0
        for name, marker, size in (("this_record", "o", 16), ("qtrack", "s", 18)):
            pts = points[y][name]
            for st in O.STATUSES:
                sel = [p for p in pts if p["status"] == st]
                if not sel:
                    continue
                ax.scatter([p["lon"] for p in sel], [p["lat"] for p in sel], s=size, marker=marker, color=COLORS[st],
                           edgecolors="k" if name == "qtrack" else "none", linewidths=0.4, zorder=3,
                           label=f"{'this record' if name == 'this_record' else 'QTrack'}, {LABELS[st]}")
                n += len(sel)
        ax.set_xlim(-20, 52)
        ax.set_ylim(-2, 27)
        ax.set_ylabel("latitude")
        ax.text(0.99, 0.04, f"{y}: {n} cohort starts", transform=ax.transAxes, fontsize=8, ha="right", va="bottom")
        panel_label(ax, "abc"[years.index(y)], size=11)
        drawn[y] = n
    axes[-1, 0].set_xlabel("longitude")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("First recorded positions of the cohort by record and outcome, with the predeclared longitude band edges", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return drawn


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--origin", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and figures beside artifacts are never overwritten")
    origin = json.load(open(args.origin))
    arts = verified_season_artifacts(origin, args.artifacts)
    drawn = render(origin, arts, args.out)
    print(f"wrote {args.out}: " + ", ".join(f"{y} {n} starts" for y, n in drawn.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
