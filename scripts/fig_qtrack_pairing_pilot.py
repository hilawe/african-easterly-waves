#!/usr/bin/env python3
"""One figure for the pairing pilot, from its retained per-season artifacts: for each
season, the first observation in coverage of every assigned pair on both sides (joined
by a line), of every eligible Africa-origin track of this record left without a
candidate, and of every eligible QTrack Atlantic-filter system left without a candidate.
Positions are read from the artifacts, nothing is recomputed, no map projection is drawn.

    .venv/bin/python3 scripts/fig_qtrack_pairing_pilot.py --artifacts <dir> --years 1990 2002 2007 --out <png>
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def first_observations(art):
    """{id: first observation in coverage} for both eligible populations, from the inputs
    the artifact names, refused unless their digests are the artifact's."""
    import hashlib
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import qtrack_pairing_pilot as M
    for key in ("this_record", "qtrack"):
        path = art["inputs"][key]["path"]
        if hashlib.sha256(open(path, "rb").read()).hexdigest() != art["inputs"][key]["sha256"]:
            raise SystemExit(f"REFUSED: {path} is not the file the artifact names")
    window = tuple(art["rules"]["window_days_since_1900"])
    region = tuple(art["rules"]["region_lat_lon"])
    ours, _ = M.load_ours(art["inputs"]["this_record"]["path"])
    q, _ = M.load_qtrack(art["inputs"]["qtrack"]["path"])
    a_el, _, _ = M.populations(ours, window, region)
    b_el, _, _ = M.populations(q, window, region)
    first = lambda t: {"lon": float(t["lon"][0]), "lat": float(t["lat"][0])}
    return {t["id"]: first(t) for t in a_el}, {t["id"]: first(t) for t in b_el}


def render(artifacts, out, tolerance="500", dpi=150):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aew.plotting import panel_label
    fig, axes = plt.subplots(len(artifacts), 1, figsize=(11, 3.2 * len(artifacts)), squeeze=False)
    drawn = {}
    for ax, (year, art) in zip(axes[:, 0], artifacts):
        r = art["results"][tolerance]
        region = art["rules"]["region_lat_lon"]
        africa = set(art["populations"]["a"]["africa_origin_in_season_ids"])
        atlantic = set(art["populations"]["b"]["atlantic_filter_ids"])
        a_none = {u["id"] for u in r["a"]["no_candidate"]} & africa
        b_none = {u["id"] for u in r["b"]["no_candidate"]} & atlantic
        a_first = {p["a"]: p["a_first"] for p in r["pairs"]}
        # the artifact carries first observations only for pairs; the unpaired tracks' positions are
        # read from the inputs the artifact names, verified by digest, through the pilot's own loaders
        # and coverage rule, so they are the same restricted tracks the pairing saw
        firsts_a, firsts_b = first_observations(art)
        for p in r["pairs"]:
            ax.plot([p["a_first"]["lon"], p["b_first"]["lon"]], [p["a_first"]["lat"], p["b_first"]["lat"]], color="#6b6b6b", lw=0.6, zorder=1)
        ax.scatter([p["a_first"]["lon"] for p in r["pairs"]], [p["a_first"]["lat"] for p in r["pairs"]], s=14, color="#1f5fa8", label="this record, paired (first observation)", zorder=3)
        ax.scatter([p["b_first"]["lon"] for p in r["pairs"]], [p["b_first"]["lat"] for p in r["pairs"]], s=14, marker="s", color="#c8552d", label="QTrack, paired", zorder=3)
        xa = [firsts_a[i]["lon"] for i in a_none]
        ya = [firsts_a[i]["lat"] for i in a_none]
        xb = [firsts_b[i]["lon"] for i in b_none]
        yb = [firsts_b[i]["lat"] for i in b_none]
        ax.scatter(xa, ya, s=10, facecolors="none", edgecolors="#1f5fa8", label="this record, Africa-origin, no candidate", zorder=2)
        ax.scatter(xb, yb, s=10, marker="s", facecolors="none", edgecolors="#c8552d", label="QTrack, Atlantic filter, no candidate", zorder=2)
        ax.plot([region[2], region[3], region[3], region[2], region[2]], [region[0], region[0], region[1], region[1], region[0]], color="k", lw=0.6, ls=":")
        ax.set_xlim(region[2] - 3, region[3] + 8)
        ax.set_ylim(region[0] - 3, region[1] + 3)
        ax.set_ylabel("latitude")
        ax.text(0.99, 0.04, f"{year}: {len(r['pairs'])} pairs at {tolerance} km, {len(a_none)} of {len(africa & set(firsts_a))} eligible Africa-origin tracks without a candidate, "
                f"{len(b_none)} of {len(atlantic & set(firsts_b))} eligible Atlantic-filter systems without one",
                transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom")
        panel_label(ax, "abc"[len(drawn)], size=11)
        drawn[year] = {"pairs": len(r["pairs"]), "a_none": len(a_none), "b_none": len(b_none), "drawn_a_none": len(xa), "drawn_b_none": len(xb)}
    axes[-1, 0].set_xlabel("longitude")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.965))   # above the panels, so no annotation is covered
    fig.suptitle(f"Stored-track correspondences, this record against QTrack, first observations in coverage, {tolerance} km", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return drawn


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--tolerance", default="500")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and figures beside artifacts are never overwritten")
    arts = [(y, json.load(open(os.path.join(args.artifacts, f"pairing_{y}_2026-09-27.json")))) for y in args.years]
    drawn = render(arts, args.out, args.tolerance)
    print(f"wrote {args.out}: " + ", ".join(f"{y} {d['pairs']} pairs" for y, d in drawn.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
