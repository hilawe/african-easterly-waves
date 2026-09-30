#!/usr/bin/env python3
"""The principal figure of the intercomparison, from retained artifacts only: (a) the
annual cohort and Atlantic-side counts of both records over the thirty common years,
so the denominators are visible; (b) the pooled Atlantic-side fraction by predeclared
band of first longitude, each bar labeled with its numerator and denominator; (c) the
stratum 10 E to 30 E, 5 to 15 N by month of first observation, likewise labeled. Every
number is a count of stored tracks under the descriptive measurement, and nothing here
is a physical validation. The origin re-tabulation and the monthly tabulation are read
after their season-artifact digests are verified against each other.

    .venv/bin/python3 scripts/fig_intercomparison_principal.py --origin <json> --by-month <json> --out <png>
"""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

BANDS = ("west of 10 W", "10 W to 10 E", "10 E to 30 E", "east of 30 E")
MONTHS = (("6", "June"), ("7", "July"), ("8", "August"), ("9", "September"))
COLORS = {"this_record": "#1f5fa8", "qtrack": "#c8552d"}
LABELS = {"this_record": "this record", "qtrack": "QTrack"}


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def verified(origin_path, by_month_path):
    origin = json.load(open(origin_path))
    by_month = json.load(open(by_month_path))
    if by_month["inputs"]["origin"]["sha256"] != _sha256(origin_path):
        raise SystemExit("REFUSED: the monthly tabulation was not made from this origin artifact")
    if by_month["years"] != origin["years"]:
        raise SystemExit("REFUSED: the two artifacts cover different years")
    return origin, by_month


def annual(origin):
    years = [str(y) for y in origin["years"]]
    out = {n: {"cohort": [], "atlantic_side": []} for n in COLORS}
    for y in years:
        for n in COLORS:
            tot = origin["seasons"][y][n]["by_longitude_band"]["total"]
            out[n]["cohort"].append(tot["cohort"])
            out[n]["atlantic_side"].append(tot["atlantic_side"])
    return years, out


def render(origin, by_month, out, dpi=160, size=(9.5, 11.5), suptitle=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from aew.plotting import panel_label
    years, ann = annual(origin)
    plt.rcParams.update({"axes.unicode_minus": False})
    compact = size[0] <= 7.0                          # drawn at the printed width
    labels = dict(LABELS, this_record="version 2") if compact else LABELS
    ylab = "percent on the Atlantic side" if compact else "recorded on the Atlantic side, percent of cohort"
    if compact:
        plt.rcParams.update({"axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    fig, axes = plt.subplots(3, 1, figsize=size)
    # (a) annual counts with denominators
    ax = axes[0]
    x = np.arange(len(years))
    for i, n in enumerate(COLORS):
        off = -0.2 if i == 0 else 0.2
        ax.bar(x + off, ann[n]["cohort"], width=0.4, color=COLORS[n], alpha=0.3, label=f"{labels[n]}, cohort")
        ax.bar(x + off, ann[n]["atlantic_side"], width=0.4, color=COLORS[n], label=f"{labels[n]}, recorded on the Atlantic side")
    ax.set_xticks(x[::3])
    ax.set_xticklabels(years[::3], fontsize=8)
    ax.set_ylabel("stored tracks per season")
    ax.set_ylim(0, 1.3 * max(max(ann[n]["cohort"]) for n in COLORS))            # room for the legend above the tallest bar
    ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper right")
    panel_label(ax, "a", size=11)
    # (b) pooled fraction by band with numerator and denominator
    ax = axes[1]
    xb = np.arange(len(BANDS))
    for i, n in enumerate(COLORS):
        off = -0.2 if i == 0 else 0.2
        rows = origin["pooled_all_years"][n]["by_longitude_band"]
        fr = [100.0 * rows[b]["atlantic_side"] / rows[b]["cohort"] if rows[b]["cohort"] else 0.0 for b in BANDS]
        ax.bar(xb + off, fr, width=0.4, color=COLORS[n], label=labels[n])
        for k, b in enumerate(BANDS):
            ax.text(xb[k] + off, fr[k] + 1.5, f"{rows[b]['atlantic_side']} of {rows[b]['cohort']}", ha="center", fontsize=7)
    ax.set_xticks(xb)
    ax.set_xticklabels([b.replace(" W", " W").replace(" E", " E") for b in BANDS], fontsize=8.5)
    ax.set_ylabel(ylab)
    ax.set_ylim(0, 125 if compact else 110)
    ax.set_xlabel("band of first recorded longitude, thirty seasons pooled")
    ax.legend(fontsize=8, frameon=False)
    panel_label(ax, "b", size=11)
    # (c) the stratum by month
    ax = axes[2]
    xm = np.arange(len(MONTHS))
    for i, n in enumerate(COLORS):
        off = -0.2 if i == 0 else 0.2
        rows = by_month["by_month"][n]
        fr = [100.0 * rows[m]["atlantic_side"] / rows[m]["cohort"] if rows[m]["cohort"] else 0.0 for m, _ in MONTHS]
        ax.bar(xm + off, fr, width=0.4, color=COLORS[n], label=labels[n])
        for k, (m, _) in enumerate(MONTHS):
            ax.text(xm[k] + off, fr[k] + 1.5, f"{rows[m]['atlantic_side']} of {rows[m]['cohort']}", ha="center", fontsize=7)
    ax.set_xticks(xm)
    ax.set_xticklabels([lab for _, lab in MONTHS], fontsize=8.5)
    ax.set_ylabel(ylab)
    ax.set_ylim(0, 80)
    ax.set_xlabel("month of first observation, starts in 10 E to 30 E and 5 to 15 N, thirty seasons pooled")
    ax.legend(fontsize=8, frameon=False)
    panel_label(ax, "c", size=11)
    if suptitle:
        fig.suptitle("A descriptive intercomparison of stored tracks, 1981 to 2010: counts, denominators and the origin-dependent contrast", fontsize=9.5)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()
    fmt = os.path.splitext(out)[1].lstrip(".") or "png"
    with open(out, "xb") as fh:                                                 # exclusive creation: never overwritten, even by a concurrent run
        fig.savefig(fh, format=fmt, dpi=dpi)
    plt.close(fig)
    return {"years": len(years), "bands": len(BANDS), "months": len(MONTHS)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--origin", required=True)
    ap.add_argument("--by-month", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=float, default=9.5, help="figure width in inches, where 6.5 prints at full text width unscaled")
    ap.add_argument("--height", type=float, default=11.5)
    ap.add_argument("--no-suptitle", action="store_true", help="leave the title to the caption")
    args = ap.parse_args(argv)
    origin, by_month = verified(args.origin, args.by_month)
    try:
        drawn = render(origin, by_month, args.out, size=(args.width, args.height), suptitle=not args.no_suptitle)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {args.out} exists and figures beside artifacts are never overwritten")
    print(f"wrote {args.out}: {drawn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
