#!/usr/bin/env python3
"""One compact figure of the protocol campaign, read from the campaign summary.

Four panels: (a) Africa-origin season tracks by year on both reanalyses, with the
archive's count where it has one; (b) the paired difference by year, ERA5 minus
ERA-Interim, with the published record's interannual spread marked; (c) ERA5 against
ERA-Interim by year with the one-to-one line and the Pearson correlation the summary
records; (d) mean starts by month and by ten-degree band of genesis longitude on both
sides. Every value is read from the summary, nothing is recomputed except the axis
limits, and the figure annotates the summary's own correlation rather than a fresh one.
No trend line and no test is drawn, since none was run.

    .venv/bin/python3 scripts/fig_protocol_campaign.py --summary <json> --out <png>
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

LABEL = {"v1": "ERA-Interim", "port": "ERA5"}       # the campaign's sides; the sensitivity passes its own
COLOR = {"v1": "#1f5fa8", "port": "#c8552d", "archive": "#6b6b6b"}


def _band_label(key):
    def side(x):
        x = int(x)
        return f"{abs(x)}W" if x < 0 else (f"{x}E" if x > 0 else "0")
    a, b = key.split("..")
    return f"{side(a)} to {side(b)}"


# the positions a year label may take around its point, tried in order; the first is where
# every label went before 2026-10-02, kept identical so a figure without a collision does
# not change
LABEL_CORNERS = (dict(xytext=(3, 2)),
                 dict(xytext=(-3, 2), ha="right"),
                 dict(xytext=(3, -2), va="top"),
                 dict(xytext=(-3, -2), ha="right", va="top"),
                 # four more, beside and above or below, for a crowd the corners cannot clear
                 # (the settled 32-year figure left 1981 and 2005 with no clear corner)
                 dict(xytext=(5, 0), va="center"),
                 dict(xytext=(-5, 0), ha="right", va="center"),
                 dict(xytext=(0, 5), ha="center"),
                 dict(xytext=(0, -5), ha="center", va="top"))


def label_years(fig, ax, points, labeled, marker_size=18, fontsize=6.5, obstacles=()):
    """Annotate each (year, x, y) in `labeled` at the first corner of LABEL_CORNERS whose
    text box covers no other plotted point and no label already placed, so a label never
    sits on data. When every corner is blocked, the corner covering the fewest is used.
    The 32-year archive spread first labeled 1986, beside 1987, and the fixed corner put the
    label across both points. THE REQUIREMENT, after three repairs: in the SAVED figure no
    year label covers a point (within one marker radius) or another label, and when that is
    impossible the render names every label that does. Eight positions cannot clear every crowd, so each placed label
    carries `obstructed_by`, the count of points and labels it covers, and the caller
    reports any label above zero rather than calling the figure clear. Returns the text
    artists placed, so a caller can place them again once the layout is final."""
    renderer = fig.canvas.get_renderer()
    pts = ax.transData.transform(points)
    # other text already in the panel (the correlation note, the panel letter) is in the way
    # too: a review found 2007 over the correlation note with nothing reported
    fixed = [o.get_window_extent(renderer) for o in obstacles]
    radius = (marker_size ** 0.5) / 2 * fig.dpi / 72.0                       # marker radius in pixels
    placed, texts = [], []
    for y, x1, x2 in labeled:
        own = ax.transData.transform((x1, x2))
        best = None
        for corner in LABEL_CORNERS:
            t = ax.annotate(str(y), (x1, x2), fontsize=fontsize, textcoords="offset points", **corner)
            bb = t.get_window_extent(renderer)
            hits = sum(1 for p in pts if not (abs(p[0] - own[0]) < 1e-6 and abs(p[1] - own[1]) < 1e-6)
                       and bb.x0 - radius <= p[0] <= bb.x1 + radius and bb.y0 - radius <= p[1] <= bb.y1 + radius)
            hits += sum(1 for other in placed if bb.overlaps(other)) + sum(1 for o in fixed if bb.overlaps(o))
            if best is None or hits < best[0]:
                if best is not None:
                    best[1].remove()
                best = (hits, t, bb)
            else:
                t.remove()
            if hits == 0:
                break
        placed.append(best[2])
        texts.append(best[1])
    # a later label can land on an earlier one that was clear when placed, so what each
    # label covers is counted once more against every point and every other final label
    boxes = [t.get_window_extent(renderer) for t in texts]
    for i, (t, bb) in enumerate(zip(texts, boxes)):
        own = ax.transData.transform(t.xy)
        t.obstructed_by = sum(1 for p in pts if not (abs(p[0] - own[0]) < 1e-6 and abs(p[1] - own[1]) < 1e-6)
                              and bb.x0 - radius <= p[0] <= bb.x1 + radius and bb.y0 - radius <= p[1] <= bb.y1 + radius) \
            + sum(1 for j, other in enumerate(boxes) if j != i and bb.overlaps(other)) \
            + sum(1 for o in fixed if bb.overlaps(o))
    return texts


def render(summary, out, dpi=150, labels=None, title=None, size=(11, 7.5)):
    """`labels` names side A ("v1") and side B ("port"); the campaign's defaults are the
    two reanalyses, the sensitivity passes the two threshold pairs. Every axis label and
    the difference's sense (B minus A) follow from them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    LABEL = dict(globals()["LABEL"], **(labels or {}))
    rows, ag = summary["years"], summary["aggregates"]
    years = np.array(sorted(int(y) for y in rows))
    v1 = np.array([rows[str(y)]["africa_origin"]["v1"] for y in years], float)
    port = np.array([rows[str(y)]["africa_origin"]["port"] for y in years], float)
    arch = np.array([rows[str(y)]["published_context_tracks"] if rows[str(y)]["published_context_tracks"] is not None else np.nan
                     for y in years], float)
    spread = summary.get("published_spread") or ag["archive"].get("interannual_sd")

    plt.rcParams.update({"axes.unicode_minus": False})
    compact = size[0] <= 7.0                          # at the printed width, labels shorten and corners stay clear
    if compact:
        plt.rcParams.update({"axes.labelsize": 8.5})
    fig, axes = plt.subplots(2, 2, figsize=size)
    (a, b), (c, d) = axes

    a.plot(years, v1, "o-", color=COLOR["v1"], ms=3.5, lw=1.2, label=LABEL["v1"])
    a.plot(years, port, "s-", color=COLOR["port"], ms=3.5, lw=1.2, label=LABEL["port"])
    have = ~np.isnan(arch)
    if have.any():
        a.plot(years[have], arch[have], "^--", color=COLOR["archive"], ms=3.5, lw=1.0, label="archived record, its own rules")
    a.set_ylabel("Africa-origin tracks in season")
    a.set_xlabel("Year")
    if compact:
        a.set_ylim(top=max(np.nanmax(arch) if have.any() else 0, v1.max(), port.max()) * 1.25)
        a.legend(fontsize=7, frameon=False, loc="upper right", ncol=1, handlelength=1.8, borderaxespad=0.3)
    else:
        a.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.88))

    diff = np.array([rows[str(y)]["africa_origin"]["port_minus_v1"] for y in years], float)
    if not np.array_equal(diff, port - v1):
        raise SystemExit("REFUSED: the summary's stored differences disagree with its counts")
    b.bar(years, diff, color=np.where(diff < 0, COLOR["v1"], COLOR["port"]), width=0.8)
    b.axhline(0, color="k", lw=0.8)
    if spread:
        for s in (spread, -spread):
            b.axhline(s, color=COLOR["archive"], lw=0.8, ls=":")
        # the caption sits on whichever spread line the bars leave clear
        clear = -spread if diff.min() > -spread and diff.max() > spread else spread
        if compact:                                   # a key in the clear lower-left corner, not a label on a line
            b.set_ylim(bottom=min(diff.min(), -spread) * 1.3)
            b.text(0.02, 0.03, "dotted, archived record's spread", transform=b.transAxes, fontsize=6.5,
                   va="bottom", ha="left", color=COLOR["archive"])
        else:
            b.text(years[-1] + 0.6, clear, "published record interannual spread", fontsize=7,
                   va="bottom" if clear > 0 else "top", ha="right", color=COLOR["archive"])
    b.set_ylabel(f"{LABEL['port']} minus {LABEL['v1']}" + ("" if compact else ", tracks"))
    b.set_xlabel("Year")

    lo, hi = min(v1.min(), port.min()) - 5, max(v1.max(), port.max()) + 5
    if compact:
        lo, hi = lo - 12, hi + 12                     # room for the corner label and the annotation
    c.plot([lo, hi], [lo, hi], color="k", lw=0.8, ls="--")
    c.scatter(v1, port, s=18, color=COLOR["port"], edgecolor="k", linewidth=0.4)
    beyond = [(abs(x2 - x1), y, x1, x2) for y, x1, x2 in zip(years, v1, port) if abs(x2 - x1) > (spread or 0)]
    to_label = [(y, x1, x2) for _, y, x1, x2 in sorted(beyond, reverse=True)[:10]]   # the ten largest, so a wholesale shift stays legible
    year_texts = label_years(fig, c, np.column_stack([v1, port]), to_label)
    r = ag["africa_origin"]["pearson_correlation_v1_port"]
    c.text(0.96, 0.05, f"Pearson correlation {r:.3f}\n{len(years)} years, means {ag['africa_origin']['mean']['v1']:.1f} and {ag['africa_origin']['mean']['port']:.1f}",
           transform=c.transAxes, fontsize=7 if compact else 8, va="bottom", ha="right")
    c.set_xlim(lo, hi)
    c.set_ylim(lo, hi)
    c.set_xlabel(f"{LABEL['v1']}, tracks in season")
    c.set_ylabel(f"{LABEL['port']}, tracks in season")
    c.set_aspect("equal")

    months = ag["months_mean_starts"]
    bands = {k: v for k, v in ag["bands_mean_starts"].items() if v["v1"] > 0 or v["port"] > 0}
    band_keys = sorted(bands, key=lambda k: int(k.split("..")[0]))
    labels = [{"6": "Jun", "7": "Jul", "8": "Aug", "9": "Sep"}[m] for m in ("6", "7", "8", "9")] + [_band_label(k) for k in band_keys]
    mv1 = [months[m]["v1"] for m in ("6", "7", "8", "9")] + [bands[k]["v1"] for k in band_keys]
    mport = [months[m]["port"] for m in ("6", "7", "8", "9")] + [bands[k]["port"] for k in band_keys]
    x = np.arange(len(labels))
    d.bar(x - 0.2, mv1, 0.4, color=COLOR["v1"], label=LABEL["v1"])
    d.bar(x + 0.2, mport, 0.4, color=COLOR["port"], label=LABEL["port"])
    d.axvline(3.5, color="k", lw=0.6)
    d.set_xticks(x)
    d.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    d.set_ylabel("Mean starts per year")
    top = max(mv1 + mport) * (1.75 if compact else 1.35)
    d.set_ylim(0, top)
    ty = top * (0.80 if compact else 0.96)
    d.text(1.5, ty, "by month", ha="center", va="top", fontsize=7 if compact else 8)
    d.text(3.5 + (len(labels) - 4) / 2, ty, "by band of genesis longitude", ha="center", va="top", fontsize=7 if compact else 8)
    if compact:
        d.legend(fontsize=7, frameon=False, loc="upper right", ncol=2, columnspacing=1.0, handlelength=1.5)
    else:
        d.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.9))

    from aew.plotting import panel_label
    for ax, tag in zip((a, b, c, d), "abcd"):
        panel_label(ax, tag, size=11)
        ax.tick_params(labelsize=7.5 if compact else 8)
    if title != "":                                   # an empty title leaves the caption to carry it
        fig.suptitle(title or "Protocol campaign, 1979 to 2010, one implementation and one set of rules on both reanalyses", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()
    # the layout moved the axes, so the year labels are placed again against the final
    # geometry. Drawing first settles it, because the scatter panel's equal aspect is only
    # applied at draw time, and a confirmation review found a label counted clear before
    # that covering a point after it.
    # and at the resolution it is saved at, since text extents do not scale exactly with dpi
    # (a review found 1981 over 1989 at 40 dpi with both counted clear at the default 100)
    fig.set_dpi(dpi)
    fig.canvas.draw()
    for t in year_texts:
        t.remove()
    final_labels = label_years(fig, c, np.column_stack([v1, port]), to_label, obstacles=list(c.texts))
    obstructed = [t.get_text() for t in final_labels if t.obstructed_by]
    fmt = os.path.splitext(out)[1].lstrip(".") or "png"
    with open(out, "xb") as fh:                       # exclusive creation, never overwritten
        fig.savefig(fh, format=fmt, dpi=dpi)
    plt.close(fig)
    return {"years": years.tolist(), "v1": v1.tolist(), "port": port.tolist(), "difference": diff.tolist(),
            "archive": arch.tolist(), "group_labels": labels, "group_v1": mv1, "group_port": mport,
            "labels_obstructed": obstructed}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--side-a", default=None, help="the label of side A (v1), the campaign's default is ERA-Interim")
    ap.add_argument("--side-b", default=None, help="the label of side B (port), the campaign's default is ERA5")
    ap.add_argument("--title", default=None, help='the in-figure title, where "" draws none')
    ap.add_argument("--width", type=float, default=11.0, help="figure width in inches, where 6.5 prints at full text width unscaled")
    ap.add_argument("--height", type=float, default=7.5)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and figures beside artifacts are never overwritten")
    summary = json.load(open(args.summary))
    labels = {k: v for k, v in (("v1", args.side_a), ("port", args.side_b)) if v}
    drawn = render(summary, args.out, args.dpi, labels=labels, title=args.title, size=(args.width, args.height))
    print(f"wrote {args.out} from {args.summary}")
    if drawn["labels_obstructed"]:
        print("NOTE: no clear corner was found for the year labels " + ", ".join(drawn["labels_obstructed"])
              + ", which each cover a point or another label. Check panel (c) before using the figure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
