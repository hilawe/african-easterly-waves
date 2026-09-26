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

LABEL = {"v1": "ERA-Interim", "port": "ERA5"}
COLOR = {"v1": "#1f5fa8", "port": "#c8552d", "archive": "#6b6b6b"}


def _band_label(key):
    def side(x):
        x = int(x)
        return f"{abs(x)}W" if x < 0 else (f"{x}E" if x > 0 else "0")
    a, b = key.split("..")
    return f"{side(a)} to {side(b)}"


def render(summary, out, dpi=150):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, ag = summary["years"], summary["aggregates"]
    years = np.array(sorted(int(y) for y in rows))
    v1 = np.array([rows[str(y)]["africa_origin"]["v1"] for y in years], float)
    port = np.array([rows[str(y)]["africa_origin"]["port"] for y in years], float)
    arch = np.array([rows[str(y)]["published_context_tracks"] if rows[str(y)]["published_context_tracks"] is not None else np.nan
                     for y in years], float)
    spread = summary.get("published_spread") or ag["archive"].get("interannual_sd")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    (a, b), (c, d) = axes

    a.plot(years, v1, "o-", color=COLOR["v1"], ms=3.5, lw=1.2, label=LABEL["v1"])
    a.plot(years, port, "s-", color=COLOR["port"], ms=3.5, lw=1.2, label=LABEL["port"])
    have = ~np.isnan(arch)
    if have.any():
        a.plot(years[have], arch[have], "^--", color=COLOR["archive"], ms=3.5, lw=1.0, label="archived record, its own rules")
    a.set_ylabel("Africa-origin tracks in season")
    a.set_xlabel("Year")
    a.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.88))

    diff = np.array([rows[str(y)]["africa_origin"]["port_minus_v1"] for y in years], float)
    if not np.array_equal(diff, port - v1):
        raise SystemExit("REFUSED: the summary's stored differences disagree with its counts")
    b.bar(years, diff, color=np.where(diff < 0, COLOR["v1"], COLOR["port"]), width=0.8)
    b.axhline(0, color="k", lw=0.8)
    if spread:
        for s in (spread, -spread):
            b.axhline(s, color=COLOR["archive"], lw=0.8, ls=":")
        b.text(years[-1] + 0.6, spread, "published record interannual spread", fontsize=7, va="bottom", ha="right", color=COLOR["archive"])
    b.set_ylabel("ERA5 minus ERA-Interim, tracks")
    b.set_xlabel("Year")

    lo, hi = min(v1.min(), port.min()) - 5, max(v1.max(), port.max()) + 5
    c.plot([lo, hi], [lo, hi], color="k", lw=0.8, ls="--")
    c.scatter(v1, port, s=18, color=COLOR["port"], edgecolor="k", linewidth=0.4)
    for y, x1, x2 in zip(years, v1, port):
        if abs(x2 - x1) > (spread or 0):
            c.annotate(str(y), (x1, x2), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
    r = ag["africa_origin"]["pearson_correlation_v1_port"]
    c.text(0.96, 0.05, f"Pearson correlation {r:.3f}\n{len(years)} years, means {ag['africa_origin']['mean']['v1']:.1f} and {ag['africa_origin']['mean']['port']:.1f}",
           transform=c.transAxes, fontsize=8, va="bottom", ha="right")
    c.set_xlim(lo, hi)
    c.set_ylim(lo, hi)
    c.set_xlabel("ERA-Interim, tracks in season")
    c.set_ylabel("ERA5, tracks in season")
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
    top = max(mv1 + mport) * 1.35
    d.set_ylim(0, top)
    d.text(1.5, top * 0.96, "by month", ha="center", va="top", fontsize=8)
    d.text(3.5 + (len(labels) - 4) / 2, top * 0.96, "by band of genesis longitude", ha="center", va="top", fontsize=8)
    d.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.9))

    from aew.plotting import panel_label
    for ax, tag in zip((a, b, c, d), "abcd"):
        panel_label(ax, tag, size=11)
        ax.tick_params(labelsize=8)
    fig.suptitle("Protocol campaign, 1979 to 2010, one implementation and one set of rules on both reanalyses", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return {"years": years.tolist(), "v1": v1.tolist(), "port": port.tolist(), "difference": diff.tolist(),
            "archive": arch.tolist(), "group_labels": labels, "group_v1": mv1, "group_port": mport}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and figures beside artifacts are never overwritten")
    summary = json.load(open(args.summary))
    render(summary, args.out, args.dpi)
    print(f"wrote {args.out} from {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
