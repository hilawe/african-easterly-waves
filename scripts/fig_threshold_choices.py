#!/usr/bin/env python3
"""Annual Africa-origin track counts, 1979 to 2010, under two threshold choices for each
reanalysis, drawn from the retained threshold-experiment summaries only.

Panel (a) is ERA-Interim under its own calibrated pair and under the version 1 constants.
Panel (b) is ERA5 under its own calibrated pair and under ERA-Interim's calibrated pair.
Both panels share one year axis and one count axis. Each panel states the two means, the
mean of the per-year difference (the alternative choice minus the dataset's own
calibrated pair) and the Pearson correlation of the two annual series.

Nothing is drawn until every number is checked. The per-year series are read from the
two experiment summaries. Each is compared year by year with the same series in a third
summary that also holds it (ERA-Interim's calibrated series and ERA5 under ERA-Interim's
pair). The means, the mean difference and the correlation are recomputed from the plotted
series and must equal the summaries' stored aggregates and, at the precision printed in
the manuscript, the values stated there. Any disagreement refuses the figure.

    .venv/bin/python3 scripts/fig_threshold_choices.py --eraint-experiment <json> \
        --era5-experiment <json> --crosscheck <json> --out-stem <path without extension>

writes <stem>.pdf (vector) and <stem>.png (preview), each by exclusive creation.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

YEARS = list(range(1979, 2011))
# The values the manuscript prints (section 4 and Table 2): own mean, alternative mean,
# mean difference, correlation, at the printed precision.
MANUSCRIPT_VALUES = {"eraint": (126.5, 195.9, 69.4, 0.616), "era5": (126.0, 119.6, -6.5, 0.842)}
PANELS = (("eraint", "ERA-Interim", "own calibrated pair", "version 1 constants"),
          ("era5", "ERA5", "own calibrated pair", "ERA-Interim's calibrated pair"))
OWN_STYLE = dict(color="#0072B2", ls="-", marker="o", ms=3.2, lw=1.3)
ALT_STYLE = dict(color="#D55E00", ls=(0, (4, 2)), marker="s", ms=3.2, lw=1.3, mfc="white")


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _series(summary, side):
    rows = summary["years"]
    missing = [y for y in YEARS if str(y) not in rows]
    if missing:
        raise SystemExit(f"REFUSED: the summary lacks years {missing}")
    return np.array([rows[str(y)]["africa_origin"][side] for y in YEARS], dtype=float)


def statistics(own, alt):
    diff = alt - own
    return {"own_mean": float(own.mean()), "alt_mean": float(alt.mean()),
            "mean_difference": float(diff.mean()), "r": float(np.corrcoef(own, alt)[0, 1])}


def verified(eraint_path, era5_path, cross_path, expected=None):
    expected = MANUSCRIPT_VALUES if expected is None else expected
    E = json.load(open(eraint_path))
    F = json.load(open(era5_path))
    X = json.load(open(cross_path))
    data = {"eraint": (_series(E, "v1"), _series(E, "port")),
            "era5": (_series(F, "v1"), _series(F, "port"))}
    if not np.array_equal(data["eraint"][0], _series(X, "v1")):
        raise SystemExit("REFUSED: ERA-Interim's calibrated series differs between the two summaries holding it")
    if not np.array_equal(data["era5"][1], _series(X, "port")):
        raise SystemExit("REFUSED: ERA5 under ERA-Interim's pair differs between the two summaries holding it")
    stats = {}
    for key, summary in (("eraint", E), ("era5", F)):
        own, alt = data[key]
        s = statistics(own, alt)
        A = summary["aggregates"]["africa_origin"]
        stored = (A["mean"]["v1"], A["mean"]["port"], A["difference_port_minus_v1"]["mean"],
                  A["pearson_correlation_v1_port"])
        got = (s["own_mean"], s["alt_mean"], s["mean_difference"], s["r"])
        if not np.allclose(got, stored, rtol=0, atol=1e-9):
            raise SystemExit(f"REFUSED: {key} recomputes {got}, the summary stores {stored}")
        printed = (round(got[0], 1), round(got[1], 1), round(got[2], 1), round(got[3], 3))
        if printed != tuple(expected[key]):
            raise SystemExit(f"REFUSED: {key} gives {printed}, the manuscript prints {tuple(expected[key])}")
        stats[key] = s
    digests = {"eraint_experiment": _sha256(eraint_path), "era5_experiment": _sha256(era5_path),
               "crosscheck": _sha256(cross_path)}
    return data, stats, digests


def render(data, stats, stem, dpi=300):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aew.plotting import panel_label
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.2), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.22, "left": 0.1, "right": 0.985,
                                          "top": 0.955, "bottom": 0.085})
    years = np.array(YEARS)
    for k, (key, name, own_label, alt_label) in enumerate(PANELS):
        ax = axes[k]
        own, alt = data[key]
        s = stats[key]
        ax.plot(years, own, label=own_label, **OWN_STYLE)
        ax.plot(years, alt, label=alt_label, **ALT_STYLE)
        ax.set_title(f"{name}, 1979 to 2010", fontsize=9.5, pad=3)
        ax.set_ylabel("Africa-origin tracks per season", fontsize=8)
        ax.grid(axis="y", color="0.88", lw=0.5)
        ax.tick_params(labelsize=7.5)
        ax.legend(loc="upper right", fontsize=7.5, frameon=True, framealpha=0.9, edgecolor="none",
                  ncol=2, handlelength=2.6)
        diff = s["mean_difference"]
        ax.text(0.99, 0.04,
                f"means {s['own_mean']:.1f} and {s['alt_mean']:.1f}; mean change {diff:+.1f}; "
                f"correlation {s['r']:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, edgecolor="none"))
        panel_label(ax, "ab"[k], size=11)
    axes[0].set_ylim(60, 280)
    axes[0].set_yticks(np.arange(80, 281, 40))
    axes[1].set_xlim(1978, 2011)
    axes[1].set_xticks(np.arange(1980, 2011, 5))
    axes[1].set_xlabel("Year", fontsize=8)
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
    ap.add_argument("--eraint-experiment", required=True)
    ap.add_argument("--era5-experiment", required=True)
    ap.add_argument("--crosscheck", required=True)
    ap.add_argument("--out-stem", required=True)
    args = ap.parse_args(argv)
    data, stats, digests = verified(args.eraint_experiment, args.era5_experiment, args.crosscheck)
    try:
        written = render(data, stats, args.out_stem)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {args.out_stem}.* exists and figures beside artifacts are never overwritten")
    for key, s in stats.items():
        print(f"{key}: means {s['own_mean']:.4f} and {s['alt_mean']:.4f}, mean change "
              f"{s['mean_difference']:+.4f}, correlation {s['r']:.4f}")
    print("inputs " + ", ".join(f"{k} {v[:8]}" for k, v in digests.items()) + "; wrote " + ", ".join(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
