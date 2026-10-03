"""The campaign figure: rendered from a summary alone, it writes one file, refuses to
overwrite it, and annotates the summary's own correlation rather than recomputing one."""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _load():
    path = os.environ.get("FIG_SCRIPT", os.path.join(ROOT, "scripts", "fig_protocol_campaign.py"))
    spec = importlib.util.spec_from_file_location("fig_protocol_campaign", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _summary():
    years = {}
    for i, y in enumerate(range(1990, 1994)):
        years[str(y)] = {"africa_origin": {"v1": 100 + 10 * i, "port": 90 + 15 * i, "port_minus_v1": 5 * i - 10},
                         "published_context_tracks": 150 + i if i < 3 else None}
    return {"years": years, "aggregates": {
        "africa_origin": {"mean": {"v1": 115.0, "port": 112.5}, "pearson_correlation_v1_port": 0.123},
        "archive": {"interannual_sd": 17.5},
        "months_mean_starts": {m: {"v1": 30.0, "port": 31.0} for m in ("6", "7", "8", "9")},
        "bands_mean_starts": {"-10..+0": {"v1": 14.0, "port": 13.0}, "+0..+10": {"v1": 13.0, "port": 12.0},
                              "-60..-50": {"v1": 0.0, "port": 0.0}}}}


def test_figure_is_written_once_and_never_overwritten(tmp_path, monkeypatch):
    F = _load()
    s = tmp_path / "summary.json"
    s.write_text(json.dumps(_summary()))
    out = tmp_path / "fig.png"
    assert F.main(["--summary", str(s), "--out", str(out), "--dpi", "60"]) == 0
    assert out.exists() and out.stat().st_size > 1000
    with pytest.raises(SystemExit):
        F.main(["--summary", str(s), "--out", str(out)])


def test_the_drawn_values_are_the_summary_values_and_a_disagreeing_difference_is_refused(tmp_path):
    F = _load()
    drawn = F.render(_summary(), str(tmp_path / "f.png"), dpi=60)
    assert drawn["v1"] == [100, 110, 120, 130] and drawn["port"] == [90, 105, 120, 135]
    assert drawn["difference"] == [-10, -5, 0, 5] and drawn["archive"][:3] == [150, 151, 152]
    assert drawn["group_labels"] == ["Jun", "Jul", "Aug", "Sep", "10W to 0", "0 to 10E"]      # bands west to east, empty ones dropped
    assert drawn["group_v1"] == [30.0] * 4 + [14.0, 13.0] and drawn["group_port"] == [31.0] * 4 + [13.0, 12.0]
    s = _summary()
    s["years"]["1991"]["africa_origin"]["port_minus_v1"] = -4
    with pytest.raises(SystemExit):
        F.render(s, str(tmp_path / "g.png"), dpi=60)


def test_the_panel_label_and_the_legend_do_not_overlap_in_panel_a(tmp_path, monkeypatch):
    F = _load()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    captured = {}
    original = plt.subplots

    def keep(*a, **k):
        fig, axes = original(*a, **k)
        captured["fig"], captured["axes"] = fig, axes
        return fig, axes
    monkeypatch.setattr(plt, "subplots", keep)
    monkeypatch.setattr(plt, "close", lambda fig: None)
    F.render(_summary(), str(tmp_path / "f.png"), dpi=60)
    fig, a = captured["fig"], captured["axes"][0][0]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    label = [t for t in a.texts if t.get_text() == "(a)"][0].get_window_extent(r)
    legend = a.get_legend().get_window_extent(r)
    assert not label.overlaps(legend)


def test_at_most_ten_years_are_annotated_and_the_spread_caption_avoids_the_bars(tmp_path, monkeypatch):
    F = _load()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    captured = {}
    original = plt.subplots

    def keep(*a, **k):
        fig, axes = original(*a, **k)
        captured["axes"] = axes
        return fig, axes
    monkeypatch.setattr(plt, "subplots", keep)
    monkeypatch.setattr(plt, "close", lambda fig: None)
    s = _summary()
    for i, y in enumerate(range(1990, 2010)):                                     # twenty years, every one far above the spread
        s["years"][str(y)] = {"africa_origin": {"v1": 100 + i, "port": 180 + i, "port_minus_v1": 80}, "published_context_tracks": None}
    F.render(s, str(tmp_path / "f.png"), dpi=60)
    (a, b), (c, d) = captured["axes"]
    assert len([t for t in c.texts if t.get_text().isdigit()]) == 10
    caption = [t for t in b.texts if "spread" in t.get_text()][0]
    assert caption.get_position()[1] < 0                                           # all bars positive, so the caption sits on the lower line
    F.render(_summary(), str(tmp_path / "g.png"), dpi=60)
    (a, b), (c, d) = captured["axes"]
    assert [t for t in b.texts if "spread" in t.get_text()][0].get_position()[1] > 0   # mixed signs, the upper line as before


def test_side_labels_name_the_axes_and_default_to_the_reanalyses(tmp_path, monkeypatch):
    F = _load()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    captured = {}
    original = plt.subplots

    def keep(*a, **k):
        fig, axes = original(*a, **k)
        captured["axes"] = axes
        return fig, axes
    monkeypatch.setattr(plt, "subplots", keep)
    monkeypatch.setattr(plt, "close", lambda fig: None)
    F.render(_summary(), str(tmp_path / "a.png"), dpi=60)
    (a, b), (c, d) = captured["axes"]
    assert b.get_ylabel() == "ERA5 minus ERA-Interim, tracks" and c.get_xlabel().startswith("ERA-Interim")
    F.render(_summary(), str(tmp_path / "b.png"), dpi=60, labels={"v1": "Rule A", "port": "archived constants"}, title="t")
    (a, b), (c, d) = captured["axes"]
    assert b.get_ylabel() == "archived constants minus Rule A, tracks"
    assert c.get_xlabel() == "Rule A, tracks in season" and c.get_ylabel() == "archived constants, tracks in season"
    assert [t.get_text() for t in a.get_legend().get_texts()][:2] == ["Rule A", "archived constants"]


def test_the_annotated_correlation_is_the_summary_value(tmp_path, monkeypatch):
    F = _load()
    seen = []
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes
    original = Axes.text

    def spy(self, *a, **k):
        seen.append(a[2] if len(a) > 2 else k.get("s"))
        return original(self, *a, **k)
    monkeypatch.setattr(Axes, "text", spy)
    F.render(_summary(), str(tmp_path / "f.png"), dpi=60)
    assert any(isinstance(t, str) and "Pearson correlation 0.123" in t for t in seen)


def _axes_with(points):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(3, 3))
    pts = np.array(points, float)
    ax.scatter(pts[:, 0], pts[:, 1], s=18)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    fig.canvas.draw()
    return fig, ax, pts


def test_a_year_label_moves_off_a_neighbor_it_would_cover():
    """The specimen is 1986 beside 1987 under the 32-year archive spread: the first corner
    put the label across the neighbor, and another corner is clear."""
    M = _load()
    fig, ax, pts = _axes_with([[10, 10], [11.0, 10.5]])          # the neighbor sits up and right
    (t,) = M.label_years(fig, ax, pts, [(1986, 10, 10)])
    bb = t.get_window_extent(fig.canvas.get_renderer())
    nx, ny = ax.transData.transform(pts[1])
    assert not (bb.x0 <= nx <= bb.x1 and bb.y0 <= ny <= bb.y1)
    assert tuple(t.xyann) != (3, 2)


def test_an_unobstructed_label_keeps_the_original_corner():
    M = _load()
    fig, ax, pts = _axes_with([[10, 10], [2, 18]])
    (t,) = M.label_years(fig, ax, pts, [(1990, 10, 10)])
    assert tuple(t.xyann) == (3, 2) and t.get_ha() == "left"


def test_two_labels_never_overlap_when_a_corner_is_free():
    M = _load()
    # two points stacked half a unit apart: neither label covers the other point, but at the
    # first corner the two labels overlap each other, as 1989 and 2007 did in the v3 figure
    fig, ax, pts = _axes_with([[10, 10], [10, 10.5]])
    a, b = M.label_years(fig, ax, pts, [(1989, 10, 10), (2007, 10, 10.5)])
    r = fig.canvas.get_renderer()
    assert not a.get_window_extent(r).overlaps(b.get_window_extent(r))


def test_a_label_with_no_clear_corner_is_reported_not_called_clear():
    """Four corners cannot clear a crowd: the label takes the least covered corner and
    carries how much it covers, so the figure can say so."""
    M = _load()
    crowd = [[10, 10]] + [[10 + dx, 10 + dy] for dx in (-0.9, -0.5, 0.5, 0.9) for dy in (-0.4, 0.4)]
    fig, ax, pts = _axes_with(crowd)
    (t,) = M.label_years(fig, ax, pts, [(1994, 10, 10)])
    assert t.obstructed_by > 0


def test_a_clear_label_reports_nothing():
    M = _load()
    fig, ax, pts = _axes_with([[10, 10], [2, 18]])
    (t,) = M.label_years(fig, ax, pts, [(1990, 10, 10)])
    assert t.obstructed_by == 0


def test_every_label_that_covers_a_point_or_label_reports_it():
    """The invariant over many crowded layouts: a final label whose box overlaps another
    final label, or covers a point other than its own, has a positive obstruction count.
    Counting at placement time missed an earlier label covered by a later one."""
    import numpy as np
    M = _load()
    rng = np.random.default_rng(20261002)
    for _ in range(40):
        pts_list = (10 + rng.normal(0, 0.6, size=(10, 2))).tolist()
        fig, ax, pts = _axes_with(pts_list)
        texts = M.label_years(fig, ax, pts, [(1980 + i, x, y) for i, (x, y) in enumerate(pts_list[:6])])
        r = fig.canvas.get_renderer()
        boxes = [t.get_window_extent(r) for t in texts]
        for i, (t, bb) in enumerate(zip(texts, boxes)):
            if any(j != i and bb.overlaps(o) for j, o in enumerate(boxes)):
                assert t.obstructed_by > 0, t.get_text()
        import matplotlib.pyplot as plt
        plt.close(fig)


CROWD = {1980: (98, 127), 1981: (105, 117), 1982: (92, 126), 1983: (93, 113), 1984: (78, 121), 1985: (93, 115),
         1986: (100, 128), 1987: (96, 125), 1988: (98, 140), 1989: (98, 128), 1990: (91, 104), 1991: (101, 126)}


def test_labels_are_counted_against_the_geometry_the_figure_is_saved_with(tmp_path, monkeypatch):
    """A confirmation review's specimen: the scatter panel's equal aspect is applied only at
    draw time, so labels placed and counted before drawing could cover a point in the saved
    figure while reporting nothing. Every label covering a point in the SAVED geometry must
    be among those the render reports."""
    import numpy as np
    from matplotlib.figure import Figure
    M = _load()
    summary = _summary()
    summary["years"] = {str(y): {"africa_origin": {"v1": a, "port": b, "port_minus_v1": b - a}, "published_context_tracks": None}
                        for y, (a, b) in CROWD.items()}
    summary["published_spread"] = 1
    unreported, original = [], Figure.savefig

    def save_then_audit(fig, *args, **kwargs):
        original(fig, *args, **kwargs)
        fig.canvas.draw()
        ax = fig.axes[2]
        r = fig.canvas.get_renderer()
        labels = [t for t in ax.texts if t.get_text().isdigit()]
        pts = ax.transData.transform(ax.collections[0].get_offsets())
        radius = (18 ** 0.5) / 2 * fig.dpi / 72
        for t in labels:
            bb, own = t.get_window_extent(r), ax.transData.transform(t.xy)
            covers = any(not np.allclose(p, own, atol=1e-6) and bb.x0 - radius <= p[0] <= bb.x1 + radius
                         and bb.y0 - radius <= p[1] <= bb.y1 + radius for p in pts)
            if covers and not t.obstructed_by:
                unreported.append(t.get_text())
    monkeypatch.setattr(Figure, "savefig", save_then_audit)
    drawn = M.render(summary, str(tmp_path / "crowd.png"), dpi=100, size=(6.5, 4.4), title="")
    assert unreported == [], f"labels covering a point in the saved figure but not reported: {unreported}"
    assert set(drawn["labels_obstructed"]) <= {str(y) for y in CROWD}


def _audit_saved_labels(M, summary, path, dpi, size, monkeypatch):
    """Render, and during the actual save draw list every year label that covers a point,
    another label or other panel text without being reported (a review's instrument)."""
    import numpy as np
    from matplotlib.figure import Figure
    original, false_clear = Figure.savefig, []

    def save_audit(fig, *a, **k):
        def audit(event):
            ax, r = fig.axes[2], event.renderer
            labels = [t for t in ax.texts if t.get_text().isdigit()]
            other = [t for t in ax.texts if not t.get_text().isdigit()]
            pts = ax.transData.transform(ax.collections[0].get_offsets())
            radius = 18 ** .5 / 2 * fig.dpi / 72
            boxes = [t.get_window_extent(r) for t in labels]
            for i, (t, bb) in enumerate(zip(labels, boxes)):
                own = ax.transData.transform(t.xy)
                covers = any(not np.allclose(p, own, atol=1e-6, rtol=0) and bb.x0 - radius <= p[0] <= bb.x1 + radius
                             and bb.y0 - radius <= p[1] <= bb.y1 + radius for p in pts)
                overlaps = any(j != i and bb.overlaps(boxes[j]) for j in range(len(boxes))) \
                    or any(bb.overlaps(o.get_window_extent(r)) for o in other)
                if (covers or overlaps) and not t.obstructed_by:
                    false_clear.append(t.get_text())
        cid = fig.canvas.mpl_connect("draw_event", audit)
        try:
            return original(fig, *a, **k)
        finally:
            fig.canvas.mpl_disconnect(cid)
    monkeypatch.setattr(Figure, "savefig", save_audit)
    M.render(summary, path, dpi=dpi, size=size, title="")
    monkeypatch.setattr(Figure, "savefig", original)
    return false_clear


@pytest.mark.parametrize("dpi", [40, 100])
def test_no_saved_label_overlap_goes_unreported_at_the_saved_resolution(tmp_path, monkeypatch, dpi):
    """Labels were measured at the default 100 dpi and saved at another, and the panel's
    other text was not an obstacle (a review found 1981 over 1989 at 40 dpi, and 2007 over
    the correlation note). Seeded 32-year summaries, two sizes, audited at save time."""
    import numpy as np
    M = _load()
    for seed in range(4):
        rng = np.random.default_rng(seed)
        v1 = np.round(rng.normal(125, 14, 32))
        port = np.round(v1 + rng.normal(0, 20, 32))
        summary = _summary()
        summary["years"] = {str(1979 + i): {"africa_origin": {"v1": float(a), "port": float(b), "port_minus_v1": float(b - a)},
                                            "published_context_tracks": None} for i, (a, b) in enumerate(zip(v1, port))}
        for size in ((6.5, 4.4), (6.5, 5.4)):
            unreported = _audit_saved_labels(M, summary, str(tmp_path / f"s{seed}_{size[1]}_{dpi}.png"), dpi, size, monkeypatch)
            assert unreported == [], f"seed {seed}, size {size}, {dpi} dpi: {unreported}"
