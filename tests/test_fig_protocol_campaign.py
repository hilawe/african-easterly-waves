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
