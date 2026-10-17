"""The filtered seasonal view: three focused examples through the study's own filter
(a steady standing input attenuated, an oscillating standing input keeping its fixed
longitude and its variation, a one-day pulse answering over a longer span), the wrapper's
whole-window rule and unavailable counts, and figures that carry the footprint and nothing
of the tracker. Passing these verifies the implementation under these examples. It
classifies no real streak."""
import importlib.util
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("filtered_view_under_test", os.path.join(ROOT, "scripts", "pilot_season_filtered_view.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _grid(F):
    lons = np.arange(20.0, 75.5, 0.5)
    t = np.arange(488) * F.P.DT_DAYS
    return lons, t


def test_a_steady_standing_input_is_strongly_attenuated():
    F = _load()
    lons, t = _grid(F)
    field = 5.0 * np.exp(-((lons[None, :] - 50.0) / 5.0) ** 2) * np.ones((t.size, 1))   # constant in time, fixed longitude
    f = F.P.filter_supported(field, F.P.lanczos_bandpass())
    kept = np.isfinite(f)
    assert kept.sum() > 0 and np.max(np.abs(f[kept])) < 0.05 * 5.0, "a steady input must pass at under 5 percent of its amplitude"


def test_an_oscillating_standing_input_keeps_its_fixed_longitude_and_its_variation():
    F = _load()
    lons, t = _grid(F)
    j = int(np.argmin(np.abs(lons - 50.0)))
    field = np.exp(-((lons[None, :] - 50.0) / 5.0) ** 2) * np.sin(2 * np.pi * t[:, None] / 4.0)   # a four-day antinode at 50 E
    f = F.P.filter_supported(field, F.P.lanczos_bandpass())
    rows = np.flatnonzero(np.isfinite(f[:, j]))
    live = rows[np.max(np.abs(f[rows]), axis=1) > 1e-6]
    assert np.all(np.argmax(np.abs(f[live]), axis=1) == j), "the longitude of maximum amplitude must not move"
    gain = np.max(np.abs(f[rows, j])) / 1.0
    assert 0.8 < gain < 1.1, "a four-day oscillation lies inside the passband and keeps its variation"
    assert np.sum(np.sign(f[rows, j])[1:] * np.sign(f[rows, j])[:-1] < 0) >= 40, "the oscillation is kept, not flattened"


def test_a_one_day_pulse_answers_over_a_longer_span_with_sign_changes():
    F = _load()
    fp = F.footprint_response(F.P.lanczos_bandpass())
    assert fp["pulse_days"] == 1.0 and fp["apparent_span_days"] > 1.0, "the response outlasts the input"
    assert fp["sign_changes"] >= 2, "the response oscillates"
    assert abs(fp["steady_input_response"]) < 0.05, "the weights sum near zero, which is why a steady input is attenuated"
    assert fp["apparent_span_days"] >= 2.5, "the apparent span reaches at least the shortest period in the band"


def test_the_wrapper_keeps_whole_windows_only_and_counts_the_unavailable():
    F = _load()
    rng = np.random.default_rng(2)
    series = rng.normal(0, 1, (488, 3))
    series[200, 1] = np.nan                                                   # one gap in the middle column
    means = {"southern": {"series": series, "support": np.ones((488, 3)), "unavailable": 1, "samples": series.size}}
    filt = F.filtered_means(means, F.P.lanczos_bandpass())["southern"]
    f = filt["series"]
    assert np.isnan(f[:40, 0]).all() and np.isnan(f[-40:, 0]).all() and np.isfinite(f[40:-40, 0]).all(), "40 steps lost at each end of a full run"
    assert np.isnan(f[160:241, 1]).all() and np.isfinite(f[40:160, 1]).all() and np.isfinite(f[241:448, 1]).all(), "a gap removes every window that touches it"
    assert filt["unavailable"] == 80 * 3 + 81 and filt["unavailable_before_filter"] == 1


def test_figures_carry_the_footprint_and_draw_nothing_of_the_tracker(tmp_path):
    pytest.importorskip("matplotlib")
    import datetime as dt
    F = _load()
    times = [dt.datetime(2001, 6, 1) + dt.timedelta(hours=6 * k) for k in range(488)]
    lons = np.arange(20.0, 75.5, 0.5)
    rng = np.random.default_rng(1)
    filt = {}
    for b in ("southern", "northern"):
        series = rng.normal(0, 1, (488, lons.size)); series[:40] = np.nan; series[-40:] = np.nan
        filt[b] = {"series": series, "support": np.ones((488, lons.size)), "unavailable": int(np.isnan(series).sum()), "samples": series.size, "unavailable_before_filter": 0}
    figs = F.draw(2001, 700, lons, times, filt, str(tmp_path / "view"), "test")
    assert [f["name"] for f in figs] == ["season", "June", "July", "August", "September"]
    assert figs[0]["steps"] == 488 and figs[0]["unavailable_in_window"]["southern"] == 80 * lons.size
    assert all(f["footprint_in_title"] and os.path.exists(f["path"]) and len(f["sha256"]) == 64 for f in figs)
    src = open(os.path.join(ROOT, "scripts", "pilot_season_filtered_view.py")).read()
    body = src.split("def draw(")[1].split("def main(")[0]
    assert "ten days before to ten days after" in body and "not a lifetime" in body, "every title states the footprint and its limit"
    assert "track" not in body.replace("tracker drawn", "") and "candidate" not in body and "history" not in body
