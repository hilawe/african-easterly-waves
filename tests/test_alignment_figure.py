"""The supplement figure of the alignment check is drawn from the saved profiles of both grids.

The figure script computes no composite. It reads the profile and the summary row of each
grid, and it must refuse to draw a curve that does not reproduce its own summary.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import check_track_alignment_region as A  # noqa: E402
import fig_track_alignment as G  # noqa: E402


def profile(step, shift, half=12.0, n=50):
    off = np.arange(-half, half + step / 2, step)
    val = 1e-5 * np.exp(-0.5 * ((off - shift) / 3.0) ** 2) - 2e-6
    return pd.DataFrame(dict(offset_deg=off, mean_curvature_vorticity=val, n=float(n)))


def deposit(tmp_path, change=None):
    rows = []
    for grid, name, step, shift, n_arg in (("global band", "alignment_band_profile.csv", 1.5, 0.6, 47),
                                           ("regional", "alignment_region_profile.csv", 0.5, 0.2, 44)):
        prof = profile(step, shift)
        peak, centroid = G.peak_and_centroid(prof)
        prof.to_csv(tmp_path / name, index=False, float_format="%.9g")
        rows.append(dict(grid=grid, seasons=25, grid_step_deg=step, n_matched=50, n_window_outside_grid=0, peak_deg=peak,
                         centroid_deg=centroid, median_argmax_deg=0.0, n_argmax=n_arg))
    d = pd.DataFrame(rows)
    if change:
        change(d)
    d.to_csv(tmp_path / "alignment_region.csv", index=False, float_format="%.6f")
    return str(tmp_path)


def test_the_centroid_of_a_profile_is_the_one_the_composite_states():
    # the figure's own arithmetic against the composite's, on a lobe that is not symmetric
    c = A.Composite(0.5, wrap=False)
    lon = np.arange(-45.0, 75.5, 0.5)
    times = pd.date_range("2000-07-01", periods=3, freq="6h")
    lat = np.arange(0.0, 21.0, 1.0)
    crv = np.zeros((3, lat.size, lon.size))
    for i, lo in enumerate((-10.0, 0.0, 10.0)):
        crv[i] = (np.exp(-0.5 * ((lon - lo - 1.0) / 1.5) ** 2) + 0.6 * np.exp(-0.5 * ((lon - lo - 4.0) / 1.0) ** 2))[None, :]
    c.add(times, lat, lon, crv, times.values, np.full(3, 10.0), np.array([-10.0, 0.0, 10.0]))
    s, prof = c.summary()
    peak, centroid = G.peak_and_centroid(prof)
    assert peak == s["peak_deg"] and centroid == pytest.approx(s["centroid_deg"], abs=1e-12)
    assert abs(centroid - peak) > 0.1


def test_the_figure_is_drawn_from_both_saved_profiles_with_each_grids_own_summary(tmp_path):
    data = G.load(deposit(tmp_path))
    assert [d["grid"] for d in data] == ["global band", "regional"]
    assert [len(d["prof"]) for d in data] == [17, 49]
    lines = G.draw(data, str(tmp_path / "f.png"))
    assert os.path.getsize(tmp_path / "f.png") > 10_000
    assert lines[0].startswith("50 trough observations on each grid")
    assert lines[1].startswith("1.5 degree: peak +0.0, centroid +") and lines[1].endswith("median of maxima +0.0 (n = 47)")
    assert lines[2].startswith("0.5 degree: peak +0.0, centroid +") and lines[2].endswith("median of maxima +0.0 (n = 44)")
    assert max(len(line) for line in lines) <= 80                                 # a longer line leaves the frame at this size
    assert not any(ch in "".join(lines) for ch in "\u2212\u2013\u2014")          # no dash or math minus in the figure text


@pytest.mark.parametrize("change, said", [
    (lambda d: d.__setitem__("centroid_deg", d.centroid_deg + 0.001), "its summary states"),
    (lambda d: d.__setitem__("peak_deg", d.peak_deg + 0.5), "its summary states"),
    (lambda d: d.__setitem__("grid_step_deg", [0.5, 0.5]), "is not a window about zero on a grid of"),
    (lambda d: d.__setitem__("grid", ["global band", "another"]), "holds the grids"),
])
def test_a_profile_that_does_not_reproduce_its_summary_is_not_drawn(tmp_path, change, said):
    dep = deposit(tmp_path, change)
    with pytest.raises(SystemExit, match="REFUSED: .*" + said):
        G.load(dep)


def test_a_window_that_is_not_centered_on_zero_is_refused(tmp_path):
    dep = deposit(tmp_path)
    p = pd.read_csv(tmp_path / "alignment_region_profile.csv").iloc[1:]
    p.to_csv(tmp_path / "alignment_region_profile.csv", index=False, float_format="%.9g")
    with pytest.raises(SystemExit, match="REFUSED: alignment_region_profile.csv is not a window about zero"):
        G.load(dep)
