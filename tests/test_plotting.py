import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np

from aew.plotting import hovmoller, save


def _demo_arrays():
    lon = np.arange(-40.0, 80.0, 4.0)
    lag = np.arange(-6.0, 7.0)
    shaded = np.outer(np.cos(lag / 3.0), np.sin(lon / 20.0))
    contour = np.outer(np.sin(lag / 3.0), np.cos(lon / 20.0))
    return shaded, lon, lag, contour


def test_hovmoller_returns_fig_and_axes():
    shaded, lon, lag, contour = _demo_arrays()
    fig, ax = hovmoller(shaded, lon, lag, contour=contour, base_lon=0.0,
                        lon_range=(-40, 80), title="t", provenance="n=10")
    assert fig is not None and ax is not None
    # y-axis inverted (lag -6 at top): ylim[0] > ylim[1]
    ylo, yhi = ax.get_ylim()
    assert ylo > yhi
    assert ax.get_xlim() == (-40, 80)


def test_hovmoller_shaded_only():
    shaded, lon, lag, _ = _demo_arrays()
    fig, ax = hovmoller(shaded, lon, lag)
    assert ax.collections  # contourf drew something


def test_save_writes_png(tmp_path):
    shaded, lon, lag, _ = _demo_arrays()
    fig, ax = hovmoller(shaded, lon, lag)
    out = save(fig, str(tmp_path / "h.png"))
    import os
    assert os.path.exists(out) and os.path.getsize(out) > 0
