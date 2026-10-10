"""The wide box is the control run's domain widened by the margin on every side, so the
same builder serves the 40 E and the 60 E controls, and the 40 E constants are that box."""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load(name):
    spec = importlib.util.spec_from_file_location(name + "_under_test", os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_wide_box_is_the_controls_domain_plus_the_margin_on_every_side():
    W = _load("pilot_wide_case")
    assert W.MARGIN_DEGREES == 4.0
    assert W.wide_box({"lat": [-35.0, 35.0], "lon": [-140.0, 40.0]}) == ((-39.0, 39.0), (-144.0, 44.0))
    assert W.wide_box({"lat": [-35.0, 35.0], "lon": [-140.0, 60.0]}) == ((-39.0, 39.0), (-144.0, 64.0))
    assert (W.WIDE_LAT, W.WIDE_LON) == W.wide_box({"lat": [-35.0, 35.0], "lon": [-140.0, 40.0]})      # the 40 E constants are that box
    assert W.wide_box({"lat": [-35.0, 35.0], "lon": [-140.0, 40.0]}, margin=2.0) == ((-37.0, 37.0), (-142.0, 42.0))
