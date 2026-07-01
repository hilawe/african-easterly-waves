import numpy as np

from aew.data.huang import load_huang_month
from aew.tracks import Tracks


def _write_sample(path):
    path.write_text(
        "ID\tLifetime(hour)\tgLat(N)\tgLon(E)\twLat(N)\twLon(E)\tSize(km^2)\t"
        "Eccentricity\tBTavg(K)\tBTmin(K)\tTime(UTC)\tSpeed(km/h)\tDirection(deg)\t\n"
        "1 6 10.0000 350.0000 10.0 350.0 2.600000e+04 0.50 225.0 210.0 2000-07-01-00 30.0 -90.0\t\n"
        "1 6 10.5000 348.0000 10.5 348.0 3.000000e+04 0.55 224.0 208.0 2000-07-01-03 30.0 -90.0\t\n"
        "2 3 5.0000 20.0000 5.0 20.0 5.0e+03 0.10 230.0 224.0 2000-07-01-00 NaN NaN\t\n"
    )


def test_load_huang_month_parses_and_wraps(tmp_path):
    p = tmp_path / "MCS_record_2000-07.txt"
    _write_sample(p)
    tr = load_huang_month(str(p))
    assert isinstance(tr, Tracks)
    assert len(tr) == 3
    # lon 350 -> -10 (wrapped), 20 stays
    assert tr.lon.min() == -12.0 or np.isclose(sorted(tr.lon)[0], -12.0)  # 348 -> -12
    assert -10.0 in np.round(tr.lon, 0)
    # radius from size: sqrt(26000/pi) ~ 91 km
    assert np.isclose(tr.variables["radius_km"][0], np.sqrt(26000 / np.pi))
    # track ids namespaced by year-month
    assert all(tid.startswith("2000-07_") for tid in tr.variables["track_id"])


def test_load_huang_month_radius_cut(tmp_path):
    p = tmp_path / "MCS_record_2000-07.txt"
    _write_sample(p)
    tr = load_huang_month(str(p), min_radius_km=90.0)
    # the 5000 km^2 system (radius ~40 km) is dropped; two ~90+ km points remain
    assert len(tr) == 2
    assert tr.variables["radius_km"].min() >= 90.0
