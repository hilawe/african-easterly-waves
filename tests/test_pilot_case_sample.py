"""The first-pass sample follows the frozen rules and nothing else. Per season it takes the
three eastern histories with the most stored observations (ties to the earlier first
detection), one comparison history first detected between 20 and 40 E in the same months
and band with the lower-median number of observations, and the history, if any, with an
observation within 5 degrees and 24 hours of the documented event point."""
import importlib.util
import os
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load(name):
    spec = importlib.util.spec_from_file_location(name + "_under_test", os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _days(y, m, d, hour=0.0):
    return (date(y, m, d) - date(1900, 1, 1)).days + hour / 24.0


def _track(t0, lat0, lons):
    n = len(lons)
    return {"time": [t0 + 0.25 * k for k in range(n)], "lat": [lat0] * n, "lon": list(lons)}


def test_the_sample_rules_pick_by_observations_then_earlier_first_detection_and_the_median_comparison():
    P = _load("pilot_case_sample")
    finished = [_track(_days(1990, 7, 1), 10.0, [45.0, 44.0, 43.0]),                 # 0 eastern, 3 obs
                _track(_days(1990, 7, 2), 10.0, [50.0, 49.0, 48.0, 47.0, 46.0]),      # 1 eastern, 5 obs
                _track(_days(1990, 8, 1), 10.0, [45.0, 44.0, 43.0]),                 # 2 eastern, 3 obs, later than 0
                _track(_days(1990, 7, 3), 10.0, [55.0, 54.0, 53.0, 52.0]),           # 3 eastern, 4 obs
                _track(_days(1990, 6, 1), 10.0, [41.0, 40.0]),                       # 4 eastern, 2 obs
                _track(_days(1990, 7, 1), 10.0, [30.0, 29.0, 28.0, 27.0]),           # 5 comparison, 4 obs
                _track(_days(1990, 7, 5), 10.0, [35.0, 34.0]),                       # 6 comparison, 2 obs
                _track(_days(1990, 7, 6), 10.0, [25.0, 24.0, 23.0, 22.0, 21.0, 20.0]),  # 7 comparison, 6 obs
                _track(_days(1990, 7, 1), 30.0, [30.0, 29.0, 28.0]),                 # 8 outside the band
                _track(_days(1990, 5, 1), 10.0, [30.0, 29.0, 28.0])]                 # 9 outside the months
    eastern = [0, 1, 2, 3, 4]
    sample = P.sample(finished, eastern, start_east_of=40.0, lat_range=(5.0, 20.0), n_eastern=3)
    assert sample["eastern"] == [1, 3, 0]                                              # 5, 4, then the two 3-observation histories by earlier first detection
    assert sample["comparison_population"] == [5, 6, 7] and sample["comparison"] == 5     # observations 4, 2, 6, lower median 4
    sample_two = P.sample(finished[:7], eastern, start_east_of=40.0, lat_range=(5.0, 20.0), n_eastern=3)
    assert sample_two["comparison_population"] == [5, 6] and sample_two["comparison"] == 6   # the lower median of 2 and 4 is 2
    assert P.event_history(finished, lat=10.0, lon=44.5, day=_days(1990, 7, 1, 6)) == {"index": 0, "distance_deg": 0.5, "hours": 0.0}
    assert P.event_history(finished, lat=10.0, lon=44.5, day=_days(1990, 7, 5, 6)) == {"index": None, "nearest": {"index": 0, "distance_deg": 0.5, "hours": 96.0}}   # nothing within a day
    assert P.event_history(finished, lat=20.0, lon=60.0, day=_days(1990, 7, 1)) == {"index": None, "nearest": {"index": 3, "distance_deg": 11.180339887498949, "hours": 48.0}}
