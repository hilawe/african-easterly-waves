"""The threshold figure plots only series that agree across the summaries holding them,
recomputes every printed statistic, refuses any disagreement with the stored aggregates or
the manuscript's values, and never overwrites a figure."""
import importlib.util
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def _load():
    spec = importlib.util.spec_from_file_location("fig_threshold_choices",
                                                  os.path.join(ROOT, "scripts", "fig_threshold_choices.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()
rng = np.random.default_rng(7)


def _summary(v1, port):
    diff = port - v1
    return {"years": {str(y): {"africa_origin": {"v1": int(a), "port": int(b)}} for y, a, b in zip(M.YEARS, v1, port)},
            "aggregates": {"africa_origin": {"mean": {"v1": float(v1.mean()), "port": float(port.mean())},
                                             "difference_port_minus_v1": {"mean": float(diff.mean())},
                                             "pearson_correlation_v1_port": float(np.corrcoef(v1, port)[0, 1])}}}


def _world(tmp_path):
    ei = rng.integers(100, 160, len(M.YEARS)).astype(float)
    ei_alt = ei + rng.integers(50, 90, len(M.YEARS))
    e5 = rng.integers(100, 160, len(M.YEARS)).astype(float)
    e5_alt = e5 - rng.integers(0, 12, len(M.YEARS))
    paths = {}
    for name, doc in (("eraint", _summary(ei, ei_alt)), ("era5", _summary(e5, e5_alt)), ("cross", _summary(ei, e5_alt))):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(doc))
        paths[name] = p

    def printed(own, alt):
        s = M.statistics(own, alt)
        return (round(s["own_mean"], 1), round(s["alt_mean"], 1), round(s["mean_difference"], 1), round(s["r"], 3))
    expected = {"eraint": printed(ei, ei_alt), "era5": printed(e5, e5_alt)}
    return paths, expected


def test_consistent_summaries_pass_and_the_statistics_are_recomputed(tmp_path):
    paths, expected = _world(tmp_path)
    data, stats, digests = M.verified(str(paths["eraint"]), str(paths["era5"]), str(paths["cross"]), expected)
    assert set(stats) == {"eraint", "era5"} and len(digests) == 3
    own, alt = data["eraint"]
    assert stats["eraint"]["mean_difference"] == pytest.approx(float((alt - own).mean()))


def test_a_series_that_differs_between_summaries_is_refused(tmp_path):
    paths, expected = _world(tmp_path)
    doc = json.loads(paths["cross"].read_text())
    doc["years"]["1990"]["africa_origin"]["v1"] += 1
    paths["cross"].write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="differs between the two summaries"):
        M.verified(str(paths["eraint"]), str(paths["era5"]), str(paths["cross"]), expected)


def test_a_stored_aggregate_that_disagrees_is_refused(tmp_path):
    paths, expected = _world(tmp_path)
    doc = json.loads(paths["era5"].read_text())
    doc["aggregates"]["africa_origin"]["pearson_correlation_v1_port"] += 0.01
    paths["era5"].write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="the summary stores"):
        M.verified(str(paths["eraint"]), str(paths["era5"]), str(paths["cross"]), expected)


def test_values_that_disagree_with_the_manuscript_are_refused(tmp_path):
    paths, expected = _world(tmp_path)
    wrong = dict(expected)
    wrong["eraint"] = (expected["eraint"][0], expected["eraint"][1], expected["eraint"][2] + 0.1, expected["eraint"][3])
    with pytest.raises(SystemExit, match="manuscript prints"):
        M.verified(str(paths["eraint"]), str(paths["era5"]), str(paths["cross"]), wrong)


def test_the_retained_manuscript_values_are_the_section_4_numbers():
    assert M.MANUSCRIPT_VALUES == {"eraint": (126.5, 195.9, 69.4, 0.616), "era5": (126.0, 119.6, -6.5, 0.842)}


def test_the_figure_is_written_once_in_vector_and_preview_form(tmp_path):
    paths, expected = _world(tmp_path)
    data, stats, _ = M.verified(str(paths["eraint"]), str(paths["era5"]), str(paths["cross"]), expected)
    written = M.render(data, stats, str(tmp_path / "thr"))
    assert [os.path.basename(p) for p in written] == ["thr.pdf", "thr.png"]
    with pytest.raises(FileExistsError):
        M.render(data, stats, str(tmp_path / "thr"))
