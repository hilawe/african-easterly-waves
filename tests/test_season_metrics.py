"""The season metrics' predeclared rules, on fixtures.

Season membership is by the first observation's month; a year with no track in a group
counts as zero in the published spread and is never dropped; the sparse rule and the
zero-variance rule report counts only; the distribution comparison reports the KS
statistic and percentile differences, not a median; band edges are [lo, hi); the harness
control fails on one moved observation or a different case; and the artifact says it is
descriptive with no tolerance applied.
"""
import importlib.util
import io
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
pytest.importorskip("scipy")

DAY = 33025.0   # 1990-06-03


def _load():
    path = os.environ.get("SEASON_SCRIPT", os.path.join(ROOT, "scripts", "season_metrics.py"))
    spec = importlib.util.spec_from_file_location("season_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _track(t0, n, lon0, lat0=10.0, dlon=-0.5):
    t = t0 + 0.25 * np.arange(n)
    return {"time": t, "lat": np.full(n, lat0), "lon": lon0 + dlon * np.arange(n)}


def _mat_bytes(tracks, case="case-x"):
    from scipy.io import savemat
    payload = {"n": float(len(tracks)), "case_id": case}
    for i, tr in enumerate(tracks):
        payload[f"time{i}"], payload[f"lat{i}"], payload[f"lon{i}"] = tr["time"], tr["lat"], tr["lon"]
    buf = io.BytesIO()
    savemat(buf, payload)
    return buf.getvalue()


def test_season_membership_is_by_the_first_observation():
    S = _load()
    may31 = S.date_of(33022.0)
    assert (may31.year, may31.month, may31.day) == (1990, 5, 31)
    starts_may = _track(33022.0, 12, 10.0)         # runs into June, starts in May
    starts_june = _track(33023.0, 4, 10.0)
    assert not S.in_season(starts_may, 1990)
    assert S.in_season(starts_june, 1990)
    assert not S.in_season(starts_june, 1991)


def _regions(tmp_path):
    """Two square polygons in the archive's file layout: a higher-priority region NAL over
    the box 30 W to 20 W, 0 to 20 N, and AFR over 20 W to 40 E, 0 to 20 N, sharing an edge."""
    from scipy.io import savemat
    tmp_path.mkdir(parents=True, exist_ok=True)
    boxes = {"north_atlantic": (-30, -20), "africa": (-20, 40)}
    for code, name in [x for x in [("NEP", "northeast_pacific"), ("SEP", "southeast_pacific"),
                                   ("CAM", "central_america"), ("SAM", "south_america"),
                                   ("NAL", "north_atlantic"), ("SAL", "south_atlantic"), ("AFR", "africa")]]:
        lo, hi = boxes.get(name, (200, 201))       # the others sit off the map
        arr = np.array([[lo, hi, hi, lo, lo], [0, 0, 20, 20, 0]], float)
        savemat(tmp_path / f"{name}.mat", {name: arr})
    return tmp_path


def test_record_membership_is_the_archives_polygon_rule_at_the_first_observation(tmp_path):
    S = _load()
    regions, digests = S.load_regions(str(_regions(tmp_path)))
    assert [c for c, _ in regions] == ["NEP", "SEP", "CAM", "SAM", "NAL", "SAL", "AFR"] and len(digests) == 7
    starts_outside_ends_inside = _track(DAY, 8, -25.0, dlon=+2.0)      # 25 W to 11 W: NAL by its start
    starts_inside_ends_outside = _track(DAY, 8, 38.0, dlon=+1.0)       # 38 E to 45 E: AFR by its start
    assert S.source_region(starts_outside_ends_inside, regions) == "NAL"
    assert not S.in_record_domain(starts_outside_ends_inside, regions)
    assert S.in_record_domain(starts_inside_ends_outside, regions)
    # the shared edge at 20 W goes to the higher-priority region, and rounding to the
    # quarter degree decides membership: 19.9 W rounds to 20 W, on the edge, so NAL
    assert S.source_region(_track(DAY, 8, -19.9), regions) == "NAL"
    assert S.source_region(_track(DAY, 8, -19.8), regions) == "AFR"
    assert S.source_region(_track(DAY, 8, 10.0, lat0=20.0), regions) == "AFR"       # on the north edge
    assert S.source_region(_track(DAY, 8, 10.0, lat0=20.2), regions) == "OTH"       # rounds to 20.25, outside
    # TIES ROUND AWAY FROM ZERO, as MATLAB does, on both signs. 20.125 / 0.25 = 80.5 goes to
    # 81 (20.25) under MATLAB and to 80 (20.0) under Python's half-to-even, so a first
    # observation on that tie sits outside the north edge under the archive's rule and on it
    # under Python's, and -20.125 / 0.25 = -80.5 goes to -81 (20.25 W) under MATLAB.
    assert S.matlab_round(80.5) == 81.0 and S.matlab_round(-80.5) == -81.0
    # just below a tie stays down on both signs, where floor(a + 0.5) would round up
    below = 0.49999999999999994
    assert S.matlab_round(below) == 0.0 and S.matlab_round(-below) == 0.0
    # 80 + below is 80.5 exactly in double precision, so the below-tie case at magnitude
    # 80 is the largest double under 80.5, four units in the last place down
    assert S.matlab_round(80.5 - 2 ** -44) == 80.0 and S.matlab_round(-(80.5 - 2 ** -44)) == -80.0
    assert S.matlab_round(79.5) == 80.0 and S.matlab_round(-79.5) == -80.0
    assert S.source_region(_track(DAY, 8, 10.0, lat0=20.125), regions) == "OTH"     # MATLAB: 20.25, outside
    assert S.source_region(_track(DAY, 8, -20.125), regions) == "NAL"               # MATLAB: 20.25 W
    ends_in_june_starts_in_may = _track(33022.0, 12, 10.0)              # May 31 to June 3
    starts_in_september_ends_in_october = _track(33144.0, 12, 10.0)     # September 30 to October 3
    assert not S.in_season(ends_in_june_starts_in_may, 1990)
    assert S.in_season(starts_in_september_ends_in_october, 1990)


def test_spacing_reports_intervals_gaps_and_duration_not_a_count():
    S = _load()
    contiguous = _track(DAY, 5, 10.0)                                  # four intervals of 0.25
    gapped = {"time": DAY + np.array([0.0, 0.25, 0.75, 1.0]), "lat": np.full(4, 10.0), "lon": np.full(4, 10.0)}
    out = S.spacing([contiguous, gapped])
    assert out["intervals"] == 7 and out["fraction_of_intervals_one_timestep"] == pytest.approx(6 / 7)
    assert out["largest_interval_days"] == 0.5 and out["tracks_with_a_gap"] == 1
    assert out["duration_days_percentiles"]["50"] == pytest.approx(1.0)     # both span one day
    assert S.features(contiguous)["lifetime"] == 5                          # five observations, one day
    # a population of single-observation tracks has no intervals and must not raise
    single = {"time": np.array([DAY]), "lat": np.array([10.0]), "lon": np.array([10.0])}
    only = S.spacing([single, single])
    assert only["intervals"] == 0 and only["fraction_of_intervals_one_timestep"] is None and only["tracks_with_a_gap"] == 0


def test_band_edges_are_half_open_and_outside_is_named():
    S = _load()
    assert S.band_of(10.0) == "+10..+20"
    assert S.band_of(9.999) == "+0..+10"
    assert S.band_of(-60.0) == "-60..-50"
    assert S.band_of(60.0) == "outside"


def test_published_spread_counts_an_empty_year_as_zero():
    S = _load()
    by_year = {1990: [_track(DAY, 6, 25.0), _track(DAY + 1, 6, 25.0)],
               1991: [_track(DAY + 365, 6, 25.0)],
               1992: []}
    spread = S.published_spread(by_year)
    season = spread["season"]
    assert season["counts_by_year"] == {"1990": 2, "1991": 1, "1992": 0}
    assert season["mean"] == pytest.approx(1.0)
    assert season["sd"] == pytest.approx(np.std([2, 1, 0], ddof=1))
    assert spread["band +20..+30"]["counts_by_year"] == {"1990": 2, "1991": 1, "1992": 0}
    # the record's own distributions travel with the spread, pooled over the years
    assert spread["season"]["pooled_tracks"] == 3
    assert spread["season"]["percentiles"]["lifetime"] == {"10": 6.0, "50": 6.0, "90": 6.0}
    assert spread["band +20..+30"]["percentiles"]["genesis_lon"]["50"] == pytest.approx(25.0)
    assert spread["month 7"]["percentiles"]["lifetime"] is None


def test_sparse_and_zero_variance_groups_are_counts_only():
    S = _load()
    v1 = [_track(DAY + i, 8, 25.0) for i in range(6)]          # six in band +20..+30, June
    port = [_track(DAY + i, 8, 25.0) for i in range(6)]
    spread = {"season": {"mean": 5.0, "sd": 2.0}, "month 6": {"mean": 5.0, "sd": 0.0},
              "band +20..+30": {"mean": 5.0, "sd": 1.5}}
    comparison, counts_only = S.compare(v1, port, spread)
    assert "distributions" in comparison["season"] and "distributions" in comparison["band +20..+30"]
    assert comparison["month 6"]["reported_as"] == "counts only"
    assert "zero interannual variance" in comparison["month 6"]["why"]
    assert "distributions" not in comparison["month 6"]
    # four on the port side in July: sparse, whatever the published spread says
    v1 += [_track(DAY + 40 + i, 8, 5.0) for i in range(6)]
    port += [_track(DAY + 40 + i, 8, 5.0) for i in range(4)]
    spread["month 7"] = {"mean": 5.0, "sd": 2.0}
    comparison, counts_only = S.compare(v1, port, spread)
    assert comparison["month 7"]["reported_as"] == "counts only" and "fewer than" in comparison["month 7"]["why"]
    assert comparison["month 7"]["v1"] == 6 and comparison["month 7"]["port"] == 4
    assert set(counts_only) >= {"month 6", "month 7"}


def test_distribution_comparison_reports_ks_and_percentiles_not_a_median():
    S = _load()
    a = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    b = [10, 10, 10, 10, 10, 10, 30, 30, 30, 30]      # same median, different distribution
    out = S.distribution_comparison(a, b)
    assert out["percentiles"]["50"]["port_minus_v1"] == pytest.approx(0.0)
    assert out["percentiles"]["90"]["port_minus_v1"] == pytest.approx(20.0)
    assert out["ks_statistic"] == pytest.approx(0.4)


def test_difference_is_scaled_by_the_published_sd():
    S = _load()
    v1 = [_track(DAY + i, 8, 25.0) for i in range(10)]
    port = [_track(DAY + i, 8, 25.0) for i in range(6)]
    port.append({k: v.copy() for k, v in port[0].items()})      # one exact duplicate: 8 of 56 observations
    spread = {"season": {"mean": 8.0, "sd": 2.0, "distinct_waves_sd": 4.0, "duplication_fraction_sd": 0.1},
              "band +20..+30": {"mean": 8.0, "sd": 8.0, "distinct_waves_sd": 1.0, "duplication_fraction_sd": 0.05},
              "month 6": {"mean": 8.0, "sd": 0.5, "distinct_waves_sd": 2.0, "duplication_fraction_sd": 0.02}}
    comparison, _ = S.compare(v1, port, spread)
    assert comparison["season"]["port_minus_v1"] == -3
    assert comparison["season"]["difference_over_published_sd"] == pytest.approx(-1.5)
    # EACH GROUP IS SCALED BY ITS OWN SPREAD, and each quantity by its own: the same count
    # difference reads -0.375 in the band and -6 in the month; the distinct-wave difference
    # (ten distinct against six, the duplicate joining its original) by the distinct-wave
    # spread; and the duplication difference (8 of 56 against none) by the duplication
    # spread, none of them by the count's
    assert comparison["band +20..+30"]["difference_over_published_sd"] == pytest.approx(-0.375)
    assert comparison["month 6"]["difference_over_published_sd"] == pytest.approx(-6.0)
    assert comparison["season"]["distinct_waves"]["port_minus_v1"] == -4
    assert comparison["season"]["distinct_waves"]["difference_over_published_sd"] == pytest.approx(-1.0)
    assert comparison["band +20..+30"]["distinct_waves"]["difference_over_published_sd"] == pytest.approx(-4.0)
    assert comparison["season"]["duplication"]["port_minus_v1_points"] == pytest.approx(100 * 8 / 56)
    assert comparison["season"]["duplication"]["difference_over_published_sd"] == pytest.approx((8 / 56) / 0.1)
    assert comparison["band +20..+30"]["duplication"]["difference_over_published_sd"] == pytest.approx((8 / 56) / 0.05)


def test_published_spread_carries_distinct_waves_and_duplication_per_group():
    S = _load()
    a = _track(DAY, 8, 25.0)
    copy = {"time": a["time"].copy(), "lat": a["lat"].copy(), "lon": a["lon"].copy()}   # an exact duplicate
    by_year = {1990: [a, copy, _track(DAY + 3, 8, 25.0)], 1991: [_track(DAY + 365, 8, 25.0)], 1992: []}
    spread = S.published_spread(by_year)
    g = spread["band +20..+30"]
    assert g["counts_by_year"] == {"1990": 3, "1991": 1, "1992": 0}
    assert g["distinct_waves_mean"] == pytest.approx(1.0)            # 2, 1, 0
    assert g["distinct_waves_sd"] == pytest.approx(np.std([2, 1, 0], ddof=1))
    assert g["duplication_fraction_mean"] == pytest.approx((8 / 24 + 0.0) / 2)   # 1992 undefined, not zero
    assert g["duplication_fraction_sd"] == pytest.approx(np.std([8 / 24, 0.0], ddof=1))


def test_distinct_waves_join_near_copies_only():
    S = _load()
    a = _track(DAY, 8, 25.0)
    near = {"time": a["time"].copy(), "lat": a["lat"] + 0.2, "lon": a["lon"] + 0.2}
    far = {"time": a["time"].copy(), "lat": a["lat"] + 5.0, "lon": a["lon"]}
    brief = {"time": a["time"][:2].copy(), "lat": a["lat"][:2].copy(), "lon": a["lon"][:2].copy()}
    assert S.distinct_waves([a, near, far, brief]) == 3


def test_harness_control_fails_on_one_moved_observation_or_another_case():
    S = _load()
    tracks = [_track(DAY, 6, 25.0), _track(DAY + 2, 5, 5.0)]
    same = S.harness_control(_mat_bytes(tracks), _mat_bytes(tracks))
    assert same["passed"] and same["tracks_new"] == 2
    moved = [dict(tracks[0]), dict(tracks[1])]
    moved[1] = {"time": tracks[1]["time"], "lat": tracks[1]["lat"] + 1e-6, "lon": tracks[1]["lon"]}
    assert not S.harness_control(_mat_bytes(moved), _mat_bytes(tracks))["passed"]
    assert not S.harness_control(_mat_bytes(tracks, case="other"), _mat_bytes(tracks))["passed"]


def test_track_files_must_carry_a_nonempty_case_id_and_a_whole_count():
    S = _load()
    tracks = [_track(DAY, 6, 25.0)]
    with pytest.raises(SystemExit):
        S.read_mat_tracks(_mat_bytes(tracks, case=" "))
    from scipy.io import savemat
    buf = io.BytesIO()
    savemat(buf, {"n": -1.0, "case_id": "c", "time0": tracks[0]["time"], "lat0": tracks[0]["lat"], "lon0": tracks[0]["lon"]})
    with pytest.raises(SystemExit):
        S.read_mat_tracks(buf.getvalue())
    # a count that is an array is not one number, whatever its first element says
    buf = io.BytesIO()
    savemat(buf, {"n": np.array([0.0, 1.0]), "case_id": "c", "time0": tracks[0]["time"], "lat0": tracks[0]["lat"], "lon0": tracks[0]["lon"]})
    with pytest.raises(SystemExit):
        S.read_mat_tracks(buf.getvalue())


def test_command_path_publishes_a_descriptive_artifact(tmp_path, monkeypatch):
    S = _load()
    tracks = [_track(DAY + i, 8, 25.0 - i) for i in range(6)]
    outside = [_track(DAY + 10, 8, -100.0), _track(DAY + 11, 8, 45.0)]     # a Pacific start and one east of 40 E
    (tmp_path / "v1.mat").write_bytes(_mat_bytes(tracks + outside))
    (tmp_path / "port.mat").write_bytes(_mat_bytes(tracks[:5] + outside[:1]))
    record = {1990: [_track(DAY + i, 8, 25.0) for i in range(6)], 1991: [_track(DAY + 365, 8, 25.0)]}
    # the record is parsed FROM THE HASHED BYTES, so the fake keys on the bytes it was handed
    monkeypatch.setattr(S, "read_record_bytes", lambda blob: record[int(blob.decode())])
    rec = tmp_path / "rec"
    rec.mkdir()
    for y in record:
        (rec / f"ERA-Int_ew_700hPa_{y}_AFR.nc").write_bytes(str(y).encode())
    out = tmp_path / "out.json"
    regions_dir = _regions(tmp_path / "regions_dir") if not (tmp_path / "regions_dir").exists() else tmp_path / "regions_dir"
    S.main(["--v1", str(tmp_path / "v1.mat"), "--port", str(tmp_path / "port.mat"), "--year", "1990",
            "--record-dir", str(rec), "--record-years", "1990-1991", "--regions-dir", str(regions_dir), "--out", str(out)])
    import json
    art = json.loads(out.read_text())
    assert art["descriptive"] is True and art["tolerances_applied"] is None
    # THE COMPARISON IS OVER THE RECORD'S DOMAIN, and the whole-domain totals sit beside it
    assert art["comparison"]["season"]["v1"] == 6 and art["comparison"]["season"]["port"] == 5
    assert art["columns"]["v1"]["tracks_in_season_whole_domain"] == 8
    assert art["columns"]["port"]["tracks_in_season_whole_domain"] == 6
    assert art["columns"]["v1"]["tracks_in_season_record_domain"] == 6
    assert "domain" in art["definitions"]
    assert art["published_spread"]["season"]["counts_by_year"] == {"1990": 6, "1991": 1}
    assert set(art["inputs_sha256"]) == {"v1", "port"}
    # ARTIFACT COMPLETENESS: every group carries the count, distinct waves and duplication,
    # and every non-sparse group its distributions
    for k, g in art["comparison"].items():
        assert {"v1", "port", "distinct_waves", "duplication"} <= set(g), k
        assert ("distributions" in g) != (g.get("reported_as") == "counts only"), k
    assert all(name in art["comparison"]["season"]["distributions"] for name in S.DISTRIBUTIONS)
    assert art["columns"]["v1"]["spacing"]["tracks"] == 6 and "duration_days_percentiles" in art["columns"]["v1"]["spacing"]
    assert art["columns"]["v1"]["source_regions_in_season"] == {"AFR": 6, "OTH": 2}
    assert set(art["region_polygons_sha256"]) == {n for _, n in S.REGION_FILES}
    with pytest.raises(SystemExit):
        S.main(["--v1", str(tmp_path / "v1.mat"), "--port", str(tmp_path / "port.mat"), "--year", "1990",
                "--record-dir", str(rec), "--record-years", "1990-1991", "--regions-dir", str(regions_dir), "--out", str(out)])


def test_boundary_predicates_on_the_first_and_last_observation():
    S = _load()
    y = 1990
    june1, oct1, dec31 = S._day(y, 6, 1), S._day(y, 10, 1), S._day(y, 12, 31)
    spans_june_with_a_gap = {"time": np.array([june1 - 0.25, june1 + 0.25]), "lat": np.zeros(2), "lon": np.zeros(2)}
    ends_before_october = _track(oct1 - 2.0, 8, 10.0)                       # last observation September 30 18Z
    crosses_october = _track(oct1 - 0.5, 8, 10.0)                            # September 30 12Z to October 2
    ends_dec31 = _track(dec31 - 1.0, 5, 10.0)                                # last observation December 31 00Z
    starts_in_may_ends_in_may = _track(june1 - 3.0, 4, 10.0)
    out = S.boundary_lines([spans_june_with_a_gap, ends_before_october, crosses_october, ends_dec31, starts_in_may_ends_in_may], y)
    assert out["crossing_in"]["tracks"] == 1 and out["crossing_in"]["observations_at_or_after_june_1"] == 1
    assert out["crossing_out"]["tracks"] == 1
    assert out["year_end_potentially_censored"]["tracks"] == 1
    # EVERY December 31 time counts, and the following January 1 does not
    for last in (dec31, dec31 + 0.25, dec31 + 0.5, dec31 + 0.75):
        t = {"time": np.array([last - 0.5, last - 0.25, last]), "lat": np.zeros(3), "lon": np.zeros(3)}
        assert S.boundary_lines([t], y)["year_end_potentially_censored"]["tracks"] == 1, last
    jan1 = {"time": np.array([dec31 + 0.75, dec31 + 1.0]), "lat": np.zeros(2), "lon": np.zeros(2)}
    assert S.boundary_lines([jan1], y)["year_end_potentially_censored"]["tracks"] == 0
    # one timestep earlier and the October crossing is not one
    just_short = _track(oct1 - 2.0, 8, 10.0)
    assert S.boundary_lines([just_short], y)["crossing_out"]["tracks"] == 0
    # THE EDGES: a track whose first observation is June 1 00Z is a season member, not a
    # crossing in, and a track whose last observation is October 1 00Z exactly crosses out
    starts_on_june_1 = _track(june1, 8, 10.0)
    assert S.boundary_lines([starts_on_june_1], y)["crossing_in"]["tracks"] == 0
    ends_on_oct_1 = _track(oct1 - 1.75, 8, 10.0)
    assert ends_on_oct_1["time"][-1] == oct1
    assert S.boundary_lines([ends_on_oct_1], y)["crossing_out"]["tracks"] == 1


def _mat_with_producer(tracks, case, settings, label, dataset, stage="tracking", year=1990, sources=None, prefix=None):
    from scipy.io import savemat
    S = _load()
    inventory = {name: "a" * 64 for name in S.expected_tracking_sources()}
    payload = {"n": float(len(tracks)), "case_id": case,
               "producer_json": json.dumps({"protocol_settings": settings, "stage": stage,
                                            "dataset_specific": {"label": label, "dataset": dataset, "year": year,
                                                                 "prefix": prefix or {"eraint": "eraint", "era5": "era5"}.get(dataset, dataset)},
                                            "source_sha256": sources if sources is not None else inventory})}
    for i, tr in enumerate(tracks):
        payload[f"time{i}"], payload[f"lat{i}"], payload[f"lon{i}"] = tr["time"], tr["lat"], tr["lon"]
    buf = io.BytesIO()
    savemat(buf, payload)
    return buf.getvalue()


def test_reanalysis_mode_keeps_both_identities_and_requires_equal_settings(tmp_path, monkeypatch):
    import json as _json
    S = _load()
    import hashlib
    tracks = [_track(DAY + i, 8, 25.0 - i) for i in range(6)]
    regions_dir = _regions(tmp_path / "regions_dir")
    real = json.load(open(os.path.join(ROOT, "docs", "aewc_v2", "protocol", "manifest_2026-09-25.json")))
    manifest = json.loads(json.dumps(real))
    manifest["membership"]["regions_dir"] = str(regions_dir)
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))
    settings = {**{k: manifest[k] for k in S.PROTOCOL_KEYS}, "manifest_sha256": hashlib.sha256(mpath.read_bytes()).hexdigest()}
    (tmp_path / "a.mat").write_bytes(_mat_with_producer(tracks, "case-a", settings, "ERA-Interim", "eraint"))
    (tmp_path / "b.mat").write_bytes(_mat_with_producer(tracks[:5], "case-b", settings, "ERA5", "era5"))
    other = dict(settings, years=[1981, 2010])
    (tmp_path / "c.mat").write_bytes(_mat_with_producer(tracks[:5], "case-c", other, "ERA5", "era5"))
    full = {name: "a" * 64 for name in S.expected_tracking_sources()}
    (tmp_path / "d.mat").write_bytes(_mat_with_producer(tracks[:5], "case-d", settings, "ERA5", "era5", sources=dict(full, **{"src/aew/v1port/pipeline.py": "b" * 64})))
    (tmp_path / "h.mat").write_bytes(_mat_with_producer(tracks[:5], "case-h", settings, "ERA5", "era5", sources={"src/aew/v1port/pipeline.py": "a" * 64}))
    (tmp_path / "i.mat").write_bytes(_mat_with_producer(tracks[:5], "case-i", settings, "ERA5", "eraint"))        # label not the manifest's for eraint
    (tmp_path / "j.mat").write_bytes(_mat_with_producer(tracks[:5], "case-j", settings, "Other", "other"))        # not a manifest dataset
    (tmp_path / "e.mat").write_bytes(_mat_with_producer(tracks[:5], "case-e", settings, "ERA5", "era5", stage="inputs"))
    (tmp_path / "f.mat").write_bytes(_mat_with_producer(tracks[:5], "case-f", {}, "ERA5", "era5"))
    outside = tracks[:5] + [_track(DAY + 400, 8, 25.0)]           # an observation in the next year
    (tmp_path / "g.mat").write_bytes(_mat_with_producer(outside, "case-g", settings, "ERA5", "era5"))
    record = {1990: [_track(DAY + i, 8, 25.0) for i in range(6)], 1991: [_track(DAY + 365, 8, 25.0)]}
    monkeypatch.setattr(S, "read_record_bytes", lambda blob: record[int(blob.decode())])
    rec = tmp_path / "rec"
    rec.mkdir()
    for y in record:
        (rec / f"ERA-Int_ew_700hPa_{y}_AFR.nc").write_bytes(str(y).encode())
    common = ["--year", "1990", "--record-dir", str(rec), "--record-years", "1990-1991", "--regions-dir", str(regions_dir),
              "--manifest", str(mpath)]
    out = tmp_path / "out.json"
    S.main(["--mode", "reanalysis", "--v1", str(tmp_path / "a.mat"), "--port", str(tmp_path / "b.mat"), *common, "--out", str(out)])
    art = _json.loads(out.read_text())
    assert art["mode"] == "reanalysis" and art["case_id"] is None
    assert art["sides"]["v1"]["label"] == "ERA-Interim" and art["sides"]["v1"]["case_id"] == "case-a"
    assert art["sides"]["v1"]["manifest_sha256"] == settings["manifest_sha256"] and art["inputs_sha256"]["manifest"]["sha256"] == settings["manifest_sha256"]
    assert art["sides"]["port"]["label"] == "ERA5" and art["sides"]["protocol_settings_equal"] is True
    assert art["comparison"]["season"]["v1"] == 6 and art["comparison"]["season"]["port"] == 5
    assert "boundaries" in art["columns"]["v1"] and art["columns"]["v1"]["boundaries"]["crossing_in"]["tracks"] == 0
    with pytest.raises(SystemExit):        # differing protocol settings
        S.main(["--mode", "reanalysis", "--v1", str(tmp_path / "a.mat"), "--port", str(tmp_path / "c.mat"), *common, "--out", str(tmp_path / "o2.json")])
    for name in ("d", "e", "f", "g", "h", "i", "j"):      # differing sources, inputs stage, empty settings, outside the year, incomplete inventory, wrong label, unknown dataset
        with pytest.raises(SystemExit):
            S.main(["--mode", "reanalysis", "--v1", str(tmp_path / "a.mat"), "--port", str(tmp_path / f"{name}.mat"), *common, "--out", str(tmp_path / f"o_{name}.json")])
    # THE PRODUCER GATE ON ITS OWN, since a later gate would mask these in the command path
    prod = {"protocol_settings": settings, "stage": "tracking",
              "dataset_specific": {"label": "ERA5", "dataset": "era5", "year": 1990, "prefix": "era5"},
              "source_sha256": full}
    assert S.producer_problems(prod, manifest, settings["manifest_sha256"], 1990, "x") == []
    wrong_label = json.loads(json.dumps(prod))
    wrong_label["dataset_specific"]["label"] = "Wrong"
    assert any("label or prefix" in p for p in S.producer_problems(wrong_label, manifest, settings["manifest_sha256"], 1990, "x"))
    partial = json.loads(json.dumps(prod))
    partial["source_sha256"] = {"src/aew/v1port/pipeline.py": "a" * 64}
    assert any("inventory" in p for p in S.producer_problems(partial, manifest, settings["manifest_sha256"], 1990, "x"))
    short_digest = json.loads(json.dumps(prod))
    short_digest["source_sha256"] = dict(full, **{"src/aew/v1port/pipeline.py": "abc"})
    assert any("inventory" in p for p in S.producer_problems(short_digest, manifest, settings["manifest_sha256"], 1990, "x"))
    newline_digest = json.loads(json.dumps(prod))
    newline_digest["source_sha256"] = dict(full, **{"src/aew/v1port/pipeline.py": "a" * 64 + "\n"})
    assert any("inventory" in p for p in S.producer_problems(newline_digest, manifest, settings["manifest_sha256"], 1990, "x"))
    other_rule = json.loads(json.dumps(manifest))
    other_rule["membership"]["rule"] = "the last observation inside the outline"
    assert "membership rule" in S.manifest_agrees_with_this_instrument(other_rule)
    with pytest.raises(SystemExit):        # no manifest
        S.main(["--mode", "reanalysis", "--v1", str(tmp_path / "a.mat"), "--port", str(tmp_path / "b.mat"), *common[:-2], "--out", str(tmp_path / "o5.json")])
    with pytest.raises(SystemExit):        # equal identities are not two datasets
        S.main(["--mode", "reanalysis", "--v1", str(tmp_path / "a.mat"), "--port", str(tmp_path / "a.mat"), *common, "--out", str(tmp_path / "o3.json")])
    with pytest.raises(SystemExit):        # implementation mode still refuses different identities
        S.main(["--v1", str(tmp_path / "a.mat"), "--port", str(tmp_path / "b.mat"), *common, "--out", str(tmp_path / "o4.json")])
