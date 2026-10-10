"""The field-sequence reading aid's contracts: the cadence selects the declared steps, and
the figures carry the fields only, with the dates the record names."""
import importlib.util
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load(name, folder="scripts"):
    spec = importlib.util.spec_from_file_location(name + "_under_test", os.path.join(ROOT, folder, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_cadence_selects_the_declared_steps_from_the_start():
    S = _load("pilot_field_sequence")
    times = 100.0 + np.arange(0, 20) * 0.25                     # six-hourly steps
    assert S.sequence_steps(times, 100.5, 102.0, 12) == [2, 4, 6, 8]
    assert S.sequence_steps(times, 100.5, 102.0, 6) == [2, 3, 4, 5, 6, 7, 8]
    assert S.sequence_steps(times, 100.25, 100.75, 24) == [1]
    assert S.to_day("1900-01-02T12") == 1.5


def test_the_sequence_figures_carry_the_dates_and_no_tracker_output(tmp_path):
    pytest.importorskip("matplotlib")
    S = _load("pilot_field_sequence")
    T = _load("test_pilot_alledge_replay", "tests")
    wide = T.analytic_case(39.0, -144.0, 43.0)
    lat, lon = np.asarray(wide["latgrid"], float), np.asarray(wide["longrid"], float)
    box = [float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max())]
    figs = S.draw_sequence(wide, [0, 1, 2], box, str(tmp_path / "seq"), "test", panels=2)
    assert [f["steps"] for f in figs] == [[0, 1], [2]] and all(os.path.exists(f["path"]) for f in figs)
    assert len(figs[0]["dates"]) == 2 and figs[0]["dates"][0].endswith("Z")
    import matplotlib.image as mpimg
    widths = [mpimg.imread(f["path"]).shape[1] for f in figs]
    assert widths[1] < widths[0]                                         # the one-map figure is sized to its one map
    src = open(os.path.join(ROOT, "scripts", "pilot_field_sequence.py")).read()
    assert "candidate" not in src.split("def draw_sequence")[1].split("def main")[0].replace("no axis, candidate", "")   # the drawing code touches no candidate


def _wide_with_distinct_u(steps=4):
    T = _load("test_pilot_alledge_replay", "tests")
    wide = T.analytic_case(10.0, 28.0, 34.0, steps=steps)
    lat, lon = np.asarray(wide["latgrid"], float), np.asarray(wide["longrid"], float)
    wide["u"] = np.stack([1000.0 * k + 10.0 * lat + 0.1 * lon for k in range(steps)])    # every cell and step distinct, and distinct from v
    return wide


def test_the_default_draws_the_meridional_wind_as_before():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = _load("pilot_field_sequence")
    wide = _wide_with_distinct_u()
    rows, cols = np.arange(wide["latgrid"].shape[0]), np.arange(wide["longrid"].shape[1])
    arr = S.panel_arrays(wide, 1, rows, cols, "v")
    fig, ax = plt.subplots()
    S.draw_panel(ax, wide["longrid"][0], wide["latgrid"][:, 0], arr, "v", extended=False)
    np.testing.assert_array_equal(np.asarray(ax.collections[0].get_array()).reshape(arr["shaded"].shape), wide["v"][1])
    assert ax.collections[0].get_clim() == (-10, 10)
    plt.close(fig)


def test_the_zonal_mode_plots_the_zonal_array_at_its_coordinates_and_dates(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = _load("pilot_field_sequence")
    wide = _wide_with_distinct_u()
    lat1, lon1 = wide["latgrid"][:, 0], wide["longrid"][0]
    rows, cols = np.arange(lat1.size), np.arange(lon1.size)
    arr = S.panel_arrays(wide, 2, rows, cols, "u")
    fig, ax = plt.subplots()
    mesh = S.draw_panel(ax, lon1, lat1, arr, "u", extended=True)
    plotted = np.ma.getdata(mesh.get_array()).reshape(lat1.size, lon1.size)
    np.testing.assert_array_equal(plotted, wide["u"][2])                     # the zonal array of this step, not v, not another step
    corners = mesh.get_coordinates()
    centers = 0.25 * (corners[:-1, :-1] + corners[1:, :-1] + corners[:-1, 1:] + corners[1:, 1:])
    np.testing.assert_allclose(centers[..., 0], np.broadcast_to(lon1, plotted.shape))
    np.testing.assert_allclose(centers[..., 1], np.broadcast_to(lat1[:, None], plotted.shape))
    assert mesh.get_clim() == (-20.0, 20.0) and not np.array_equal(mesh.cmap.get_over(), mesh.cmap.get_under())
    plt.close(fig)
    box = [float(lat1.min()), float(lat1.max()), float(lon1.min()), float(lon1.max())]
    figs = S.draw_sequence(wide, [1, 3], box, str(tmp_path / "zon"), "test", panels=2, shade="u")
    expected = [S.date_of(wide["time"][k]) for k in (1, 3)]
    assert figs[0]["dates"] == expected and [p["date"] for p in figs[0]["panels"]] == expected
    assert [p["step"] for p in figs[0]["panels"]] == [1, 3]


def _write_sp(path, lats, lons, days, values, units="Pa"):
    import netCDF4
    d = netCDF4.Dataset(path, "w")
    d.createDimension("valid_time", len(days)); d.createDimension("latitude", len(lats)); d.createDimension("longitude", len(lons))
    t = d.createVariable("valid_time", "f8", ("valid_time",)); t.units = "seconds since 1970-01-01"
    t[:] = (np.asarray(days, float) - 25567.0) * 86400.0
    d.createVariable("latitude", "f8", ("latitude",))[:] = lats
    d.createVariable("longitude", "f8", ("longitude",))[:] = lons
    sp = d.createVariable("sp", "f8", ("valid_time", "latitude", "longitude")); sp.units = units
    sp[:] = values
    d.close()


def test_the_terrain_mask_reads_aligned_surface_pressure_and_keeps_only_cells_above_ground(tmp_path):
    pytest.importorskip("netCDF4")
    S = _load("pilot_field_sequence")
    lats, lons, days = np.arange(2.0, -0.25, -0.5), np.arange(30.0, 31.25, 0.5), [33124.0, 33124.5]   # half-degree grid
    vals = np.full((2, lats.size, lons.size), 90000.0)
    vals[0, 0, 0] = 70000.0            # equal to the level, below ground by the convention
    vals[0, 2, 2] = 69000.0            # below the level
    vals[1, 4, 0] = np.nan             # no surface pressure
    vals[0, 1, 1] = 50000.0            # a half-degree cell the plotted grid never reads
    path = str(tmp_path / "sp.nc"); _write_sp(path, lats, lons, days, vals)
    sp = S.surface_pressure_on(path, [2.0, 1.0, 0.0], [30.0, 31.0], days)
    assert sp.shape == (2, 3, 2) and sp[0, 0, 0] == 70000.0 and sp[0, 1, 1] == 69000.0 and np.isnan(sp[1, 2, 0])
    wind = np.ones((3, 2)); wind[2, 1] = np.nan
    cls0, cls1 = S.terrain_classes(sp[0], wind), S.terrain_classes(sp[1], wind)
    assert cls0[0, 0] == "terrain" and cls0[1, 1] == "terrain" and cls0[0, 1] == "valid" and cls0[2, 1] == "wind_missing"
    assert cls1[2, 0] == "sp_missing" and cls1[0, 0] == "valid"
    for kwargs, message in ((dict(lat_vals=[2.0], lon_vals=[30.25], step_days=days), "longitude"),
                            (dict(lat_vals=[2.0], lon_vals=[30.0], step_days=[33124.0 + 2.0 / 1440.0]), "time"),
                            (dict(lat_vals=[0.75], lon_vals=[30.0], step_days=days), "latitude")):
        with pytest.raises(SystemExit, match=message):
            S.surface_pressure_on(path, **kwargs)
    hpa = str(tmp_path / "sp_hpa.nc"); _write_sp(hpa, lats, lons, days, vals / 100.0, units="hPa")
    with pytest.raises(SystemExit, match="not Pa"):
        S.surface_pressure_on(hpa, [2.0], [30.0], days)


def test_masked_cells_leave_the_shading_the_vectors_and_the_counts(tmp_path):
    pytest.importorskip("matplotlib")
    S = _load("pilot_field_sequence")
    wide = _wide_with_distinct_u(steps=2)
    lat1, lon1 = wide["latgrid"][:, 0], wide["longrid"][0]
    wide["u"][0] = 0.0                                                     # within the scale everywhere, then two extremes
    wide["u"][0, 3, 2] = -99.0                                               # an extreme value inside terrain
    wide["u"][0, 5, 4] = -25.0                                               # an extreme value above ground
    sp = np.full((1,) + wide["u"].shape[1:], 95000.0); sp[0, 3, 2] = 60000.0
    arr = S.panel_arrays(wide, 0, np.arange(lat1.size), np.arange(lon1.size), "u", sp[0])
    assert np.isnan(arr["shaded"][3, 2]) and np.isnan(arr["u"][3, 2]) and np.isnan(arr["v"][3, 2])
    box = [float(lat1.min()), float(lat1.max()), float(lon1.min()), float(lon1.max())]
    figs = S.draw_sequence(wide, [0], box, str(tmp_path / "mask"), "test", panels=1, shade="u", sp=sp)
    p = figs[0]["panels"][0]
    assert p["classes"]["terrain"] == 1 and p["classes"]["valid"] == p["cells"] - 1
    assert p["out_of_range"]["below"] == 1                                   # the -25 counted, the -99 under ground not


def test_the_reference_is_drawn_only_for_segments_A_and_B_at_their_declared_times(tmp_path):
    pytest.importorskip("matplotlib")
    S = _load("pilot_field_sequence")
    wide = _wide_with_distinct_u(steps=4)
    lat1, lon1 = wide["latgrid"][:, 0], wide["longrid"][0]
    stamp = lambda k: S.date_of(wide["time"][k]).replace(" ", "T")[:13]   # noqa: E731
    reference = {"maps": [{"date": stamp(1), "segment": "A", "axis_lon": 31.0, "lat_range": [2.0, 6.0], "scored": True},
                          {"date": stamp(2), "segment": "tail", "axis_lon": 30.0, "lat_range": [2.0, 6.0], "scored": False},
                          {"date": "1900-01-01T00", "segment": "B", "axis_lon": 31.0, "lat_range": [2.0, 6.0], "scored": True}]}
    box = [float(lat1.min()), float(lat1.max()), float(lon1.min()), float(lon1.max())]
    figs = S.draw_sequence(wide, [0, 1, 2, 3], box, str(tmp_path / "ref"), "test", panels=4, shade="u", reference=reference)
    drawn = [(p["step"], p["reference_overlay"]["segment"]) for p in figs[0]["panels"] if "reference_overlay" in p]
    assert drawn == [(1, "A")]
    ov = figs[0]["panels"][1]["reference_overlay"]
    assert ov["segment_cells"] == 5 and ov["u_on_segment_valid"]["n"] == 5 and ov["axis_lon"] == 31.0


def test_figures_without_a_terrain_mask_are_labeled_unmasked(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = _load("pilot_field_sequence")
    wide = _wide_with_distinct_u(steps=1)
    lat1, lon1 = wide["latgrid"][:, 0], wide["longrid"][0]
    box = [float(lat1.min()), float(lat1.max()), float(lon1.min()), float(lon1.max())]
    titles = []
    real_savefig = plt.Figure.savefig

    def capture(self, *a, **k):
        titles.append(" ".join(self._suptitle.get_text().split()))     # titles are broken into lines, words unchanged
        return real_savefig(self, *a, **k)
    plt.Figure.savefig = capture
    try:
        S.draw_sequence(wide, [0], box, str(tmp_path / "dflt"), "test", panels=1)                       # the default path
        S.draw_sequence(wide, [0], box, str(tmp_path / "zon"), "test", panels=1, shade="u")              # zonal, no mask
        sp = np.full((1,) + wide["u"].shape[1:], 95000.0)
        S.draw_sequence(wide, [0], box, str(tmp_path / "msk"), "test", panels=1, shade="u", sp=sp)       # zonal, masked
    finally:
        plt.Figure.savefig = real_savefig
    assert S.UNMASKED_LABEL in titles[0] and S.UNMASKED_LABEL in titles[1]
    assert S.UNMASKED_LABEL not in titles[2] and "masked below model ground" in titles[2]


def test_a_figure_holds_exactly_its_maps_with_longitude_labels_on_every_column():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = _load("pilot_field_sequence")
    assert [S.panel_grid(n) for n in (1, 3, 4, 5, 8)] == [(1, 1), (1, 3), (1, 4), (2, 4), (2, 4)]
    sizes = {}
    for n in (3, 5, 8):
        fig, used = S.make_figure(plt, n)
        assert len(used) == n and fig.axes == used                      # no hidden axis left in the grid
        fig.canvas.draw()
        ncols = S.panel_grid(n)[1]
        labeled = [i for i, ax in enumerate(used) if any(t.get_visible() and t.get_text() for t in ax.get_xticklabels())]
        assert labeled == [i for i in range(n) if i + ncols >= n]       # the lowest map of every column, and no other
        assert sorted({i % ncols for i in labeled}) == list(range(ncols))
        sizes[n] = tuple(fig.get_size_inches())
        plt.close(fig)
    assert sizes[3][0] < sizes[8][0] and sizes[3][1] < sizes[8][1]      # sized to its maps, not a fixed eight-slot grid


def _write_level(path, name, lats, lons, days, values, level, units="m s**-1"):
    import netCDF4
    d = netCDF4.Dataset(path, "w")
    d.createDimension("valid_time", len(days)); d.createDimension("pressure_level", 1)
    d.createDimension("latitude", len(lats)); d.createDimension("longitude", len(lons))
    t = d.createVariable("valid_time", "f8", ("valid_time",)); t.units = "seconds since 1970-01-01"
    t[:] = (np.asarray(days, float) - 25567.0) * 86400.0
    d.createVariable("pressure_level", "f8", ("pressure_level",))[:] = [level]
    d.createVariable("latitude", "f8", ("latitude",))[:] = lats
    d.createVariable("longitude", "f8", ("longitude",))[:] = lons
    v = d.createVariable(name, "f8", ("valid_time", "pressure_level", "latitude", "longitude")); v.units = units
    v[:] = values
    d.close()


def test_a_level_file_is_read_at_the_plotted_points_and_its_level_is_checked(tmp_path):
    pytest.importorskip("netCDF4")
    S = _load("pilot_field_sequence")
    lats, lons, days = np.arange(2.0, -0.25, -0.5), np.arange(30.0, 31.25, 0.5), [33124.0, 33124.5]
    vals = np.zeros((2, 1, lats.size, lons.size))
    for k in range(2):
        for i, la in enumerate(lats):
            for j, lo in enumerate(lons):
                vals[k, 0, i, j] = 100.0 * k + 10.0 * la + lo          # value encodes step, latitude and longitude
    path = str(tmp_path / "u600.nc"); _write_level(path, "u", lats, lons, days, vals, 600.0)
    got = S.field_on(path, "u", "m s**-1", [2.0, 0.0], [31.0, 30.0], [days[1]], 600.0)
    assert got.shape == (1, 2, 2)
    np.testing.assert_array_equal(got[0], [[100.0 + 20.0 + 31.0, 100.0 + 20.0 + 30.0], [100.0 + 31.0, 100.0 + 30.0]])
    with pytest.raises(SystemExit, match="not the single level 700"):
        S.field_on(path, "u", "m s**-1", [2.0], [30.0], days, 700.0)
    with pytest.raises(SystemExit, match="not m/s|not knots"):
        S.field_on(path, "u", "knots", [2.0], [30.0], days, 600.0)
    with pytest.raises(SystemExit, match="no variable 'v'"):
        S.field_on(path, "v", "m s**-1", [2.0], [30.0], days, 600.0)


def test_the_terrain_rule_uses_the_level_drawn():
    S = _load("pilot_field_sequence")
    wide = _wide_with_distinct_u(steps=1)
    rows, cols = np.arange(wide["latgrid"].shape[0]), np.arange(wide["longrid"].shape[1])
    sp = np.full(wide["u"].shape[1:], 95000.0); sp[2, 3] = 65000.0                  # 650 hPa: above 600, below 700
    at600 = S.panel_arrays(wide, 0, rows, cols, "v", sp, 600.0)
    at700 = S.panel_arrays(wide, 0, rows, cols, "v", sp, 700.0)
    assert at600["classes"][2, 3] == "valid" and np.isfinite(at600["v"][2, 3])
    assert at700["classes"][2, 3] == "terrain" and np.isnan(at700["v"][2, 3])


def test_steps_drawn_from_a_level_file_are_renumbered_to_the_wide_case():
    S = _load("pilot_field_sequence")
    figs = [{"steps": [0, 1], "dates": ["a", "b"], "panels": [{"step": 0, "date": "a"}, {"step": 1, "date": "b"}]}]
    out = S.remap_steps(figs, [1004, 1006])
    assert out[0]["steps"] == [1004, 1006] and [p["step"] for p in out[0]["panels"]] == [1004, 1006]
    assert figs[0]["steps"] == [0, 1]                                               # the input is not mutated


def test_another_level_without_level_files_is_refused(tmp_path):
    S = _load("pilot_field_sequence")
    with pytest.raises(SystemExit, match="another level needs --level-files"):
        S.main(["--wide-dir", "x", "--start", "1990-09-07T00", "--end", "1990-09-07T12", "--out", str(tmp_path / "o.json"),
                "--png-prefix", str(tmp_path / "p"), "--label", "t", "--year", "1990", "--box", "0", "30", "20", "62", "--level-hpa", "600"])


def test_a_year_without_a_wide_case_takes_its_grid_and_times_from_the_level_file(tmp_path):
    pytest.importorskip("netCDF4")
    S = _load("pilot_field_sequence")
    lats, lons = np.arange(2.0, -1.25, -0.5), np.arange(30.0, 31.25, 0.5)
    days = [S.to_day("2000-07-28T00"), S.to_day("2000-07-28T06")]
    path = str(tmp_path / "u700.nc"); _write_level(path, "u", lats, lons, days, np.zeros((2, 1, lats.size, lons.size)), 700.0)
    g = S.grid_from_level_file(path, 2000)
    assert g["latgrid"][:, 0].tolist() == [2.0, 1.0, 0.0, -1.0] and g["longrid"][0].tolist() == [30.0, 31.0]   # integer points only, file order
    np.testing.assert_allclose(g["time"], days)
    with pytest.raises(SystemExit, match="not 1999 alone"):
        S.grid_from_level_file(path, 1999)
    with pytest.raises(SystemExit, match="must come from --level-files"):
        S.main(["--start", "2000-07-28T00", "--end", "2000-07-28T06", "--out", str(tmp_path / "o.json"), "--png-prefix", str(tmp_path / "p"),
                "--label", "t", "--year", "2000", "--box", "0", "2", "30", "31"])


def test_relative_vorticity_matches_linear_fields_on_the_sphere():
    S = _load("pilot_field_sequence")
    lat1, lon1 = np.array([12.0, 11.0, 10.0, 9.0]), np.array([20.0, 21.0, 22.0, 23.0])   # north to south, as ERA5
    a, phi, lam = S.EARTH_RADIUS_M, np.deg2rad(lat1)[:, None], np.deg2rad(lon1)[None, :]
    ok = np.full((4, 4), "valid", dtype=object)
    s = 3e-5
    zeta, cls = S.relative_vorticity(np.zeros((4, 4)), s * a * np.cos(phi) * lam, lat1, lon1, ok)   # dv/dx = s exactly
    np.testing.assert_allclose(zeta[1:-1, 1:-1], s, rtol=1e-9)
    c = 2e-5
    u = c * a * phi * np.ones((1, 4))                                                       # du/dy = c, plus the metric term
    zeta, cls = S.relative_vorticity(u, np.zeros((4, 4)), lat1, lon1, ok)
    np.testing.assert_allclose(zeta[1:-1, 1:-1], (-c + c * phi * np.tan(phi))[1:-1] * np.ones((1, 2)), rtol=1e-9)
    assert np.isnan(zeta[0]).all() and np.isnan(zeta[:, 0]).all() and set(cls[0]) == {"edge"}


def test_vorticity_needs_every_stencil_neighbor_above_ground():
    S = _load("pilot_field_sequence")
    lat1, lon1 = np.arange(14.0, 9.0, -1.0), np.arange(20.0, 25.0)
    wind = np.full((5, 5), "valid", dtype=object); wind[2, 2] = "terrain"
    zeta, cls = S.relative_vorticity(np.ones((5, 5)), np.ones((5, 5)), lat1, lon1, wind)
    assert cls[2, 2] == "terrain" and np.isnan(zeta[2, 2])
    for r, c in ((1, 2), (3, 2), (2, 1), (2, 3)):
        assert cls[r, c] == "neighbor_invalid" and np.isnan(zeta[r, c])
    for r, c in ((1, 1), (1, 3), (3, 1), (3, 3)):                                          # diagonal cells are outside the stencil
        assert cls[r, c] == "valid" and np.isfinite(zeta[r, c])


def test_markers_are_drawn_only_at_their_own_time_and_never_joined(tmp_path):
    pytest.importorskip("matplotlib")
    S = _load("pilot_field_sequence")
    wide = _wide_with_distinct_u(steps=2)
    lat, lon = np.asarray(wide["latgrid"], float), np.asarray(wide["longrid"], float)
    box = [float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max())]
    t1 = S.date_of(np.asarray(wide["time"], float).ravel()[1]).replace(" ", "T").rstrip("Z")
    mlon, mlat = float(lon[0, 2]), float(lat[2, 0])
    markers = {"sets": [{"label": "center", "marker": "o", "color": "#00a000", "positions": [{"time": t1, "lat": mlat, "lon": mlon}]},
                        {"label": "tick", "marker": "|", "color": "#00bcd4", "positions": [{"time": t1, "lat_range": [mlat - 1, mlat + 1], "lon": mlon}]}]}
    figs = S.draw_sequence(wide, [0, 1], box, str(tmp_path / "mk"), "test", panels=2, shade="zeta", markers=markers)
    p0, p1 = figs[0]["panels"]
    assert p0["markers_drawn"] == [] and [d["set"] for d in p1["markers_drawn"]] == ["center", "tick"]
    assert set(p1["classes"]) == set(S.ZETA_CLASSES) and p1["classes"]["edge"] > 0
