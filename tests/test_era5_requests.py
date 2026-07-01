from aew.data.era5 import VAR_CDS, cds_request


def test_pressure_level_variable_routes_to_pressure_levels_dataset():
    dataset, req = cds_request("r700", 2001, months=(6, 7, 8, 9))
    assert dataset == "reanalysis-era5-pressure-levels"
    assert req["variable"] == "relative_humidity"
    assert req["pressure_level"] == "700"
    assert req["year"] == "2001"
    assert req["month"] == ["06", "07", "08", "09"]


def test_single_level_variable_routes_to_single_levels_dataset_without_level():
    dataset, req = cds_request("tcwv", 2000)
    assert dataset == "reanalysis-era5-single-levels"
    assert req["variable"] == "total_column_water_vapour"
    assert "pressure_level" not in req
    assert req["month"] == [f"{m:02d}" for m in range(1, 13)]


def test_wind_keys_unchanged():
    # the validated wave-series path must keep resolving exactly as before
    assert VAR_CDS["v700"] == ("v_component_of_wind", "700")
    dataset, req = cds_request("v700", 2000)
    assert dataset == "reanalysis-era5-pressure-levels"
    assert req["pressure_level"] == "700"


def test_area_and_grid_pass_through():
    _, req = cds_request("tcwv", 2002, area=[35.0, -45.0, -25.0, 75.0], grid=(0.5, 0.5))
    assert req["area"] == [35.0, -45.0, -25.0, 75.0]
    assert req["grid"] == [0.5, 0.5]
