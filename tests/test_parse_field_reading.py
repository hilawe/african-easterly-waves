"""The field-reading parser: both row styles seen in the readings, ranges, placeholder
dashes, two rows at one date, unparsable rows kept and named, and the form screen."""
import importlib.util
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_spec = importlib.util.spec_from_file_location("parse_field_reading_under_test", os.path.join(ROOT, "scripts", "parse_field_reading.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

TEXT = """Per-map table (section 4):
Sequence A, 2006-07-30 00Z: identified; 50; 10 to 20; northerlies west, southerlies east.
Sequence A, 2006-07-30 12Z: weak; 45 to 47; 15 to 25; same signature, somewhat diffuse
Sequence A, 2006-07-31 00Z: indeterminate; \u2014; \u2014; weak signal, competing flow
Sequence A, 2006-07-31 12Z: none.
Sequence A, 2006-08-01 00Z: identified; 22; 10 to 20; first feature
Sequence A, 2006-08-01 00Z: identified; 58; 10 to 20; second feature
Sequence A, 2006-08-01 12Z: identified; 55; 10 to 20; x
Sequence A, 2006-08-02 00Z: identified; ; 10 to 20; a row with no longitude
Sequence B, 2006-09-08 00Z: weak; 48; 15 to 25; blue west-central, red north/south
"""
DATES = ["2006-07-30T00", "2006-07-30T12", "2006-07-31T00", "2006-07-31T12", "2006-08-01T00", "2006-08-01T12", "2006-08-02T00", "2006-08-02T12"]


def test_rows_parse_with_values_ranges_placeholders_and_both_styles():
    rows = P.parse_rows(TEXT)
    assert len(rows) == 9
    r0, r1, r2, r3 = rows[:4]
    assert (r0["date"], r0["status"], r0["axis_lon"], r0["lat_range"]) == ("2006-07-30T00", "identified", 50.0, [10.0, 20.0])
    assert (r1["status"], r1["axis_lon_range"], r1["lat_range"]) == ("weak", [45.0, 47.0], [15.0, 25.0]) and r1["signature"].startswith("same signature")
    assert r2["status"] == "indeterminate" and "axis_lon" not in r2 and r3["status"] == "none"
    assert rows[7]["status"] == "unparsed" and "problem" in rows[7]
    assert rows[8]["sequence"] == "B" and rows[8]["axis_lon"] == 48.0


def test_the_form_screen_counts_duplicates_gaps_statuses_latitudes_and_steps():
    rows = P.parse_rows(TEXT)
    form = P.form_screen(rows, {"A": DATES})["A"]
    assert form["maps_in_figure"] == 8 and form["rows"] == 8 and form["dates_read"] == 7
    assert form["dates_with_two_or_more_rows"] == ["2006-08-01T00"] and form["dates_missing"] == ["2006-08-02T12"] and form["dates_not_in_figure"] == []
    assert form["status_counts"] == {"identified": 4, "weak": 1, "indeterminate": 1, "none": 1, "unparsed": 1}
    assert form["distinct_latitude_ranges"] == [[10.0, 20.0], [15.0, 25.0]]
    # steps count only consecutive figure dates scored once each: 30 Jul 00Z to 12Z (50 to 46), the rest are gaps, duplicates or unparsed
    assert form["longitude_step_deg"]["n"] == 1 and form["longitude_step_deg"]["mean"] == -4.0 and form["longitude_step_deg"]["restarts_over_10_deg"] == 0
    assert form["longitudes"] == {"min": 22.0, "max": 58.0}
