"""The season-selection rule: highest and lowest count, ties to the earlier year, the pilot
year and absent years excluded, and the table printed with its digests."""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
pytest.importorskip("scipy")


def _load():
    path = os.environ.get("SELECT_SCRIPT", os.path.join(ROOT, "scripts", "select_seasons.py"))
    spec = importlib.util.spec_from_file_location("select_seasons", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_choose_takes_the_extremes_with_ties_to_the_earlier_year_and_excludes_the_pilot():
    S = _load()
    counts = {1983: 150, 1984: 200, 1985: 200, 1986: 120, 1987: 120, 1990: 999}
    assert S.choose(counts, {1990}) == (1984, 1986)
    assert S.choose(counts, set()) == (1990, 1986)
    assert S.choose({}, {1990}) == (None, None)


def test_command_path_counts_by_first_observation_and_records_absent_years(tmp_path, monkeypatch, capsys):
    S = _load()

    def track(day0, n):
        return {"time": day0 + 0.25 * np.arange(n), "meanlat": np.full(n, 10.0), "meanlon": np.full(n, 5.0)}
    june = 33023.0    # 1990-06-01 in days since 1900; the same calendar offset works per year
    record = {1990: [track(june, 6)] * 3,
              1991: [track(june + 365, 6)] * 5,
              1992: [track(june + 731, 6)] * 2 + [track(june + 731 - 3, 6)]}   # 1992 is a leap year, and one starts in May
    monkeypatch.setattr(S.M, "read_record_bytes", lambda blob: [
        {"time": t["time"], "lat": t["meanlat"], "lon": t["meanlon"]} for t in record[int(blob.decode())]])
    rec = tmp_path / "rec"
    rec.mkdir()
    for y in record:
        (rec / f"ERA-Int_ew_700hPa_{y}_AFR.nc").write_bytes(str(y).encode())
    out = tmp_path / "sel.json"
    S.main(["--record-dir", str(rec), "--years", "1990-1993", "--exclude", "1990", "--out", str(out)])
    printed = capsys.readouterr().out
    assert "1991     5  <- highest" in printed and "1992     2  <- lowest" in printed and "1990     3  (excluded)" in printed
    art = json.loads(out.read_text())
    assert art["counts"] == {"1990": 3, "1991": 5, "1992": 2}
    assert art["absent"] == [1993]
    assert art["selected"] == {"highest": 1991, "lowest": 1992}
    assert set(art["record_sha256"]) == {"1990", "1991", "1992"}
