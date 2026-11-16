#!/usr/bin/env python
"""The class contrasts split exactly by scale: years, cells, waves and passages.

The canonical split classes each trough against thresholds pooled over all years, so a
contrast between the classes mixes four things: which YEARS the classes are drawn from,
which year-month-longitude CELLS inside a year, which WAVES inside a cell, and which
PASSAGES of one wave. This script separates them with an identity. Each observation's
value is written as

    its year mean
  + its cell mean less its year mean
  + the mean of its wave's observations in that cell, less the cell mean
  + its departure from that wave mean

and the class difference of each term is one part of the contrast. The four parts add to
the contrast exactly. The means run over a stated BACKGROUND, either every eligible trough
(for the measures the deposit holds for every eligible trough) or the classified
observations (for the measures it holds only for them).

Uncertainty is a wave-cluster bootstrap in which whole waves enter with a multiplicity,
and the year means, cell means and class means are all recomputed in every replicate, so
an interval on a part carries the uncertainty of the background it is measured against.
The class labels are attributes of the observations and are not recomputed. One matrix of
multiplicities serves every measure and labeling of a sample, so differences between
measures are paired.

AMENDED AFTER THE FIRST RUN (2026-11-13). The interval above was the one declared, and the
first run showed that it is shifted for two of the parts. A replicate holds fewer distinct
waves in each cell than the sample does, so its cell means follow its waves more closely,
and part of the contrast between waves moves into the contrast between cells. For the
forward response the point value fell outside its own interval. Every row therefore now
carries, beside the declared interval, the median of its replicates, which shows the
shift, and a second interval in which the year, cell and wave means are held at their
sample values and only the class means are resampled. The second interval is centered on
the point value and describes the split of this sample with its background taken as
given. Neither interval removes the cause, which is that a mean taken over a few waves
carries some of what distinguishes those waves. The amendment adds fields and changes no
declared one.

CORRECTED AFTER THE FIRST RUN (2026-11-13). The first run found a trough's longitude bin
by dividing by the bin width, which put one trough at longitude -1.1e-16 in the bin east
of 0 where the canonical comparison puts it west of 0. The class labels were not affected,
since they are checked against the deposit row by row, but that trough's cell was, and the
parts between cells, waves and passages moved in the third decimal. The bin is now found
by comparison and is checked against the canonical rule for every trough.

Four labelings are compared:

  canonical      terciles of the forward response within longitude-month cells
  year-matched   the same within year-month-longitude cells (30-observation minimum)
  era            the same within longitude-month cells taken separately before and from 1998
  merged         canonical terciles of a response recounted after same-time detections
                 lying within a stated distance of each other are merged into one

AMENDED 2026-11-13, LABELS FROM ONE SATELLITE POSITION. The catalog file pools the rows of
every satellite position of its source and drops the position code that each source row
carries, and every close same-time pair in it is one record from each of two positions
(read from the position code of the source rows on 2026-11-13). When the pulled source
rows are named with --positions, two more labelings are computed from the rows of the MET
position alone, the one position that views the whole corridor in every season:

  MET alone                 canonical terciles of a response recounted from that position
  MET alone, year-matched   the same response within year-month-longitude cells

Before either is computed the pulled rows of every position must equal the catalog's
detections inside the pulled box, detection for detection, and must give the deposited
response and antecedent count for every eligible trough. Without --positions the script
computes what it computed before, and the rows of the earlier labelings do not change
either way, since a sample's replicates depend on its name and the seed alone.

AMENDED 2026-11-14, A SECOND CATALOG ON THE SAME TROUGHS. Since the input rules of
2026-11-13 the deposit counts its classes from the MET position on the troughs whose
counting windows pass the availability criterion, so the canonical labels of such a deposit
are already those of one position. When a second catalog is named with --alt-catalog, two
more labelings are computed from its counts on the same troughs, the canonical form and
the year-matched form, under the name given with --alt-label. Before either is computed
the response recounted from that catalog must equal the response of the deposit's own
sensitivity build on that catalog, deposit/sens_pooled/troughs_pooled.csv, trough for
trough. The number of eligible troughs and waves that the checks require is read from the
deposit's count rows unless --expect names it.

A pure consumer of the deposit and of the cloud-system catalog. It integrates nothing,
draws from its own random streams (one per sample, keyed by the sample's name) and writes
no file that another script reads. Before anything is computed it checks that it rebuilds
the deposited class labels, the deposited response and antecedent counts, and the
registry's contrasts, and it stops if any of them differs.

Writes scale_split_record.json and scale_split.csv to --outdir.
"""

import argparse
import gzip
import hashlib
import json
import os
import zlib

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from aew.environment import forward_response, stratified_terciles, terciles

N_BOOT = 20_000
CHUNK = 1000
EDGES = np.arange(-30, 41, 10.0)
M_EDGES = np.array([6.5, 7.5, 8.5, 9.5])
MIN_BIN = 30
RESP_WIN_H, DLON, LAT_LO, LAT_HI = 24.0, 8.0, 5.0, 15.0
ERA_YEAR = 1998
MERGE_DEG = 1.0
MERGE_DEG_WIDE = 2.0
MERGE_BOX = (0.0, 20.0, -45.0, 55.0)          # lat_lo, lat_hi, lon_lo, lon_hi
POSITIONS = ("GOE", "GOW", "GMS", "MET", "INS")   # the satellite positions of the cloud-system source
POSITION = "MET"                                   # the position kept by the labelings from one position
MIN_RADIUS_KM = 101                                # the smallest radius the catalog file holds
PULL_BOX = (-2.0, 22.0, -50.0, 60.0)               # lat_lo, lat_hi, lon_lo, lon_hi of the pulled source rows
ONE_POSITION = (f"{POSITION} alone", f"{POSITION} alone, year-matched")
ALT_LABEL = "every position pooled"                # the default name of the labelings from a second catalog
ALT_DEPOSIT = "sens_pooled"                        # the deposit's own build on the second catalog
EPOCHS = (("1983 to 1993", 1983, 1993), ("1994 to 2007", 1994, 2007))
ERAS = (("1983 to 1997", 1983, 1997), ("1998 to 2007", 1998, 2007))
SECTORS = (("30 W to 0", -30.0, 0.0), ("0 to 10 E", 0.0, 10.0), ("10 E to 40 E", 10.0, 40.0))
PARTS = ("years", "cells", "waves", "passages")
QUANTITIES = ("total",) + PARTS + ("years and cells", "waves and passages")

ELIGIBLE = (("inflow_rh", "inflow RH at -72 h, 700 hPa"), ("box_rh", "fixed box RH at -24 h, 700 hPa"),
            ("tcwv", "column water vapor, mm"), ("response", "forward response, count"),
            ("antecedent", "antecedent count"))
LEADS = (24, 36, 48, 60, 72)


def classified_measures(level):
    out = [(f"rh_m{h}", f"inflow RH at -{h} h, {level} hPa", "lagrangian_rh", -float(h)) for h in LEADS]
    out += [(f"thetae_m{h}", f"inflow theta-e at -{h} h, {level} hPa, K", "thetae", -float(h)) for h in (24, 72)]
    out += [("mix_m72", f"inflow mixing ratio at -72 h, {level} hPa, g/kg", "decomp_mix", -72.0),
            ("T_m72", f"inflow temperature at -72 h, {level} hPa, K", "decomp_T", -72.0)]
    if level == 850:
        out.append(("box_rh_m24", "fixed box RH at -24 h, 850 hPa", "eulerian_box", -24.0))
    return out


# --------------------------------------------------------------------------- labels
def cell_terciles(response, keys, min_bin=MIN_BIN):
    """Lower and upper tercile masks of ``response`` within the cells given by ``keys``.

    ``keys`` is a list of equal-length arrays. A row with a missing key lies in no cell.
    Thresholds are the canonical ones (``aew.environment.terciles``, the linear quantiles
    at one third and two thirds with boundary-equal values entering the outer class), and
    a cell under ``min_bin`` rows takes no part.
    """
    response = np.asarray(response, float)
    frame = pd.DataFrame({f"k{i}": np.asarray(k, float) for i, k in enumerate(keys)})
    lo = np.zeros(response.size, bool)
    hi = np.zeros(response.size, bool)
    whole = np.flatnonzero(frame.notna().all(axis=1).values)
    for _, pos in frame.iloc[whole].groupby(list(frame.columns), sort=False).indices.items():
        idx = whole[np.asarray(pos)]
        if idx.size < min_bin:
            continue
        bl, bh = terciles(response[idx])
        lo[idx[bl]] = True
        hi[idx[bh]] = True
    return lo, hi


def lon_bin(lon):
    """Index of the canonical half-open 10-degree bin, NaN outside [-30, 40).

    Found by comparing with the edges, as the canonical split does, and not by arithmetic.
    A longitude a rounding error below an edge belongs to the bin below it, and dividing
    by the bin width puts it in the bin above (one trough of the record sits at -1.1e-16).
    """
    lon = np.asarray(lon, float)
    b = np.searchsorted(EDGES, lon, side="right") - 1.0
    return np.where((lon >= EDGES[0]) & (lon < EDGES[-1]), b, np.nan)


# --------------------------------------------------------------------------- the catalog
def merge_same_time(t_ns, lon, lat, max_deg, box=MERGE_BOX):
    """Merge detections that carry one time stamp and lie within ``max_deg`` of each other.

    Inside ``box`` the detections of a time stamp are joined wherever two of them are no
    farther apart than ``max_deg`` degrees (longitude differences scaled by the cosine of
    the pair's mean latitude), and every connected group is replaced by one detection at
    the group's mean position. Detections outside the box are returned unchanged. Returns
    the merged (time, lon, lat) and the number of detections removed.
    """
    t_ns = np.asarray(t_ns).astype("datetime64[ns]").astype("int64")
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    inside = (lat >= box[0]) & (lat <= box[1]) & (lon >= box[2]) & (lon <= box[3])
    keep_t, keep_x, keep_y = [t_ns[~inside]], [lon[~inside]], [lat[~inside]]
    ti, xi, yi = t_ns[inside], lon[inside], lat[inside]
    if ti.size == 0:                                 # nothing inside the box, so nothing to merge
        return t_ns.astype("datetime64[ns]"), lon.copy(), lat.copy(), 0
    order = np.argsort(ti, kind="stable")
    ti, xi, yi = ti[order], xi[order], yi[order]
    starts = np.flatnonzero(np.r_[True, ti[1:] != ti[:-1]])
    ends = np.r_[starts[1:], ti.size]
    removed = 0
    for a, b in zip(starts, ends):
        n = b - a
        if n == 1:
            keep_t.append(ti[a:b]); keep_x.append(xi[a:b]); keep_y.append(yi[a:b])
            continue
        x, y = xi[a:b], yi[a:b]
        dy = y[:, None] - y[None, :]
        dx = (x[:, None] - x[None, :]) * np.cos(np.deg2rad(0.5 * (y[:, None] + y[None, :])))
        near = np.hypot(dx, dy) <= max_deg
        k, lab = connected_components(coo_matrix(near), directed=False)
        removed += n - k
        sums_x = np.bincount(lab, weights=x, minlength=k)
        sums_y = np.bincount(lab, weights=y, minlength=k)
        cnt = np.bincount(lab, minlength=k)
        keep_t.append(np.full(k, ti[a])); keep_x.append(sums_x / cnt); keep_y.append(sums_y / cnt)
    return (np.concatenate(keep_t).astype("datetime64[ns]"), np.concatenate(keep_x),
            np.concatenate(keep_y), int(removed))


# --------------------------------------------------------------------------- one satellite position
def stored_coordinates(lon, lat):
    """Longitude and latitude as the catalog file stores them.

    The file holds single-precision values, and a western longitude in it is the
    single-precision eastern value less 360, taken in single precision. A count compares
    these values with a trough's longitude and with the latitude limits, so a detection
    lying exactly on a limit in tenths of a degree falls on one side or the other by the
    way its coordinate was formed. The pulled rows give longitude from 180 W to 180 E.
    """
    lon = np.asarray(lon, float)
    east = np.where(lon < 0.0, lon + 360.0, lon).astype("f4")
    x = np.where(east > np.float32(180.0), east - np.float32(360.0), east)
    return x.astype(float), np.asarray(lat, float).astype("f4").astype(float)


def load_positions(path, years):
    """The pulled source rows of at least ``MIN_RADIUS_KM`` in radius, with their position.

    A row of the pull reads ``R position year month day hour number radius latitude
    longitude``. Lines of any other kind are passed over. Returns the time, the stored
    coordinates and the position code of every row of the given years.
    """
    sat, when, lon, lat = [], [], [], []
    read = 0
    with gzip.open(path, "rt", encoding="ascii", errors="replace") as f:
        for line in f:
            if not line.startswith("R "):
                continue
            p = line.split()
            if len(p) != 10:
                refuse(f"a pulled row has {len(p)} fields where 10 are expected")
            if p[1] not in POSITIONS:
                refuse(f"a pulled row carries the unknown position {p[1]!r}")
            read += 1
            if int(p[7]) < MIN_RADIUS_KM:
                continue
            sat.append(p[1])
            when.append((int(p[2]), int(p[3]), int(p[4]), int(p[5])))
            lat.append(float(p[8]))
            lon.append(float(p[9]))
    if not sat:
        refuse("the pull holds no row of the smallest radius or more")
    w = np.array(when)
    t = pd.to_datetime(dict(year=w[:, 0], month=w[:, 1], day=w[:, 2], hour=w[:, 3])).values.astype("datetime64[ns]")
    x, y = stored_coordinates(lon, lat)
    keep = np.isin(w[:, 0], years)
    return dict(t=t[keep], x=x[keep], y=y[keep], sat=np.array(sat)[keep], rows_read=int(read))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- the split
class Sample:
    """One set of observations with its wave, year and cell indices."""

    def __init__(self, wave, year, cell):
        self.wave_ids, self.w = np.unique(np.asarray(wave), return_inverse=True)
        self.year_ids, self.y = np.unique(np.asarray(year), return_inverse=True)
        self.cell_ids, self.c = np.unique(np.asarray(cell), return_inverse=True)
        self.W, self.Y, self.C = self.wave_ids.size, self.year_ids.size, self.cell_ids.size
        self.n = self.w.size

    def _sum(self, cols, ncol, weights, mask):
        out = np.zeros((self.W, ncol))
        np.add.at(out, (self.w[mask], cols[mask]), weights[mask])
        return out

    def measure(self, x, lo, hi, background):
        """Matrices of one measure under one labeling, ready for any multiplicities."""
        x = np.asarray(x, float)
        fin = np.isfinite(x)
        bg = np.asarray(background, bool) & fin
        lo = np.asarray(lo, bool) & fin
        hi = np.asarray(hi, bool) & fin
        if ((lo | hi) & ~bg).any():
            raise ValueError("a classified observation lies outside the background")
        if (lo & hi).any():
            raise ValueError("an observation is in both classes")
        xz = np.where(fin, x, 0.0)
        one = np.ones(self.n)
        Sy, Ny = self._sum(self.y, self.Y, xz, bg), self._sum(self.y, self.Y, one, bg)
        Sc, Nc = self._sum(self.c, self.C, xz, bg), self._sum(self.c, self.C, one, bg)
        with np.errstate(invalid="ignore", divide="ignore"):
            wbar = Sc[self.w, self.c] / Nc[self.w, self.c]     # the wave's mean in its cell
            yfull = (Sy.sum(axis=0) / Ny.sum(axis=0))[self.y]  # the sample's own year and cell means
            cfull = (Sc.sum(axis=0) / Nc.sum(axis=0))[self.c]
        held = np.column_stack([yfull, cfull - yfull, wbar - cfull, xz - wbar])
        cls = {}
        for name, m in (("lo", lo), ("hi", hi)):
            F = np.zeros((self.W, 4))
            np.add.at(F, self.w[m], held[m])
            cls[name] = dict(
                Ay=self._sum(self.y, self.Y, one, m), Ac=self._sum(self.c, self.C, one, m),
                S=np.bincount(self.w[m], weights=xz[m], minlength=self.W),
                N=np.bincount(self.w[m], minlength=self.W).astype(float),
                T=np.bincount(self.w[m], weights=wbar[m], minlength=self.W), F=F,
                n=int(m.sum()), waves=int(np.unique(self.w[m]).size))
        return dict(Sy=Sy, Ny=Ny, Sc=Sc, Nc=Nc, cls=cls)


def split_values(mats, M):
    """The contrast and its four parts for every row of the multiplicity matrix ``M``.

    Returns a dict of arrays of length ``M.shape[0]``. A replicate in which a class holds
    no observation is NaN throughout.
    """
    M = np.asarray(M, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        dy, dc = M @ mats["Ny"], M @ mats["Nc"]
        ybar = np.where(dy > 0, (M @ mats["Sy"]) / dy, 0.0)
        cbar = np.where(dc > 0, (M @ mats["Sc"]) / dc, 0.0)
        side = {}
        for name, k in mats["cls"].items():
            n = M @ k["N"]
            n = np.where(n > 0, n, np.nan)
            tot = (M @ k["S"]) / n
            p_year = (ybar * (M @ k["Ay"])).sum(axis=1) / n
            p_cell = (cbar * (M @ k["Ac"])).sum(axis=1) / n
            p_wave = (M @ k["T"]) / n
            side[name] = dict(total=tot, years=p_year, cells=p_cell - p_year,
                              waves=p_wave - p_cell, passages=tot - p_wave)
    out = {q: side["hi"][q] - side["lo"][q] for q in ("total",) + PARTS}
    out["years and cells"] = out["years"] + out["cells"]
    out["waves and passages"] = out["waves"] + out["passages"]
    return out


def split_values_held(mats, M):
    """The same quantities with the year, cell and wave means held at their sample values.

    Each observation keeps the four terms it has in the sample, and a replicate resamples
    the class means of those terms alone. With every multiplicity equal to one this is the
    sample's own split, the same as ``split_values``.
    """
    M = np.asarray(M, float)
    side = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for name, k in mats["cls"].items():
            n = M @ k["N"]
            n = np.where(n > 0, n, np.nan)
            side[name] = (M @ k["F"]) / n[:, None]
    d = side["hi"] - side["lo"]
    out = {q: d[:, i] for i, q in enumerate(PARTS)}
    out["total"] = d.sum(axis=1)
    out["years and cells"] = out["years"] + out["cells"]
    out["waves and passages"] = out["waves"] + out["passages"]
    return out


def multiplicities(n_waves, n_boot, name, seed):
    """Chunks of wave multiplicities for the sample called ``name``.

    Each replicate draws ``n_waves`` waves with replacement, which is the paper's wave
    resampling. The stream depends on the seed and the sample's name alone, so the order
    in which samples are computed cannot move a number.
    """
    rng = np.random.default_rng([int(seed), zlib.crc32(name.encode("utf-8"))])
    p = np.full(n_waves, 1.0 / n_waves)
    done = 0
    while done < n_boot:
        b = min(CHUNK, n_boot - done)
        yield rng.multinomial(n_waves, p, size=b).astype(float)
        done += b


def interval(point, reps):
    reps = np.asarray(reps, float)
    ok = np.isfinite(reps)
    if not ok.any() or not np.isfinite(point):
        raise ValueError("no finite replicate, or a point value that is not finite")
    lo, hi = np.percentile(reps[ok], [2.5, 97.5])
    return dict(diff=float(point), low=float(lo), high=float(hi), draws_used=int(ok.sum()))


def both(point, reps, held):
    """The declared interval of one quantity with the two additions of the amendment."""
    out = interval(point, reps)
    h = interval(point, held)
    out.update(replicate_median=float(np.nanmedian(reps)), held_low=h["low"], held_high=h["high"])
    return out


def run_sample(sample, name, jobs, n_boot, seed, pairs=()):
    """Point values, replicate arrays and intervals for every job of one sample.

    ``jobs`` maps a key to the matrices of one measure under one labeling. ``pairs`` lists
    (first key, second key) whose difference, first less second, is wanted on the same
    replicates. Returns (rows, replicate arrays by key, paired rows). The replicate arrays
    are a dict with the declared replicates under "recomputed" and those of the amendment
    under "held".
    """
    ones = np.ones((1, sample.W))
    point = {k: {q: float(v[0]) for q, v in split_values(m, ones).items()} for k, m in jobs.items()}
    for k, m in jobs.items():
        again = split_values_held(m, ones)
        assert all(abs(float(again[q][0]) - point[k][q]) < 1e-8 for q in QUANTITIES), k
    reps = {kind: {k: {q: [] for q in QUANTITIES} for k in jobs} for kind in ("recomputed", "held")}
    for M in multiplicities(sample.W, n_boot, name, seed):
        for k, m in jobs.items():
            for q, v in split_values(m, M).items():
                reps["recomputed"][k][q].append(v)
            for q, v in split_values_held(m, M).items():
                reps["held"][k][q].append(v)
    reps = {kind: {k: {q: np.concatenate(v) for q, v in d.items()} for k, d in by.items()} for kind, by in reps.items()}
    rows = {}
    for k, m in jobs.items():
        assert abs(sum(point[k][p] for p in PARTS) - point[k]["total"]) < 1e-9
        rows[k] = dict(n_quiet=m["cls"]["lo"]["n"], n_active=m["cls"]["hi"]["n"],
                       waves_quiet=m["cls"]["lo"]["waves"], waves_active=m["cls"]["hi"]["waves"],
                       waves_resampled=int(sample.W),
                       quantities={q: both(point[k][q], reps["recomputed"][k][q], reps["held"][k][q]) for q in QUANTITIES})
    paired = {}
    for a, b in pairs:
        paired[(a, b)] = {q: both(point[a][q] - point[b][q], reps["recomputed"][a][q] - reps["recomputed"][b][q],
                                  reps["held"][a][q] - reps["held"][b][q]) for q in QUANTITIES}
    return rows, reps, paired


# --------------------------------------------------------------------------- inputs and checks
def refuse(msg):
    raise SystemExit(f"REFUSED: {msg}")


def load_inputs(dep):
    e = pd.read_csv(os.path.join(dep, "control_model_design_eligible.csv"))
    e["time"] = pd.to_datetime(e["time"])
    e["month"] = e.time.dt.month.astype(float)
    e["lonbin"] = lon_bin(e.lon.values)
    tr = pd.read_csv(os.path.join(dep, "troughs_pooled.csv"))
    tr["time"] = pd.to_datetime(tr["time"])
    j = e.merge(tr[["traj_id", "time", "label", "response"]].rename(columns={"traj_id": "wave", "response": "response_dep"}),
                on=["wave", "time"], how="left", validate="one_to_one")
    if len(j) != len(e) or j.label.isna().any():
        refuse("the eligible troughs and the deposited trough table do not pair one to one")
    return j


def registry(dep):
    r = pd.read_csv(os.path.join(dep, "canonical_numbers.csv"))
    r = r[(r.tier == "pooled") & (r.kind == "contrast")]

    def get(level, statistic, hour):
        m = (r.statistic == statistic) & np.isclose(r.time_rel_h, hour)
        m &= r.level.isna() if level is None else np.isclose(r.level, float(level))
        if int(m.sum()) != 1:
            refuse(f"registry row {level} {statistic} {hour} found {int(m.sum())} times")
        row = r[m].iloc[0]
        return float(row["diff"]), int(row.n_quiet), int(row.n_active)
    return get


def expected_sample(dep):
    """The eligible troughs and waves that the deposit's own count rows state."""
    r = pd.read_csv(os.path.join(dep, "canonical_numbers.csv"))
    out = []
    for stat in ("n_corridor_troughs", "n_waves"):
        m = (r.tier == "pooled") & (r.statistic == stat)
        if int(m.sum()) != 1 or not np.isfinite(r["diff"][m]).all():
            refuse(f"the deposit states {stat} {int(m.sum())} times or without a value")
        out.append(int(round(float(r["diff"][m].iloc[0]))))
    return tuple(out)


def alt_names(label):
    return (label, f"{label}, year-matched")


def check_alt(e, dep, resp, label):
    """The second catalog bound to the deposit's own build on it. Stops on a mismatch.

    The labelings from a second catalog are a statement about the deposit's sensitivity
    build only if both count the same thing, so the response recounted here must equal
    the response of that build for every eligible trough, paired by wave and time.
    """
    path = os.path.join(dep, ALT_DEPOSIT, "troughs_pooled.csv")
    if not os.path.exists(path):
        refuse(f"{path} is absent, so the counts from the second catalog cannot be bound to the deposit")
    tr = pd.read_csv(path)
    tr["time"] = pd.to_datetime(tr["time"])
    j = e[["wave", "time"]].merge(tr[["traj_id", "time", "response"]].rename(columns={"traj_id": "wave"}),
                                  on=["wave", "time"], how="left", validate="one_to_one")
    if len(j) != len(e) or j.response.isna().any():
        refuse("the eligible troughs and the trough table of the second catalog's build do not pair one to one")
    if not np.array_equal(np.asarray(resp, float), j.response.values.astype(float)):
        refuse("the response recounted from the second catalog differs from the deposit's build on it")
    return {f"{label}: the response recounted from the second catalog equals the deposit's build on it": int(len(e))}


def load_catalog(path, years):
    cs = xr.open_dataset(path)
    t = pd.DatetimeIndex(cs["time"].values)
    x = np.asarray(cs["lon"].values, float)
    y = np.asarray(cs["lat"].values, float)
    cs.close()
    m = np.isin(t.year, years)
    return t[m].values, x[m], y[m]


def counts(e, t, x, y):
    resp = forward_response(e.time.values, e.lon.values, t, x, y, RESP_WIN_H, DLON, LAT_LO, LAT_HI)
    prior = e.time.values.astype("datetime64[ns]") - np.timedelta64(48, "h")
    ante = forward_response(prior, e.lon.values, t, x, y, 24.0, DLON + 2.0, LAT_LO, LAT_HI)
    return resp, ante


def contrast(x, lo, hi):
    x = np.asarray(x, float)
    return float(np.nanmean(x[hi]) - np.nanmean(x[lo]))


def finite_close(value, reference, tolerance):
    """True only when both numbers are finite and no farther apart than ``tolerance``.

    A comparison written as ``abs(a - b) > tolerance`` is false when either number is
    missing, so a missing contrast would pass a check built on it.
    """
    return bool(np.isfinite(value) and np.isfinite(reference) and abs(value - reference) <= tolerance)


def check_inputs(e, lab, dep, cat_counts, expect=(11457, 1002)):
    """Every binding check. Returns the record of what was compared; stops on a mismatch."""
    out = {}
    if (len(e), e.wave.nunique()) != tuple(expect):
        refuse(f"expected {expect[0]:,} eligible troughs on {expect[1]:,} waves, found {len(e)} on {e.wave.nunique()}")
    lo, hi = lab["canonical"]
    dep_lo = (e.label == "MCS-quiet").values
    dep_hi = (e.label == "MCS-active").values
    if not (np.array_equal(lo, dep_lo) and np.array_equal(hi, dep_hi)):
        refuse("the rebuilt canonical labels differ from the deposited labels")
    out["canonical labels equal the deposit, row by row"] = [int(lo.sum()), int(hi.sum())]
    for i in range(EDGES.size - 1):
        inside = (e.lon.values >= EDGES[i]) & (e.lon.values < EDGES[i + 1])
        if not np.array_equal(inside, e.lonbin.values == i):
            refuse(f"the longitude bin {i} differs from the canonical comparison with its edges")
    out["every longitude bin equals the canonical comparison with its edges"] = int(np.isfinite(e.lonbin.values).sum())
    lo2, hi2 = cell_terciles(e.response.values, [e.month.values, e.lonbin.values])
    if not (np.array_equal(lo, lo2) and np.array_equal(hi, hi2)):
        refuse("the cell-tercile routine does not reproduce the canonical split")
    out["the cell-tercile routine reproduces the canonical split"] = True
    ys = pd.read_csv(os.path.join(dep, "yearstrat_sensitivity.csv")).set_index("split")
    ly, hy = lab["year-matched"]
    d = contrast(e.inflow_rh, ly, hy)
    if (int(ly.sum()), int(hy.sum())) != (int(ys.n_quiet["yearmonthlon"]), int(ys.n_active["yearmonthlon"])) \
            or not finite_close(d, float(ys["diff"]["yearmonthlon"]), 1e-5):
        refuse("the year-matched labels do not reproduce deposit/yearstrat_sensitivity.csv")
    out["year-matched labels reproduce the deposited sensitivity"] = [int(ly.sum()), int(hy.sum()), d]
    get = registry(dep)
    for col, level, stat, hour in (("inflow_rh", 700, "lagrangian_rh", -72.0), ("box_rh", 700, "eulerian_box", -24.0),
                                   ("tcwv", None, "tcwv_box", -24.0), ("antecedent", 700, "antecedent_convection", -48.0)):
        ref, nq, na = get(level, stat, hour)
        d = contrast(e[col], lo, hi)
        if not finite_close(d, ref, 1e-5) or (nq, na) != (int(lo.sum()), int(hi.sum())):
            refuse(f"{col}: contrast {d:.6f} against the registry's {ref:.6f}")
        out[f"registry {stat}"] = [d, ref]
    for name, a, b in EPOCHS:
        m = ((e.year >= a) & (e.year <= b)).values
        ref, nq, na = get(700, f"lagrangian_rh_epoch_{a}_{b}", -72.0)
        d = contrast(e.inflow_rh, lo & m, hi & m)
        if not finite_close(d, ref, 1e-5) or (nq, na) != (int((lo & m).sum()), int((hi & m).sum())):
            refuse(f"epoch {name}: contrast {d:.6f} against the registry's {ref:.6f}")
        out[f"registry epoch {name}"] = [d, ref]
    resp, ante = cat_counts
    if not (np.array_equal(resp, e.response.values.astype(float)) and np.array_equal(resp, e.response_dep.values.astype(float))):
        refuse("the response recounted from the catalog differs from the deposited response")
    if not np.array_equal(ante, e.antecedent.values.astype(float)):
        refuse("the antecedent count recounted from the catalog differs from the deposited one")
    out["response and antecedent recounted from the catalog equal the deposit"] = int(resp.size)
    return out


def check_positions(e, catalog, pos, cat_counts):
    """The pulled rows bound to the catalog and to the deposit. Stops on a mismatch.

    The labelings from one position are only as good as the claim that the pulled rows are
    the catalog's own detections with a position attached. Three things are therefore
    required: the windows of the counts lie inside the box the rows were pulled for, the
    rows of every position equal the catalog's detections inside that box one for one in
    time and stored position, and they give the same response and antecedent count as the
    catalog for every eligible trough.
    """
    t, x, y = catalog
    out = {}
    reach = DLON + 2.0
    west, east = float(e.lon.min()) - reach, float(e.lon.max()) + reach
    if not (PULL_BOX[0] <= LAT_LO and LAT_HI <= PULL_BOX[1] and PULL_BOX[2] <= west and east <= PULL_BOX[3]):
        refuse("the counts reach outside the box the source rows were pulled for")
    out["positions: the windows of the counts lie inside the pulled box"] = [west, east]
    inside = (y >= PULL_BOX[0]) & (y <= PULL_BOX[1]) & (x >= PULL_BOX[2]) & (x <= PULL_BOX[3])
    a = [np.asarray(t)[inside].astype("datetime64[ns]").astype("int64"), x[inside], y[inside]]
    b = [pos["t"].astype("datetime64[ns]").astype("int64"), pos["x"], pos["y"]]
    ia, ib = np.lexsort((a[2], a[1], a[0])), np.lexsort((b[2], b[1], b[0]))
    if a[0].size != b[0].size or not all(np.array_equal(u[ia], v[ib]) for u, v in zip(a, b)):
        refuse("the pulled rows of every position do not equal the catalog's detections inside the pulled box")
    out["positions: the rows of every position equal the catalog inside the pulled box, one for one"] = int(a[0].size)
    resp, ante = counts(e, pos["t"], pos["x"], pos["y"])
    if not (np.array_equal(resp, cat_counts[0]) and np.array_equal(ante, cat_counts[1])):
        refuse("the counts from the pulled rows of every position differ from the counts from the catalog")
    out["positions: the rows of every position give the deposited response and antecedent count"] = int(resp.size)
    by = {k: int((pos["sat"] == k).sum()) for k in POSITIONS if (pos["sat"] == k).any()}
    if POSITION not in by:
        refuse(f"the pull holds no row of the {POSITION} position")
    out["positions: rows by position"] = by
    return out


def attach_cases(e, dep, level, lo, hi, get):
    """The classified-only measures of one level, aligned to the eligible rows and checked."""
    c = pd.read_csv(os.path.join(dep, f"cases_pooled_{level}.csv"))
    c["time"] = pd.to_datetime(c["time"])
    cols = [m[0] for m in classified_measures(level)]
    j = e[["wave", "time"]].merge(c[["traj_id", "time", "label"] + cols].rename(columns={"traj_id": "wave"}),
                                  on=["wave", "time"], how="left", validate="one_to_one")
    have = j.label.notna().values
    if not np.array_equal(have, lo | hi):
        refuse(f"the {level} hPa case table does not hold exactly the classified observations")
    expected = np.where(lo, "MCS-quiet", "MCS-active")
    if not np.array_equal(j.label.values[have], expected[have]):
        refuse(f"the {level} hPa case labels differ from the canonical labels")
    checks = {}
    for col, _, stat, hour in classified_measures(level):
        x = j[col].values
        ref, nq, na = get(level, stat, hour)
        d = contrast(x, lo, hi)
        n = (int((lo & np.isfinite(x)).sum()), int((hi & np.isfinite(x)).sum()))
        if not finite_close(d, ref, 1e-4) or n != (nq, na):
            refuse(f"{level} hPa {col}: contrast {d:.6f} on {n} against the registry's {ref:.6f} on {(nq, na)}")
        checks[f"{level} {col}"] = [d, ref]
    if level == 700 and np.nanmax(np.abs(j.rh_m72.values[have] - e.inflow_rh.values[have])) > 6e-5:
        refuse("the case table's -72 h humidity differs from the eligible table's")
    return j[cols], checks


def label_balance(e, lab, cell):
    """How evenly each labeling fills its own cells and the two eras. Added after the first run.

    Thresholds with boundary-equal values entering the outer class do not give equal
    classes when counts tie, so a labeling that sets its thresholds inside a cell does not
    balance the classes there exactly. This counts how far from balance each one is.
    """
    out = {}
    late = e.year.values >= ERA_YEAR
    for k, (a, b) in lab.items():
        d = pd.DataFrame(dict(cell=cell, lo=a, hi=b))
        g = d[d.lo | d.hi].groupby("cell").agg(lo=("lo", "sum"), hi=("hi", "sum"))
        share = g.hi / (g.hi + g.lo)
        out[k] = dict(cells_with_a_class=int(len(g)), cells_with_unequal_classes=int((g.lo != g.hi).sum()),
                      largest_count_difference=int((g.hi - g.lo).abs().max()),
                      active_share_lowest=float(share.min()), active_share_highest=float(share.max()),
                      active_share_before_1998=float(b[~late].sum() / max(1, (a | b)[~late].sum())),
                      active_share_from_1998=float(b[late].sum() / max(1, (a | b)[late].sum())))
    return out


def bind_outputs(rows, get, matched):
    """The finished rows against the registry. Stops on a mismatch, returns how many were bound.

    ``rows`` is the record's list of rows, ``get`` the registry reader, and ``matched`` the
    deposited year-matched contrast with its class sizes. Every contrast the registry also
    holds must come out of the split equal to it, on the same class sizes.
    """
    H, BOX, TCWV, _, ANTE = [name for _, name in ELIGIBLE]
    every, cls = "every eligible trough", "classified observations"
    want = {("whole sample", H, "canonical", every): (get(700, "lagrangian_rh", -72.0), 1e-5),
            ("whole sample", BOX, "canonical", every): (get(700, "eulerian_box", -24.0), 1e-5),
            ("whole sample", TCWV, "canonical", every): (get(None, "tcwv_box", -24.0), 1e-5),
            ("whole sample", ANTE, "canonical", every): (get(700, "antecedent_convection", -48.0), 1e-5),
            ("whole sample", H, "canonical", cls): (get(700, "lagrangian_rh", -72.0), 1e-5),
            ("whole sample", H, "year-matched", every): (tuple(matched), 1e-5)}
    for name, a, b in EPOCHS:
        want[(name, H, "canonical", every)] = (get(700, f"lagrangian_rh_epoch_{a}_{b}", -72.0), 1e-5)
    for level in (700, 850):
        for _, name, stat, hour in classified_measures(level):
            want.setdefault(("whole sample", name, "canonical", cls), (get(level, stat, hour), 1e-4))
    have, seen = {}, set()
    for r in rows:
        key = (r["sample"], r["measure"], r["labels"], r["background"])
        if key + (r["quantity"],) in seen:
            refuse(f"two rows for {key + (r['quantity'],)}")
        seen.add(key + (r["quantity"],))
        if r["quantity"] == "total":
            have[key] = r
    for key, ((ref, nq, na), tol) in want.items():
        if key not in have:
            refuse(f"no row for {key}")
        r = have[key]
        if not finite_close(r["diff"], ref, tol) or (r["n_quiet"], r["n_active"]) != (int(nq), int(na)):
            refuse(f"{key}: {r['diff']:.6f} on {(r['n_quiet'], r['n_active'])} against the registry's {ref:.6f} on {(int(nq), int(na))}")
    return len(want)


# --------------------------------------------------------------------------- the run
def build(dep, csct, n_boot, seed, check_only=False, expect=None, positions=None, alt_catalog=None, alt_label=ALT_LABEL):
    e = load_inputs(dep)
    if expect is None:
        expect = expected_sample(dep)
    years = np.arange(int(e.year.min()), int(e.year.max()) + 1)
    t, x, y = load_catalog(csct, years)
    resp0, ante0 = counts(e, t, x, y)

    lab = {}
    lab["canonical"] = stratified_terciles(e.response.values.astype(float), e.lon.values, EDGES, min_bin=MIN_BIN,
                                           strat2=e.month.values, edges2=M_EDGES)
    lab["year-matched"] = cell_terciles(e.response.values, [e.year.values, e.month.values, e.lonbin.values])
    era = (e.year.values >= ERA_YEAR).astype(float)
    lab["era"] = cell_terciles(e.response.values, [era, e.month.values, e.lonbin.values])
    checks = check_inputs(e, lab, dep, (resp0, ante0), expect=expect)
    pos = None
    if positions is not None:                        # the pulled rows are bound before anything is computed from them
        pos = load_positions(positions, years)
        checks.update(check_positions(e, (t, x, y), pos, (resp0, ante0)))
    alt = None
    if alt_catalog is not None:                      # the second catalog is bound before anything is computed from it
        if alt_label in ("canonical", "year-matched", "era") + ONE_POSITION or alt_label.startswith("merged"):
            refuse(f"the name {alt_label!r} belongs to another labeling")
        at, ax, ay = load_catalog(alt_catalog, years)
        ar, aa = counts(e, at, ax, ay)
        checks.update(check_alt(e, dep, ar, alt_label))
        alt = dict(response=ar, antecedent=aa, detections=int(at.size), names=alt_names(alt_label))
    get = registry(dep)
    lo, hi = lab["canonical"]
    cases = {}
    for level in (700, 850):
        cases[level], ck = attach_cases(e, dep, level, lo, hi, get)
        checks.update(ck)
    if check_only:                                   # the inputs are bound and nothing is computed
        return dict(checks=checks)

    merged = {}
    for deg in (MERGE_DEG, MERGE_DEG_WIDE):
        mt, mx, my, removed = merge_same_time(t, x, y, deg)
        r, a = counts(e, mt, mx, my)
        merged[deg] = dict(response=r, antecedent=a, removed=removed, detections=int(t.size))
        lab[f"merged {deg:g}"] = stratified_terciles(r, e.lon.values, EDGES, min_bin=MIN_BIN,
                                                     strat2=e.month.values, edges2=M_EDGES)

    one = None
    if pos is not None:                              # the counts and the classes from one position alone
        own = pos["sat"] == POSITION
        r, a = counts(e, pos["t"][own], pos["x"][own], pos["y"][own])
        one = dict(response=r, antecedent=a, rows=int(own.size), rows_of_position=int(own.sum()))
        lab[ONE_POSITION[0]] = stratified_terciles(r, e.lon.values, EDGES, min_bin=MIN_BIN,
                                                   strat2=e.month.values, edges2=M_EDGES)
        lab[ONE_POSITION[1]] = cell_terciles(r, [e.year.values, e.month.values, e.lonbin.values])
    if alt is not None:                              # the classes from the second catalog on the same troughs
        lab[alt["names"][0]] = stratified_terciles(alt["response"], e.lon.values, EDGES, min_bin=MIN_BIN,
                                                   strat2=e.month.values, edges2=M_EDGES)
        lab[alt["names"][1]] = cell_terciles(alt["response"], [e.year.values, e.month.values, e.lonbin.values])
    recounted = ([(ONE_POSITION, one)] if one is not None else []) + ([(alt["names"], alt)] if alt is not None else [])

    keep = np.isfinite(e.lonbin.values)                  # the three troughs at exactly 40 E lie in no cell
    e = e[keep].reset_index(drop=True)
    lab = {k: (a[keep], b[keep]) for k, (a, b) in lab.items()}
    cases = {lv: df[keep].reset_index(drop=True) for lv, df in cases.items()}
    for deg in merged:
        merged[deg]["response"] = merged[deg]["response"][keep]
        merged[deg]["antecedent"] = merged[deg]["antecedent"][keep]
    for _, rc in recounted:
        rc["response"], rc["antecedent"] = rc["response"][keep], rc["antecedent"][keep]
    cell = e.year.values * 1000 + e.month.values.astype(int) * 10 + e.lonbin.values.astype(int)
    everyone = np.ones(len(e), bool)
    classified = lab["canonical"][0] | lab["canonical"][1]

    def jobs_for(sample, idx, with_cases):
        """Every measure under every labeling for the observations ``idx`` of ``e``."""
        J = {}
        for col, name in ELIGIBLE:
            for L in ("canonical", "year-matched", "era"):
                J[(name, L, "every eligible trough")] = sample.measure(e[col].values[idx], lab[L][0][idx], lab[L][1][idx], everyone[idx])
            key = f"merged {MERGE_DEG:g}"
            src = merged[MERGE_DEG][col] if col in ("response", "antecedent") else e[col].values
            J[(name, key, "every eligible trough")] = sample.measure(src[idx], lab[key][0][idx], lab[key][1][idx], everyone[idx])
            for names, rc in recounted:                  # both counts are the ones recounted for that labeling
                for key in names:
                    own = rc[col] if col in ("response", "antecedent") else e[col].values
                    J[(name, key, "every eligible trough")] = sample.measure(own[idx], lab[key][0][idx], lab[key][1][idx], everyone[idx])
        key = f"merged {MERGE_DEG_WIDE:g}"
        J[(ELIGIBLE[0][1], key, "every eligible trough")] = sample.measure(e.inflow_rh.values[idx], lab[key][0][idx], lab[key][1][idx], everyone[idx])
        if with_cases:
            c_lo, c_hi = lab["canonical"][0][idx], lab["canonical"][1][idx]
            J[(ELIGIBLE[0][1], "canonical", "classified observations")] = sample.measure(
                e.inflow_rh.values[idx], c_lo, c_hi, classified[idx])
            for level in (700, 850):
                for col, name, _, _ in classified_measures(level):
                    if (name, "canonical", "classified observations") in J:
                        continue          # the 700 hPa value at -72 h is already there at full precision
                    xv = cases[level][col].values[idx]
                    J[(name, "canonical", "classified observations")] = sample.measure(xv, c_lo, c_hi, classified[idx] & np.isfinite(xv))
        return J

    H, BOX = ELIGIBLE[0][1], ELIGIBLE[1][1]
    record = dict(n_boot=int(n_boot), seed=int(seed), checks=checks, rows=[], paired=[], merge={}, labels={})
    for k, (a, b) in lab.items():
        record["labels"][k] = dict(quiet=int(a.sum()), active=int(b.sum()),
                                   year_gap=float(e.year.values[b].mean() - e.year.values[a].mean()))
    record["balance"] = label_balance(e, lab, cell)

    def emit(sample_name, rows, paired):
        for (measure, L, bg), r in rows.items():
            for q, v in r["quantities"].items():
                record["rows"].append(dict(sample=sample_name, measure=measure, labels=L, background=bg, quantity=q,
                                           **v, n_quiet=r["n_quiet"], n_active=r["n_active"], waves_quiet=r["waves_quiet"],
                                           waves_active=r["waves_active"], waves_resampled=r["waves_resampled"]))
        for (a, b), qs in paired.items():
            for q, v in qs.items():
                record["paired"].append(dict(sample=sample_name, first=list(a), second=list(b), quantity=q, **v))

    # the whole sample, with the paired comparisons declared for it
    idx = np.arange(len(e))
    s = Sample(e.wave.values, e.year.values, cell)
    J = jobs_for(s, idx, with_cases=True)
    pairs = [((H, L, "every eligible trough"), (BOX, L, "every eligible trough")) for L in ("canonical", "year-matched", f"merged {MERGE_DEG:g}")]
    pairs.append((("inflow RH at -72 h, 700 hPa", "canonical", "classified observations"),
                  ("inflow RH at -24 h, 700 hPa", "canonical", "classified observations")))
    pairs.append(((H, "canonical", "every eligible trough"), (H, "canonical", "classified observations")))
    for names, _ in recounted:
        pairs += [((H, L, "every eligible trough"), (BOX, L, "every eligible trough")) for L in names]
        pairs.append(((H, "canonical", "every eligible trough"), (H, names[0], "every eligible trough")))
    rows, _, paired = run_sample(s, "whole sample", J, n_boot, seed, pairs)
    emit("whole sample", rows, paired)

    # epochs and eras, each pair of disjoint samples compared on independent replicates
    for group, spans in (("epochs", EPOCHS), ("eras", ERAS)):
        held = {}
        for name, a, b in spans:
            idx = np.flatnonzero((e.year.values >= a) & (e.year.values <= b))
            s = Sample(e.wave.values[idx], e.year.values[idx], cell[idx])
            rows, reps, _ = run_sample(s, name, jobs_for(s, idx, with_cases=False), n_boot, seed)
            emit(name, rows, {})
            held[name] = (rows, reps)
        (n1, (r1, p1)), (n2, (r2, p2)) = list(held.items())
        for key in r1:
            if key[0] != H:
                continue
            for q in QUANTITIES:
                v = both(r1[key]["quantities"][q]["diff"] - r2[key]["quantities"][q]["diff"],
                         p1["recomputed"][key][q] - p2["recomputed"][key][q], p1["held"][key][q] - p2["held"][key][q])
                record["paired"].append(dict(sample=f"{n1} less {n2}", first=list(key), second=list(key), quantity=q, **v))

    # longitude sectors, every mean recomputed inside the sector
    for name, a, b in SECTORS:
        idx = np.flatnonzero((e.lon.values >= a) & (e.lon.values < b))
        s = Sample(e.wave.values[idx], e.year.values[idx], cell[idx])
        rows, _, _ = run_sample(s, name, jobs_for(s, idx, with_cases=False), n_boot, seed)
        emit(name, rows, {})

    # what the merge did to the counts
    era_late = e.year.values >= ERA_YEAR
    for deg, mg in merged.items():
        byb = {}
        for bi in range(7):
            m = e.lonbin.values == bi
            byb[f"{int(EDGES[bi])} to {int(EDGES[bi + 1])}"] = dict(
                before=float(mg["response"][m & ~era_late].mean()), after=float(mg["response"][m & era_late].mean()),
                unmerged_before=float(e.response.values[m & ~era_late].mean()), unmerged_after=float(e.response.values[m & era_late].mean()))
        lo_m, hi_m = lab[f"merged {deg:g}"]
        share = {}
        for sname, a, b in SECTORS:
            m = (e.lon.values >= a) & (e.lon.values < b)
            for ename, late in (("1983 to 1997", False), ("1998 to 2007", True)):
                mm = m & (era_late == late)
                share[f"{sname}, {ename}"] = dict(merged=float(hi_m[mm].sum() / max(1, (hi_m[mm] | lo_m[mm]).sum())),
                                                  canonical=float(lab["canonical"][1][mm].sum() / max(1, (classified[mm]).sum())))
        record["merge"][f"{deg:g}"] = dict(detections=mg["detections"], removed=mg["removed"], response_by_bin=byb, active_share=share,
                                           agree_with_canonical=float(((lo_m == lab["canonical"][0]) & (hi_m == lab["canonical"][1])).mean()))
    # what the recounted labelings did to the counts and the classes, beside the deposit's own
    def recount_summary(names, rc):
        byb = {}
        for bi in range(7):
            m = e.lonbin.values == bi
            row = {}
            for col in ("response", "antecedent"):
                row[col] = dict(before=float(rc[col][m & ~era_late].mean()), after=float(rc[col][m & era_late].mean()),
                                pooled_before=float(e[col].values[m & ~era_late].mean()), pooled_after=float(e[col].values[m & era_late].mean()))
            byb[f"{int(EDGES[bi])} to {int(EDGES[bi + 1])}"] = row
        share = {}
        for sname, a, b in SECTORS:
            m = (e.lon.values >= a) & (e.lon.values < b)
            for ename, late in (("1983 to 1997", False), ("1998 to 2007", True)):
                mm = m & (era_late == late)
                share[f"{sname}, {ename}"] = {k: float(lab[k][1][mm].sum() / max(1, (lab[k][0][mm] | lab[k][1][mm]).sum()))
                                               for k in ("canonical",) + tuple(names)}
        cls = ("MCS-quiet", "neither class", "MCS-active")
        was = np.where(lab["canonical"][0], 0, np.where(lab["canonical"][1], 2, 1))
        now = np.where(lab[names[0]][0], 0, np.where(lab[names[0]][1], 2, 1))
        return dict(by_bin=byb, active_share=share,
                    crosstab={f"{cls[i]} to {cls[k]}": int(((was == i) & (now == k)).sum()) for i in range(3) for k in range(3)},
                    agree_with_canonical=float((was == now).mean()))

    if one is not None:
        record["position"] = dict(position=POSITION, smallest_radius_km=MIN_RADIUS_KM, pull_sha256=sha256(positions), rows=one["rows"],
                                  rows_of_position=one["rows_of_position"], **recount_summary(ONE_POSITION, one))
    if alt is not None:
        record["alt_catalog"] = dict(label=alt_label, sha256=sha256(alt_catalog), detections=alt["detections"],
                                     catalog_sha256=sha256(csct), **recount_summary(alt["names"], alt))
    ys = pd.read_csv(os.path.join(dep, "yearstrat_sensitivity.csv")).set_index("split")
    matched = (float(ys["diff"]["yearmonthlon"]), int(ys.n_quiet["yearmonthlon"]), int(ys.n_active["yearmonthlon"]))
    record["checks"]["totals equal to the registry after the run"] = bind_outputs(record["rows"], get, matched)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposit", default="deposit")
    ap.add_argument("--csct", default="data/original/csct/csct_africa_cs245.nc")
    ap.add_argument("--outdir", default="deposit")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--positions", default=None,
                    help="the pulled source rows with their satellite position; adds the labelings from one position")
    ap.add_argument("--alt-catalog", default=None,
                    help="a second catalog; adds two labelings from its counts on the same troughs")
    ap.add_argument("--alt-label", default=ALT_LABEL, help="the name of the labelings from the second catalog")
    ap.add_argument("--expect", type=int, nargs=2, default=None, metavar=("TROUGHS", "WAVES"),
                    help="the eligible troughs and waves the checks require; read from the deposit when absent")
    ap.add_argument("--check-only", action="store_true",
                    help="bind the inputs to the deposit and the registry, compute nothing, write nothing")
    a = ap.parse_args()
    more = dict(positions=a.positions, alt_catalog=a.alt_catalog, alt_label=a.alt_label,
                expect=tuple(a.expect) if a.expect else None)
    if a.check_only:
        checks = build(a.deposit, a.csct, a.n_boot, a.seed, check_only=True, **more)["checks"]
        print(f"{len(checks)} checks passed, nothing computed")
        return
    record = build(a.deposit, a.csct, a.n_boot, a.seed, **more)
    payload = json.dumps(record, indent=1, allow_nan=False)      # stops here if any number is missing
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, "scale_split_record.json")
    with open(out, "w") as f:
        f.write(payload)
    flat = pd.DataFrame(record["rows"])
    flat.to_csv(os.path.join(a.outdir, "scale_split.csv"), index=False, float_format="%.6f")
    print(f"wrote {out}: {len(record['rows'])} rows, {len(record['paired'])} paired rows, {record['n_boot']} replicates")


if __name__ == "__main__":
    main()
