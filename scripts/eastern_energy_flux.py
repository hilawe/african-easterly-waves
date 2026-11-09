"""The energy flux of the regression composite at the published budget base of Diaz and Aiyyer
(2013), declared in `EASTERN_ENERGY_FLUX_2026-11-04.md`.

The ERA5 fields of `scripts/download_era5_flux_fields.py` are band-passed with the reference,
the 2 to 10 day 700 hPa meridional wind at 10 N, 15 W, and regressed on it at lags of -4 to +4
days. From the composites come the terms of the paper's eddy kinetic energy budget (section 4
of the record), three reproduction products (section 5) and the decision reading, the energy
velocity of the composite over a box one wavelength wide around the base (section 6), with
intervals from whole-season resamples. Before any ERA5 field is read, the energy velocity is
scored on six shallow-water wave packets of known motion (section 7), and the reading refuses
unless that scoring is adequate and the gates of section 8 pass.

Modes: `constructions` writes the packet record, `read` the ERA5 record.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eastern_phase_envelope as PE  # noqa: E402

with open(os.path.abspath(__file__), "rb") as _fh:
    SCRIPT_SHA256_AT_LOAD = hashlib.sha256(_fh.read()).hexdigest()

A = 6.371e6
OMEGA = 7.292e-5
G = 9.80665
RD = 287.04
F0 = 2.0 * OMEGA * np.sin(np.deg2rad(10.0))
M_PER_DEG_DAY = A * np.cos(np.deg2rad(10.0)) * np.pi / 180.0 / 86400.0   # 1 deg/day at 10 N in m/s
SLOW = PE.SLOW

BASE_LON, BASE_LAT = -15.0, 10.0
BOX_LONS = (-30.0, 0.0)
BOX_LATS = (6.0, 14.0)
NORTH_LATS = (15.0, 20.0)
WINDOW_STEPS = 7                       # lags of -42 to +42 hours at 6 hours


# ------------------------------------------------------------------ the estimator

def ddlat(X, lats, axis):
    """d(X)/dy = (1/a) d(X)/d(theta), centered inside and one-sided at the edges."""
    return np.gradient(X, np.deg2rad(lats) * A, axis=axis)


def energy_velocity(u, v, phi, U, lats, lat_range=BOX_LATS, f0=F0, weights_p=None, ape=None):
    """Section 6. The flux part, the advective part and their sum, in m/s, from fields shaped
    (..., latitude, longitude) restricted to the box's longitudes and lags, with the latitudes
    `lats` reaching at least one row beyond `lat_range` on each side. `U` is the mean zonal wind
    broadcast to the same shape. Sums run over the box's latitudes weighted by their cosine, over
    every other axis equally or, for a level axis at position 1, by `weights_p`. When `ape` is
    given, the eddy available potential energy is added to K in the normalization (development
    change 1, section 10)."""
    lats = np.asarray(lats, float)
    ua = u + ddlat(phi, lats, axis=-2) / f0
    rows = (lats >= lat_range[0] - 1e-9) & (lats <= lat_range[1] + 1e-9)
    if rows[0] or rows[-1]:
        raise SystemExit("REFUSED: the latitudes do not reach beyond the box for the derivative")
    w = np.cos(np.deg2rad(lats))[rows][:, None]
    if weights_p is not None:
        shape = [1] * u.ndim
        shape[1] = len(weights_p)
        w = w * np.asarray(weights_p, float).reshape(shape)

    def total(X):
        return float((X[..., rows, :] * w).sum())
    K = 0.5 * (u ** 2 + v ** 2)
    k = total(K if ape is None else K + ape)
    c_rel, c_adv = total(ua * phi) / k, total(U * K) / k
    return {"c_rel": c_rel, "c_adv": c_adv, "c_E": c_rel + c_adv}


def in_deg_per_day(r):
    return {key: val / M_PER_DEG_DAY for key, val in r.items()}


def point_class(c):
    """The class of a point speed alone against the 1 deg/day line (section 7)."""
    return "eastward" if c >= SLOW else ("westward" if c <= -SLOW else "standing")


# ------------------------------------------------------------------ the shallow-water packets

SW_LONS = np.arange(-90.0, 90.0, 1.0)
SW_LATS = np.arange(0.0, 25.0 + 0.5, 1.0)
SW_DX = A * np.cos(np.deg2rad(10.0)) * np.pi / 180.0
SW_DY = A * np.pi / 180.0
SW_DT = 300.0
SW_DAYS = 8.0
SW_SAMPLE = 6 * 3600.0
CONSTRUCTIONS = {
    "U-8 L30 H300": {"U": -8.0, "wavelength": 30.0, "H": 300.0},
    "U-2 L30 H300": {"U": -2.0, "wavelength": 30.0, "H": 300.0},
    "U+2 L30 H300": {"U": 2.0, "wavelength": 30.0, "H": 300.0},
    "U-8 L20 H300": {"U": -8.0, "wavelength": 20.0, "H": 300.0},
    "U+0 L20 H300": {"U": 0.0, "wavelength": 20.0, "H": 300.0},
    "U-8 L30 H5000": {"U": -8.0, "wavelength": 30.0, "H": 5000.0},
}
VERSIONS = {0: "as declared", 1: "after change 1", 2: "after changes 1 and 2"}
FULL_LATS = (1.0, 24.0)               # change 2, every row whose centered derivative exists
ENVELOPE_DEG = 10.0
ARRIVAL_DAYS = 3.0
TRUTH_DAYS = (1.0, 7.0)


def coriolis(lats=SW_LATS):
    return 2.0 * OMEGA * np.sin(np.deg2rad(lats))


def theory_group_speed(U, wavelength, H, lats=SW_LATS):
    """The group speed in deg/day of the gravest channel mode by shallow-water Rossby theory at
    the channel's middle, used only to place the packet so it reaches the base mid-run."""
    lat0 = 0.5 * (lats[0] + lats[-1])
    f, beta = 2 * OMEGA * np.sin(np.deg2rad(lat0)), 2 * OMEGA * np.cos(np.deg2rad(lat0)) / A
    F = f ** 2 / (G * H)
    beta_eff = beta + F * U
    l = np.pi / ((lats[-1] - lats[0]) * SW_DY)

    def omega(k):
        return U * k - beta_eff * k / (k ** 2 + l ** 2 + F)
    k = 2 * np.pi / (wavelength * SW_DX)
    return (omega(k * 1.001) - omega(k * 0.999)) / (0.002 * k) / M_PER_DEG_DAY


def _ddx(X):
    k = 2j * np.pi * np.fft.rfftfreq(X.shape[-1], d=SW_DX)
    return np.fft.irfft(k * np.fft.rfft(X, axis=-1), n=X.shape[-1], axis=-1)


def _ddy(X):
    return np.gradient(X, SW_DY, axis=0, edge_order=2)


def balanced_state(psi, f):
    """u, v and the geopotential in linear balance with the streamfunction `psi`, the
    geopotential solving lap(phi) = div(f grad psi) with d(phi)/dy = f d(psi)/dy at the walls,
    one Fourier mode in longitude at a time."""
    u, v = -_ddy(psi), _ddx(psi)
    rhs = _ddx(f[:, None] * _ddx(psi)) + _ddy(f[:, None] * _ddy(psi))
    ny, nx = psi.shape
    R = np.fft.rfft(rhs, axis=-1)
    bc = np.fft.rfft(f[:, None] * _ddy(psi), axis=-1)
    k = 2 * np.pi * np.fft.rfftfreq(nx, d=SW_DX)
    P = np.zeros_like(R)
    for m in range(R.shape[1]):
        M = np.zeros((ny, ny), complex)
        b = R[:, m].copy()
        for j in range(ny):
            M[j, j] = -2.0 / SW_DY ** 2 - k[m] ** 2
            if 0 < j < ny - 1:
                M[j, j - 1] = M[j, j + 1] = 1.0 / SW_DY ** 2
        M[0, 1] = M[-1, -2] = 2.0 / SW_DY ** 2
        b[0] += 2.0 * bc[0, m] / SW_DY
        b[-1] -= 2.0 * bc[-1, m] / SW_DY
        P[:, m] = np.linalg.lstsq(M, b, rcond=None)[0]
    phi = np.fft.irfft(P, n=nx, axis=-1)
    return u, v, phi - phi.mean()


def tendencies(u, v, phi, U, f, phibar):
    du = -U * _ddx(u) + f[:, None] * v - _ddx(phi)
    dv = -U * _ddx(v) - f[:, None] * u - _ddy(phi)
    dv[0] = dv[-1] = 0.0
    dphi = -U * _ddx(phi) + f[:, None] * U * v - phibar[:, None] * (_ddx(u) + _ddy(v))
    return du, dv, dphi


def run_packet(U, wavelength, H, x0, days=SW_DAYS, dt=SW_DT):
    """Integrate the linear shallow-water equations about a uniform zonal flow U in geostrophic
    balance from a balanced packet centered at x0, sampled every 6 hours, as (times in days,
    u, v, phi) with arrays shaped (time, latitude, longitude)."""
    f = coriolis()
    phibar = G * H - np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * U * SW_DY)])
    x = SW_LONS[None, :] - x0
    env = np.exp(-x ** 2 / (2 * ENVELOPE_DEG ** 2)) * np.cos(2 * np.pi * x / wavelength)
    psi = 1e6 * env * np.sin(np.pi * (SW_LATS[:, None] - SW_LATS[0]) / (SW_LATS[-1] - SW_LATS[0]))
    u, v, phi = balanced_state(psi, f)
    v[0] = v[-1] = 0.0
    every = int(round(SW_SAMPLE / dt))
    out_t, out = [0.0], [(u.copy(), v.copy(), phi.copy())]
    for n in range(1, int(round(days * 86400 / dt)) + 1):
        k1 = tendencies(u, v, phi, U, f, phibar)
        k2 = tendencies(*(a + 0.5 * dt * b for a, b in zip((u, v, phi), k1)), U, f, phibar)
        k3 = tendencies(*(a + 0.5 * dt * b for a, b in zip((u, v, phi), k2)), U, f, phibar)
        k4 = tendencies(*(a + dt * b for a, b in zip((u, v, phi), k3)), U, f, phibar)
        u, v, phi = (a + dt / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4) for a, b1, b2, b3, b4 in zip((u, v, phi), k1, k2, k3, k4))
        if n % every == 0:
            out_t.append(n * dt / 86400.0)
            out.append((u.copy(), v.copy(), phi.copy()))
    t = np.array(out_t)
    return t, np.array([o[0] for o in out]), np.array([o[1] for o in out]), np.array([o[2] for o in out]), phibar


def energy_centroid(u, v, phi, phibar, lons=SW_LONS):
    """The longitude of the total-energy centroid at each time, a circular mean over the
    periodic channel unwrapped in time."""
    e = 0.5 * (phibar[None, :, None] * (u ** 2 + v ** 2) + phi ** 2)
    length = lons[-1] - lons[0] + (lons[1] - lons[0])
    z = (e * np.exp(2j * np.pi * (lons - lons[0]) / length)[None, None, :]).sum(axis=(1, 2))
    ang = np.unwrap(np.angle(z))
    return lons[0] + ang * length / (2 * np.pi)


def construction_reading(name, spec):
    """One construction (section 7): the true speed from the energy centroid, and the
    estimator's speeds from the box around the base and the window around the time the
    meridional wind at the base is largest, all in deg/day."""
    U, lam, H = spec["U"], spec["wavelength"], spec["H"]
    x0 = BASE_LON - theory_group_speed(U, lam, H) * ARRIVAL_DAYS
    t, u, v, phi, phibar = run_packet(U, lam, H, x0)
    xc = energy_centroid(u, v, phi, phibar)
    sel = (t >= TRUTH_DAYS[0] - 1e-9) & (t <= TRUTH_DAYS[1] + 1e-9)
    truth = float(np.polyfit(t[sel], xc[sel], 1)[0])
    ib, jb = int(np.flatnonzero(SW_LATS == BASE_LAT)[0]), int(np.flatnonzero(SW_LONS == BASE_LON)[0])
    ok = np.arange(len(t))[WINDOW_STEPS:len(t) - WINDOW_STEPS]
    i0 = int(ok[np.argmax(np.abs(v[ok, ib, jb]))])
    win = slice(i0 - WINDOW_STEPS, i0 + WINDOW_STEPS + 1)
    cols = (SW_LONS >= BOX_LONS[0] - 1e-9) & (SW_LONS <= BOX_LONS[1] + 1e-9)
    w_u, w_v, w_phi = u[win][..., cols], v[win][..., cols], phi[win][..., cols]
    ape = w_phi ** 2 / (2.0 * G * H)
    est = {VERSIONS[0]: in_deg_per_day(energy_velocity(w_u, w_v, w_phi, U, SW_LATS)),
           VERSIONS[1]: in_deg_per_day(energy_velocity(w_u, w_v, w_phi, U, SW_LATS, ape=ape)),
           VERSIONS[2]: in_deg_per_day(energy_velocity(w_u, w_v, w_phi, U, SW_LATS, lat_range=FULL_LATS, ape=ape))}
    return {"spec": spec, "x0": float(x0), "theory": float(theory_group_speed(U, lam, H)), "truth": truth,
            "lag0_day": float(t[i0]), "versions": est,
            "energy_drift": float(np.sum(0.5 * (phibar[None, :, None] * (u[-1:] ** 2 + v[-1:] ** 2) + phi[-1:] ** 2)) /
                                  np.sum(0.5 * (phibar[None, :, None] * (u[:1] ** 2 + v[:1] ** 2) + phi[:1] ** 2)) - 1.0)}


def adequacy(readings, version=VERSIONS[0], tol=SLOW):
    """Section 7. Every construction gives c_E within `tol` deg/day of its true speed and the
    same point class, in the estimator's `version`."""
    fails = []
    for name, r in readings.items():
        c = r["versions"][version]["c_E"]
        if abs(c - r["truth"]) > tol:
            fails.append(f"{name} reads {c:+.2f} against {r['truth']:+.2f}")
        if point_class(c) != point_class(r["truth"]):
            fails.append(f"{name} reads {point_class(c)}, true {point_class(r['truth'])}")
    return {"adequate": not fails, "failures": fails}


def run_constructions(names=tuple(CONSTRUCTIONS)):
    return {name: construction_reading(name, CONSTRUCTIONS[name]) for name in names}


# ------------------------------------------------------------------ the ERA5 composites

import download_era5_flux_fields as D  # noqa: E402
import eastern_lag_regression as LR  # noqa: E402
import eastern_phase_envelope_matched as M  # noqa: E402
import eastern_phase_envelope_maxima as MX  # noqa: E402
import eastern_phase_envelope_probes as PR  # noqa: E402

FROZEN = {"eastern_phase_envelope": (PE, "49469e14779b69b27bf6861c0e652f34f754ceb09470f3816202d10dd83f9b6a"),
          "eastern_lag_regression": (LR, "22d4d4ec22a235be9a28c2b0d8facc33118c4f37cd0dfa062c04eef7a2d2f4c5"),
          "eastern_phase_envelope_matched": (M, "a2c5f8cdd25c72b16af44738d3b56a3de85c2a3b0e0b719ba30893e41bb3c006"),
          "eastern_phase_envelope_probes": (PR, "4c09a8c191cbf0336da7b0d491e0ed329f1c7e91629525e5f40281459228bec2"),
          "eastern_phase_envelope_maxima": (MX, "86c1e050c4c3524586e27cb93f2179d7ef2b414a2908f88137c3352cd635f98c"),
          "download_era5_flux_fields": (None, "6cd966a9dad85ba8022514df851893933c89b05069ef5c72be714692ea2e17a0")}
VARS = ("u", "v", "z", "t", "w")
LEVELS = np.array(D.LEVELS, float)                  # 950 to 350 hPa
P_PA = LEVELS * 100.0
GRID_LATS = np.arange(D.AREA[2], D.AREA[0] + 0.5, D.GRID)
GRID_LONS = np.arange(D.AREA[1], D.AREA[3] + 0.5, D.GRID)
LAGS = np.arange(-4 * PE.OBS_PER_DAY, 4 * PE.OBS_PER_DAY + 1)
WINDOW = np.abs(LAGS) <= WINDOW_STEPS
ERA5_FULL_LATS = (-4.0, 29.0)                        # change 2 on ERA5
MAP_LONS = (-45.0, 0.0)
BOX_COLS = (GRID_LONS >= BOX_LONS[0] - 1e-9) & (GRID_LONS <= BOX_LONS[1] + 1e-9)
KAPPA = 2.0 / 7.0
GATE4_MIN_R = 0.98
BLOCK_COLUMNS = 512                                  # columns per call of the frozen filter


def check_frozen():
    """Gate 2 of section 8."""
    for name, (mod, want) in FROZEN.items():
        got = mod.SCRIPT_SHA256_AT_LOAD if mod is not None else PE.sha256_of(os.path.join(HERE, name + ".py"))
        if got != want:
            raise SystemExit(f"REFUSED: {name} has the digest {got}, not the recorded {want}")
    return True


def pressure_weights(levels_hpa=None):
    """Trapezoid weights in pressure over the levels, summing to one (the vertical mean of
    section 4)."""
    p = np.asarray(LEVELS if levels_hpa is None else levels_hpa, float)
    w = np.zeros_like(p)
    dp = np.abs(np.diff(p))
    w[:-1] += 0.5 * dp
    w[1:] += 0.5 * dp
    return w / w.sum()


def load(var, year, flux_dir=D.OUT_DIR):
    """One season of one variable as (time, level, latitude, longitude) with levels from 950 to
    350 hPa and latitudes and longitudes ascending, after the downloader's check (gate 1)."""
    import xarray as xr
    if year == 2005:
        raise SystemExit("REFUSED: the 2005 season stays unopened")
    path = D.path(var, year, flux_dir)
    D.check(path, var, year)
    with xr.open_dataset(path) as ds:
        da = ds[var].sortby("valid_time").sortby("latitude").sortby("longitude")
        da = da.sel(pressure_level=list(LEVELS))
        if not (np.allclose(da["latitude"].values, GRID_LATS) and np.allclose(da["longitude"].values, GRID_LONS)):
            raise SystemExit(f"REFUSED: {path} is not on the declared grid")
        return np.asarray(da.transpose("valid_time", "pressure_level", "latitude", "longitude").values, float)


def lag_sums(X, r, lags=LAGS):
    """MX.real_sums at the lags given."""
    T, nx = X.shape
    nl = len(lags)
    n = np.zeros(nl); sa = np.zeros(nl); saa = np.zeros(nl); sb = np.zeros((nl, nx)); sab = np.zeros((nl, nx))
    for li, L in enumerate(lags):
        L = int(L)
        t0, t1 = max(0, -L), min(T, T - L)
        a = r[t0:t1]; B = X[t0 + L:t1 + L]
        n[li] = t1 - t0; sa[li] = a.sum(); saa[li] = float((a * a).sum())
        sb[li] = B.sum(axis=0); sab[li] = (B * a[:, None]).sum(axis=0)
    return {"n": n, "sa": sa, "saa": saa, "sb": sb, "sab": sab}


def reference_series(year, flux_dir=D.OUT_DIR):
    """The unfiltered 700 hPa meridional wind at 10 N, 15 W from the new files (section 3)."""
    V = load("v", year, flux_dir)
    k = int(np.flatnonzero(LEVELS == 700.0)[0])
    i = int(np.flatnonzero(np.isclose(GRID_LATS, BASE_LAT))[0]); j = int(np.flatnonzero(np.isclose(GRID_LONS, BASE_LON))[0])
    return V[:, k, i, j]


def filtered_block(flat, r, batch=BLOCK_COLUMNS):
    """MX.block over `batch` columns at a time, which bounds the filter's temporary to a batch
    instead of the whole grid. Each call filters its columns with the same reference, and every
    call must return the same reference block, so the result equals one call over all columns."""
    parts, rf = [], None
    for c0 in range(0, flat.shape[1], batch):
        F, rr = MX.block(flat[:, c0:c0 + batch], r)
        if rf is None:
            rf = rr
        elif rr.shape != rf.shape or not np.array_equal(rr, rf):
            raise SystemExit("REFUSED: the column batches gave different reference blocks")
        parts.append(F)
    return np.concatenate(parts, axis=1), rf


def season_sums(year, flux_dir=D.OUT_DIR, variables=VARS):
    """Every variable's lag sums against the reference over the whole grid, the same sums kept
    for the box's longitudes and the window's lags, and the season's unfiltered sums for the
    mean flow and the mean temperature."""
    r = reference_series(year, flux_dir)
    shape = (len(LEVELS), len(GRID_LATS), len(GRID_LONS))
    out = {}
    for var in variables:
        X = load(var, year, flux_dir)
        if X.shape[1:] != shape or X.shape[0] != len(r):
            raise SystemExit(f"REFUSED: {var} {year} has the shape {X.shape}")
        flat = X.reshape(X.shape[0], -1)
        F, rf = filtered_block(flat, r)
        s = lag_sums(F, rf)
        box = {k: (v[WINDOW].reshape((-1,) + shape)[..., BOX_COLS].reshape(int(WINDOW.sum()), -1) if k in ("sb", "sab") else v[WINDOW])
               for k, v in s.items()}
        out[var] = {"all": s, "box": box, "mean_sum": X.sum(axis=0), "mean_n": X.shape[0]}
    return out


def composite(st, w=None, shape=None):
    """MX.regression on stacked per-season sums, reshaped to (lag,) + `shape`."""
    R = MX.regression(st, w)
    return R if shape is None else R.reshape((R.shape[0],) + tuple(shape))


# ------------------------------------------------------------------ the terms

def ddlon(X, lats, lons):
    """d(X)/dx = (1 / (a cos theta)) d(X)/d(lambda) over the last two axes (latitude, longitude)."""
    return np.gradient(X, np.deg2rad(lons) * A, axis=-1) / np.cos(np.deg2rad(lats))[:, None]


def divergence(fx, fy, lats, lons):
    """Horizontal divergence on the sphere of a field shaped (..., latitude, longitude)."""
    c = np.cos(np.deg2rad(lats))[:, None]
    return ddlon(fx, lats, lons) + np.gradient(fy * c, np.deg2rad(lats) * A, axis=-2) / c


def budget_terms(cmp, mean, lats=None, lons=None, levels=None):
    """Section 4 from composites `cmp` (u, v, z, t, w, each lag by level by latitude by
    longitude) and mean fields `mean` (u, v, level by latitude by longitude)."""
    lats, lons, levels = (GRID_LATS if lats is None else lats), (GRID_LONS if lons is None else lons), (LEVELS if levels is None else levels)
    u, v, phi, T, om = cmp["u"], cmp["v"], cmp["z"], cmp["t"], cmp["w"]
    U, V = mean["u"][None], mean["v"][None]
    ua = u + ddlat(phi, lats, axis=-2) / F0
    va = v - ddlon(phi, lats, lons) / F0
    alpha = RD * T / (np.asarray(levels, float) * 100.0)[None, :, None, None]
    out = {"K": 0.5 * (u ** 2 + v ** 2), "flux_x": ua * phi, "flux_y": va * phi,
           "gfc": -divergence(u * phi, v * phi, lats, lons), "baroclinic": -om * alpha,
           "barotropic": -(u ** 2 * ddlon(U, lats, lons) + u * v * (ddlat(U, lats, axis=-2) + ddlon(V, lats, lons)) + v ** 2 * ddlat(V, lats, axis=-2))}
    out["total"] = out["gfc"] + out["baroclinic"] + out["barotropic"]
    return out


def vertical_mean(X, levels=None, axis=1):
    w = pressure_weights(LEVELS if levels is None else levels)
    return np.tensordot(X, w, axes=([axis], [0]))


def band_mean(X, lats, band):
    """Cosine-weighted mean over the latitudes of `band` (inclusive), latitude on axis -2."""
    rows = (lats >= band[0] - 1e-9) & (lats <= band[1] + 1e-9)
    w = np.cos(np.deg2rad(lats[rows]))
    return np.tensordot(X[..., rows, :], w / w.sum(), axes=([-2], [0]))


def static_stability(Tbar, levels):
    """sigma = -(R_d Tbar / p) d(ln thetabar)/dp at each level, from the box-mean temperature."""
    p = np.asarray(levels, float) * 100.0
    theta = Tbar * (1.0e5 / p) ** KAPPA
    return -(RD * Tbar / p) * np.gradient(np.log(theta), p)


def eddy_ape(T, sigma, levels):
    """(R_d T / p)^2 / (2 sigma), the eddy available potential energy of change 1, with the level
    on axis 1."""
    p = (np.asarray(levels, float) * 100.0)[None, :, None, None]
    return (RD * T / p) ** 2 / (2.0 * sigma[None, :, None, None])


def box_reading(cmp, mean, lats=None, lat_range=ERA5_FULL_LATS, bottom_hpa=None):
    """Section 6 after changes 1 and 2, and the exact form and departures of section 11, from box
    composites (u, v, z, t: window lag by level by latitude by box longitude) and the box's
    mean zonal wind and temperature (level by latitude by box longitude), in deg/day. With
    `bottom_hpa` the vertical mean starts at that level (the sensitivity of section 4)."""
    lats = GRID_LATS if lats is None else lats
    keep = np.ones(len(LEVELS), bool) if bottom_hpa is None else LEVELS <= bottom_hpa + 1e-9
    levels = LEVELS[keep]
    cmp = {k: x[:, keep] for k, x in cmp.items()}
    mean = {k: x[keep] for k, x in mean.items()}
    u, v, phi, T = cmp["u"], cmp["v"], cmp["z"], cmp["t"]
    rows = (lats >= lat_range[0] - 1e-9) & (lats <= lat_range[1] + 1e-9)
    wt = np.cos(np.deg2rad(lats[rows]))
    Tbar = np.tensordot(mean["t"][:, rows, :].mean(axis=-1), wt / wt.sum(), axes=([1], [0]))
    sigma = static_stability(Tbar, levels)
    if not (sigma > 0).all():
        raise SystemExit("REFUSED: the box-mean static stability is not positive at every level")
    ape = eddy_ape(T, sigma, levels)
    U = np.broadcast_to(mean["u"][None], u.shape)
    wp = pressure_weights(levels)
    v2 = in_deg_per_day(energy_velocity(u, v, phi, U, lats, lat_range=lat_range, weights_p=wp, ape=ape))
    w = np.cos(np.deg2rad(lats))[rows][:, None] * wp.reshape(1, -1, 1, 1)

    def total(X):
        return float((X[..., rows, :] * w).sum())
    K = 0.5 * (u ** 2 + v ** 2)
    e = total(K + ape)
    exact = (total(U * (K + ape)) + total(u * phi)) / e / M_PER_DEG_DAY
    return {**v2, "exact": exact, "ape_share": total(ape) / e,
            "edge_term": total(ddlat(phi, lats, axis=-2) / F0 * phi) / e / M_PER_DEG_DAY,
            "advective_shortfall": total(U * ape) / e / M_PER_DEG_DAY}


# ------------------------------------------------------------------ the reading

def add_sums(total, s):
    return s if total is None else {k: total[k] + s[k] for k in s}


def accumulate(years, flux_dir=D.OUT_DIR, variables=VARS):
    """The pooled sums over the whole grid, the per-season box sums and the per-season mean sums
    of every variable over `years`, one season and variable at a time."""
    if 2005 in years:
        raise SystemExit("REFUSED: the 2005 season stays unopened")
    totals, box, means = {v: None for v in variables}, {v: [] for v in variables}, {v: [] for v in variables}
    for y in years:
        ss = season_sums(y, flux_dir, variables)
        for var in variables:
            totals[var] = add_sums(totals[var], ss[var]["all"])
            box[var].append(ss[var]["box"])
            means[var].append((ss[var]["mean_sum"], ss[var]["mean_n"]))
    return totals, {v: PE.stack(b) for v, b in box.items()}, means


def weighted_mean(means, w=None):
    w = np.ones(len(means)) if w is None else np.asarray(w, float)
    return sum(wi * m for wi, (m, _) in zip(w, means)) / sum(wi * n for wi, (_, n) in zip(w, means))


def box_inputs(box, means, w=None):
    shape = (len(LEVELS), len(GRID_LATS), int(BOX_COLS.sum()))
    cmp = {v: composite(box[v], w, shape) for v in ("u", "v", "z", "t")}
    mean = {v: weighted_mean(means[v], w)[..., BOX_COLS] for v in ("u", "t")}
    return cmp, mean


def decision(box, means, n_boot=PE.N_BOOT, bottom_hpa=None, lat_range=ERA5_FULL_LATS):
    """The point reading of section 6 after changes 1 and 2 with the exact form beside it, their
    intervals from whole-season resamples and their classes, with the latitudes and lower
    boundary they were read over."""
    point = box_reading(*box_inputs(box, means), lat_range=lat_range, bottom_hpa=bottom_hpa)
    W = PE.boot_weights(len(means["u"]), n_boot)
    draws = [box_reading(*box_inputs(box, means, w), lat_range=lat_range, bottom_hpa=bottom_hpa) for w in W]
    out = {"point": point, "lat_range": [float(x) for x in lat_range],
           "bottom_hpa": float(LEVELS.max() if bottom_hpa is None else bottom_hpa)}
    for key in ("c_E", "c_rel", "c_adv", "exact"):
        lo, hi, und = PE.interval([d[key] for d in draws])
        out[key] = {"point": point[key], "interval": [lo, hi], "undefined": und,
                    "class": PE.classify(point[key], lo, hi, und, np.inf)}
    return out


def products(terms, lats=None, lons=None):
    """Section 5's three products from the budget terms."""
    lats, lons = (GRID_LATS if lats is None else lats), (GRID_LONS if lons is None else lons)
    j = int(np.flatnonzero(np.isclose(lons, BASE_LON))[0])
    section = terms["flux_x"][WINDOW][..., j].mean(axis=0)                       # level by latitude
    wp = pressure_weights()
    vmean_section = np.tensordot(wp, section, axes=([0], [0]))                  # by latitude
    k650 = int(np.flatnonzero(LEVELS == 650.0)[0])
    low = LEVELS >= 750.0

    def lat_mean(x, band):
        rows = (lats >= band[0] - 1e-9) & (lats <= band[1] + 1e-9)
        c = np.cos(np.deg2rad(lats[rows]))
        return float((x[rows] * c).sum() / c.sum())
    w_low = pressure_weights(LEVELS[low])
    summary = {"650 hPa, 12 to 18 N": lat_mean(section[k650], (12.0, 18.0)),
               "950 to 750 hPa, 6 to 14 N": lat_mean(np.tensordot(w_low, section[low], axes=([0], [0])), (6.0, 14.0)),
               "vertical mean, 6 to 14 N": lat_mean(vmean_section, (6.0, 14.0)),
               "vertical mean, 15 to 20 N": lat_mean(vmean_section, (15.0, 20.0)),
               "vertical mean, 21 to 25 N": lat_mean(vmean_section, (21.0, 25.0))}
    cols = (lons >= MAP_LONS[0] - 1e-9) & (lons <= MAP_LONS[1] + 1e-9)
    hov = {band: {k: band_mean(vertical_mean(terms[k]), lats, rng)[:, cols] for k in ("K", "gfc", "baroclinic", "barotropic", "total")}
           for band, rng in (("6 to 14 N", BOX_LATS), ("15 to 20 N", NORTH_LATS))}
    kmax = {band: [float(lons[cols][int(np.argmax(row))]) for row in hov[band]["K"]] for band in hov}
    plan = {f"{int(L * 6)} h": {k: vertical_mean(terms[k])[int(np.flatnonzero(LAGS == L)[0])] for k in ("K", "gfc", "flux_x", "flux_y")}
            for L in (0, 4, 8)}
    return {"section": section, "section_vertical_mean": vmean_section, "section_summary": summary,
            "hovmoller": hov, "k_max_longitude": kmax, "plan": plan}


def retained_row(year, level=700, lat=BASE_LAT, lons=np.arange(-45.0, 35.5, 1.0)):
    """One season's retained 0.5 degree meridional wind along `lat` at the integer longitudes."""
    import netCDF4
    if year == 2005:
        raise SystemExit("REFUSED: the 2005 season stays unopened")
    d = netCDF4.Dataset(os.path.join("data/era5/region6h", f"era5_v{level}_{year}_6h_region.nc"))
    try:
        la = np.asarray(d.variables["latitude"][:], float); lo = np.asarray(d.variables["longitude"][:], float)
        i = int(np.flatnonzero(np.isclose(la, lat))[0]); cols = [int(np.flatnonzero(np.isclose(lo, x))[0]) for x in lons]
        V = np.asarray(d.variables["v"][:, 0, i, :], float)[:, cols]
    finally:
        d.close()
    return V


def gate4(cmp_v, years):
    """Gate 4 of section 8. The new files' 700 hPa composite along 10 N against the retained
    files' at the integer longitudes from 45 W to 35 E."""
    lons = np.arange(-45.0, 35.5, 1.0)
    st = PE.stack([lag_sums(*MX.block(retained_row(y, lons=lons), PR.point_series(y, 700, BASE_LON))) for y in years])
    old = composite(st)
    k = int(np.flatnonzero(LEVELS == 700.0)[0]); i = int(np.flatnonzero(np.isclose(GRID_LATS, BASE_LAT))[0])
    cols = [int(np.flatnonzero(np.isclose(GRID_LONS, x))[0]) for x in lons]
    new = cmp_v[:, k, i][:, cols]
    if not (np.isfinite(new).all() and np.isfinite(old).all()):
        raise SystemExit("REFUSED: a 700 hPa composite of gate 4 holds a value that is not finite")
    with np.errstate(invalid="ignore", divide="ignore"):
        r = float(np.corrcoef(new.ravel(), old.ravel())[0, 1])
    if not np.isfinite(r) or r < GATE4_MIN_R:
        raise SystemExit(f"REFUSED: the new 700 hPa composite correlates at {r:.4f} with the retained one, below {GATE4_MIN_R}")
    return {"r": r, "max_abs_difference": float(np.abs(new - old).max())}


def check_constructions(path):
    """Gate 3 of section 8. Written by this script at its digest, its verdicts recomputed, and
    adequate after changes 1 and 2."""
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    if rec["generated_by"]["sha256"] != SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the constructions were run by a different version of this script")
    if set(rec["results"]) != set(CONSTRUCTIONS):
        raise SystemExit("REFUSED: the constructions record does not hold the six declared constructions")
    for v in VERSIONS.values():
        if adequacy(rec["results"], v) != rec["verdicts"].get(v):
            raise SystemExit("REFUSED: a stored verdict disagrees with the rule applied to the stored readings")
    if not rec["verdicts"][VERSIONS[2]]["adequate"]:
        raise SystemExit("REFUSED: the estimator after changes 1 and 2 is not adequate on the constructions")
    return rec


PLAN_LATS, PLAN_LONS = (0.0, 30.0), (-40.0, 10.0)


def terrain_shares(years, lat_range=ERA5_FULL_LATS, lon_range=BOX_LONS):
    """The share of samples below the retained surface pressure at every level over the box, or
    over the plan views' area with `lat_range` and `lon_range` set to it."""
    import netCDF4
    below = np.zeros(len(LEVELS)); count = 0
    for y in years:
        d = netCDF4.Dataset(os.path.join("data/era5/region6h", f"era5_sp_{y}_6h_region.nc"))
        try:
            la = np.asarray(d.variables["latitude"][:], float); lo = np.asarray(d.variables["longitude"][:], float)
            ri = np.flatnonzero(np.isin(np.round(la, 6), GRID_LATS[(GRID_LATS >= lat_range[0]) & (GRID_LATS <= lat_range[1])]))
            ci = np.flatnonzero(np.isin(np.round(lo, 6), GRID_LONS[(GRID_LONS >= lon_range[0]) & (GRID_LONS <= lon_range[1])]))
            sp = np.asarray(d.variables["sp"][:], float)[:, ri][:, :, ci] / 100.0
        finally:
            d.close()
        below += np.array([(L >= sp).sum() for L in LEVELS]); count += sp.size
    return {f"{int(L)} hPa": float(b / count) for L, b in zip(LEVELS, below)}


def centroid_partner():
    """The frozen centroid at 700 hPa, base 15 W, with the 10 N point reference (section 6)."""
    seasons, refs = MX.era5_inputs(700, BASE_LON)
    r = PR.reference_results(seasons, refs[MX.POINT_REF], BASE_LON)
    return MX.centroid_summary(json.loads(json.dumps(r)))


def outcome(c_class):
    """Section 9's row from the class of c_E over the box."""
    if c_class not in ("eastward", "westward", "standing", "unresolved"):
        raise SystemExit(f"REFUSED: {c_class} is not a class")
    return c_class


# ------------------------------------------------------------------ records

def refuse_existing(paths):
    for p in paths:
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and records are never overwritten")


def generated_by():
    return {"script": "scripts/eastern_energy_flux.py", "sha256": SCRIPT_SHA256_AT_LOAD,
            "declared": "EASTERN_ENERGY_FLUX_2026-11-04.md", "frozen_sha256": PE.SCRIPT_SHA256_AT_LOAD,
            "estimator": {"box_lons": list(BOX_LONS), "box_lats": list(BOX_LATS), "construction_full_lats": list(FULL_LATS),
                          "reading_full_lats": list(ERA5_FULL_LATS),
                          "window_steps": WINDOW_STEPS, "f0": F0, "slow_deg_per_day": SLOW, "versions": VERSIONS}}


def constructions_text(rec):
    L = ["# the energy velocity on six shallow-water packets (sections 7 and 10)", ""]
    for version, v in rec["verdicts"].items():
        L += [f"The estimator {version} adequate: {'yes' if v['adequate'] else 'no'}." + ("" if v["adequate"] else " " + "; ".join(v["failures"])), ""]
    L += ["| construction | U (m/s) | wavelength (deg) | H (m) | theory | truth | version | c_rel | c_adv | c_E | point class | true class |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in rec["results"].items():
        sp = r["spec"]
        for version, e in r["versions"].items():
            L.append(f"| {name} | {sp['U']:+.0f} | {sp['wavelength']:.0f} | {sp['H']:.0f} | {r['theory']:+.2f} | {r['truth']:+.2f} | {version} | "
                     f"{e['c_rel']:+.2f} | {e['c_adv']:+.2f} | {e['c_E']:+.2f} | {point_class(e['c_E'])} | {point_class(r['truth'])} |")
    L += ["", "Speeds in deg/day at 10 N. Truth is the least-squares speed of the total-energy centroid over days 1 to 7.", ""]
    return "\n".join(L)


def reading_text(rec):
    d, L = rec["decision"], []
    L += ["# the energy flux of the regression composite at 10 N, 15 W, 700 hPa (sections 9 to 12)", "",
          f"The outcome row of section 9 is {rec['outcome']}. The frozen centroid at the same base and reference reads "
          f"{rec['centroid']['envelope_class']} ({_f(rec['centroid']['envelope_speed'])}, {_f(rec['centroid']['envelope_interval'][0])} to "
          f"{_f(rec['centroid']['envelope_interval'][1])}).", "",
          "| reading | estimator | point | interval | class |", "| --- | --- | --- | --- | --- |"]
    for label, key in (("950 to 350 hPa", "primary"), ("850 to 350 hPa", "from_850")):
        for est in ("c_E", "c_rel", "c_adv", "exact"):
            x = d[key][est]
            L.append(f"| {label} | {est} | {_f(x['point'])} | {_f(x['interval'][0])} to {_f(x['interval'][1])} | {x['class']} |")
    p = d["primary"]["point"]
    L += ["", f"Potential energy's share {p['ape_share']:.3f}, edge term {_f(p['edge_term'])}, advective shortfall {_f(p['advective_shortfall'])} deg/day.", "",
          "| section summary (zonal ageostrophic flux at 15 W, lags -42 to +42 h) | m^3/s^3 |", "| --- | --- |"]
    L += [f"| {k} | {v:+.3g} |" for k, v in rec["products"]["section_summary"].items()]
    L += ["", "| lag (h) | " + " | ".join(f"largest K, {b}" for b in rec["products"]["k_max_longitude"]) + " |", "| --- |" + " --- |" * len(rec["products"]["k_max_longitude"])]
    for i, lag in enumerate(LAGS):
        L.append(f"| {int(lag) * 6:+d} | " + " | ".join(f"{v[i]:+.0f}" for v in rec["products"]["k_max_longitude"].values()) + " |")
    L += ["", "| level | share below surface pressure, box | share below surface pressure, plan views |", "| --- | --- | --- |"]
    L += [f"| {k} | {v:.4f} | {rec['terrain_plan'][k]:.4f} |" for k, v in rec["terrain_box"].items()]
    L += ["", f"Gate 4 correlation {rec['gate4']['r']:.4f}, largest difference {rec['gate4']['max_abs_difference']:.3f} m/s.", "",
          f"Speeds in deg/day at 10 N (1 deg/day = {M_PER_DEG_DAY:.4f} m/s), read over {_lat(d['primary']['lat_range'])} with the box's "
          f"longitudes and lags. Pressure levels are not masked below the surface.", ""]
    return "\n".join(L)


def _f(v):
    return "none" if v is None else f"{v:+.2f}"


def _lat(rng):
    return " to ".join(f"{abs(x):g} {'S' if x < 0 else 'N'}" for x in rng)


def draw(paths, prod, lats=None, lons=None):
    """The three figures, every title saying the levels are unmasked, the plan views drawn over
    PLAN_LATS and PLAN_LONS, the area of their below-ground shares, with one arrow scale across
    the three lags. Returns the titles and the plan views' drawn extent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lats, lons = (GRID_LATS if lats is None else lats), (GRID_LONS if lons is None else lons)
    titles = {}
    north = lats >= 0
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    sec = prod["section"][:, north]
    lim = np.abs(sec).max()
    cs = axes[0].contourf(lats[north], LEVELS, sec, levels=np.linspace(-lim, lim, 21), cmap="RdBu_r")
    axes[0].invert_yaxis(); axes[0].set_xlabel("latitude (deg N)"); axes[0].set_ylabel("pressure (hPa)")
    axes[0].set_title("zonal ageostrophic geopotential flux at 15 W, lags -42 to +42 h", fontsize=9)
    fig.colorbar(cs, ax=axes[0], label="m^3 s^-3")
    axes[1].plot(lats[north], prod["section_vertical_mean"][north], "k"); axes[1].axhline(0, color="0.5", lw=0.8)
    axes[1].set_xlabel("latitude (deg N)"); axes[1].set_title("its vertical mean, 950 to 350 hPa, in m^3 s^-3", fontsize=9)
    titles["section"] = "Regression on the 700 hPa meridional wind at 10 N, 15 W, 17 seasons. Pressure levels unmasked below the surface."
    fig.suptitle(titles["section"], fontsize=9)
    fig.savefig(paths["section"], dpi=90); plt.close(fig)
    cols = (lons >= MAP_LONS[0] - 1e-9) & (lons <= MAP_LONS[1] + 1e-9)
    names = ("K", "baroclinic", "barotropic", "gfc", "total")
    fig, axes = plt.subplots(2, 5, figsize=(16, 7), sharex=True, sharey=True, constrained_layout=True)
    for r_i, band in enumerate(prod["hovmoller"]):
        for c_i, k in enumerate(names):
            X = prod["hovmoller"][band][k]
            lim = np.abs(X).max()
            cs = axes[r_i, c_i].contourf(lons[cols], LAGS / PE.OBS_PER_DAY, X, levels=np.linspace(-lim, lim, 21) if k != "K" else 20, cmap="RdBu_r" if k != "K" else "Greys")
            axes[r_i, c_i].contour(lons[cols], LAGS / PE.OBS_PER_DAY, prod["hovmoller"][band]["K"], 6, colors="k", linewidths=0.5)
            axes[r_i, c_i].set_title(f"{k}, {band}", fontsize=9)
            fig.colorbar(cs, ax=axes[r_i, c_i], shrink=0.8, label="J/kg" if k == "K" else "W/kg")
        axes[r_i, 0].set_ylabel("lag (days)")
    for c_i in range(5):
        axes[-1, c_i].set_xlabel("longitude (deg E)")
    titles["hovmoller"] = "Vertical means from 950 to 350 hPa, pressure levels unmasked below the surface. Energy in J/kg, terms in W/kg. Black contours mark K."
    fig.suptitle(titles["hovmoller"], fontsize=9)
    fig.savefig(paths["hovmoller"], dpi=80); plt.close(fig)
    rows = (lats >= PLAN_LATS[0] - 1e-9) & (lats <= PLAN_LATS[1] + 1e-9)
    pcols = (lons >= PLAN_LONS[0] - 1e-9) & (lons <= PLAN_LONS[1] + 1e-9)
    la, lo = lats[rows], lons[pcols]
    crop = {lag: {k: P[k][np.ix_(rows, pcols)] for k in ("K", "gfc", "flux_x", "flux_y")} for lag, P in prod["plan"].items()}
    vmax = max(float(np.hypot(P["flux_x"], P["flux_y"]).max()) for P in crop.values())
    key = float(10 ** np.floor(np.log10(vmax))) if vmax > 0 else 1.0
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True, sharey=True, constrained_layout=True)
    for r_i, (lag, P) in enumerate(crop.items()):
        for c_i, k in enumerate(("K", "gfc")):
            X = P[k]
            lim = np.abs(X).max()
            cs = axes[r_i, c_i].contourf(lo, la, X, levels=20 if k == "K" else np.linspace(-lim, lim, 21), cmap="Greys" if k == "K" else "RdBu_r")
            q = axes[r_i, c_i].quiver(lo[::3], la[::3], P["flux_x"][::3, ::3], P["flux_y"][::3, ::3], scale=8.0 * vmax, scale_units="width")
            axes[r_i, c_i].quiverkey(q, 0.85, 1.03, key, f"{key:g} m^3 s^-3", labelpos="E", fontproperties={"size": 8})
            axes[r_i, c_i].set_title(f"{k} and ageostrophic flux, lag {lag}", fontsize=9, loc="left")
            fig.colorbar(cs, ax=axes[r_i, c_i], shrink=0.8, label="J/kg" if k == "K" else "W/kg")
        axes[r_i, 0].set_ylabel("latitude (deg N)")
    for c_i in range(2):
        axes[-1, c_i].set_xlabel("longitude (deg E)")
    titles["plan"] = ("Vertical means from 950 to 350 hPa of the regression composites, pressure levels unmasked below the surface. "
                      "One arrow scale for all panels.")
    fig.suptitle(titles["plan"], fontsize=9)
    fig.savefig(paths["plan"], dpi=80); plt.close(fig)
    return {"titles": titles, "plan_extent": [float(lo[0]), float(lo[-1]), float(la[0]), float(la[-1])]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("constructions", "read"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--constructions", default=None)
    ap.add_argument("--matched", default=None)
    ap.add_argument("--flux-dir", default=D.OUT_DIR)
    a = ap.parse_args(argv)
    if a.mode == "constructions":
        pj, pm = (os.path.join(a.out_dir, f"eval60_energy_flux_constructions{a.tag}.{e}") for e in ("json", "md"))
        refuse_existing([pj, pm])
        res = json.loads(json.dumps(run_constructions()))
        rec = {"generated_by": generated_by(), "results": res, "verdicts": {v: adequacy(res, v) for v in VERSIONS.values()}}
        with open(pj, "w", encoding="utf-8") as fh:
            json.dump(rec, fh)
        text = constructions_text(rec)
        with open(pm, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(text)
        return 0
    if not (a.constructions and a.matched):
        raise SystemExit("REFUSED: --constructions and --matched are required for the reading")
    paths = {k: os.path.join(a.out_dir, f"eval60_energy_flux{s}{a.tag}.{e}") for k, s, e in
             (("record", "", "json"), ("tables", "_tables", "md"), ("section", "_section", "png"), ("hovmoller", "_hovmoller", "png"), ("plan", "_plan", "png"))}
    refuse_existing(paths.values())
    check_frozen()
    M.check_frozen()
    check_constructions(a.constructions)
    years = D.YEARS
    if 2005 in years:
        raise SystemExit("REFUSED: the 2005 season stays unopened")
    totals, box, means = accumulate(years, a.flux_dir)
    shape = (len(LEVELS), len(GRID_LATS), len(GRID_LONS))
    cmp = {v: composite(PE.stack([totals[v]]), shape=shape) for v in VARS}
    g4 = gate4(cmp["v"], years)
    mean = {v: weighted_mean(means[v]) for v in ("u", "v", "t")}
    prod = products(budget_terms(cmp, mean))
    dec = {"primary": decision(box, means), "from_850": decision(box, means, bottom_hpa=850.0)}
    with open(a.matched, encoding="utf-8") as fh:
        stored = json.load(fh)["results"]["primary and transect"]["700"]["-15"]
    rec = {"generated_by": {**generated_by(), "constructions": {"path": a.constructions, "sha256": PE.sha256_of(a.constructions)},
                            "matched": {"path": a.matched, "sha256": PE.sha256_of(a.matched)},
                            "inputs": {os.path.basename(D.path(v, y, a.flux_dir)): PE.sha256_of(D.path(v, y, a.flux_dir)) for y in years for v in VARS},
                            "gates": "1 every file checked, 2 frozen digests, 3 constructions adequate after changes 1 and 2, 4 the 700 hPa composite against the retained files, 5 no 2005"},
           "decision": dec, "outcome": outcome(dec["primary"]["c_E"]["class"]), "centroid": centroid_partner(),
           "stored_band_mean_centroid": {k: stored[k] for k in ("envelope_class", "envelope_speed", "envelope_interval")},
           "gate4": g4, "terrain_box": terrain_shares(years), "terrain_plan": terrain_shares(years, PLAN_LATS, PLAN_LONS),
           "products": {"section_summary": prod["section_summary"], "k_max_longitude": prod["k_max_longitude"],
                        "section": prod["section"].tolist(), "section_vertical_mean": prod["section_vertical_mean"].tolist(),
                        "hovmoller": {b: {k: x.tolist() for k, x in h.items()} for b, h in prod["hovmoller"].items()}}}
    rec = json.loads(json.dumps(rec))
    draw(paths, prod)
    with open(paths["record"], "w", encoding="utf-8") as fh:
        json.dump(rec, fh)
    text = reading_text(rec)
    with open(paths["tables"], "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
