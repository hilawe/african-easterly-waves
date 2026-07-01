#!/usr/bin/env python
"""Ethiopian Highlands and Darfur as AEW convective-genesis regions.

Characterizes where and when convective-system families initiate over the two eastern
trigger regions, and the low-level flow around genesis, using the ISCCP CT family genesis
slice (with reliable ctime) and ERA5 daily v700. Three regions are compared: the Ethiopian
Highlands, Darfur, and the West African Sahel as a downstream reference.

Panels: (1) genesis density with the region boxes; (2) diurnal cycle of genesis (the
terrain-triggered afternoon-initiation signature); (3) superposed-epoch v700 around
genesis (the mean low-level meridional flow before and after initiation).
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr

from aew.binning import bin_sum
from aew.data.ct import from_ct_genesis

REGIONS = {
    "Ethiopian Highlands": dict(lon=(35, 42), lat=(6, 14), color="tab:red"),
    "Darfur":              dict(lon=(22, 27), lat=(10, 16), color="tab:purple"),
    "West African Sahel":  dict(lon=(-10, 10), lat=(8, 16), color="tab:blue"),
}


def in_box(tr, box):
    return ((tr.lon >= box["lon"][0]) & (tr.lon <= box["lon"][1])
            & (tr.lat >= box["lat"][0]) & (tr.lat <= box["lat"][1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct", default="data/original/CT/ct_genesis_time.nc")
    ap.add_argument("--era5", default="data/era5/era5_v700_1984-2007_daily.nc")
    ap.add_argument("--out", default="fig_east_genesis.png")
    a = ap.parse_args()

    g = from_ct_genesis(a.ct)
    t = pd.DatetimeIndex(g.time)
    jas = np.isin(t.month, [7, 8, 9]) & ~pd.isna(t)
    g = g.filter(jas)
    t = pd.DatetimeIndex(g.time)
    hour = t.hour.values

    # ERA5 daily v700 region-mean series for the superposed-epoch analysis
    ds = xr.open_dataset(a.era5)["v700"].squeeze(drop=True)
    latn = "latitude" if "latitude" in ds.coords else "lat"
    lonn = "longitude" if "longitude" in ds.coords else "lon"
    vtime = pd.DatetimeIndex(ds["time"].values)

    def roi_v700_series(box):
        sub = ds.sel({latn: slice(box["lat"][0], box["lat"][1]),
                      lonn: slice(box["lon"][0], box["lon"][1])})
        if sub[latn].size == 0:  # descending lat
            sub = ds.sel({latn: slice(box["lat"][1], box["lat"][0]),
                          lonn: slice(box["lon"][0], box["lon"][1])})
        return pd.Series(np.asarray(sub.mean((latn, lonn)).values, float), index=vtime)

    lags = np.arange(-5, 6)
    print(f"{'region':22s} {'n_gen':>6} {'deep%':>6} {'medLife':>7} {'peakHrUTC':>9}")
    sea = {}
    for name, box in REGIONS.items():
        m = in_box(g, box)
        sub_t = pd.DatetimeIndex(g.time[m])
        deep = np.mean(g.variables["tmincl_K"][m] < 200) * 100
        medlife = np.median(g.variables["cslife"][m])
        hh = pd.DatetimeIndex(g.time[m]).hour.values
        peak_hr = int(np.bincount(hh, minlength=24).argmax())
        print(f"{name:22s} {m.sum():6d} {deep:6.0f} {medlife:7.0f} {peak_hr:9d}")

        # superposed epoch of ROI-mean v700 about unique genesis days
        vser = roi_v700_series(box)
        vpos = pd.Series(np.arange(vser.size), index=vser.index)
        days = pd.DatetimeIndex(pd.DatetimeIndex(sub_t).floor("D").unique())
        days = days[days.isin(vser.index)]
        comp = np.full(lags.size, np.nan)
        for i, L in enumerate(lags):
            tgt = days + pd.Timedelta(days=int(L))
            tgt = tgt[tgt.isin(vser.index)]
            comp[i] = vser.iloc[vpos.loc[tgt].to_numpy()].mean()
        sea[name] = comp

    # genesis density for the map
    lon_c = np.arange(-20, 55, 2.0)
    lat_c = np.arange(0, 25, 2.0)
    dens, _ = bin_sum(lon_c, lat_c, g.lon, g.lat, z=np.ones(len(g)), variant="fixed")

    import matplotlib
    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(14, 4.5))
    ax0 = fig.add_subplot(1, 3, 1, projection=proj)
    pc = ax0.pcolormesh(lon_c, lat_c, dens, cmap="YlOrRd", vmin=0, transform=proj, shading="auto")
    ax0.coastlines(resolution="110m", linewidth=0.5)
    ax0.set_extent([-20, 52, 0, 22], crs=proj)
    ax0.set_xticks(np.arange(-20, 51, 20), crs=proj); ax0.set_yticks(np.arange(0, 21, 10), crs=proj)
    ax0.tick_params(labelsize=8)
    for name, box in REGIONS.items():
        w, e = box["lon"]; s, n = box["lat"]
        ax0.plot([w, e, e, w, w], [s, s, n, n, s], color=box["color"], lw=2, transform=proj)
    ax0.set_title("CT genesis density (JAS) with regions", fontsize=10)
    fig.colorbar(pc, ax=ax0, shrink=0.7, label="genesis count")

    ax1 = fig.add_subplot(1, 3, 2)
    for name, box in REGIONS.items():
        m = in_box(g, box)
        hh = pd.DatetimeIndex(g.time[m]).hour.values
        frac = np.bincount(hh, minlength=24) / max(m.sum(), 1)
        hrs = np.arange(24)
        sel = frac > 0
        ax1.plot(hrs[sel], frac[sel] * 100, "o-", color=box["color"], ms=3, label=name)
    ax1.set_xlabel("genesis hour (UTC)"); ax1.set_ylabel("% of region's genesis")
    ax1.set_title("Diurnal cycle of genesis"); ax1.grid(alpha=0.3); ax1.legend(fontsize=7)

    ax2 = fig.add_subplot(1, 3, 3)
    for name, box in REGIONS.items():
        ax2.plot(lags, sea[name], "o-", color=box["color"], ms=3, label=name)
    ax2.axvline(0, color="green", lw=1.5); ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xlabel("days relative to genesis"); ax2.set_ylabel("region-mean v700 (m/s)")
    ax2.set_title("Low-level meridional flow around genesis"); ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)

    fig.tight_layout(); fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
