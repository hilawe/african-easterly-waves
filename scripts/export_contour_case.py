#!/usr/bin/env python3
"""Export the masked advection fields version 1 contours, plus the port's own axes."""
import os, sys
import numpy as np
from scipy.io import savemat
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "src"))
from aew.v1port import climatology as clim, load as L, pipeline as P
from aew.v1port.detection import MAX_ZONAL_WIND, _prepare, trough_axes

S = os.environ["AEW_ORACLE_DIR"]
D = "data/eraint/v1port_buffered"
with np.load(os.environ["AEW_CLIMO_CACHE"], allow_pickle=False) as z:
    climo = {"keys":[tuple(int(x) for x in r) for r in z["keys"]], "mean":z["mean"],
             "counts":z["counts"], "short_steps":[]}
times, latgrid, longrid, u, v = L.load_year(1990, D, "eraint")
curv = P.curvature_from_winds(latgrid, longrid, u, v)
anom = P.anomaly_from_climatology(curv, times, climo); del curv
adv = P._advection(latgrid, longrid, u, v, anom)
nat = abs(float(latgrid[1,0]-latgrid[0,0]))
C = {n: clim.gaussian_decimate(f, nat, P.COARSE_RESOLUTION_DEG)
     for n,f in (("u",u),("anomaly",anom),("advection",adv))}
del adv
lat_c, lon_c = P.coarse_grid(latgrid, longrid, nat, P.COARSE_RESOLUTION_DEG)
rc, cc = P._subset(lat_c, lon_c, P.DOMAIN_LAT, P.DOMAIN_LON)
LG, LO = lat_c[np.ix_(rc,cc)], lon_c[np.ix_(rc,cc)]
ct, _ = P.thresholds_for("ERA-Int", 700)
out = {}
# a whole season rather than a handful, since spurious hits are expected to be rare and
# a rare event needs a population to be measured on
for step in range(600, 900, 4):
    wind = clim.smooth9(C["u"][step][np.ix_(rc,cc)])
    field = _prepare(C["anomaly"][step][np.ix_(rc,cc)], LG[:,0])
    a = clim.smooth9(C["advection"][step][np.ix_(rc,cc)])
    west = wind > MAX_ZONAL_WIND
    a = np.where(west, np.nan, a); field = np.where(west, np.nan, field)
    weak = field < ct
    a = np.where(weak, np.nan, a)
    axes = trough_axes(LG, LO, a)
    out[f"s{step}"] = {
        "latgrid": LG, "longrid": LO, "adv": a,
        "port_n": float(len(axes)),
        "port_lat": np.array([float(np.mean(x[0])) for x in axes], dtype=float),
        "port_lon": np.array([float(np.mean(x[1])) for x in axes], dtype=float),
    }
savemat(os.path.join(S, "contour_cases.mat"), out, do_compression=True)
print(f"wrote {len(out)} timesteps; port axes per step: "
      f"{np.mean([v['port_n'] for v in out.values()]):.1f} mean")
