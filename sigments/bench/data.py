"""Scenes with reference crowns: the two Krynki windows of the dead-pine exercise."""
from paths import DATA
import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd

ROOT = DATA
SCENES = {"krynki1": ("dane/krynki1_orto.tif", "dane/krynki1_korony.gpkg"),
          "krynki2": ("dane/krynki2_orto.tif", "kontrola/krynki2_korony.gpkg")}


def load(name, window=None):
    """Return dict(data (bands, rows, cols) float64, valid, crowns (rows, cols) int32 with 0 = none, n_crowns)."""
    tif, ref = SCENES[name]
    with rasterio.open(ROOT + tif) as src:
        data = src.read().astype(np.float64)
        tf = src.transform
        bands = src.descriptions
    valid = ~(data == 0).all(axis=0)
    g = gpd.read_file(ROOT + ref)
    g = g[~g.geometry.is_empty & (g.geometry.area > 0)].reset_index(drop=True)
    crowns = rasterize([(geom, i + 1) for i, geom in enumerate(g.geometry)], out_shape=valid.shape,
                       transform=tf, fill=0, dtype="int32")
    crowns[~valid] = 0
    if window is not None:
        r0, r1, c0, c1 = window
        data, valid, crowns = data[:, r0:r1, c0:c1].copy(), valid[r0:r1, c0:c1].copy(), crowns[r0:r1, c0:c1].copy()
    return {"name": name, "data": data, "valid": valid, "crowns": crowns, "bands": bands,
            "n_crowns": int(len(np.unique(crowns)) - 1)}
