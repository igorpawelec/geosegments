"""
pygeoadaptels — Vectorization of adaptel labels to polygons.

Uses only rasterio.features.shapes + fiona for polygon export.
No geopandas or shapely required (following Netzel's recommendation).

Based on:
    https://github.com/rasterio/rasterio/blob/main/examples/rasterio_polygonize.py
    https://gis.stackexchange.com/questions/417383/how-to-apply-gdal-polygonize
"""

import math
import os
import numpy as np

# Features are written in batches rather than one at a time.
#
# A GeoPackage is SQLite, and one write per feature is one transaction per row.
# Measured on a 3000x3000 adaptel raster (300 809 polygons): 204.3 s writing
# feature by feature against 16.8 s in batches of 20 000 -- the same polygons,
# read back and counted, 12x faster. A Shapefile takes 12.1 s either way, so
# only the GeoPackage path was ever paying this, which is unfortunate because
# it is the format QGIS Processing offers first. On a full orthophoto the
# difference is about fifty minutes against four.
#
# The spatial index was the other suspect and it is innocent: creating the
# layer with SPATIAL_INDEX=NO changed 204.3 s into 195.9 s, i.e. nothing. It is
# left enabled, since 300 000 polygons without one are painful to pan around.
WRITE_BATCH = 20000

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _ensure_deps():
    """Lazy import rasterio.features.shapes and fiona."""
    try:
        from rasterio.features import shapes
    except ImportError as e:
        raise ImportError(
            "rasterio is required for vectorization.\n"
            "Install:  conda install -c conda-forge rasterio"
        ) from e
    try:
        import fiona
    except ImportError as e:
        raise ImportError(
            "fiona is required for writing vector files.\n"
            "Install:  conda install -c conda-forge fiona"
        ) from e
    return shapes, fiona


def _ring_area_length(ring):
    """Area and length of one closed ring of (x, y) vertices.

    The shoelace sum is taken on coordinates moved to the ring's first
    vertex. In map coordinates the products are of the order of 1e11 and
    their differences of the order of the pixel area, so the plain sum
    carries an error of about 1e-4 m2, which is visible on a polygon of one
    0.25 m pixel; moved to the origin it is exact to the last digits.
    """
    ox, oy = ring[0][0], ring[0][1]
    x0 = y0 = 0.0
    area2 = 0.0
    length = 0.0
    for pt in ring[1:]:
        x1, y1 = pt[0] - ox, pt[1] - oy
        area2 += x0 * y1 - x1 * y0
        length += math.hypot(x1 - x0, y1 - y0)
        x0, y0 = x1, y1
    return abs(area2) / 2.0, length


def polygon_area_perimeter(geom):
    """Area and perimeter of a GeoJSON-like Polygon, in the units of its CRS.

    The area is the outer ring's less the holes'; the perimeter is the length
    of every ring, the holes' included, because a hole is boundary too.
    """
    rings = geom["coordinates"]
    area, perimeter = _ring_area_length(rings[0])
    for hole in rings[1:]:
        a, length = _ring_area_length(hole)
        area -= a
        perimeter += length
    return area, perimeter


def vectorize_adaptels(labels, transform, crs_wkt,
                       output_path,
                       driver="ESRI Shapefile",
                       nodata=-9999,
                       connectivity=4,
                       compute_area=True,
                       quiet=False):
    """
    Convert adaptel label raster to polygon vector file.

    Uses rasterio.features.shapes for polygonization and fiona for
    writing — no geopandas or shapely needed.

    Parameters
    ----------
    labels : np.ndarray, shape (rows, cols), dtype int32
        Adaptel label raster from create_adaptels() or adaptels_from_array().
    transform : affine.Affine
        Geotransform from the input raster (rasterio src.transform).
    crs_wkt : str
        Coordinate reference system as WKT string (rasterio src.crs.to_wkt()).
    output_path : str or Path
        Output file path. Extension determines format:
        .shp → ESRI Shapefile (default), .gpkg → GeoPackage, .geojson → GeoJSON.
        If no recognized extension, uses `driver` parameter.
    driver : str, optional
        Fiona driver override. Default "ESRI Shapefile".
    nodata : int, optional
        Nodata value in labels array. Default -9999.
    connectivity : int, optional
        Pixel connectivity for polygonization: 4 or 8. Default 4.
    compute_area : bool, optional
        If True, write each polygon's area (``area_m2``) and perimeter
        (``perimeter``), measured on its own geometry in the units of the
        CRS: square metres and metres for a projected one. The perimeter
        runs along pixel edges and counts the edges of holes. A label left
        in several pieces is several features, each with the area and
        perimeter of its piece, so the areas add up to the labelled area.
        Default True.

        Until 0.10.5 ``area_m2`` was the area of the whole label, repeated
        on every piece of it, and ``perimeter`` was that of a circle of the
        same area, ``2 * sqrt(pi * area)``.

    Returns
    -------
    n_polygons : int
        Number of polygons written.

    Examples
    --------
    >>> from pygeoadaptels import create_adaptels
    >>> from pygeoadaptels.vectorize import vectorize_adaptels
    >>> import rasterio
    >>>
    >>> labels, n = create_adaptels('input.tif', threshold=60.0)
    >>> with rasterio.open('input.tif') as src:
    ...     vectorize_adaptels(labels, src.transform, src.crs.to_wkt(),
    ...                        'adaptels.shp')
    """
    shapes, fiona = _ensure_deps()
    output_path = str(output_path)

    # Auto-detect driver from extension
    ext = os.path.splitext(output_path)[1].lower()
    ext_drivers = {".shp": "ESRI Shapefile", ".gpkg": "GPKG", ".geojson": "GeoJSON"}
    if ext in ext_drivers:
        driver = ext_drivers[ext]

    # Prepare data
    data = labels.astype(np.int32)
    mask = (data != int(nodata)) & (data >= 0)

    # Schema
    props = {'adaptel_id': 'int'}
    if compute_area:
        props['area_m2'] = 'float'
        props['perimeter'] = 'float'

    schema = {
        'geometry': 'Polygon',
        'properties': props,
    }

    # Polygonize + write
    n_polygons = 0
    poly_iter = shapes(data, mask=mask, transform=transform,
                       connectivity=connectivity)
    if _HAS_TQDM and not quiet:
        # Only for the bar's total, so it is computed only when there is a bar:
        # it sorts every valid pixel, which is pure waste under ``quiet``.
        poly_iter = _tqdm(poly_iter, desc="Vectorizing", unit="poly",
                          total=int(len(np.unique(data[mask]))))

    with fiona.open(
        output_path,
        'w',
        driver=driver,
        crs_wkt=crs_wkt,
        schema=schema
    ) as dst:
        batch = []
        for geom, value in poly_iter:
            adaptel_id = int(value)
            if adaptel_id < 0:
                continue

            feature = {
                'geometry': geom,
                'properties': {'adaptel_id': adaptel_id},
            }

            if compute_area:
                # From the geometry, not from the label's pixel count: a
                # label in two pieces is two features, and each must carry
                # its own area, or the attribute sums to more than the map.
                # Four decimals, because one 0.25 m pixel is 0.0625 m2.
                area, perimeter = polygon_area_perimeter(geom)
                feature['properties']['area_m2'] = round(area, 4)
                feature['properties']['perimeter'] = round(perimeter, 4)

            batch.append(feature)
            n_polygons += 1
            if len(batch) >= WRITE_BATCH:
                dst.writerecords(batch)
                batch = []
        if batch:
            dst.writerecords(batch)

    return n_polygons


def vectorize_from_file(input_raster, output_path,
                        driver="ESRI Shapefile",
                        nodata=None, connectivity=4,
                        compute_area=True, quiet=False):
    """
    Convenience wrapper: read adaptel raster from file and vectorize.

    Parameters
    ----------
    input_raster : str or Path
        Path to adaptel GeoTIFF (output of create_adaptels).
    output_path : str or Path
        Output vector file path.
    driver : str
        Fiona driver. Default "ESRI Shapefile".
    nodata : int, optional
        Override nodata. If None, reads from raster metadata.
    connectivity : int
        Pixel connectivity (4 or 8).
    compute_area : bool
        Write each polygon's own area and perimeter (see
        :func:`vectorize_adaptels`).

    Returns
    -------
    n_polygons : int

    Examples
    --------
    >>> from pygeoadaptels.vectorize import vectorize_from_file
    >>> n = vectorize_from_file('adaptels.tif', 'adaptels.shp')
    >>> print(f"Wrote {n} polygons")
    """
    try:
        import rasterio
    except ImportError as e:
        raise ImportError("rasterio required. Install: conda install -c conda-forge rasterio") from e

    with rasterio.open(str(input_raster)) as src:
        data = src.read(1).astype(np.int32)
        transform = src.transform
        crs_wkt = src.crs.to_wkt()
        if nodata is None:
            nodata = src.nodata if src.nodata is not None else -9999

    return vectorize_adaptels(
        data, transform, crs_wkt, output_path,
        driver=driver, nodata=int(nodata),
        connectivity=connectivity, compute_area=compute_area,
        quiet=quiet
    )
