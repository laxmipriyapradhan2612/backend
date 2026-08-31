"""
gee_utils.py
------------
All Google Earth Engine logic for Verdant's satellite land analysis.

This module:
  1. Authenticates/initializes Earth Engine with a service account.
  2. Pulls a cloud-masked Sentinel-2 SR composite over an area of interest (AOI).
  3. Computes NDVI (vegetation) and NDWI (water) indices.
  4. Produces a browser-renderable tile URL for the NDVI layer (colored to match
     the existing Verdant legend: red=stressed, yellow=moderate, green=healthy).
  5. Computes zonal statistics (mean NDVI) and a healthy/moderate/stressed
     pixel-area breakdown for the AOI.

Nothing here touches the frontend directly -- main.py exposes it over HTTP.
"""

import datetime
import ee


# ---------------------------------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------------------------------

def initialize_ee(project_id: str, service_account: str | None = None,
                   key_path: str | None = None) -> None:
    """
    Initialize the Earth Engine session.

    Two supported auth modes:
      - Service account (recommended for a server): pass service_account +
        key_path (path to the downloaded JSON key).
      - Already-authenticated local credentials (e.g. you ran
        `earthengine authenticate` once on this machine): leave both None.
    """
    if service_account and key_path:
        credentials = ee.ServiceAccountCredentials(service_account, key_path)
        ee.Initialize(credentials, project=project_id)
    else:
        ee.Initialize(project=project_id)


# ---------------------------------------------------------------------------
# CLOUD MASKING
# ---------------------------------------------------------------------------

def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """Mask clouds/cirrus using the Sentinel-2 QA60 band, scale reflectance."""
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = (
        qa.bitwiseAnd(cloud_bit_mask).eq(0)
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )
    return image.updateMask(mask).divide(10000).copyProperties(
        image, ["system:time_start"]
    )


# ---------------------------------------------------------------------------
# COMPOSITE + INDICES
# ---------------------------------------------------------------------------

def build_aoi(lat: float, lon: float, buffer_km: float) -> ee.Geometry:
    point = ee.Geometry.Point([lon, lat])
    return point.buffer(buffer_km * 1000).bounds()


def geojson_to_ee_geometry(geojson: dict) -> ee.Geometry:
    """Convert a GeoJSON Polygon/MultiPolygon (e.g. from Nominatim) into an ee.Geometry."""
    return ee.Geometry(geojson)


def get_sentinel_composite(lat: float, lon: float, buffer_km: float = 15,
                            days_back: int = 30, aoi_geometry: ee.Geometry | None = None):
    """
    Returns (composite, ndvi, ndwi, aoi, image_count) for the given area.

    If aoi_geometry is provided (e.g. a real city boundary polygon), the
    composite is clipped to that exact shape instead of a rectangular
    buffer -- this is what keeps the colored output from spilling past the
    actual city border.
    """
    aoi = aoi_geometry if aoi_geometry is not None else build_aoi(lat, lon, buffer_km)
    end = ee.Date(datetime.datetime.utcnow().isoformat())

    def collection_for_window(days):
        start = end.advance(-days, "day")
        return (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
            .map(mask_s2_clouds)
        )

    collection = collection_for_window(days_back)
    count = collection.size().getInfo()

    # Widen the window automatically if nothing usable was found.
    if count == 0:
        collection = collection_for_window(days_back * 3)
        count = collection.size().getInfo()

    composite = collection.median().clip(aoi)
    ndvi = composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = composite.normalizedDifference(["B3", "B8"]).rename("NDWI")

    return composite, ndvi, ndwi, aoi, count


# ---------------------------------------------------------------------------
# TILE URL FOR LEAFLET
# ---------------------------------------------------------------------------

def get_ndvi_tile_url(ndvi: ee.Image) -> str:
    """
    Dense multi-stop palette so the layer reads as a real classified raster
    (like a QGIS-style vegetation-index render) rather than 3 flat blocks:
      bare/stressed -> red/orange -> transitioning -> yellow ->
      moderate -> yellow-green -> healthy -> green -> dense canopy -> dark green
    """
    vis_params = {
        "min": 0.0,
        "max": 0.85,
        "palette": [
            "#8B0000", "#D7301F", "#FC8D59", "#FDCC8A",
            "#FFFFBF", "#D9EF8B", "#A6D96A", "#66BD63",
            "#1A9850", "#006837",
        ],
    }
    map_id = ndvi.getMapId(vis_params)
    return map_id["tile_fetcher"].url_format


def get_true_color_tile_url(composite: ee.Image) -> str:
    """Optional natural-color Sentinel-2 tile (B4/B3/B2), useful as a base layer."""
    vis_params = {"bands": ["B4", "B3", "B2"], "min": 0.0, "max": 0.3}
    map_id = composite.getMapId(vis_params)
    return map_id["tile_fetcher"].url_format


# ---------------------------------------------------------------------------
# ZONAL STATISTICS
# ---------------------------------------------------------------------------

def zonal_ndvi_stats(ndvi: ee.Image, aoi: ee.Geometry, scale: int = 20) -> dict:
    reducer = ee.Reducer.mean().combine(ee.Reducer.minMax(), sharedInputs=True)
    stats = ndvi.reduceRegion(reducer=reducer, geometry=aoi, scale=scale, maxPixels=1e9)
    result = stats.getInfo()
    return {
        "ndvi_mean": result.get("NDVI_mean"),
        "ndvi_min": result.get("NDVI_min"),
        "ndvi_max": result.get("NDVI_max"),
    }


def classify_zone(ndvi_val: float | None, ndwi_val: float | None) -> tuple[str, str]:
    """
    Rule-based classification for a single zone, combining vegetation health
    (NDVI) with a moisture proxy (NDWI). Colors match the existing map legend.
    """
    if ndvi_val is None:
        return "no_data", "#999999"

    if ndvi_val > 0.5:
        classification, color = "healthy", "#65c969"
    elif ndvi_val > 0.3:
        classification, color = "moderate", "#e8da5e"
    else:
        classification, color = "stressed", "#f95d6a"

    # A vegetated zone that's clearly moisture-deficient gets downgraded a level.
    if ndwi_val is not None and ndwi_val < -0.3 and classification == "healthy":
        classification, color = "moderate", "#e8da5e"

    return classification, color


def get_zone_grid_stats(lat: float, lon: float, buffer_km: float = 10,
                         days_back: int = 30, grid_size: int = 2):
    """
    Splits the AOI into a grid_size x grid_size grid (default 2x2 = 4 zones),
    computes mean NDVI + NDWI per cell in a single Earth Engine call, and
    classifies each cell. Returns (geojson_feature_collection, image_count).
    """
    aoi = build_aoi(lat, lon, buffer_km)
    bounds_coords = aoi.bounds().getInfo()["coordinates"][0]
    lons = [c[0] for c in bounds_coords]
    lats = [c[1] for c in bounds_coords]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    lon_step = (max_lon - min_lon) / grid_size
    lat_step = (max_lat - min_lat) / grid_size

    _, ndvi, ndwi, _, image_count = get_sentinel_composite(lat, lon, buffer_km, days_back)
    combined = ndvi.addBands(ndwi)

    cells = []
    zone_num = 1
    for row in range(grid_size):
        for col in range(grid_size):
            cell_min_lon = min_lon + col * lon_step
            cell_max_lon = min_lon + (col + 1) * lon_step
            cell_min_lat = min_lat + row * lat_step
            cell_max_lat = min_lat + (row + 1) * lat_step
            geom = ee.Geometry.Rectangle(
                [cell_min_lon, cell_min_lat, cell_max_lon, cell_max_lat]
            )
            cells.append(ee.Feature(geom, {"zone_id": zone_num}))
            zone_num += 1

    fc = ee.FeatureCollection(cells)
    # One network round-trip covers every zone.
    stats_fc = combined.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=20)
    result = stats_fc.getInfo()

    features = []
    for f in result["features"]:
        props = f["properties"]
        ndvi_val = props.get("NDVI")
        ndwi_val = props.get("NDWI")
        classification, color = classify_zone(ndvi_val, ndwi_val)
        features.append({
            "type": "Feature",
            "geometry": f["geometry"],
            "properties": {
                "zone": f"Zone {props['zone_id']}",
                "ndvi_mean": round(ndvi_val, 3) if ndvi_val is not None else None,
                "ndwi_mean": round(ndwi_val, 3) if ndwi_val is not None else None,
                "classification": classification,
                "color": color,
            },
        })

    return {"type": "FeatureCollection", "features": features}, image_count


def classify_health_breakdown(ndvi: ee.Image, aoi: ee.Geometry, scale: int = 20) -> dict:
    """
    Percent of vegetated pixel area falling into healthy / moderate / stressed
    bands, matching the map legend thresholds.
    """
    pixel_area = ee.Image.pixelArea()

    healthy_mask = ndvi.gt(0.6)
    moderate_mask = ndvi.gt(0.3).And(ndvi.lte(0.6))
    stressed_mask = ndvi.lte(0.3)

    combined = ee.Image.cat([
        pixel_area.updateMask(healthy_mask).rename("healthy"),
        pixel_area.updateMask(moderate_mask).rename("moderate"),
        pixel_area.updateMask(stressed_mask).rename("stressed"),
        pixel_area.rename("total"),
    ])

    sums = combined.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=scale, maxPixels=1e9
    ).getInfo()

    total = sums.get("total") or 1.0
    return {
        "healthy_pct": 100.0 * (sums.get("healthy") or 0.0) / total,
        "moderate_pct": 100.0 * (sums.get("moderate") or 0.0) / total,
        "stressed_pct": 100.0 * (sums.get("stressed") or 0.0) / total,
    }