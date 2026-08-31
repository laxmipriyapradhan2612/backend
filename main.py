import os
from typing import Any, Optional
 
import ee
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
 
from gee_utils import (
    build_aoi,
    classify_health_breakdown,
    geojson_to_ee_geometry,
    get_ndvi_tile_url,
    get_sentinel_composite,
    get_zone_grid_stats,
    initialize_ee,
    zonal_ndvi_stats,
)
from osm_utils import fetch_osm_boundary_overpass
 
app = FastAPI(title="Verdant Satellite Analysis API")
 
# Wide open for local dev. Lock this down to your actual frontend origin
# (e.g. "http://localhost:8000" or your deployed domain) before going live.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
GEE_PROJECT_ID = os.environ.get("GEE_PROJECT_ID", "")
GEE_SERVICE_ACCOUNT = os.environ.get("GEE_SERVICE_ACCOUNT")  # optional
GEE_KEY_PATH = os.environ.get("GEE_KEY_PATH")  # optional
 
 
@app.on_event("startup")
def startup() -> None:
    if not GEE_PROJECT_ID:
        # Fail loudly rather than silently returning fake data later.
        raise RuntimeError(
            "GEE_PROJECT_ID env var is not set. Create a Google Cloud project, "
            "enable the Earth Engine API for it, and set GEE_PROJECT_ID before "
            "starting the server. See README.md."
        )
    initialize_ee(GEE_PROJECT_ID, GEE_SERVICE_ACCOUNT, GEE_KEY_PATH)
 
 
@app.get("/api/geocode")
def geocode(city: str = Query(...), state: str = Query(""), country: str = "India"):
    """Resolve a city/state name to coordinates via OpenStreetMap Nominatim."""
    query = f"{city}, {state}, {country}" if state else f"{city}, {country}"
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        # polygon_geojson=1 asks Nominatim for the actual admin boundary shape,
        # not just a center point, when the match is a city/town/etc.
        # limit=5 so we can skip past point-only matches to a polygon one.
        params={"q": query, "format": "json", "limit": 5, "polygon_geojson": 1},
        headers={"User-Agent": "verdant-agri-app/1.0 (contact: info@verdant.com)"},
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise HTTPException(status_code=404, detail=f"Could not locate '{query}'")
 
    # Prefer the first result that already has a real polygon; otherwise
    # just take the top result's point and try to find a boundary separately.
    r = results[0]
    boundary = None
    for candidate in results:
        geojson = candidate.get("geojson")
        if geojson and geojson.get("type") in ("Polygon", "MultiPolygon"):
            r = candidate
            boundary = geojson
            break
 
    # Nominatim's plain search often has no polygon for sub-city localities
    # (e.g. "Dwarka" inside Delhi). Try Overpass as a fallback before giving up.
    if boundary is None:
        boundary = fetch_osm_boundary_overpass(city, float(r["lat"]), float(r["lon"]))
 
    return {
        "lat": float(r["lat"]),
        "lon": float(r["lon"]),
        "display_name": r["display_name"],
        "bbox": [float(x) for x in r["boundingbox"]],  # [south, north, west, east]
        "boundary_geojson": boundary,  # None if nothing usable was found anywhere
    }
 
 
class AnalyzeRequest(BaseModel):
    lat: float
    lon: float
    buffer_km: float = 10
    days_back: int = 30
    boundary_geojson: Optional[dict[str, Any]] = None  # real city polygon, if we have one
 
 
@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    """
    Live Sentinel-2 land analysis. If boundary_geojson is provided (a real
    city polygon from /api/geocode), the NDVI raster is clipped to that exact
    shape -- color won't spill past the actual city border. Otherwise falls
    back to a circular buffer around (lat, lon).
    """
    aoi_geometry = None
    if req.boundary_geojson:
        try:
            aoi_geometry = geojson_to_ee_geometry(req.boundary_geojson)
        except Exception:
            aoi_geometry = None  # bad/unsupported geometry -> fall back to buffer
 
    try:
        composite, ndvi, ndwi, aoi, image_count = get_sentinel_composite(
            req.lat, req.lon, buffer_km=req.buffer_km, days_back=req.days_back,
            aoi_geometry=aoi_geometry,
        )
        tile_url = get_ndvi_tile_url(ndvi)
        stats = zonal_ndvi_stats(ndvi, aoi)
        health = classify_health_breakdown(ndvi, aoi)
        bounds = aoi.bounds().getInfo()
    except ee.EEException as exc:
        raise HTTPException(status_code=502, detail=f"Earth Engine error: {exc}")
 
    return {
        "tile_url": tile_url,
        "image_count": image_count,
        "stats": stats,
        "health_breakdown": health,
        "aoi_bounds": bounds,
        "center": {"lat": req.lat, "lon": req.lon},
        "buffer_km": req.buffer_km,
        "clipped_to_boundary": aoi_geometry is not None,
    }
 
 
@app.get("/api/zones")
def zones(
    lat: float = Query(...),
    lon: float = Query(...),
    buffer_km: float = Query(10, gt=0, le=100),
    days_back: int = Query(30, gt=0, le=365),
    grid_size: int = Query(2, ge=1, le=6),  # 2 = 2x2 = 4 zones
):
    """
    Splits the area into grid_size x grid_size zones and returns a GeoJSON
    FeatureCollection, one polygon per zone, each carrying NDVI/NDWI means
    and a healthy/moderate/stressed classification + display color.
    """
    try:
        geojson, image_count = get_zone_grid_stats(
            lat, lon, buffer_km=buffer_km, days_back=days_back, grid_size=grid_size
        )
    except ee.EEException as exc:
        raise HTTPException(status_code=502, detail=f"Earth Engine error: {exc}")
 
    return {"geojson": geojson, "image_count": image_count}
 
 
@app.get("/health")
def health_check():
    return {"status": "ok"}
 