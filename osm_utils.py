import requests
 
 
def fetch_osm_boundary_overpass(place_name: str, near_lat: float, near_lon: float,
                                 radius_km: float = 30) -> dict | None:
    """
    Searches Overpass for an administrative/place boundary relation or way
    with a matching name within radius_km of (near_lat, near_lon).
    Returns a GeoJSON Polygon dict, or None if nothing usable was found.
    """
    query = f"""
    [out:json][timeout:25];
    (
      relation["boundary"="administrative"]["name"~"{place_name}",i](around:{radius_km * 1000},{near_lat},{near_lon});
      relation["place"]["name"~"{place_name}",i](around:{radius_km * 1000},{near_lat},{near_lon});
      way["boundary"="administrative"]["name"~"{place_name}",i](around:{radius_km * 1000},{near_lat},{near_lon});
    );
    out geom;
    """
    try:
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=30,
        )
    except requests.RequestException:
        return None
 
    if resp.status_code != 200:
        return None
 
    elements = resp.json().get("elements", [])
    if not elements:
        return None
 
    el = elements[0]
 
    # A single closed "way" is the simple case -- its geometry IS the ring.
    if el["type"] == "way" and "geometry" in el:
        coords = [[pt["lon"], pt["lat"]] for pt in el["geometry"]]
        if len(coords) < 3:
            return None
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        return {"type": "Polygon", "coordinates": [coords]}
 
    # A "relation" is made of member ways -- stitch the outer ring together.
    # (Simplified: assumes the outer members are already contiguous, which
    # covers most single-ring administrative boundaries. Complex multi-part
    # boundaries may need proper ring assembly -- fine for an MVP.)
    if el["type"] == "relation" and "members" in el:
        outer_coords = []
        for member in el["members"]:
            if member.get("role") == "outer" and "geometry" in member:
                outer_coords.extend([[pt["lon"], pt["lat"]] for pt in member["geometry"]])
        if len(outer_coords) >= 3:
            if outer_coords[0] != outer_coords[-1]:
                outer_coords.append(outer_coords[0])
            return {"type": "Polygon", "coordinates": [outer_coords]}
 
    return None
 