import os
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Polygon

BOUNDARY_PLACE = "Bengaluru, India"
CELL_KM = 2.0  # ~2 km squares 
OUT_GEOJSON = "data/processed/zones.geojson"
OUT_PARQUET = "data/processed/zones.parquet"


def fetch_boundary(place: str) -> gpd.GeoDataFrame:
    """
    Fetch administrative boundary polygon for the place from OSM.
    Returns a single-polygon GeoDataFrame in EPSG:4326.
    """
    gdf = ox.geocode_to_gdf(place).set_crs(4326)
    # keep largest polygon if multipolygon
    gdf["__area"] = gdf.to_crs(3857).area
    gdf = gdf.loc[[gdf["__area"].idxmax()]].drop(columns="__area")
    # fix tiny topology issues
    gdf["geometry"] = gdf.buffer(0)
    return gdf


def make_square_grid(boundary_gdf: gpd.GeoDataFrame, cell_km: float) -> gpd.GeoDataFrame:
    """
    Create a square grid (~cell_km x cell_km) clipped to boundary.
    Returns GeoDataFrame in EPSG:4326 with ZoneID + centroid lat/lon.
    """
    boundary_m = boundary_gdf.to_crs(3857)  # meters
    xmin, ymin, xmax, ymax = boundary_m.total_bounds
    step = float(cell_km) * 1000.0

    xs = np.arange(xmin, xmax + step, step)
    ys = np.arange(ymin, ymax + step, step)

    cells = []
    for x in xs[:-1]:
        for y in ys[:-1]:
            cells.append(Polygon([(x, y), (x + step, y), (x + step, y + step), (x, y + step)]))

    grid = gpd.GeoDataFrame(geometry=cells, crs=3857)

    # intersect with boundary to clip to city
    clipped = gpd.overlay(grid, boundary_m[["geometry"]], how="intersection")
    clipped = clipped.explode(index_parts=False, ignore_index=True)

    # drop slivers < 1% of nominal cell area
    cell_area = step * step
    clipped["__area"] = clipped.area
    clipped = clipped[clipped["__area"] >= 0.01 * cell_area].copy()
    clipped = clipped.drop(columns="__area").to_crs(4326)

    # Accurate centroids (compute in projected CRS, then back to WGS84)
    clipped_3857 = clipped.to_crs(3857)
    cent_3857 = clipped_3857.centroid
    cent_wgs84 = gpd.GeoSeries(cent_3857, crs=3857).to_crs(4326)

    clipped["centroid"] = cent_wgs84
    clipped["centroid_lat"] = cent_wgs84.y
    clipped["centroid_lon"] = cent_wgs84.x
    clipped["ZoneID"] = [f"BLR-{i:04d}" for i in range(1, len(clipped) + 1)]

    # final column order
    clipped = clipped[["ZoneID", "geometry", "centroid_lat", "centroid_lon"]]
    return clipped


def add_octant_labels(zones_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add Region8 label per zone based on centroid angle from city center:
    E, NE, N, NW, W, SW, S, SE.
    """
    lat0 = zones_gdf["centroid_lat"].mean()
    lon0 = zones_gdf["centroid_lon"].mean()
    dy = zones_gdf["centroid_lat"] - lat0
    dx = zones_gdf["centroid_lon"] - lon0
    ang = np.degrees(np.arctan2(dy, dx))  # -180..180, 0=East, 90=North

    def to_octant(a: float) -> str:
        if -22.5 <= a < 22.5:
            return "E"
        if 22.5 <= a < 67.5:
            return "NE"
        if 67.5 <= a < 112.5:
            return "N"
        if 112.5 <= a < 157.5:
            return "NW"
        if a >= 157.5 or a < -157.5:
            return "W"
        if -157.5 <= a < -112.5:
            return "SW"
        if -112.5 <= a < -67.5:
            return "S"
        if -67.5 <= a < -22.5:
            return "SE"
        return "NA"

    zones_gdf["Region8"] = [to_octant(a) for a in ang]
    return zones_gdf


def coverage_report(boundary: gpd.GeoDataFrame, zones: gpd.GeoDataFrame) -> dict:
    """
    Simple coverage report of grid vs boundary (in km² and %).
    """
    b_area = float(boundary.to_crs(3857).area.sum())
    z_union = zones.to_crs(3857).dissolve().geometry.iloc[0]
    z_area = float(z_union.area)
    covered = min((z_area / b_area) * 100.0, 100.0)
    return {
        "boundary_km2": round(b_area / 1e6, 2),
        "zones_km2": round(z_area / 1e6, 2),
        "coverage_pct": round(covered, 2),
        "n_zones": int(len(zones)),
        "avg_cell_km2": round((zones.to_crs(3857).area.mean() / 1e6), 3),
    }


def main():
    os.makedirs("data/processed", exist_ok=True)

    print(f"[phase1] fetching boundary for: {BOUNDARY_PLACE}")
    boundary = fetch_boundary(BOUNDARY_PLACE)

    print(f"[phase1] building grid ~{CELL_KM} km cells …")
    zones = make_square_grid(boundary, CELL_KM)

    print("[phase1] tagging regions (Region8: N/NE/E/SE/S/SW/W/NW) …")
    zones = add_octant_labels(zones)

    print(f"[phase1] writing GeoJSON → {OUT_GEOJSON}")
    zones.to_file(OUT_GEOJSON, driver="GeoJSON")

    # parquet (requires pyarrow or fastparquet)
    try:
        import pyarrow  # noqa: F401
        zones.to_parquet(OUT_PARQUET)
        print(f"[phase1] writing Parquet → {OUT_PARQUET}")
    except Exception as e:
        print(f"[phase1] parquet not written (install pyarrow). Reason: {e}")

    rpt = coverage_report(boundary, zones)
    print("[phase1] Coverage:", rpt)


if __name__ == "__main__":
    main()
