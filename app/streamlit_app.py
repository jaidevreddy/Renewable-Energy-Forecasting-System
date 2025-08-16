# app/streamlit_app.py
import os
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import streamlit as st
from shapely.geometry import Point

# ─────────────────────────────
# App config
# ─────────────────────────────
st.set_page_config(page_title="Bengaluru Solar — Phase 7", layout="wide")
st.title("Bengaluru Solar — Suitability & Home Check")

# File paths
GEO_PARQUET  = "data/processed/suitability_solar_geo.parquet"   # must contain: ZoneID, score_0_100, annual_kwh, geometry
CITY_CENTER  = (12.9716, 77.5946)

# ─────────────────────────────
# Helpers
# ─────────────────────────────
@st.cache_data
def load_geo() -> gpd.GeoDataFrame:
    if not os.path.exists(GEO_PARQUET):
        st.error(f"Missing: {GEO_PARQUET}. Re-run 06A & 06B to create it.")
        st.stop()
    gdf = gpd.read_parquet(GEO_PARQUET)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    # sanity columns
    need = {"ZoneID", "annual_kwh", "score_0_100", "geometry"}
    missing = need - set(gdf.columns)
    if missing:
        st.error(f"{GEO_PARQUET} missing columns: {missing}")
        st.stop()
    # normalized annual_kwh for coloring
    ak = gdf["annual_kwh"]
    gdf["annual_kwh_norm"] = (ak - ak.min()) / (ak.max() - ak.min()) if ak.max() > ak.min() else 0.0
    # types
    gdf["ZoneID"] = gdf["ZoneID"].astype(str)
    return gdf

def zone_join_point(gdf: gpd.GeoDataFrame, lat: float, lon: float):
    pt = gpd.GeoDataFrame({"_":[0]}, geometry=[Point(lon, lat)], crs=4326)
    try:
        hit = gpd.sjoin(pt, gdf, how="left", predicate="within")
    except Exception as e:
        st.error(f"Spatial join failed: {e}")
        return None
    if hit.empty:
        return None
    row = hit.iloc[0]
    return row if pd.notna(row.get("ZoneID")) else None

def get_browser_location():
    """
    Returns (lat, lon) if streamlit-js-eval is installed and user allows location.
    Otherwise returns None.
    """
    try:
        from streamlit_js_eval import get_geolocation  # pip install streamlit-js-eval
        loc = get_geolocation()
        if isinstance(loc, dict):
            lat = loc.get("coords", {}).get("latitude")
            lon = loc.get("coords", {}).get("longitude")
            if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                return float(lat), float(lon)
    except Exception:
        pass
    return None

def color_expr(metric_key: str):
    # JS expressions for deck.gl fill color
    if metric_key == "score_0_100":
        # Red → Green with some orange; alpha 160
        return "[255*(1 - score_0_100/100), 180*(score_0_100/100), 60, 160]"
    # annual_kwh normalized 0..1 prepared as annual_kwh_norm
    return "[255*(1 - annual_kwh_norm), 180*(annual_kwh_norm), 60, 160]"

# ─────────────────────────────
# Data
# ─────────────────────────────
gdf = load_geo()

# ─────────────────────────────
# Tabs (only two: Map, Home)
# ─────────────────────────────
tab_map, tab_home = st.tabs(["Suitability Map", "Check My Home"])

# ─────────────────────────────
# Tab 1: Suitability Map
# ─────────────────────────────
with tab_map:
    st.subheader("Solar Suitability Map (223 zones)")

    metric_choice = st.selectbox(
        "Color by",
        options=["score_0_100", "annual_kwh"],
        index=0,
        help="Switch between normalized score (0–100) and model-estimated annual kWh."
    )

    layer = pdk.Layer(
        "GeoJsonLayer",
        data=gdf.__geo_interface__,
        pickable=True,
        stroked=True,
        filled=True,
        get_fill_color=color_expr(metric_choice),
        get_line_color=[60, 60, 60],
        line_width_min_pixels=1,
    )
    tooltip = {"text": "Zone: {ZoneID}\nScore: {score_0_100}\nAnnual kWh: {annual_kwh}"}
    view = pdk.ViewState(latitude=CITY_CENTER[0], longitude=CITY_CENTER[1], zoom=10.5)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip))

    st.markdown("#### Top 10 Zones")
    colname = metric_choice
    cols_to_show = ["ZoneID", "annual_kwh", "score_0_100"]
    top10_df = gdf[cols_to_show].sort_values(colname, ascending=False).head(10).reset_index(drop=True)
    st.dataframe(top10_df)

    # ⬇️ Download full CSV (all zones) sorted by the selected metric
    full_df = gdf[cols_to_show].sort_values(colname, ascending=False).reset_index(drop=True)
    csv_bytes = full_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇️ Download full zones CSV (sorted by {colname})",
        data=csv_bytes,
        file_name=f"bengaluru_solar_{colname}_zones.csv",
        mime="text/csv",
        help="Exports all 223 zones with ZoneID, annual_kwh, score_0_100."
    )

# ─────────────────────────────
# Tab 2: Check My Home (no verdict line)
# ─────────────────────────────
with tab_home:
    st.subheader("Check if my home is good for solar")

    # Try geolocation (if extension installed and user allows)
    geoloc = None
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("📍 Use my location"):
            geoloc = get_browser_location()
            if geoloc is None:
                st.info(
                    "Geolocation unavailable. Install `streamlit-js-eval` and allow location, "
                    "or enter coordinates manually."
                )
    with colB:
        st.caption("Or enter coordinates below:")

    default_lat = geoloc[0] if geoloc else CITY_CENTER[0]
    default_lon = geoloc[1] if geoloc else CITY_CENTER[1]

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        lat = st.number_input("Latitude", value=float(default_lat), format="%.6f")
    with c2:
        lon = st.number_input("Longitude", value=float(default_lon), format="%.6f")
    with c3:
        kw = st.slider("Planned system size (kW)", 1.0, 10.0, 2.0, 0.5)

    if st.button("Check Suitability"):
        row = zone_join_point(gdf, lat, lon)
        if row is None:
            st.warning("Couldn't map your location to a zone. Try nudging the point.")
        else:
            zone = row["ZoneID"]
            score = int(row["score_0_100"]) if pd.notna(row["score_0_100"]) else None
            annual_kwh_per_kw = float(row["annual_kwh"]) if pd.notna(row["annual_kwh"]) else None

            if (score is None) or (annual_kwh_per_kw is None):
                st.error(f"Zone {zone} found, but suitability values missing.")
            else:
                est_annual = kw * annual_kwh_per_kw

                st.success(f"Zone **{zone}**")
                a, b, c = st.columns(3)
                a.metric("Suitability score (0–100)", score)
                b.metric("Est. annual energy (kWh)", f"{est_annual:,.0f}")
                c.metric("System size (kW)", f"{kw:.1f}")
                # No 'Verdict: ...' line here (as requested)
                st.caption("Notes:\n- Score is relative to other Bengaluru zones.\n- Estimated energy scales ~linearly with kW.")

st.markdown("---")
st.caption("Bengaluru Solar • Built with Streamlit + GeoPandas + PyDeck")
