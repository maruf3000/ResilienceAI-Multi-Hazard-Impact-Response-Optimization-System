import streamlit as st
import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
import tempfile, os
from rasterio.transform import rowcol
from rasterio.warp import reproject, Resampling
import folium
from streamlit_folium import st_folium

st.set_page_config(layout="wide")
st.title("🌍 Multi-Hazard Impact Dashboard")

# =========================
# SESSION STATE
# =========================
if "result" not in st.session_state:
    st.session_state.result = None

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Configuration")

disaster = st.sidebar.selectbox("Select Disaster", ["Flood", "Landslide"])
analysis = st.sidebar.selectbox("Select Analysis", ["Road", "Hospital", "School", "Community"])

st.sidebar.header("📂 Upload Files")

def save_files(files):
    tmp = tempfile.mkdtemp()
    for f in files:
        with open(os.path.join(tmp, f.name), "wb") as out:
            out.write(f.getbuffer())
    return tmp

# Inputs
shape_f = None
if analysis in ["Road", "Hospital", "School"]:
    shape_f = st.sidebar.file_uploader("Upload Infrastructure Shapefile", accept_multiple_files=True)

sens_f = st.sidebar.file_uploader("Sensitivity TIFF")
adapt_f = st.sidebar.file_uploader("Adaptive Capacity TIFF")

flood_f = None
landslide_f = None

if disaster == "Flood":
    flood_f = st.sidebar.file_uploader("Flood TIFF (0/1)")
else:
    landslide_f = st.sidebar.file_uploader("Landslide Points (Shapefile)", accept_multiple_files=True)

pop_f = None
if analysis == "Community":
    pop_f = st.sidebar.file_uploader("Population TIFF")

run = st.sidebar.button("🚀 Run Analysis")

# =========================
# HELPER FUNCTIONS
# =========================
def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-6)

def align_raster(src_array, src_meta, target_meta):
    dst = np.empty((target_meta["height"], target_meta["width"]), dtype=np.float32)

    reproject(
        source=src_array,
        destination=dst,
        src_transform=src_meta["transform"],
        src_crs=src_meta["crs"],
        dst_transform=target_meta["transform"],
        dst_crs=target_meta["crs"],
        resampling=Resampling.nearest
    )
    return dst

# =========================
# RUN ANALYSIS
# =========================
if run:

    with st.spinner("⏳ Processing..."):

        progress = st.progress(0)

        sens_src = rasterio.open(sens_f)
        adapt_src = rasterio.open(adapt_f)

        sens = sens_src.read(1)
        adapt = adapt_src.read(1)

        progress.progress(20)

        # =========================
        # COMMUNITY
        # =========================
        if analysis == "Community":

            pop_src = rasterio.open(pop_f)
            pop = pop_src.read(1)

            if disaster == "Flood":

                flood_src = rasterio.open(flood_f)
                flood = flood_src.read(1)

                # 🔥 FIX: ALIGN RASTER
                flood_aligned = align_raster(
                    flood,
                    flood_src.meta,
                    pop_src.meta
                )

                exposure = flood_aligned * pop

            else:
                # Landslide → distance-based exposure
                ls = gpd.read_file(save_files(landslide_f))
                ls = ls.to_crs(pop_src.crs)

                exposure = np.zeros_like(pop, dtype=float)

                for geom in ls.geometry:
                    row, col = rowcol(pop_src.transform, geom.x, geom.y)
                    if 0 <= row < pop.shape[0] and 0 <= col < pop.shape[1]:
                        exposure[row, col] += 1

            vulnerability = normalize(sens) - normalize(adapt)
            risk = normalize(exposure) + vulnerability
            priority = normalize(risk)

            st.session_state.result = {
                "type": "raster",
                "exposure": exposure,
                "priority": priority,
                "transform": pop_src.transform
            }

            progress.progress(100)

        # =========================
        # INFRASTRUCTURE
        # =========================
        else:

            gdf = gpd.read_file(save_files(shape_f))

            if disaster == "Flood":
                flood_src = rasterio.open(flood_f)
                flood = flood_src.read(1)
                gdf = gdf.to_crs(flood_src.crs)
            else:
                ls = gpd.read_file(save_files(landslide_f))
                gdf = gdf.to_crs(ls.crs)

            exposure = []
            sens_list = []
            adapt_list = []

            for geom in gdf.geometry:

                try:
                    if disaster == "Flood":
                        row, col = rowcol(flood_src.transform, geom.centroid.x, geom.centroid.y)
                        exp = flood[row, col]
                    else:
                        # distance to nearest landslide
                        dist = ls.distance(geom).min()
                        exp = 1 / (dist + 1)

                    row, col = rowcol(sens_src.transform, geom.centroid.x, geom.centroid.y)
                    s = sens[row, col]
                    a = adapt[row, col]

                except:
                    exp, s, a = 0, 0, 0

                exposure.append(exp)
                sens_list.append(s)
                adapt_list.append(a)

            gdf["exposure"] = normalize(np.array(exposure))
            gdf["sensitivity"] = normalize(np.array(sens_list))
            gdf["adaptive_capacity"] = normalize(np.array(adapt_list))

            gdf["risk"] = (
                0.5*gdf["exposure"] +
                0.3*gdf["sensitivity"] -
                0.2*gdf["adaptive_capacity"]
            )

            q1, q2 = gdf["risk"].quantile([0.33, 0.66])

            gdf["priority"] = gdf["risk"].apply(
                lambda x: "High" if x > q2 else "Medium" if x > q1 else "Low"
            )

            st.session_state.result = {"type": "vector", "data": gdf}

            progress.progress(100)

    st.success("✅ Analysis Completed")

# =========================
# DISPLAY
# =========================
if st.session_state.result:

    result = st.session_state.result

    # =========================
    # 📊 SUMMARY SECTION
    # =========================
    st.subheader("📊 Analysis Summary")

    if result["type"] == "vector":
        gdf = result["data"]

        # 🔹 IMPACT SUMMARY
        st.markdown("### 🚨 Impact Summary (Exposure)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Exposure", round(gdf["exposure"].mean(), 3))
        col2.metric("Max Exposure", round(gdf["exposure"].max(), 3))
        col3.metric("Highly Exposed (>0.7)", (gdf["exposure"] > 0.7).sum())

        # 🔹 PRIORITY SUMMARY
        st.markdown("### 🎯 Priority Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("High Priority", (gdf["priority"]=="High").sum())
        col2.metric("Medium Priority", (gdf["priority"]=="Medium").sum())
        col3.metric("Low Priority", (gdf["priority"]=="Low").sum())

    else:
        exposure = result["exposure"]
        priority = result["priority"]

        # 🔹 IMPACT SUMMARY
        st.markdown("### 🚨 Impact Summary (Flooded Population / Exposure)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Affected Population", int(np.sum(exposure)))
        col2.metric("Max Exposure Pixel", round(np.max(exposure), 2))
        col3.metric("Highly Exposed Pixels (>0.7)", int(np.sum(exposure > 0.7)))

        # 🔹 PRIORITY SUMMARY
        st.markdown("### 🎯 Priority Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Max Priority Score", round(np.max(priority), 2))
        col2.metric("Mean Priority Score", round(np.mean(priority), 2))
        col3.metric("High Priority Pixels (>0.7)", int(np.sum(priority > 0.7)))

    # =========================
    # 🗺️ MAPS
    # =========================
    st.subheader("🗺️ Impact Map")

    m1 = folium.Map(location=[22.5, 91.8], zoom_start=9)

    if result["type"] == "vector":
        folium.GeoJson(
            result["data"],
            tooltip=folium.GeoJsonTooltip(fields=["exposure"])
        ).add_to(m1)

    st_folium(m1, width=1000, height=400)

    st.subheader("🗺️ Priority Map")

    m2 = folium.Map(location=[22.5, 91.8], zoom_start=9)

    if result["type"] == "vector":
        color = {"High":"red","Medium":"orange","Low":"green"}

        folium.GeoJson(
            result["data"],
            style_function=lambda x: {
                "color": color[x["properties"]["priority"]],
                "weight": 3
            },
            tooltip=folium.GeoJsonTooltip(fields=["priority"])
        ).add_to(m2)

    st_folium(m2, width=1000, height=400)

    # =========================
    # 📋 TABLE
    # =========================
    st.subheader("📋 Detailed Results Table")

    if result["type"] == "vector":
        st.dataframe(gdf.drop(columns="geometry"))
    else:
        st.info("Raster result: Table view not available (use map + summary)")