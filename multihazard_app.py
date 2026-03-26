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
from folium.plugins import MarkerCluster

st.set_page_config(layout="wide")
st.title("🌍 Multi-Hazard Impact Dashboard (Optimized)")

# =========================
# SESSION STATE
# =========================
if "result" not in st.session_state:
    st.session_state.result = None
if "files_uploaded" not in st.session_state:
    st.session_state.files_uploaded = {}

# =========================
# SIDEBAR CONFIG
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

# =========================
# UPLOAD FILES
# =========================
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
upzila_f = None
if analysis == "Community":
    pop_f = st.sidebar.file_uploader("Population TIFF")
    upzila_f = st.sidebar.file_uploader("Upazila Shapefile (Optional)")

run = st.sidebar.button("🚀 Run Analysis")

# =========================
# HELPER FUNCTIONS
# =========================
@st.cache_data
def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-6)

@st.cache_data
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

@st.cache_data
def read_raster(file):
    src = rasterio.open(file)
    arr = src.read(1).astype(np.float32)
    return arr, src.meta

@st.cache_data
def read_vector(files):
    tmp = save_files(files)
    gdf = gpd.read_file(tmp)
    return gdf

# =========================
# RUN ANALYSIS
# =========================
if run:
    st.session_state.result = None
    with st.spinner("⏳ Running Analysis..."):
        progress = st.progress(0)

        # =========================
        # Load Rasters
        # =========================
        sens, sens_meta = read_raster(sens_f)
        adapt, adapt_meta = read_raster(adapt_f)
        progress.progress(20)

        # =========================
        # COMMUNITY ANALYSIS
        # =========================
        if analysis == "Community":
            pop, pop_meta = read_raster(pop_f)

            if disaster == "Flood":
                flood, flood_meta = read_raster(flood_f)
                flood_aligned = align_raster(flood, flood_meta, pop_meta)
                exposure = flood_aligned * pop
            else:
                # Landslide points → distance-based exposure
                ls = read_vector(landslide_f).to_crs(pop_meta["crs"])
                exposure = np.zeros_like(pop, dtype=float)
                for geom in ls.geometry:
                    row, col = rowcol(pop_meta["transform"], geom.x, geom.y)
                    if 0 <= row < pop.shape[0] and 0 <= col < pop.shape[1]:
                        exposure[row, col] += 1

            vulnerability = normalize(sens) - normalize(adapt)
            risk = normalize(exposure) + vulnerability
            priority = normalize(risk)

            st.session_state.result = {
                "type": "raster",
                "exposure": exposure,
                "priority": priority,
                "transform": pop_meta["transform"]
            }
            progress.progress(100)

        # =========================
        # INFRASTRUCTURE ANALYSIS
        # =========================
        else:
            gdf = read_vector(shape_f)

            if disaster == "Flood":
                flood, flood_meta = read_raster(flood_f)
                gdf = gdf.to_crs(flood_meta["crs"])
            else:
                ls = read_vector(landslide_f)
                gdf = gdf.to_crs(ls.crs)

            exposure_list, sens_list, adapt_list = [], [], []

            for geom in gdf.geometry:
                try:
                    if disaster == "Flood":
                        row, col = rowcol(flood_meta["transform"], geom.centroid.x, geom.centroid.y)
                        exp = flood[row, col]
                    else:
                        dist = ls.distance(geom).min()
                        exp = 1 / (dist + 1)
                    row, col = rowcol(sens_meta["transform"], geom.centroid.x, geom.centroid.y)
                    s = sens[row, col]
                    a = adapt[row, col]
                except:
                    exp, s, a = 0, 0, 0
                exposure_list.append(exp)
                sens_list.append(s)
                adapt_list.append(a)

            gdf["exposure"] = normalize(np.array(exposure_list))
            gdf["sensitivity"] = normalize(np.array(sens_list))
            gdf["adaptive_capacity"] = normalize(np.array(adapt_list))

            gdf["risk"] = 0.5*gdf["exposure"] + 0.3*gdf["sensitivity"] - 0.2*gdf["adaptive_capacity"]
            q1, q2 = gdf["risk"].quantile([0.33, 0.66])
            gdf["priority"] = gdf["risk"].apply(lambda x: "High" if x>q2 else "Medium" if x>q1 else "Low")

            st.session_state.result = {"type": "vector", "data": gdf}
            progress.progress(100)

    st.success("✅ Analysis Completed")

# =========================
# DISPLAY RESULTS
# =========================
if st.session_state.result:
    result = st.session_state.result

    # ===== SUMMARY =====
    st.subheader("📊 Summary")
    if result["type"] == "vector":
        gdf = result["data"]
        col1, col2, col3 = st.columns(3)
        col1.metric("High Priority", (gdf["priority"]=="High").sum())
        col2.metric("Medium", (gdf["priority"]=="Medium").sum())
        col3.metric("Low", (gdf["priority"]=="Low").sum())
    else:
        total = int(np.sum(result["exposure"]))
        st.metric("Affected Population", total)

    # ===== IMPACT MAP =====
    st.subheader("🗺️ Impact Map")
    m1 = folium.Map(location=[22.5,91.8], zoom_start=9)
    if result["type"] == "vector":
        if analysis in ["Hospital","School"]:
            marker_cluster = MarkerCluster().add_to(m1)
            for idx, row in gdf.iterrows():
                folium.Marker([row.geometry.y, row.geometry.x]).add_to(marker_cluster)
        else:
            folium.GeoJson(gdf).add_to(m1)
    st_folium(m1, width=1000, height=400)

    # ===== PRIORITY MAP =====
    st.subheader("🗺️ Priority Map")
    m2 = folium.Map(location=[22.5,91.8], zoom_start=9)
    if result["type"] == "vector":
        color = {"High":"red","Medium":"orange","Low":"green"}
        folium.GeoJson(gdf, style_function=lambda x: {"color":color[x["properties"]["priority"]]}).add_to(m2)
    st_folium(m2, width=1000, height=400)

    # ===== TABLE & DOWNLOAD =====
    st.subheader("📋 Data Table & Download")
    if result["type"] == "vector":
        st.dataframe(gdf.drop(columns="geometry"))
        csv = gdf.drop(columns="geometry").to_csv(index=False)
        st.download_button("⬇️ Download CSV", csv, "result.csv", "text/csv")
    else:
        # raster → flatten into table
        df = pd.DataFrame({"exposure": result["exposure"].flatten(), "priority": result["priority"].flatten()})
        st.dataframe(df.head(50))
        csv = df.to_csv(index=False)
        st.download_button("⬇️ Download CSV", csv, "result.csv", "text/csv")
