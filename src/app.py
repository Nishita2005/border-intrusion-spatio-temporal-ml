import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from pykalman import KalmanFilter
import joblib
import os
import re
import math

# --- 1. SETTINGS & HARDCODED PATHS ---
st.set_page_config(page_title="Spacio-Temporal Command Center", layout="wide")

# Absolute paths for Windows environment
BASE_DIR = r"C:\Users\HP\OneDrive\Desktop\border-intrusion-spacio-temporal-ml"
MODEL_PATH = os.path.join(BASE_DIR, "models", "border_intruder_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "border_data.csv")

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        st.error(f"Critical Files Missing! Check folder: {BASE_DIR}")
        st.stop()
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    return model, df

model, df = load_assets()

# --- 2. SESSION STATE ---
if "selected_id" not in st.session_state:
    st.session_state.selected_id = 0

# --- 3. SIDEBAR: TACTICAL CONTROLS & UPDATED LEGEND ---
st.sidebar.title("🛡️ Tactical Controls")
current_step = st.sidebar.slider("Timeline Step", 10, len(df), 615)
display_df = df.iloc[:current_step].copy()

# Sync target selection
if st.session_state.selected_id >= len(display_df):
    st.session_state.selected_id = len(display_df) - 1

target_id = st.session_state.selected_id
target_row = display_df.loc[target_id]

# REWRITTEN LEGEND: Matching your specific color scheme
st.sidebar.markdown("### 🗺️ Map Legend")
st.sidebar.info("""
- 🔴 **Red Point**: Raw Sensor Detection
- 🔵 **Solid Cyan Line**: Kalman Filtered History
- ⚪ **Dotted White Line**: Predicted Movement
- 🔥 **Glow Effect**: Threat Density Heatmap
""")

# --- 4. BEHAVIOR & ANALYSIS ---
def classify_behavior(row):
    if row["speed"] > 3.5: return "Evasive"
    if row["speed"] < 0.5: return "Stationary"
    return "Patrol"

display_df["behavior"] = display_df.apply(classify_behavior, axis=1)

st.sidebar.markdown("---")
st.sidebar.subheader(f"🎯 Analysis: Track ID {target_id}")

if target_row["label"] == 1:
    st.sidebar.error(f"⚠️ THREAT: {target_row.get('object_type', 'Intruder')}")
else:
    st.sidebar.success(f"✅ CLEAR: {target_row.get('object_type', 'Neutral')}")

# Intelligence Metrics
c1, c2 = st.sidebar.columns(2)
c1.metric("Speed", f"{target_row['speed']:.2f} m/s")
c2.metric("Angle", f"{target_row['angle_change']:.2f}°")

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Threat Confidence")
st.sidebar.progress(min(target_row["speed"]/10, 1.0), text="Speed Anomaly")
st.sidebar.progress(min(target_row["angle_change"]/100, 1.0), text="Maneuver Anomaly")

# --- 5. MAP ENGINE ---
st.title("🛡️ Spacio-Temporal Command Center")

m = folium.Map(
    location=[target_row["latitude"], target_row["longitude"]],
    zoom_start=14,
    tiles="CartoDB dark_matter"
)

# A. RED ZONE REMOVED (As requested)

# B. HISTORY (Solid Cyan) & PREDICTION (Dotted White)
# Filter track history for the specific object type
track = display_df[
    (display_df['object_type'] == target_row['object_type']) & 
    (display_df.index <= target_id)
].tail(25)

if len(track) > 3:
    obs = track[['latitude', 'longitude']].values
    kf = KalmanFilter(initial_state_mean=obs[0], n_dim_obs=2)
    state_means, _ = kf.smooth(obs)
    
    # 1. DRAW HISTORY: Solid Cyan Line
    folium.PolyLine(
        state_means.tolist(), 
        color="#00FFFF", 
        weight=4, 
        opacity=0.8, 
        tooltip="Historical Path"
    ).add_to(m)
    
    # 2. DRAW PREDICTION: Dotted White Line
    current_pos = state_means[-1]
    # Estimate next position based on velocity and current angle
    prediction_dist = target_row['speed'] * 0.0004 
    rad = math.radians(target_row['angle_change'])
    
    next_pos = [
        current_pos[0] + (prediction_dist * math.cos(rad)),
        current_pos[1] + (prediction_dist * math.sin(rad))
    ]
    
    folium.PolyLine(
        locations=[current_pos.tolist(), next_pos],
        color="white",
        weight=3,
        dash_array='5, 10', # Creates the dotted effect
        opacity=0.9,
        tooltip="Predicted Heading"
    ).add_to(m)

# C. THREAT HEATMAP
heat_data = display_df[display_df['label'] == 1][['latitude', 'longitude']].values.tolist()
if heat_data:
    HeatMap(heat_data, radius=15, blur=10, min_opacity=0.4).add_to(m)

# D. LIVE MARKERS
for i, row in display_df.tail(60).iterrows():
    is_target = (i == target_id)
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=14 if is_target else 6,
        color="red" if row["label"] == 1 else "#00FFFF",
        fill=True,
        fill_opacity=0.9 if is_target else 0.6,
        popup=f"ID: {i}"
    ).add_to(m)

# --- 6. RENDER & INTERACTION ---
output = st_folium(m, width=1100, height=600, key="tactical_map")

# Handle map clicks to switch targets
if output and output.get("last_object_clicked_popup"):
    clicked_id = re.search(r"ID:\s*(\d+)", output["last_object_clicked_popup"])
    if clicked_id:
        st.session_state.selected_id = int(clicked_id.group(1))
        st.rerun()

# --- 7. INTELLIGENCE LOG ---
st.markdown("---")
st.subheader("📋 Active Threat Intelligence Log")
st.table(display_df[display_df["label"] == 1].tail(5)[["object_type", "speed", "behavior"]])