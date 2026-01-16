import streamlit as st
import pandas as pd
import numpy as np
import folium
import joblib
from pathlib import Path
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# =========================
# 1. LOAD YOUR TRAINED ML BRAIN
# =========================
@st.cache_resource
def get_model():
    # Load the pipeline we just fixed and pushed to Git
    return joblib.load("models/border_intruder_model.pkl")

pipeline = get_model()

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Elite Border Command", layout="wide", page_icon="🛡️")

# Custom CSS for a "Tactical" look
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4451; }
    </style>
    """, unsafe_allow_html=True)

st.title(" Spacio-Temporal Border Command Center")
st.caption("AI-Powered Intrusion Detection & Tactical Analytics")

# =========================
# 2. DATA GENERATION (Simulating Real Sensors)
# =========================
def generate_tactical_data(n=50):
    # Simulating a specific border coordinate (e.g., Kutch region)
    base_lat, base_lon = 23.8, 68.7
    lats = base_lat + np.random.uniform(-0.1, 0.1, n)
    lons = base_lon + np.random.uniform(-0.1, 0.1, n)
    
    data = pd.DataFrame({
        "track_id": range(1000, 1000 + n),
        "lat": lats,
        "lon": lons,
        "speed": np.random.uniform(2, 45, n),
        "angle_change": np.random.uniform(0, 180, n),
        "sensor_confidence": np.random.uniform(0.7, 1.0, n),
        "object_type": np.random.choice(["Drone", "Human", "Animal", "Vehicle"], n),
        "terrain": np.random.choice(["Plain", "Marshy", "Mountain"], n),
        "visibility": np.random.choice(["Clear", "Foggy", "Night"], n)
    })
    
    
    probs = pipeline.predict_proba(data)[:, 1] 
    data["threat_score"] = probs
    data["is_intruder"] = pipeline.predict(data)
    
    return data

if 'df' not in st.session_state:
    st.session_state.df = generate_tactical_data()

# =========================
# 4. SIDEBAR CONTROLS
# =========================
st.sidebar.header("Tactical Filters")
min_threat = st.sidebar.slider("Threat Filter Threshold", 0.0, 1.0, 0.5)
map_style = st.sidebar.selectbox(" Satellite View", ["Dark Tactical", "Satellite"])
if st.sidebar.button("Refresh Sensor Feed"):
    st.session_state.df = generate_tactical_data()
    st.rerun()
# =========================
# 4.5 APPLY FILTERS 
# =========================
# This creates the variable 'filtered_df' that the map needs!
df = st.session_state.df
filtered_df = df[df['threat_score'] >= min_threat]
# =========================
# =========================
# 3. TOP TIER METRICS (Filtered)
# =========================
# Use filtered_df instead of df so the numbers change with the slider!
display_df = filtered_df 

intruders_count = display_df[display_df['is_intruder'] == 1].shape[0]
avg_threat = display_df['threat_score'].mean() if not display_df.empty else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Active Tracks (Filtered)", len(display_df))
m2.metric("Detected Intruders", intruders_count)
m3.metric("Avg Threat Level", f"{avg_threat:.2%}")
m4.metric("System Status", "ALERTS ACTIVE" if intruders_count > 0 else "SECURE")


# 5. THE MAP (Decision Intelligence)
# =========================
st.subheader("Live Border Surveillance Map")

# A. Define the Border & Warning Zone Coordinates
# These are simulated to cross through your sensor data range (23.8, 68.7)
border_line = [
    [23.75, 68.6], [23.80, 68.75], [23.85, 68.85], [23.90, 68.95]
]
# Offset the warning zone slightly to the left/bottom of the border
warning_zone = [[p[0]-0.008, p[1]-0.008] for p in border_line]

# B. Initialize Map
tiles = "CartoDB dark_matter" if map_style == "Dark Tactical" else "Esri.WorldImagery"
m = folium.Map(location=[23.8, 68.7], zoom_start=11, tiles=tiles)

# C. Draw the Visual Barriers (The Geofences)
# The Warning Zone (Yellow)
folium.PolyLine(
    locations=warning_zone,
    color="#FFFF00",
    weight=4,
    opacity=0.3,
    tooltip="Warning Zone: Entry Detected"
).add_to(m)

# The Main Border (Red)
folium.PolyLine(
    locations=border_line,
    color="#FF4B4B",
    weight=6,
    opacity=0.9,
    dash_array='10, 10',
    tooltip="International Border (RESTRICTED)"
).add_to(m)

# D. Draw the Threat Heatmap
if not filtered_df.empty:
    # Only show heatmap for scores > 0.7 to keep it "clean"
    heat_data = filtered_df[filtered_df['threat_score'] > 0.7][['lat', 'lon']].values.tolist()
    if heat_data:
        HeatMap(heat_data, radius=15, blur=18, min_opacity=0.4).add_to(m)

# E. Plot Every Track (The Dots)
for _, row in filtered_df.iterrows():
    # Use Neon colors for high visibility on dark maps
    color = "#FF0000" if row['is_intruder'] == 1 else "#00FF00"
    
    folium.CircleMarker(
        location=[row.lat, row.lon],
        radius=7,
        color=color,
        fill=True,
        fill_opacity=0.8,
        popup=f"""
        <b>Track ID:</b> {row.track_id}<br>
        <b>Type:</b> {row.object_type}<br>
        <b>Threat Score:</b> {row.threat_score:.2f}<br>
        <b>Status:</b> {'⚠️ INTRUDER' if row['is_intruder'] == 1 else '✅ NORMAL'}
        """
    ).add_to(m)

# F. THE TACTICAL LEGEND (Floating HTML Overlay)
legend_html = '''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 200px; height: 170px; 
     background-color: rgba(14, 17, 23, 0.9); z-index:9999; font-size:14px;
     color: white; border:1px solid #3e4451; border-radius:8px; padding: 12px;
     font-family: sans-serif;">
     <b style="color:#FF4B4B">Tactical Legend</b><br>
     <hr style="margin: 5px 0; border-color: #3e4451;">
     <span style="color:#FF4B4B"><b>--</b></span> Border Line<br>
     <span style="color:#FFFF00"><b>--</b></span> Warning Zone<br>
     <span style="color:#FF0000">●</span> Intruder (Confirmed)<br>
     <span style="color:#00FF00">●</span> Normal Activity<br>
     <span style="color:orange">🔥</span> Threat Density
     </div>
     '''
m.get_root().html.add_child(folium.Element(legend_html))

# G. Render in Streamlit
st_folium(m, height=550, use_container_width=True)

# ==========================================
# 6. FEATURE IMPORTANCE (Explainable AI)
# ==========================================

col_left, col_right = st.columns([1, 1]) 

with col_left:
    st.subheader(" Why is this a Threat?")
    # Use the pipeline to show what features matter most
    rf_model = pipeline.named_steps['classifier']
    imp_df = pd.DataFrame({
        "Feature": ["Speed", "Angle", "Confidence"],
        "Importance": rf_model.feature_importances_[:3] 
    }).sort_values("Importance", ascending=False)
    st.bar_chart(imp_df.set_index("Feature"))

with col_right:
    st.subheader("Top Priority Threats")
    # Show the actual data table for the intruders
    st.dataframe(
        filtered_df[filtered_df['is_intruder'] == 1].sort_values("threat_score", ascending=False),
        use_container_width=True
    )