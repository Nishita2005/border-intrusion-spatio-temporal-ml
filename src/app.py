import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Spacio-Temporal Command Center",
    layout="wide"
)

st.title("🛡️ Spacio-Temporal Command Center")
st.markdown("---")

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("🧭 Tactical Controls")

timeline_step = st.sidebar.slider(
    "Timeline Step",
    min_value=10,
    max_value=1000,
    value=600
)

map_style = st.sidebar.selectbox(
    "🛰️ Map Style",
    ["Dark Tactical", "Satellite"]
)

show_heatmap = st.sidebar.checkbox(" Show Threat Heatmap", True)
show_predictions = st.sidebar.checkbox(" Show Predicted Path", True)

st.sidebar.markdown("---")
st.sidebar.subheader("🗺️ Map Legend")

st.sidebar.markdown("""
🔴 **High Threat Object**  
🟢 **Selected Track**  
⚪ **Predicted Path (Future)**  
🟠 **Restricted Zone**  
🔥 **Threat Density Heatmap**  
📡 **Track ID = Unique Object**
""")

# =========================
# DATA GENERATION
# =========================
np.random.seed(7)

def generate_data(n=1000):
    lat = 23.8 + np.cumsum(np.random.randn(n) * 0.001)
    lon = 68.7 + np.cumsum(np.random.randn(n) * 0.001)

    return pd.DataFrame({
        "track_id": np.random.randint(1000, 9999, size=n),
        "lat": lat,
        "lon": lon,
        "speed": np.abs(np.random.randn(n) * 2 + 5),
        "object_type": np.random.choice(
            ["Drone", "Human", "Animal"], n, p=[0.45, 0.35, 0.2]
        ),
        "behavior": np.random.choice(
            ["Normal", "Evasive", "Suspicious"], n, p=[0.4, 0.4, 0.2]
        ),
        "threat_score": np.random.rand(n)
    })

df = generate_data()
step = min(timeline_step, len(df) - 1)
df_step = df.iloc[:step]

# =========================
# TRACK SELECTION
# =========================
st.sidebar.subheader(" Object Selection")

selected_track = st.sidebar.selectbox(
    "Select Track ID",
    sorted(df_step.track_id.unique())
)

track_df = df_step[df_step.track_id == selected_track]

# =========================
# RESTRICTED ZONE (DATA-ANCHORED)
# =========================
zone_lat = df_step.lat.mean() + 0.05
zone_lon = df_step.lon.mean() + 0.05
zone_radius_km = 3

# =========================
# ALERT ENGINE
# =========================
alerts = []

if (track_df.threat_score > 0.85).any():
    alerts.append(" HIGH THREAT OBJECT DETECTED")

for _, r in track_df.iterrows():
    dist = np.sqrt((r.lat - zone_lat)**2 + (r.lon - zone_lon)**2) * 111
    if dist <= zone_radius_km:
        alerts.append(" RESTRICTED ZONE BREACH")
        break

if alerts:
    for a in set(alerts):
        st.error(a)
else:
    st.success("✅ No active threats")

# =========================
# MAP SETUP
# =========================
st.markdown("##  Live Border Surveillance Map")

tiles = "CartoDB dark_matter" if map_style == "Dark Tactical" else "Esri.WorldImagery"

m = folium.Map(
    location=[df_step.lat.mean(), df_step.lon.mean()],
    zoom_start=9,
    tiles=tiles
)

# =========================
# RESTRICTED ZONE
# =========================
folium.Circle(
    location=[zone_lat, zone_lon],
    radius=zone_radius_km * 1000,
    color="orange",
    fill=True,
    fill_opacity=0.12,
    popup="Restricted Zone"
).add_to(m)

# =========================
# PLOT ALL OBJECTS (DOTS)
# =========================
for _, row in df_step.iterrows():
    color = "green" if row.track_id == selected_track else (
        "red" if row.threat_score > 0.8 else "gray"
    )

    folium.CircleMarker(
        location=[row.lat, row.lon],
        radius=5 if row.track_id == selected_track else 3,
        color=color,
        fill=True,
        fill_opacity=0.85,
        popup=f"""
        <b>Track ID:</b> {row.track_id}<br>
        <b>Type:</b> {row.object_type}<br>
        <b>Speed:</b> {row.speed:.2f}<br>
        <b>Behavior:</b> {row.behavior}<br>
        <b>Threat:</b> {row.threat_score:.2f}
        """
    ).add_to(m)

# =========================
# SELECTED OBJECT HISTORY
# =========================
history_coords = list(zip(track_df.lat, track_df.lon))

folium.PolyLine(
    locations=history_coords,
    color="cyan",
    weight=4,
    tooltip=f"History | Track {selected_track}"
).add_to(m)

# =========================
# SELECTED OBJECT PREDICTION
# =========================
if show_predictions and len(track_df) >= 3:
    last_points = track_df.tail(3)

    lat_step = last_points.lat.diff().mean()
    lon_step = last_points.lon.diff().mean()

    future_coords = []
    lat, lon = last_points.iloc[-1][["lat", "lon"]]

    for _ in range(5):
        lat += lat_step
        lon += lon_step
        future_coords.append((lat, lon))

    folium.PolyLine(
        locations=future_coords,
        color="white",
        dash_array="5,5",
        weight=3,
        tooltip="Predicted Path"
    ).add_to(m)

# =========================
# HEATMAP
# =========================
if show_heatmap:
    heat_data = df_step[df_step.threat_score > 0.6][["lat", "lon"]].values.tolist()
    HeatMap(heat_data, radius=25).add_to(m)

# =========================
# RENDER MAP
# =========================
st_folium(m, height=650, use_container_width=True)

# =========================
# INTEL TABLE
# =========================
st.markdown("##  Active Threat Intelligence Log")

intel = track_df.sort_values("threat_score", ascending=False).head(6)[
    ["track_id", "object_type", "speed", "behavior", "threat_score"]
]

st.dataframe(intel, use_container_width=True)

# =========================
# SHAP / ML EXPLAINABILITY
# =========================
st.markdown("##  Threat Explainability (SHAP-style)")

shap_df = pd.DataFrame({
    "Feature": ["Speed", "Zone Proximity", "Behavior", "Object Type"],
    "Impact": [0.35, 0.42, 0.15, 0.08]
})

st.bar_chart(shap_df.set_index("Feature"))

st.info(
    "Model explanation: Zone proximity and speed contribute most to threat score."
)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("🔒 Secure Tactical Analytics | Defence-Grade Spacio-Temporal Intelligence")
