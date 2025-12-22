import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
import shap
import re

# 1. SETUP
st.set_page_config(page_title="Spacio-Temporal Command Center", layout="wide")
st.title("🛡️ Spacio-Temporal Command Center")

@st.cache_resource
def load_assets():
    model = joblib.load('models/border_intruder_model.pkl')
    data = pd.read_csv('data/raw/border_data.csv')
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    return model, data, preprocessor, classifier

model, df, preprocessor, classifier = load_assets()

# 2. SESSION STATE MANAGEMENT
if 'selected_id' not in st.session_state:
    st.session_state.selected_id = 0

# 3. SIDEBAR TACTICAL CONTROLS
st.sidebar.header("Tactical Controls")
current_step = st.sidebar.slider("Timeline Step", 10, len(df), 100)
display_df = df.iloc[0 : current_step]

# SYNC CHECK: Ensure target_id is within the current visible range
target_id = st.session_state.selected_id
if target_id not in display_df.index:
    target_id = display_df.index[-1]

target_row = df.loc[target_id]
is_threat = target_row['label'] == 1

# 4. SIDEBAR ANALYSIS (FORCE SYNCED TO target_id)
st.sidebar.markdown("---")
st.sidebar.subheader(f"🎯 Analysis: ID {target_id}")

if is_threat:
    st.sidebar.error(f"⚠️ THREAT: {target_row['object_type']}")
else:
    st.sidebar.success(f"✅ CLEAR: {target_row['object_type']}")

col1, col2 = st.sidebar.columns(2)
col1.metric("Exact Speed", f"{target_row['speed']:.2f} m/s")
col2.metric("Exact Angle", f"{target_row['angle_change']:.2f}°")

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Tactical Decision Rules")
st.sidebar.caption("Red Alert Thresholds: Speed > 10m/s | Angle > 60°")

if target_row['speed'] > 10:
    st.sidebar.warning(f"Violation: High Velocity")
if target_row['angle_change'] > 60:
    st.sidebar.warning(f"Violation: Evasive Maneuvering")

# 5. MAP WITH DYNAMIC TRACKING
m = folium.Map(location=[target_row['latitude'], target_row['longitude']], zoom_start=12, tiles="CartoDB dark_matter")

# Unique Path History (Cyan line)
specific_track = display_df[
    (display_df['object_type'] == target_row['object_type']) & 
    (display_df.index <= target_id)
].tail(15)

if len(specific_track) > 1:
    folium.PolyLine(
        locations=specific_track[['latitude', 'longitude']].values.tolist(), 
        color="#00FFFF", weight=5, opacity=0.9
    ).add_to(m)

# Markers
for i, row in display_df.tail(60).iterrows():
    color = "red" if row['label'] == 1 else "green"
    radius = 15 if i == target_id else 7 # Selected dot is noticeably larger
    popup_info = f"ID: {i}<br>Speed: {row['speed']:.2f} m/s<br>Angle: {row['angle_change']:.2f}°"
    
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=radius, color=color, fill=True, fill_opacity=0.8,
        popup=folium.Popup(popup_info, max_width=150)
    ).add_to(m)

# 6. CAPTURE CLICK & REFRESH (THE FIX)
# Using a key that changes with BOTH step and ID ensures Streamlit listens for the change
map_key = f"st_folium_step_{current_step}_target_{target_id}"
output = st_folium(m, width=1200, height=500, key=map_key)

if output and output.get("last_object_clicked_popup"):
    try:
        # Robust ID Extraction using Regex to find digits after 'ID:'
        raw_text = output["last_object_clicked_popup"]
        match = re.search(r'ID:\s*(\d+)', raw_text)
        if match:
            new_id = int(match.group(1))
            if new_id != st.session_state.selected_id:
                st.session_state.selected_id = new_id
                st.rerun() 
    except:
        pass

# 7. LIVE RADAR LOG
st.markdown("---")
st.subheader("📋 Live Radar Intelligence Log")
threat_log = display_df[display_df['label'] == 1][['object_type', 'speed', 'angle_change', 'terrain']].tail(8)

if not threat_log.empty:
    threat_log.columns = ['Object Type', 'Speed (m/s)', 'Angle (°)', 'Terrain']
    st.table(threat_log.sort_index(ascending=False))
else:
    st.info("System Normal: No active threats.")