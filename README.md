# Border Intrusion Pattern Analysis using Spatio-Temporal Machine Learning

## Problem Statement
Border surveillance systems generate large volumes of movement data from sensors,
patrol logs, and monitoring infrastructure. Manual monitoring is inefficient and
reactive. This project aims to analyze spatio-temporal movement data to identify
potentially suspicious patterns and high-risk regions along border areas.

## Objectives
- Analyze spatial and temporal movement patterns
- Classify movement behavior as normal or potentially suspicious
- Identify high-risk border zones and time windows
- Generate risk scores to support early warning and prioritization

## Project Scope
- Uses simulated and publicly available datasets
- Does not rely on classified or sensitive data
- Focuses on pattern analysis and anomaly detection
- Emphasizes ethical and responsible ML usage

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- GeoPandas
- Matplotlib, Seaborn
- Folium
- Streamlit / Flask (for visualization)

## Project Structure
data/ - raw and processed datasets
notebooks/ - exploration and analysis notebooks
src/ - core Python scripts
models/ - trained models
visuals/ - plots and maps
docs/ - project documentation

## Current Operational Features
- **Dynamic Threat Classification**: Real-time classification of objects based on spacio-temporal movement patterns.
- **Click-to-Analyze Synchronization**: Interactive map interface where clicking a target instantly synchronizes the tactical sidebar with precise telemetry.
- **Trajectory Reconstruction**: Visualizes the unique historical "breadcrumb" path of a selected ID to identify movement intent.
- **Tactical Telemetry Dashboard**: High-precision metrics for Speed (m/s) and Angle Change (°) to assist in human-in-the-loop decision making.
- **Intelligence Log & Reporting**: Automated generation of threat summaries and downloadable CSV history for forensic mission debriefing.

## Tactical Decision Logic
The system classifies threats based on verified physical motion thresholds:
- **High Velocity Alert**: Triggered when a target exceeds **10.0 m/s**.
- **Evasive Maneuver Alert**: Triggered by an angular change exceeding **60.0°**.

## Tech Stack
- **Frontend**: Streamlit
- **Mapping**: Folium / Streamlit-Folium
- **ML Engine**: Scikit-Learn (Random Forest Classifier)
- **Explainability**: SHAP (SHapley Additive exPlanations)

## Roadmap (In-Development)
- **Multi-Sensor Data Fusion**: Integrating simulated Radar, Thermal, and Acoustic sensor inputs.
- **Predictive Intercept Point**: project target's future location based on current vector analysis.
- **Dynamic Geofencing**: Automated risk-profiling for high-priority border sectors.


## Current Status
Project initialization and data design phase.

## Author
Student project for learning applied spatio-temporal machine learning.

