🛡️ BorderShield: Spatio-Temporal Intrusion Analysis
BorderShield is an AI-powered surveillance framework designed to transform raw movement data into actionable intelligence. 
By leveraging spatio-temporal machine learning, the system identifies suspicious patterns, reconstructs trajectories, and quantifies risk in real-time.

Project Overview 
Manual border monitoring is reactive and prone to fatigue. BorderShield automates the detection of anomalies by analyzing the "physics of movement." The system doesn't just see where an object is; it understands how it is moving to predict intent.

Core Capabilities:-
Dynamic Threat Classification: Real-time categorization of movement using Random Forest classifiers.
Trajectory Reconstruction: Visualizes "breadcrumb" paths to identify loitering, direct approach, or evasive maneuvers.
Tactical Telemetry: Instant calculation of velocity (m/s) and angular deviation (degrees).
Explainable AI (XAI): Uses SHAP to explain why a specific target was flagged as a high-risk threat.

Technical Architecture 
Tactical Decision Logic
The system employs a multi-stage heuristic and ML pipeline to flag threats:
Velocity Threshold: Targets exceeding 10.0(m/s) are flagged for rapid response.
Evasive Maneuver Logic: Sudden angular changes > 60 degrees trigger "Evasive" alerts (indicative of path-finding or patrol avoidance).
Spatio-Temporal Features: Features include distance_to_border, acceleration, and bearing_drift.

Tech Stack
LayerTechnology
Frontend/UI- Streamlit, Folium 
Data Processing- GeoPandas, NumPy, Pandas
Machine Learning- Scikit-Learn (Random Forest), SHAP
Visualization- Matplotlib, Seaborn
DevOps- Pytest, Pre-commit, GitHub Actions

Getting Started
Prerequisites
Python 3.9+ ,Virtual Environment (recommended)
Installation 
1. Clone & SetupBashgit clone https://github.com/Nishita2005/border-intrusion-analysis.git
cd border-intrusion-analysis
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
2. Install Dependencies
Bashpip install -r requirements.txt
pip install -r dev-requirements.txt
3. Initialize Environment
Bashpre-commit install
pytest -q

Roadmap & Evolution
Phase 1: Core ML Engine & Static Visualization. 
Phase 2: Interactive Tactical Dashboard (Streamlit Integration).
Phase 3: Multi-Sensor Fusion (Simulated Radar & Acoustic data).
Phase 4: Predictive Intercept Modeling (Vector-based future-positioning).
Phase 5: Dynamic Geofencing (Sector-based risk weighting).

 Project Structure
├── data/           # Simulated & Public datasets
├── docs/           # Technical specs & documentation
├── models/         # Serialized (.pkl) ML models
├── notebooks/      # EDA & Model Prototyping
├── src/            # Core logic (Feature Eng, Threat Logic)
├── visuals/        # Exported tactical maps & plots
└── tests/          # Unit tests for telemetry logic

Ethical Usage
This project is developed for educational purposes using simulated and public datasets. It adheres to ethical AI principles, emphasizing transparency (via SHAP) and human-in-the-loop decision-making. No sensitive or classified data is utilized.Author: [Nishita Pandey/Nishita2005]Student Project - Applied Spatio-Temporal ML
