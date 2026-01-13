from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from src.kalman_filter import apply_kalman_filter


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)

    a = sin(dphi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2.0) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def calculate_features(df):
    # -------------------------------
    # 1. Timestamp handling
    # -------------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # -------------------------------
    # 2. Agent ID safety (CRITICAL)
    # -------------------------------
    if "agent_id" not in df.columns:
        df["agent_id"] = "ID_000"
    else:
        df["agent_id"] = df["agent_id"].astype(str)

    # -------------------------------
    # 3. Sort temporally per agent
    # -------------------------------
    df = df.sort_values(["agent_id", "timestamp"]).reset_index(drop=True)

    # -------------------------------
    # 4. Kalman smoothing
    # -------------------------------
    df = apply_kalman_filter(df)

    # -------------------------------
    # 5. Feature columns
    # -------------------------------
    df["dist_moved_m"] = 0.0
    df["time_delta_s"] = 0.0
    df["speed_m_s"] = 0.0
    df["direction"] = 0.0
    df["angle_change"] = 0.0

    # -------------------------------
    # 6. Motion features per agent
    # -------------------------------
    for agent, idx in df.groupby("agent_id").groups.items():
        idx = list(idx)
        if len(idx) < 2:
            continue

        for i in range(1, len(idx)):
            cur = idx[i]
            prev = idx[i - 1]

            lat1, lon1 = df.at[prev, "lat_kalman"], df.at[prev, "lon_kalman"]
            lat2, lon2 = df.at[cur, "lat_kalman"], df.at[cur, "lon_kalman"]

            dist = haversine(lat1, lon1, lat2, lon2)
            df.at[cur, "dist_moved_m"] = dist

            td = (df.at[cur, "timestamp"] - df.at[prev, "timestamp"]).total_seconds()
            df.at[cur, "time_delta_s"] = td

            if td > 0:
                df.at[cur, "speed_m_s"] = dist / td

            dx = radians(lat2 - lat1)
            dy = radians(lon2 - lon1)
            df.at[cur, "direction"] = atan2(dx, dy) if (dx or dy) else 0.0

        for i in range(1, len(idx)):
            cur = idx[i]
            prev = idx[i - 1]
            diff = df.at[cur, "direction"] - df.at[prev, "direction"]
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            df.at[cur, "angle_change"] = abs(diff)

    # -------------------------------
    # 7. Spatial risk feature
    # -------------------------------
    df["dist_to_border"] = (df["longitude"] - 70.0).abs()

    return df


if __name__ == "__main__":
    raw_path = Path("data/raw/border_data.csv")

    if not raw_path.exists():
        raise FileNotFoundError("Run generate_data.py first")

    df = pd.read_csv(raw_path)
    processed_df = calculate_features(df)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_df.to_csv(out_dir / "featured_border_data.csv", index=False)
    print("Kalman-smoothed features saved")
