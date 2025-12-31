from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


def haversine(lat1, lon1, lat2, lon2):
    # returns distance in meters between two lat/lon points
    R = 6371000.0
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2.0) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def calculate_features(df):
    # Ensure timestamps are datetimes
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort within each agent
    df = df.sort_values(["agent_id", "timestamp"]).reset_index(drop=True)

    # Prepare columns
    df["dist_moved_m"] = 0.0
    df["time_delta_s"] = 0.0
    df["speed_m_s"] = 0.0
    df["direction"] = 0.0
    df["angle_change"] = 0.0

    # Compute per-agent diffs
    for agent, group_idx in df.groupby("agent_id").groups.items():
        idx = list(group_idx)
        if len(idx) < 2:
            continue
        for i in range(1, len(idx)):
            cur = idx[i]
            prev = idx[i - 1]
            lat1, lon1 = df.at[prev, "latitude"], df.at[prev, "longitude"]
            lat2, lon2 = df.at[cur, "latitude"], df.at[cur, "longitude"]

            dist = haversine(lat1, lon1, lat2, lon2)
            df.at[cur, "dist_moved_m"] = dist

            # time delta
            if "timestamp" in df.columns:
                td = (
                    df.at[cur, "timestamp"] - df.at[prev, "timestamp"]
                ).total_seconds()
                td = td if td > 0 else np.nan
                df.at[cur, "time_delta_s"] = td
                if pd.notna(td) and td > 0:
                    df.at[cur, "speed_m_s"] = dist / td

            # direction (radians)
            dx = radians(lat2 - lat1)
            dy = radians(lon2 - lon1)
            df.at[cur, "direction"] = atan2(dx, dy) if (dx != 0 or dy != 0) else 0.0

        # angle change per agent
        agent_idx = idx
        for i in range(1, len(agent_idx)):
            cur = agent_idx[i]
            prev = agent_idx[i - 1]
            a1 = df.at[prev, "direction"]
            a2 = df.at[cur, "direction"]
            diff = a2 - a1
            # normalize to [-pi, pi]
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            df.at[cur, "angle_change"] = abs(diff)

    # Distance to border (longitude difference)
    df["dist_to_border"] = df["longitude"].sub(70.0).abs()

    return df


if __name__ == "__main__":
    raw_candidates = [
        Path("data/raw/synthetic_border_data.csv"),
        Path("data/raw/border_data.csv"),
    ]
    src_path = None
    for p in raw_candidates:
        if p.exists():
            src_path = p
            break

    if src_path is None:
        print("Error: Raw data file not found! Run generate_data.py first.")
    else:
        df = pd.read_csv(src_path)
        processed_df = calculate_features(df)

        out_dir = Path("data/processed")
        out_dir.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(out_dir / "featured_border_data.csv", index=False)
        print("Success! Saved to data/processed/featured_border_data.csv")
