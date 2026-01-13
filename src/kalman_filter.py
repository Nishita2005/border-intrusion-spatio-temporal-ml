import numpy as np
import pandas as pd


def apply_kalman_filter(df):
    """
    Applies Kalman filter to latitude & longitude per agent.
    Assumes constant velocity model.
    """

    df = df.copy()
    df["lat_kalman"] = df["latitude"]
    df["lon_kalman"] = df["longitude"]

    for agent_id, group in df.groupby("agent_id"):
        group = group.sort_values("timestamp")

        # State: [lat, lon, v_lat, v_lon]
        x = np.array([
            group.iloc[0]["latitude"],
            group.iloc[0]["longitude"],
            0.0,
            0.0
        ])

        # Covariance
        P = np.eye(4) * 0.01

        # Transition matrix
        F = np.eye(4)
        F[0, 2] = 1
        F[1, 3] = 1

        # Measurement matrix
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Noise matrices
        R = np.eye(2) * 0.0001   # GPS noise
        Q = np.eye(4) * 0.00001  # Process noise

        for idx in group.index:
            z = np.array([
                df.loc[idx, "latitude"],
                df.loc[idx, "longitude"]
            ])

            # Predict
            x = F @ x
            P = F @ P @ F.T + Q

            # Update
            y = z - (H @ x)
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)

            x = x + K @ y
            P = (np.eye(4) - K @ H) @ P

            df.loc[idx, "lat_kalman"] = x[0]
            df.loc[idx, "lon_kalman"] = x[1]

    return df
