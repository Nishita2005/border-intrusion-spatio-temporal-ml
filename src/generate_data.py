import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import config


def generate_scientific_data(n_points=1000):
    data = []

    # Coordinates for the Rann of Kutch border area
    base_lat, base_lon = 23.8, 69.5

    # Metadata categories
    object_types = ["Human", "Animal", "Vehicle", "Drone"]
    terrains = ["Salt Flat", "Marshy", "Sandy"]
    visibilities = ["Clear", "Foggy", "Night"]

    for i in range(n_points):
        # 50/50 split
        label = 1 if i > (n_points // 2) else 0

        lat = base_lat + random.uniform(-0.05, 0.05)
        lon = base_lon + random.uniform(-0.05, 0.05)

        if label == 1:  # Intruder signature
            speed = random.uniform(2.0, 8.0)
            angle = random.uniform(45, 120)
        else:  # Patrol signature
            speed = random.uniform(3.0, 5.0)
            angle = random.uniform(0, 15)

        data.append(
            {
                "timestamp": (datetime.now() + timedelta(seconds=i)).isoformat(),
                "latitude": lat,
                "longitude": lon,
                "speed": speed,
                "angle_change": angle,
                "object_type": random.choice(object_types),
                "terrain": random.choice(terrains),
                "visibility": random.choice(visibilities),
                "label": label,
            }
        )

    # Create DataFrame
    df = pd.DataFrame(data)

    # ==============================
    # DAY 3: PARTIAL SENSOR CONFIDENCE
    # ==============================

    # Base imperfect sensor confidence
    df["sensor_confidence"] = [
        random.uniform(0.4, 1.0) for _ in range(len(df))
    ]

    # Intruders are harder to detect (stealth / camouflage)
    intruder_mask = df["label"] == 1
    df.loc[intruder_mask, "sensor_confidence"] *= random.uniform(0.6, 0.85)

    # Random sensor degradation (weather / jamming)
    degrade_idx = df.sample(frac=0.1, random_state=42).index
    df.loc[degrade_idx, "sensor_confidence"] *= 0.5

    # Clamp values to valid range
    df["sensor_confidence"] = df["sensor_confidence"].clip(0.1, 1.0)
    df["sensor_confidence"] = df["sensor_confidence"].round(2)

    # Save data
    out = Path(config.raw_data_path())
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("✅ Day 3 Complete: Dataset with partial sensor confidence generated")


if __name__ == "__main__":
    generate_scientific_data()
