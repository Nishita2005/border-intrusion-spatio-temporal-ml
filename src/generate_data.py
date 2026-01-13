import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import config

# =========================
# BORDER DEFINITIONS
# =========================
BORDER_LON = 69.50
AMBIGUITY_WIDTH = 0.005  # ~500 meters


def generate_scientific_data(n_points=1000):
    data = []

    # Base coordinates (Rann of Kutch region)
    base_lat, base_lon = 23.8, 69.5

    object_types = ["Human", "Animal", "Vehicle", "Drone"]
    terrains = ["Salt Flat", "Marshy", "Sandy"]
    visibilities = ["Clear", "Foggy", "Night"]

    start_time = datetime.now()

    for i in range(n_points):
        # -------------------------
        # LOCATION GENERATION
        # -------------------------
        lon_offset = random.uniform(-0.02, 0.02)
        lon = base_lon + lon_offset
        lat = base_lat + random.uniform(-0.05, 0.05)

        distance_from_border = abs(lon - BORDER_LON)
        is_ambiguous = distance_from_border <= AMBIGUITY_WIDTH

        # -------------------------
        # LABEL LOGIC
        # -------------------------
        if is_ambiguous:
            label = random.choice([0, 1])  # Confusion zone
        else:
            label = 1 if lon > BORDER_LON else 0

        # -------------------------
        # MOVEMENT SIGNATURES
        # -------------------------
        if is_ambiguous:
            speed = random.uniform(2.5, 6.0)
            angle = random.uniform(10, 80)
        elif label == 1:  # Intruder
            speed = random.uniform(2.0, 8.0)
            angle = random.uniform(45, 120)
        else:  # Patrol
            speed = random.uniform(3.0, 5.0)
            angle = random.uniform(0, 15)

        # -------------------------
        # SENSOR CONFIDENCE
        # -------------------------
        if is_ambiguous:
            sensor_confidence = round(random.uniform(0.55, 0.75), 2)
        else:
            sensor_confidence = round(random.uniform(0.85, 0.99), 2)

        data.append(
            {
                "timestamp": (start_time + timedelta(seconds=i)).isoformat(),
                "latitude": lat,
                "longitude": lon,
                "speed": speed,
                "angle_change": angle,
                "object_type": random.choice(object_types),
                "terrain": random.choice(terrains),
                "visibility": random.choice(visibilities),
                "sensor_confidence": sensor_confidence,
                "ambiguity_zone": int(is_ambiguous),
                "label": label,
            }
        )

    df = pd.DataFrame(data)

    out = Path(config.raw_data_path())
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(" Dataset with BORDER AMBIGUITY ZONES generated")


if __name__ == "__main__":
    generate_scientific_data()
