import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import config


def generate_scientific_data(n_points=1200):
    """
     HARD NEGATIVE SAMPLES
    - Some patrols behave like intruders
    - Some intruders behave stealthily
    - Forces model to learn CONTEXT, not shortcuts
    """

    data = []

    # Rann of Kutch border reference
    base_lat, base_lon = 23.8, 69.5

    object_types = ["Human", "Animal", "Vehicle", "Drone"]
    terrains = ["Salt Flat", "Marshy", "Sandy"]
    visibilities = ["Clear", "Foggy", "Night"]

    for i in range(n_points):
        # Balanced classes
        label = 1 if i >= (n_points // 2) else 0

        lat = base_lat + random.uniform(-0.05, 0.05)
        lon = base_lon + random.uniform(-0.05, 0.05)

        # --- HARD NEGATIVE LOGIC ---
        hard_case = random.random()

        if label == 0:  # PATROL
            if hard_case < 0.3:
                # HARD NEGATIVE PATROL (looks like intruder)
                speed = random.uniform(6.0, 9.0)
                angle = random.uniform(50, 120)
            else:
                # Normal patrol
                speed = random.uniform(2.5, 4.5)
                angle = random.uniform(0, 20)

        else:  # INTRUDER
            if hard_case < 0.3:
                # STEALTH INTRUDER (looks innocent)
                speed = random.uniform(1.0, 3.0)
                angle = random.uniform(0, 15)
            else:
                # Normal intruder
                speed = random.uniform(4.5, 8.5)
                angle = random.uniform(40, 130)

        # Sensor confidence now wider (prepping Day 6)
        sensor_confidence = round(random.uniform(0.55, 0.98), 2)

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
                "sensor_confidence": sensor_confidence,
                "label": label,
            }
        )

    df = pd.DataFrame(data)

    out = Path(config.raw_data_path())
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("Dataset with HARD NEGATIVE SAMPLES generated")


if __name__ == "__main__":
    generate_scientific_data()
