from pathlib import Path

import pandas as pd

from src import config
from src.train_model import train_elite_model


def test_train_creates_model(tmp_path):
    # Prepare a tiny dataset matching expected columns
    rows = []
    for i in range(20):
        rows.append(
            {
                "speed": 1.0 + i * 0.1,
                "angle_change": 0.1 * i,
                "sensor_confidence": 0.9,
                "object_type": "Human",
                "terrain": "Sandy",
                "visibility": "Clear",
                "label": 0 if i < 10 else 1,
            }
        )

    df = pd.DataFrame(rows)
    raw = Path(config.raw_data_path())
    raw.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw, index=False)

    # Run training (should write model file)
    train_elite_model()

    model_file = Path(config.model_path())
    assert model_file.exists()

    # Clean up artifacts
    try:
        model_file.unlink()
        raw.unlink()
    except OSError:
        pass
