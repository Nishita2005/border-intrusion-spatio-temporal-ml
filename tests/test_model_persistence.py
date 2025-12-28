import pandas as pd
from pathlib import Path
from src.train_model import train_elite_model
from src import config
import joblib


def test_model_pipeline_persistence(tmp_path):
    # Create minimal dataset expected by train_elite_model
    rows = []
    for i in range(40):
        rows.append(
            {
                "speed": 1.0 + (i % 5) * 0.2,
                "angle_change": 0.1 * (i % 3),
                "sensor_confidence": 0.9,
                "object_type": "Human",
                "terrain": "Sandy",
                "visibility": "Clear",
                "label": 0 if i < 30 else 1,
            }
        )

    df = pd.DataFrame(rows)
    raw = Path(config.raw_data_path())
    raw.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw, index=False)

    # Train and persist model
    train_elite_model()

    model_file = Path(config.model_path())
    assert model_file.exists()

    # Load and check pipeline has required steps
    pipeline = joblib.load(model_file)
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "named_steps")

    # Clean up
    try:
        model_file.unlink()
        raw.unlink()
    except Exception:
        pass
