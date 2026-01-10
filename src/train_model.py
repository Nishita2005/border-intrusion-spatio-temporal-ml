from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


def train_elite_model():
    # --------------------------------------------------
    # 1. Load processed data
    # --------------------------------------------------
    raw_path = Path(config.raw_data_path())
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")

    df = pd.read_csv(raw_path)

    # --------------------------------------------------
    # 2. Ensure temporal correctness (CRITICAL)
    # --------------------------------------------------
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column missing from dataset")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --------------------------------------------------
    # 3. Define features & target
    # --------------------------------------------------
    numeric_features = [
        "speed",
        "angle_change",
        "sensor_confidence",
    ]

    categorical_features = [
        "object_type",
        "terrain",
        "visibility",
    ]

    X = df[numeric_features + categorical_features]
    y = df["label"]

    # --------------------------------------------------
    # 4. Preprocessing pipeline
    # --------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # --------------------------------------------------
    # 5. Model pipeline (Defense-grade)
    # --------------------------------------------------
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # --------------------------------------------------
    # 6. TEMPORAL TRAIN–TEST SPLIT (NO LEAKAGE)
    # --------------------------------------------------
    split_idx = int(0.8 * len(df))

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]

    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    # --------------------------------------------------
    # 7. Train model
    # --------------------------------------------------
    pipeline.fit(X_train, y_train)

    # --------------------------------------------------
    # 8. Evaluate (train vs test)
    # --------------------------------------------------
    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)

    print(f"Train Accuracy : {train_acc:.3f}")
    print(f"Test Accuracy  : {test_acc:.3f}")

    # --------------------------------------------------
    # 9. Save model
    # --------------------------------------------------
    models_dir = Path(config.models_dir())
    models_dir.mkdir(parents=True, exist_ok=True)

    model_out = Path(config.model_path())
    joblib.dump(pipeline, model_out)

    print("Elite Model trained with TEMPORAL validation")
    print(f"Model saved to: {model_out}")


if __name__ == "__main__":
    train_elite_model()
