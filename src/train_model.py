from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def train_elite_model():
    # 1. Load the new data
    raw_path = Path(config.raw_data_path())
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")
    df = pd.read_csv(raw_path)

    # 2. Define our features
    numeric_features = ["speed", "angle_change", "sensor_confidence"]
    categorical_features = ["object_type", "terrain", "visibility"]

    X = df[numeric_features + categorical_features]
    y = df["label"]

    # 3. Create a "Preprocessor" (Handles numbers and text automatically)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(), categorical_features),
        ]
    )

    # 4. Create the "Elite Pipeline"
    # This bundles the preprocessing and the AI model into one file!
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # 5. Split and Train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)

    # 6. Save the entire Pipeline
    models_dir = Path(config.models_dir())
    models_dir.mkdir(parents=True, exist_ok=True)
    model_out = Path(config.model_path())
    joblib.dump(pipeline, model_out)
    acc = pipeline.score(X_test, y_test)
    print("Step 2 Complete: Elite Model trained (Accuracy: {:.2f})".format(acc))


if __name__ == "__main__":
    train_elite_model()
