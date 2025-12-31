import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

from src import config


def evaluate_elite_system():
    # 1. Load Model and Data
    model_path = Path(config.model_path())
    pipeline = joblib.load(model_path)
    raw_path = Path(config.raw_data_path())
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")
    df = pd.read_csv(raw_path)

    X = df.drop(columns=["label", "timestamp"])
    y = df["label"]

    print("--- 🛡️ Military Grade Validation ---")

    # 2. K-Fold Cross Validation (The Stress Test)
    scores = cross_val_score(pipeline, X, y, cv=5)
    print("Cross-Validation Scores: {}".format(scores))
    mean = scores.mean()
    err = scores.std() * 2
    print("Mean Reliability: {:.4f} (+/- {:.4f})".format(mean, err))

    # 3. Confusion Matrix (The 'False Alarm' Check)
    y_pred = pipeline.predict(X)
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Intruder"],
        yticklabels=["Normal", "Intruder"],
    )
    plt.title("Confusion Matrix: Detection Accuracy")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    visuals = Path(config.visuals_dir())
    visuals.mkdir(parents=True, exist_ok=True)
    plt.savefig(visuals / "confusion_matrix.png")
    print("✅ Validation Complete. Matrix saved to visuals/ folder.")


if __name__ == "__main__":
    os.makedirs("visuals", exist_ok=True)
    evaluate_elite_system()
