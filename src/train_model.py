from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score # Added cross_val_score

import config

def train_elite_model():
    # 1. Load data
    raw_path = Path(config.raw_data_path())
    df = pd.read_csv(raw_path)

    # 2. Define features
    numeric_features = ["speed", "angle_change", "sensor_confidence"]
    categorical_features = ["object_type", "terrain", "visibility"]
    X = df[numeric_features + categorical_features]
    y = df["label"]

    # 3. Pipeline Setup
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # UPDATED CLASSIFIER: Added constraints to fix 1.00 training accuracy
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,         # Prevents memorization
                    min_samples_leaf=5,   # Ensures general rules
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=True, random_state=42, stratify=y
    )

    # --------------------------------------------------
    # 5. NEW: CROSS-VALIDATION (5-Fold)
    # --------------------------------------------------
    print("\n--- Running 5-Fold Cross-Validation ---")
    # We run this on the training set to see how stable the model is
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    
    print(f"CV Individual Scores: {cv_scores}")
    print(f"CV Mean Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # 6. Final Fit and Evaluation
    pipeline.fit(X_train, y_train)
    
    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)
    
    print(f"\n--- Final Results ---")
    print(f"Train Accuracy : {train_acc:.3f}") # This should now be < 1.00
    print(f"Test Accuracy  : {test_acc:.3f}")

    # --------------------------------------------------
    # 7. NEW: Feature Importance
    # --------------------------------------------------
    # 1. Get the trained model from the pipeline
    rf_model = pipeline.named_steps['classifier']
    
    # 2. Get feature names after OneHotEncoding
    # We combine the numeric names with the new categorical names
    cat_encoder = pipeline.named_steps['preprocessor'].transformers_[1][1]
    encoded_cat_names = cat_encoder.get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(encoded_cat_names)

    # 3. Create importance series and sort
    importances = rf_model.feature_importances_
    feat_importances = pd.Series(importances, index=all_feature_names).sort_values(ascending=True)

    # 4. Plot
    plt.figure(figsize=(10, 6))
    feat_importances.plot(kind='barh', color='skyblue')
    plt.title("Critical Features for Intrusion Detection")
    plt.xlabel("Relative Importance Score")
    
    # Save to visuals folder
    importance_path = Path("visuals/feature_importance.png")
    plt.tight_layout()
    plt.savefig(importance_path)
    plt.close()
    print(f"Feature importance plot saved to: {importance_path}")

    # 8. Visualization & Save (Logic remains the same)
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig("visuals/confusion_matrix.png")
    
    joblib.dump(pipeline, config.model_path())
    print(f"Model saved successfully.")

if __name__ == "__main__":
    train_elite_model()