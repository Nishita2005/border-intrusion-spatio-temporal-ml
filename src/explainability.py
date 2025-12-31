"""
SHAP-based model explainability module.

Provides tools for generating SHAP explanations for model predictions.
"""

from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src import config


def load_model_and_preprocessor():
    """Load the trained pipeline model."""
    model_path = Path(config.model_path())
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    pipeline = joblib.load(model_path)
    return pipeline


def load_training_data():
    """Load training data for SHAP explainer background."""
    raw_path = Path(config.raw_data_path())
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")
    df = pd.read_csv(raw_path)
    
    numeric_features = ["speed", "angle_change", "sensor_confidence"]
    categorical_features = ["object_type", "terrain", "visibility"]
    
    X = df[numeric_features + categorical_features]
    return X, numeric_features, categorical_features


def generate_shap_summary_plot(output_path: Optional[str] = None):
    """
    Generate SHAP summary plot (bar chart) showing global feature importance.
    
    Args:
        output_path: Optional path to save the plot. If None, uses visuals/shap_summary.png
    """
    pipeline = load_model_and_preprocessor()
    X, numeric_features, categorical_features = load_training_data()
    
    # Transform data using the pipeline's preprocessor
    X_transformed = pipeline.named_steps['preprocessor'].transform(X)
    
    # Use a sample for performance (SHAP can be slow on large datasets)
    sample_size = min(100, len(X_transformed))
    X_sample = X_transformed[:sample_size]
    
    # Create TreeExplainer for RandomForest
    explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
    shap_values = explainer.shap_values(X_sample)
    
    # For binary classification, use the positive class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Generate summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=(numeric_features + categorical_features),
        plot_type="bar",
        show=False
    )
    
    if output_path is None:
        output_path = Path(config.visuals_dir()) / "shap_summary.png"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved to {output_path}")
    return str(output_path)


def generate_shap_dependence_plots(output_dir: Optional[str] = None):
    """
    Generate SHAP dependence plots for top features.
    
    Args:
        output_dir: Optional directory to save plots. If None, uses visuals/
    """
    pipeline = load_model_and_preprocessor()
    X, numeric_features, categorical_features = load_training_data()
    
    # Transform data
    X_transformed = pipeline.named_steps['preprocessor'].transform(X)
    
    # Use a sample
    sample_size = min(100, len(X_transformed))
    X_sample = X_transformed[:sample_size]
    
    # Create TreeExplainer
    explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
    shap_values = explainer.shap_values(X_sample)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    if output_dir is None:
        output_dir = config.visuals_dir()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots for top 3 features
    all_feature_names = numeric_features + categorical_features
    feature_importance = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(feature_importance)[-3:][::-1]
    
    for idx in top_features_idx:
        feature_name = all_feature_names[idx]
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            idx,
            shap_values,
            X_sample,
            feature_names=all_feature_names,
            show=False
        )
        output_path = output_dir / f"shap_dependence_{feature_name}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved dependence plot: {output_path}")


def explain_single_prediction(X_sample: pd.DataFrame) -> str:
    """
    Generate SHAP force plot for a single prediction (or small batch).
    
    Args:
        X_sample: DataFrame with 1 or few rows to explain
        
    Returns:
        Path to saved HTML force plot
    """
    pipeline = load_model_and_preprocessor()
    X_all, numeric_features, categorical_features = load_training_data()
    
    # Transform sample
    X_sample_transformed = pipeline.named_steps['preprocessor'].transform(X_sample)
    X_all_transformed = pipeline.named_steps['preprocessor'].transform(X_all)
    
    # Create explainer with background data
    background_sample = X_all_transformed[:min(50, len(X_all_transformed))]
    explainer = shap.TreeExplainer(
        pipeline.named_steps['classifier'],
        background_sample
    )
    
    shap_values = explainer.shap_values(X_sample_transformed)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Generate force plot (HTML)
    output_dir = Path(config.visuals_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "shap_force_plot.html"
    
    # Create force plot for first sample
    shap.force_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
        shap_values[0],
        X_sample_transformed[0],
        feature_names=(numeric_features + categorical_features),
        matplotlib=False
    ).save_html(str(output_path))
    
    print(f"SHAP force plot saved to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    print("Generating SHAP explainability artifacts...")
    generate_shap_summary_plot()
    generate_shap_dependence_plots()
    print("Done! Check visuals/ for plots.")
