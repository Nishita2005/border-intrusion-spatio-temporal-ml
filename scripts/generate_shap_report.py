#!/usr/bin/env python
"""
SHAP explainability runner.

Generates SHAP summary plots and dependence plots for model explainability.
Run this after training the model to generate visualizations in visuals/.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.explainability import (
    generate_shap_summary_plot,
    generate_shap_dependence_plots,
)


def main():
    print("=" * 60)
    print("SHAP Model Explainability Report Generator")
    print("=" * 60)
    
    try:
        print("\n1. Generating SHAP summary plot (global feature importance)...")
        generate_shap_summary_plot()
        
        print("\n2. Generating SHAP dependence plots (top features)...")
        generate_shap_dependence_plots()
        
        print("\n" + "=" * 60)
        print("✓ SHAP explainability artifacts generated successfully!")
        print("  Check visuals/ folder for:")
        print("    - shap_summary.png (feature importance bar chart)")
        print("    - shap_dependence_*.png (individual feature plots)")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the model has been trained first using: python -m src.train_model")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating SHAP plots: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
