"""Simple orchestration script to run generate -> preprocess -> train -> evaluate -> visualize
This is a convenience script for demoing the full pipeline locally.
"""

from src import (
    evaluate_model,
    generate_data,
    preprocess_data,
    train_model,
    visualize_data,
)


def run_all(n_points: int = 500):
    print("1/5: Generating data...")
    generate_data.generate_scientific_data(n_points=n_points)

    print("2/5: Preprocessing data...")
    # The preprocess script can be imported and used programmatically
    import pandas as pd

    raw_candidates = [
        "data/raw/synthetic_border_data.csv",
        "data/raw/border_data.csv",
    ]
    for p in raw_candidates:
        try:
            df = pd.read_csv(p)
            processed = preprocess_data.calculate_features(df)
            processed.to_csv("data/processed/featured_border_data.csv", index=False)
            break
        except Exception:
            continue

    print("3/5: Training model...")
    train_model.train_elite_model()

    print("4/5: Evaluating model...")
    evaluate_model.evaluate_elite_system()

    print("5/5: Visualizing results...")
    visualize_data.plot_movements()


if __name__ == "__main__":
    run_all(n_points=500)
