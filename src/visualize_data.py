from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src import config


def plot_movements():
    # Load the featured data
    file_path = Path(config.processed_data_path())
    if not file_path.exists():
        raise FileNotFoundError(f"Processed featured data not found: {file_path}")
    df = pd.read_csv(file_path)

    plt.figure(figsize=(12, 8))

    # Draw a vertical line for the "Border" at Lon 70.0
    plt.axvline(x=70.0, color="black", linestyle="--", label="Border Line")

    # Plot each agent's path
    for agent_id in df["agent_id"].unique():
        agent_data = df[df["agent_id"] == agent_id]

        # Color based on label: 0=Green (Normal), 1=Red (Suspicious)
        color = "green" if agent_data["label"].iloc[0] == 0 else "red"

        plt.plot(
            agent_data["longitude"],
            agent_data["latitude"],
            color=color,
            alpha=0.6,
            marker="o",
            markersize=2,
        )

    plt.title("Border Movement Simulation: Normal vs Suspicious Paths")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    # Clean up legend so it doesn't show every single ID
    plt.plot([], [], color="green", label="Normal (Patrol)")
    plt.plot([], [], color="red", label="Suspicious (Zig-Zag)")
    plt.legend()

    # Save the visual
    visuals = Path(config.visuals_dir())
    visuals.mkdir(parents=True, exist_ok=True)
    plt.savefig(visuals / "movement_map.png")
    print(f"Success! Map saved to {visuals / 'movement_map.png'}")
    plt.show()


if __name__ == "__main__":
    plot_movements()
