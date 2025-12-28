from pathlib import Path
import os


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))


def data_dir():
    return PROJECT_ROOT / "data"


def raw_data_path(filename="border_data.csv"):
    return data_dir() / "raw" / filename


def processed_data_path(filename="featured_border_data.csv"):
    return data_dir() / "processed" / filename


def models_dir():
    return PROJECT_ROOT / "models"


def model_path(filename="border_intruder_model.pkl"):
    return models_dir() / filename


def visuals_dir():
    return PROJECT_ROOT / "visuals"
