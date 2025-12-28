import pandas as pd
import pytest
from src.preprocess_data import calculate_features


def test_calculate_features_empty():
    df = pd.DataFrame(columns=["agent_id", "timestamp", "latitude", "longitude"]) 
    out = calculate_features(df)
    # Should return DataFrame with expected columns even if empty
    assert "dist_moved_m" in out.columns


def test_calculate_features_missing_columns():
    # Missing latitude/longitude should raise KeyError
    df = pd.DataFrame([{"agent_id": 1, "timestamp": "2025-01-01T00:00:00"}])
    with pytest.raises(KeyError):
        calculate_features(df)
