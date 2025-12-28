import pandas as pd
import numpy as np
from src.preprocess_data import calculate_features


def test_calculate_features_basic():
    # Create a small two-agent dataset
    data = [
        {
            "agent_id": 1,
            "timestamp": "2025-01-01T00:00:00",
            "latitude": 10.0,
            "longitude": 20.0,
        },
        {
            "agent_id": 1,
            "timestamp": "2025-01-01T00:00:10",
            "latitude": 10.0001,
            "longitude": 20.0001,
        },
        {
            "agent_id": 2,
            "timestamp": "2025-01-01T00:00:00",
            "latitude": 11.0,
            "longitude": 21.0,
        },
    ]
    df = pd.DataFrame(data)
    out = calculate_features(df)

    # Columns created
    assert "speed_m_s" in out.columns
    assert "dist_moved_m" in out.columns
    assert "angle_change" in out.columns

    # First row of each agent should have zero distance/speed
    first_agent_mask = out["agent_id"] == 1
    assert out[first_agent_mask].iloc[0]["dist_moved_m"] == 0
    assert out[first_agent_mask].iloc[0]["speed_m_s"] == 0

    # Second row of agent 1 should have non-negative speed
    sp = out[first_agent_mask].iloc[1]["speed_m_s"]
    assert sp >= 0 or np.isnan(sp)
