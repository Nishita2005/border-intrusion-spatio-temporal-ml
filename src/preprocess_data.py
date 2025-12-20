import pandas as pd
import numpy as np

def calculate_features(df):
    # Sort by agent and time to ensure calculations are in order
    df = df.sort_values(['agent_id', 'timestamp'])
    
    # 1. Calculate Speed (Distance between Lat/Lon points)
    # Simple Euclidean distance as a proxy for speed
    df['dist_moved'] = np.sqrt(df['latitude'].diff()**2 + df['longitude'].diff()**2)
    # Reset distance for the first point of every new agent
    df.loc[df['agent_id'] != df['agent_id'].shift(), 'dist_moved'] = 0
    
    # 2. Distance to Border (How far is Longitude from 70.0?)
    df['dist_to_border'] = np.abs(df['longitude'] - 70.0)
    
    # 3. Heading Change (Movement Direction)
    # Calculate the angle of movement
    df['direction'] = np.arctan2(df['latitude'].diff(), df['longitude'].diff())
    # Calculate the change in direction (detecting zig-zags)
    df['angle_change'] = df['direction'].diff().abs()
    # Reset for new agents
    df.loc[df['agent_id'] != df['agent_id'].shift(), 'angle_change'] = 0
    
    return df

if __name__ == "__main__":
    # Load the raw data we made yesterday
    raw_data_path = "data/raw/synthetic_border_data.csv"
    if not pd.io.common.file_exists(raw_data_path):
        print("Error: Raw data file not found! Run generate_data.py first.")
    else:
        df = pd.read_csv(raw_data_path)
        processed_df = calculate_features(df)
        
        # Save to the 'processed' folder
        import os
        os.makedirs('data/processed', exist_ok=True)
        processed_df.to_csv("data/processed/featured_border_data.csv", index=False)
        
        print("Success! Feature Engineering complete.")
        print("New file saved: data/processed/featured_border_data.csv")