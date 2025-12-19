import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_movement_data(num_agents=15):
    all_rows = []
    
    for i in range(num_agents):
        # Assign a unique ID to each "person"
        agent_id = f"ID_{i:03d}"
        
        # 30% of agents will be 'Suspicious' (Label 1), 70% 'Normal' (Label 0)
        is_suspicious = np.random.random() < 0.3
        label = 1 if is_suspicious else 0
        
        # Start everyone at a random spot near the "border" (Lon 70.0)
        lat = np.random.uniform(24.0, 26.0)
        lon = np.random.uniform(69.0, 69.5)
        timestamp = datetime.now()
        
        for step in range(40): # Each person takes 40 "steps"
            if is_suspicious:
                # ZIG-ZAG: Random small moves in any direction
                lat += np.random.uniform(-0.005, 0.005)
                lon += np.random.uniform(-0.002, 0.010) # Generally moving toward border
            else:
                # PATROL: Steady, predictable straight line
                lat += 0.002
                lon += 0.0005
            
            all_rows.append({
                "agent_id": agent_id,
                "timestamp": timestamp + timedelta(minutes=step * 5),
                "latitude": lat,
                "longitude": lon,
                "label": label
            })
            
    return pd.DataFrame(all_rows)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Create the data
    df = create_movement_data(20)
    
    # 2. Ensure the data/raw directory exists
    os.makedirs('data/raw', exist_ok=True)
    
    # 3. Save it to a CSV file
    file_path = "data/raw/synthetic_border_data.csv"
    df.to_csv(file_path, index=False)
    
    print(f"Success! Generated {len(df)} rows of movement data.")
    print(f"File saved at: {file_path}")