import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

def generate_scientific_data(n_points=1000):
    data = []
    # Coordinates for the Rann of Kutch border area
    base_lat, base_lon = 23.8, 69.5
    
    # "Elite" Metadata Categories
    object_types = ['Human', 'Animal', 'Vehicle', 'Drone']
    terrains = ['Salt Flat', 'Marshy', 'Sandy']
    visibilities = ['Clear', 'Foggy', 'Night']
    
    for i in range(n_points):
        # 50/50 Split between Normal and Intruder
        label = 1 if i > (n_points // 2) else 0 
        lat = base_lat + random.uniform(-0.05, 0.05)
        lon = base_lon + random.uniform(-0.05, 0.05)
        
        if label == 1: # Intruder Signature
            speed = random.uniform(2.0, 8.0)
            angle = random.uniform(45, 120)
        else: # Patrol Signature
            speed = random.uniform(3.0, 5.0)
            angle = random.uniform(0, 15)
            
        data.append({
            'timestamp': (datetime.now() + timedelta(seconds=i)).isoformat(),
            'latitude': lat,
            'longitude': lon,
            'speed': speed,
            'angle_change': angle,
            'object_type': random.choice(object_types),
            'terrain': random.choice(terrains),
            'visibility': random.choice(visibilities),
            'sensor_confidence': round(random.uniform(0.85, 0.99), 2),
            'label': label
        })

    df = pd.DataFrame(data)
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/border_data.csv', index=False)
    print("✅ Step 1 Complete: Elite Scientific Dataset Created!")

if __name__ == "__main__":
    generate_scientific_data()