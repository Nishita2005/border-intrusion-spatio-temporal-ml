import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_movements():
    # Load the featured data
    file_path = "data/processed/featured_border_data.csv"
    df = pd.read_csv(file_path)
    
    plt.figure(figsize=(12, 8))
    
    # Draw a vertical line for the "Border" at Lon 70.0
    plt.axvline(x=70.0, color='black', linestyle='--', label='Border Line')
    
    # Plot each agent's path
    for agent_id in df['agent_id'].unique():
        agent_data = df[df['agent_id'] == agent_id]
        
        # Color based on label: 0=Green (Normal), 1=Red (Suspicious)
        color = 'green' if agent_data['label'].iloc[0] == 0 else 'red'
        label_text = "Normal" if agent_data['label'].iloc[0] == 0 else "Suspicious"
        
        plt.plot(agent_data['longitude'], agent_data['latitude'], 
                 color=color, alpha=0.6, marker='o', markersize=2)

    plt.title("Border Movement Simulation: Normal vs Suspicious Paths")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    # Clean up legend so it doesn't show every single ID
    plt.plot([], [], color='green', label='Normal (Patrol)')
    plt.plot([], [], color='red', label='Suspicious (Zig-Zag)')
    plt.legend()
    
    # Save the visual
    os.makedirs('visuals', exist_ok=True)
    plt.savefig('visuals/movement_map.png')
    print("Success! Map saved to visuals/movement_map.png")
    plt.show()

if __name__ == "__main__":
    plot_movements()