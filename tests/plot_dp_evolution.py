
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_dp():
    csv_path = 'diagnostic_results.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run diagnose_detailed.py first.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Filter for the first 2 meters where the action happens
    df_zoom = df[df['z (m)'] <= 2.0]
    
    plt.figure(figsize=(10, 6))
    
    # Plot Dp
    ax1 = plt.gca()
    lns1 = ax1.plot(df['z (m)'], df['d_p (um)'], 'b-o', label='Particle Diameter (um)', markersize=4)
    ax1.set_xlabel('Axial Position z (m)')
    ax1.set_ylabel('Particle Diameter (um)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plot Conversion on second axis
    ax2 = ax1.twinx()
    lns2 = ax2.plot(df['z (m)'], df['Conversion_Coal (%)'], 'r--', label='Carbon Conversion (%)', alpha=0.7)
    ax2.set_ylabel('Carbon Conversion (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Plot O2 on third axis
    ax3 = ax1.twinx()
    # Offset the third axis to the right
    ax3.spines['right'].set_position(('outward', 60))
    lns3 = ax3.plot(df['z (m)'], df['F_O2'], 'g-^', label='Gas O2 (mol/s)', markersize=4, alpha=0.8)
    ax3.set_ylabel('Gas O2 (mol/s)', color='g')
    ax3.tick_params(axis='y', labelcolor='g')
    
    plt.title('Axial Evolution of Dp, Conversion, and Oxygen Consumption')
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center right')
    
    plt.tight_layout()
    plot_path = 'dp_o2_evolution_profile.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    plot_dp()
