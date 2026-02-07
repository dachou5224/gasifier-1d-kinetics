
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_profiles(csv_path='diagnostic_results.csv', output_dir='plots'):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(csv_path)
    z = df['z (m)']
    
    # --- Data Processing ---
    # 1. Calculate Gas Volume Percentage (Vol% - Dry Basis)
    # Species columns: F_O2, F_CH4, F_CO, F_CO2, F_H2S, F_H2, F_N2
    # H2O is excluded for Dry Basis reporting
    dry_cols = ['F_O2', 'F_CH4', 'F_CO', 'F_CO2', 'F_H2', 'F_H2S', 'F_N2']
    df['F_dry_total'] = df[dry_cols].sum(axis=1)
    for col in dry_cols:
        sp_name = col.replace('F_', '')
        df[f'Vol%_{sp_name}'] = (df[col] / (df['F_dry_total'] + 1e-9)) * 100.0
        
    # 2. Calculate Carbon Conversion (X)
    # X = 1 - (m_c / m_c0)
    # m_c = SolidMass * CarbonFrac
    df['CarbonMass'] = df['SolidMass (kg/s)'] * df['CarbonFrac']
    m_c0 = df['CarbonMass'].iloc[0]
    df['Conversion_X'] = 1.0 - (df['CarbonMass'] / (m_c0 + 1e-9))
    df['Conversion_X'] = np.clip(df['Conversion_X'], 0.0, 1.0)

    # --- Plotting ---
    
    # Plot A: Merged Reaction Rates (Homogeneous & Heterogeneous)
    plt.figure(figsize=(10, 6))
    
    het_rates = [c for c in df.columns if 'Rate_Het' in c]
    homo_rates = [c for c in df.columns if 'Rate_Homo' in c]
    
    for r in het_rates:
        label = f"Het: {r.replace('Rate_Het_', '')}"
        plt.plot(z, df[r], label=label, linewidth=1.5)
        
    for r in homo_rates:
        label = f"Homo: {r.replace('Rate_Homo_', '')}"
        plt.plot(z, df[r], label=label, linewidth=1.2, linestyle='--')
        
    plt.xlabel('Axial Position (m)')
    plt.ylabel('Reaction Rate (mol/s)')
    plt.title('Merged Reaction Rates (Heterogeneous & Homogeneous)')
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'merged_rates.png'), dpi=300)
    plt.close()

    # Plot B: Comprehensive Profiles (Temperature, Conversion, Gas Vol%)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # 1. Temperature (Primary Y-axis)
    ax1.set_xlabel('Axial Position (m)')
    ax1.set_ylabel('Temperature (°C)', color='red')
    # Convert T (K) to T (C) if it looks like Kelvin (e.g. > 200)
    T_C = df['T (K)'] - 273.15 if df['T (K)'].mean() > 200 else df['T (K)']
    p1, = ax1.plot(z, T_C, 'r-', linewidth=2.5, label='Temperature')
    ax1.tick_params(axis='y', labelcolor='red')
    
    # 2. Carbon Conversion (Secondary Y-axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Carbon Conversion (X)', color='black')
    p2, = ax2.plot(z, df['Conversion_X'], 'k--', linewidth=2, label='Conversion (X)')
    ax2.set_ylim(-0.05, 1.05)
    
    # 3. Gas Composition Vol% (Shared with Ax2 or extra?)
    # We plot Vol% on a 0-100% scale. Let's use Ax2 and set ylim 0-100 if we want it together,
    # or use another axis. Since user asked "merged", let's put Vol% on a 0-100 scale on Ax2.
    # Wait, X and Vol% have different scales (0-1 vs 0-100). 
    # Let's use ax2 for both by scaling X * 100 or using a 3rd axis.
    # User said "气体组成的范围应该在0-100%之间".
    
    # Let's make ax2 for 0-100 range.
    # Plot Title to mention Dry Basis
    plt.title('Comprehensive Gasifier Profiles: Temp, Xc, and Gas Vol% (Dry Basis)')
    ax2.set_ylabel('Percentage (Dry Basis %)', color='blue')
    ax2.set_ylim(0, 105)
    
    # Re-plot Conversion as % (Reference line)
    p2, = ax2.plot(z, df['Conversion_Coal (%)'], 'k--', linewidth=2, label='Conversion (%)')
    
    vol_species = ['Vol%_CO', 'Vol%_H2', 'Vol%_CO2', 'Vol%_CH4', 'Vol%_O2']
    colors = ['green', 'purple', 'orange', 'brown', 'blue']
    plots_vol = []
    for sp, color in zip(vol_species, colors):
        label = sp.replace('Vol%_', '')
        p_vol, = ax2.plot(z, df[sp], label=f"{label} (vol%)", color=color, alpha=0.8)
        plots_vol.append(p_vol)
        
    plt.title('Comprehensive Gasifier Profiles: Temperature, Conversion, and Gas Vol%')
    
    # Legend handling
    lns = [p1, p2] + plots_vol
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center right', fontsize='small')
    
    ax1.grid(True, linestyle=':', alpha=0.5)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_profiles.png'), dpi=300)
    plt.close()

    print(f"[Plot] Enhanced merged profiles generated in {output_dir}/")

if __name__ == "__main__":
    plot_profiles()
