"""
02_extract_glacier_points.py
----------------------------
Extracts the in-situ Ground-Penetrating Radar (GPR) survey points for
Urumqi Glacier No. 1 (GlaThiDa ID: 2083) from the point database, calculates
bedrock elevation (Bedrock = Surface - Thickness), and plots the 2D survey transects.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)

data_file = os.path.join("data", "glacier_urumqi_no1_points.csv")

if os.path.exists(data_file):
    df = pd.read_csv(data_file)
    print(f"Loaded existing clean dataset: {data_file}")
else:
    # Scan raw GlaThiDa TTT.csv if present
    ttt_path = os.path.join("..", "Glaciers", "glathida-3.1.0", "data", "TTT.csv")
    print(f"Extracting Urumqi Glacier No. 1 (ID: 2083) from {ttt_path}...")
    matched = []
    for chunk in pd.read_csv(ttt_path, chunksize=100000, low_memory=False):
        sub = chunk[chunk['GlaThiDa_ID'] == 2083]
        if not sub.empty:
            matched.append(sub)
    df = pd.concat(matched, ignore_index=True)
    df['BEDROCK_ELEVATION'] = df['ELEVATION'] - df['THICKNESS']
    df.to_csv(data_file, index=False)
    print(f"Saved extracted points to: {data_file}")

# Glaciological Summary
print("\n" + "=" * 55)
print("     URUMQI GLACIER NO. 1 (GPR SURVEY SUMMARY)")
print("=" * 55)
print(f"Total GPR Survey Points : {len(df)}")
print(f"Surface Elevation Range : {df['ELEVATION'].min():.1f} m to {df['ELEVATION'].max():.1f} m")
print(f"Ice Thickness Range     : {df['THICKNESS'].min():.1f} m to {df['THICKNESS'].max():.1f} m")
print(f"Mean Ice Thickness      : {df['THICKNESS'].mean():.1f} m")
print(f"Subglacial Bedrock Range: {df['BEDROCK_ELEVATION'].min():.1f} m to {df['BEDROCK_ELEVATION'].max():.1f} m")
print("=" * 55)

# Plot 2D GPR Survey Transects
plt.figure(figsize=(9, 6))
scatter = plt.scatter(
    df['POINT_LON'], df['POINT_LAT'],
    c=df['THICKNESS'], cmap='Blues',
    edgecolors='black', s=25, alpha=0.9
)
cbar = plt.colorbar(scatter)
cbar.set_label('Ice Thickness (meters)', fontsize=11, fontweight='bold')

plt.title("Urumqi Glacier No. 1: Real GPR Radar Survey Points (GlaThiDa)", fontsize=12, fontweight='bold')
plt.xlabel("Longitude (°E)", fontsize=11)
plt.ylabel("Latitude (°N)", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)

plot_file = os.path.join("figures", "glacier_gpr_survey_map.png")
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Saved survey map plot to: {plot_file}")
plt.show()
