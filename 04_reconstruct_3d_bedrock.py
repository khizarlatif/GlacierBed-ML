"""
04_reconstruct_3d_bedrock.py
----------------------------
Reconstructs continuous 2D ice thickness maps across unmeasured gaps,
models the 3D subglacial bedrock topography (Bedrock = Surface - Thickness),
and calculates total glacier ice volume in cubic kilometers (km³) and million m³.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestRegressor
from scipy.interpolate import griddata

os.makedirs("figures", exist_ok=True)

# 1. Load the real GPR survey points
data_path = os.path.join("data", "glacier_urumqi_no1_points.csv")
df = pd.read_csv(data_path)
print(f"Loaded {len(df)} survey points for Urumqi Glacier No. 1.")

# 2. Train the full Random Forest Model
X = df[['POINT_LON', 'POINT_LAT', 'ELEVATION']]
y = df['THICKNESS']

print("Training Random Forest on full survey dataset...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# 3. Create a continuous 2D spatial grid over the glacier footprint
grid_res = 80
lon_grid = np.linspace(df['POINT_LON'].min(), df['POINT_LON'].max(), grid_res)
lat_grid = np.linspace(df['POINT_LAT'].min(), df['POINT_LAT'].max(), grid_res)
LON, LAT = np.meshgrid(lon_grid, lat_grid)

# Interpolate surface elevation onto the grid
SURF = griddata((df['POINT_LON'], df['POINT_LAT']), df['ELEVATION'], (LON, LAT), method='linear')

# Mask out areas outside the surveyed glacier boundary
mask = ~np.isnan(SURF)

# 4. Predict ice thickness across all unmeasured gaps
grid_features = pd.DataFrame({
    'POINT_LON': LON[mask],
    'POINT_LAT': LAT[mask],
    'ELEVATION': SURF[mask]
})

THICK = np.full(LON.shape, np.nan)
THICK[mask] = rf.predict(grid_features)

# Calculate Subglacial Bedrock: Bedrock = Surface - Thickness
BED = SURF - THICK

# 5. Calculate Real Physical Ice Volume (in km³ and million m³)
meters_per_deg_lat = 111139.0
meters_per_deg_lon = 111139.0 * np.cos(np.radians(43.11))

dx = (df['POINT_LON'].max() - df['POINT_LON'].min()) / (grid_res - 1) * meters_per_deg_lon
dy = (df['POINT_LAT'].max() - df['POINT_LAT'].min()) / (grid_res - 1) * meters_per_deg_lat
cell_area_m2 = dx * dy

valid_thickness = THICK[mask]
total_volume_m3 = np.sum(valid_thickness) * cell_area_m2
total_volume_km3 = total_volume_m3 / 1e9
glacier_area_km2 = len(valid_thickness) * cell_area_m2 / 1e6

print("\n" + "=" * 60)
print("     FINAL GLACIOLOGICAL INVENTORY & 3D RECONSTRUCTION")
print("=" * 60)
print(f"Glacier Footprint Area     : {glacier_area_km2:.2f} km²")
print(f"Mean Modeled Ice Thickness : {np.mean(valid_thickness):.1f} meters")
print(f"Max Modeled Ice Thickness  : {np.max(valid_thickness):.1f} meters")
print(f"Total Modeled Ice Volume   : {total_volume_km3:.4f} km³ ({total_volume_m3/1e6:.1f} million m³)")
print("=" * 60)

# =============================================================================
# 6. VISUALIZATION: 2D Continuous Map + 3D Subglacial Bedrock Surface
# =============================================================================
fig = plt.figure(figsize=(15, 6))

# Panel 1: 2D Continuous Reconstructed Ice Thickness (Filling the Gaps)
ax1 = fig.add_subplot(1, 2, 1)
c1 = ax1.contourf(LON, LAT, THICK, levels=25, cmap='Blues')
ax1.scatter(df['POINT_LON'], df['POINT_LAT'], color='black', s=4, alpha=0.35, label='Original GPR Transects')
cb1 = fig.colorbar(c1, ax=ax1)
cb1.set_label('Predicted Ice Thickness (m)', fontweight='bold')
ax1.set_title("2D Continuous Ice Thickness Map (Gaps Reconstructed)", fontweight='bold')
ax1.set_xlabel("Longitude (°E)", fontweight='bold')
ax1.set_ylabel("Latitude (°N)", fontweight='bold')
ax1.legend(loc='lower left')
ax1.grid(True, linestyle=':', alpha=0.6)

# Panel 2: 3D Perspective View (Surface Mesh vs Subglacial Bedrock)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf_plot = ax2.plot_surface(LON, LAT, SURF, color='#6baed6', alpha=0.55, edgecolor='none', label='Ice Surface')
bed_plot = ax2.plot_surface(LON, LAT, BED, cmap='copper', alpha=0.9, edgecolor='none')

ax2.set_title("3D Subglacial Bedrock Topography & Ice Surface", fontweight='bold', pad=15)
ax2.set_xlabel("Longitude (°E)", fontweight='bold', labelpad=8)
ax2.set_ylabel("Latitude (°N)", fontweight='bold', labelpad=8)
ax2.set_zlabel("Elevation (m a.s.l.)", fontweight='bold', labelpad=8)
ax2.view_init(elev=28, azim=-65)

plt.tight_layout()
final_img = os.path.join("figures", "glacier_3d_bedrock_reconstruction.png")
plt.savefig(final_img, dpi=300, bbox_inches='tight')
print(f"\nSaved final 3D reconstruction figure to: {final_img}")
plt.show()
