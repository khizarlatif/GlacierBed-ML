"""
05_export_to_qgis.py
--------------------
Exports the in-situ GPR survey points as a GeoJSON vector layer and
exports the reconstructed ice thickness and subglacial bedrock elevation
as standard georeferenced GeoTIFF rasters (EPSG:4326) for analysis in QGIS.
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.transform import from_bounds
from sklearn.ensemble import RandomForestRegressor
from scipy.interpolate import griddata

os.makedirs("gis_outputs", exist_ok=True)

# 1. Load the clean glacier survey points
data_path = os.path.join("data", "glacier_urumqi_no1_points.csv")
df = pd.read_csv(data_path)
print(f"Loaded {len(df)} points from {data_path}.")

# 2. Export GPR Survey Points as GeoJSON (Vector Layer for QGIS)
geometry = [Point(xy) for xy in zip(df['POINT_LON'], df['POINT_LAT'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
geojson_file = os.path.join("gis_outputs", "urumqi_gpr_survey_points.geojson")
gdf.to_file(geojson_file, driver="GeoJSON")
print(f"[VECTOR] Exported GPR survey points to: {geojson_file}")

# 3. Train the Random Forest Model
X = df[['POINT_LON', 'POINT_LAT', 'ELEVATION']]
y = df['THICKNESS']
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# 4. Generate 2D Raster Grid (100x100 resolution)
grid_res = 100
min_lon, max_lon = df['POINT_LON'].min(), df['POINT_LON'].max()
min_lat, max_lat = df['POINT_LAT'].min(), df['POINT_LAT'].max()

lon_grid = np.linspace(min_lon, max_lon, grid_res)
lat_grid = np.linspace(min_lat, max_lat, grid_res)
LON, LAT = np.meshgrid(lon_grid, lat_grid)

# Interpolate surface elevation and mask
SURF = griddata((df['POINT_LON'], df['POINT_LAT']), df['ELEVATION'], (LON, LAT), method='linear')
mask = ~np.isnan(SURF)

grid_features = pd.DataFrame({
    'POINT_LON': LON[mask],
    'POINT_LAT': LAT[mask],
    'ELEVATION': SURF[mask]
})

# Predict Ice Thickness and compute Bedrock
THICK = np.full(LON.shape, -9999.0, dtype=np.float32)
THICK[mask] = rf.predict(grid_features)

BED = np.full(LON.shape, -9999.0, dtype=np.float32)
BED[mask] = SURF[mask] - THICK[mask]

# 5. Export Georeferenced GeoTIFF Rasters using Rasterio
transform = from_bounds(min_lon, min_lat, max_lon, max_lat, grid_res, grid_res)

# Flip vertically (rasters read from North to South)
thick_raster = np.flipud(THICK)
bed_raster = np.flipud(BED)

# Save Ice Thickness GeoTIFF
thick_tif = os.path.join("gis_outputs", "urumqi_predicted_ice_thickness.tif")
with rasterio.open(
    thick_tif, 'w',
    driver='GTiff', height=grid_res, width=grid_res, count=1,
    dtype='float32', crs='EPSG:4326', transform=transform, nodata=-9999.0
) as dst:
    dst.write(thick_raster, 1)
print(f"[RASTER] Exported Ice Thickness GeoTIFF to: {thick_tif}")

# Save Subglacial Bedrock GeoTIFF
bed_tif = os.path.join("gis_outputs", "urumqi_subglacial_bedrock.tif")
with rasterio.open(
    bed_tif, 'w',
    driver='GTiff', height=grid_res, width=grid_res, count=1,
    dtype='float32', crs='EPSG:4326', transform=transform, nodata=-9999.0
) as dst:
    dst.write(bed_raster, 1)
print(f"[RASTER] Exported Subglacial Bedrock GeoTIFF to: {bed_tif}")

print("\nAll layers successfully generated in 'gis_outputs/' for QGIS!")
