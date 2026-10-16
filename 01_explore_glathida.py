"""
01_explore_glathida.py
----------------------
Explores the global Glacier Thickness Database (GlaThiDa 3.1.0) compiled by the
World Glacier Monitoring Service (WGMS). Inspects surveyed glaciers, regional
distributions, and dense Ground-Penetrating Radar (GPR) survey availability.
"""

import os
import pandas as pd

# Path to GlaThiDa overview table
t_path = os.path.join("..", "Glaciers", "glathida-3.1.0", "data", "T.csv")
if not os.path.exists(t_path):
    t_path = os.path.join("..", "Glaciers", "Data", "glathida-3.1.0", "glathida-3.1.0", "data", "T.csv")

print("=" * 65)
print("     EXPLORING GLATHIDA 3.1.0 GLOBAL GLACIER INVENTORY")
print("=" * 65)

if os.path.exists(t_path):
    df_glaciers = pd.read_csv(t_path, low_memory=False)
    print(f"Total surveyed glaciers cataloged worldwide: {len(df_glaciers)}")
    
    # Filter for glaciers with dense in-situ point surveys
    surveyed = df_glaciers.dropna(subset=['NUMBER_OF_SURVEY_POINTS'])
    print(f"Glaciers with dense in-situ radar point surveys: {len(surveyed)}")
    
    # Glaciers in Asia / China region
    asia_glaciers = surveyed[surveyed['POLITICAL_UNIT'] == 'CN'].sort_values(
        by='NUMBER_OF_SURVEY_POINTS', ascending=False
    )
    print(f"\nTop surveyed glaciers in China (Central Asia / Tien Shan):")
    cols = ['GlaThiDa_ID', 'GLACIER_NAME', 'LAT', 'LON', 'NUMBER_OF_SURVEY_POINTS', 'MAXIMUM_THICKNESS']
    print(asia_glaciers[cols].head(5).to_string(index=False))
else:
    print(f"Note: T.csv overview file not found at {t_path}. Ensure GlaThiDa 3.1.0 is downloaded.")

print("=" * 65)
