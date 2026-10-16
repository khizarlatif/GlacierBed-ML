"""
03_train_kriging_vs_rf.py
-------------------------
Benchmarks Geostatistical Spatial Interpolation (Ordinary Kriging with a
Spherical Variogram) against Machine Learning (Random Forest Regressor)
on 1,386 in-situ Ground-Penetrating Radar (GPR) points from Urumqi Glacier No. 1.
Evaluates prediction error (RMSE, MAE, R²) on an unseen 20% validation split.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from pykrige.ok import OrdinaryKriging

os.makedirs("figures", exist_ok=True)

# 1. Load the clean glacier survey points
data_path = os.path.join("data", "glacier_urumqi_no1_points.csv")
df = pd.read_csv(data_path)
print(f"Loaded {len(df)} survey points for Urumqi Glacier No. 1.")

# 2. Define Features (X) and Target (y)
features = ['POINT_LON', 'POINT_LAT', 'ELEVATION']
X = df[features]
y = df['THICKNESS']

# 3. Split into 80% Training and 20% Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"Training on {len(X_train)} points | Testing on {len(X_test)} unseen points.")

# =============================================================================
# MODEL 1: Ordinary Kriging (Geostatistics)
# =============================================================================
print("\n[1/2] Training Ordinary Kriging (Spherical Variogram)...")
ok_model = OrdinaryKriging(
    X_train['POINT_LON'].values,
    X_train['POINT_LAT'].values,
    y_train.values,
    variogram_model='spherical',
    verbose=False,
    enable_plotting=False
)

# Predict on unseen 20% test points
y_pred_ok, _ = ok_model.execute('points', X_test['POINT_LON'].values, X_test['POINT_LAT'].values)

rmse_ok = root_mean_squared_error(y_test, y_pred_ok)
mae_ok = mean_absolute_error(y_test, y_pred_ok)
r2_ok = r2_score(y_test, y_pred_ok)

# =============================================================================
# MODEL 2: Random Forest Regressor (Machine Learning)
# =============================================================================
print("[2/2] Training Random Forest Regressor (100 Decision Trees)...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on unseen 20% test points
y_pred_rf = rf_model.predict(X_test)

rmse_rf = root_mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# =============================================================================
# SUMMARY REPORT
# =============================================================================
print("\n" + "=" * 65)
print("     BENCHMARK EVALUATION: KRIGING vs RANDOM FOREST")
print("=" * 65)
print(f"{'Metric':<25} | {'Ordinary Kriging':<18} | {'Random Forest':<15}")
print("-" * 65)
print(f"{'RMSE (Mean Error)':<25} | {rmse_ok:.2f} meters{'':<11} | {rmse_rf:.2f} meters")
print(f"{'MAE (Absolute Error)':<25} | {mae_ok:.2f} meters{'':<11} | {mae_rf:.2f} meters")
print(f"{'R² Score (Accuracy)':<25} | {r2_ok:.3f}{'':<17} | {r2_rf:.3f}")
print("=" * 65)

# =============================================================================
# FIGURE: 1-to-1 Scatter Validation Comparison
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Ordinary Kriging
ax1.scatter(y_test, y_pred_ok, color='#1f77b4', edgecolors='black', alpha=0.75, s=35)
ax1.plot([0, 150], [0, 150], 'r--', lw=2, label='1:1 Perfect Prediction')
ax1.set_title(f"Ordinary Kriging\nRMSE = {rmse_ok:.2f} m | R² = {r2_ok:.3f}", fontweight='bold')
ax1.set_xlabel("Actual GPR Thickness (m)", fontweight='bold')
ax1.set_ylabel("Predicted Thickness (m)", fontweight='bold')
ax1.set_xlim(0, 150)
ax1.set_ylim(0, 150)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)

# Panel 2: Random Forest
ax2.scatter(y_test, y_pred_rf, color='#2ca02c', edgecolors='black', alpha=0.75, s=35)
ax2.plot([0, 150], [0, 150], 'r--', lw=2, label='1:1 Perfect Prediction')
ax2.set_title(f"Random Forest Regressor\nRMSE = {rmse_rf:.2f} m | R² = {r2_rf:.3f}", fontweight='bold')
ax2.set_xlabel("Actual GPR Thickness (m)", fontweight='bold')
ax2.set_ylabel("Predicted Thickness (m)", fontweight='bold')
ax2.set_xlim(0, 150)
ax2.set_ylim(0, 150)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
comp_plot = os.path.join("figures", "model_evaluation_kriging_vs_rf.png")
plt.savefig(comp_plot, dpi=300, bbox_inches='tight')
print(f"\nSaved evaluation comparison plot to: {comp_plot}")
plt.show()
