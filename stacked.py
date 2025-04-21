import pandas as pd
import numpy as np
import glob
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load and combine CSVs
csv_files = glob.glob("ML_Training/*.csv")
df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)
combined_df = combined_df.dropna()

# Feature engineering
combined_df['delta_lat'] = combined_df['WP Lat'] - combined_df['Lat']
combined_df['delta_lon'] = combined_df['WP Lon'] - combined_df['Lon']
combined_df['relative_wind'] = combined_df['Wind Angle (deg)'] - combined_df['Heading (deg)']
combined_df['Speed (m/s)'] = combined_df['Speed (knots)'] * 0.514444  # If you have Speed (knots) in your data

# Define features and targets
features = [
    'Heading Error (deg)',
    'Distance to WP (m)',
    'Wind Angle (deg)',
    'Wind Speed (knots)'
]
targets = ['RZ Thrust', 'Forward Thrust']

X = combined_df[features].values
y = combined_df[targets].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# ---------- Random Forest ----------
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on train and test
rf_train_preds = rf_model.predict(X_train)
rf_test_preds = rf_model.predict(X_test)

# ---------- Stacked MLP ----------
X_train_stacked = np.hstack((X_train, rf_train_preds))
X_test_stacked = np.hstack((X_test, rf_test_preds))

mlp_model = MLPRegressor(
    hidden_layer_sizes=(64, 64),
    activation='relu',
    max_iter=10000,
    learning_rate='adaptive',
    alpha=0.001,
    early_stopping=True,
    verbose=True,
    random_state=42
)

mlp_model.fit(X_train_stacked, y_train)

# Plot loss
plt.figure(figsize=(8,5))
plt.plot(mlp_model.loss_curve_)
plt.title('MLP Training Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
# Predict and Evaluate
y_pred = mlp_model.predict(X_test_stacked)
print("Stacked MLP R2 Score:", r2_score(y_test, y_pred))
print("Stacked MLP MSE:", mean_squared_error(y_test, y_pred))

# ---------- Save Models ----------
joblib.dump(rf_model, "ML_Models/rf_model.pkl")
joblib.dump(mlp_model, "ML_Models/stacked_mlp_model.pkl")
joblib.dump(scaler, "ML_Models/feature_scaler.pkl")

print("Models saved successfully!")
