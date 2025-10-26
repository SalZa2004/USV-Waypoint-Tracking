from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pandas as pd
import glob
import joblib

# Load and prepare data
scaler = StandardScaler()
csv_files = glob.glob("wind_data/*.csv")
df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True).dropna()

# Feature Engineering
combined_df['delta_lat'] = combined_df['WP Lat'] - combined_df['Lat']
combined_df['delta_lon'] = combined_df['WP Lon'] - combined_df['Lon']
combined_df['relative_wind'] = combined_df['Wind Angle (deg)'] - combined_df['Heading (deg)']

features = ['XTE (m)','Heading Error (deg)', 'Distance to WP (m)','Wind Angle (deg)', 'Wind Speed (knots)']
targets = ['RZ Thrust', 'Forward Thrust']

X = combined_df[features].values
y = combined_df[targets].values
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# Train Random Forest
xgb_model = xgb.XGBRegressor(tree_method="hist", device="cuda")
xgb_model.fit(X_train, y_train)

# Evaluation
y_pred = xgb_model.predict(X_test)
print("XGBoost R2 Score:", r2_score(y_test, y_pred))
print("Random Forest MSE:", mean_squared_error(y_test, y_pred))

# Save model and scaler
joblib.dump(xgb_model, "ML_Models/xgb_trained.pkl")
joblib.dump(scaler, "ML_Models/xgb_scaler.pkl")