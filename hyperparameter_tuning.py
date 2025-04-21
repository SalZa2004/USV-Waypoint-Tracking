from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import glob
import pandas as pd

csv_files = glob.glob("wind_data/*.csv")
# Combine all into one DataFrame
df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)
combined_df = combined_df.dropna()
# Add computed features
combined_df['delta_lat'] = combined_df['WP Lat'] - combined_df['Lat']
combined_df['delta_lon'] = combined_df['WP Lon'] - combined_df['Lon']
combined_df['relative_wind'] = combined_df['Wind Angle (deg)'] - combined_df['Heading (deg)']
features = ['Heading Error (deg)', 'Distance to WP (m)', 'Speed (knots)', 'relative_wind','Wind Speed (knots)']
targets = ['RZ Thrust', 'Forward Thrust']
X = combined_df[features].values
y = combined_df[targets].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# Your features and labels

# Define pipeline
pipeline = make_pipeline(
    StandardScaler(),
    MLPRegressor(max_iter=8000, early_stopping=True, random_state=42)
)

# Define parameter grid
param_grid = {
    'mlpregressor__hidden_layer_sizes': [(64, 64), (128, 64), (128, 128)],
    'mlpregressor__activation': ['relu', 'tanh'],
    'mlpregressor__learning_rate': ['adaptive', 'constant'],
    'mlpregressor__alpha': [0.0001, 0.001, 0.01]  # L2 regularization
}

# Grid search
grid = GridSearchCV(pipeline, param_grid, cv=4, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best score:", -grid.best_score_)