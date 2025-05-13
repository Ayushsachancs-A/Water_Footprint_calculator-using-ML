import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Load the dataset
data = pd.read_csv('water_footprint.csv')

# Feature selection
features = data.drop(columns=['WaterFootprint'])
target = data['WaterFootprint']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create DMatrix for XGBoost
train_data = xgb.DMatrix(X_train, label=y_train)
test_data = xgb.DMatrix(X_test, label=y_test)

# Define parameters
params = {
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'eta': 0.05,
    'max_depth': 6
}

# Train the model
evals = [(test_data, 'eval'), (train_data, 'train')]
model = xgb.train(params, train_data, num_boost_round=1000, evals=evals, early_stopping_rounds=10)

# Predict and evaluate
y_pred = model.predict(test_data)

# Calculate various metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calculate MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
accuracy = 100 - mape

# Print evaluation metrics
print("\nModel Evaluation Metrics:")
print("=" * 50)
print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared Score: {r2:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Model Accuracy: {accuracy:.2f}%")
print("=" * 50)

# Save the model
model.save_model('waterfootprint_xgboost_model.json')
