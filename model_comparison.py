import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import numpy as np
import time

# Load the dataset
data = pd.read_csv('water_footprint.csv')

# Feature selection
features = data.drop(columns=['WaterFootprint'])
target = data['WaterFootprint']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize models
models = {
    'XGBoost': xgb.XGBRegressor(objective='reg:linear', eta=0.05, max_depth=6),
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Support Vector Machine': SVR(kernel='rbf'),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    accuracy = 100 - mape
    training_time = time.time() - start_time
    
    # Store results
    results[name] = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'Accuracy': accuracy,
        'Training Time (s)': training_time
    }

# Print comparison table
print("\nModel Comparison Results:")
print("=" * 120)
print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R2 Score':<10} {'MAPE (%)':<10} {'Accuracy (%)':<10} {'Time (s)':<10}")
print("-" * 120)

for name, metrics in results.items():
    print(f"{name:<20} {metrics['RMSE']:.4f}    {metrics['MAE']:.4f}    {metrics['R2']:.4f}    {metrics['MAPE']:.2f}%      {metrics['Accuracy']:.2f}%      {metrics['Training Time (s)']:.2f}")

print("=" * 120)

# Find the best model based on accuracy
best_model_name = max(results.items(), key=lambda x: x[1]['Accuracy'])[0]
print(f"\nBest performing model: {best_model_name} with {results[best_model_name]['Accuracy']:.2f}% accuracy")

# Save the best model if it's XGBoost
if best_model_name == 'XGBoost':
    models['XGBoost'].save_model('waterfootprint_xgboost_model.json')
    print("Saved the best model (XGBoost) to 'waterfootprint_xgboost_model.json'") 