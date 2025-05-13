import pandas as pd
import numpy as np

# Define the size of the dataset
size = 100000

# Generate random data
data = {
    'Average faucet durations (minutes)': np.random.randint(0, 31, size),
    'Faucet pressure (low/high)': np.random.randint(0, 2, size),
    'Uses dishwasher (yes/no)': np.random.randint(0, 2, size),
    'Times dishwasher used daily': np.random.randint(0, 5, size),
    'Times toilet flushed daily': np.random.randint(0, 21, size),
    'Toilet type (low flow/dual flush)': np.random.randint(0, 2, size),
    'Times showered daily': np.random.randint(0, 4, size),
    'Average shower duration (minutes)': np.random.randint(0, 61, size),
    'Has garden (yes/no)': np.random.randint(0, 2, size),
    'Garden type': np.random.randint(0, 3, size),
    'Times garden watered weekly': np.random.randint(0, 8, size),
    'Times clothes washed weekly': np.random.randint(0, 15, size),
    'Washing machine type': np.random.randint(0, 2, size),
    'Times mopped weekly': np.random.randint(0, 8, size),
    'Mopping method': np.random.randint(0, 2, size),
    'Vehicle type': np.random.randint(0, 2, size),
    'Times vehicle washed weekly': np.random.randint(0, 8, size),
    'Has RO (yes/no)': np.random.randint(0, 2, size),
    'House size (square feet)': np.random.randint(500, 5001, size),
    'Household members': np.random.randint(1, 11, size)
}

# Calculate daily water footprint
def calculate_water_footprint(row):
    faucet_flow_rate = 6 if row['Faucet pressure (low/high)'] == 0 else 9
    toilet_water_usage = 4.5 if row['Toilet type (low flow/dual flush)'] == 0 else 9
    garden_water_usage = 0 if row['Garden type'] == 0 else (20 if row['Garden type'] == 1 else 30)
    washing_machine_usage = 100 if row['Washing machine type'] == 0 else 60
    mopping_usage = 10 if row['Mopping method'] == 0 else 5
    vehicle_wash_usage = 50 if row['Vehicle type'] == 1 else 200
    ro_waste_usage = 10 if row['Has RO (yes/no)'] == 1 else 0
    
    daily_water_footprint = (
        row['Average faucet durations (minutes)'] * faucet_flow_rate +
        row['Uses dishwasher (yes/no)'] * row['Times dishwasher used daily'] * 15 +
        row['Times toilet flushed daily'] * toilet_water_usage +
        row['Times showered daily'] * row['Average shower duration (minutes)'] * 10 +
        row['Has garden (yes/no)'] * garden_water_usage * row['Times garden watered weekly'] / 7 +
        row['Times clothes washed weekly'] * washing_machine_usage / 7 +
        row['Times mopped weekly'] * mopping_usage / 7 +
        row['Times vehicle washed weekly'] * vehicle_wash_usage / 7 +
        ro_waste_usage
    )
    return daily_water_footprint

# Apply the calculation to each row
df = pd.DataFrame(data)
df['WaterFootprint'] = df.apply(calculate_water_footprint, axis=1)

# Save to CSV
df.to_csv('water_footprint.csv', index=False)

print("Dataset generated and saved as 'water_footprint.csv'")
