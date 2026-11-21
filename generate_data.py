
import numpy as np
import pandas as pd
import os

# Make results reproducible
np.random.seed(42)

# Create data folder
os.makedirs("data", exist_ok=True)

# Generate normal machine sensor readings
normal_readings = []

# Simulate three operating states
states = [
    (30, 2),
    (50, 3),
    (70, 2.5)
]

for mean, std in states:
    readings = np.random.normal(
        loc=mean,
        scale=std,
        size=1000
    )
    normal_readings.extend(readings)

normal_data = pd.DataFrame({
    "sensor_value": normal_readings
})

normal_data.to_csv(
    "data/normal_sensor_data.csv",
    index=False
)

# Generate test sensor readings
test_readings = []

for mean, std in states:
    readings = np.random.normal(
        loc=mean,
        scale=std,
        size=300
    )
    test_readings.extend(readings)

# Introduce unusual sensor readings
anomaly_positions = [350, 351, 352, 700, 701, 702]

for position in anomaly_positions:
    test_readings[position] = np.random.normal(110, 3)

test_data = pd.DataFrame({
    "sensor_value": test_readings
})

# Save actual anomaly labels for evaluation
test_data["true_anomaly"] = 0
test_data.loc[anomaly_positions, "true_anomaly"] = 1

test_data.to_csv(
    "data/test_sensor_data.csv",
    index=False
)

print("Dataset generation completed!")
print("Normal dataset shape:", normal_data.shape)
print("Test dataset shape:", test_data.shape)
print("Number of actual anomalies:", len(anomaly_positions))
