
import pandas as pd

# Load both datasets
normal_data = pd.read_csv("data/normal_sensor_data.csv")
test_data = pd.read_csv("data/test_sensor_data.csv")

# Display normal dataset
print("NORMAL DATASET")
print(normal_data.head())
print(normal_data.describe())

# Display test dataset
print("\nTEST DATASET")
print(test_data.head())
print(test_data.describe())

# Count actual anomalies
print("\nANOMALY LABEL COUNTS")
print(test_data["true_anomaly"].value_counts())
