
import pandas as pd
import matplotlib.pyplot as plt

normal_data = pd.read_csv("data/normal_sensor_data.csv")
test_data = pd.read_csv("data/test_sensor_data.csv")

# Create two graphs side by side
fig, axes = plt.subplots(2, 1, figsize=(12, 9))

# First graph: Normal readings
axes[0].plot(
    normal_data["sensor_value"],
    color="blue",
    label="Normal Sensor Readings"
)
axes[0].set_title("Normal Machine Sensor Readings")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Sensor Value")
axes[0].legend()
axes[0].grid(True)

# Second graph: Test readings
axes[1].plot(
    test_data["sensor_value"],
    color="blue",
    label="Test Sensor Readings"
)

anomalies = test_data[test_data["true_anomaly"] == 1]

axes[1].scatter(
    anomalies.index,
    anomalies["sensor_value"],
    color="red",
    label="Actual Anomalies",
    zorder=5
)

axes[1].set_title("Test Sensor Readings with Actual Anomalies")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Sensor Value")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
