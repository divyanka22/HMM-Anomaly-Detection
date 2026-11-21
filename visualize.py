
import pandas as pd
import matplotlib.pyplot as plt

# Load test dataset and detection results
df = pd.read_csv("data/test_sensor_data.csv")
results = pd.read_csv("data/detection_results.csv")

# Extract sensor readings
sensor_values = df["sensor_value"]
true_anomalies = df["true_anomaly"]

# Create graph
plt.figure(figsize=(16, 7))

# Plot original sensor readings
plt.plot(
    sensor_values.index,
    sensor_values,
    label="Sensor Readings",
    color="blue",
    linewidth=1
)

# Highlight actual anomalies
actual_indices = df.index[true_anomalies == 1]

plt.scatter(
    actual_indices,
    sensor_values.iloc[actual_indices],
    color="red",
    label="Actual Anomalies",
    s=70,
    zorder=3
)

# Highlight HMM-detected windows
window_size = 20

for i, row in results[results["anomaly_detected"] == 1].iterrows():
    start = int(row["window_start"])
    end = min(start + window_size, len(df))

    plt.axvspan(
        start,
        end,
        color="orange",
        alpha=0.18,
        label="HMM Detected Windows" if i == results[
            results["anomaly_detected"] == 1
        ].index[0] else ""
    )

# Graph details
plt.title("HMM-Based Sensor Anomaly Detection")
plt.xlabel("Data Point / Time Index")
plt.ylabel("Sensor Value")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Save graph
plt.savefig("anomaly_detection_visualization.png", dpi=300)

plt.show()

print("Visualization saved successfully!")
