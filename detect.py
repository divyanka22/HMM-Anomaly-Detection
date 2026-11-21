
import numpy as np
import pandas as pd
import joblib

# Load trained HMM
model = joblib.load("models/hmm_model.pkl")

# Load normal training data
normal_data = pd.read_csv("data/normal_sensor_data.csv")
normal_values = normal_data["sensor_value"].values

# Load test data
test_data = pd.read_csv("data/test_sensor_data.csv")
test_values = test_data["sensor_value"].values

# Sliding window settings
window_size = 20
step_size = 5

# Function to calculate window log-likelihood
def calculate_window_scores(values):

    scores = []
    positions = []

    for start in range(
        0,
        len(values) - window_size + 1,
        step_size
    ):

        window = values[
            start:start + window_size
        ].reshape(-1, 1)

        score = model.score(window)

        scores.append(score)
        positions.append(start)

    return np.array(scores), np.array(positions)


# Calculate normal training window scores
normal_scores, _ = calculate_window_scores(
    normal_values
)

# Calculate anomaly threshold
threshold = np.percentile(normal_scores, 1)

print("ANOMALY THRESHOLD:", round(threshold, 3))

# Calculate test window scores
test_scores, positions = calculate_window_scores(
    test_values
)

# Identify anomalous windows
anomaly_flags = test_scores < threshold

print("\nDETECTION RESULTS")
print("Total test windows:", len(test_scores))
print("Anomalous windows:", np.sum(anomaly_flags))
print("Normal windows:", np.sum(~anomaly_flags))

# Save detection results
results = pd.DataFrame({
    "window_start": positions,
    "log_likelihood": test_scores,
    "anomaly_detected": anomaly_flags.astype(int)
})

results.to_csv(
    "data/detection_results.csv",
    index=False
)

print("\nDetection results saved successfully!")
