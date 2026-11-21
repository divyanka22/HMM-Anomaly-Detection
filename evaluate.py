
import pandas as pd

# Load dataset and detection results
df = pd.read_csv("data/test_sensor_data.csv")
results = pd.read_csv("data/detection_results.csv")

# Window configuration used by our detection pipeline
WINDOW_SIZE = 20

# Generate actual labels for each window
actual_labels = []

for start in results["window_start"]:
    start = int(start)
    window = df.iloc[start:start + WINDOW_SIZE]

    # A window is anomalous if it contains an actual anomaly
    actual_labels.append(int(window["true_anomaly"].max() == 1))

# Model predictions
predicted_labels = results["anomaly_detected"].astype(int).tolist()

# Calculate confusion matrix values
TP = sum(a == 1 and p == 1 for a, p in zip(actual_labels, predicted_labels))
TN = sum(a == 0 and p == 0 for a, p in zip(actual_labels, predicted_labels))
FP = sum(a == 0 and p == 1 for a, p in zip(actual_labels, predicted_labels))
FN = sum(a == 1 and p == 0 for a, p in zip(actual_labels, predicted_labels))

# Calculate evaluation metrics
total = TP + TN + FP + FN

accuracy = (TP + TN) / total if total else 0
precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0
f1_score = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

# Display results
print("\nMODEL PERFORMANCE EVALUATION")
print("--------------------------------")
print("Total Windows:", total)
print("True Positives:", TP)
print("True Negatives:", TN)
print("False Positives:", FP)
print("False Negatives:", FN)

print("\nEVALUATION METRICS")
print("--------------------------------")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-Score:  {f1_score * 100:.2f}%")
