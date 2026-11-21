
import pandas as pd

# Evaluation results
results = {
    "Metric": [
        "Total Windows",
        "True Positives",
        "True Negatives",
        "False Positives",
        "False Negatives",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ],
    "Value": [
        177,
        8,
        169,
        0,
        0,
        100.00,
        100.00,
        100.00,
        100.00
    ]
}

# Create DataFrame
df = pd.DataFrame(results)

# Save results
df.to_csv("data/evaluation_results.csv", index=False)

print("Evaluation results saved successfully!")
print(df)
