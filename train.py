
import numpy as np
import pandas as pd
import joblib
import os

from hmmlearn.hmm import GaussianHMM

# Load normal sensor data
data = pd.read_csv("data/normal_sensor_data.csv")
X = data["sensor_value"].values.reshape(-1, 1)

print("Training data shape:", X.shape)

state_options = [2, 3, 4]
seeds = range(10)

best_model = None
best_bic = float("inf")
best_states = None

print("\nMODEL SELECTION RESULTS")

for n_states in state_options:

    best_candidate = None
    best_log_likelihood = -float("inf")

    # Try 10 different initializations
    for seed in seeds:

        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=300,
            random_state=seed
        )

        model.fit(X)
        log_likelihood = model.score(X)

        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_candidate = model

    # Calculate BIC for the best run
    n_parameters = n_states ** 2 + 2 * n_states - 1

    bic = (
        -2 * best_log_likelihood
        + n_parameters * np.log(len(X))
    )

    print(
        f"Hidden States: {n_states}, "
        f"Best Log Likelihood: {best_log_likelihood:.2f}, "
        f"BIC: {bic:.2f}"
    )

    if bic < best_bic:
        best_bic = bic
        best_model = best_candidate
        best_states = n_states

# Save the selected model
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/hmm_model.pkl")

print("\nBEST MODEL SELECTED")
print("Number of hidden states:", best_states)
print("Best BIC:", round(best_bic, 2))
print("Model saved successfully!")
