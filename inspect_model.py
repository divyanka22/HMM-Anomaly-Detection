
import joblib
import numpy as np

# Load the trained HMM
model = joblib.load("models/hmm_model.pkl")

print("NUMBER OF HIDDEN STATES:")
print(model.n_components)

print("\nLEARNED STATE MEANS:")
print(model.means_.flatten())

print("\nLEARNED STATE STANDARD DEVIATIONS:")
print(np.sqrt(model.covars_.reshape(-1)))

print("\nSTATE TRANSITION PROBABILITIES:")
print(np.round(model.transmat_, 3))
