# Import necessary libraries
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import sys
from numpy.lib.stride_tricks import sliding_window_view

# ----------------------------- #
#       Argument Parsing        #
# ----------------------------- #

# Read command-line arguments
model_name = sys.argv[1]      # Model type: 'mlp', 'lstm', or 'cnn'
attack_name = sys.argv[2]     # Attack type: 'fgsm', 'bim', 'pgd', 'random', or 'original'
dataset_name = sys.argv[3]    # Name of CSV file (excluding extension); must be in 'dataset/' directory

# ----------------------------- #
#       Threshold Settings      #
# ----------------------------- #

# Predefined anomaly score thresholds for each model type
threshold = {
    "mlp": 0.9178198173696142,
    "lstm": 0.7413586421503263,
    "cnn": 0.9214007646727854
}

# ----------------------------- #
#     Sliding Window Utility    #
# ----------------------------- #

def prepare_input_data(timeseries_data, n_features):
    """
    Generates input-output pairs using a sliding window over time-series data.
    """
    windows = sliding_window_view(timeseries_data, window_shape=n_features)
    windows = windows[:-1]  # Drop last window (no corresponding y)
    y = timeseries_data[n_features:]
    return windows, y

# ----------------------------- #
#       Prediction Utility      #
# ----------------------------- #

def predict(test_X, model):
    """
    Predict future values using the loaded model.
    """
    return model.predict(test_X)

# ----------------------------- #
#      Score Computation        #
# ----------------------------- #

def compute_scores(y, pred_y):
    """
    Computes anomaly scores based on prediction error trends.
    """
    diff = abs(pred_y - y)
    cumsums = diff.cumsum().reshape(-1, 1)
    cumsums = cumsums - diff
    numbers = np.arange(cumsums.shape[0]).reshape(-1, 1)
    avgs = np.divide(cumsums, numbers, where=numbers != 0, out=np.full_like(cumsums, np.nan))
    scores = diff / avgs
    return scores

# ----------------------------- #
#         Main Pipeline         #
# ----------------------------- #

# Fixed input window size used by the model (e.g., 60 minutes)
n_steps = 60

# Load dataset
df = pd.read_csv(f'dataset/{dataset_name}.csv')

# Prepare sliding window input-output pairs
test_X, test_y = prepare_input_data(df['consumption'], n_steps)
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1])

# Load pre-trained model
forecasting_model = load_model(f"models/{model_name}_{attack_name}")

# Make predictions
pred_y = predict(np.asarray(test_X).astype(np.float32), forecasting_model)

# Save predictions into DataFrame starting from the first predicted point (after n_steps)
df.loc[n_steps:, "pred"] = pred_y

# Convert test_y to correct shape for score computation
test_y = test_y.values.reshape(-1, 1)

# Compute anomaly scores
scores = compute_scores(test_y, pred_y)
df.loc[n_steps:, "scores"] = scores

# Generate binary anomaly labels based on model-specific threshold
anomalies = (scores > threshold[model_name]).reshape(-1,)
df.loc[n_steps:, "anomalies"] = anomalies

# Save modified DataFrame with predictions and anomalies
df.to_csv(f"{dataset_name}_modified.csv", index=False)
