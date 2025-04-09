import pandas as pd
import numpy as np

def modified_z_score(data: pd.DataFrame, threshold: float = 3.5):
    """
    Apply the modified Z-score method to detect anomalies in each feature.

    Parameters:
        data (pd.DataFrame): DataFrame with numeric features.
        threshold (float): Threshold to flag anomalies.

    Returns:
        anomalies (pd.DataFrame): Boolean DataFrame where True indicates an anomaly.
        z_scores (pd.DataFrame): Modified Z-scores for each data point.
    """
    median = data.median()
    mad = (data - median).abs().median()
    mad.replace(0, 1e-9, inplace=True)  # Prevent division by zero

    mod_z_scores = 0.6745 * (data - median) / mad
    abs_scores = mod_z_scores.abs()
    anomalies = abs_scores > threshold

    return anomalies, abs_scores


# === Load Data ===
file_path = "cpu_memory_data (1).csv"  # Adjust path if needed
df = pd.read_csv(file_path)

# === Drop timestamp and select numeric features ===
numeric_df = df.drop(columns=["timestamp"])

# === Apply anomaly detection ===
anomaly_flags, z_scores = modified_z_score(numeric_df, threshold=3.5)

# === Count anomalies per feature ===
anomaly_counts = anomaly_flags.sum()
print("Anomalies per feature:\n", anomaly_counts)

# === View sample anomaly rows ===
sample_anomalies = df[anomaly_flags.any(axis=1)].head()
print("\nSample anomalies:\n", sample_anomalies)

# === Optional: Save anomalies to CSV ===
# df[anomaly_flags.any(axis=1)].to_csv("detected_anomalies.csv", index=False)
