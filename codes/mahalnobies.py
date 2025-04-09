import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import precision_score, recall_score, f1_score

# === Load the dataset ===
file_path = "cpu_memory_data_with_anomalies.csv"
df = pd.read_csv(file_path)

# === Check for ground truth column ===
if "is_anomaly" not in df.columns:
    raise ValueError("Ground truth column 'is_anomaly' is missing in the dataset.")

# === Prepare data ===
numeric_df = df.drop(columns=["timestamp", "is_anomaly"])
true_labels = df["is_anomaly"]

# === Compute Mahalanobis distance ===
cov_matrix = np.cov(numeric_df, rowvar=False)
inv_cov_matrix = np.linalg.inv(cov_matrix)
mean_vector = numeric_df.mean().values

distances = numeric_df.apply(lambda row: mahalanobis(row, mean_vector, inv_cov_matrix), axis=1)
threshold = distances.mean() + 3 * distances.std()
pred_labels = (distances > threshold).astype(int)

# === Print metrics ===
precision = precision_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)

print("📊 Mahalanobis Distance Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# === Optional: View anomaly sample ===
print("\nSample detected anomalies:")
print(df[pred_labels == 1].head())

# === Optional: Save to CSV ===
# df[pred_labels == 1].to_csv("mahalanobis_anomalies.csv", index=False)
