import pandas as pd
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# === Load the dataset ===
file_path = "cpu_memory_data_with_anomalies.csv"
df = pd.read_csv(file_path)

# === Check for ground truth column ===
if "is_anomaly" not in df.columns:
    raise ValueError("Ground truth column 'is_anomaly' is missing in the dataset.")

# === Prepare data ===
features = df.drop(columns=["timestamp", "is_anomaly"])
true_labels = df["is_anomaly"]

# === Scale the data ===
scaled = StandardScaler().fit_transform(features)

# === Apply HDBSCAN ===
clusterer = hdbscan.HDBSCAN(min_cluster_size=50)
cluster_labels = clusterer.fit_predict(scaled)
pred_labels = (cluster_labels == -1).astype(int)  # -1 is noise

# === Print metrics ===
precision = precision_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)

print("📊 HDBSCAN Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# === Optional: View anomaly sample ===
print("\nSample detected anomalies:")
print(df[pred_labels == 1].head())

# === Optional: Save to CSV ===
# df[pred_labels == 1].to_csv("hdbscan_anomalies.csv", index=False)
