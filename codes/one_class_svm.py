import pandas as pd
from sklearn.svm import OneClassSVM
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

# === Apply One-Class SVM ===
model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.01)
model.fit(numeric_df)
pred_labels = model.predict(numeric_df)
pred_labels = (pred_labels == -1).astype(int)

# === Print metrics ===
precision = precision_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)

print("📊 One-Class SVM Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
