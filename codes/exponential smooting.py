import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import precision_score, recall_score, f1_score

# === Load the dataset ===
file_path = "cpu_memory_data_with_anomalies.csv"
df = pd.read_csv(file_path)

# === Check for ground truth column ===
if "is_anomaly" not in df.columns:
    raise ValueError("Ground truth column 'is_anomaly' is missing in the dataset.")

true_labels = df["is_anomaly"]
cpu = df["cpu_utilization"]

# === Apply Exponential Smoothing ===
model = ExponentialSmoothing(cpu, trend=None, seasonal=None).fit()
fitted = model.fittedvalues
residuals = cpu - fitted

# === Identify anomalies based on residuals
threshold = residuals.std() * 3
pred_labels = (abs(residuals) > threshold).astype(int)

# === Print metrics ===
precision = precision_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)

print("📊 Exponential Smoothing Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# === Optional: View anomaly sample ===
print("\nSample detected anomalies:")
print(df[pred_labels == 1].head())

# === Optional: Save to CSV ===
# df[pred_labels == 1].to_csv("exp_smoothing_anomalies.csv", index=False)
