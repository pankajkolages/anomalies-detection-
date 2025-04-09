import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def iqr_anomaly_detection(data: pd.DataFrame, factor: float = 1.5):
    """
    Detect anomalies using the IQR method for each numeric column.

    Parameters:
        data (pd.DataFrame): Numeric data
        factor (float): Multiplier for IQR to define outlier range

    Returns:
        anomalies (pd.DataFrame): Boolean DataFrame where True indicates an anomaly
        iqr_bounds (dict): Lower and upper bounds for each column
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    anomalies = (data < lower_bound) | (data > upper_bound)

    iqr_bounds = {
        col: (lower_bound[col], upper_bound[col]) for col in data.columns
    }

    return anomalies, iqr_bounds


# === Load the dataset ===
file_path = "cpu_memory_data_with_anomalies.csv"  # Adjust if needed
df = pd.read_csv(file_path)

# === Check for ground truth column ===
if "is_anomaly" not in df.columns:
    raise ValueError("Ground truth column 'is_anomaly' is missing in the dataset.")

# === Prepare data ===
numeric_df = df.drop(columns=["timestamp", "is_anomaly"])
true_labels = df["is_anomaly"]

# === Apply IQR anomaly detection ===
anomaly_flags, bounds = iqr_anomaly_detection(numeric_df, factor=1.5)

# === Combine anomaly results from all features (any=True) ===
pred_labels = anomaly_flags.any(axis=1).astype(int)

# === Print metrics ===
precision = precision_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)

print("📊 IQR-Based Anomaly Detection Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# === Optional: View anomaly sample ===
print("\nSample detected anomalies:")
print(df[pred_labels == 1].head())

# === Optional: Save to CSV ===
# df[pred_labels == 1].to_csv("iqr_anomalies_with_labels.csv", index=False)
