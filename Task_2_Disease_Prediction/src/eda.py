import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

# 1. Resolve paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.normpath(os.path.join(base_dir, "..", "data"))
csv_path = os.path.join(data_dir, "breast_cancer_data.csv")

# 2. Fetch dataset
cancer = load_breast_cancer(as_frame=True)
df = cancer.frame

# Rename target column for clinical clarity (0: Malignant, 1: Benign)
df.rename(columns={"target": "diagnosis"}, inplace=True)

# 3. Save to data directory for reproducibility
os.makedirs(data_dir, exist_ok=True)
df.to_csv(csv_path, index=False)
print(f"Dataset successfully saved to: {csv_path}\n")

# 4. Exploratory Inspection
print("=" * 50)
print("CLINICAL DATASET SUMMARY")
print("=" * 50)
print(f"Total Patient Records: {df.shape[0]}")
print(f"Diagnostic Features:   {df.shape[1] - 1}\n")

print("--- Class Breakdown ---")
# 0 = Malignant (Tumor), 1 = Benign (Non-tumorous)
counts = df["diagnosis"].value_counts()
print(f"Benign (1):    {counts.get(1, 0)} ({counts.get(1, 0)/len(df)*100:.1f}%)")
print(f"Malignant (0): {counts.get(0, 0)} ({counts.get(0, 0)/len(df)*100:.1f}%)")

print("\n--- Missing Values ---")
print(f"Total null records: {df.isnull().sum().sum()}")

print("\n--- Key Clinical Biomarkers (Sample 3 records) ---")
preview_cols = ["mean radius", "mean texture", "mean perimeter", "mean area", "diagnosis"]
print(df[preview_cols].head(3))