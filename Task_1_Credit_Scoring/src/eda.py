import os
import pandas as pd

# Find the path to the CSV file automatically
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "..", "data", "german_credit_data.csv")

print(f"Loading data from: {dataset_path}\n")

# Load data
df = pd.read_csv(dataset_path)

print("=" * 40)
print("SUCCESS: DATASET LOADED")
print("=" * 40)
print(f"Total Rows: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}\n")

print("--- Column List ---")
print(list(df.columns))

print("\n--- First 3 Records ---")
print(df.head(3))