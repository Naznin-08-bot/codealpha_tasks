import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "..", "data", "german_credit_data.csv")

df = pd.read_csv(dataset_path)

# Look for the target column
possible_targets = [col for col in df.columns if col.lower() in ['status', 'risk', 'credit_risk', 'class']]
print("Detected Target Column(s):", possible_targets)

target = possible_targets[0] if possible_targets else df.columns[-1]
print(f"\nUsing Target Column: '{target}'")
print("\nTarget Value Counts:")
print(df[target].value_counts(dropna=False))
print("\nPercentage Distribution:")
print(df[target].value_counts(normalize=True) * 100)