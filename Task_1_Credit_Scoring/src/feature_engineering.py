import os
import pandas as pd

def load_and_engineer_features():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "..", "data", "german_credit_data.csv")
    
    df = pd.read_csv(dataset_path)
    
    # 1. Clean column names (strip whitespace and lowercase)
    df.columns = df.columns.str.strip().str.lower()
    
    # 2. Identify the target column
    target_col = 'status' if 'status' in df.columns else ('risk' if 'risk' in df.columns else df.columns[-1])
    
    # Standardize target: 1 for High Risk / Default, 0 for Low Risk / Good
    # In German credit data, 1 often means bad/default or 0/1 indicator
    if df[target_col].dtype == object:
        df['target'] = df[target_col].apply(lambda x: 1 if str(x).lower() in ['bad', '1', 'default'] else 0)
    else:
        # If already numerical, keep as binary 0/1
        df['target'] = (df[target_col] == df[target_col].max()).astype(int)
        
    df = df.drop(columns=[target_col])
    
    # 3. Domain Feature Engineering
    # Metric A: Monthly installment burden (amount divided by duration in months)
    if 'amount' in df.columns and 'duration' in df.columns:
        df['monthly_burden'] = df['amount'] / df['duration']
        
    # Metric B: Credit loan amount relative to borrower age
    if 'amount' in df.columns and 'age' in df.columns:
        df['credit_to_age_ratio'] = df['amount'] / df['age']

    return df

if __name__ == "__main__":
    processed_df = load_and_engineer_features()
    print("Feature engineering complete.")
    print(f"Engineered Dataset Shape: {processed_df.shape}")
    print(processed_df[['monthly_burden', 'credit_to_age_ratio', 'target']].head())