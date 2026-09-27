import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

# Models requested in brief
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 1. Load Data
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.normpath(os.path.join(base_dir, "..", "data", "breast_cancer_data.csv"))
df = pd.read_csv(csv_path)

# Malignant = 1 (positive condition of interest), Benign = 0
# Original dataset has 0: Malignant, 1: Benign; invert so 1 means High Risk / Cancerous
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"].apply(lambda val: 1 if val == 0 else 0)

# 2. Stratified Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Model Zoo Setup
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Support Vector Machine (RBF Kernel)": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}

print("=" * 65)
print("DISEASE PREDICTION MODEL BENCHMARK (MALIGNANT DETECTION)")
print("=" * 65)

# 4. Pipeline Execution & Scoring
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n>>> Model: {name}")
    print(f"ROC-AUC Score: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Benign (0)", "Malignant (1)"]))
    print("-" * 65)