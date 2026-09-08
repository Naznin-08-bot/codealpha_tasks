import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# -----------------------------------------------------------------
# 1. Load Data & Apply Feature Engineering
# -----------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "german_credit_data.csv")
df = pd.read_csv(data_path)

# Drop redundant index column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Identify Target Column (Standard Kaggle German Credit uses 'Risk' or 'status')
target_col = "Risk" if "Risk" in df.columns else "status"

# Encode target: 1 for Bad (Defaulter/High Risk), 0 for Good
if df[target_col].dtype == object:
    df["target"] = df[target_col].map({"bad": 1, "good": 0})
else:
    # If already numerical (e.g., 1 and 2 or 0 and 1)
    df["target"] = df[target_col].apply(lambda x: 1 if x in [1, "bad"] else 0)

df = df.drop(columns=[target_col])

# Fill missing categorical values if present
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].fillna("Unknown")

# Feature Engineering
if "amount" in df.columns and "duration" in df.columns:
    df["monthly_burden"] = df["amount"] / (df["duration"] + 1e-5)
if "age" in df.columns and "amount" in df.columns:
    df["amount_to_age_ratio"] = df["amount"] / df["age"]

# -----------------------------------------------------------------
# 2. Train-Test Split (Stratified)
# -----------------------------------------------------------------
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Identify numerical and categorical columns
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# -----------------------------------------------------------------
# 3. Build Preprocessor Pipeline
# -----------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features)
    ]
)

# -----------------------------------------------------------------
# 4. Define Candidate Models
# -----------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", max_depth=5, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        class_weight="balanced", n_estimators=100, max_depth=6, random_state=42
    )
}

# -----------------------------------------------------------------
# 5. Train and Evaluate
# -----------------------------------------------------------------
print("=" * 60)
print("CREDITWORTHINESS MODEL BENCHMARKING RESULTS")
print("=" * 60)

for name, model in models.items():
    clf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    
    clf_pipeline.fit(X_train, y_train)
    y_pred = clf_pipeline.predict(X_test)
    y_prob = clf_pipeline.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\nModel: {name}")
    print(f"ROC-AUC Score: {auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Good Credit (0)", "Default Risk (1)"]))
    print("-" * 60)