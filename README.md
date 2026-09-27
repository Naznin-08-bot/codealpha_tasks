```markdown
# Machine Learning Projects Portfolio

A collection of applied machine learning pipelines demonstrating end-to-end data preprocessing, feature engineering, multi-algorithm benchmarking, and domain-specific evaluation metrics.

---

## Projects Included

### 1. [Project 1: Creditworthiness & Default Risk Prediction](./Task_1_Credit_Scoring)
* **Objective:** Predict borrower default risk using demographic and historical credit records.
* **Dataset:** German Credit Risk Benchmark.
* **Techniques:** Feature engineering (monthly debt burden, debt-to-age ratio), data cleaning, handling imbalanced classes with stratified splits.
* **Algorithms:** Logistic Regression, Decision Tree Classifier, Random Forest Classifier.
* **Primary Metrics:** Recall (Default Class) and ROC-AUC.

### 2. [Project 2: Diagnostic Disease Prediction from Medical Data](./Task_2_Disease_Prediction)
* **Objective:** Classify breast tumor fine needle aspirate (FNA) biopsies as benign or malignant.
* **Dataset:** UCI Breast Cancer Wisconsin (Diagnostic) Dataset.
* **Techniques:** Diagnostic feature scaling via `StandardScaler`, stratified train-test splitting.
* **Algorithms:** Logistic Regression, Support Vector Machine (SVM), Random Forest, XGBoost.
* **Primary Metrics:** Recall / Sensitivity (minimizing critical False Negatives) and ROC-AUC.

---

## Repository Structure

```text
ml-portfolio-projects/
├── Task_1_Credit_Scoring/
│   ├── data/
│   │   └── german_credit_data.csv
│   ├── src/
│   │   ├── eda.py
│   │   └── train_models.py
│   └── README.md
├── Task_2_Disease_Prediction/
│   ├── data/
│   │   └── breast_cancer_data.csv
│   ├── src/
│   │   ├── eda.py
│   │   └── train_models.py
│   └── README.md
├── .gitignore
└── README.md