# CodeAlpha Machine Learning Internship — Tasks Portfolio
**Repository Name:** `codealpha_tasks`

This repository contains the machine learning solutions developed for the **CodeAlpha Machine Learning Internship**.

---

## Completed Tasks

### 1. [Task 1: Creditworthiness Prediction](./Task_1_Credit_Scoring)
* **Objective:** Assess borrower creditworthiness to predict potential default risk for lending institutions.
* **Dataset:** German Credit Risk Benchmark.
* **Methodology:** Feature engineering (monthly repayment burden, debt-to-age ratio), data cleaning, handling imbalanced classes with stratified splits.
* **Algorithms:** Logistic Regression, Decision Tree Classifier, Random Forest Classifier.
* **Key Metrics:** Recall (Default class) and ROC-AUC.

### 2. [Task 4: Disease Prediction from Medical Data](./Task_4_Disease_Prediction)
* **Objective:** Classify breast tumor fine needle aspirate (FNA) biopsies as benign or malignant.
* **Dataset:** UCI Breast Cancer Wisconsin (Diagnostic) Dataset.
* **Methodology:** Diagnostic feature scaling via `StandardScaler`, stratified train-test splitting.
* **Algorithms:** Logistic Regression, Support Vector Machine (SVM), Random Forest, XGBoost.
* **Key Metrics:** Recall / Sensitivity (minimizing critical False Negatives) and ROC-AUC.

---

## Repository Structure

```text
codealpha_tasks/
├── Task_1_Credit_Scoring/
│   ├── data/
│   │   └── german_credit_data.csv
│   ├── src/
│   │   ├── eda.py
│   │   └── train_models.py
│   └── README.md
├── Task_4_Disease_Prediction/
│   ├── data/
│   │   └── breast_cancer_data.csv
│   ├── src/
│   │   ├── eda.py
│   │   └── train_models.py
│   └── README.md
├── .gitignore
└── README.md