# Creditworthiness Prediction Using Financial Data
**CodeAlpha Machine Learning Internship — Task 1**

## Project Overview
This project builds a credit scoring classification model to assess borrower creditworthiness using historical financial records. The model predicts the likelihood of financial default to assist lending institutions in mitigating credit risk.

## Dataset & Domain Context
* **Dataset:** German Credit Risk Benchmark
* **Target Variable:** Binary Classification (`0` = Safe/Low Risk, `1` = Default/High Risk)
* **Key Features:** Credit history, loan duration, loan amount, installment rate, employment duration, savings, and account status.

## Feature Engineering
To reflect financial capacity and debt burden, the following indicators were engineered:
* **Monthly Financial Burden:** Loan amount divided by duration to capture periodic repayment pressure.
* **Debt-to-Age Exposure:** Credit amount relative to borrower age, assessing leverage risk across career stages.
* **Categorical Handling:** Encoding banking indicators with explicit imputation for missing records.

## Models Benchmarked
To counter class imbalance inherent to loan default datasets, algorithms were configured with balanced class weighting:
1. **Logistic Regression:** Linear baseline providing interpretable risk coefficients.
2. **Decision Tree Classifier:** Non-linear rule-based estimator.
3. **Random Forest Classifier:** Bagging ensemble model to reduce variance and capture complex interactions.

## Evaluation Strategy
In credit risk, **Recall for Default Risk (Class 1)** and **ROC-AUC** take priority over simple accuracy. Failing to identify an actual defaulter (False Negative) results in direct capital write-offs, whereas rejecting a good applicant (False Positive) only incurs lost interest opportunity.

## Project Structure
```text
CodeAlpha_Credit_Scoring/
├── data/
│   └── german_credit_data.csv
├── notebooks/
├── src/
│   ├── eda.py
│   └── train_models.py
├── .gitignore
└── README.md
