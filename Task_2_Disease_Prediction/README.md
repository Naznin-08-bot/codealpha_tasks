# Disease Prediction from Medical Data
**Machine Learning Portfolio — Project 2**

## Project Overview
This project develops an automated diagnostic classification model to predict whether a breast tumor is benign or malignant using patient biopsy measurements.

## Dataset & Attributes
* **Source:** UCI Breast Cancer Wisconsin (Diagnostic) Benchmark
* **Target Classes:** `0` = Benign (Non-cancerous), `1` = Malignant (Cancerous / High Risk)
* **Features:** 30 continuous clinical features derived from cellular nuclei images (radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, fractal dimension).

## Models Evaluated
1. **Logistic Regression** (Standardized baseline)
2. **Support Vector Machine (SVM)** (RBF Kernel)
3. **Random Forest Classifier**
4. **XGBoost Classifier**

## Clinical Evaluation
In medical diagnostics, **Recall for Malignant cases (Sensitivity)** is prioritized. Missing a malignant case (False Negative) delays critical treatment, whereas a false alarm (False Positive) can be verified via secondary diagnostics.

## Execution
```bash
python Task_2_Disease_Prediction/src/train_models.py