# Disease Prediction from Medical Data
**CodeAlpha Machine Learning Internship — Task 4**

## Project Overview
This project develops a diagnostic machine learning pipeline to classify breast tumor masses as either malignant or benign based on digitized fine needle aspirate (FNA) measurements.

## Clinical Context & Dataset
* **Dataset:** UCI Breast Cancer Wisconsin (Diagnostic) Dataset
* **Target Classes:** 
  * `0`: Benign (Non-cancerous)
  * `1`: Malignant (Cancerous / High Risk)
* **Clinical Features:** 30 continuous real-valued cellular biomarkers computed from digitized images (mean radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension).

## Models Benchmarked
1. **Logistic Regression:** Linear probabilistic baseline with standard scaling.
2. **Support Vector Machine (SVM):** High-dimensional separation using an RBF kernel.
3. **Random Forest Classifier:** Non-linear decision ensemble.
4. **XGBoost Classifier:** Sequential gradient boosting optimizing classification log-loss.

## Evaluation Priority
In clinical disease diagnostics, **Recall for Malignant cases (Sensitivity)** is the primary metric. Missing a cancerous mass (False Negative) has severe health consequences, whereas a false alarm (False Positive) can be resolved with follow-up testing.

## Execution
```bash
python Task_4_Disease_Prediction/src/train_models.py