# 💳 Credit Card Score Modeling via Classification & Risk Techniques

This project predicts **credit card default risk** using customer billing and payment history.  
It applies **feature engineering, class imbalance handling, model benchmarking, and explainability techniques** to build a recall-optimized solution suitable for financial risk mitigation.

---

## 🚀 Project Overview
- **Objective:** Predict whether a customer will default in the next month.  
- **Data:** Tabular dataset with ~19% default cases (~80–20 imbalance).  
- **Techniques Used:**
  - Feature engineering (credit utilization, pay-to-bill ratio, late payment flags, etc.)
  - SMOTE resampling to balance classes
  - Threshold tuning (~0.38) for recall optimization
  - Model benchmarking: Logistic Regression, Random Forest, XGBoost, LightGBM
  - Explainability with SHAP values
  - Visualization of financial behavior with Seaborn/Matplotlib

---

## 📊 Results
- **Best Model:** LightGBM  
- **Metrics (validation set):**
  - Recall ≈ 0.82  
  - Precision ≈ 0.55  
  - F2 Score ≈ 0.75  
  - ROC–AUC ≈ 0.84  

📌 Focused on recall to minimize false negatives (undetected defaulters), aligning with financial risk priorities.

---

## 🛠 Tech Stack
- **Python** (pandas, numpy, scikit-learn, lightgbm, xgboost, imbalanced-learn)  
- **Visualization:** seaborn, matplotlib  
- **Explainability:** shap  

