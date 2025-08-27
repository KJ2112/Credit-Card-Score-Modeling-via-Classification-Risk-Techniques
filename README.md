#  Credit Card Score Modeling via Classification & Risk Techniques

This project tackles **credit card default prediction** using billing and payment history.  
It features targeted **feature engineering**, **class imbalance handling**, model benchmarking, and **explainability** to create a **recall-optimized** solution aligned with financial risk mitigation.

---

##  Project Overview

- **Objective:** Predict whether a customer will default in the next month.
- **Dataset:** Tabular data with ~19% default rate (~80–20 class imbalance).
- **Key Techniques:**
  - Feature engineering: credit utilization, pay-to-bill ratios, delinquency flags, trend indicators.
  - Class balancing via **SMOTE** (applied on training data only).
  - Threshold tuning (~0.38) to maximize recall.
  - Model comparisons: Logistic Regression, Random Forest, XGBoost, LightGBM.
  - Explainability using **SHAP**, and visual insights via **Seaborn**.

---

##  Results

- **Chosen Model:** LightGBM
- **Validation Performance:**
  - Recall ≈ 0.82 (high sensitivity to defaulters)
  - Precision ≈ 0.55
  - F2-score ≈ 0.75 (emphasizing recall)
  - ROC–AUC ≈ 0.84

Our focus on recall ensures minimal missed defaulters (false negatives), adhering to the business requirement of early risk detection.

---

##  Tech Stack

- **Python libraries:** pandas, numpy, scikit-learn, imbalanced-learn, lightgbm, xgboost  
- **Visualization:** seaborn, matplotlib  
- **Explainability:** shap  

---

##  Repository Structure

```text
📦 Credit-Card-Score-Modeling-via-Classification-Risk-Techniques
 ┣ LICENSE
 ┣ README.md
 ┣ credit card project.ipynb
 ┣ creditcardreport.pdf
 ┗ data/
   ┣ train.csv (or dataset link)
   ┗ validate.csv
