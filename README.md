# Credit Card Fraud Detection - ML Pipeline

This project builds a complete **machine learning classification pipeline** on the imbalanced Credit Card Fraud dataset. It follows industry-standard practices with preprocessing, feature engineering, resampling, and model evaluation.

---

## Dataset
- **Source**: Credit Card Fraud dataset (Kaggle)
- **Target**: `Class` (0 = Non-Fraud, 1 = Fraud)
- Highly imbalanced: Fraudulent transactions are very rare compared to normal ones.

---

## Methodology
1. **Exploratory Data Analysis (EDA)** with visualizations  
2. **Feature Engineering** (log transformation of `Amount`, extracting transaction hour)  
3. **Preprocessing Pipeline** (scaling, handling imbalance with SMOTE)  
4. **Model Training & Comparison**: Logistic Regression, Random Forest, and SVM  
5. **Evaluation** using Precision, Recall, F1-Score, and ROC-AUC  
6. **Feature Importance Analysis** for interpretability  

---

## Tech Stack
- Python  
- Pandas, NumPy  
- scikit-learn  
- imbalanced-learn (SMOTE)  
- Matplotlib, Seaborn  

---

## Results
- Random Forest achieved the best balance of **Precision, Recall, and ROC-AUC**.  
- Feature engineering and SMOTE significantly improved fraud detection performance.  

---

## 📂 Project Structure
 - classification.ipynb   
 - README.md               
