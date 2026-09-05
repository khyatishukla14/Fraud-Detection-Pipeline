# Credit Card Fraud Detection — AI Risk Manager

A fraud **risk scoring and decision system** for merchant transactions. The goal isn't to predict a label in isolation — it's to answer the question a merchant actually cares about: **how much money does this save, and what does it cost in blocked good sales?**

Built on the public Credit Card Fraud dataset (284,807 transactions, 0.17% fraud).

---

## Headline Result

On a held-out set of **85,443 transactions** the model cuts fraud loss by **78%**:

| Policy | Total cost on test set |
|---|---|
| Approve everything (no model) | ₹176,567 |
| **Model @ cost-optimal threshold (0.31)** | **₹38,768** |
| Model @ naive threshold (0.5) | ₹42,189 |
| Block everything | ₹5,354,382 |

Tuning the decision threshold against a **rupee cost function** — instead of accepting the default 0.5 — recovers an extra **₹3,421** that F1-optimization would never have found.

---

## Why this isn't just a classifier

**1. Temporal split, not a random shuffle.**
The dataset is chronologically ordered. A random split trains the model on transactions that happened *after* the ones it's tested on — quiet leakage that inflates every metric. Here the first 70% (by time) is train, the last 30% is test. The model is evaluated exactly the way it would run in production.

**2. PR-AUC as the primary metric, not ROC-AUC.**
At 0.17% prevalence ROC-AUC is actively misleading. In the results below, Logistic Regression posts the *highest* ROC-AUC (0.978) while running at **3% precision** — flagging ~2,800 good transactions for every 97 frauds caught. A merchant selecting on ROC-AUC would ship the worst model on the list.

**3. The threshold is chosen by cost, not by F1.**
Every decision is priced:
- **Missed fraud (FN):** merchant loses the full transaction amount + ₹1,500 chargeback fee
- **False alarm (FP):** blocked good sale — 15% margin lost + ₹50 friction cost

The threshold sweep finds the point that minimizes actual rupees lost on real held-out transactions.

**4. A three-tier policy, not a binary cutoff.**
Real payment risk teams challenge ambiguous transactions rather than declining them, because a hard block costs the entire sale while an OTP costs only friction:

| Decision | Transactions | Actual fraud | Fraud rate |
|---|---|---|---|
| Approve (score < 0.16) | 84,996 | 17 | 0.02% |
| Step-up verification (0.16–0.46) | 356 | 9 | 2.5% |
| Block (score ≥ 0.46) | 91 | 82 | **90.1%** |

The block band is **90% precise** — when this system declines outright, it's almost always right. Only 0.5% of transactions see any friction at all.

**5. SMOTE is tested, not assumed.**
"Apply SMOTE" is the reflex on imbalanced data. Measured honestly inside cross-validation (SMOTE fit *within* each fold via an `imblearn` pipeline, to avoid the leakage of resampling before the split):

- SMOTE + Logistic Regression: **PR-AUC 0.7535**
- `class_weight='balanced'` only: **PR-AUC 0.7507**

A 0.003 difference, well inside the fold-to-fold standard deviation (±0.05). **SMOTE bought nothing measurable here** — it's generating synthetic transaction records for no gain.

---

## Model Comparison

Chronological split, full 85,443-row test set, 108 fraud cases:

| Model | PR-AUC | ROC-AUC | Precision (fraud) | Recall (fraud) |
|---|---|---|---|---|
| **Random Forest** | **0.806** | 0.965 | 0.932 | 0.759 |
| Logistic Regression | 0.781 | 0.978 | 0.030 | 0.898 |
| SVM (20K subsample) | 0.731 | 0.977 | 0.043 | 0.889 |

Top predictive features: `V14` (0.22), `V10` (0.11), `V12` (0.10).

---

## Methodology

1. **EDA** on the imbalanced distribution
2. **Feature engineering** — `Amount_log`, `Hour`; raw `Time` dropped from the feature matrix (it's a dataset artifact, not signal)
3. **Temporal train/test split** — no shuffling
4. **Imbalance handling** — SMOTE vs. `class_weight`, compared inside CV with no leakage
5. **Model comparison** — Logistic Regression, Random Forest, SVM, ranked by PR-AUC
6. **Hyperparameter tuning** — `GridSearchCV` optimizing average precision
7. **Cost-based threshold optimization** in ₹
8. **Three-tier decision policy** — Approve / Step-up / Block

---

## Honest Limitations

- The cost model uses **assumed parameters** (₹1,500 chargeback fee, 15% margin, ₹50 friction). The framework transfers to real merchant economics; the exact rupee figures depend on those inputs.
- Recall at the operating point is **76%** — roughly 1 in 4 frauds still gets through. This reduces loss, it does not eliminate it.
- The test set contains **108 fraud cases**, so metrics carry meaningful variance.
- SVM was trained on a 20,000-row subsample (RBF kernels scale quadratically); LR and RF use the full resampled training set.

---

## Scope

**Defense only.** This is a detection and risk-scoring system trained on a public dataset. It contains no fraud-generation, evasion, or offense-capable component of any kind.

---

## Tech Stack
Python · pandas · NumPy · scikit-learn · imbalanced-learn · Matplotlib · Seaborn

## Project Structure
- `Classification.ipynb` — full pipeline with executed outputs
- `README.md`
