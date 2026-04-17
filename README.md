# Titanic Survivor Prediction

A machine learning project built for the [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic) — the introductory challenge where the goal is to predict which passengers survived the shipwreck based on passenger data.

This is part of a structured learning path toward AI Engineering, focusing on building end-to-end ML pipelines with scikit-learn.

---

## Problem Statement

On April 15, 1912, the RMS Titanic sank after hitting an iceberg, killing 1,502 of the 2,224 passengers and crew on board. The question this model attempts to answer is:

> *Given a passenger's characteristics — class, sex, age, and family size — can we predict whether they survived?*

This is a **binary classification** problem. The target variable is `Survived` (1 = survived, 0 = deceased).

---

## Dataset

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 891 | Labelled passenger data used for training |
| `test.csv` | 418 | Unlabelled data — predictions submitted to Kaggle |

**Features used:**

| Feature | Type | Description |
|---------|------|-------------|
| `Pclass` | Raw | Ticket class (1st, 2nd, 3rd) — proxy for socio-economic status |
| `Sex` | Raw | Gender (label-encoded: female=0, male=1) |
| `Age` | Raw | Age in years — imputed per title-group median where missing |
| `Title` | Engineered | Extracted from passenger name (Mr, Mrs, Miss, Master, Rev, Dr, Col) |
| `FamilySizeGroup` | Engineered | 0=Solo, 1=Regular (2–4), 2=Large (5+) |
| `CabinDeck` | Engineered | First letter of cabin number (A–G, T); "U" if unknown — fixed map encoded |
| `IsChild` | Engineered | 1 if passenger age < 12, else 0 |

---

## Methodology

### Data Cleaning
Seven title categories present in training but **absent from test** (`Lady`, `Countess`, `Sir`, `Capt`, `Don`, `Jonkheer`, `Major`) were removed from training. Each had n=1–2 occurrences with extreme survival rates (0% or 100%) driven by passenger class, not the title itself — keeping them would introduce noise the model can never generalise from.

### Feature Engineering

- **Title** — extracted from `Name` via regex. Aliases normalised (`Mlle→Miss`, `Mme→Mrs`). Encoded with a fixed map to prevent train/test category misalignment.
- **Age imputation** — missing Age filled using per-title median (e.g. Master→4.5, Mr→30, Mrs→35) rather than the global median, preserving age signal within each demographic group.
- **FamilySizeGroup** — `SibSp + Parch + 1` bucketed into Solo / Regular / Large. Small families survived at higher rates than solo travellers or very large groups.
- **CabinDeck** — first letter of the cabin number, with "U" for unknown. Encoded with a fixed map (A=0…U=8); correlates with passenger class and survival.
- **IsChild** — binary flag for passengers under 12; children had higher survival rates ("women and children first").

### Models
Five classifiers trained and evaluated with 5-fold cross-validation:
- Decision Tree, Random Forest, Logistic Regression (scaled), SVM (scaled), XGBoost
- Final submission uses a **soft-voting ensemble of XGBoost + Random Forest**

---

## Results

| Model | CV Accuracy (5-fold) |
|-------|----------------------|
| **Ensemble (XGB + RF)** | **0.8449** |
| **XGBoost** | **0.8449** |
| SVM | 0.8290 |
| Random Forest | 0.8234 |
| Decision Tree | 0.8143 |
| Logistic Regression | 0.7973 |

---

## How to Run

**Requirements:** Python 3.10+

### Install dependencies

```bash
pip install pandas scikit-learn xgboost
```

### Run the pipeline

```bash
py main.py
```

**To change which features the model uses**, edit the `FEATURES` list at the top of `main.py` — comment out any features you want to exclude.

---

## Project Structure

```
├── train.csv                   # Training data
├── test.csv                    # Test data (no labels)
├── main.py                     # Main pipeline — loads, cleans, preprocesses, trains, evaluates
├── preprocessing.py            # Feature engineering and preprocessing pipeline
├── EDA-Analysis.ipynb          # Exploratory data analysis
├── models/
│   ├── decision_tree.py        # Decision Tree wrapper
│   ├── random_forest.py        # Random Forest wrapper
│   ├── logistic_regression.py  # Logistic Regression wrapper (StandardScaler pipeline)
│   ├── svm_model.py            # SVM wrapper (StandardScaler pipeline)
│   └── xgboost_model.py        # XGBoost wrapper with hyperparameter tuning
└── docs/
    └── competition.md          # Full competition brief
```
