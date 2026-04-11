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
| `Title` | Engineered | Extracted from passenger name (Mr, Mrs, Miss, Master, Rev, Rare) |
| `FamilySizeGroup` | Engineered | 0=Solo, 1=Regular (2–4), 2=Large (5+) |
| `AgeGroup` | Engineered | Age binned into 5 groups (Child / Teen / Adult / Middle-aged / Senior) |
| `HasCabin` | Engineered | 1 if a cabin number was recorded, else 0 |

---

## Methodology

### Data Cleaning
Seven title categories present in training but **absent from test** (`Lady`, `Countess`, `Sir`, `Capt`, `Don`, `Jonkheer`, `Major`) were removed from training. Each had n=1–2 occurrences with extreme survival rates (0% or 100%) driven by passenger class, not the title itself — keeping them would introduce noise the model can never generalise from.

### Feature Engineering

- **Title** — extracted from `Name` via regex. Aliases normalised (`Mlle→Miss`, `Mme→Mrs`). Titles with fewer than 10 occurrences collapsed into `"Rare"`, with one exception: `Rev` is kept as its own category because all 6 Reverends in the training data died (0% survival rate), which is a meaningful and consistent pattern.
- **Age imputation** — missing Age filled using per-title median (e.g. Master→4.5, Mr→30, Mrs→35) rather than the global mean, preserving age signal within each demographic group.
- **FamilySizeGroup** — `SibSp + Parch + 1` bucketed into Solo / Regular / Large. Small families survived at higher rates than solo travellers or very large groups.
- **HasCabin** — binary flag; cabin records are concentrated in 1st class and correlate with survival.

### Models
Four classifiers trained and evaluated with 5-fold cross-validation:
- Decision Tree, Random Forest, Logistic Regression, XGBoost
- XGBoost tuned with `RandomizedSearchCV` (50 iterations) over `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`

---

## Results

| Model | CV Accuracy (5-fold) |
|-------|----------------------|
| **XGBoost (tuned)** | **0.8474** |
| Decision Tree | 0.8272 |
| Random Forest | 0.8249 |
| Logistic Regression | 0.7879 |

Best XGBoost parameters: `max_depth=6`, `learning_rate=0.05`, `n_estimators=100`, `subsample=0.8`, `colsample_bytree=0.8`.

---

## How to Run

**Requirements:** Python 3.10+

```bash
pip install pandas scikit-learn xgboost
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
│   ├── logistic_regression.py  # Logistic Regression wrapper
│   └── xgboost_model.py        # XGBoost wrapper with hyperparameter tuning
└── docs/
    └── competition.md          # Full competition brief
```
