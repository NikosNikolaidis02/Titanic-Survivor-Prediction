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

**Key features used:**

| Feature | Description |
|---------|-------------|
| `Pclass` | Ticket class (1st, 2nd, 3rd) — proxy for socio-economic status |
| `Sex` | Gender |
| `Age` | Age in years (median-imputed where missing) |
| `SibSp` | Number of siblings / spouses aboard |
| `Parch` | Number of parents / children aboard |
| `Fare` | Ticket fare (median-imputed where missing) |
| `Embarked` | Port of embarkation (C / Q / S) |
| `HasCabin` | Engineered binary flag — whether a cabin number was recorded |

---

## How to Run

**Requirements:** Python 3.10+, pandas, scikit-learn

```bash
pip install pandas scikit-learn
python main.py
```

**To change which features the model uses**, edit the `FEATURES` list at the top of `main.py` — comment out any features you want to exclude.

---

## Project Structure

```
├── train.csv                  # Training data
├── test.csv                   # Test data (no labels)
├── gender_submission.csv      # Example submission format
├── main.py                    # Main pipeline
└── docs/
    └── competition.md         # Full competition brief
```
