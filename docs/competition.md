# Titanic - Machine Learning from Disaster

## Overview

Use machine learning to predict which passengers survived the Titanic shipwreck.

**Platform:** Kaggle
**Type:** Binary Classification
**Metric:** Accuracy (percentage of passengers correctly predicted)

---

## The Challenge

On April 15, 1912, the RMS Titanic sank after colliding with an iceberg during her maiden voyage. Of 2,224 passengers and crew, 1,502 died. While survival had some element of luck, certain groups of people were more likely to survive than others.

**Goal:** Build a predictive model that answers — *"what sorts of people were more likely to survive?"* — using passenger data (name, age, gender, socio-economic class, etc.).

---

## Data

| File | Description |
|------|-------------|
| `train.csv` | 891 passengers with known survival outcomes (ground truth) |
| `test.csv` | 418 passengers — predict survival for these |
| `gender_submission.csv` | Example submission file |

### Available Features

- `PassengerId` — Unique passenger identifier
- `Survived` — Target variable (0 = deceased, 1 = survived) *(train only)*
- `Pclass` — Ticket class (1st, 2nd, 3rd) — proxy for socio-economic status
- `Name` — Passenger name
- `Sex` — Gender
- `Age` — Age in years
- `SibSp` — Number of siblings/spouses aboard
- `Parch` — Number of parents/children aboard
- `Ticket` — Ticket number
- `Fare` — Passenger fare
- `Cabin` — Cabin number
- `Embarked` — Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

---

## Evaluation

- **Metric:** Accuracy — percentage of passengers correctly classified
- **Task:** Predict `0` (deceased) or `1` (survived) for each passenger in `test.csv`

---

## Submission Format

A CSV file with exactly **418 entries + 1 header row**, two columns only:

```
PassengerId,Survived
892,0
893,1
894,0
...
```

**Limits:** 10 submissions per day.
