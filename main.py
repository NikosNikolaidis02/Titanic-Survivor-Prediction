import pandas as pd
from preprocessing import preprocess
from models.decision_tree import DecisionTreeModel
from models.random_forest import RandomForestModel
from models.logistic_regression import LogisticRegressionModel
from models.xgboost_model import XGBoostModel

# --- Parameters: choose which features to include in the model ---
FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    #"Fare",
    #"SibSp",
    #"Parch",
    #"Embarked",
    #"HasCabin",
    #"CabinDeck",
    "Title",
    #"HasSiblings",
    #"AgeGroup",
    #"FareGroup",
    "FamilySizeGroup",
    "FarePerPerson",
]

# --- Load data ---
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# --- Missing values per column ---
print("=== Missing values (train) ===")
print(train.isnull().sum()[train.isnull().sum() > 0].to_string())
print("\n=== Missing values (test) ===")
print(test.isnull().sum()[test.isnull().sum() > 0].to_string())
print()

# --- Preprocessing ---

# Drop rows whose titles never appear in test — n=1/2 each, extreme rates, no generalisation value
TRAIN_ONLY_TITLES = {"Lady", "Countess", "Don", "Jonkheer", "Capt", "Sir", "Major"}
_titles_raw = train["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False).replace(
    {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
)
dropped = _titles_raw.isin(TRAIN_ONLY_TITLES).sum()
train = train[~_titles_raw.isin(TRAIN_ONLY_TITLES)].reset_index(drop=True)
print(f"Dropped {dropped} rows with train-only titles: {TRAIN_ONLY_TITLES}\n")

# Compute imputation statistics from cleaned training data only
age_median = train["Age"].mean()
fare_median = train["Fare"].mean()
embarked_mode = train["Embarked"].mode()[0]

TITLE_MIN_COUNT = 10

# Compute title-based statistics from cleaned training data only
_train = train.copy()
_train["Title"] = _train["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
_train["Title"] = _train["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
title_counts = _train["Title"].value_counts()
rare_titles = set(title_counts[title_counts < TITLE_MIN_COUNT].index)
rare_titles.discard("Rev")  # kept as own category — all Revs in training died
age_by_title = _train.groupby("Title")["Age"].median().to_dict()
print(f"Title counts:\n{title_counts.to_string()}")
print(f"\nRare titles (< {TITLE_MIN_COUNT}): {rare_titles}")
print(f"Rev kept separately (n={title_counts.get('Rev', 0)}, 0% survival)\n")

X_train = preprocess(train, FEATURES, age_median, fare_median, embarked_mode, age_by_title, rare_titles)
y_train = train["Survived"]
X_test = preprocess(test, FEATURES, age_median, fare_median, embarked_mode, age_by_title, rare_titles)

# --- Decision Tree ---
print("--- Decision Tree ---")
dt = DecisionTreeModel(features=FEATURES)
dt.train(X_train, y_train)
dt.cross_validate(X_train, y_train)
dt.feature_importance()

# --- Random Forest ---
print("\n--- Random Forest ---")
rf = RandomForestModel(features=FEATURES)
rf.train(X_train, y_train)
rf.cross_validate(X_train, y_train)
rf.feature_importance()

# --- Logistic Regression ---
print("\n--- Logistic Regression ---")
lr = LogisticRegressionModel(features=FEATURES)
lr.train(X_train, y_train)
lr.cross_validate(X_train, y_train)
lr.feature_importance()

# --- XGBoost ---
print("\n--- XGBoost ---")
xgb = XGBoostModel(features=FEATURES)
xgb.tune(X_train, y_train)
xgb.train(X_train, y_train)
xgb.cross_validate(X_train, y_train)
xgb.feature_importance()

# --- Generate submission file ---
model = xgb  # swap to whichever model performed best
predictions = model.predict(X_test)
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})
submission.to_csv("submission.csv", index=False)
print("submission.csv saved.")
