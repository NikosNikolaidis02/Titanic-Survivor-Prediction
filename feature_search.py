import pandas as pd
from itertools import combinations
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from preprocessing import preprocess

# --- Load and clean data (mirrors main.py exactly) ---
train = pd.read_csv("train.csv")

TRAIN_ONLY_TITLES = {"Lady", "Countess", "Don", "Jonkheer", "Capt", "Sir", "Major"}
_titles_raw = train["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False).replace(
    {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
)
train = train[~_titles_raw.isin(TRAIN_ONLY_TITLES)].reset_index(drop=True)

age_median    = train["Age"].mean()
fare_median   = train["Fare"].mean()
embarked_mode = train["Embarked"].mode()[0]

_train = train.copy()
_train["Title"] = _train["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
_train["Title"] = _train["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
age_by_title = _train.groupby("Title")["Age"].median().to_dict()

y = train["Survived"]

# --- Feature pool ---
POOL = [
    "Pclass", "Sex", "Age", "Title", "FamilySizeGroup",
    "FarePerPerson", "CabinDeck", "HasCabin", "Embarked", "AgeGroup",
]

# --- XGBoost with best params found earlier (no tuning per combination) ---
model = XGBClassifier(
    max_depth=6, learning_rate=0.05, n_estimators=100,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
    random_state=42, eval_metric="logloss", verbosity=0,
)

# --- Exhaustive search over all combinations of 2+ features ---
total = sum(1 for r in range(2, len(POOL) + 1) for _ in combinations(POOL, r))
print(f"Testing {total} combinations...\n")

best_score    = 0
best_features = None
best_std      = None
done          = 0

for r in range(2, len(POOL) + 1):
    for feature_set in combinations(POOL, r):
        features = list(feature_set)
        X = preprocess(train, features, age_median, fare_median, embarked_mode, age_by_title)
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        if scores.mean() > best_score:
            best_score    = scores.mean()
            best_features = features
            best_std      = scores.std()
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{total} checked | best so far: {best_score:.4f} {best_features}")

print(f"\n{'='*60}")
print(f"Best CV accuracy : {best_score:.4f} (+/- {best_std:.4f})")
print(f"Best feature set : {best_features}")
print(f"{'='*60}")
