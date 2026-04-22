import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from preprocessing import preprocess
from models.decision_tree import DecisionTreeModel
from models.random_forest import RandomForestModel
from models.logistic_regression import LogisticRegressionModel
from models.xgboost_model import XGBoostModel
from models.svm_model import SVMModel

# --- Parameters: choose which features to include in the model ---
FEATURES = ["Pclass", "Sex", "Age", "Title", "FamilySizeGroup", "CabinDeck", "IsChild"]

# --- Load data ---
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# --- Drop rows whose titles never appear in test ---
TRAIN_ONLY_TITLES = {"Lady", "Countess", "Don", "Jonkheer", "Capt", "Sir", "Major"}
_titles_raw = train["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False).replace(
    {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
)
train = train[~_titles_raw.isin(TRAIN_ONLY_TITLES)].reset_index(drop=True)

# --- Compute imputation statistics from cleaned training data only ---
age_median = train["Age"].mean()
fare_median = train["Fare"].mean()
embarked_mode = train["Embarked"].mode()[0]

_train = train.copy()
_train["Title"] = _train["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
_train["Title"] = _train["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
age_by_title = _train.groupby("Title")["Age"].median().to_dict()

# Ticket frequency computed on train+test combined — ticket count is a property of
# the ticket itself, not the target, so this does not leak survival labels.
ticket_freq = pd.concat([train["Ticket"], test["Ticket"]]).value_counts().to_dict()

X_train = preprocess(train, FEATURES, age_median, fare_median, embarked_mode, age_by_title, ticket_freq=ticket_freq)
y_train = train["Survived"]
X_test = preprocess(test, FEATURES, age_median, fare_median, embarked_mode, age_by_title, ticket_freq=ticket_freq)

# --- Train models ---
dt = DecisionTreeModel(features=FEATURES)
dt.train(X_train, y_train)

rf = RandomForestModel(features=FEATURES)
rf.train(X_train, y_train)

lr = LogisticRegressionModel(features=FEATURES)
lr.train(X_train, y_train)

svm = SVMModel(features=FEATURES)
svm.train(X_train, y_train)

xgb = XGBoostModel(features=FEATURES)
xgb.train(X_train, y_train)

ensemble = VotingClassifier(
    estimators=[("xgb", xgb.model), ("rf", rf.model)],
    voting="soft",
)
ensemble.fit(X_train, y_train)

# --- Cross-validation scores ---
cv_results = {
    "Decision Tree":       cross_val_score(dt.model,       X_train, y_train, cv=5),
    "Random Forest":       cross_val_score(rf.model,       X_train, y_train, cv=5),
    "Logistic Regression": cross_val_score(lr.model,       X_train, y_train, cv=5),
    "SVM":                 cross_val_score(svm.model,      X_train, y_train, cv=5),
    "XGBoost":             cross_val_score(xgb.model,      X_train, y_train, cv=5),
    "Ensemble (XGB + RF)": cross_val_score(ensemble,       X_train, y_train, cv=5),
}

# --- Print summary table ---
col_model, col_cv, col_std = 22, 13, 9
print(f"{'Model':<{col_model}}  {'CV Accuracy':>{col_cv}}  {'(+/-)':<{col_std}}")
print("-" * (col_model + col_cv + col_std + 4))
for name, scores in cv_results.items():
    print(f"{name:<{col_model}}  {scores.mean():>{col_cv}.4f}  {scores.std():<{col_std}.4f}")
