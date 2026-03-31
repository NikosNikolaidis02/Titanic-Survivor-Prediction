import pandas as pd
from preprocessing import compute_imputation_stats, preprocess
from models.decision_tree import DecisionTreeModel
from models.random_forest import RandomForestModel
from models.logistic_regression import LogisticRegressionModel
from models.xgboost_model import XGBoostModel

# --- Parameters: choose which features to include in the model ---
FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    #"SibSp",
    "Parch",
    #"Embarked",
    "HasCabin",
    #"HasSiblings",
    #"AgeGroup",
    #"FareGroup",
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
impute_stats = compute_imputation_stats(train)
X_train = preprocess(train, FEATURES, impute_stats)
y_train = train["Survived"]
X_test = preprocess(test, FEATURES, impute_stats)

# --- Decision Tree ---
print("--- Decision Tree ---")
dt = DecisionTreeModel(features=FEATURES)
dt.train(X_train, y_train)
dt.evaluate(X_train, y_train)
dt.feature_importance()

# --- Random Forest ---
print("\n--- Random Forest ---")
rf = RandomForestModel(features=FEATURES)
rf.train(X_train, y_train)
rf.evaluate(X_train, y_train)
rf.feature_importance()

# --- Logistic Regression ---
print("\n--- Logistic Regression ---")
lr = LogisticRegressionModel(features=FEATURES)
lr.train(X_train, y_train)
lr.evaluate(X_train, y_train)
lr.feature_importance()

# --- XGBoost ---
print("\n--- XGBoost ---")
xgb = XGBoostModel(features=FEATURES)
xgb.train(X_train, y_train)
xgb.evaluate(X_train, y_train)
xgb.feature_importance()

# --- Generate submission file ---
# model = xgb  # swap to whichever model performed best
# predictions = model.predict(X_test)
# submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})
# submission.to_csv("submission.csv", index=False)
# print("submission.csv saved.")
