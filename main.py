import pandas as pd
from preprocessing import compute_imputation_stats, preprocess
from models.decision_tree import DecisionTreeModel

# --- Parameters: choose which features to include in the model ---
FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    #"SibSp",
    #"Parch",
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
dt = DecisionTreeModel(features=FEATURES)
dt.train(X_train, y_train)
dt.evaluate(X_train, y_train)
dt.cross_validate(X_train, y_train)
dt.feature_importance()

# --- Generate submission file ---
# predictions = dt.predict(X_test)
# submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})
# submission.to_csv("submission_dt.csv", index=False)
# print("submission_dt.csv saved.")
