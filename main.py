import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# --- Parameters: choose which features to include in the model ---
FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    #"Parch",
    "Fare",
    #"Embarked",
    "HasCabin",
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

# --- Compute imputation statistics from train only ---
impute_stats = {
    "Age": train["Age"].median(),
    "Fare": train["Fare"].median(),
    "Embarked": train["Embarked"].mode()[0],
}

# --- Preprocessing ---
def preprocess(df: pd.DataFrame, features: list, stats: dict) -> pd.DataFrame:
    df = df.copy()

    # Engineer HasCabin before selecting features
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    df = df[features].copy()

    if "Sex" in features:
        df["Sex"] = LabelEncoder().fit_transform(df["Sex"])  # male=1, female=0

    if "Embarked" in features:
        df["Embarked"] = df["Embarked"].fillna(stats["Embarked"])
        df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])

    if "Age" in features:
        df["Age"] = df["Age"].fillna(stats["Age"])

    if "Fare" in features:
        df["Fare"] = df["Fare"].fillna(stats["Fare"])

    return df

X_train = preprocess(train, FEATURES, impute_stats)
y_train = train["Survived"]
X_test = preprocess(test, FEATURES, impute_stats)

# --- Train decision tree ---
model = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=10)
model.fit(X_train, y_train)

# --- Accuracy on training data ---
y_pred_train = model.predict(X_train)
print(f"Training accuracy: {accuracy_score(y_train, y_pred_train):.4f}")

# --- Cross-validation accuracy ---
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"\nCross-validation accuracy (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# --- Feature importance ---
importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature importance:")
print(importance.to_string())

# --- Generate submission file ---
# predictions = model.predict(X_test)
# submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})
# submission.to_csv("submission.csv", index=False)
# print("submission.csv saved.")
