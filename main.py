import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# --- Parameters: choose which features to include in the model ---
FEATURES = [
    #"Pclass",
    "Sex",
    "Age",
    #"SibSp",
    #"Parch",
    "Fare",
    #"Embarked",
    "Cabin",
]

# --- Load data ---
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Handling Missing Data
train = train.dropna()

# --- Preprocessing ---
def preprocess(df: pd.DataFrame, features: list) -> pd.DataFrame:
    df = df[features].copy()

    if "Sex" in features:
        df["Sex"] = LabelEncoder().fit_transform(df["Sex"])  # male=1, female=0
        
    if "Cabin" in features:
        df["Cabin"] = LabelEncoder().fit_transform(df["Cabin"])

    if "Embarked" in features:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
        df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])

    '''
    if "Age" in features:
        df["Age"] = df["Age"].fillna(df["Age"].median())

    if "Fare" in features:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    '''

    return df

X_train = preprocess(train, FEATURES)
y_train = train["Survived"]
X_test = preprocess(test, FEATURES)

# --- Train decision tree ---
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# --- Accuracy on training data ---
y_pred_train = model.predict(X_train)
print(f"Training accuracy: {accuracy_score(y_train, y_pred_train):.4f}")

# --- Feature importance ---
importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature importance:")
print(importance.to_string())

# --- Generate submission file ---
# predictions = model.predict(X_test)
# submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})
# submission.to_csv("submission.csv", index=False)
# print("submission.csv saved.")
