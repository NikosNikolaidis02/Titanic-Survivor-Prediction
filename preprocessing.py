import pandas as pd
from sklearn.preprocessing import LabelEncoder


def compute_imputation_stats(train: pd.DataFrame) -> dict:
    """Compute imputation statistics from training data only."""
    return {
        "Age": train["Age"].mean(),
        "Fare": train["Fare"].mean(),
        "Embarked": train["Embarked"].mode()[0],
    }


def preprocess(df: pd.DataFrame, features: list, stats: dict) -> pd.DataFrame:
    """
    Apply feature engineering and preprocessing.

    Engineered features available:
        - HasCabin    : 1 if cabin number was recorded, else 0
        - HasSiblings : 1 if passenger had siblings/spouses aboard, else 0
        - AgeGroup    : 0=Child(<20), 1=Teen(20-35), 2=Adult(35-50), 3=Middle-aged(50-75), 4=Senior(75+)
        - FareGroup   : 0=Low(Q1), 1=Medium(Q2), 2=High(Q3), 3=Premium(Q4)

    Args:
        df       : Raw dataframe (train or test).
        features : List of feature column names to include in the output.
        stats    : Imputation statistics computed from training data.

    Returns:
        Preprocessed dataframe with only the selected features.
    """
    df = df.copy()

    # --- Feature engineering ---
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    df["HasSiblings"] = (df["SibSp"] > 0).astype(int)

    age = df["Age"].fillna(stats["Age"])
    df["AgeGroup"] = pd.cut(
        age,
        bins=[0, 20, 35, 50, 75, 100],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)

    fare = df["Fare"].fillna(stats["Fare"])
    df["FareGroup"] = pd.cut(
        fare,
        bins=[-1, 8, 15, 31, 513],
        labels=[0, 1, 2, 3],
    ).astype(int)

    # --- Select features ---
    df = df[features].copy()

    # --- Encoding and imputation ---
    if "Sex" in features:
        df["Sex"] = LabelEncoder().fit_transform(df["Sex"])  # female=0, male=1

    if "Embarked" in features:
        df["Embarked"] = df["Embarked"].fillna(stats["Embarked"])
        df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])

    if "Age" in features:
        df["Age"] = df["Age"].fillna(stats["Age"])

    if "Fare" in features:
        df["Fare"] = df["Fare"].fillna(stats["Fare"])

    return df
