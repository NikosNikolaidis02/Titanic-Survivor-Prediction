import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess(
    df: pd.DataFrame,
    features: list,
    age_median: float = None,
    fare_median: float = None,
    embarked_mode: str = None,
) -> pd.DataFrame:
    """
    Apply feature engineering and preprocessing.
    Imputation statistics should be computed from the training set and passed in
    to avoid leaking test distribution into imputed values.

    Engineered features available:
        - HasCabin    : 1 if cabin number was recorded, else 0
        - CabinDeck   : First letter of Cabin (A–G, T); "U" if unknown — label encoded
        - HasSiblings : 1 if passenger had siblings/spouses aboard, else 0
        - AgeGroup    : 0=Child(<20), 1=Teen(20-35), 2=Adult(35-50), 3=Middle-aged(50-75), 4=Senior(75+)
        - FareGroup   : 0=Low(Q1), 1=Medium(Q2), 2=High(Q3), 3=Premium(Q4)

    Args:
        df            : Raw dataframe (train or test).
        features      : List of feature column names to include in the output.
        age_median    : Imputation value for Age; computed from df if not provided.
        fare_median   : Imputation value for Fare; computed from df if not provided.
        embarked_mode : Imputation value for Embarked; computed from df if not provided.

    Returns:
        Preprocessed dataframe with only the selected features.
    """
    df = df.copy()

    # --- Imputation statistics (use training stats if provided) ---
    if age_median is None:
        age_median = df["Age"].mean()
    if fare_median is None:
        fare_median = df["Fare"].mean()
    if embarked_mode is None:
        embarked_mode = df["Embarked"].mode()[0]

    # --- Feature engineering ---
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    df["CabinDeck"] = df["Cabin"].str[0].fillna("U")

    df["HasSiblings"] = (df["SibSp"] > 0).astype(int)

    age = df["Age"].fillna(age_median)
    df["AgeGroup"] = pd.cut(
        age,
        bins=[0, 20, 35, 50, 75, 100],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)

    fare = df["Fare"].fillna(fare_median)
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
        df["Embarked"] = df["Embarked"].fillna(embarked_mode)
        df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])

    if "CabinDeck" in features:
        df["CabinDeck"] = LabelEncoder().fit_transform(df["CabinDeck"])

    if "Age" in features:
        df["Age"] = df["Age"].fillna(age_median)

    if "Fare" in features:
        df["Fare"] = df["Fare"].fillna(fare_median)

    return df
