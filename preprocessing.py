import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess(
    df: pd.DataFrame,
    features: list,
    age_median: float = None,
    fare_median: float = None,
    embarked_mode: str = None,
    age_by_title: dict = None,
    rare_titles: set = None,
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
        age_by_title  : Dict mapping Title → median Age (computed from train). When provided,
                        imputes missing Age per title group; falls back to age_median.
        rare_titles   : Set of title strings to consolidate into "Rare". Computed from train
                        using a frequency threshold; applied after alias normalisation.

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

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["FamilySizeGroup"] = pd.cut(
        df["FamilySize"],
        bins=[0, 1, 4, 20],
        labels=[0, 1, 2],  # 0=Solo, 1=Regular(2–4), 2=Big(5+)
    ).astype(int)

    # --- Title extraction ---
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    # Normalise aliases before anything else
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    # --- Age imputation (title-based if available, global median fallback) ---
    # Uses pre-consolidation titles for more accurate per-title medians
    if age_by_title:
        df["Age"] = df.apply(
            lambda row: age_by_title.get(row["Title"], age_median) if pd.isna(row["Age"]) else row["Age"],
            axis=1,
        )
    else:
        df["Age"] = df["Age"].fillna(age_median)

    # --- Consolidate rare titles (after age imputation to preserve per-title accuracy) ---
    if rare_titles:
        df["Title"] = df["Title"].apply(lambda t: "Rare" if t in rare_titles else t)

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 20, 35, 50, 75, 100],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)

    fare = df["Fare"].fillna(fare_median)
    df["FareGroup"] = pd.cut(
        fare,
        bins=[-1, 8, 15, 31, 513],
        labels=[0, 1, 2, 3],
    ).astype(int)
    df["FarePerPerson"] = fare / df["FamilySize"]

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
        df["Age"] = df["Age"].fillna(age_median)  # safety fallback

    if "Fare" in features:
        df["Fare"] = df["Fare"].fillna(fare_median)

    if "Title" in features:
        # Fixed map covering all titles present after train-only row removal.
        # Unseen titles in test (e.g. Dona) fall back to 7.
        title_map = {"Master": 0, "Miss": 1, "Mr": 2, "Mrs": 3, "Rev": 4, "Dr": 5, "Col": 6}
        df["Title"] = df["Title"].map(title_map).fillna(7).astype(int)

    return df
