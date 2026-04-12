import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


class SVMModel:
    """SVM classifier wrapper for the Titanic pipeline.

    Uses a Pipeline(StandardScaler → SVC) to ensure features are scaled
    within each CV fold, preventing data leakage.
    """

    def __init__(self, features: list, C: float = 1.0, kernel: str = "rbf", random_state: int = 42):
        self.features = features
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("svc",    SVC(C=C, kernel=kernel, probability=True, random_state=random_state)),
        ])

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the pipeline on training data."""
        self.model.fit(X, y)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Print accuracy on the given dataset."""
        y_pred = self.model.predict(X)
        print("\nAccuracy:", accuracy_score(y, y_pred) * 100)

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> None:
        """Print cross-validation accuracy."""
        scores = cross_val_score(self.model, X, y, cv=cv)
        print(f"\nCross-validation accuracy ({cv}-fold): {scores.mean():.4f} (+/- {scores.std():.4f})")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return predictions for the given dataset."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        """Return class probabilities (needed for soft voting)."""
        return self.model.predict_proba(X)
