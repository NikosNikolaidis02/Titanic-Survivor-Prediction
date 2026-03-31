import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score


class LogisticRegressionModel:
    """Logistic Regression classifier wrapper for the Titanic pipeline."""

    def __init__(self, features: list, max_iter: int = 1000, random_state: int = 42):
        self.features = features
        self.model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model on training data."""
        self.model.fit(X, y)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Print confusion matrix, accuracy, and classification report."""
        y_pred = self.model.predict(X)
        #print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
        print("\nAccuracy:", accuracy_score(y, y_pred) * 100)
        #print("\nClassification Report:\n", classification_report(y, y_pred))

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> None:
        """Print cross-validation accuracy."""
        scores = cross_val_score(self.model, X, y, cv=cv)
        print(f"\nCross-validation accuracy ({cv}-fold): {scores.mean():.4f} (+/- {scores.std():.4f})")

    def feature_importance(self) -> None:
        """Print feature coefficients ranked by absolute value."""
        importance = pd.Series(
            self.model.coef_[0], index=self.features
        ).abs().sort_values(ascending=False)
        print("\nFeature coefficients (absolute value):")
        print(importance.to_string())

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return predictions for the given dataset."""
        return self.model.predict(X)
