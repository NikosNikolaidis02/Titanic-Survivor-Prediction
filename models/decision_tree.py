import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score


class DecisionTreeModel:
    """Decision Tree classifier wrapper for the Titanic pipeline."""

    def __init__(self, features: list, max_depth: int = 5, min_samples_leaf: int = 10, random_state: int = 42):
        self.features = features
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model on training data."""
        self.model.fit(X, y)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Print confusion matrix, accuracy, and classification report."""
        y_pred = self.model.predict(X)
        print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
        print("\nAccuracy:", accuracy_score(y, y_pred) * 100)
        print("\nClassification Report:\n", classification_report(y, y_pred))

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> None:
        """Print cross-validation accuracy."""
        scores = cross_val_score(self.model, X, y, cv=cv)
        print(f"\nCross-validation accuracy ({cv}-fold): {scores.mean():.4f} (+/- {scores.std():.4f})")

    def feature_importance(self) -> None:
        """Print features ranked by importance."""
        importance = pd.Series(self.model.feature_importances_, index=self.features).sort_values(ascending=False)
        print("\nFeature importance:")
        print(importance.to_string())

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return predictions for the given dataset."""
        return self.model.predict(X)
