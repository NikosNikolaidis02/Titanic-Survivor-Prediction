import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, RandomizedSearchCV


class XGBoostModel:
    """XGBoost classifier wrapper for the Titanic pipeline."""

    def __init__(self, features: list, n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.05, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, min_child_weight: int = 1,
                 random_state: int = 42):
        self.features = features
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
        )

    def tune(self, X: pd.DataFrame, y: pd.Series, n_iter: int = 50, cv: int = 5, verbose: bool = True) -> None:
        """Search for best hyperparameters and update self.model with the best estimator."""
        param_grid = {
            "n_estimators":     [100, 200, 300, 500],
            "max_depth":        [3, 4, 5, 6],
            "learning_rate":    [0.01, 0.05, 0.1, 0.2],
            "subsample":        [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
        }
        search = RandomizedSearchCV(
            self.model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="accuracy",
            cv=cv,
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X, y)
        if verbose:
            print(f"Best CV accuracy: {search.best_score_:.4f}")
            print(f"Best params: {search.best_params_}")
        self.model = search.best_estimator_

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
        """Print features ranked by importance."""
        importance = pd.Series(
            self.model.feature_importances_, index=self.features
        ).sort_values(ascending=False)
        print("\nFeature importance:")
        print(importance.to_string())

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return predictions for the given dataset."""
        return self.model.predict(X)
