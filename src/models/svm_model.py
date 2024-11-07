from typing import Dict, Any, Optional
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from .base_model import BaseModel


class SVMModel(BaseModel):
    """Support Vector Machine model implementation."""

    def __init__(self):
        self.model = None
        self.best_params = None
        self.best_score = None

    def get_default_hyperparameters(self) -> Dict[str, Any]:
        return {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        }

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = (
            hyperparameters if hyperparameters else self.get_default_hyperparameters()
        )
        base_model = SVC()

        grid_search = GridSearchCV(base_model, params, cv=5, n_jobs=-1, verbose=1)

        grid_search.fit(X, y)

        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_

        return {"best_params": self.best_params, "best_score": self.best_score}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)

    def get_model_name(self) -> str:
        return "svm"

    def get_model_params(self) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model must be fitted before getting parameters")
        return {"best_params": self.best_params, "best_score": self.best_score}
