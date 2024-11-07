from typing import Dict, Any, Optional, Union
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from .base_model import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest model implementation."""

    def __init__(self):
        self.model = None
        self.best_params = None
        self.best_score = None

    def get_default_hyperparameters(self) -> Dict[str, list]:
        return {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, 30],
            "min_samples_split": [2, 5, 10],
        }

    def fit(
        self, X: np.ndarray, y: np.ndarray, hyperparameters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        params = (
            hyperparameters if hyperparameters else self.get_default_hyperparameters()
        )
        base_model = RandomForestClassifier(random_state=42)

        grid_search = GridSearchCV(base_model, params, cv=5, n_jobs=-1, verbose=1)

        grid_search.fit(X, y)

        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = float(
            grid_search.best_score_
        ) 

        # Convert numpy types to Python types for JSON serialization
        best_params_json = {}
        for key, value in self.best_params.items():
            if isinstance(value, np.integer):
                best_params_json[key] = int(value)
            elif isinstance(value, np.floating):
                best_params_json[key] = float(value)
            else:
                best_params_json[key] = value

        return {"best_params": best_params_json, "best_score": self.best_score}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        predictions = self.model.predict(X)
        return predictions.astype(float)

    def get_model_name(self) -> str:
        return "random_forest"

    def get_model_params(self) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model must be fitted before getting parameters")
        return {"best_params": self.best_params, "best_score": self.best_score}
