from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BaseModel(ABC):
    """Abstract base class for all ML models in the API."""

    @abstractmethod
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """Return default hyperparameters for the model."""
        pass

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Fit the model with given hyperparameters.

        Args:
            X: Training features
            y: Training labels
            hyperparameters: Optional custom hyperparameters

        Returns:
            Dict containing training results
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.

        Args:
            X: Features to predict on

        Returns:
            Predicted values
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of the model."""
        pass

    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """Return the current model parameters."""
        pass
