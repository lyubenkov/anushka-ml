from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime
import yaml
from ..models.base_model import BaseModel
from ..models.random_forest_model import RandomForestModel
from ..models.svm_model import SVMModel
from .exceptions import ModelNotFoundError, InvalidHyperparametersError
from ..core import logger

logger = logger.get_logger(__name__)


class ModelManager:
    """Manager class for handling multiple ML models."""

    def __init__(self, config_path: Optional[str] = None, max_active_models: int = 10):
        """Initialize the model manager."""
        self.models: Dict[str, BaseModel] = {}
        self.max_active_models = max_active_models
        self.available_model_classes = {
            "random_forest": RandomForestModel,
            "svm": SVMModel,
        }

        # Load configuration if provided
        self.config = self._load_config(config_path) if config_path else {}
        logger.info("ModelManager initialized successfully")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {str(e)}")
            return {}

    def _validate_data(self, X: np.ndarray, y: np.ndarray):
        """Validate input data."""
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("Features and labels must be numpy arrays")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Number of samples don't match: X has {X.shape[0]}, y has {y.shape[0]}"
            )

        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("Data contains NaN values")

    def get_available_models(self) -> List[str]:
        """Return list of available model types."""
        return list(self.available_model_classes.keys())

    def create_model(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        hyperparameters: Optional[Dict] = None,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create and train a new model."""
        logger.info(f"Creating model of type {model_type}")

        try:
            # Check model limit only if it's a new model (not retraining)
            if (
                model_id not in self.models
                and len(self.models) >= self.max_active_models
            ):
                raise ValueError(
                    f"Maximum number of active models ({self.max_active_models}) reached. "
                    "Please delete some models before creating new ones."
                )
            # Validate model type
            if model_type not in self.available_model_classes:
                raise ValueError(
                    f"Model type {model_type} not available. Choose from {self.get_available_models()}"
                )

            # Validate data
            self._validate_data(X, y)

            # Generate model_id if not provided
            if model_id is None:
                model_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Create and train model
            model_class = self.available_model_classes[model_type]
            model = model_class()

            logger.info(f"Training model {model_id}")
            results = model.fit(X, y, hyperparameters)

            # Store trained model
            self.models[model_id] = model

            logger.info(f"Model {model_id} trained successfully")
            return {"model_id": model_id, "results": results}

        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise

    def predict(self, model_id: str, X: np.ndarray) -> np.ndarray:
        """Make predictions using a trained model."""
        try:
            if model_id not in self.models:
                raise ModelNotFoundError(f"Model {model_id} not found")

            if not isinstance(X, np.ndarray):
                raise ValueError("Features must be a numpy array")

            if np.isnan(X).any():
                raise ValueError("Features contain NaN values")

            logger.info(f"Making predictions with model {model_id}")
            predictions = self.models[model_id].predict(X)
            return predictions

        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

    def delete_model(self, model_id: str) -> None:
        """Delete a trained model."""
        try:
            if model_id not in self.models:
                raise ModelNotFoundError(f"Model {model_id} not found")

            del self.models[model_id]
            logger.info(f"Model {model_id} deleted successfully")

        except Exception as e:
            logger.error(f"Error deleting model: {str(e)}")
            raise
