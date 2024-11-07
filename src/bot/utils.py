import pandas as pd
import numpy as np
from io import BytesIO
import json
from typing import Tuple, Dict, Any
import requests
from ..core.logger import get_logger

logger = get_logger(__name__)


class MLBotClient:
    """Client for interacting with ML API from the bot."""

    def __init__(self, api_url: str):
        self.api_url = api_url

    def get_available_models(self) -> Dict[str, Any]:
        """Get available models from API."""
        response = requests.get(f"{self.api_url}/models/available")
        return response.json()

    def train_model(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        model_type: str,
        hyperparameters: Dict = None,
    ) -> Dict[str, Any]:
        """Train a new model."""
        data = {
            "features": features.tolist(),
            "labels": labels.tolist(),
            "model_type": model_type,
            "hyperparameters": hyperparameters,
        }
        response = requests.post(f"{self.api_url}/models/train", json=data)
        return response.json()

    def predict(self, features: np.ndarray, model_id: str) -> Dict[str, Any]:
        """Make predictions using trained model."""
        data = {"features": features.tolist(), "model_id": model_id}
        response = requests.post(f"{self.api_url}/models/predict", json=data)
        return response.json()

    def delete_model(self, model_id: str) -> Dict[str, Any]:
        """Delete a model."""
        response = requests.delete(f"{self.api_url}/models/{model_id}")
        return response.json()


def process_data_file(
    file_content: bytes, is_training: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Process uploaded data file."""
    try:
        # Log attempt to read as CSV
        logger.info("Attempting to read as CSV...")
        df = pd.read_csv(BytesIO(file_content))
    except Exception as csv_error:
        logger.error(f"CSV read failed: {csv_error}")
        try:
            # Log attempt to read as Excel
            logger.info("Attempting to read as Excel...")
            df = pd.read_excel(BytesIO(file_content))
        except Exception as excel_error:
            logger.error(f"Excel read failed: {excel_error}")
            raise ValueError("File must be CSV or Excel format")

    if is_training:
        if "label" in df.columns:
            target_col = "label"
        else:
            raise ValueError("File must contain 'label' column")

        # Split features and target
        y = df[target_col].values
        X = df.drop(columns=[target_col]).values

        return X, y
    else:
        return df.values, None


def format_model_info(model_info: Dict[str, Any]) -> str:
    """Format model information for display."""
    return (
        f"Model ID: {model_info['model_id']}\n"
        f"Type: {model_info['model_type']}\n"
        f"Status: {model_info['status']}\n"
        f"Performance:\n"
        f"{json.dumps(model_info['performance_metrics'], indent=2)}"
    )


def format_prediction_result(result: Dict[str, Any]) -> str:
    """Format prediction results for display."""
    return (
        f"Model ID: {result['model_id']}\n"
        f"Predictions: {result['predictions']}\n"
        f"Prediction time: {result['prediction_time']:.3f} seconds"
    )
