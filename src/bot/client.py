import aiohttp
from typing import Dict, List, Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class MLBotClient:
    """Async client for ML API."""

    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")

    async def _make_request(
        self, method: str, endpoint: str, data: Optional[Dict] = None
    ) -> Dict:
        """Make request to API."""
        url = f"{self.api_url}{endpoint}"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.request(method, url, json=data) as response:
                    response_data = await response.json()
                    if not response.ok:
                        logger.error(f"API error: {response_data}")
                        raise ValueError(
                            response_data.get("detail", "Unknown API error")
                        )
                    return response_data
            except aiohttp.ClientError as e:
                logger.error(f"Request error: {e}")
                raise

    async def health_check(self) -> Dict[str, str]:
        """Check API health."""
        return await self._make_request("GET", "/api/v1/health")

    async def train_model(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        model_type: str,
        hyperparameters: Optional[Dict] = None,
    ) -> Dict:
        """Train a new model."""
        data = {
            "model_type": model_type,
            "features": features.tolist(),
            "labels": labels.tolist(),
            "hyperparameters": hyperparameters,
        }
        return await self._make_request("POST", "/api/v1/models/train", data)

    async def predict(self, features: np.ndarray, model_id: str) -> Dict:
        """Make predictions using a trained model."""
        data = {"model_id": model_id, "features": features.tolist()}
        return await self._make_request("POST", "/api/v1/models/predict", data)

    async def get_available_models(self) -> Dict:
        """Get list of available models."""
        return await self._make_request("GET", "/api/v1/models/available")

    async def delete_model(self, model_id: str) -> Dict:
        """Delete a trained model."""
        return await self._make_request("DELETE", f"/api/v1/models/{model_id}")
