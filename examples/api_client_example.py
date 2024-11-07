import requests
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class MLAPIClient:
    """Client for interacting with the ML Model API."""

    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url

    def healthcheck(self):
        """Healthcheck request"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

    def get_available_models(self):
        """Get list of available models."""
        response = requests.get(f"{self.base_url}/models/available")
        return response.json()

    def train_model(
        self, features, labels, model_type, hyperparameters=None, model_id=None
    ):
        """Train a new model."""
        data = {
            "features": features.tolist(),
            "labels": labels.tolist(),
            "model_type": model_type,
            "hyperparameters": hyperparameters,
            "model_id": model_id,
        }
        response = requests.post(f"{self.base_url}/models/train", json=data)
        return response.json()

    def predict(self, features, model_id):
        """Make predictions using a trained model."""
        data = {"features": features.tolist(), "model_id": model_id}
        response = requests.post(f"{self.base_url}/models/predict", json=data)
        return response.json()

    def delete_model(self, model_id: str):
        """Delete a trained model."""
        response = requests.delete(f"{self.base_url}/models/{model_id}")
        return response.status_code

    def delete_all_models(self):
        """Get all active models and delete them."""
        # Get current active models
        response = self.get_available_models()
        active_models = response.get("active_models", {})

        deleted_models = []
        failed_deletions = []

        # Delete each model
        for model_id in active_models:
            try:
                status = self.delete_model(model_id)
                if status == 200:
                    deleted_models.append(model_id)
                else:
                    failed_deletions.append((model_id, status))
            except Exception as e:
                failed_deletions.append((model_id, str(e)))

        return {"deleted_models": deleted_models, "failed_deletions": failed_deletions}


def main():
    # Create synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Initialize client
    client = MLAPIClient()

    # Healthcheck
    healthcheck = client.healthcheck()
    print("Healthcheck:", healthcheck)

    # Get available models
    available_models = client.get_available_models()
    print("Available models:", available_models)

    # Train a random forest model
    hyperparameters = {"n_estimators": [50, 100], "max_depth": [10, 20]}
    print("Hyperparameters:", hyperparameters)

    rf_result = client.train_model(X_train, y_train, "random_forest", hyperparameters)
    print("\nTraining result:", rf_result)
    rf_model_id = rf_result["model_id"]

    # Get available models
    available_models = client.get_available_models()
    print("Available models:", available_models)

    # Retrain a random forest model
    hyperparameters = {"n_estimators": [40, 150], "max_depth": [15, 30]}

    rf_result = client.train_model(
        X_train,
        y_train,
        "random_forest",
        hyperparameters,
        model_id=rf_model_id,
    )
    print("\nRetraining result:", rf_result)

    # Make predictions with the retrained Random Forest model
    rf_predictions = client.predict(X_test, rf_model_id)
    print("\nRandom Forest predictions:", rf_predictions)

    # Train an SVM model
    svm_hyperparameters = {"C": [1.0], "kernel": ["rbf"]}
    svm_result = client.train_model(X_train, y_train, "svm", svm_hyperparameters)
    print("\nSVM training result:", svm_result)
    svm_model_id = svm_result["model_id"]

    # Make prediction with the SVM model
    svm_predictions = client.predict(X_test, svm_model_id)
    print("\nSVM predictions:", svm_predictions)

    # Get and delete all active models
    print("\nCurrent active models:", client.get_available_models())

    result = client.delete_all_models()
    print("\nDeleted models:", result["deleted_models"])
    if result["failed_deletions"]:
        print("Failed deletions:", result["failed_deletions"])

    # Verify all models are deleted
    print("\nRemaining models:", client.get_available_models())


if __name__ == "__main__":
    main()
