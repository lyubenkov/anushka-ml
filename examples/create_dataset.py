import pandas as pd
import numpy as np
from sklearn.datasets import make_classification


def create_dataset(n_samples=100, n_features=4):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )

    df = pd.DataFrame(data=X, columns=[f"feature_{i+1}" for i in range(n_features)])

    df["label"] = y

    # Round values to 3 decimal places for readability
    df = df.round(3)

    return df


def create_prediction_dataset(n_samples=5, n_features=4):
    np.random.seed(42)

    X = np.random.randn(n_samples, n_features)

    df = pd.DataFrame(data=X, columns=[f"feature_{i+1}" for i in range(n_features)])

    df = df.round(3)

    return df


def create_test_files():
    """Create test Excel files for different models."""
    df_clf = create_dataset(n_samples=100, n_features=4)
    df_pred = create_prediction_dataset(n_samples=5, n_features=4)

    df_clf.to_excel("dataset.xlsx", index=False)
    df_pred.to_excel("prediction_data.xlsx", index=False)

    print("\nDataset (dataset.xlsx):")
    print(f"Shape: {df_clf.shape}")
    print("\nFirst few rows:")
    print(df_clf.head())
    print("\nFeature statistics:")
    print(df_clf.describe())

    print("\nPrediction dataset (prediction_data.xlsx):")
    print(f"Shape: {df_pred.shape}")
    print("\nFirst few rows:")
    print(df_pred.head())
    print("\nFeature statistics:")
    print(df_pred.describe())


if __name__ == "__main__":
    create_test_files()
