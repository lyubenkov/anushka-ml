class ModelNotFoundError(Exception):
    """Raised when a requested model is not found."""

    pass


class InvalidHyperparametersError(Exception):
    """Raised when invalid hyperparameters are provided."""

    pass
