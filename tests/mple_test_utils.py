import numpy as np


def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Concordance-Correlation-Coefficient
    Measures the goodness of fit of data and predictions (closeness to the identity line)
    Can be defined as 1 - mean_distance_from_identity/mean_distance_from_identity_assuming_independence
    """
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
