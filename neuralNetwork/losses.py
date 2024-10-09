import numpy as np

def mse(y_true, y_pred):
    """Calculate the Mean Squared Error (MSE) loss.

    Parameters:
    y_true (list)   - True labels.
    y_pred (list)   - Predicted labels.

    Returns:
    (float) - The calculated MSE loss.
    """
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    """Calculates the derivative of the Mean Squared Error (MSE) loss with respect to the predictions.

    Parameters:
    y_true (list) - True labels.
    y_pred (list) - Predicted labels.

    Returns:
    (list) - The derivative of the MSE loss with respect to y_pred.
    """
    return 2 * (y_pred - y_true) / np.size(y_true)

def binary_cross_entropy(y_true, y_pred):
    """Calculates the Binary Cross-Entropy loss.

    Parameters:
    y_true (list) - True labels.
    y_pred (list) - Predicted labels.

    Returns:
    (float) - The calculated Binary Cross-Entropy loss.
    """
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    """Calculates the derivative of the Binary Cross-Entropy loss with respect to the predictions.

    Parameters:
    y_true (list) - True labels.
    y_pred (list) - Predicted labels.

    Returns:
    (list) - The derivative of the Binary Cross-Entropy loss with respect to y_pred.
    """
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)
