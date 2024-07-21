"""Neural network activation functions."""

# Author: Genevieve Hayes
# License: BSD 3 clause

import numpy as np
import torch
import torch.nn.functional as F


def identity(x: np.ndarray, deriv: bool = False) -> np.ndarray:
    """Linear activation function

    Parameters
    ----------
    x : np.ndarray
        Array containing input data.

    deriv : bool, default: False
        Whether to return the function or its derivative.
        Set True for derivative.

    Returns
    -------
    np.ndarray
        Value of activation function at x
    """
    if not deriv:
        fx = x
    else:
        fx = np.ones(np.shape(x))

    return fx


def relu(x: np.ndarray, deriv: bool = False) -> np.ndarray:
    """ReLU activation function

    Parameters
    ----------
    x : np.ndarray
        Array containing input data.

    deriv : bool, default: False
        Whether to return the function or its derivative.
        Set True for derivative.

    Returns
    -------
    np.ndarray
        Value of activation function at x
    """
    x_tensor = torch.tensor(x, dtype=torch.float32)
    fx = F.relu(x_tensor).numpy()

    if deriv:
        fx = (fx > 0).astype(float)

    return fx


def sigmoid(x: np.ndarray, deriv: bool = False) -> np.ndarray:
    """Sigmoid activation function

    Parameters
    ----------
    x : np.ndarray
        Array containing input data.

    deriv : bool, default: False
        Whether to return the function or its derivative.
        Set True for derivative.

    Returns
    -------
    np.ndarray
        Value of activation function at x
    """
    x_tensor = torch.tensor(x, dtype=torch.float32)
    fx = torch.sigmoid(x_tensor).numpy()

    if deriv:
        fx = fx * (1 - fx)

    return fx


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax activation function

    Parameters
    ----------
    x : np.ndarray
        Array containing input data.

    Returns
    -------
    np.ndarray
        Value of activation function at x
    """
    x_tensor = torch.tensor(x, dtype=torch.float32)
    fx = F.softmax(x_tensor, dim=1).numpy()

    return fx


def tanh(x: np.ndarray, deriv: bool = False) -> np.ndarray:
    """Hyperbolic tan activation function

    Parameters
    ----------
    x : np.ndarray
        Array containing input data.

    deriv : bool, default: False
        Whether to return the function or its derivative.
        Set True for derivative.

    Returns
    -------
    np.ndarray
        Value of activation function at x
    """
    x_tensor = torch.tensor(x, dtype=torch.float32)
    fx = torch.tanh(x_tensor).numpy()

    if deriv:
        fx = 1 - fx**2

    return fx
