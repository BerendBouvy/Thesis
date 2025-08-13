import numpy as np


def polynomial(x: np.ndarray, p: int) -> np.ndarray:
    """Compute the polynomial of degree p for each element in x.
    """
    
    return np.power(x, p)

def exp(x: np.ndarray, c: float) -> np.ndarray:
    """Compute the exponential function e^(c * x) for each element in x.
    """
    if not isinstance(c, (int, float)) or abs(c) > 10:
        raise ValueError("Coefficient c must be a float or integer and within the range [-10, 10].")
    if not isinstance(x, np.ndarray):
        raise ValueError("Input x must be a numpy ndarray.")
    return np.exp(c * x)

def log(x: np.ndarray, eps: float) -> np.ndarray:
    
    """Compute the natural logarithm of (x**2 + eps) for each element in x.
    """
    if not isinstance(eps, (int, float)) or eps <= 0 or eps > 1:
        raise ValueError("Epsilon must be a positive float or integer and within the range (0, 1].")
    if not isinstance(x, np.ndarray):
        raise ValueError("Input x must be a numpy ndarray.")
    return np.log(x**2 + eps)

def smooth_abs(x: np.ndarray, eps: float) -> np.ndarray:
    """Compute the smoothed absolute value of x using the given epsilon.
    """
    if not isinstance(eps, (int, float)) or eps <= 0:
        raise ValueError("Epsilon must be a positive float or integer.")
    return np.sqrt(x**2 + eps)

def tanh(x: np.ndarray) -> np.ndarray:
    """Compute the hyperbolic tangent of x for each element in x.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("Input x must be a numpy ndarray.")
    return np.tanh(x)

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute the sigmoid function of x for each element in x.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("Input x must be a numpy ndarray.")
    return 1 / (1 + np.exp(-x))

def sin(x: np.ndarray) -> np.ndarray:
    """Compute the sine of x for each element in x.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("Input x must be a numpy ndarray.")
    return np.sin(x)

def cos(x: np.ndarray) -> np.ndarray:
    """Compute the cosine of x for each element in x.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("Input x must be a numpy ndarray.")
    return np.cos(x) 