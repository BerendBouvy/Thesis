import numpy as np

def polynomialprod(x: np.ndarray, y: np.ndarray, p: int, q: int) -> np.ndarray:
    """Compute the polynomial of degree p in x and degree q in y.
    """
    return np.power(x, p) * np.power(y, q)

def polynomialsum(x: np.ndarray, y: np.ndarray, p: int, q: int) -> np.ndarray:
    """Compute the polynomial of degree p in x and degree q in y, summed together.
    """
    return np.power(x, p) + np.power(y, q)

def exp(x: np.ndarray, y: np.ndarray, c: float) -> np.ndarray:
    """Compute the exponential function e^(c * (x + y)) for each element in x and y.
    """
    return np.exp(c * (x + y))

def log(x: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
    """Compute the natural logarithm of ((x + y)**2 + eps) for each element in x and y.
    """
    return np.log((x + y)**2 + eps)

def ratio1(x: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
    """Compute the ratio of x to (y + eps) for each element in x and y.
    """
    return x / (y + eps)

def ratio2(x: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
    """Compute the ratio of (x + y) to (x - y + eps) for each element in x and y.
    """
    return (x + y) / (x - y + eps)

def sin1(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the sine of (x + y) for each element in x and y.
    """
    return np.sin(x + y)

def sin2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the sine of (x * y) for each element in x and y.
    """
    return np.sin(x * y)

