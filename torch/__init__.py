import numpy as np

def randn(*shape):
    """Return an array of random numbers with the given shape."""
    return np.random.randn(*shape)

__all__ = ["randn"]
