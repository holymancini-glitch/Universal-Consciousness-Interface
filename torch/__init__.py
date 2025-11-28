import numpy as np


class Tensor(np.ndarray):
    """Mock torch.Tensor using numpy arrays."""

    def __new__(cls, data, dtype=None):
        if isinstance(data, np.ndarray):
            obj = np.asarray(data, dtype=dtype).view(cls)
        else:
            obj = np.asarray(data, dtype=dtype).view(cls)
        return obj

    def size(self, dim=None):
        """Return tensor size."""
        if dim is None:
            return self.shape
        return self.shape[dim]

    def dim(self):
        """Return number of dimensions."""
        return self.ndim

    def cuda(self):
        """Mock cuda() - returns self."""
        return self

    def cpu(self):
        """Mock cpu() - returns self."""
        return self

    def to(self, *args, **kwargs):
        """Mock to() - returns self."""
        return self

    def detach(self):
        """Mock detach() - returns copy."""
        return self.copy()

    def numpy(self):
        """Return as numpy array."""
        return np.asarray(self)


def tensor(data, dtype=None):
    """Create a mock tensor."""
    return Tensor(data, dtype=dtype)


def randn(*shape):
    """Return an array of random numbers with the given shape."""
    return Tensor(np.random.randn(*shape))


def zeros(*shape, dtype=None):
    """Return a tensor of zeros."""
    return Tensor(np.zeros(shape, dtype=dtype))


def ones(*shape, dtype=None):
    """Return a tensor of ones."""
    return Tensor(np.ones(shape, dtype=dtype))


def rand(*shape):
    """Return a tensor of random values in [0, 1)."""
    return Tensor(np.random.rand(*shape))


def arange(*args, **kwargs):
    """Return a tensor with evenly spaced values."""
    return Tensor(np.arange(*args, **kwargs))


def cat(tensors, dim=0):
    """Concatenate tensors along a dimension."""
    arrays = [np.asarray(t) for t in tensors]
    return Tensor(np.concatenate(arrays, axis=dim))


def stack(tensors, dim=0):
    """Stack tensors along a new dimension."""
    arrays = [np.asarray(t) for t in tensors]
    return Tensor(np.stack(arrays, axis=dim))


# Mock dtype objects
class dtype:
    float32 = np.float32
    float64 = np.float64
    int32 = np.int32
    int64 = np.int64
    bool = np.bool_


float32 = dtype.float32
float64 = dtype.float64
int32 = dtype.int32
int64 = dtype.int64


__all__ = [
    "Tensor", "tensor", "randn", "zeros", "ones", "rand", "arange",
    "cat", "stack", "dtype", "float32", "float64", "int32", "int64"
]
