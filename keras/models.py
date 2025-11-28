# Mock keras.models module
import numpy as np


class Model:
    """Mock Keras Model."""

    def __init__(self, inputs=None, outputs=None, name=None):
        self.inputs = inputs
        self.outputs = outputs
        self.name = name
        self._is_compiled = False

    def compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
        """Mock compile method."""
        self._is_compiled = True

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1,
            validation_data=None, **kwargs):
        """Mock fit method."""
        return MockHistory()

    def predict(self, x, batch_size=None, verbose=0, **kwargs):
        """Mock predict method - returns random data."""
        if isinstance(x, np.ndarray):
            # Return appropriate shape based on input
            batch_size = x.shape[0]
            # Default output size
            output_size = 10
            return np.random.randn(batch_size, output_size)
        return np.random.randn(1, 10)

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, **kwargs):
        """Mock evaluate method."""
        return [0.0, 0.0]  # Mock loss and metric

    def save(self, filepath, **kwargs):
        """Mock save method."""
        pass

    def load_weights(self, filepath, **kwargs):
        """Mock load_weights method."""
        pass

    def summary(self, **kwargs):
        """Mock summary method."""
        print(f"Model: {self.name}")
        print("_" * 60)


class MockHistory:
    """Mock training history object."""

    def __init__(self):
        self.history = {
            'loss': [0.5, 0.4, 0.3],
            'accuracy': [0.7, 0.8, 0.85]
        }


def load_model(filepath, **kwargs):
    """Mock load_model function."""
    return Model()


__all__ = ['Model', 'load_model']
