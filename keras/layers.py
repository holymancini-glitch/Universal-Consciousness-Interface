# Mock keras.layers module
import numpy as np


class Layer:
    """Base mock Keras layer."""

    def __init__(self, **kwargs):
        self.trainable = kwargs.get('trainable', True)
        self.name = kwargs.get('name', None)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, inputs):
        return inputs


class Input(Layer):
    """Mock Keras Input layer."""

    def __init__(self, shape=None, batch_size=None, name=None, dtype=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.shape = shape
        self.batch_size = batch_size
        self.dtype = dtype


class Dense(Layer):
    """Mock Keras Dense layer."""

    def __init__(self, units, activation=None, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias

    def call(self, inputs):
        # Return random output with correct shape
        if isinstance(inputs, np.ndarray):
            batch_shape = inputs.shape[:-1]
            return np.random.randn(*batch_shape, self.units)
        return np.random.randn(1, self.units)


class GRU(Layer):
    """Mock Keras GRU layer."""

    def __init__(self, units, activation='tanh', recurrent_activation='sigmoid',
                 use_bias=True, return_sequences=True, return_state=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias
        self.return_sequences = return_sequences
        self.return_state = return_state

    def call(self, inputs, initial_state=None):
        # Mock GRU output
        if isinstance(inputs, np.ndarray):
            batch_size = inputs.shape[0]
            seq_len = inputs.shape[1] if len(inputs.shape) > 2 else 1

            if self.return_sequences:
                output = np.random.randn(batch_size, seq_len, self.units)
            else:
                output = np.random.randn(batch_size, self.units)

            if self.return_state:
                state = np.random.randn(batch_size, self.units)
                return output, state
            return output
        return np.random.randn(1, self.units)


class LSTM(Layer):
    """Mock Keras LSTM layer."""

    def __init__(self, units, activation='tanh', recurrent_activation='sigmoid',
                 use_bias=True, return_sequences=True, return_state=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias
        self.return_sequences = return_sequences
        self.return_state = return_state

    def call(self, inputs, initial_state=None):
        if isinstance(inputs, np.ndarray):
            batch_size = inputs.shape[0]
            seq_len = inputs.shape[1] if len(inputs.shape) > 2 else 1

            if self.return_sequences:
                output = np.random.randn(batch_size, seq_len, self.units)
            else:
                output = np.random.randn(batch_size, self.units)

            if self.return_state:
                h_state = np.random.randn(batch_size, self.units)
                c_state = np.random.randn(batch_size, self.units)
                return output, h_state, c_state
            return output
        return np.random.randn(1, self.units)


class Dropout(Layer):
    """Mock Keras Dropout layer."""

    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        return inputs


class Embedding(Layer):
    """Mock Keras Embedding layer."""

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        if isinstance(inputs, np.ndarray):
            return np.random.randn(*inputs.shape, self.output_dim)
        return np.random.randn(1, self.output_dim)


__all__ = ['Layer', 'Input', 'Dense', 'GRU', 'LSTM', 'Dropout', 'Embedding']
