# Mock torch.nn module to satisfy imports when PyTorch is not installed
import numpy as np


class Module:
    """Mock torch.nn.Module for testing without PyTorch."""

    def __init__(self):
        self.training = True
        self._parameters = {}
        self._modules = {}

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self, mode=True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def parameters(self):
        return []

    def to(self, *args, **kwargs):
        return self


class Linear(Module):
    """Mock torch.nn.Linear layer."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.randn(out_features, in_features)
        self.bias = np.random.randn(out_features) if bias else None

    def forward(self, x):
        result = np.dot(x, self.weight.T)
        if self.bias is not None:
            result += self.bias
        return result


class LSTM(Module):
    """Mock torch.nn.LSTM layer."""

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, dropout=0.0, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional

    def forward(self, x, hidden=None):
        # Return mock output
        batch_size = x.shape[0] if self.batch_first else x.shape[1]
        seq_len = x.shape[1] if self.batch_first else x.shape[0]
        num_directions = 2 if self.bidirectional else 1

        output_shape = (batch_size, seq_len, self.hidden_size * num_directions) if self.batch_first else (seq_len, batch_size, self.hidden_size * num_directions)
        output = np.random.randn(*output_shape)

        h_n = np.random.randn(self.num_layers * num_directions, batch_size, self.hidden_size)
        c_n = np.random.randn(self.num_layers * num_directions, batch_size, self.hidden_size)

        return output, (h_n, c_n)


class GRU(Module):
    """Mock torch.nn.GRU layer."""

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, dropout=0.0, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional

    def forward(self, x, hidden=None):
        batch_size = x.shape[0] if self.batch_first else x.shape[1]
        seq_len = x.shape[1] if self.batch_first else x.shape[0]
        num_directions = 2 if self.bidirectional else 1

        output_shape = (batch_size, seq_len, self.hidden_size * num_directions) if self.batch_first else (seq_len, batch_size, self.hidden_size * num_directions)
        output = np.random.randn(*output_shape)

        h_n = np.random.randn(self.num_layers * num_directions, batch_size, self.hidden_size)

        return output, h_n


class Dropout(Module):
    """Mock torch.nn.Dropout layer."""

    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        if self.training:
            mask = np.random.binomial(1, 1 - self.p, size=x.shape) / (1 - self.p)
            return x * mask
        return x


class LayerNorm(Module):
    """Mock torch.nn.LayerNorm layer."""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + self.eps)


class Embedding(Module):
    """Mock torch.nn.Embedding layer."""

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = np.random.randn(num_embeddings, embedding_dim)

    def forward(self, x):
        return self.weight[x]


class MultiheadAttention(Module):
    """Mock torch.nn.MultiheadAttention layer."""

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, batch_first=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        # Return mock output
        batch_size = query.shape[0] if self.batch_first else query.shape[1]
        seq_len = query.shape[1] if self.batch_first else query.shape[0]

        output_shape = (batch_size, seq_len, self.embed_dim) if self.batch_first else (seq_len, batch_size, self.embed_dim)
        output = np.random.randn(*output_shape)

        if need_weights:
            attn_weights = np.random.rand(batch_size, seq_len, seq_len)
            return output, attn_weights
        return output, None


class Sequential(Module):
    """Mock torch.nn.Sequential container."""

    def __init__(self, *args):
        super().__init__()
        self.layers = list(args)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Activation functions
class ReLU(Module):
    def forward(self, x):
        return np.maximum(0, x)


class Tanh(Module):
    def forward(self, x):
        return np.tanh(x)


class Sigmoid(Module):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))


class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=self.dim, keepdims=True))
        return exp_x / np.sum(exp_x, axis=self.dim, keepdims=True)


# Loss functions
class CrossEntropyLoss(Module):
    def forward(self, input, target):
        return 0.0  # Mock loss


class MSELoss(Module):
    def forward(self, input, target):
        return 0.0  # Mock loss


__all__ = [
    'Module', 'Linear', 'LSTM', 'GRU', 'Dropout', 'LayerNorm',
    'Embedding', 'MultiheadAttention', 'Sequential',
    'ReLU', 'Tanh', 'Sigmoid', 'Softmax',
    'CrossEntropyLoss', 'MSELoss'
]
