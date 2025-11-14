"""
Centralized Test Utilities for Universal Consciousness Interface

This module provides common mock implementations and test utilities
to reduce code duplication across the test suite and core modules.
"""

import random
import statistics
import math
from typing import Any, List, Optional, Union


class MockNumPy:
    """Mock NumPy implementation for systems without numpy installed"""

    def __init__(self):
        self.random = self.MockRandom()

    class MockRandom:
        """Mock numpy.random module"""

        @staticmethod
        def random():
            return random.random()

        @staticmethod
        def randn(*shape):
            """Generate random numbers from standard normal distribution"""
            if len(shape) == 0:
                return random.gauss(0, 1)
            elif len(shape) == 1:
                return [random.gauss(0, 1) for _ in range(shape[0])]
            elif len(shape) == 2:
                return [[random.gauss(0, 1) for _ in range(shape[1])]
                        for _ in range(shape[0])]
            else:
                # For higher dimensions, return nested lists
                def generate_nested(dims):
                    if len(dims) == 1:
                        return [random.gauss(0, 1) for _ in range(dims[0])]
                    return [generate_nested(dims[1:]) for _ in range(dims[0])]
                return generate_nested(shape)

        @staticmethod
        def rand(*shape):
            """Generate random numbers from uniform distribution [0, 1)"""
            if len(shape) == 0:
                return random.random()
            elif len(shape) == 1:
                return [random.random() for _ in range(shape[0])]
            elif len(shape) == 2:
                return [[random.random() for _ in range(shape[1])]
                        for _ in range(shape[0])]
            else:
                def generate_nested(dims):
                    if len(dims) == 1:
                        return [random.random() for _ in range(dims[0])]
                    return [generate_nested(dims[1:]) for _ in range(dims[0])]
                return generate_nested(shape)

        @staticmethod
        def choice(seq, size=None):
            if size is None:
                return random.choice(seq)
            return [random.choice(seq) for _ in range(size)]

    @staticmethod
    def array(data, dtype=None):
        """Mock array creation - returns list"""
        return list(data) if hasattr(data, '__iter__') else [data]

    @staticmethod
    def zeros(shape):
        """Create array of zeros"""
        if isinstance(shape, int):
            return [0.0] * shape
        elif len(shape) == 1:
            return [0.0] * shape[0]
        elif len(shape) == 2:
            return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]
        return []

    @staticmethod
    def ones(shape):
        """Create array of ones"""
        if isinstance(shape, int):
            return [1.0] * shape
        elif len(shape) == 1:
            return [1.0] * shape[0]
        elif len(shape) == 2:
            return [[1.0 for _ in range(shape[1])] for _ in range(shape[0])]
        return []

    @staticmethod
    def mean(values, axis=None):
        """Calculate mean of values"""
        if not values:
            return 0.0
        try:
            return statistics.mean(values)
        except (TypeError, statistics.StatisticsError):
            # Handle nested lists
            flat = [v for v in values if isinstance(v, (int, float))]
            return statistics.mean(flat) if flat else 0.0

    @staticmethod
    def std(values, axis=None):
        """Calculate standard deviation"""
        if not values or len(values) < 2:
            return 0.0
        try:
            return statistics.stdev(values)
        except (TypeError, statistics.StatisticsError):
            flat = [v for v in values if isinstance(v, (int, float))]
            return statistics.stdev(flat) if len(flat) >= 2 else 0.0

    @staticmethod
    def var(values, axis=None):
        """Calculate variance"""
        if not values or len(values) < 2:
            return 0.0
        try:
            return statistics.variance(values)
        except (TypeError, statistics.StatisticsError):
            flat = [v for v in values if isinstance(v, (int, float))]
            return statistics.variance(flat) if len(flat) >= 2 else 0.0

    @staticmethod
    def sqrt(value):
        """Calculate square root"""
        if isinstance(value, (list, tuple)):
            return [math.sqrt(v) if v >= 0 else 0 for v in value]
        return math.sqrt(value) if value >= 0 else 0

    @staticmethod
    def exp(value):
        """Calculate exponential"""
        if isinstance(value, (list, tuple)):
            return [math.exp(min(v, 700)) for v in value]  # Prevent overflow
        return math.exp(min(value, 700))

    @staticmethod
    def sum(values, axis=None):
        """Calculate sum"""
        if isinstance(values, (int, float)):
            return values
        return sum(v for v in values if isinstance(v, (int, float)))

    @staticmethod
    def clip(values, min_val, max_val):
        """Clip values to range"""
        if isinstance(values, (int, float)):
            return max(min_val, min(max_val, values))
        return [max(min_val, min(max_val, v)) for v in values]

    @staticmethod
    def tanh(value):
        """Calculate hyperbolic tangent"""
        if isinstance(value, (list, tuple)):
            return [math.tanh(v) for v in value]
        return math.tanh(value)


class MockTorch:
    """Mock PyTorch implementation for systems without torch installed"""

    class Tensor:
        """Mock torch.Tensor"""
        def __init__(self, data):
            self.data = data if isinstance(data, list) else [data]

        def numpy(self):
            return self.data

        def item(self):
            return self.data[0] if self.data else 0

        def size(self):
            return len(self.data)

        def to(self, device):
            """Mock device transfer"""
            return self

        def __repr__(self):
            return f"Tensor({self.data})"

    class nn:
        """Mock torch.nn module"""
        class Module:
            """Mock nn.Module"""
            def __init__(self):
                self.training = True

            def train(self, mode=True):
                self.training = mode
                return self

            def eval(self):
                self.training = False
                return self

            def forward(self, x):
                return x

            def parameters(self):
                return []

            def to(self, device):
                """Mock device transfer"""
                return self

    @staticmethod
    def tensor(data, dtype=None, device=None):
        """Create mock tensor"""
        return MockTorch.Tensor(data)

    @staticmethod
    def randn(*shape, device=None):
        """Generate random tensor"""
        if len(shape) == 1:
            data = [random.gauss(0, 1) for _ in range(shape[0])]
        elif len(shape) == 2:
            data = [[random.gauss(0, 1) for _ in range(shape[1])]
                    for _ in range(shape[0])]
        else:
            data = [random.gauss(0, 1)]
        return MockTorch.Tensor(data)

    @staticmethod
    def zeros(*shape, device=None):
        """Create tensor of zeros"""
        if len(shape) == 1:
            data = [0.0] * shape[0]
        elif len(shape) == 2:
            data = [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]
        else:
            data = [0.0]
        return MockTorch.Tensor(data)

    @staticmethod
    def cat(tensors, dim=0):
        """Concatenate tensors"""
        combined = []
        for t in tensors:
            if isinstance(t, MockTorch.Tensor):
                combined.extend(t.data)
            else:
                combined.append(t)
        return MockTorch.Tensor(combined)


class MockNetworkX:
    """Mock NetworkX implementation for graph operations"""

    class Graph:
        """Mock networkx.Graph"""
        def __init__(self):
            self.nodes_dict = {}
            self.edges_list = []

        def add_node(self, node, **attr):
            self.nodes_dict[node] = attr

        def add_edge(self, u, v, **attr):
            self.edges_list.append((u, v, attr))

        def nodes(self, data=False):
            if data:
                return list(self.nodes_dict.items())
            return list(self.nodes_dict.keys())

        def edges(self, data=False):
            if data:
                return self.edges_list
            return [(u, v) for u, v, _ in self.edges_list]

        def number_of_nodes(self):
            return len(self.nodes_dict)

        def number_of_edges(self):
            return len(self.edges_list)

    @staticmethod
    def spring_layout(graph, **kwargs):
        """Mock spring layout - returns random positions"""
        return {node: (random.random(), random.random())
                for node in graph.nodes()}


def get_numpy():
    """Get numpy or mock implementation"""
    try:
        import numpy as np
        return np
    except ImportError:
        return MockNumPy()


def get_torch():
    """Get torch or mock implementation"""
    try:
        import torch
        return torch
    except ImportError:
        return MockTorch()


def get_networkx():
    """Get networkx or mock implementation"""
    try:
        import networkx as nx
        return nx
    except ImportError:
        return MockNetworkX()


__all__ = [
    'MockNumPy',
    'MockTorch',
    'MockNetworkX',
    'get_numpy',
    'get_torch',
    'get_networkx',
]
