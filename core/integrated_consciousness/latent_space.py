"""
Latent Space Management for Integrated Consciousness System

This module provides vector space management with GPU acceleration for
efficient processing of consciousness representations.
"""

import logging
import numpy as np
import torch
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Union


logger = logging.getLogger(__name__)


class LatentSpace:
    """
    Enhanced vector space management with GPU acceleration.

    Manages high-dimensional vector representations of consciousness states
    with automatic GPU transfer and history tracking.

    Attributes:
        dimensions: Dimensionality of the vector space
        use_gpu: Whether GPU acceleration is enabled
        device: PyTorch device (cuda or cpu)
        vectors: Dictionary mapping vector IDs to tensors
        vector_history: Recent vector operations history
        dimension_weights: Per-dimension weighting factors
    """

    def __init__(self, dimensions: int = 256, use_gpu: bool = True):
        """
        Initialize latent space with specified dimensions.

        Args:
            dimensions: Dimensionality of vectors (default: 256)
            use_gpu: Enable GPU acceleration if available (default: True)
        """
        self.dimensions = dimensions
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        # Use PyTorch tensors for GPU acceleration
        self.vectors: Dict[int, torch.Tensor] = {}
        self.vector_history = deque(maxlen=1000)
        self.dimension_weights = torch.ones(dimensions, device=self.device)

        logger.info(f"LatentSpace initialized: {dimensions}D, GPU: {self.use_gpu}")

    def add_vector(self, vector_id: int, vector: Union[List[float], np.ndarray, torch.Tensor]):
        """
        Add vector with automatic GPU transfer.

        Converts input to PyTorch tensor and transfers to appropriate device
        (GPU or CPU) for processing.

        Args:
            vector_id: Unique identifier for the vector
            vector: Vector data (list, numpy array, or torch tensor)

        Raises:
            ValueError: If vector dimension doesn't match space dimensions
        """
        if isinstance(vector, (list, np.ndarray)):
            vector_tensor = torch.tensor(vector, dtype=torch.float32, device=self.device)
        else:
            vector_tensor = vector.to(self.device)

        if vector_tensor.shape[0] != self.dimensions:
            raise ValueError(f"Vector dimension must be {self.dimensions}")

        self.vectors[vector_id] = vector_tensor
        self.vector_history.append((vector_id, datetime.now()))

    def get_vector(self, vector_id: int) -> Optional[torch.Tensor]:
        """
        Get vector by ID.

        Args:
            vector_id: Identifier of the vector to retrieve

        Returns:
            PyTorch tensor if found, None otherwise
        """
        return self.vectors.get(vector_id)

    def update_vector(self, vector_id: int, new_vector: Union[List[float], np.ndarray, torch.Tensor]):
        """
        Update existing vector.

        Args:
            vector_id: Identifier of the vector to update
            new_vector: New vector data

        Raises:
            ValueError: If vector ID doesn't exist
        """
        if vector_id not in self.vectors:
            raise ValueError(f"Vector ID {vector_id} not found")

        if isinstance(new_vector, (list, np.ndarray)):
            new_vector = torch.tensor(new_vector, dtype=torch.float32, device=self.device)

        self.vectors[vector_id] = new_vector.to(self.device)

    def get_all_vectors(self) -> Dict[int, torch.Tensor]:
        """
        Get all vectors in the space.

        Returns:
            Dictionary mapping vector IDs to tensors
        """
        return self.vectors


__all__ = ['LatentSpace']
