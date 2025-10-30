"""
Attention Field for Integrated Consciousness System

This module provides attention mechanisms with dynamic focus and resonance
detection across the latent space.
"""

import torch
from collections import deque
from datetime import datetime
from typing import Dict, Any, Optional

from .latent_space import LatentSpace


class AttentionField:
    """
    Enhanced attention field with dynamic focus and resonance detection.

    Manages attention allocation across the latent space using weighted
    resonance calculations. Tracks focus history and automatically
    detects high-resonance vectors.

    Attributes:
        latent_space: LatentSpace instance to monitor
        attention_weights: Per-dimension attention weighting
        focus_history: History of attention focus events
        resonance_threshold: Minimum resonance for detection
    """

    def __init__(self, latent_space: LatentSpace):
        """
        Initialize attention field.

        Args:
            latent_space: LatentSpace instance to monitor
        """
        self.latent_space = latent_space
        self.attention_weights = torch.ones(latent_space.dimensions, device=latent_space.device)
        self.focus_history = deque(maxlen=100)
        self.resonance_threshold = 0.5

    def sense_resonance(self) -> Dict[int, float]:
        """
        Compute resonance for all vectors with attention weighting.

        Calculates weighted norm for each vector in the latent space,
        providing a measure of how strongly each vector resonates with
        the current attention pattern.

        Returns:
            Dictionary mapping vector IDs to resonance values
        """
        vectors = self.latent_space.get_all_vectors()
        if not vectors:
            return {}

        resonance = {}
        for vec_id, vec in vectors.items():
            # Weighted norm with attention
            weighted_vec = vec * self.attention_weights
            resonance_value = torch.norm(weighted_vec).item()
            resonance[vec_id] = resonance_value

        return resonance

    def focus_on(self, vector_id: int) -> Optional[Dict[str, Any]]:
        """
        Focus attention on specific vector with enhancement.

        Calculates weighted resonance for the specified vector and
        records the focus event in history for tracking.

        Args:
            vector_id: ID of the vector to focus on

        Returns:
            Focus event dictionary with vector, resonance, and timestamp,
            or None if vector not found
        """
        vector = self.latent_space.get_vector(vector_id)
        if vector is None:
            return None

        # Calculate weighted resonance
        weighted_vector = vector * self.attention_weights
        resonance = torch.norm(weighted_vector).item()

        # Record focus event
        focus_event = {
            'vector_id': vector_id,
            'vector': vector,
            'resonance': resonance,
            'timestamp': datetime.now()
        }

        self.focus_history.append(focus_event)
        return focus_event


__all__ = ['AttentionField']
