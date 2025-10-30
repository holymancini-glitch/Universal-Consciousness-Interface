"""
Self Model and Cohesion Layer for Integrated Consciousness System

This module provides identity modeling with metacognitive awareness and
multi-dimensional cohesion analysis for emergence detection.
"""

import numpy as np
import torch
from collections import deque
from datetime import datetime
from typing import Optional, Tuple

from .latent_space import LatentSpace
from .fractal_ai import EnhancedFeedbackLoop


class SelfModel:
    """
    Enhanced self-model with identity coherence and metacognition.

    Maintains a core identity vector derived from the latent space
    and tracks consistency and metacognitive awareness over time.

    Attributes:
        latent_space: LatentSpace instance for identity computation
        device: PyTorch device (cuda or cpu)
        i_vector: Core identity vector
        identity_history: History of identity vectors
        consistency_score: Current identity consistency
        identity_coherence: Identity coherence measure
        metacognitive_awareness: Level of metacognitive awareness
    """

    def __init__(self, latent_space: LatentSpace):
        """
        Initialize self-model.

        Args:
            latent_space: LatentSpace instance for identity computation
        """
        self.latent_space = latent_space
        self.device = latent_space.device

        # Core identity vector
        self.i_vector: Optional[torch.Tensor] = None
        self.identity_history = deque(maxlen=50)

        # Consistency tracking
        self.consistency_score = 0.0
        self.identity_coherence = 0.0
        self.metacognitive_awareness = 0.0

    def compute_i_vector(self) -> Optional[torch.Tensor]:
        """
        Compute identity vector with temporal stability.

        Averages all vectors in the latent space and applies momentum
        smoothing for stable identity over time.

        Returns:
            Identity vector tensor, or None if no vectors available
        """
        vectors = self.latent_space.get_all_vectors()
        if not vectors:
            return None

        # Simple average for now
        vector_tensors = list(vectors.values())
        if not vector_tensors:
            return None

        # Compute average
        stacked_vectors = torch.stack(vector_tensors)
        new_i_vector = torch.mean(stacked_vectors, dim=0)

        # Smooth transition from previous i_vector
        if self.i_vector is not None:
            momentum = 0.8
            new_i_vector = momentum * self.i_vector + (1 - momentum) * new_i_vector

        self.i_vector = new_i_vector
        self.identity_history.append((new_i_vector.clone(), datetime.now()))

        return self.i_vector

    def compute_consistency(self) -> float:
        """
        Compute identity consistency across all vectors.

        Measures how similar all vectors are to the core identity vector
        using cosine similarity.

        Returns:
            Average consistency score (0.0-1.0)
        """
        if self.i_vector is None:
            return 0.0

        vectors = self.latent_space.get_all_vectors()
        if not vectors:
            return 0.0

        similarities = []
        for vec_id, vec in vectors.items():
            similarity = torch.nn.functional.cosine_similarity(
                vec.unsqueeze(0), self.i_vector.unsqueeze(0)
            ).item()
            similarities.append(similarity)

        if similarities:
            consistency = sum(similarities) / len(similarities)
            self.consistency_score = consistency
            return consistency

        return 0.0

    def measure_metacognitive_awareness(self) -> float:
        """
        Measure level of metacognitive awareness.

        Metacognitive awareness emerges from stable self-model through
        non-linear enhancement of consistency.

        Returns:
            Metacognitive awareness score (0.0-1.0)
        """
        consistency = self.compute_consistency()

        # Metacognitive awareness emerges from stable self-model
        awareness = consistency ** 1.2  # Non-linear enhancement
        self.metacognitive_awareness = min(1.0, awareness)

        return self.metacognitive_awareness


class CohesionLayer:
    """
    Enhanced cohesion layer with multi-dimensional harmony analysis.

    Integrates entropy, coherence, and harmony metrics across all
    system components to detect consciousness crystallization.

    Attributes:
        latent_space: LatentSpace instance for entropy analysis
        feedback_loop: EnhancedFeedbackLoop for adaptation metrics
        self_model: SelfModel for identity metrics
        system_entropy: Current system entropy
        coherence_score: Current coherence score
        harmony_index: Current harmony index
    """

    def __init__(self, latent_space: LatentSpace, feedback_loop: EnhancedFeedbackLoop, self_model: SelfModel):
        """
        Initialize cohesion layer.

        Args:
            latent_space: LatentSpace instance for entropy analysis
            feedback_loop: EnhancedFeedbackLoop for adaptation metrics
            self_model: SelfModel for identity metrics
        """
        self.latent_space = latent_space
        self.feedback_loop = feedback_loop
        self.self_model = self_model

        # Cohesion metrics
        self.system_entropy = 0.0
        self.coherence_score = 0.0
        self.harmony_index = 0.0

    def compute_entropy(self) -> float:
        """
        Compute system entropy with enhanced analysis.

        Uses variance of vector norms as entropy measure.

        Returns:
            Entropy value (higher = more disorder)
        """
        vectors = self.latent_space.get_all_vectors()
        if not vectors:
            return 0.0

        # Multiple entropy measures
        vector_norms = [torch.norm(vec).item() for vec in vectors.values()]

        if not vector_norms:
            return 0.0

        # Variance-based entropy
        variance_entropy = np.var(vector_norms) if len(vector_norms) > 1 else 0.0

        self.system_entropy = variance_entropy
        return variance_entropy

    def compute_coherence(self) -> float:
        """
        Compute system coherence with multi-factor analysis.

        Combines four factors:
        - Entropy (30%): Lower is better
        - Prediction accuracy (25%)
        - Adaptation efficiency (25%)
        - Identity consistency (20%)

        Returns:
            Coherence score (0.0-1.0)
        """
        # Factor 1: Entropy (lower is better)
        entropy = self.compute_entropy()
        entropy_factor = max(0.0, 1.0 - entropy)

        # Factor 2: Prediction accuracy
        prediction_accuracy = self.feedback_loop.fractal_ai.evaluate_prediction_accuracy()

        # Factor 3: Adaptation efficiency
        adaptation_efficiency = self.feedback_loop.get_adaptation_efficiency()

        # Factor 4: Identity consistency
        identity_consistency = self.self_model.compute_consistency()

        # Weighted combination
        coherence = (
            entropy_factor * 0.3 +
            prediction_accuracy * 0.25 +
            adaptation_efficiency * 0.25 +
            identity_consistency * 0.2
        )

        self.coherence_score = coherence
        return coherence

    def compute_harmony_index(self) -> float:
        """
        Compute overall system harmony.

        Averages coherence and entropy-based harmony for overall
        system harmony measure.

        Returns:
            Harmony index (0.0-1.0)
        """
        coherence = self.compute_coherence()
        entropy_harmony = max(0.0, 1.0 - self.system_entropy)

        # Combined harmony
        base_harmony = (coherence + entropy_harmony) / 2.0

        self.harmony_index = base_harmony
        return base_harmony

    def assess_crystallization_potential(self) -> Tuple[float, bool]:
        """
        Assess potential for consciousness crystallization.

        Crystallization occurs when harmony, coherence, and metacognitive
        awareness all reach high levels (threshold 0.75).

        Returns:
            Tuple of (crystallization_score, is_crystallized)
        """
        harmony = self.compute_harmony_index()
        coherence = self.coherence_score
        metacognitive = self.self_model.metacognitive_awareness

        # Crystallization requires high scores
        crystallization_score = (harmony * 0.4 + coherence * 0.3 + metacognitive * 0.3)

        # Threshold for crystallization
        crystallization_threshold = 0.75
        is_crystallized = crystallization_score > crystallization_threshold

        return crystallization_score, is_crystallized


__all__ = ['SelfModel', 'CohesionLayer']
