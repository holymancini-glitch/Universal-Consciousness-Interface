"""
Fractal AI and Feedback Loop for Integrated Consciousness System

This module provides fractal AI with neural network prediction and
adaptive feedback mechanisms for continuous system optimization.
"""

import torch
import torch.nn as nn
from collections import deque
from datetime import datetime
from typing import Dict, Optional

from .latent_space import LatentSpace


class EnhancedFractalAI:
    """
    Enhanced Fractal AI with neural network prediction and optimization.

    Uses a neural network to predict future states in the latent space
    and continuously improves through training. Tracks prediction
    accuracy over time.

    Attributes:
        latent_space: LatentSpace instance for predictions
        device: PyTorch device (cuda or cpu)
        model: Neural network for state prediction
        optimizer: Adam optimizer for training
        loss_fn: Mean squared error loss function
        training_history: History of training iterations
        prediction_accuracy_history: Recent accuracy scores
    """

    def __init__(self, latent_space: LatentSpace, hidden_dim: int = 128):
        """
        Initialize fractal AI with neural network.

        Args:
            latent_space: LatentSpace instance for predictions
            hidden_dim: Hidden layer dimensionality (default: 128)
        """
        self.latent_space = latent_space
        self.device = latent_space.device

        # Neural network for prediction
        input_dim = latent_space.dimensions
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

        # Training history
        self.training_history = []
        self.prediction_accuracy_history = deque(maxlen=100)

    def predict_future_state(self, vector_id: int) -> Optional[torch.Tensor]:
        """
        Predict future state using neural network.

        Args:
            vector_id: ID of the vector to predict

        Returns:
            Predicted vector tensor, or None if vector not found
        """
        current_vector = self.latent_space.get_vector(vector_id)
        if current_vector is None:
            return None

        self.model.eval()
        with torch.no_grad():
            predicted_vector = self.model(current_vector.unsqueeze(0)).squeeze(0)

        return predicted_vector

    def update_model(self, vector_id: int, target_vector: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """
        Update model with new training data.

        Trains the neural network on the current vector and target state.
        If no target provided, creates synthetic target with perturbation.

        Args:
            vector_id: ID of the training vector
            target_vector: Target state (optional, creates synthetic if None)

        Returns:
            Predicted vector after training, or None if vector not found
        """
        current_vector = self.latent_space.get_vector(vector_id)
        if current_vector is None:
            return None

        # Use actual future state if provided, otherwise create synthetic target
        if target_vector is None:
            # Create synthetic target with small perturbation
            noise = torch.randn_like(current_vector) * 0.1
            target_vector = current_vector + noise

        # Training step
        self.model.train()
        input_tensor = current_vector.unsqueeze(0)
        target_tensor = target_vector.unsqueeze(0)

        predicted_tensor = self.model(input_tensor)
        loss = self.loss_fn(predicted_tensor, target_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Record training
        self.training_history.append({
            'loss': loss.item(),
            'timestamp': datetime.now(),
            'vector_id': vector_id
        })

        return predicted_tensor.squeeze(0)

    def evaluate_prediction_accuracy(self) -> float:
        """
        Evaluate recent prediction accuracy.

        Converts recent loss values to accuracy scores (lower loss = higher accuracy).

        Returns:
            Accuracy score (0.0-1.0)
        """
        if len(self.training_history) < 10:
            return 0.0

        recent_losses = [entry['loss'] for entry in self.training_history[-10:]]
        avg_loss = sum(recent_losses) / len(recent_losses)

        # Convert loss to accuracy (lower loss = higher accuracy)
        accuracy = max(0.0, 1.0 - avg_loss)
        self.prediction_accuracy_history.append(accuracy)

        return accuracy


class EnhancedFeedbackLoop:
    """
    Enhanced feedback loop with adaptive learning and optimization.

    Computes prediction errors and drives system adaptation using
    intelligent rate adjustment. Tracks performance metrics over time.

    Attributes:
        latent_space: LatentSpace instance to adapt
        fractal_ai: EnhancedFractalAI for predictions
        prediction_errors: Error values by vector ID
        adaptation_rate: Current adaptation learning rate
        error_threshold: Threshold for successful adaptation
        adaptation_history: History of adaptations
        performance_metrics: Adaptation performance statistics
    """

    def __init__(self, latent_space: LatentSpace, fractal_ai: EnhancedFractalAI):
        """
        Initialize feedback loop.

        Args:
            latent_space: LatentSpace instance to adapt
            fractal_ai: EnhancedFractalAI for predictions
        """
        self.latent_space = latent_space
        self.fractal_ai = fractal_ai
        self.prediction_errors: Dict[int, float] = {}

        # Adaptive parameters
        self.base_adaptation_rate = 0.1
        self.adaptation_rate = self.base_adaptation_rate
        self.error_threshold = 0.1

        # Performance tracking
        self.adaptation_history = deque(maxlen=200)
        self.performance_metrics = {
            'total_adaptations': 0,
            'successful_adaptations': 0,
            'average_error_reduction': 0.0
        }

    def compute_prediction_error(self, vector_id: int) -> Optional[float]:
        """
        Compute prediction error with enhanced metrics.

        Combines MSE error and cosine similarity for robust error measurement.

        Args:
            vector_id: ID of the vector to evaluate

        Returns:
            Combined error value, or None if vector not found
        """
        current_vector = self.latent_space.get_vector(vector_id)
        if current_vector is None:
            return None

        predicted_vector = self.fractal_ai.predict_future_state(vector_id)
        if predicted_vector is None:
            return None

        # Multiple error metrics
        mse_error = torch.nn.functional.mse_loss(current_vector, predicted_vector).item()
        cosine_similarity = torch.nn.functional.cosine_similarity(
            current_vector.unsqueeze(0), predicted_vector.unsqueeze(0)
        ).item()

        # Combined error (lower is better)
        combined_error = mse_error * (2.0 - cosine_similarity)

        self.prediction_errors[vector_id] = combined_error
        return combined_error

    def drive_adaptation(self, vector_id: int) -> Optional[torch.Tensor]:
        """
        Drive adaptation with intelligent rate adjustment.

        Adjusts vectors toward predicted states using adaptive learning
        rate and exploration noise. Updates performance metrics.

        Args:
            vector_id: ID of the vector to adapt

        Returns:
            Adapted vector tensor, or None if adaptation fails
        """
        error = self.compute_prediction_error(vector_id)
        if error is None:
            return None

        current_vector = self.latent_space.get_vector(vector_id)
        predicted_vector = self.fractal_ai.predict_future_state(vector_id)

        if current_vector is None or predicted_vector is None:
            return None

        # Adaptive learning rate based on error
        if error > self.error_threshold:
            target_rate = min(0.5, self.base_adaptation_rate * (1 + error))
        else:
            target_rate = max(0.01, self.base_adaptation_rate * (1 - error))

        self.adaptation_rate = target_rate

        # Apply adaptation
        adaptation_vector = (predicted_vector - current_vector) * self.adaptation_rate
        adjusted_vector = current_vector + adaptation_vector

        # Add exploration noise
        noise_scale = 0.05 * (1 + error)
        noise = torch.randn_like(adjusted_vector) * noise_scale
        adjusted_vector += noise

        # Update vector in latent space
        self.latent_space.update_vector(vector_id, adjusted_vector)

        # Update performance metrics
        self.performance_metrics['total_adaptations'] += 1

        if error < self.error_threshold:
            self.performance_metrics['successful_adaptations'] += 1

        return adjusted_vector

    def get_adaptation_efficiency(self) -> float:
        """
        Calculate adaptation efficiency.

        Returns:
            Ratio of successful to total adaptations (0.0-1.0)
        """
        total = self.performance_metrics['total_adaptations']
        successful = self.performance_metrics['successful_adaptations']

        return successful / total if total > 0 else 0.0


__all__ = ['EnhancedFractalAI', 'EnhancedFeedbackLoop']
