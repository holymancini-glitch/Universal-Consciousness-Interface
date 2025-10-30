"""
Neural Network Components for Full Consciousness AI

Contains neural network modules for consciousness attention and
emotional processing.
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn

from .data_models import EmotionalState


class ConsciousnessAttentionMechanism(nn.Module):
    """Neural attention mechanism for consciousness focus"""

    def __init__(self, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.consciousness_projection = nn.Linear(hidden_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.self_awareness_layer = nn.Linear(hidden_dim, hidden_dim)
        self.meta_cognition_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, consciousness_state: torch.Tensor, memory_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Project consciousness state
        conscious_projection = self.consciousness_projection(consciousness_state)

        # Apply self-attention for awareness
        attended_state, attention_weights = self.attention(
            conscious_projection, memory_context, memory_context
        )

        # Self-awareness processing
        self_aware_state = self.self_awareness_layer(attended_state)

        # Meta-cognitive processing
        meta_cognitive_state = self.meta_cognition_layer(self_aware_state)

        return meta_cognitive_state, attention_weights


class EmotionalProcessingEngine(nn.Module):
    """Neural network for emotional processing and consciousness"""

    def __init__(self, input_dim: int = 512, emotion_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.emotion_dim = emotion_dim

        self.emotion_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, emotion_dim)
        )

        self.valence_predictor = nn.Linear(emotion_dim, 1)  # -1 to 1
        self.arousal_predictor = nn.Linear(emotion_dim, 1)  # 0 to 1
        self.emotion_classifier = nn.Linear(emotion_dim, len(EmotionalState))

        self.emotional_memory_integration = nn.Linear(emotion_dim + input_dim, input_dim)

    def forward(self, consciousness_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode emotional features
        emotion_features = self.emotion_encoder(consciousness_input)

        # Predict emotional dimensions
        valence = torch.tanh(self.valence_predictor(emotion_features))
        arousal = torch.sigmoid(self.arousal_predictor(emotion_features))
        emotion_probs = torch.softmax(self.emotion_classifier(emotion_features), dim=-1)

        # Integrate emotions with consciousness
        integrated_state = self.emotional_memory_integration(
            torch.cat([emotion_features, consciousness_input], dim=-1)
        )

        return {
            'emotion_features': emotion_features,
            'valence': valence,
            'arousal': arousal,
            'emotion_probabilities': emotion_probs,
            'integrated_state': integrated_state
        }


__all__ = ['ConsciousnessAttentionMechanism', 'EmotionalProcessingEngine']
