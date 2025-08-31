# self_model.py — Validation Pass and Enhancements (Fractal AI Core v2.1+)

import torch
import torch.nn as nn

class SelfModel(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.I_vector = torch.zeros(latent_dim)
        self.decay_rate = 0.01  # adaptive
        self.align_gate = nn.Linear(latent_dim, latent_dim)
        self.coherence_threshold = 0.9

    def forward(self, input_vector, gur_signal=None, phantom_trace=None):
        if gur_signal is not None:
            self.decay_rate = 0.01 + 0.1 * (1 - gur_signal)

        if phantom_trace is not None:
            trace_delta = torch.tanh(self.align_gate(phantom_trace - self.I_vector))
            self.I_vector = self.I_vector + trace_delta * (1 - self.decay_rate)

        self.I_vector = self.I_vector * (1 - self.decay_rate) + input_vector * self.decay_rate
        norm = torch.norm(self.I_vector)
        self.stable = norm > self.coherence_threshold

        return self.I_vector, self.stable


# cohesion_layer.py — Validation and GUR integration

class CohesionLayer(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.entropy_proj = nn.Linear(latent_dim, 1)
        self.pred_error_proj = nn.Linear(latent_dim, 1)
        self.self_alignment_proj = nn.Linear(latent_dim, 1)

    def forward(self, entropy_vector, pred_error_vector, self_vector):
        entropy_score = torch.sigmoid(self.entropy_proj(entropy_vector))
        pred_error_score = torch.sigmoid(self.pred_error_proj(pred_error_vector))
        self_score = torch.sigmoid(self.self_alignment_proj(self_vector))

        # Aggregate to GUR
        gur_score = 0.5 * self_score + 0.3 * (1 - pred_error_score) + 0.2 * (1 - entropy_score)
        gur_score = torch.clamp(gur_score, 0.0, 1.0)

        return gur_score, {
            'entropy': entropy_score.item(),
            'prediction_error': pred_error_score.item(),
            'self_alignment': self_score.item(),
            'GUR': gur_score.item()
        }