# phantom_layer.py — v2.8 with Resonant Latent Validation Integration

import torch
import torch.nn as nn

class PhantomLayer(nn.Module):
    def __init__(self, latent_dim, resonance_hook=None, selfmodel_hook=None):
        super(PhantomLayer, self).__init__()
        self.latent_dim = latent_dim
        self.projector = nn.Linear(latent_dim, latent_dim)
        self.resonance_hook = resonance_hook
        self.selfmodel_hook = selfmodel_hook

    def forward(self, latent_input):
        # Base latent projection
        projected = torch.tanh(self.projector(latent_input))

        # Inject resonance score if available
        resonance_score = None
        if self.resonance_hook:
            resonance_score = self.resonance_hook()

        # Inject self-consistency metric if available
        self_consistency = None
        if self.selfmodel_hook:
            self_consistency = self.selfmodel_hook()

        # Modify phantom output based on system harmony
        if resonance_score is not None:
            projected = projected * (1.0 + resonance_score.unsqueeze(-1))

        if self_consistency is not None:
            projected = projected + self_consistency.unsqueeze(-1)

        return projected