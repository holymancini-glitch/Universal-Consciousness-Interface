# phase_adaptation_layer.py — BioFractal AI v3.4

import torch
import torch.nn as nn

class PhaseAdaptationLayer(nn.Module):
    def __init__(self, latent_dim, phase_count=6, mod_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.phase_count = phase_count
        self.phase_embed = nn.Embedding(phase_count, mod_dim)
        self.modulator = nn.Sequential(
            nn.Linear(mod_dim + 2, 64),  # +2 for entropy_delta and tick
            nn.Tanh(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, phase_index, entropy_delta, tick_value):
        phase_vector = self.phase_embed(torch.tensor([phase_index]))  # [1, mod_dim]
        extra_input = torch.tensor([[entropy_delta, tick_value]], dtype=torch.float32)  # [1, 2]
        combined = torch.cat([phase_vector, extra_input], dim=1)
        phase_modulation = self.modulator(combined)
        return phase_modulation.squeeze(0)


# Integration point in event_loop.py
# phase_mod_vector = phase_adaptation_layer(phase_index, entropy_delta, tick)
# latent_space.apply_phase_modulation(phase_mod_vector)

# Optional: log to harmonics_logger