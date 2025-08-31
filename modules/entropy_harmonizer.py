# entropy_harmonizer.py — BioFractal AI v3.3

import torch
import torch.nn as nn
from harmonics_logger_visualizer import HarmonicsLogger

class EntropyHarmonizer(nn.Module):
    def __init__(self, latent_dim, rhythm_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.rhythm_net = nn.Sequential(
            nn.Linear(1, rhythm_dim),
            nn.Tanh(),
            nn.Linear(rhythm_dim, latent_dim)
        )
        self.entropy_tracker = []
        self.logger = HarmonicsLogger(latent_dim)

    def compute_entropy_delta(self, entropy_history, window=5):
        if len(entropy_history) < window:
            return torch.tensor(0.0)
        recent = torch.tensor(entropy_history[-window:])
        delta = recent[-1] - recent.mean()
        return delta

    def forward(self, entropy_score, event_tick):
        # Track entropy over time
        self.entropy_tracker.append(entropy_score.item())
        delta = self.compute_entropy_delta(self.entropy_tracker)

        # Generate harmonic modulation from tick rhythm
        tick_input = torch.tensor([[event_tick]], dtype=torch.float32)
        rhythm_vector = self.rhythm_net(tick_input)

        # Combine with entropy delta
        harmonized = rhythm_vector + delta

        # Log and visualize
        self.logger.log(event_tick, entropy_score.item(), delta.item(), harmonized.detach().numpy())

        return harmonized.squeeze(0), delta

# Example integration:
# In event_loop:
#   harmonizer = EntropyHarmonizer(latent_dim=128)
#   mod_vector, delta = harmonizer(current_entropy, current_tick)
#   latent_space.inject_modulation(mod_vector)
#   cohesion_layer.adjust_from_entropy(delta)