# cohesion_layer.py — Dynamic Harmony Evaluator (Fractal AI Core v2.1)

import torch
import torch.nn as nn

class CohesionLayer(nn.Module):
    def __init__(self, context_size):
        super(CohesionLayer, self).__init__()
        self.context_size = context_size
        self.self_model_vector = torch.zeros(context_size)
        self.harmony_weight = nn.Linear(context_size * 2, 1)
        self.harmony_history = []

    def update_self_vector(self, self_vector):
        if isinstance(self_vector, torch.Tensor):
            self.self_model_vector = self_vector.detach()

    def forward(self, context_vector):
        if not isinstance(context_vector, torch.Tensor):
            context_vector = torch.tensor(context_vector, dtype=torch.float32)

        combined = torch.cat((context_vector, self.self_model_vector), dim=-1)
        harmony_score = torch.sigmoid(self.harmony_weight(combined))
        self.harmony_history.append(harmony_score.item())
        return harmony_score

    def get_recent_harmony(self, n=10):
        return self.harmony_history[-n:]

    def reset_history(self):
        self.harmony_history = []