# selfmodel_cohesion_integration.py — BioFractal AI v3.3 Extension Layer

import torch
import torch.nn as nn

class SelfModel(nn.Module):
    def __init__(self, latent_dim):
        super(SelfModel, self).__init__()
        self.latent_dim = latent_dim
        self.self_vector = nn.Parameter(torch.zeros(latent_dim))
        self.coherence_tracker = nn.CosineSimilarity(dim=0)

    def forward(self, current_latent):
        coherence_score = self.coherence_tracker(current_latent, self.self_vector)
        return coherence_score

    def update_self_vector(self, current_latent, alpha=0.05):
        # Soft update toward current latent state
        with torch.no_grad():
            self.self_vector.data = (1 - alpha) * self.self_vector + alpha * current_latent

class CohesionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(CohesionLayer, self).__init__()
        self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        # Estimate global harmony (0 to 1)
        return self.sigmoid(self.linear(features))

# Example integration usage
def integrate_selfmodel_cohesion(feedback_output, self_model, cohesion_layer):
    current_context = feedback_output['context']
    self_model.update_self_vector(current_context)
    coherence_score = self_model(current_context)
    cohesion_score = cohesion_layer(current_context)

    feedback_output['coherence_score'] = coherence_score.item()
    feedback_output['cohesion_score'] = cohesion_score.item()
    return feedback_output