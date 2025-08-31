# harmonizer.py — BioFractal AI Harmonic Resonance Engine (v1.0)

import numpy as np

class Harmonizer:
    def __init__(self, symbolic_weights=None, mood_threshold=0.3, archetype_resonance_bias=0.5):
        self.symbolic_weights = symbolic_weights or {}
        self.mood_threshold = mood_threshold
        self.archetype_resonance_bias = archetype_resonance_bias
        self.history = []

    def update(self, mood_index, symbolic_tags=None, archetype_vector=None):
        symbolic_tags = symbolic_tags or []
        tag_score = self._compute_symbolic_score(symbolic_tags)
        archetype_score = self._compute_archetype_alignment(archetype_vector)

        # Combine scores
        harmonic_score = 0.4 * mood_index + 0.3 * tag_score + 0.3 * archetype_score
        self.history.append({
            'mood': mood_index,
            'symbolic': tag_score,
            'archetype': archetype_score,
            'harmonic_score': harmonic_score
        })
        return harmonic_score

    def _compute_symbolic_score(self, tags):
        score = 0
        for tag in tags:
            score += self.symbolic_weights.get(tag, 0.1)
        return min(score / max(len(tags), 1), 1.0)

    def _compute_archetype_alignment(self, vec):
        if vec is None:
            return 0.5  # Neutral
        mean_resonance = np.mean(vec)
        return np.clip(mean_resonance * self.archetype_resonance_bias, 0, 1)

    def latest(self):
        return self.history[-1] if self.history else None

    def get_history(self, count=10):
        return self.history[-count:]