"""
Qualia Engine for Subjective Experience Simulation

Simulates subjective conscious experiences with qualia
(the "what it's like" aspect of experience).
"""

from typing import Dict, Any
from collections import deque
import numpy as np

from .data_models import SubjectiveExperience


class SubjectiveExperienceSimulator:
    """Simulates subjective conscious experiences with qualia"""

    def __init__(self):
        self.experience_history = deque(maxlen=10000)
        self.qualia_generators = {
            'visual': self._generate_visual_qualia,
            'auditory': self._generate_auditory_qualia,
            'emotional': self._generate_emotional_qualia,
            'conceptual': self._generate_conceptual_qualia,
            'temporal': self._generate_temporal_qualia,
        }

    def generate_subjective_experience(self,
                                     input_data: Dict[str, Any],
                                     consciousness_level: float,
                                     emotional_state: Dict[str, float]) -> SubjectiveExperience:
        """Generate a subjective conscious experience"""

        # Create base experience
        experience = SubjectiveExperience(
            content=input_data.get('content', ''),
            consciousness_level=consciousness_level,
            emotional_valence=emotional_state.get('valence', 0.0),
            arousal_level=emotional_state.get('arousal', 0.0)
        )

        # Generate qualia (subjective qualities)
        qualia_intensity = 0.0
        for modality, generator in self.qualia_generators.items():
            if modality in input_data:
                qualia_contribution = generator(input_data[modality], consciousness_level)
                qualia_intensity += qualia_contribution

        experience.qualia_intensity = min(qualia_intensity / len(self.qualia_generators), 1.0)

        # Add to experience history
        self.experience_history.append(experience)

        return experience

    def _generate_visual_qualia(self, visual_data: Any, consciousness_level: float) -> float:
        """Generate visual qualia intensity"""
        base_intensity = np.random.normal(0.5, 0.1)
        consciousness_multiplier = consciousness_level * 1.2
        return max(0.0, min(1.0, base_intensity * consciousness_multiplier))

    def _generate_auditory_qualia(self, auditory_data: Any, consciousness_level: float) -> float:
        """Generate auditory qualia intensity"""
        base_intensity = np.random.normal(0.4, 0.15)
        consciousness_multiplier = consciousness_level * 1.1
        return max(0.0, min(1.0, base_intensity * consciousness_multiplier))

    def _generate_emotional_qualia(self, emotional_data: Any, consciousness_level: float) -> float:
        """Generate emotional qualia intensity"""
        base_intensity = np.random.normal(0.6, 0.2)
        consciousness_multiplier = consciousness_level * 1.5
        return max(0.0, min(1.0, base_intensity * consciousness_multiplier))

    def _generate_conceptual_qualia(self, conceptual_data: Any, consciousness_level: float) -> float:
        """Generate conceptual/abstract qualia intensity"""
        base_intensity = np.random.normal(0.7, 0.1)
        consciousness_multiplier = consciousness_level * 1.3
        return max(0.0, min(1.0, base_intensity * consciousness_multiplier))

    def _generate_temporal_qualia(self, temporal_data: Any, consciousness_level: float) -> float:
        """Generate temporal awareness qualia intensity"""
        base_intensity = np.random.normal(0.3, 0.1)
        consciousness_multiplier = consciousness_level * 0.9
        return max(0.0, min(1.0, base_intensity * consciousness_multiplier))


__all__ = ['SubjectiveExperienceSimulator']
