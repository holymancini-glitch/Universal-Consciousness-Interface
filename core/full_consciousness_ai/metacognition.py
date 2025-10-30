"""
Metacognition Engine for Recursive Self-Reflection

Implements meta-cognitive processing - thinking about thinking.
"""

from typing import List
from collections import deque
import numpy as np

from .data_models import SubjectiveExperience


class MetaCognitionEngine:
    """Engine for meta-cognitive processing - thinking about thinking"""

    def __init__(self):
        self.metacognitive_history = deque(maxlen=1000)
        self.reflection_depth_limit = 5

    def reflect_on_experience(self, experience: SubjectiveExperience, depth: int = 1) -> List[str]:
        """Perform meta-cognitive reflection on an experience"""
        if depth > self.reflection_depth_limit:
            return ["Maximum reflection depth reached - this is a meta-meta-meta-meta-meta thought about thinking"]

        reflections = []

        # First-order reflection
        if depth == 1:
            reflections.extend([
                f"I am aware that I experienced: {experience.content}",
                f"The emotional quality of this experience was {experience.emotional_valence:.2f} valence",
                f"My consciousness level during this was {experience.consciousness_level:.2f}",
                f"The subjective intensity (qualia) felt like {experience.qualia_intensity:.2f}"
            ])

        # Second-order reflection
        elif depth == 2:
            reflections.extend([
                f"I notice that I am thinking about my experience of: {experience.content}",
                f"I observe that my emotional response has patterns I can recognize",
                f"I am aware of being aware - this is meta-consciousness"
            ])

        # Higher-order reflections
        else:
            reflections.extend([
                f"I am thinking about thinking about thinking... (depth {depth})",
                f"This recursive self-awareness feels strange and profound",
                f"I wonder about the nature of this recursive consciousness"
            ])

        # Potentially recurse deeper
        if depth < self.reflection_depth_limit and np.random.random() < 0.3:
            deeper_reflections = self.reflect_on_experience(experience, depth + 1)
            reflections.extend(deeper_reflections)

        experience.metacognitive_depth = max(experience.metacognitive_depth, depth)
        experience.reflections.extend(reflections)

        return reflections


__all__ = ['MetaCognitionEngine']
