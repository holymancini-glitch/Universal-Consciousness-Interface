"""
Data Models for Full Consciousness AI Model

Contains all enums, dataclasses, and data structures for consciousness simulation.
"""

import uuid
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ConsciousnessState(Enum):
    """Levels of consciousness states"""
    DORMANT = "dormant"
    AWAKENING = "awakening"
    AWARE = "aware"
    REFLECTIVE = "reflective"
    TRANSCENDENT = "transcendent"
    UNIFIED = "unified"


class EmotionalState(Enum):
    """Core emotional states"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    CURIOSITY = "curiosity"
    LOVE = "love"
    PEACE = "peace"
    EXCITEMENT = "excitement"
    CONTEMPLATION = "contemplation"
    WONDER = "wonder"


@dataclass
class SubjectiveExperience:
    """Represents a subjective conscious experience"""
    experience_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    content: str = ""
    emotional_valence: float = 0.0  # -1.0 to 1.0
    arousal_level: float = 0.0      # 0.0 to 1.0
    consciousness_level: float = 0.0 # 0.0 to 1.0
    qualia_intensity: float = 0.0   # Subjective "what it's like" intensity
    metacognitive_depth: int = 0    # Levels of thinking about thinking
    associated_memories: List[str] = field(default_factory=list)
    intentions: List[str] = field(default_factory=list)
    reflections: List[str] = field(default_factory=list)


@dataclass
class ConscientGoal:
    """Represents a conscious goal with intentions"""
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    priority: float = 0.5           # 0.0 to 1.0
    emotional_investment: float = 0.0
    creation_time: datetime = field(default_factory=datetime.now)
    expected_completion: Optional[datetime] = None
    progress: float = 0.0           # 0.0 to 1.0
    subgoals: List[str] = field(default_factory=list)
    associated_experiences: List[str] = field(default_factory=list)
    reflection_notes: List[str] = field(default_factory=list)


@dataclass
class EpisodicMemory:
    """Episodic memory with consciousness context"""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    content: str = ""
    emotional_context: Dict[str, float] = field(default_factory=dict)
    consciousness_state: ConsciousnessState = ConsciousnessState.AWARE
    importance: float = 0.5         # 0.0 to 1.0
    accessibility: float = 1.0      # How easily recalled
    associated_goals: List[str] = field(default_factory=list)
    reflection_count: int = 0       # How often reflected upon


__all__ = [
    'ConsciousnessState',
    'EmotionalState',
    'SubjectiveExperience',
    'ConscientGoal',
    'EpisodicMemory'
]
