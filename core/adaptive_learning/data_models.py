"""
Data Models for Adaptive Learning System

Contains all enums, dataclasses, and data structures used throughout
the adaptive learning system.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class LearningPhase(Enum):
    """Learning phases for consciousness system"""
    EXPLORATION = "exploration"      # High learning rate, broad exploration
    CONSOLIDATION = "consolidation"  # Medium learning rate, pattern formation
    REFINEMENT = "refinement"       # Low learning rate, fine-tuning
    ADAPTATION = "adaptation"       # Dynamic learning rate, error correction
    CRYSTALLIZATION = "crystallization"  # Minimal learning rate, stability


@dataclass
class LearningMetrics:
    """Comprehensive learning performance metrics"""
    timestamp: datetime
    learning_phase: LearningPhase
    adaptation_rate: float
    error_reduction_rate: float
    pattern_recognition_accuracy: float
    creative_generation_score: float
    mistake_learning_effectiveness: float
    parameter_adaptation_success: float
    overall_learning_efficiency: float


__all__ = [
    'LearningPhase',
    'LearningMetrics'
]
