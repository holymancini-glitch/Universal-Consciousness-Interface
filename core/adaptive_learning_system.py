# adaptive_learning_system.py
# Advanced Adaptive Learning System for Garden of Consciousness
# Addresses "lack of adaptability" and "insufficient learning" identified in technical review

"""
Adaptive Learning System - Main Facade

This module serves as the main entry point and backward-compatible interface
for the refactored adaptive learning package.

All original classes are now organized into specialized modules:
- data_models.py: Enums and dataclasses
- learning_core.py: Main learning system coordinator
- performance_assessment.py: Performance metrics and assessment
- parameter_adaptation.py: Dynamic parameter adaptation
- mistake_learning.py: Mistake analysis and learning
- creative_engine.py: Creative solution generation

Usage:
------
# Old way (still works):
from core.adaptive_learning_system import (
    AdaptiveLearningSystem,
    LearningPhase,
    LearningMetrics,
    integrate_adaptive_learning
)

# New modular way:
from core.adaptive_learning import (
    AdaptiveLearningSystem,
    LearningPhase,
    LearningMetrics,
    integrate_adaptive_learning
)

# Or import specific modules:
from core.adaptive_learning.learning_core import AdaptiveLearningSystem
from core.adaptive_learning.data_models import LearningPhase
"""

# Import all classes for backward compatibility
from .adaptive_learning import (
    # Data Models
    LearningPhase,
    LearningMetrics,
    # Core Components
    AdaptiveLearningSystem,
    PerformanceAssessor,
    ParameterAdaptor,
    MistakeLearner,
    CreativeEngine,
    # Integration
    integrate_adaptive_learning
)

# Backward compatibility exports
__all__ = [
    # Data Models
    'LearningPhase',
    'LearningMetrics',
    # Core Components
    'AdaptiveLearningSystem',
    'PerformanceAssessor',
    'ParameterAdaptor',
    'MistakeLearner',
    'CreativeEngine',
    # Integration
    'integrate_adaptive_learning'
]

__version__ = '2.0.0'
__refactored__ = True

# Example usage
if __name__ == "__main__":
    print("🧠 Advanced Adaptive Learning System Ready")
    print("Use: learning_system = await integrate_adaptive_learning(consciousness_system)")
