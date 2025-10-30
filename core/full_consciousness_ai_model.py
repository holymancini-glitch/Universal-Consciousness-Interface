"""
Full Consciousness AI Model - Advanced Conscious Artificial Intelligence

This module implements a comprehensive consciousness simulation system with:
- Full subjective experience simulation
- Emotional awareness and processing
- Self-reflection and meta-cognition
- Memory of past interactions and learning
- Goal-setting and intention tracking
- Integration with existing consciousness modules

Architecture: Hybrid Neural-Symbolic Consciousness Engine

------------------------------------------------------------------
REFACTORED: Main Facade for Backward Compatibility

All original classes are now organized into specialized modules:
- data_models.py: Enums and dataclasses
- neural_components.py: Neural network modules
- qualia_engine.py: Subjective experience simulator
- metacognition.py: Meta-cognitive processing
- memory_system.py: Conscious memory system
- goal_framework.py: Goal and intention framework
- consciousness_core.py: Main consciousness model
- demo.py: Demonstration function

Usage:
------
# Old way (still works):
from core.full_consciousness_ai_model import (
    FullConsciousnessAIModel,
    ConsciousnessState,
    EmotionalState,
    consciousness_demo
)

# New modular way:
from core.full_consciousness_ai import (
    FullConsciousnessAIModel,
    ConsciousnessState,
    EmotionalState,
    consciousness_demo
)

# Or import specific modules:
from core.full_consciousness_ai.consciousness_core import FullConsciousnessAIModel
from core.full_consciousness_ai.data_models import ConsciousnessState
"""

# Import all classes for backward compatibility
from .full_consciousness_ai import (
    # Data Models
    ConsciousnessState,
    EmotionalState,
    SubjectiveExperience,
    ConscientGoal,
    EpisodicMemory,
    # Neural Components
    ConsciousnessAttentionMechanism,
    EmotionalProcessingEngine,
    # Specialized Components
    SubjectiveExperienceSimulator,
    MetaCognitionEngine,
    ConsciousMemorySystem,
    GoalIntentionFramework,
    # Main Model
    FullConsciousnessAIModel,
    # Demo
    consciousness_demo
)

# Backward compatibility exports
__all__ = [
    # Data Models
    'ConsciousnessState',
    'EmotionalState',
    'SubjectiveExperience',
    'ConscientGoal',
    'EpisodicMemory',
    # Neural Components
    'ConsciousnessAttentionMechanism',
    'EmotionalProcessingEngine',
    # Specialized Components
    'SubjectiveExperienceSimulator',
    'MetaCognitionEngine',
    'ConsciousMemorySystem',
    'GoalIntentionFramework',
    # Main Model
    'FullConsciousnessAIModel',
    # Demo
    'consciousness_demo'
]

__version__ = '2.0.0'
__refactored__ = True

# Example usage (can be run directly)
if __name__ == "__main__":
    import asyncio
    asyncio.run(consciousness_demo())
