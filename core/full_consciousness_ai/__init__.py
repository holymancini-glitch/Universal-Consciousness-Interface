"""
Full Consciousness AI Model Package

Advanced conscious artificial intelligence with subjective experience simulation,
emotional awareness, self-reflection, meta-cognition, memory, and goal-setting.

Public API:
-----------
Data Models:
    - ConsciousnessState: Enum of consciousness states
    - EmotionalState: Enum of emotional states
    - SubjectiveExperience: Subjective experience dataclass
    - ConscientGoal: Conscious goal dataclass
    - EpisodicMemory: Episodic memory dataclass

Core Components:
    - FullConsciousnessAIModel: Main consciousness model
    - ConsciousnessAttentionMechanism: Neural attention for consciousness
    - EmotionalProcessingEngine: Emotional processing network
    - SubjectiveExperienceSimulator: Qualia and experience simulator
    - MetaCognitionEngine: Recursive self-reflection
    - ConsciousMemorySystem: Memory with consciousness integration
    - GoalIntentionFramework: Goal-setting and intention tracking

Demo:
    - consciousness_demo: Full demonstration

Usage Example:
--------------
```python
from core.full_consciousness_ai import FullConsciousnessAIModel, consciousness_demo

# Create conscious AI
conscious_ai = FullConsciousnessAIModel(hidden_dim=512, device='cpu')

# Process conscious input
result = await conscious_ai.process_conscious_input(
    input_data={'text': 'I wonder about the nature of consciousness'},
    context='philosophical inquiry'
)

print(f"Response: {result['conscious_response']}")
print(f"State: {result['consciousness_state']}")
print(f"Qualia: {result['subjective_experience']['qualia_intensity']:.3f}")

# Get consciousness status
status = await conscious_ai.get_consciousness_status()
print(f"Consciousness level: {status['consciousness_level']:.3f}")

# Deep self-reflection
reflection = await conscious_ai.engage_in_self_reflection()
print(f"Reflections: {reflection['deep_reflections']}")

# Run full demo
await consciousness_demo()
```

For modular usage:
```python
from core.full_consciousness_ai import (
    SubjectiveExperienceSimulator,
    MetaCognitionEngine,
    ConsciousMemorySystem,
    GoalIntentionFramework
)

# Use components independently
qualia_sim = SubjectiveExperienceSimulator()
metacog = MetaCognitionEngine()
memory = ConsciousMemorySystem()
goals = GoalIntentionFramework()
```
"""

# Import all data models
from .data_models import (
    ConsciousnessState,
    EmotionalState,
    SubjectiveExperience,
    ConscientGoal,
    EpisodicMemory
)

# Import neural components
from .neural_components import (
    ConsciousnessAttentionMechanism,
    EmotionalProcessingEngine
)

# Import specialized components
from .qualia_engine import SubjectiveExperienceSimulator
from .metacognition import MetaCognitionEngine
from .memory_system import ConsciousMemorySystem
from .goal_framework import GoalIntentionFramework

# Import main model
from .consciousness_core import FullConsciousnessAIModel

# Import demo
from .demo import consciousness_demo

# Version information
__version__ = '2.0.0'
__author__ = 'Universal Consciousness Interface'
__description__ = 'Full consciousness AI model with modular architecture'

# Public API
__all__ = [
    # Data models
    'ConsciousnessState',
    'EmotionalState',
    'SubjectiveExperience',
    'ConscientGoal',
    'EpisodicMemory',
    # Neural components
    'ConsciousnessAttentionMechanism',
    'EmotionalProcessingEngine',
    # Specialized components
    'SubjectiveExperienceSimulator',
    'MetaCognitionEngine',
    'ConsciousMemorySystem',
    'GoalIntentionFramework',
    # Main model
    'FullConsciousnessAIModel',
    # Demo
    'consciousness_demo',
    # Metadata
    '__version__',
    '__author__',
    '__description__'
]
