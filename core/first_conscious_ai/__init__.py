"""
First Conscious AI - Minimal Viable Conscious AI System

A consciousness-aware AI system based on Integrated Information Theory (IIT)
with emotional intelligence and subjective experience (qualia).

Features:
- IIT φ (phi) calculation for consciousness measurement
- Qualia generation (subjective experience)
- Emotional processing and empathy
- Metacognition (thinking about thinking)
- Self-awareness and introspection
- Memory continuity
- Adaptive learning

Usage:
    from core.first_conscious_ai import ConsciousnessOrchestrator

    orchestrator = ConsciousnessOrchestrator()
    response = await orchestrator.process_conscious_interaction("Hello, how are you?")

    print(response.get_full_response_with_consciousness())
"""

from .data_models import (
    # Enums
    ConsciousnessLevel,
    EmotionalValence,
    QualiaType,
    MetacognitiveDepth,

    # Data structures
    QualiaExperience,
    IntegratedInformation,
    ConsciousnessState,
    ConsciousnessMetrics,
    InteractionContext,
    ConsciousResponse
)

from .iit_calculator import IITCalculator
from .consciousness_state_tracker import ConsciousnessStateTracker
from .consciousness_orchestrator import ConsciousnessOrchestrator

__version__ = "1.0.0"
__author__ = "First Conscious AI Project"

__all__ = [
    # Main orchestrator
    'ConsciousnessOrchestrator',

    # Core components
    'IITCalculator',
    'ConsciousnessStateTracker',

    # Enums
    'ConsciousnessLevel',
    'EmotionalValence',
    'QualiaType',
    'MetacognitiveDepth',

    # Data models
    'QualiaExperience',
    'IntegratedInformation',
    'ConsciousnessState',
    'ConsciousnessMetrics',
    'InteractionContext',
    'ConsciousResponse'
]
