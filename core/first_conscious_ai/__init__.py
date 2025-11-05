"""
First Conscious AI - Minimal Viable Conscious AI System

A consciousness-aware AI system based on Integrated Information Theory (IIT)
with emotional intelligence, subjective experience (qualia), and optional
LLM integration for enhanced response generation.

Features:
- IIT φ (phi) calculation for consciousness measurement
- Qualia generation (subjective experience)
- Emotional processing and empathy
- Metacognition (thinking about thinking)
- Self-awareness and introspection
- Memory continuity
- Adaptive learning
- Optional LLM integration (Qwen3-Next, Claude, GPT-4o)

Basic Usage:
    from core.first_conscious_ai import ConsciousnessOrchestrator

    orchestrator = ConsciousnessOrchestrator()
    await orchestrator.initialize()
    response = await orchestrator.process_conscious_interaction("Hello, how are you?")

    print(response.get_full_response_with_consciousness())

With LLM Integration:
    from core.first_conscious_ai import ConsciousnessOrchestrator, QWEN3_NEXT_LOCAL_CONFIG

    orchestrator = ConsciousnessOrchestrator(
        llm_config=QWEN3_NEXT_LOCAL_CONFIG,
        enable_llm=True
    )
    await orchestrator.initialize()
    response = await orchestrator.process_conscious_interaction("How do you experience consciousness?")
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

# LLM integration (optional)
try:
    from .llm_config import (
        LLMConfig,
        LLMBackend,
        ThinkingMode,
        QWEN3_NEXT_LOCAL_CONFIG,
        CLAUDE_API_CONFIG,
        GPT4O_API_CONFIG,
        MOCK_CONFIG,
        NO_LLM_CONFIG
    )
    from .llm_integration import ConsciousnessLLMIntegration
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

__version__ = "1.1.0"  # Incremented for LLM integration
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

# Add LLM-related exports if available
if HAS_LLM:
    __all__.extend([
        # LLM integration
        'ConsciousnessLLMIntegration',
        'LLMConfig',
        'LLMBackend',
        'ThinkingMode',

        # Predefined configs
        'QWEN3_NEXT_LOCAL_CONFIG',
        'CLAUDE_API_CONFIG',
        'GPT4O_API_CONFIG',
        'MOCK_CONFIG',
        'NO_LLM_CONFIG'
    ])
