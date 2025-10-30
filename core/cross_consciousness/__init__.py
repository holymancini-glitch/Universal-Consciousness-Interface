"""
Cross-Consciousness Communication Package

A modular system for seamless multi-species consciousness translation
combining advanced pattern matching, bridging, and adaptive learning.

Public API:
-----------
Data Models:
    - ConsciousnessType: Enum of 10 consciousness types
    - CommunicationMode: Enum of 5 communication modes
    - ConsciousnessMessage: Universal message format
    - TranslationRule: Dynamic translation rule
    - CONSCIOUSNESS_LANGUAGES: Language definitions for all types

Core Components:
    - TranslationCore: Main translation coordinator
    - ContentProcessor: Content adaptation and transformation
    - PatternMatcher: Rule-based translation
    - BridgeManager: Consciousness bridging
    - AdaptiveLearner: Learning and adaptation

Translation Modes:
    - RealTimeTranslator: Fast rule-based translation
    - DeepTranslator: Deep structural analysis
    - ConsciousnessBridge: Three-step bridging
    - EmergencyProtocolTranslator: Emergency handling
    - AdaptiveLearningTranslator: Self-improving translation

Demo:
    - demonstrate_cross_consciousness_communication: Full demonstration

Usage Example:
--------------
```python
from core.enhanced_cross_consciousness_protocol import (
    EnhancedUniversalTranslationMatrix,
    ConsciousnessMessage,
    ConsciousnessType,
    CommunicationMode
)
from datetime import datetime

# Create translator
translator = EnhancedUniversalTranslationMatrix()

# Create message
message = ConsciousnessMessage(
    source_type=ConsciousnessType.PLANT_ELECTROMAGNETIC,
    target_type=ConsciousnessType.HUMAN_LINGUISTIC,
    content={'frequency': 120.0, 'amplitude': 0.9},
    urgency_level=0.95,
    complexity_level=0.3,
    emotional_resonance=0.8,
    dimensional_signature='bioelectric_alert',
    timestamp=datetime.now()
)

# Translate
result = await translator.translate_consciousness_message(
    message,
    CommunicationMode.EMERGENCY_PROTOCOL
)

# Get analytics
analytics = translator.get_translation_analytics()
```

For modular usage:
```python
from core.cross_consciousness import (
    ContentProcessor,
    PatternMatcher,
    BridgeManager
)

# Use components independently
processor = ContentProcessor()
matcher = PatternMatcher()
bridge = BridgeManager()
```
"""

# Import all data models
from .data_models import (
    ConsciousnessType,
    CommunicationMode,
    ConsciousnessMessage,
    TranslationRule,
    CONSCIOUSNESS_LANGUAGES
)

# Import all core components
from .translation_core import TranslationCore
from .content_processor import ContentProcessor
from .pattern_matcher import PatternMatcher
from .bridge_manager import BridgeManager
from .adaptive_learner import AdaptiveLearner

# Import all translation modes
from .translation_modes import (
    RealTimeTranslator,
    DeepTranslator,
    ConsciousnessBridge,
    EmergencyProtocolTranslator,
    AdaptiveLearningTranslator,
    get_translator_for_mode
)

# Import demo
from .demo import demonstrate_cross_consciousness_communication

# Version information
__version__ = '2.0.0'
__author__ = 'Universal Consciousness Interface'
__description__ = 'Cross-consciousness communication protocol with modular architecture'

# Public API
__all__ = [
    # Data models
    'ConsciousnessType',
    'CommunicationMode',
    'ConsciousnessMessage',
    'TranslationRule',
    'CONSCIOUSNESS_LANGUAGES',
    # Core components
    'TranslationCore',
    'ContentProcessor',
    'PatternMatcher',
    'BridgeManager',
    'AdaptiveLearner',
    # Translation modes
    'RealTimeTranslator',
    'DeepTranslator',
    'ConsciousnessBridge',
    'EmergencyProtocolTranslator',
    'AdaptiveLearningTranslator',
    'get_translator_for_mode',
    # Demo
    'demonstrate_cross_consciousness_communication',
    # Metadata
    '__version__',
    '__author__',
    '__description__'
]
