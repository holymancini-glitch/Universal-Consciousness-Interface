"""
Mycelium Language Generation Package

A modular system for generating novel languages based on mycelium network
communication patterns, combining chemical signals, electrical pulses, and
network topology into linguistic structures.

Public API:
-----------
Data Models:
    - MyceliumCommunicationType: Enum of signal types
    - MyceliumSignal: Individual communication signal
    - MyceliumWord: Word in mycelium language
    - MyceliumSentence: Sentence structure

Core Classes:
    - VocabularyManager: Manages phonetic/chemical vocabularies
    - NetworkProcessor: Handles network topology and signals
    - BiochemicalTranslator: Translates signals to phonetics
    - PatternAnalyzer: Analyzes syntactic and semantic patterns
    - LanguageSynthesizer: Generates words and sentences
    - EvolutionEngine: Evolves language patterns over time

Usage Example:
--------------
```python
from core.mycelium_language import (
    VocabularyManager,
    NetworkProcessor,
    BiochemicalTranslator,
    PatternAnalyzer,
    LanguageSynthesizer,
    EvolutionEngine
)

# Initialize components
vocab = VocabularyManager()
network = NetworkProcessor(network_size=1000)
translator = BiochemicalTranslator(vocab)
analyzer = PatternAnalyzer(network)
synthesizer = LanguageSynthesizer(translator, analyzer)
evolution = EvolutionEngine(vocab.phonetic_library)

# Generate signals and create language
signals = network.generate_sample_signals(10)
tokens = await translator.process_signals_to_tokens(signals)
words = await synthesizer.generate_words_from_patterns(tokens, 'network_cognition')
```
"""

# Import all data models
from .data_models import (
    MyceliumCommunicationType,
    MyceliumSignal,
    MyceliumWord,
    MyceliumSentence,
    SignalList,
    WordList,
    SentenceList
)

# Import all core classes
from .vocabulary_manager import VocabularyManager
from .network_processor import NetworkProcessor
from .biochemical_translator import BiochemicalTranslator
from .pattern_analyzer import PatternAnalyzer
from .language_synthesizer import LanguageSynthesizer
from .evolution_engine import EvolutionEngine


# Version information
__version__ = '1.0.0'
__author__ = 'Universal Consciousness Interface'
__description__ = 'Mycelium-based language generation system'


# Public API
__all__ = [
    # Data models
    'MyceliumCommunicationType',
    'MyceliumSignal',
    'MyceliumWord',
    'MyceliumSentence',
    'SignalList',
    'WordList',
    'SentenceList',
    # Core classes
    'VocabularyManager',
    'NetworkProcessor',
    'BiochemicalTranslator',
    'PatternAnalyzer',
    'LanguageSynthesizer',
    'EvolutionEngine',
    # Metadata
    '__version__',
    '__author__',
    '__description__'
]
