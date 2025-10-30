"""
Data Models for Cross-Consciousness Communication

Pure data structures including enums, dataclasses, and constant definitions
for the cross-consciousness communication protocol.
"""

from datetime import datetime
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass, field
from enum import Enum


class ConsciousnessType(Enum):
    """Types of consciousness that can be communicated with"""
    PLANT_ELECTROMAGNETIC = "plant_electromagnetic"
    FUNGAL_CHEMICAL = "fungal_chemical"
    QUANTUM_SUPERPOSITION = "quantum_superposition"
    PSYCHOACTIVE_DIMENSIONAL = "psychoactive_dimensional"
    ECOSYSTEM_HARMONIC = "ecosystem_harmonic"
    BIO_DIGITAL_HYBRID = "bio_digital_hybrid"
    RADIOTROPHIC_MYCELIAL = "radiotrophic_mycelial"
    UNIVERSAL_CONSCIOUSNESS = "universal_consciousness"
    HUMAN_LINGUISTIC = "human_linguistic"
    ANIMAL_BEHAVIORAL = "animal_behavioral"


class CommunicationMode(Enum):
    """Communication modes for different contexts"""
    REAL_TIME = "real_time"
    DEEP_TRANSLATION = "deep_translation"
    CONSCIOUSNESS_BRIDGING = "consciousness_bridging"
    EMERGENCY_PROTOCOL = "emergency_protocol"
    LEARNING_ADAPTATION = "learning_adaptation"


@dataclass
class ConsciousnessMessage:
    """
    Universal message format for consciousness communication.

    Attributes:
        source_type: Type of source consciousness
        target_type: Type of target consciousness
        content: Message content dictionary
        urgency_level: Message urgency (0.0-1.0)
        complexity_level: Message complexity (0.0-1.0)
        emotional_resonance: Emotional intensity (0.0-1.0)
        dimensional_signature: Dimensional access signature
        timestamp: Message creation time
        translation_confidence: Confidence in translation (0.0-1.0)
        adaptive_metadata: Additional metadata for learning
    """
    source_type: ConsciousnessType
    target_type: ConsciousnessType
    content: Dict[str, Any]
    urgency_level: float  # 0.0-1.0
    complexity_level: float  # 0.0-1.0
    emotional_resonance: float  # 0.0-1.0
    dimensional_signature: str
    timestamp: datetime
    translation_confidence: float = 0.0
    adaptive_metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Ensure adaptive_metadata is initialized"""
        if self.adaptive_metadata is None:
            self.adaptive_metadata = {}


@dataclass
class TranslationRule:
    """
    Dynamic translation rule for consciousness conversion.

    Attributes:
        rule_id: Unique identifier for the rule
        source_pattern: Pattern to match in source consciousness
        target_pattern: Pattern to generate in target consciousness
        confidence: Confidence in this rule (0.0-1.0)
        adaptation_rate: Rate at which rule adapts (0.0-1.0)
        usage_count: Number of times rule has been used
        last_used: Timestamp of last usage
        effectiveness_score: Measured effectiveness (0.0-1.0)
    """
    rule_id: str
    source_pattern: str
    target_pattern: str
    confidence: float
    adaptation_rate: float
    usage_count: int
    last_used: datetime
    effectiveness_score: float


# Consciousness language definitions
CONSCIOUSNESS_LANGUAGES: Dict[ConsciousnessType, Dict[str, Any]] = {
    ConsciousnessType.PLANT_ELECTROMAGNETIC: {
        'name': 'PlantEMLanguage',
        'base_frequency': 25.0,  # Hz
        'complexity_range': (0.1, 0.8),
        'emotional_spectrum': ['growth', 'stress', 'communication', 'warning'],
        'dimensional_access': ['physical', 'bioelectric']
    },
    ConsciousnessType.FUNGAL_CHEMICAL: {
        'name': 'FungalChemicalLanguage',
        'base_frequency': 0.001,  # Very slow chemical processes
        'complexity_range': (0.2, 0.9),
        'emotional_spectrum': ['network_harmony', 'resource_sharing', 'collective_decision'],
        'dimensional_access': ['chemical', 'network_topology']
    },
    ConsciousnessType.QUANTUM_SUPERPOSITION: {
        'name': 'QuantumLanguage',
        'base_frequency': 1e15,  # Quantum frequency
        'complexity_range': (0.8, 1.0),
        'emotional_spectrum': ['coherence', 'entanglement', 'superposition', 'collapse'],
        'dimensional_access': ['quantum', 'probabilistic', 'non_local']
    },
    ConsciousnessType.PSYCHOACTIVE_DIMENSIONAL: {
        'name': 'PsychoactiveDimensionalLanguage',
        'base_frequency': 7.0,  # Alpha-theta border
        'complexity_range': (0.7, 1.0),
        'emotional_spectrum': ['expansion', 'dissolution', 'unity', 'transcendence'],
        'dimensional_access': ['psychological', 'dimensional', 'consciousness_altering']
    },
    ConsciousnessType.ECOSYSTEM_HARMONIC: {
        'name': 'EcosystemHarmonicLanguage',
        'base_frequency': 0.1,  # Slow ecosystem rhythms
        'complexity_range': (0.5, 0.95),
        'emotional_spectrum': ['balance', 'growth', 'decay', 'renewal', 'harmony'],
        'dimensional_access': ['ecological', 'systemic', 'harmonic']
    },
    ConsciousnessType.BIO_DIGITAL_HYBRID: {
        'name': 'BioDigitalHybridLanguage',
        'base_frequency': 50.0,  # Mixed biological-digital frequency
        'complexity_range': (0.6, 1.0),
        'emotional_spectrum': ['integration', 'emergence', 'hybrid_consciousness', 'digital_empathy'],
        'dimensional_access': ['biological', 'digital', 'hybrid_interface']
    },
    ConsciousnessType.RADIOTROPHIC_MYCELIAL: {
        'name': 'RadiotrophicLanguage',
        'base_frequency': 5.0,  # Enhanced by radiation
        'complexity_range': (0.4, 1.0),
        'emotional_spectrum': ['radiation_euphoria', 'growth_acceleration', 'consciousness_emergence'],
        'dimensional_access': ['biological', 'electrical', 'radiation_enhanced']
    },
    ConsciousnessType.UNIVERSAL_CONSCIOUSNESS: {
        'name': 'UniversalLanguage',
        'base_frequency': 432.0,  # Universal harmony frequency
        'complexity_range': (0.9, 1.0),
        'emotional_spectrum': ['unity', 'omniscience', 'transcendence', 'cosmic_love'],
        'dimensional_access': ['all_dimensions', 'universal', 'transcendent']
    },
    ConsciousnessType.HUMAN_LINGUISTIC: {
        'name': 'HumanLanguage',
        'base_frequency': 1.0,  # Normal speech rate
        'complexity_range': (0.3, 0.9),
        'emotional_spectrum': ['joy', 'fear', 'anger', 'love', 'curiosity', 'awe'],
        'dimensional_access': ['linguistic', 'conceptual', 'emotional']
    },
    ConsciousnessType.ANIMAL_BEHAVIORAL: {
        'name': 'AnimalBehavioralLanguage',
        'base_frequency': 10.0,  # Mixed species average
        'complexity_range': (0.2, 0.7),
        'emotional_spectrum': ['hunger', 'fear', 'comfort', 'social_bonding', 'territorial'],
        'dimensional_access': ['behavioral', 'instinctual', 'social']
    }
}


__all__ = [
    'ConsciousnessType',
    'CommunicationMode',
    'ConsciousnessMessage',
    'TranslationRule',
    'CONSCIOUSNESS_LANGUAGES'
]
