"""
Data Models for Mycelium Language Generator

This module contains all data structures used throughout the mycelium language
generation system, including enums, dataclasses, and type aliases.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Tuple


class MyceliumCommunicationType(Enum):
    """
    Types of mycelium communication signals.

    Mycelium networks communicate through multiple modalities including
    chemical gradients, electrical pulses, nutrient flow patterns, and
    network-level resonance effects.
    """
    CHEMICAL_GRADIENT = "chemical_gradient"
    ELECTRICAL_PULSE = "electrical_pulse"
    NUTRIENT_FLOW = "nutrient_flow"
    HYPHAL_GROWTH = "hyphal_growth"
    SPORE_RELEASE = "spore_release"
    ENZYMATIC_SIGNAL = "enzymatic_signal"
    NETWORK_RESONANCE = "network_resonance"


@dataclass
class MyceliumSignal:
    """
    Individual mycelium communication signal.

    Represents a single communication event in the mycelium network,
    capturing both the physical characteristics (intensity, duration,
    spatial pattern) and the chemical/electrical signatures.

    Attributes:
        signal_type: The type of communication signal
        intensity: Signal strength (0.0-1.0)
        duration: How long the signal lasts (seconds)
        spatial_pattern: Description of spatial distribution
        chemical_composition: Dictionary of chemical compounds and concentrations
        electrical_frequency: Electrical oscillation frequency (Hz)
        timestamp: When the signal was generated
        network_location: 3D coordinates in the network (x, y, z)
    """
    signal_type: MyceliumCommunicationType
    intensity: float
    duration: float
    spatial_pattern: str
    chemical_composition: Dict[str, float]
    electrical_frequency: float
    timestamp: datetime
    network_location: Tuple[float, float, float]  # 3D coordinates


@dataclass
class MyceliumWord:
    """
    A word in the mycelium language.

    Words are generated from patterns of mycelium signals, combining
    phonetic, chemical, and electrical signatures to create meaning.

    Attributes:
        phonetic_pattern: The sound pattern of the word
        chemical_signature: Dictionary of chemical compounds that define this word
        electrical_signature: Electrical frequency associated with the word
        meaning_concept: The semantic concept this word represents
        context_cluster: The semantic cluster this word belongs to
        formation_signals: The original signals that formed this word
    """
    phonetic_pattern: str
    chemical_signature: Dict[str, float]
    electrical_signature: float
    meaning_concept: str
    context_cluster: str
    formation_signals: List[MyceliumSignal]


@dataclass
class MyceliumSentence:
    """
    A sentence in the mycelium language.

    Sentences are structured sequences of mycelium words that follow
    network topology patterns and express complex semantic relationships.

    Attributes:
        words: List of MyceliumWord objects in the sentence
        syntactic_structure: The grammatical structure pattern
        semantic_flow: Dictionary describing the flow of meaning
        network_topology: The network pattern that shaped the structure
        temporal_pattern: The timing pattern of communication
        consciousness_level: The consciousness level expressed in the sentence
    """
    words: List[MyceliumWord]
    syntactic_structure: str
    semantic_flow: Dict[str, Any]
    network_topology: str
    temporal_pattern: str
    consciousness_level: str


# Type aliases for convenience
SignalList = List[MyceliumSignal]
WordList = List[MyceliumWord]
SentenceList = List[MyceliumSentence]


__all__ = [
    'MyceliumCommunicationType',
    'MyceliumSignal',
    'MyceliumWord',
    'MyceliumSentence',
    'SignalList',
    'WordList',
    'SentenceList'
]
