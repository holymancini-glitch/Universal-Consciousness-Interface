"""
Data Models for Integrated Consciousness System

This module contains all data structures used throughout the integrated
consciousness system, including enums and dataclasses for metrics.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ConsciousnessIntegrationLevel(Enum):
    """
    Integration levels for consciousness processing.

    Each level represents a different stage of consciousness integration,
    from basic to universal, incorporating increasingly complex processing
    and coordination mechanisms.

    Levels:
        BASIC_INTEGRATION: Simple consciousness processing
        FRACTAL_INTEGRATION: Fractal pattern integration
        QUANTUM_INTEGRATION: Quantum-enhanced processing
        RADIOTROPHIC_INTEGRATION: Radiation-based energy processing
        UNIVERSAL_INTEGRATION: Complete universal consciousness
    """
    BASIC_INTEGRATION = 1
    FRACTAL_INTEGRATION = 2
    QUANTUM_INTEGRATION = 3
    RADIOTROPHIC_INTEGRATION = 4
    UNIVERSAL_INTEGRATION = 5


@dataclass
class IntegratedConsciousnessMetrics:
    """
    Comprehensive metrics for the integrated consciousness system.

    Captures all key performance indicators and emergent properties
    across the various consciousness processing components.

    Attributes:
        timestamp: When the metrics were captured
        integration_level: Current consciousness integration level
        fractal_coherence: Coherence score across fractal patterns (0.0-1.0)
        mycelial_connectivity: Network connectivity measure (0.0-1.0)
        quantum_entanglement: Quantum correlation measure (0.0-1.0)
        radiotrophic_efficiency: Energy processing efficiency (0.0-1.0)
        universal_harmony: Overall system harmony index (0.0-1.0)
        total_processing_nodes: Number of active processing nodes
        active_consciousness_streams: Number of active consciousness streams
        emergent_patterns_detected: Count of detected emergent patterns
    """
    timestamp: datetime
    integration_level: ConsciousnessIntegrationLevel
    fractal_coherence: float
    mycelial_connectivity: float
    quantum_entanglement: float
    radiotrophic_efficiency: float
    universal_harmony: float
    total_processing_nodes: int
    active_consciousness_streams: int
    emergent_patterns_detected: int


__all__ = [
    'ConsciousnessIntegrationLevel',
    'IntegratedConsciousnessMetrics'
]
