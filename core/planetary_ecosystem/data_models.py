"""
Data Models for Planetary Ecosystem Consciousness Network

Contains all enums, dataclasses, and data structures used throughout
the planetary ecosystem consciousness system.
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class EcosystemType(Enum):
    """Types of ecosystems in the planetary network"""
    FOREST = "forest"
    OCEAN = "ocean"
    DESERT = "desert"
    GRASSLAND = "grassland"
    WETLAND = "wetland"
    TUNDRA = "tundra"
    MOUNTAIN = "mountain"
    URBAN = "urban"
    CORAL_REEF = "coral_reef"
    RAINFOREST = "rainforest"


class ConsciousnessIndicator(Enum):
    """Indicators of ecosystem consciousness"""
    BIOLOGICAL_DENSITY = "biological_density"
    NETWORK_CONNECTIVITY = "network_connectivity"
    CHEMICAL_COMMUNICATION = "chemical_communication"
    ELECTRICAL_ACTIVITY = "electrical_activity"
    THERMAL_REGULATION = "thermal_regulation"
    WATER_CYCLING = "water_cycling"
    NUTRIENT_FLOW = "nutrient_flow"
    GENETIC_DIVERSITY = "genetic_diversity"
    SYMBIOTIC_RELATIONSHIPS = "symbiotic_relationships"
    COLLECTIVE_BEHAVIOR = "collective_behavior"


@dataclass
class EcosystemNode:
    """Represents a node in the planetary ecosystem consciousness network"""
    id: str
    ecosystem_type: EcosystemType
    location: Tuple[float, float]  # latitude, longitude
    consciousness_level: float
    health_status: float
    connectivity_score: float
    data_sources: List[str]
    last_updated: datetime
    biodiversity_index: float
    communication_signals: Dict[str, Any]


@dataclass
class PlanetaryConsciousnessState:
    """Represents the overall planetary consciousness state"""
    global_awareness: float
    ecosystem_distribution: Dict[EcosystemType, float]
    consciousness_hotspots: List[Dict[str, Any]]
    environmental_stress_indicators: Dict[str, float]
    collective_intelligence: float
    network_coherence: float
    timestamp: datetime
    planetary_health: float
    climate_stability: float
    regenerative_capacity: float


__all__ = [
    'EcosystemType',
    'ConsciousnessIndicator',
    'EcosystemNode',
    'PlanetaryConsciousnessState'
]
