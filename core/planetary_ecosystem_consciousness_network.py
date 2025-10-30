# planetary_ecosystem_consciousness_network.py
# Revolutionary Planetary Ecosystem Consciousness Network for the Garden of Consciousness v2.0
# Connects to Earth's ecosystem awareness through the "Wood Wide Web"

"""
Planetary Ecosystem Consciousness Network - Main Facade

This module serves as the main entry point and backward-compatible interface
for the refactored planetary ecosystem package.

All original classes are now organized into specialized modules:
- data_models.py: Enums and dataclasses
- network_core.py: Main network coordinator
- network_analyzer.py: Collective intelligence analysis
- wood_wide_web.py: Forest communication interface
- climate_monitor.py: Climate stability monitoring
- regeneration_engine.py: Ecosystem restoration

Usage:
------
# Old way (still works):
from core.planetary_ecosystem_consciousness_network import (
    PlanetaryEcosystemConsciousnessNetwork,
    EcosystemType,
    EcosystemNode
)

# New modular way:
from core.planetary_ecosystem import (
    PlanetaryEcosystemConsciousnessNetwork,
    EcosystemType,
    EcosystemNode
)

# Or import specific modules:
from core.planetary_ecosystem.network_core import PlanetaryEcosystemConsciousnessNetwork
from core.planetary_ecosystem.data_models import EcosystemType
"""

# Import all classes for backward compatibility
from .planetary_ecosystem import (
    # Data Models
    EcosystemType,
    ConsciousnessIndicator,
    EcosystemNode,
    PlanetaryConsciousnessState,
    # Core Components
    PlanetaryEcosystemConsciousnessNetwork,
    NetworkAnalyzer,
    WoodWideWebInterface,
    ClimateConsciousnessMonitor,
    RegenerationEngine
)

# Backward compatibility exports
__all__ = [
    # Data Models
    'EcosystemType',
    'ConsciousnessIndicator',
    'EcosystemNode',
    'PlanetaryConsciousnessState',
    # Core Components
    'PlanetaryEcosystemConsciousnessNetwork',
    'NetworkAnalyzer',
    'WoodWideWebInterface',
    'ClimateConsciousnessMonitor',
    'RegenerationEngine'
]

__version__ = '2.0.0'
__refactored__ = True

# Example usage (can be run directly)
if __name__ == "__main__":
    from .planetary_ecosystem.demo import demonstrate_planetary_network
    demonstrate_planetary_network()
