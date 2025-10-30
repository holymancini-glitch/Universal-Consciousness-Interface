"""
Planetary Ecosystem Consciousness Network Package

A comprehensive system for connecting to Earth's ecosystem awareness
through the "Wood Wide Web" and monitoring planetary consciousness.

Public API:
-----------
Data Models:
    - EcosystemType: Enum of ecosystem types
    - ConsciousnessIndicator: Enum of consciousness indicators
    - EcosystemNode: Node representation in the network
    - PlanetaryConsciousnessState: Overall planetary state

Core Components:
    - PlanetaryEcosystemConsciousnessNetwork: Main network coordinator
    - NetworkAnalyzer: Collective intelligence analyzer
    - WoodWideWebInterface: Forest communication interface
    - ClimateConsciousnessMonitor: Climate stability monitor
    - RegenerationEngine: Ecosystem restoration engine

Demo:
    - demonstrate_planetary_network: Full demonstration

Usage Example:
--------------
```python
from core.planetary_ecosystem import (
    PlanetaryEcosystemConsciousnessNetwork,
    EcosystemNode,
    EcosystemType
)
from datetime import datetime

# Create network
network = PlanetaryEcosystemConsciousnessNetwork()

# Register ecosystem node
node = EcosystemNode(
    id="forest_001",
    ecosystem_type=EcosystemType.FOREST,
    location=(45.0, -120.0),
    consciousness_level=0.8,
    health_status=0.9,
    connectivity_score=0.85,
    data_sources=["sensors"],
    last_updated=datetime.now(),
    biodiversity_index=0.92,
    communication_signals={}
)
network.register_ecosystem_node(node)

# Assess planetary consciousness
state = network.assess_planetary_consciousness()
print(f"Global Awareness: {state.global_awareness:.3f}")

# Connect to Wood Wide Web
connection = network.connect_to_wood_wide_web()
```

For modular usage:
```python
from core.planetary_ecosystem import (
    NetworkAnalyzer,
    WoodWideWebInterface,
    RegenerationEngine
)

# Use components independently
analyzer = NetworkAnalyzer()
wood_web = WoodWideWebInterface()
regeneration = RegenerationEngine()
```
"""

# Import all data models
from .data_models import (
    EcosystemType,
    ConsciousnessIndicator,
    EcosystemNode,
    PlanetaryConsciousnessState
)

# Import all core components
from .network_core import PlanetaryEcosystemConsciousnessNetwork
from .network_analyzer import NetworkAnalyzer
from .wood_wide_web import WoodWideWebInterface
from .climate_monitor import ClimateConsciousnessMonitor
from .regeneration_engine import RegenerationEngine

# Import demo
from .demo import demonstrate_planetary_network

# Version information
__version__ = '2.0.0'
__author__ = 'Universal Consciousness Interface'
__description__ = 'Planetary ecosystem consciousness network with modular architecture'

# Public API
__all__ = [
    # Data models
    'EcosystemType',
    'ConsciousnessIndicator',
    'EcosystemNode',
    'PlanetaryConsciousnessState',
    # Core components
    'PlanetaryEcosystemConsciousnessNetwork',
    'NetworkAnalyzer',
    'WoodWideWebInterface',
    'ClimateConsciousnessMonitor',
    'RegenerationEngine',
    # Demo
    'demonstrate_planetary_network',
    # Metadata
    '__version__',
    '__author__',
    '__description__'
]
