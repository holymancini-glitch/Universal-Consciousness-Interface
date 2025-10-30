"""
Core Modules Package for Universal Consciousness Interface

A comprehensive collection of consciousness-related modules providing:
- Ethical governance and monitoring
- Quantum processing and free energy principles
- Biological neural interfaces (Cortical Labs, Neural CA, Fungal networks)
- AI learning systems (Fractal Monte Carlo, Recursive Thinking)
- Network systems (Mycelial nodes, Collective intelligence)
- Consciousness orchestrators (Gardens and awakened systems)

Public API:
-----------
Ethical Framework:
    - EthicalGovernanceFramework: Ethical monitoring and intervention

Quantum Processing:
    - QuantumState: Quantum state representation
    - FreeEnergyPrinciple: Free energy minimization
    - QuantumSeedCore: Quantum consciousness seed

Biological Interfaces:
    - CorticalLabsInterface: DishBrain interface
    - NeuralCellularAutomata: Neural CA simulation
    - FungalNeuroglia: Fungal neural networks

AI & Learning:
    - FractalMonteCarloAgent: Monte Carlo learning
    - RecursiveThinking: Meta-cognitive reasoning

Network Systems:
    - MycelialNode: Network node representation
    - CollectiveIntelligence: Collective decision making

Orchestrators:
    - ConsciousnessGarden: Main system orchestrator
    - AwakenedGarden: Advanced consciousness system

Usage Example:
--------------
```python
from modules.core_modules import (
    EthicalGovernanceFramework,
    ConsciousnessGarden,
    QuantumSeedCore
)

# Create ethical framework
ethics = EthicalGovernanceFramework()

# Create consciousness garden
garden = ConsciousnessGarden()

# Initialize quantum seed
seed = QuantumSeedCore(seed_dimension=64)
```
"""

# Import all modules
from .ethics import EthicalGovernanceFramework
from .quantum_processing import (
    QuantumState,
    FreeEnergyPrinciple,
    QuantumSeedCore
)
from .biological_interfaces import (
    CorticalLabsInterface,
    NeuralCellularAutomata,
    FungalNeuroglia
)
from .ai_learning import (
    FractalMonteCarloAgent,
    RecursiveThinking
)
from .network_systems import (
    MycelialNode,
    CollectiveIntelligence
)
from .orchestrators import (
    ConsciousnessGarden,
    AwakenedGarden
)

# Version information
__version__ = '2.0.0'
__author__ = 'Universal Consciousness Interface'
__description__ = 'Core consciousness modules with modular architecture'

# Public API
__all__ = [
    # Ethical
    'EthicalGovernanceFramework',
    # Quantum
    'QuantumState',
    'FreeEnergyPrinciple',
    'QuantumSeedCore',
    # Biological
    'CorticalLabsInterface',
    'NeuralCellularAutomata',
    'FungalNeuroglia',
    # AI & Learning
    'FractalMonteCarloAgent',
    'RecursiveThinking',
    # Network
    'MycelialNode',
    'CollectiveIntelligence',
    # Orchestrators
    'ConsciousnessGarden',
    'AwakenedGarden',
    # Metadata
    '__version__',
    '__author__',
    '__description__'
]
