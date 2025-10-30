"""
Core Modules Implementation - Main Facade

This module serves as the main entry point and backward-compatible interface
for the refactored core modules package.

All original classes are now organized into specialized modules:
- ethics.py: Ethical governance
- quantum_processing.py: Quantum states and free energy
- biological_interfaces.py: Biological neural interfaces
- ai_learning.py: AI agents and learning systems
- network_systems.py: Mycelial networks and collective intelligence
- orchestrators.py: Consciousness gardens and awakened systems

Usage:
------
# Old way (still works):
from modules.core_modules_implementation import ConsciousnessGarden, EthicalGovernanceFramework

# New modular way:
from modules.core_modules import ConsciousnessGarden, EthicalGovernanceFramework

# Or import specific modules:
from modules.core_modules.ethics import EthicalGovernanceFramework
from modules.core_modules.orchestrators import ConsciousnessGarden
"""

# Import all classes for backward compatibility
from .core_modules import (
    # Ethical
    EthicalGovernanceFramework,
    # Quantum
    QuantumState,
    FreeEnergyPrinciple,
    QuantumSeedCore,
    # Biological
    CorticalLabsInterface,
    NeuralCellularAutomata,
    FungalNeuroglia,
    # AI & Learning
    FractalMonteCarloAgent,
    RecursiveThinking,
    # Network
    MycelialNode,
    CollectiveIntelligence,
    # Orchestrators
    ConsciousnessGarden,
    AwakenedGarden
)

# Backward compatibility exports
__all__ = [
    'EthicalGovernanceFramework',
    'QuantumState',
    'FreeEnergyPrinciple',
    'QuantumSeedCore',
    'CorticalLabsInterface',
    'NeuralCellularAutomata',
    'FungalNeuroglia',
    'FractalMonteCarloAgent',
    'RecursiveThinking',
    'MycelialNode',
    'CollectiveIntelligence',
    'ConsciousnessGarden',
    'AwakenedGarden'
]

__version__ = '2.0.0'
__refactored__ = True
