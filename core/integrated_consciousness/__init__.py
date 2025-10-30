"""
Integrated Consciousness Package

A modular system for integrated consciousness processing combining fractal AI,
mycelial networks, attention mechanisms, self-modeling, and cohesion analysis.

Public API:
-----------
Data Models:
    - ConsciousnessIntegrationLevel: Enum of integration levels
    - IntegratedConsciousnessMetrics: Comprehensive system metrics

Core Classes:
    - LatentSpace: GPU-accelerated vector management
    - EnhancedMycelialEngine: Experience network with graph algorithms
    - AttentionField: Dynamic focus and resonance detection
    - EnhancedFractalAI: Neural network prediction
    - EnhancedFeedbackLoop: Adaptive learning system
    - SelfModel: Identity and metacognition modeling
    - CohesionLayer: System harmony and crystallization analysis

Testing Utilities:
    - optimize_system_performance: Configuration optimization
    - run_comprehensive_test: Full system testing

Usage Example:
--------------
```python
from core.integrated_consciousness import (
    IntegratedConsciousnessSystem,
    ConsciousnessIntegrationLevel
)

# Create system
system = IntegratedConsciousnessSystem(
    dimensions=256,
    max_nodes=2000,
    integration_level=ConsciousnessIntegrationLevel.FRACTAL_INTEGRATION
)

# Process consciousness cycle
metrics = await system.process_consciousness_cycle({
    'sensory_input': 0.5,
    'cognitive_load': 0.6
})

# Run simulation
results = await system.run_consciousness_simulation(duration_cycles=100)
```
"""

# Import all data models
from .data_models import (
    ConsciousnessIntegrationLevel,
    IntegratedConsciousnessMetrics
)

# Import all core classes
from .latent_space import LatentSpace
from .mycelial_engine import EnhancedMycelialEngine
from .attention_field import AttentionField
from .fractal_ai import EnhancedFractalAI, EnhancedFeedbackLoop
from .self_model import SelfModel, CohesionLayer

# Import testing utilities
from .testing import optimize_system_performance, run_comprehensive_test


# Version information
__version__ = '1.0.0'
__author__ = 'Universal Consciousness Interface'
__description__ = 'Integrated consciousness processing system'


# Public API
__all__ = [
    # Data models
    'ConsciousnessIntegrationLevel',
    'IntegratedConsciousnessMetrics',
    # Core classes
    'LatentSpace',
    'EnhancedMycelialEngine',
    'AttentionField',
    'EnhancedFractalAI',
    'EnhancedFeedbackLoop',
    'SelfModel',
    'CohesionLayer',
    # Testing utilities
    'optimize_system_performance',
    'run_comprehensive_test',
    # Metadata
    '__version__',
    '__author__',
    '__description__'
]
