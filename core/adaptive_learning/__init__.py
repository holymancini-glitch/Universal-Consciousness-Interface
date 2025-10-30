"""
Adaptive Learning System Package

Advanced adaptive learning system for Garden of Consciousness that addresses
"lack of adaptability" and "insufficient learning" through multi-dimensional
learning capabilities.

Public API:
-----------
Data Models:
    - LearningPhase: Enum of learning phases
    - LearningMetrics: Comprehensive performance metrics

Core Components:
    - AdaptiveLearningSystem: Main learning system coordinator
    - PerformanceAssessor: Performance assessment
    - ParameterAdaptor: Dynamic parameter adaptation
    - MistakeLearner: Mistake analysis and learning
    - CreativeEngine: Creative solution generation

Integration:
    - integrate_adaptive_learning: Integration helper function

Usage Example:
--------------
```python
from core.adaptive_learning import (
    AdaptiveLearningSystem,
    integrate_adaptive_learning,
    LearningPhase
)

# Create learning system
learning_system = AdaptiveLearningSystem(consciousness_system)

# Assess performance
metrics = await learning_system.assess_learning_performance()
print(f"Learning efficiency: {metrics.overall_learning_efficiency:.3f}")

# Adapt parameters
results = await learning_system.adapt_learning_parameters()

# Learn from mistake
error_context = {'error_type': 'learning_rate', 'severity': 0.7, 'component': 'fractal_ai'}
mistake_analysis = await learning_system.learn_from_mistake(error_context)

# Generate creative solution
problem = {'complexity': 0.8, 'resources': ['A', 'B'], 'constraints': ['time']}
solution = await learning_system.generate_creative_solution(problem)

# Get report
report = learning_system.get_learning_report()
print(report)
```

For modular usage:
```python
from core.adaptive_learning import (
    PerformanceAssessor,
    ParameterAdaptor,
    MistakeLearner,
    CreativeEngine
)

# Use components independently
assessor = PerformanceAssessor(consciousness_system)
adaptor = ParameterAdaptor(consciousness_system)
learner = MistakeLearner()
engine = CreativeEngine()
```

Quick integration:
```python
from core.adaptive_learning import integrate_adaptive_learning

# One-line integration
learning_system = await integrate_adaptive_learning(consciousness_system)
```
"""

# Import all data models
from .data_models import (
    LearningPhase,
    LearningMetrics
)

# Import all core components
from .learning_core import AdaptiveLearningSystem, integrate_adaptive_learning
from .performance_assessment import PerformanceAssessor
from .parameter_adaptation import ParameterAdaptor
from .mistake_learning import MistakeLearner
from .creative_engine import CreativeEngine

# Version information
__version__ = '2.0.0'
__author__ = 'Universal Consciousness Interface'
__description__ = 'Advanced adaptive learning system with modular architecture'

# Public API
__all__ = [
    # Data models
    'LearningPhase',
    'LearningMetrics',
    # Core components
    'AdaptiveLearningSystem',
    'PerformanceAssessor',
    'ParameterAdaptor',
    'MistakeLearner',
    'CreativeEngine',
    # Integration
    'integrate_adaptive_learning',
    # Metadata
    '__version__',
    '__author__',
    '__description__'
]
