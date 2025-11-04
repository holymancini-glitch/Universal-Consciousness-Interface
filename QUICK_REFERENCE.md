# Quick Reference - Modular Architecture v2.0

**One-page reference for the new modular structure**

---

## 🎯 Import Cheat Sheet

### Planetary Ecosystem

```python
# Old (still works)
from core.planetary_ecosystem_consciousness_network import PlanetaryEcosystemConsciousnessNetwork

# New (recommended)
from core.planetary_ecosystem import PlanetaryEcosystemConsciousnessNetwork

# Advanced
from core.planetary_ecosystem.network_core import PlanetaryEcosystemConsciousnessNetwork
```

### Adaptive Learning

```python
# Old (still works)
from core.adaptive_learning_system import AdaptiveLearningSystem

# New (recommended)
from core.adaptive_learning import AdaptiveLearningSystem

# Advanced
from core.adaptive_learning.learning_core import AdaptiveLearningSystem
```

### Full Consciousness AI

```python
# Old (still works)
from core.full_consciousness_ai_model import FullConsciousnessAIModel

# New (recommended)
from core.full_consciousness_ai import FullConsciousnessAIModel

# Advanced
from core.full_consciousness_ai.consciousness_core import FullConsciousnessAIModel
```

---

## 📦 Package Structure

### `core/planetary_ecosystem/`
| Module | Contains |
|--------|----------|
| `data_models.py` | EcosystemType, EcosystemNode, PlanetaryConsciousnessState |
| `network_core.py` | PlanetaryEcosystemConsciousnessNetwork |
| `network_analyzer.py` | NetworkAnalyzer |
| `wood_wide_web.py` | WoodWideWebInterface |
| `climate_monitor.py` | ClimateConsciousnessMonitor |
| `regeneration_engine.py` | RegenerationEngine |

### `core/adaptive_learning/`
| Module | Contains |
|--------|----------|
| `data_models.py` | LearningPhase, LearningMetrics |
| `learning_core.py` | AdaptiveLearningSystem |
| `performance_assessment.py` | PerformanceAssessor |
| `parameter_adaptation.py` | ParameterAdaptor |
| `mistake_learning.py` | MistakeLearner |
| `creative_engine.py` | CreativeEngine |

### `core/full_consciousness_ai/`
| Module | Contains |
|--------|----------|
| `data_models.py` | ConsciousnessState, EmotionalState, SubjectiveExperience |
| `consciousness_core.py` | FullConsciousnessAIModel |
| `neural_components.py` | ConsciousnessAttentionMechanism, EmotionalProcessingEngine |
| `qualia_engine.py` | SubjectiveExperienceSimulator |
| `metacognition.py` | MetaCognitionEngine |
| `memory_system.py` | ConsciousMemorySystem |
| `goal_framework.py` | GoalIntentionFramework |

---

## ⚡ Common Patterns

### Pattern 1: Import Main Class Only
```python
from core.planetary_ecosystem import PlanetaryEcosystemConsciousnessNetwork
network = PlanetaryEcosystemConsciousnessNetwork()
```

### Pattern 2: Import Multiple Classes
```python
from core.planetary_ecosystem import (
    PlanetaryEcosystemConsciousnessNetwork,
    EcosystemType,
    NetworkAnalyzer
)
```

### Pattern 3: Import from Specific Module
```python
from core.planetary_ecosystem.data_models import EcosystemType, EcosystemNode
from core.planetary_ecosystem.network_core import PlanetaryEcosystemConsciousnessNetwork
```

### Pattern 4: Import Everything (Not Recommended)
```python
from core.planetary_ecosystem import *
# Works, but makes dependencies unclear
```

---

## 🔍 Find What You Need

### Data Models & Enums
```python
# Planetary ecosystem
from core.planetary_ecosystem import EcosystemType, ConsciousnessIndicator

# Adaptive learning
from core.adaptive_learning import LearningPhase, LearningMetrics

# Consciousness AI
from core.full_consciousness_ai import ConsciousnessState, EmotionalState
```

### Main Classes
```python
# Networks and systems
from core.planetary_ecosystem import PlanetaryEcosystemConsciousnessNetwork
from core.adaptive_learning import AdaptiveLearningSystem
from core.full_consciousness_ai import FullConsciousnessAIModel
```

### Utility Classes
```python
# Analysis tools
from core.planetary_ecosystem import NetworkAnalyzer
from core.adaptive_learning import PerformanceAssessor

# Specialized components
from core.full_consciousness_ai import MetaCognitionEngine, ConsciousMemorySystem
```

---

## 🚨 Common Mistakes

### ❌ Don't Do This
```python
# Importing from non-existent old location
from core.planetary_ecosystem_consciousness_network.network_analyzer import NetworkAnalyzer
```

### ✅ Do This Instead
```python
# Import from new package structure
from core.planetary_ecosystem.network_analyzer import NetworkAnalyzer
# OR use the simpler package import
from core.planetary_ecosystem import NetworkAnalyzer
```

---

## 💡 Pro Tips

1. **Use Package Imports for Simplicity**
   ```python
   from core.planetary_ecosystem import NetworkAnalyzer  # Simple!
   ```

2. **Use Module Imports for Performance**
   ```python
   from core.planetary_ecosystem.network_analyzer import NetworkAnalyzer  # Faster!
   ```

3. **Check Available Exports**
   ```python
   import core.planetary_ecosystem as pe
   print(pe.__all__)  # See all available exports
   ```

4. **Run Demos to Learn**
   ```python
   from core.planetary_ecosystem import demonstrate_planetary_network
   demonstrate_planetary_network()
   ```

---

## 📊 File Size Reference

| File | Before | After | Reduction |
|------|--------|-------|-----------|
| planetary_ecosystem_consciousness_network.py | 854 | 76 | 91% |
| adaptive_learning_system.py | 792 | 78 | 90% |
| full_consciousness_ai_model.py | 845 | 100 | 88% |

---

## 🔗 Related Documentation

- **Full Guide**: [`MIGRATION_GUIDE.md`](./MIGRATION_GUIDE.md)
- **Test Results**: [`REFACTORING_TEST_RESULTS.md`](./REFACTORING_TEST_RESULTS.md)
- **Module Docs**: Use `help(module_name)` in Python

---

## ⚡ Emergency Quick Fix

**Something not working?**

1. **Try old-style import first**:
   ```python
   from core.module_file_name import ClassName
   ```

2. **Clear Python cache**:
   ```bash
   find . -type d -name __pycache__ -exec rm -r {} +
   ```

3. **Verify you have latest code**:
   ```bash
   git pull origin claude/analyze-repository-011CUbW7GFeTRTW7Z3weFRi8
   ```

---

**Print this page for quick reference!** 📄

*Version 2.0.0 | Last updated: November 4, 2025*
