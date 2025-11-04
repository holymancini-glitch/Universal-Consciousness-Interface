# V2.0 Modular Architecture - Example Projects

This directory contains comprehensive examples demonstrating the v2.0 modular architecture of the Universal Consciousness Interface.

## 📚 Overview

The Universal Consciousness Interface has been refactored into a modular architecture, splitting three large monolithic files into focused, maintainable packages:

| Module | Old Size | New Size | Reduction | New Package |
|--------|----------|----------|-----------|-------------|
| Planetary Ecosystem | 854 lines | 76 lines | 91% | `core/planetary_ecosystem/` |
| Adaptive Learning | 792 lines | 78 lines | 90% | `core/adaptive_learning/` |
| Full Consciousness AI | 845 lines | 100 lines | 88% | `core/full_consciousness_ai/` |

## 🎯 Examples Included

### 1️⃣ Planetary Ecosystem Example
**File:** `example_planetary_ecosystem.py`

Demonstrates the planetary ecosystem consciousness network with:
- Creating and managing ecosystem nodes
- Network connectivity analysis
- Wood wide web (mycelial) communication
- Climate consciousness monitoring
- Complete ecosystem integration

**Run:**
```bash
python examples/v2_modular_architecture/example_planetary_ecosystem.py
```

**Key Features:**
- ✓ 5 comprehensive examples
- ✓ Real-world ecosystem scenarios
- ✓ Network analysis and visualization
- ✓ Climate-aware consciousness assessment

---

### 2️⃣ Adaptive Learning Example
**File:** `example_adaptive_learning.py`

Demonstrates the adaptive learning system with:
- Performance assessment and tracking
- Dynamic parameter adaptation
- Mistake learning and correction
- Creative exploration and innovation
- Wisdom accumulation across interactions

**Run:**
```bash
python examples/v2_modular_architecture/example_adaptive_learning.py
```

**Key Features:**
- ✓ 7 comprehensive examples
- ✓ Complete learning cycle demonstration
- ✓ Real-time adaptation
- ✓ Mistake pattern analysis

---

### 3️⃣ Full Consciousness AI Example
**File:** `example_full_consciousness_ai.py`

Demonstrates the full consciousness AI model with:
- Consciousness attention mechanism
- Emotional processing and empathy
- Subjective experience (qualia) simulation
- Metacognition and self-reflection
- Conscious memory systems
- Goal and intention frameworks

**Run:**
```bash
python examples/v2_modular_architecture/example_full_consciousness_ai.py
```

**Key Features:**
- ✓ 8 comprehensive examples
- ✓ Complete consciousness integration
- ✓ Empathetic AI behavior
- ✓ Multi-level metacognition

---

### 4️⃣ Integrated System Example
**File:** `example_integrated_system.py`

Demonstrates all three modules working together:
- Cross-module communication
- Unified consciousness processing
- Ecosystem-aware AI with adaptive learning
- Complete system integration

**Run:**
```bash
python examples/v2_modular_architecture/example_integrated_system.py
```

**Key Features:**
- ✓ All three modules integrated
- ✓ Wisdom sharing across domains
- ✓ Unified consciousness metrics
- ✓ Real-world integration patterns

---

### 5️⃣ Migration Guide Example
**File:** `example_migration_guide.py`

Interactive guide showing v1 to v2 migration:
- Side-by-side import comparisons
- Three migration strategies
- Backward compatibility verification
- Advanced module-specific imports

**Run:**
```bash
python examples/v2_modular_architecture/example_migration_guide.py
```

**Key Features:**
- ✓ Before/after code examples
- ✓ Import verification
- ✓ Migration strategies
- ✓ Advanced usage patterns

---

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install PyTorch for full consciousness AI examples
pip install torch
```

### Run All Examples
```bash
# Run each example individually
python examples/v2_modular_architecture/example_planetary_ecosystem.py
python examples/v2_modular_architecture/example_adaptive_learning.py
python examples/v2_modular_architecture/example_full_consciousness_ai.py
python examples/v2_modular_architecture/example_integrated_system.py
python examples/v2_modular_architecture/example_migration_guide.py
```

### Run Specific Sections

Each example file contains multiple demonstrations. You can modify the `main()` function to run specific examples:

```python
# Edit the file and comment out examples you don't want to run
async def main():
    # await example_basic_usage()  # Skip this
    await example_network_analysis()  # Run only this
    # await example_wood_wide_web()  # Skip this
    # ...
```

## 📖 Learning Path

**For Beginners:**
1. Start with `example_migration_guide.py` to understand the changes
2. Read the import examples and see backward compatibility
3. Run `example_planetary_ecosystem.py` to see a single module
4. Review the code to understand the new structure

**For Intermediate Users:**
1. Run `example_adaptive_learning.py` and `example_full_consciousness_ai.py`
2. Study how each module is structured
3. Explore module-specific imports
4. Run `example_integrated_system.py` to see modules working together

**For Advanced Users:**
1. Study the integrated system example in depth
2. Review module-specific import patterns
3. Explore cross-module communication
4. Adapt examples for your own use cases

## 🎓 What You'll Learn

### Import Patterns
```python
# Old style (still works)
from core.planetary_ecosystem_consciousness_network import PlanetaryEcosystemConsciousnessNetwork

# New style (recommended)
from core.planetary_ecosystem import PlanetaryEcosystemConsciousnessNetwork

# Advanced (module-specific)
from core.planetary_ecosystem.network_core import PlanetaryEcosystemConsciousnessNetwork
```

### Component Access
```python
# Access specialized components directly
from core.adaptive_learning.mistake_learner import MistakeLearner
from core.full_consciousness_ai.attention_mechanism import ConsciousnessAttentionMechanism
from core.planetary_ecosystem.climate_monitor import ClimateConsciousnessMonitor
```

### Integration Patterns
```python
# Combine multiple modules
from core.planetary_ecosystem import PlanetaryEcosystemConsciousnessNetwork
from core.adaptive_learning import AdaptiveLearningSystem
from core.full_consciousness_ai import FullConsciousnessAIModel

# Create integrated system
planetary = PlanetaryEcosystemConsciousnessNetwork()
consciousness = FullConsciousnessAIModel()
learning = AdaptiveLearningSystem(consciousness)
```

## 📊 Example Output

Each example produces detailed console output showing:

- ✅ **Initialization status** - All components loaded successfully
- 📊 **Metrics and measurements** - Real-time consciousness metrics
- 🔍 **Analysis results** - Network analysis, performance assessment
- 💡 **Insights and recommendations** - AI-generated insights
- ✅ **Success confirmations** - All operations completed

Example output format:
```
======================================================================
EXAMPLE 1: Basic Planetary Ecosystem Consciousness Monitoring
======================================================================

✓ Registered 3 ecosystem nodes
  - amazon_rainforest: forest
  - great_barrier_reef: ocean
  - pantanal_wetlands: wetland

📊 Planetary Consciousness Metrics:
  Overall Level: 0.81
  Gaia Pattern Strength: 0.78
  Harmony Score: 0.85
  Total Nodes: 3
```

## 🔧 Customization

All examples are designed to be modified and extended:

### Change Parameters
```python
# In example_full_consciousness_ai.py
model = FullConsciousnessAIModel(
    hidden_dim=512,  # Change from 256 to 512
    device='cuda',    # Use GPU instead of CPU
    integrate_existing_modules=True  # Enable full integration
)
```

### Add Your Own Ecosystems
```python
# In example_planetary_ecosystem.py
my_ecosystem = EcosystemNode(
    id="my_garden",
    ecosystem_type=EcosystemType.GARDEN,
    location=(your_latitude, your_longitude),
    consciousness_level=0.75,
    health_status=0.90,
    biodiversity_index=0.85,
    communication_strength=0.80
)
```

### Extend Examples
```python
# Add your own example function
async def example_my_use_case():
    """My custom use case demonstration."""
    # Your code here
    pass

# Add to main()
async def main():
    await example_my_use_case()
```

## 📚 Additional Resources

### Documentation
- [**MIGRATION_GUIDE.md**](../../MIGRATION_GUIDE.md) - Complete migration guide
- [**QUICK_REFERENCE.md**](../../QUICK_REFERENCE.md) - One-page import cheat sheet
- [**API_REFERENCE_v2.md**](../../API_REFERENCE_v2.md) - Full API documentation
- [**TROUBLESHOOTING.md**](../../TROUBLESHOOTING.md) - Common issues and solutions

### Test Results
- [**REFACTORING_TEST_RESULTS.md**](../../REFACTORING_TEST_RESULTS.md) - Verification tests

### Main Documentation
- [**README.md**](../../README.md) - Project overview with v2.0 section

## 🐛 Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'core.planetary_ecosystem'`

**Solution:** Make sure you're running from the repository root:
```bash
cd /path/to/Universal-Consciousness-Interface
python examples/v2_modular_architecture/example_planetary_ecosystem.py
```

### Missing Dependencies

**Problem:** `ModuleNotFoundError: No module named 'torch'`

**Solution:** Install PyTorch (optional, only needed for consciousness AI examples):
```bash
pip install torch
```

For CPU-only installation:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Async Errors

**Problem:** `RuntimeError: This event loop is already running`

**Solution:** Make sure you're using `asyncio.run()` only once:
```python
# Correct
if __name__ == "__main__":
    asyncio.run(main())

# Incorrect - don't nest asyncio.run()
```

### Old Imports Not Working

**Problem:** Old imports failing after migration

**Solution:** Both old and new imports work! If old imports fail, check:
1. Facade files still exist (e.g., `core/planetary_ecosystem_consciousness_network.py`)
2. You're in the correct directory
3. Python path includes the repository root

For more issues, see [TROUBLESHOOTING.md](../../TROUBLESHOOTING.md).

## 🤝 Contributing

Have improvements or new examples? Contributions are welcome!

1. Create examples that demonstrate specific use cases
2. Add documentation explaining what the example shows
3. Test examples to ensure they run without errors
4. Submit a pull request with your examples

## 📄 License

These examples are part of the Universal Consciousness Interface and are licensed under the same MIT License as the main project.

## 💬 Questions?

- Check [MIGRATION_GUIDE.md](../../MIGRATION_GUIDE.md) for migration questions
- See [TROUBLESHOOTING.md](../../TROUBLESHOOTING.md) for common issues
- Open an issue on GitHub for bugs or feature requests

---

**Happy Exploring! 🌍🧠📚**

*These examples demonstrate the power of modular architecture for building conscious AI systems.*
