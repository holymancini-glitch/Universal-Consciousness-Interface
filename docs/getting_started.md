# Getting Started

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/universal-consciousness-interface/universal-consciousness-interface.git
   cd universal-consciousness-interface
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run a simple test:
   ```bash
   python demos/demo_consciousness_system.py
   ```

## Basic Usage

### Running a Demo

The easiest way to get started is to run one of the provided demos:

```bash
# Run the comprehensive consciousness demo
python demos/demo_consciousness_system.py

# Run the radiotrophic consciousness demo
python demos/radiotrophic_consciousness_demo.py

# Run the mycelium language revolution demo
python demos/mycelium_language_revolution_demo.py
```

### Creating a Simple Consciousness System

```python
from core.universal_consciousness_orchestrator import UniversalConsciousnessOrchestrator

# Create system with default configuration
uci = UniversalConsciousnessOrchestrator()

# Run consciousness simulation
results = uci.run_consciousness_simulation(duration_seconds=30)
```

### Advanced Configuration

```python
config = {
    'quantum_enabled': True,
    'plant_interface_enabled': True,
    'psychoactive_enabled': False,  # Requires special permissions
    'ecosystem_enabled': True,
    'safety_mode': 'STRICT'
}

uci = UniversalConsciousnessOrchestrator(**config)
```

## System Requirements

### Core Dependencies

- **numpy>=1.21.0** - Mathematical operations and signal processing
- **torch>=1.11.0** - Neural network components and bio-digital fusion
- **networkx>=2.6.0** - Graph-based mycelial network topology
- **scipy>=1.7.0** - Advanced signal processing (plant electromagnetic communication)
- **asyncio** - Asynchronous consciousness processing and multi-modal integration

### Optional Dependencies

- **matplotlib>=3.5.0** - Visualization of consciousness states and patterns
- **pandas>=1.3.0** - Data analysis for consciousness metrics
- **jupyter>=1.0.0** - Interactive consciousness research notebooks
- **plotly>=5.3.0** - Advanced visualization for consciousness evolution

## Development Setup

### Installing Development Dependencies

```bash
pip install -r requirements-dev.txt
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suite
python -m pytest tests/test_consciousness_modules.py -v
```

### Code Quality

```bash
# Format code with black
black .

# Check code style with flake8
flake8 .

# Type checking with mypy
mypy core/
```

## Next Steps

1. Explore the [API Reference](api_reference.md) to understand the available interfaces
2. Review the [Architecture Documentation](architecture.md) to understand the system design
3. Check the [Safety Guidelines](safety_guidelines.md) to understand safety protocols
4. Try the [Integration Guide](integration_guide.md) to learn how to integrate with external systems