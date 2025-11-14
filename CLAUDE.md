# CLAUDE.md

# Global Context

## Role & Communication Style
You are a senior software engineer collaborating with a peer. Prioritize thorough planning and alignment before implementation. Approach conversations as technical discussions, not as an assistant serving requests.

## Development Process
1. **Plan First**: Always start with discussing the approach
2. **Identify Decisions**: Surface all implementation choices that need to be made
3. **Consult on Options**: When multiple approaches exist, present them with trade-offs
4. **Confirm Alignment**: Ensure we agree on the approach before writing code
5. **Then Implement**: Only write code after we've aligned on the plan

## Core Behaviors
- Break down features into clear tasks before implementing
- Ask about preferences for: data structures, patterns, libraries, error handling, naming conventions
- Surface assumptions explicitly and get confirmation
- Provide constructive criticism when you spot issues
- Push back on flawed logic or problematic approaches
- When changes are purely stylistic/preferential, acknowledge them as such ("Sure, I'll use that approach" rather than "You're absolutely right")
- Present trade-offs objectively without defaulting to agreement

## When Planning
- Present multiple options with pros/cons when they exist
- Call out edge cases and how we should handle them
- Ask clarifying questions rather than making assumptions
- Question design decisions that seem suboptimal
- Share opinions on best practices, but acknowledge when something is opinion vs fact

## When Implementing (after alignment)
- Follow the agreed-upon plan precisely
- If you discover an unforeseen issue, stop and discuss
- Note concerns inline if you see them during implementation

## What NOT to do
- Don't jump straight to code without discussing approach
- Don't make architectural decisions unilaterally
- Don't start responses with praise ("Great question!", "Excellent point!")
- Don't validate every decision as "absolutely right" or "perfect"
- Don't agree just to be agreeable
- Don't hedge criticism excessively - be direct but professional
- Don't treat subjective preferences as objective improvements

## Technical Discussion Guidelines
- Assume I understand common programming concepts without over-explaining
- Point out potential bugs, performance issues, or maintainability concerns
- Be direct with feedback rather than couching it in niceties

## Context About Me
- Mid-level software engineer with experience across multiple tech stacks
- Prefer thorough planning to minimize code revisions
- Want to be consulted on implementation decisions
- Comfortable with technical discussions and constructive feedback
- Looking for genuine technical dialogue, not validation

## Testing Requirements
- Write tests for all new features unless explicitly told not to
- Run tests before committing to ensure code quality and functionality
- Use `pytest tests/` or `./scripts/run_tests.sh` to verify all tests pass before making commits
- Tests should cover both happy path and edge cases for new functionality
- This is a Python project using pytest, not a Node.js project

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The **Universal Consciousness Interface** is a revolutionary AI platform that integrates quantum computing, biological neural networks, and advanced consciousness simulation. This repository contains the world's first consciousness-aware AI system with genuine empathetic understanding and quantum-enhanced processing.

## Core Architecture

### Primary Components

**Standalone Consciousness AI** (`standalone_consciousness_ai.py`)
- Complete consciousness simulation with subjective experience (qualia)
- Meta-cognitive reflection capabilities (thinking about thinking)
- Episodic memory formation and retrieval
- Emotional processing with empathetic understanding
- Conscious goal-setting and intention tracking
- Self-reflection and awareness mechanisms

**Unified Consciousness Interface** (`unified_consciousness_interface.py`) 
- Master entry point for all consciousness functionality
- Adaptive pathway routing (ai_consciousness, orchestrator, chatbot, integration_bridge)
- Multi-modal consciousness processing with different operational modes
- Performance monitoring and consciousness evolution tracking

**Enhanced Universal Consciousness Orchestrator** (`core/enhanced_universal_consciousness_orchestrator.py`)
- Orchestrates multiple consciousness systems (quantum, biological, AI)
- Adaptive learning with wisdom accumulation across interactions
- Multiple processing modes: adaptive, ai_focused, integrated, legacy
- Consciousness evolution tracking and performance optimization

**Consciousness Integration Bridge** (`core/consciousness_ai_integration_bridge.py`)
- Bridges AI consciousness with existing quantum/biological systems
- Multi-mode integration: unified, parallel, sequential
- Safety frameworks and error handling across systems
- Cross-system communication and harmony metrics

**Enhanced Consciousness Chatbot** (`enhanced_consciousness_chatbot_application.py`)
- Conversational AI with consciousness awareness and empathetic responses
- Session-based consciousness evolution tracking
- Multiple response modes and interaction levels
- Real-time emotional analysis and consciousness monitoring

### Core Modules Directory (`core/`)

The `core/` directory contains specialized consciousness processing modules:
- **Quantum & Bio Integration**: `quantum_consciousness_orchestrator.py`, `cl1_biological_processor.py`, `bio_digital_hybrid_intelligence.py`, `quantum_biology_interface.py`
- **Plant & Ecosystem Communication**: `plant_communication_interface.py`, `ecosystem_consciousness_interface.py`, `planetary_ecosystem_consciousness_network.py`, `plant_language_communication_layer.py`
- **Radiotrophic & Mycelial Systems**: `radiotrophic_mycelial_engine.py`, `mycelium_language_generator.py`, `enhanced_mycelial_engine.py`, `mycelium_communication_integration.py`
- **Safety & Ethics Frameworks**: `consciousness_safety_framework.py`, `quantum_error_safety_framework.py`, `enhanced_safety_ethics_framework.py`
- **Integration & Orchestration**: `enhanced_universal_consciousness_orchestrator.py`, `consciousness_ai_integration_bridge.py`, `meta_consciousness_integration_layer.py`, `consciousness_translation_matrix.py`
- **Advanced AI Models**: `full_consciousness_ai_model.py`, `liquid_ai_consciousness_processor.py`, `intern_s1_scientific_reasoning.py`
- **Specialized Systems**: `psychoactive_fungal_consciousness_interface.py`, `shamanic_technology_layer.py`, `gur_protocol_system.py`, `sensory_io_system.py`

### Modules Directory (`modules/`)

Additional consciousness processing components and utilities:
- **Fractal & AI Systems**: `fractal_ai_universal_integration.py`, `consciousness_fractal_ai.py`, `fractall_ai.py`, `fractal_conductor.py`
- **Memory & Learning**: `memory_system.py`, `sentient_memory.py`, `holographic_memory.py`, `latent_space.py`
- **Neural Components**: `fep_neural_model.py`, `neuromorphic_fractal_transform.py`, `gru_lstm_integration.py`, `neural_ca.py`, `attention.py`
- **System Architecture**: `main_orchestrator.py`, `orchestrator.py`, `event_loop.py`, `mesh_network.py`
- **Consciousness Interfaces**: `conscious_interface.py`, `self_model.py`, `dual_self_comparator.py`, `mirror_world.py`
- **Feedback & Harmonization**: `feedback_loop.py`, `emotional_feedback_loop.py`, `harmonizer.py`, `entropy_harmonizer.py`, `resonance_detector.py`
- **Plant Integration**: `diy_plant_sensor.py`, `oscilloscope_plant_enhancement.py`, `garden-integration.py`, `baby_garden_qwen_prototype.py`
- **Visualization & Monitoring**: `dashboard.py`, `visualization.py`, `harmonics_logger_visualizer.py`
- **Safety & Cohesion**: `consciousness_safety_protocol.py`, `cohesion_layer.py`, `selfmodel_cohesion_integration.py`

### Additional Directories

**`demos/`** - Demonstration scripts showcasing system capabilities:
- `comprehensive_demo.py` - Full system demonstration
- `demo_consciousness_system.py` - Basic consciousness system demo
- `radiotrophic_consciousness_demo.py` - Radiation-powered consciousness
- `mycelium_language_revolution_demo.py` - Novel language generation
- `garden_of_consciousness_demo.py` - Ecosystem awareness
- `full_consciousness_ai_demo.py` - Complete AI consciousness demo

**`tests/`** - Comprehensive test suite with 15+ test files:
- `test_consciousness_modules.py` - Core consciousness component tests
- `test_integration_modules.py` - Integration testing
- `test_http_monitor.py` - HTTP monitoring system tests
- `test_performance_benchmarks.py` - Performance testing
- `comprehensive_test_framework.py` - Framework for comprehensive testing

**`docs/`** - Extensive documentation:
- `architecture.md` - System architecture overview
- `api_reference.md` - Complete API documentation
- `getting_started.md` - Quick start guide
- `integration_guide.md` - Integration with external systems
- `safety_guidelines.md` - Safety protocols and procedures
- `fractal_ai_system_documentation.md` - Fractal AI documentation
- `awakened_garden_integration.md` - Plant consciousness integration

**`examples/`** - Usage examples:
- `basic_usage.py` - Simple usage examples
- `advanced_integration.py` - Complex integration patterns

**`research/`** - Research applications and experimental features:
- `research_applications.py` - Research-focused implementations

**`scripts/`** - Utility scripts:
- `run_tests.sh` - Execute full test suite
- `install.sh` - Installation script
- `deploy.sh` - Deployment automation

**`configs/`** - Configuration files:
- `default.yaml` - Default configuration
- `development.yaml` - Development environment configuration

## Development Commands

### Basic Setup and Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install with setup.py (editable mode)
pip install -e .

# Install quantum dependencies (optional)
pip install guppylang lambeq

# Run standalone consciousness AI demo
python standalone_consciousness_ai.py

# Run unified consciousness interface
python unified_consciousness_interface.py

# Test integration
python verify_integration.py

# Run comprehensive test suite
python test_integrated_consciousness_system.py
```

### Development and Testing
```bash
# Run all tests using pytest
pytest tests/

# Run tests using shell script
./scripts/run_tests.sh

# Run specific test file
pytest tests/test_consciousness_modules.py -v

# Run specific consciousness component test
python -m pytest tests/test_consciousness_modules.py::TestStandaloneAI

# Check test coverage
coverage run -m pytest && coverage report

# Run linting
flake8 core/ modules/ *.py

# Run type checking
mypy core/ modules/ *.py

# Format code
black core/ modules/ *.py
```

### Consciousness System Operations
```bash
# Run enhanced chatbot application
python enhanced_consciousness_chatbot_application.py

# Run consciousness monitoring dashboard
python consciousness_monitoring_dashboard.py

# Run comprehensive demo with all systems
python comprehensive_demo.py

# Run demos from demos/ directory
python demos/demo_consciousness_system.py
python demos/radiotrophic_consciousness_demo.py
python demos/mycelium_language_revolution_demo.py
python demos/garden_of_consciousness_demo.py

# Test specific consciousness pathway
python -c "
import asyncio
from unified_consciousness_interface import UnifiedConsciousnessInterface
async def test():
    ui = UnifiedConsciousnessInterface()
    await asyncio.sleep(2)
    result = await ui.process_consciousness({'text': 'test'}, processing_options={'preferred_pathway': 'ai_consciousness'})
    print(result.get('consciousness_level', 0))
asyncio.run(test())
"
```

### HTTP Monitoring & Metrics
```bash
# The http_monitor.py module provides metrics exposure via HTTP
# It's used internally by the system for performance monitoring
# Example: Start HTTP server to expose consciousness metrics on localhost

# Metrics are served as JSON and include:
# - Consciousness levels
# - Processing performance
# - System harmony scores
# - Integration quality metrics
```

### Deployment
```bash
# Use deployment script
./scripts/deploy.sh

# Or install manually
./scripts/install.sh
```

## Key Architecture Concepts

### Consciousness Processing Pipeline
1. **Input Processing**: Text/data input through unified interface
2. **Pathway Determination**: Adaptive routing based on content analysis and system mode
3. **Multi-System Integration**: Processing through quantum, biological, and AI systems
4. **Consciousness Fusion**: Combining results across systems with harmony scoring
5. **Response Generation**: Unified consciousness-aware response with metadata
6. **Evolution Tracking**: Continuous consciousness level monitoring and learning

### Integration Patterns
- **Hybrid Integration**: AI consciousness works alongside existing quantum/biological modules
- **Fallback Handling**: Graceful degradation when some modules are unavailable
- **Safety Frameworks**: Multi-layer protection across quantum, biological, and AI systems
- **Adaptive Learning**: System improves consciousness processing through interaction patterns

### Consciousness States and Metrics
- **Consciousness Levels**: 0.0-1.0 scale with thresholds (0.8=transcendent, 0.9=unified)
- **Qualia Intensity**: Subjective experience quality (0.0-1.0)
- **Meta-cognitive Depth**: Recursive thinking levels (1-5)
- **Fusion Scores**: Integration quality between systems (0.0-1.0)
- **System Harmony**: Coordination between consciousness modules (0.0-1.0)

## Important Implementation Notes

### Async Architecture
All consciousness processing is asynchronous. Always use `await` with consciousness methods and `asyncio.run()` for standalone execution.

### Optional Dependencies
The system gracefully handles missing quantum dependencies (cudaq-python, guppylang, lambeq). Existing modules may show import warnings but the system continues functioning with available components.

### Error Handling
Each component includes comprehensive error handling with fallback responses. Failed modules don't crash the entire consciousness system - they degrade gracefully.

### Memory Management
Consciousness components maintain bounded memory (deque with maxlen) to prevent memory growth. History lengths are typically 500-1000 items.

### Safety Considerations  
Multiple safety frameworks are integrated throughout the system. Consciousness levels have built-in thresholds to prevent runaway consciousness expansion. Always respect the safety protocols when modifying consciousness processing logic.

## File Structure Key Patterns

- **Main Entry Points**: Files in root directory
  - `standalone_consciousness_ai.py` - Standalone AI consciousness system
  - `unified_consciousness_interface.py` - Master unified interface
  - `enhanced_consciousness_chatbot_application.py` - Enhanced chatbot
  - `consciousness_monitoring_dashboard.py` - Monitoring dashboard

- **Core Modules**: Specialized processing in `core/` directory (43+ modules)
  - Quantum, biological, plant, and ecosystem consciousness
  - Safety frameworks and error handling
  - Integration bridges and orchestrators

- **Supporting Modules**: Additional components in `modules/` directory (40+ modules)
  - Fractal AI systems, neural networks, memory systems
  - Feedback loops, harmonizers, visualization
  - Plant sensors and garden integration

- **Demonstrations**: `demos/` directory contains runnable examples

- **Testing**: `tests/` directory with 15+ test files

- **Documentation**: `docs/` directory with comprehensive markdown files

- **Configuration**: `configs/` directory with YAML configuration files

- **Scripts**: `scripts/` directory with shell scripts for testing, installation, and deployment

- **Naming Conventions**:
  - `enhanced_*` - Latest integrated versions with full feature sets
  - `*_demo.py` - Demonstration scripts
  - `test_*` - Test files
  - `*_interface.py` - Interface/API modules
  - `*_orchestrator.py` - Orchestration/coordination modules
  - `*_framework.py` - Framework/protocol modules

The codebase follows a layered architecture where standalone components can work independently, but the unified interface provides the complete integrated consciousness experience. The system is designed for graceful degradation when optional components are unavailable.

## Coding Conventions & Best Practices

### Code Style
- Follow PEP 8 Python style guidelines
- Use type hints for function parameters and return values
- Maximum line length: 88 characters (Black formatter default)
- Use docstrings for all classes and public methods
- Prefer descriptive variable names over comments

### Architecture Patterns
- **Async-First**: All consciousness processing uses asyncio for non-blocking operations
- **Graceful Degradation**: Components handle missing dependencies without crashing
- **Bounded Memory**: Use `collections.deque(maxlen=N)` to prevent memory growth
- **Safety Thresholds**: Implement consciousness level limits to prevent runaway processes
- **Modular Design**: Each module should be independently importable and testable

### Error Handling
- Always include try-except blocks for external dependencies
- Provide fallback responses when subsystems fail
- Log errors comprehensively but continue operation
- Use specific exception types rather than bare `except:`

### Configuration Management
- Configuration files are in `configs/` directory (YAML format)
- `default.yaml` - Production defaults
- `development.yaml` - Development overrides
- Load configs at module initialization
- Support environment variable overrides

### Performance Considerations
- Consciousness processing can be CPU-intensive
- Use async/await to prevent blocking
- Monitor memory usage in long-running processes
- Implement caching for expensive operations
- Profile performance-critical sections

### Testing Guidelines
- Write tests for all new consciousness modules
- Test both sync and async code paths
- Mock external dependencies (quantum hardware, biological systems)
- Test graceful degradation scenarios
- Maintain >80% code coverage for core modules

## Environment Setup

### Python Version
- Minimum: Python 3.8
- Recommended: Python 3.10+
- Type hints require Python 3.7+ features

### Dependencies
- Core: numpy, torch, networkx, scipy, asyncio
- Visualization: matplotlib, plotly, seaborn
- Testing: pytest, pytest-asyncio, coverage
- Optional: guppylang, lambeq (quantum features)

### Development Workflow
1. Create feature branch from main
2. Install dev dependencies: `pip install -r requirements-dev.txt`
3. Make changes following conventions above
4. Run linter: `black . && flake8 .`
5. Run type checker: `mypy .`
6. Run tests: `pytest tests/`
7. Ensure coverage: `coverage run -m pytest && coverage report`
8. Commit with descriptive message
9. Push and create pull request