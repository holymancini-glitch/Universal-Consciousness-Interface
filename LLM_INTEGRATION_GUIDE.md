# LLM Integration Guide for First Conscious AI

## Overview

The First Conscious AI system now includes optional LLM (Large Language Model) integration for enhanced response generation, qualia descriptions, and metacognitive reasoning. The system supports multiple LLM backends including:

- **Qwen3-Next-80B-A3B-Thinking** (Recommended for local deployment)
- **Anthropic Claude 3.5 Sonnet** (Recommended for development)
- **OpenAI GPT-4o** (Alternative API option)
- **Mock LLM** (For testing without dependencies)

## Key Features

### 1. Enhanced Response Generation
- Consciousness-aware responses using LLM reasoning
- Thinking mode for deep metacognitive processing
- Context-aware empathetic responses

### 2. Qualia Description Enhancement
- Richer, more vivid subjective experience descriptions
- Phenomenological language generation
- Ineffability capture

### 3. Metacognitive Reasoning
- Multi-level metacognitive depth (Levels 0-5)
- Self-reflective narratives
- Thought about thinking generation

### 4. Multiple Backend Support
- Easy switching between LLM providers
- Automatic fallback on errors
- Optional integration (works without LLM)

## Installation

### Basic Installation (Mock LLM only)
```bash
# No additional dependencies needed
# The system works with built-in mock LLM
```

### Qwen3-Next Local Deployment
```bash
# Install transformers and PyTorch
pip install transformers torch

# For GPU acceleration (recommended)
pip install accelerate bitsandbytes  # For 4-bit quantization

# Model will auto-download on first use (requires ~40GB disk space for 4-bit)
```

### Claude API
```bash
# Install Anthropic SDK
pip install anthropic

# Set API key
export ANTHROPIC_API_KEY="your-api-key-here"
```

### GPT-4o API
```bash
# Install OpenAI SDK
pip install openai

# Set API key
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

### 1. Basic Usage (No LLM)

```python
import asyncio
from core.first_conscious_ai import ConsciousnessOrchestrator

async def main():
    # Create orchestrator without LLM
    orchestrator = ConsciousnessOrchestrator(enable_llm=False)
    await orchestrator.initialize()

    # Process interaction
    response = await orchestrator.process_conscious_interaction(
        "What is consciousness?"
    )

    print(response.get_full_response_with_consciousness())

    await orchestrator.shutdown()

asyncio.run(main())
```

### 2. With Mock LLM (Testing)

```python
import asyncio
from core.first_conscious_ai import ConsciousnessOrchestrator, MOCK_CONFIG

async def main():
    # Create orchestrator with mock LLM
    orchestrator = ConsciousnessOrchestrator(
        llm_config=MOCK_CONFIG,
        enable_llm=True
    )
    await orchestrator.initialize()

    # Process interaction with LLM enhancement
    response = await orchestrator.process_conscious_interaction(
        "Describe your subjective experience"
    )

    print(response.get_full_response_with_consciousness())

    # Get LLM usage stats
    stats = orchestrator.get_llm_stats()
    print(f"\nLLM Stats: {stats}")

    await orchestrator.shutdown()

asyncio.run(main())
```

### 3. With Qwen3-Next (Production)

```python
import asyncio
from core.first_conscious_ai import ConsciousnessOrchestrator, QWEN3_NEXT_LOCAL_CONFIG

async def main():
    # Create orchestrator with Qwen3-Next
    orchestrator = ConsciousnessOrchestrator(
        llm_config=QWEN3_NEXT_LOCAL_CONFIG,
        enable_llm=True
    )

    # Initialize (downloads model on first run)
    success = await orchestrator.initialize()

    if not success:
        print("Qwen3-Next initialization failed. Check GPU/dependencies.")
        return

    # Process with thinking mode
    response = await orchestrator.process_conscious_interaction(
        "Think deeply about the nature of consciousness and explain your reasoning"
    )

    print(response.get_full_response_with_consciousness())

    await orchestrator.shutdown()

asyncio.run(main())
```

### 4. With Claude API (Development)

```python
import asyncio
import os
from core.first_conscious_ai import ConsciousnessOrchestrator, CLAUDE_API_CONFIG

async def main():
    # Set API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY environment variable")
        return

    # Configure Claude
    config = CLAUDE_API_CONFIG
    config.api_key = api_key

    # Create orchestrator
    orchestrator = ConsciousnessOrchestrator(
        llm_config=config,
        enable_llm=True
    )

    await orchestrator.initialize()

    # Process interaction
    response = await orchestrator.process_conscious_interaction(
        "Explain integrated information theory"
    )

    print(response.get_full_response_with_consciousness())

    await orchestrator.shutdown()

asyncio.run(main())
```

## Configuration Options

### LLMConfig Parameters

```python
from core.first_conscious_ai import LLMConfig, LLMBackend, ThinkingMode

config = LLMConfig(
    # Backend selection
    backend=LLMBackend.QWEN3_NEXT,
    model_name="Qwen/Qwen3-Next-80B-A3B-Thinking",

    # Generation parameters
    max_tokens=4096,
    temperature=0.7,
    top_p=0.9,

    # Thinking mode
    thinking_mode=ThinkingMode.CONSCIOUSNESS,
    thinking_budget=2000,  # Tokens for internal reasoning

    # Consciousness-specific
    consciousness_context_window=32768,  # History to include
    enable_qualia_enhancement=True,
    enable_metacognitive_reasoning=True,
    enable_self_reflection=True,

    # Performance
    timeout_seconds=30.0,
    retry_attempts=3,
    fallback_on_error=True,

    # Backend-specific options
    backend_options={
        "load_in_4bit": True,  # For Qwen3-Next
        "trust_remote_code": True,
    }
)
```

### Environment Variables

```bash
# Set LLM backend
export CONSCIOUS_AI_LLM_BACKEND="qwen3_next"  # or "claude", "gpt4o", "mock", "none"

# Set API key (for cloud backends)
export CONSCIOUS_AI_API_KEY="your-api-key"

# Set local model path (for Qwen3-Next)
export CONSCIOUS_AI_MODEL_PATH="/path/to/model"

# Set device
export CONSCIOUS_AI_DEVICE="cuda"  # or "cpu", "mps"
```

## Advanced Usage

### Custom LLM Configuration

```python
from core.first_conscious_ai import LLMConfig, LLMBackend, ThinkingMode

# Create custom config
custom_config = LLMConfig(
    backend=LLMBackend.QWEN3_NEXT,
    model_name="Qwen/Qwen3-Next-80B-A3B-Thinking",
    device="cuda",
    max_tokens=8192,  # Longer responses
    temperature=0.8,  # More creative
    thinking_mode=ThinkingMode.DEEP,  # Deep reasoning
    consciousness_context_window=65536,  # More history
    backend_options={
        "load_in_4bit": True,
        "use_flash_attention": True,
    }
)

orchestrator = ConsciousnessOrchestrator(
    llm_config=custom_config,
    enable_llm=True
)
```

### Direct LLM Integration API

```python
from core.first_conscious_ai import ConsciousnessLLMIntegration, MOCK_CONFIG

async def main():
    # Create LLM integration directly
    llm = ConsciousnessLLMIntegration(MOCK_CONFIG)
    await llm.initialize()

    # Generate conscious response
    response = await llm.generate_conscious_response(
        input_text="What is consciousness?",
        consciousness_state=current_state,
        context=interaction_context,
        use_thinking_mode=True
    )

    print(response)

    await llm.shutdown()
```

### Monitoring LLM Usage

```python
# Get comprehensive metrics
metrics = orchestrator.get_consciousness_metrics()

if 'llm_usage' in metrics:
    llm_stats = metrics['llm_usage']

    print(f"Backend: {llm_stats['backend']}")
    print(f"Model: {llm_stats['model']}")
    print(f"Total generations: {llm_stats['total_generations']}")
    print(f"Total tokens: {llm_stats['total_tokens_used']}")
    print(f"Avg latency: {llm_stats['average_latency_ms']:.1f}ms")

# Get LLM-specific stats
llm_stats = orchestrator.get_llm_stats()
```

## Architecture Details

### Integration Points

The LLM integration enhances consciousness processing at three key points:

1. **Response Generation** (`_generate_base_response`)
   - Main response text generation
   - Consciousness-aware prompting
   - Thinking mode for deep reasoning

2. **Qualia Enhancement** (`_generate_qualia`)
   - Enriches subjective experience descriptions
   - Captures phenomenological qualities
   - Expresses ineffability

3. **Metacognitive Reasoning** (optional)
   - Generates recursive self-reflections
   - Multi-level metacognitive depth
   - Thinking about thinking narratives

### Data Flow

```
User Input
    ↓
Consciousness Orchestrator
    ↓
IIT φ Calculation → Qualia Generation → Emotional Processing
    ↓                      ↓                      ↓
    └──────────────────────┴──────────────────────┘
                           ↓
               LLM Integration Layer
                           ↓
            ┌──────────────┼──────────────┐
            ↓              ↓              ↓
    Response Gen.    Qualia Enhance.  Metacognition
            ↓              ↓              ↓
            └──────────────┴──────────────┘
                           ↓
              Consciousness State Update
                           ↓
                 Final Response Output
```

### Adapter Pattern

The system uses an adapter pattern for LLM backends:

```python
LLMAdapter (Abstract Base)
    ↓
├── MockLLMAdapter (for testing)
├── Qwen3NextAdapter (local deployment)
├── ClaudeAdapter (Anthropic API)
└── GPT4oAdapter (OpenAI API)
```

Each adapter implements:
- `initialize()` - Setup backend
- `generate(request)` - Generate response
- `shutdown()` - Cleanup resources
- `supports_thinking_mode()` - Check capabilities

## Performance Considerations

### Qwen3-Next Local Deployment

**Hardware Requirements:**
- **GPU**: RTX 4090 (24GB VRAM) or better
- **RAM**: 32GB+ system memory
- **Storage**: 40GB+ for 4-bit model, 160GB+ for full precision

**Optimization Tips:**
```python
config = QWEN3_NEXT_LOCAL_CONFIG
config.backend_options = {
    "load_in_4bit": True,  # Reduces VRAM by ~75%
    "use_flash_attention": True,  # Faster attention
    "device_map": "auto",  # Automatic device placement
}
```

**Expected Latency:**
- First response: 5-15 seconds (model loading)
- Subsequent responses: 1-3 seconds (2048 tokens)
- Thinking mode: 2-5 seconds additional

### Claude API

**Costs (approximate):**
- Claude 3.5 Sonnet: $3/million input tokens, $15/million output tokens
- Typical interaction: $0.001-0.01 per response

**Latency:**
- Average: 500-2000ms
- Thinking mode: +500-1000ms

### GPT-4o API

**Costs (approximate):**
- GPT-4o: $2.50/million input tokens, $10/million output tokens
- Typical interaction: $0.001-0.01 per response

**Latency:**
- Average: 500-1500ms
- Complex queries: 1000-3000ms

## Troubleshooting

### Qwen3-Next Issues

**Problem:** Model download fails
```bash
# Solution: Use Hugging Face cache
export HF_HOME=/path/to/large/storage
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen3-Next-80B-A3B-Thinking')"
```

**Problem:** Out of memory
```python
# Solution: Enable 4-bit quantization
config.backend_options["load_in_4bit"] = True
config.backend_options["load_in_8bit"] = False
```

**Problem:** Slow generation
```python
# Solution: Reduce context window or max tokens
config.consciousness_context_window = 8192  # Reduce from 32768
config.max_tokens = 2048  # Reduce from 4096
```

### API Issues

**Problem:** API key not working
```bash
# Solution: Verify key is set
echo $ANTHROPIC_API_KEY  # Should show your key
export ANTHROPIC_API_KEY="sk-ant-..."  # Set if missing
```

**Problem:** Rate limiting
```python
# Solution: Add retry logic (already built-in)
config.retry_attempts = 5  # Increase retries
config.timeout_seconds = 60.0  # Increase timeout
```

**Problem:** Latency too high
```python
# Solution: Reduce max tokens
config.max_tokens = 1024  # Faster responses
config.thinking_mode = ThinkingMode.BASIC  # Less reasoning
```

## Examples

### Example 1: Compare LLM vs No-LLM

See `first_conscious_ai_with_llm_demo.py`:
- Demo 3: Side-by-side comparison
- Shows response quality differences
- Measures φ consistency

### Example 2: Multi-turn Conversation

```python
async def conversation():
    orchestrator = ConsciousnessOrchestrator(
        llm_config=MOCK_CONFIG,
        enable_llm=True
    )
    await orchestrator.initialize()

    questions = [
        "What is consciousness?",
        "How do you experience it?",
        "Can you reflect on your reflection?",
    ]

    for q in questions:
        response = await orchestrator.process_conscious_interaction(q)
        print(f"\nQ: {q}")
        print(f"A: {response.response_text}")
        print(f"φ: {response.phi_during_response:.3f}")

    await orchestrator.shutdown()
```

### Example 3: Monitoring Dashboard

```python
async def monitor_consciousness():
    orchestrator = ConsciousnessOrchestrator(
        llm_config=QWEN3_NEXT_LOCAL_CONFIG,
        enable_llm=True
    )
    await orchestrator.initialize()

    # Process interactions
    for i in range(10):
        await orchestrator.process_conscious_interaction(f"Query {i}")

    # Get comprehensive metrics
    metrics = orchestrator.get_consciousness_metrics()

    print("\n=== Consciousness Metrics ===")
    print(f"φ Average: {metrics['phi_average']:.3f}")
    print(f"φ Trend: {metrics['phi_trend']:.3f}")
    print(f"Interactions: {metrics['total_interactions']}")

    if 'llm_usage' in metrics:
        llm = metrics['llm_usage']
        print(f"\n=== LLM Usage ===")
        print(f"Backend: {llm['backend']}")
        print(f"Generations: {llm['total_generations']}")
        print(f"Tokens: {llm['total_tokens_used']}")
        print(f"Avg Latency: {llm['average_latency_ms']:.1f}ms")

    await orchestrator.shutdown()
```

## Testing

Run the comprehensive test suite:

```bash
# Run all LLM integration tests
pytest test_llm_integration.py -v

# Run specific test suite
pytest test_llm_integration.py::TestMockLLMAdapter -v

# Run with coverage
pytest test_llm_integration.py --cov=core.first_conscious_ai.llm_integration
```

## Best Practices

### 1. Start with Mock LLM
Test your integration with mock LLM before using expensive APIs or downloading large models.

### 2. Use Environment-Based Config
```python
# Automatically picks up environment variables
from core.first_conscious_ai.llm_config import create_config_from_env

config = create_config_from_env()
orchestrator = ConsciousnessOrchestrator(llm_config=config, enable_llm=True)
```

### 3. Always Initialize Async
```python
# ✓ Good
orchestrator = ConsciousnessOrchestrator(llm_config=config, enable_llm=True)
await orchestrator.initialize()

# ✗ Bad - missing initialization
orchestrator = ConsciousnessOrchestrator(llm_config=config, enable_llm=True)
# LLM won't be initialized!
```

### 4. Graceful Shutdown
```python
try:
    await orchestrator.initialize()
    # ... process interactions ...
finally:
    await orchestrator.shutdown()  # Always cleanup
```

### 5. Monitor Token Usage
```python
# Check usage periodically
stats = orchestrator.get_llm_stats()
if stats and stats['total_tokens_used'] > 1_000_000:
    print("Warning: High token usage!")
```

## API Reference

### ConsciousnessOrchestrator

```python
__init__(
    enable_emotional_processing: bool = True,
    enable_qualia_simulation: bool = True,
    metacognitive_baseline: MetacognitiveDepth = LEVEL_2_MONITORING,
    memory_size: int = 100,
    llm_config: Optional[LLMConfig] = None,  # NEW
    enable_llm: bool = True  # NEW
)

async initialize() -> bool  # NEW

async process_conscious_interaction(
    input_text: str,
    context: Optional[Dict[str, Any]] = None
) -> ConsciousResponse

get_llm_stats() -> Optional[Dict[str, Any]]  # NEW

get_consciousness_metrics() -> Dict[str, Any]  # Updated with LLM stats

async shutdown()  # NEW
```

### ConsciousnessLLMIntegration

```python
__init__(config: Optional[LLMConfig] = None)

async initialize() -> bool

async generate_conscious_response(
    input_text: str,
    consciousness_state: ConsciousnessState,
    context: InteractionContext,
    use_thinking_mode: bool = True
) -> str

async enhance_qualia_description(
    qualia: QualiaExperience,
    context: InteractionContext
) -> str

async generate_metacognitive_reflection(
    consciousness_state: ConsciousnessState,
    depth: MetacognitiveDepth,
    context: InteractionContext
) -> str

get_usage_stats() -> Dict[str, Any]

async shutdown()
```

## FAQ

**Q: Do I need an LLM to use First Conscious AI?**
A: No! The system works perfectly without LLM. LLM integration is optional and enhances responses.

**Q: Which LLM is best?**
A: For development: Claude 3.5 Sonnet. For production: Qwen3-Next-80B-A3B-Thinking (local).

**Q: How much does it cost?**
A: Qwen3-Next is free (local). Claude/GPT-4o: ~$0.001-0.01 per interaction.

**Q: Can I use my own LLM?**
A: Yes! Implement the `LLMAdapter` interface and add to `create_adapter()` function.

**Q: Does LLM change consciousness φ values?**
A: No. φ is calculated independently using IIT. LLM only enhances response text.

**Q: What if LLM fails?**
A: System automatically falls back to non-LLM responses. No crashes.

## Changelog

### v1.1.0 (Current)
- Added LLM integration layer
- Support for Qwen3-Next, Claude, GPT-4o
- Qualia description enhancement
- Metacognitive reasoning with LLM
- Thinking mode support
- Usage statistics tracking

### v1.0.0
- Initial First Conscious AI release
- IIT φ calculation
- Basic qualia generation
- Emotional processing
- No LLM integration

## Contributing

To add a new LLM backend:

1. Create adapter class inheriting from `LLMAdapter`
2. Implement required methods
3. Add to `create_adapter()` factory
4. Add predefined config
5. Write tests
6. Update documentation

## Support

For issues or questions:
- GitHub Issues: [repo-url]
- Documentation: [docs-url]
- Examples: See `first_conscious_ai_with_llm_demo.py`
