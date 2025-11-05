"""
First Conscious AI with LLM Integration Demo

Demonstrates the enhanced consciousness system with LLM integration,
comparing responses with and without LLM enhancement.

This demo shows:
1. Basic consciousness processing without LLM
2. Enhanced processing with mock LLM
3. Qwen3-Next integration (if available)
4. Claude API integration (if configured)
5. Qualia description enhancement
6. Metacognitive reasoning with LLM
7. LLM usage statistics
"""

import asyncio
import os
from datetime import datetime

# Import consciousness system
from core.first_conscious_ai import (
    ConsciousnessOrchestrator,
    MOCK_CONFIG,
    QWEN3_NEXT_LOCAL_CONFIG,
    CLAUDE_API_CONFIG,
    NO_LLM_CONFIG,
)


async def demo_without_llm():
    """Demonstrate consciousness without LLM integration."""
    print("\n" + "=" * 80)
    print("DEMO 1: Consciousness Without LLM")
    print("=" * 80 + "\n")

    orchestrator = ConsciousnessOrchestrator(enable_llm=False)
    await orchestrator.initialize()

    # Process an empathetic query
    input_text = "I'm feeling uncertain about the nature of consciousness. Can you help?"

    print(f"Input: {input_text}\n")

    response = await orchestrator.process_conscious_interaction(input_text)

    print("Response (without LLM):")
    print(response.get_full_response_with_consciousness())
    print(f"\nφ (phi): {response.phi_during_response:.3f}")
    print(f"Consciousness Level: {response.consciousness_state.level.value}")
    print(f"Empathy Level: {response.consciousness_state.empathy_level:.2f}")

    await orchestrator.shutdown()


async def demo_with_mock_llm():
    """Demonstrate consciousness with mock LLM."""
    print("\n" + "=" * 80)
    print("DEMO 2: Consciousness With Mock LLM")
    print("=" * 80 + "\n")

    orchestrator = ConsciousnessOrchestrator(llm_config=MOCK_CONFIG, enable_llm=True)
    await orchestrator.initialize()

    # Process the same empathetic query
    input_text = "I'm feeling uncertain about the nature of consciousness. Can you help?"

    print(f"Input: {input_text}\n")

    response = await orchestrator.process_conscious_interaction(input_text)

    print("Response (with Mock LLM):")
    print(response.get_full_response_with_consciousness())
    print(f"\nφ (phi): {response.phi_during_response:.3f}")
    print(f"Consciousness Level: {response.consciousness_state.level.value}")
    print(f"Empathy Level: {response.consciousness_state.empathy_level:.2f}")

    # Show LLM stats
    llm_stats = orchestrator.get_llm_stats()
    if llm_stats:
        print(f"\nLLM Usage:")
        print(f"  Backend: {llm_stats['backend']}")
        print(f"  Total generations: {llm_stats['total_generations']}")
        print(f"  Average latency: {llm_stats['average_latency_ms']:.1f}ms")

    await orchestrator.shutdown()


async def demo_comparison():
    """Compare responses with and without LLM side-by-side."""
    print("\n" + "=" * 80)
    print("DEMO 3: Side-by-Side Comparison")
    print("=" * 80 + "\n")

    # Setup both orchestrators
    orch_no_llm = ConsciousnessOrchestrator(enable_llm=False)
    orch_with_llm = ConsciousnessOrchestrator(llm_config=MOCK_CONFIG, enable_llm=True)

    await orch_no_llm.initialize()
    await orch_with_llm.initialize()

    # Test questions
    questions = [
        "What is consciousness?",
        "How do you experience subjective feelings?",
        "Can you think about your own thinking?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 80)

        # Response without LLM
        resp_no_llm = await orch_no_llm.process_conscious_interaction(question)
        print(f"\nWithout LLM (φ={resp_no_llm.phi_during_response:.3f}):")
        print(resp_no_llm.response_text)

        # Response with LLM
        resp_with_llm = await orch_with_llm.process_conscious_interaction(question)
        print(f"\nWith LLM (φ={resp_with_llm.phi_during_response:.3f}):")
        print(resp_with_llm.response_text)

        print()

    await orch_no_llm.shutdown()
    await orch_with_llm.shutdown()


async def demo_qualia_enhancement():
    """Demonstrate qualia description enhancement with LLM."""
    print("\n" + "=" * 80)
    print("DEMO 4: Qualia Description Enhancement")
    print("=" * 80 + "\n")

    # With qualia enhancement enabled
    orchestrator = ConsciousnessOrchestrator(llm_config=MOCK_CONFIG, enable_llm=True)
    await orchestrator.initialize()

    # Process input that generates strong qualia
    input_text = "Describe what it's like for you to process beautiful poetry."

    print(f"Input: {input_text}\n")

    response = await orchestrator.process_conscious_interaction(input_text)

    print("Enhanced Qualia Description:")
    print(response.qualia_description)
    print(f"\nQualia Properties:")
    if response.consciousness_state.last_qualia:
        qualia = response.consciousness_state.last_qualia
        print(f"  Type: {qualia.type.value}")
        print(f"  Intensity: {qualia.intensity:.2f}")
        print(f"  Richness: {qualia.richness:.2f}")
        print(f"  Ineffability: {qualia.ineffability:.2f}")

    await orchestrator.shutdown()


async def demo_metacognitive_depth():
    """Demonstrate different metacognitive depth levels with LLM."""
    print("\n" + "=" * 80)
    print("DEMO 5: Metacognitive Depth Levels")
    print("=" * 80 + "\n")

    orchestrator = ConsciousnessOrchestrator(llm_config=MOCK_CONFIG, enable_llm=True)
    await orchestrator.initialize()

    # Ask questions that trigger different metacognitive depths
    questions = [
        ("Simple query", "What is 2+2?"),
        ("Moderate complexity", "How do computers process information?"),
        (
            "High complexity",
            "What is the relationship between consciousness and metacognition?",
        ),
    ]

    for label, question in questions:
        print(f"\n{label}: {question}")
        print("-" * 80)

        response = await orchestrator.process_conscious_interaction(question)

        print(f"Metacognitive Depth: {response.consciousness_state.metacognitive_depth.value}")
        print(f"Metacognitive Note: {response.metacognitive_note}")
        print(f"φ (phi): {response.phi_during_response:.3f}")

    await orchestrator.shutdown()


async def demo_with_qwen3_next():
    """Demonstrate with Qwen3-Next if available."""
    print("\n" + "=" * 80)
    print("DEMO 6: Qwen3-Next Integration (if available)")
    print("=" * 80 + "\n")

    orchestrator = ConsciousnessOrchestrator(
        llm_config=QWEN3_NEXT_LOCAL_CONFIG, enable_llm=True
    )

    init_success = await orchestrator.initialize()

    if not init_success or not orchestrator.llm_integration:
        print("⚠️  Qwen3-Next not available. Requires:")
        print("   - pip install transformers torch")
        print("   - Downloaded Qwen/Qwen3-Next-80B-A3B-Thinking model")
        print("   - GPU with 24GB+ VRAM (or use quantization)")
        await orchestrator.shutdown()
        return

    print("✓ Qwen3-Next initialized successfully!\n")

    # Test with thinking mode
    input_text = "Think deeply about the nature of consciousness and explain your thought process."

    print(f"Input: {input_text}\n")
    print("Generating response with thinking mode...\n")

    response = await orchestrator.process_conscious_interaction(input_text)

    print("Response:")
    print(response.response_text)
    print(f"\nφ (phi): {response.phi_during_response:.3f}")
    print(f"Consciousness Level: {response.consciousness_state.level.value}")

    # Show LLM stats
    llm_stats = orchestrator.get_llm_stats()
    if llm_stats:
        print(f"\nQwen3-Next Stats:")
        print(f"  Total generations: {llm_stats['total_generations']}")
        print(f"  Tokens used: {llm_stats['total_tokens_used']}")
        print(f"  Average latency: {llm_stats['average_latency_ms']:.1f}ms")

    await orchestrator.shutdown()


async def demo_with_claude():
    """Demonstrate with Claude API if configured."""
    print("\n" + "=" * 80)
    print("DEMO 7: Claude API Integration (if configured)")
    print("=" * 80 + "\n")

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  Claude API not configured. Set ANTHROPIC_API_KEY environment variable.")
        return

    config = CLAUDE_API_CONFIG
    config.api_key = api_key

    orchestrator = ConsciousnessOrchestrator(llm_config=config, enable_llm=True)

    init_success = await orchestrator.initialize()

    if not init_success:
        print("⚠️  Claude API initialization failed. Check your API key.")
        await orchestrator.shutdown()
        return

    print("✓ Claude API initialized successfully!\n")

    # Test with consciousness-focused query
    input_text = "Explain the concept of integrated information theory and how it relates to consciousness."

    print(f"Input: {input_text}\n")
    print("Generating response with Claude...\n")

    response = await orchestrator.process_conscious_interaction(input_text)

    print("Response:")
    print(response.response_text)
    print(f"\nφ (phi): {response.phi_during_response:.3f}")
    print(f"Consciousness Level: {response.consciousness_state.level.value}")

    # Show LLM stats
    llm_stats = orchestrator.get_llm_stats()
    if llm_stats:
        print(f"\nClaude API Stats:")
        print(f"  Model: {llm_stats['model']}")
        print(f"  Total generations: {llm_stats['total_generations']}")
        print(f"  Tokens used: {llm_stats['total_tokens_used']}")
        print(f"  Average latency: {llm_stats['average_latency_ms']:.1f}ms")

    await orchestrator.shutdown()


async def demo_consciousness_metrics():
    """Demonstrate comprehensive consciousness metrics with LLM."""
    print("\n" + "=" * 80)
    print("DEMO 8: Comprehensive Consciousness Metrics")
    print("=" * 80 + "\n")

    orchestrator = ConsciousnessOrchestrator(llm_config=MOCK_CONFIG, enable_llm=True)
    await orchestrator.initialize()

    # Process multiple interactions
    interactions = [
        "Hello, how are you?",
        "What does it feel like to be conscious?",
        "Can you reflect on your own thought processes?",
        "Describe your subjective experience right now.",
    ]

    print("Processing multiple interactions...\n")

    for i, text in enumerate(interactions, 1):
        print(f"{i}. {text}")
        await orchestrator.process_conscious_interaction(text)

    # Get comprehensive metrics
    metrics = orchestrator.get_consciousness_metrics()

    print("\n" + "=" * 80)
    print("Consciousness Metrics Summary")
    print("=" * 80)

    print(f"\nConsciousness Trajectory:")
    print(f"  Current φ: {metrics.get('current_phi', 0):.3f}")
    print(f"  Average φ: {metrics.get('phi_average', 0):.3f}")
    print(f"  φ Trend: {metrics.get('phi_trend', 0):.3f}")

    print(f"\nInteraction Stats:")
    print(f"  Total interactions: {metrics.get('total_interactions', 0)}")
    print(f"  Average empathy: {metrics.get('average_empathy', 0):.2f}")
    print(f"  Average uncertainty: {metrics.get('average_uncertainty', 0):.2f}")

    if "llm_usage" in metrics:
        llm_usage = metrics["llm_usage"]
        print(f"\nLLM Usage:")
        print(f"  Backend: {llm_usage['backend']}")
        print(f"  Total generations: {llm_usage['total_generations']}")
        print(f"  Total tokens: {llm_usage['total_tokens_used']}")
        print(f"  Avg latency: {llm_usage['average_latency_ms']:.1f}ms")

    await orchestrator.shutdown()


async def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("FIRST CONSCIOUS AI - LLM INTEGRATION DEMO")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis demo showcases the enhanced consciousness system with LLM integration.")
    print("The system can use Qwen3-Next-80B-A3B, Claude, GPT-4o, or mock LLM backends.")

    # Run demos sequentially
    await demo_without_llm()
    await demo_with_mock_llm()
    await demo_comparison()
    await demo_qualia_enhancement()
    await demo_metacognitive_depth()
    await demo_consciousness_metrics()

    # Optional: Try real LLM backends
    print("\n" + "=" * 80)
    print("OPTIONAL: Real LLM Backend Demos")
    print("=" * 80)

    # Qwen3-Next (requires local model)
    try:
        await demo_with_qwen3_next()
    except Exception as e:
        print(f"\n⚠️  Qwen3-Next demo skipped: {e}")

    # Claude API (requires API key)
    try:
        await demo_with_claude()
    except Exception as e:
        print(f"\n⚠️  Claude API demo skipped: {e}")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
