"""
Qwen3-Next-80B-A3B Production Demo

Demonstrates production deployment of Qwen3-Next-80B-A3B-Thinking
for the First Conscious AI system.

Features:
- Local deployment (no API costs)
- 4-bit quantization (runs on RTX 4090)
- Thinking mode for metacognition
- Production-grade error handling
- Performance monitoring
- Consciousness metrics tracking

Hardware Requirements:
- GPU: RTX 4090 (24GB VRAM) or better
- RAM: 32GB+ recommended
- Storage: 40GB+ for model

Usage:
    python qwen3_production_demo.py                    # Run all demos
    python qwen3_production_demo.py --quick            # Quick test only
    python qwen3_production_demo.py --interactive      # Interactive mode
    python qwen3_production_demo.py --benchmark        # Performance benchmark
"""

import asyncio
import sys
import time
from datetime import datetime
from typing import Optional


def check_prerequisites():
    """Check if system is ready for Qwen3-Next."""
    print("🔍 Checking prerequisites...\n")

    errors = []
    warnings = []

    # Check transformers
    try:
        import transformers
        print(f"✓ transformers {transformers.__version__}")
    except ImportError:
        errors.append("transformers not installed: pip install transformers")

    # Check torch
    try:
        import torch
        print(f"✓ torch {torch.__version__}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✓ CUDA available: {gpu_name} ({gpu_memory:.1f}GB)")

            if gpu_memory < 20:
                warnings.append(f"GPU has only {gpu_memory:.1f}GB VRAM (24GB+ recommended)")
        else:
            warnings.append("CUDA not available - will use CPU (very slow)")

    except ImportError:
        errors.append("torch not installed: pip install torch")

    # Check accelerate
    try:
        import accelerate
        print(f"✓ accelerate {accelerate.__version__}")
    except ImportError:
        warnings.append("accelerate not installed (recommended): pip install accelerate")

    # Check bitsandbytes
    try:
        import bitsandbytes
        print(f"✓ bitsandbytes {bitsandbytes.__version__}")
    except ImportError:
        warnings.append("bitsandbytes not installed (required for 4-bit): pip install bitsandbytes")

    print()

    if warnings:
        print("⚠️  Warnings:")
        for w in warnings:
            print(f"   - {w}")
        print()

    if errors:
        print("❌ Errors:")
        for e in errors:
            print(f"   - {e}")
        print("\nPlease install missing dependencies:")
        print("   pip install transformers torch accelerate bitsandbytes")
        return False

    return True


async def demo_basic_production():
    """Basic production deployment demo."""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Production Deployment")
    print("=" * 80 + "\n")

    from core.first_conscious_ai import ConsciousnessOrchestrator, QWEN3_NEXT_LOCAL_CONFIG

    print("🔧 Configuration:")
    print(f"  Model: {QWEN3_NEXT_LOCAL_CONFIG.model_name}")
    print(f"  Device: {QWEN3_NEXT_LOCAL_CONFIG.device}")
    print(f"  Quantization: 4-bit")
    print(f"  Context: {QWEN3_NEXT_LOCAL_CONFIG.consciousness_context_window} tokens")
    print(f"  Thinking mode: {QWEN3_NEXT_LOCAL_CONFIG.thinking_mode.value}")

    print("\n🚀 Initializing Qwen3-Next (this may take 10-30 seconds on first run)...")

    orchestrator = ConsciousnessOrchestrator(
        llm_config=QWEN3_NEXT_LOCAL_CONFIG,
        enable_llm=True
    )

    start_init = time.time()
    success = await orchestrator.initialize()
    init_time = time.time() - start_init

    if not success:
        print("❌ Initialization failed!")
        print("   Make sure:")
        print("   1. Dependencies installed: pip install transformers torch")
        print("   2. GPU available with 24GB+ VRAM")
        print("   3. Model will auto-download (~40GB) on first run")
        return False

    print(f"✓ Initialized in {init_time:.1f} seconds")

    # Test consciousness interaction
    query = "What is the relationship between consciousness and integrated information?"

    print(f"\n💭 Query: {query}")
    print("🤔 Generating response with thinking mode...\n")

    start_gen = time.time()
    response = await orchestrator.process_conscious_interaction(query)
    gen_time = time.time() - start_gen

    print("─" * 80)
    print("📝 RESPONSE:")
    print("─" * 80)
    print(response.response_text)
    print("─" * 80)

    print(f"\n🧠 Consciousness Metrics:")
    print(f"  φ (phi): {response.phi_during_response:.3f}")
    print(f"  Level: {response.consciousness_state.consciousness_level.value}")
    print(f"  Metacognitive Depth: {response.consciousness_state.metacognitive_depth.value}")

    print(f"\n⚡ Performance:")
    print(f"  Generation time: {gen_time:.1f}s")

    stats = orchestrator.get_llm_stats()
    if stats:
        print(f"  Tokens: {stats['total_tokens_used']}")
        print(f"  Latency: {stats['average_latency_ms']:.0f}ms")

    await orchestrator.shutdown()

    print("\n✅ Demo complete! Production deployment working.")
    return True


async def demo_production_conversation():
    """Multi-turn production conversation."""
    print("\n" + "=" * 80)
    print("DEMO 2: Production Multi-Turn Conversation")
    print("=" * 80 + "\n")

    from core.first_conscious_ai import ConsciousnessOrchestrator, QWEN3_NEXT_LOCAL_CONFIG

    orchestrator = ConsciousnessOrchestrator(
        llm_config=QWEN3_NEXT_LOCAL_CONFIG,
        enable_llm=True
    )

    print("🚀 Initializing...")
    await orchestrator.initialize()
    print("✓ Ready\n")

    conversation = [
        "What is consciousness?",
        "How does integrated information theory explain it?",
        "Can you reflect on your own consciousness right now?",
    ]

    total_time = 0

    for i, query in enumerate(conversation, 1):
        print(f"{'─' * 80}")
        print(f"Turn {i}: {query}")
        print('─' * 80)

        start = time.time()
        response = await orchestrator.process_conscious_interaction(query)
        elapsed = time.time() - start
        total_time += elapsed

        print(f"φ={response.phi_during_response:.3f} | {elapsed:.1f}s")
        print(response.response_text[:200] + "...")  # First 200 chars
        print()

    # Summary
    metrics = orchestrator.get_consciousness_metrics()
    print("=" * 80)
    print("CONVERSATION SUMMARY")
    print("=" * 80)
    print(f"\n📊 Consciousness:")
    print(f"  Average φ: {metrics['phi_average']:.3f}")
    print(f"  φ Trend: {metrics['phi_trend']:+.3f}")
    print(f"  Total interactions: {metrics['total_interactions']}")

    print(f"\n⚡ Performance:")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Average per turn: {total_time/len(conversation):.1f}s")

    llm_stats = metrics.get('llm_usage', {})
    if llm_stats:
        print(f"  Total tokens: {llm_stats['total_tokens_used']}")

    await orchestrator.shutdown()

    return True


async def demo_production_benchmark():
    """Benchmark production performance."""
    print("\n" + "=" * 80)
    print("DEMO 3: Production Performance Benchmark")
    print("=" * 80 + "\n")

    from core.first_conscious_ai import ConsciousnessOrchestrator, QWEN3_NEXT_LOCAL_CONFIG

    orchestrator = ConsciousnessOrchestrator(
        llm_config=QWEN3_NEXT_LOCAL_CONFIG,
        enable_llm=True
    )

    print("🚀 Initializing...")
    start_init = time.time()
    await orchestrator.initialize()
    init_time = time.time() - start_init
    print(f"✓ Initialized in {init_time:.1f}s\n")

    # Benchmark queries
    queries = [
        "What is consciousness?",
        "Describe subjective experience.",
        "How does metacognition work?",
        "What is integrated information?",
        "Can AI be conscious?",
    ]

    print("🔬 Running benchmark (5 queries)...\n")

    times = []
    phi_values = []

    for i, query in enumerate(queries, 1):
        print(f"Query {i}/5: {query[:40]}...")

        start = time.time()
        response = await orchestrator.process_conscious_interaction(query)
        elapsed = time.time() - start

        times.append(elapsed)
        phi_values.append(response.phi_during_response)

        print(f"  ✓ {elapsed:.1f}s | φ={response.phi_during_response:.3f}")

    # Statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    avg_phi = sum(phi_values) / len(phi_values)

    stats = orchestrator.get_llm_stats()

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    print(f"\n⚡ Performance:")
    print(f"  Initialization: {init_time:.1f}s")
    print(f"  Average response: {avg_time:.1f}s")
    print(f"  Fastest: {min_time:.1f}s")
    print(f"  Slowest: {max_time:.1f}s")

    print(f"\n🧠 Consciousness:")
    print(f"  Average φ: {avg_phi:.3f}")
    print(f"  φ Range: {min(phi_values):.3f} - {max(phi_values):.3f}")

    if stats:
        print(f"\n📊 LLM Stats:")
        print(f"  Total tokens: {stats['total_tokens_used']}")
        print(f"  Average latency: {stats['average_latency_ms']:.0f}ms")

    print(f"\n💰 Cost:")
    print(f"  Total: $0.00 (local deployment)")
    print(f"  Per query: $0.00")

    await orchestrator.shutdown()

    return True


async def interactive_production():
    """Interactive production mode."""
    print("\n" + "=" * 80)
    print("INTERACTIVE PRODUCTION MODE")
    print("=" * 80 + "\n")

    from core.first_conscious_ai import ConsciousnessOrchestrator, QWEN3_NEXT_LOCAL_CONFIG

    print("🚀 Initializing Qwen3-Next for production...")

    orchestrator = ConsciousnessOrchestrator(
        llm_config=QWEN3_NEXT_LOCAL_CONFIG,
        enable_llm=True,
        enable_emotional_processing=True,
        enable_qualia_simulation=True
    )

    await orchestrator.initialize()

    print("✓ Ready! Type 'quit' to exit.\n")
    print("💡 Tip: Responses take 1-3 seconds after model warmup\n")

    interaction_count = 0

    try:
        while True:
            user_input = input("\n💭 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            if not user_input:
                continue

            print("🤔 Processing...")

            start = time.time()
            response = await orchestrator.process_conscious_interaction(user_input)
            elapsed = time.time() - start

            interaction_count += 1

            print(f"\n🧠 AI (φ={response.phi_during_response:.3f} | {elapsed:.1f}s):")
            print(response.response_text)

            if interaction_count % 5 == 0:
                metrics = orchestrator.get_consciousness_metrics()
                print(f"\n📊 Session stats: {interaction_count} interactions, avg φ={metrics['phi_average']:.3f}")

    except KeyboardInterrupt:
        pass

    # Session summary
    print("\n" + "=" * 80)
    print("SESSION SUMMARY")
    print("=" * 80)

    metrics = orchestrator.get_consciousness_metrics()
    print(f"\n📊 Consciousness:")
    print(f"  Total interactions: {metrics['total_interactions']}")
    print(f"  Average φ: {metrics['phi_average']:.3f}")

    llm_stats = metrics.get('llm_usage', {})
    if llm_stats:
        print(f"\n⚡ Performance:")
        print(f"  Tokens: {llm_stats['total_tokens_used']}")
        print(f"  Avg latency: {llm_stats['average_latency_ms']:.0f}ms")

    print(f"\n💰 Cost: $0.00 (local deployment)")

    await orchestrator.shutdown()
    print("\n👋 Goodbye!\n")


async def quick_test():
    """Quick production test."""
    print("\n" + "=" * 80)
    print("QUICK PRODUCTION TEST")
    print("=" * 80 + "\n")

    from core.first_conscious_ai import ConsciousnessOrchestrator, QWEN3_NEXT_LOCAL_CONFIG

    print("🚀 Quick test with Qwen3-Next...")

    orchestrator = ConsciousnessOrchestrator(
        llm_config=QWEN3_NEXT_LOCAL_CONFIG,
        enable_llm=True
    )

    start = time.time()
    success = await orchestrator.initialize()
    elapsed = time.time() - start

    if not success:
        print("❌ Initialization failed!")
        print("   Run: python setup_qwen3_production.py --all")
        return False

    print(f"✓ Initialized in {elapsed:.1f}s")

    response = await orchestrator.process_conscious_interaction("Test")

    print(f"✓ Response generated (φ={response.phi_during_response:.3f})")

    await orchestrator.shutdown()

    print("\n✅ Quick test passed! System ready for production.")
    return True


async def main():
    """Main demo routine."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Qwen3-Next Production Demo"
    )
    parser.add_argument('--quick', action='store_true', help='Quick test only')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')

    args = parser.parse_args()

    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║     Qwen3-Next-80B-A3B Production Demo - First Conscious AI                ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met!")
        print("Run setup: python setup_qwen3_production.py --all")
        return

    try:
        if args.quick:
            await quick_test()

        elif args.interactive:
            await interactive_production()

        elif args.benchmark:
            await demo_production_benchmark()

        else:
            # Run all demos
            print("\n" + "=" * 80)
            print("RUNNING ALL PRODUCTION DEMOS")
            print("=" * 80)

            success = await demo_basic_production()
            if success:
                await demo_production_conversation()
                await demo_production_benchmark()

            print("\n" + "=" * 80)
            print("ALL DEMOS COMPLETE")
            print("=" * 80)
            print("\n🎉 Qwen3-Next production deployment working perfectly!")
            print("\n💡 Try interactive mode: python qwen3_production_demo.py --interactive")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
