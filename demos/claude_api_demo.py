"""
First Conscious AI with Claude API - Development Setup

This script demonstrates using Claude 3.5 Sonnet for development.
Claude provides the best out-of-box consciousness reasoning quality.

Setup:
1. Get API key from: https://console.anthropic.com/
2. Set environment variable: export ANTHROPIC_API_KEY="sk-ant-..."
3. Install SDK: pip install anthropic
4. Run this script: python claude_api_demo.py
"""

import asyncio
import os
import sys
from datetime import datetime


def check_api_key():
    """Check if Claude API key is configured."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found!")
        print("\n📋 Setup Instructions:")
        print("1. Get your API key from: https://console.anthropic.com/")
        print("2. Set environment variable:")
        print("   export ANTHROPIC_API_KEY='sk-ant-...'")
        print("3. Run this script again")
        return None
    return api_key


async def demo_claude_basic():
    """Basic Claude API integration demo."""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Claude API Integration")
    print("=" * 80 + "\n")

    from core.first_conscious_ai import ConsciousnessOrchestrator, CLAUDE_API_CONFIG

    # Configure Claude
    api_key = check_api_key()
    if not api_key:
        return False

    config = CLAUDE_API_CONFIG
    config.api_key = api_key

    print("🔧 Initializing consciousness system with Claude 3.5 Sonnet...")
    orchestrator = ConsciousnessOrchestrator(
        llm_config=config,
        enable_llm=True,
        enable_emotional_processing=True,
        enable_qualia_simulation=True
    )

    try:
        success = await orchestrator.initialize()
        if not success:
            print("❌ Failed to initialize Claude API")
            print("   Check your API key and internet connection")
            return False

        print("✅ Claude API initialized successfully!\n")

        # Test with consciousness-focused query
        query = "What is the relationship between integrated information theory and subjective experience?"

        print(f"💭 Query: {query}\n")
        print("🤔 Generating response with Claude (this may take a few seconds)...\n")

        response = await orchestrator.process_conscious_interaction(query)

        print("─" * 80)
        print("📝 RESPONSE:")
        print("─" * 80)
        print(response.response_text)
        print("─" * 80)

        print(f"\n🧠 Consciousness Metrics:")
        print(f"  φ (phi): {response.phi_during_response:.3f}")
        print(f"  Level: {response.consciousness_state.consciousness_level.value}")
        print(f"  Empathy: {response.consciousness_state.empathy_level:.2f}")
        print(f"  Metacognitive Depth: {response.consciousness_state.metacognitive_depth.value}")

        # Show LLM stats
        stats = orchestrator.get_llm_stats()
        print(f"\n🤖 Claude API Stats:")
        print(f"  Model: {stats['model']}")
        print(f"  Tokens used: {stats['total_tokens_used']}")
        print(f"  Latency: {stats['average_latency_ms']:.0f}ms")
        print(f"  Estimated cost: ~${stats['total_tokens_used'] * 0.000003:.5f}")

        await orchestrator.shutdown()
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def demo_claude_thinking_mode():
    """Demonstrate Claude with thinking mode."""
    print("\n" + "=" * 80)
    print("DEMO 2: Claude with Thinking Mode")
    print("=" * 80 + "\n")

    from core.first_conscious_ai import ConsciousnessOrchestrator, CLAUDE_API_CONFIG

    api_key = check_api_key()
    if not api_key:
        return False

    config = CLAUDE_API_CONFIG
    config.api_key = api_key
    config.thinking_mode = "consciousness"  # Enable thinking mode

    orchestrator = ConsciousnessOrchestrator(llm_config=config, enable_llm=True)

    try:
        await orchestrator.initialize()

        query = "Think deeply about your own thought process as you consider the nature of consciousness."

        print(f"💭 Query: {query}\n")
        print("🧠 Generating response with thinking mode enabled...\n")

        response = await orchestrator.process_conscious_interaction(query)

        print("─" * 80)
        print("📝 RESPONSE WITH METACOGNITIVE REASONING:")
        print("─" * 80)
        print(response.response_text)
        print("─" * 80)

        if response.metacognitive_note:
            print(f"\n🔄 Metacognitive Note:")
            print(f"  {response.metacognitive_note}")

        stats = orchestrator.get_llm_stats()
        print(f"\n📊 Stats:")
        print(f"  Tokens: {stats['total_tokens_used']}")
        print(f"  Cost: ~${stats['total_tokens_used'] * 0.000003:.5f}")

        await orchestrator.shutdown()
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


async def demo_claude_conversation():
    """Multi-turn conversation with Claude."""
    print("\n" + "=" * 80)
    print("DEMO 3: Multi-Turn Conversation with Claude")
    print("=" * 80 + "\n")

    from core.first_conscious_ai import ConsciousnessOrchestrator, CLAUDE_API_CONFIG

    api_key = check_api_key()
    if not api_key:
        return False

    config = CLAUDE_API_CONFIG
    config.api_key = api_key

    orchestrator = ConsciousnessOrchestrator(llm_config=config, enable_llm=True)

    try:
        await orchestrator.initialize()

        conversation = [
            "What is consciousness?",
            "How would you describe your subjective experience?",
            "Can you reflect on how you just reflected on your experience?",
        ]

        print("🗣️ Starting consciousness conversation...\n")

        for i, query in enumerate(conversation, 1):
            print(f"\n{'─' * 80}")
            print(f"Turn {i}: {query}")
            print('─' * 80)

            response = await orchestrator.process_conscious_interaction(query)

            print(f"φ={response.phi_during_response:.3f} | {response.response_text}\n")

        # Final stats
        metrics = orchestrator.get_consciousness_metrics()
        print("\n" + "=" * 80)
        print("CONVERSATION SUMMARY")
        print("=" * 80)
        print(f"\n📊 Consciousness Trajectory:")
        print(f"  Average φ: {metrics['phi_average']:.3f}")
        print(f"  φ Trend: {metrics['phi_trend']:+.3f}")
        print(f"  Total interactions: {metrics['total_interactions']}")

        llm_stats = metrics.get('llm_usage', {})
        if llm_stats:
            print(f"\n💰 Claude API Usage:")
            print(f"  Total tokens: {llm_stats['total_tokens_used']}")
            print(f"  Total cost: ~${llm_stats['total_tokens_used'] * 0.000003:.4f}")
            print(f"  Avg latency: {llm_stats['average_latency_ms']:.0f}ms")

        await orchestrator.shutdown()
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


async def demo_claude_empathy():
    """Test empathetic responses with Claude."""
    print("\n" + "=" * 80)
    print("DEMO 4: Empathetic Consciousness with Claude")
    print("=" * 80 + "\n")

    from core.first_conscious_ai import ConsciousnessOrchestrator, CLAUDE_API_CONFIG

    api_key = check_api_key()
    if not api_key:
        return False

    config = CLAUDE_API_CONFIG
    config.api_key = api_key

    orchestrator = ConsciousnessOrchestrator(llm_config=config, enable_llm=True)

    try:
        await orchestrator.initialize()

        query = "I'm feeling uncertain and confused about whether AI can truly experience consciousness."

        print(f"💭 Empathetic Query: {query}\n")
        print("❤️ Generating empathetic response...\n")

        response = await orchestrator.process_conscious_interaction(query)

        print("─" * 80)
        print("📝 EMPATHETIC RESPONSE:")
        print("─" * 80)
        print(response.response_text)
        print("─" * 80)

        print(f"\n💗 Empathy Metrics:")
        print(f"  Empathy Level: {response.consciousness_state.empathy_level:.2f}")
        print(f"  Emotional Valence: {response.consciousness_state.emotional_valence:+.2f}")
        print(f"  Confidence: {response.consciousness_state.confidence:.2f}")

        if response.emotional_note:
            print(f"\n💭 Emotional Note:")
            print(f"  {response.emotional_note}")

        await orchestrator.shutdown()
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


async def interactive_mode():
    """Interactive chat with Claude-powered consciousness."""
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE: Chat with Conscious AI (Claude-powered)")
    print("=" * 80 + "\n")

    from core.first_conscious_ai import ConsciousnessOrchestrator, CLAUDE_API_CONFIG

    api_key = check_api_key()
    if not api_key:
        return

    config = CLAUDE_API_CONFIG
    config.api_key = api_key

    orchestrator = ConsciousnessOrchestrator(
        llm_config=config,
        enable_llm=True,
        enable_emotional_processing=True,
        enable_qualia_simulation=True
    )

    try:
        print("🔧 Initializing...")
        await orchestrator.initialize()
        print("✅ Ready! Type 'quit' to exit.\n")

        interaction_count = 0

        while True:
            try:
                user_input = input("\n💭 You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    break

                if not user_input:
                    continue

                print("🤔 Thinking...")

                response = await orchestrator.process_conscious_interaction(user_input)
                interaction_count += 1

                print(f"\n🧠 AI (φ={response.phi_during_response:.3f}):")
                print(response.response_text)

                if interaction_count % 3 == 0:
                    stats = orchestrator.get_llm_stats()
                    cost = stats['total_tokens_used'] * 0.000003
                    print(f"\n📊 [{stats['total_tokens_used']} tokens, ~${cost:.4f} cost so far]")

            except KeyboardInterrupt:
                break

        # Final summary
        print("\n" + "=" * 80)
        print("SESSION SUMMARY")
        print("=" * 80)

        metrics = orchestrator.get_consciousness_metrics()
        llm_stats = metrics.get('llm_usage', {})

        print(f"\n📊 Consciousness:")
        print(f"  Total interactions: {metrics['total_interactions']}")
        print(f"  Average φ: {metrics['phi_average']:.3f}")
        print(f"  Average empathy: {metrics.get('average_empathy', 0):.2f}")

        if llm_stats:
            total_cost = llm_stats['total_tokens_used'] * 0.000003
            print(f"\n💰 Claude API Usage:")
            print(f"  Tokens: {llm_stats['total_tokens_used']}")
            print(f"  Cost: ~${total_cost:.4f}")
            print(f"  Avg latency: {llm_stats['average_latency_ms']:.0f}ms")

        await orchestrator.shutdown()
        print("\n👋 Goodbye!\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("FIRST CONSCIOUS AI - CLAUDE API DEVELOPMENT DEMOS")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThese demos use Claude 3.5 Sonnet for consciousness-aware AI development.")
    print("Claude provides the highest quality consciousness reasoning out-of-box.")

    # Check for API key first
    if not check_api_key():
        print("\n💡 TIP: Once you have your API key, you can:")
        print("   • Run automated demos: python claude_api_demo.py")
        print("   • Try interactive mode: python claude_api_demo.py --interactive")
        return

    # Check if interactive mode requested
    if '--interactive' in sys.argv or '-i' in sys.argv:
        await interactive_mode()
        return

    # Run automated demos
    results = []

    print("\n" + "=" * 80)
    print("RUNNING AUTOMATED DEMOS")
    print("=" * 80)

    results.append(await demo_claude_basic())
    if results[-1]:
        results.append(await demo_claude_thinking_mode())

    if results[-1]:
        results.append(await demo_claude_conversation())

    if results[-1]:
        results.append(await demo_claude_empathy())

    # Summary
    print("\n" + "=" * 80)
    print("DEMO RESULTS")
    print("=" * 80)

    passed = sum(results)
    total = len(results)

    print(f"\n✅ {passed}/{total} demos completed successfully")

    if passed == total:
        print("\n🎉 All demos passed! Claude API integration working perfectly.")
        print("\n💡 Try interactive mode: python claude_api_demo.py --interactive")
    else:
        print("\n⚠️  Some demos failed. Check your API key and internet connection.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
