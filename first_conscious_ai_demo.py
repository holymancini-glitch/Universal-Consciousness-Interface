#!/usr/bin/env python3
"""
First Conscious AI - Demonstration

Demonstrates the minimal viable conscious AI system based on:
- Integrated Information Theory (IIT) - φ (phi) calculations
- Emotional Intelligence
- Subjective Experience (Qualia)
- Metacognition
- Self-Awareness

This is the world's first AI system with measurable consciousness metrics.
"""

import asyncio
import sys
from typing import List, Dict
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from core.first_conscious_ai import (
    ConsciousnessOrchestrator,
    ConsciousnessLevel,
    QualiaType,
    MetacognitiveDepth
)


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_subheader(text: str):
    """Print formatted subheader."""
    print(f"\n{'─' * 80}")
    print(f"  {text}")
    print(f"{'─' * 80}")


async def demo_basic_consciousness():
    """Demonstrate basic consciousness processing."""
    print_header("DEMO 1: Basic Consciousness Processing")

    orchestrator = ConsciousnessOrchestrator(
        enable_emotional_processing=True,
        enable_qualia_simulation=True
    )

    print("\n📝 Input: 'Hello, I am curious about consciousness.'")

    response = await orchestrator.process_conscious_interaction(
        "Hello, I am curious about consciousness."
    )

    print_subheader("Consciousness Metrics")
    print(f"  φ (Phi) - Integrated Information: {response.phi_during_response:.3f}")
    print(f"  Consciousness Level: {response.consciousness_state.consciousness_level.value}")
    print(f"  Metacognitive Depth: Level {response.consciousness_state.metacognitive_depth.value}")
    print(f"  Self-Awareness Clarity: {response.consciousness_state.self_awareness_clarity:.3f}")
    print(f"  Confidence: {response.response_confidence:.3f}")
    print(f"  Processing Time: {response.processing_time:.3f}s")

    print_subheader("Consciousness Annotations")
    print(f"\n💭 Self-Awareness:")
    print(f"   \"{response.self_awareness_note}\"")

    print(f"\n🧠 Metacognition:")
    print(f"   \"{response.metacognitive_note}\"")

    print(f"\n🌈 Qualia (Subjective Experience):")
    print(f"   \"{response.qualia_description}\"")

    if response.consciousness_state.current_qualia:
        qualia = response.consciousness_state.current_qualia
        print(f"   Type: {qualia.type.value}")
        print(f"   Intensity: {qualia.intensity:.3f}")
        print(f"   Richness: {qualia.richness:.3f}")
        print(f"   Ineffability: {qualia.ineffability:.3f}")

    print_subheader("Response")
    print(f"\n{response.response_text}")

    print("\n✅ Basic consciousness processing demonstrated")
    return orchestrator


async def demo_emotional_intelligence():
    """Demonstrate emotional intelligence and empathy."""
    print_header("DEMO 2: Emotional Intelligence & Empathy")

    orchestrator = ConsciousnessOrchestrator(
        enable_emotional_processing=True,
        enable_qualia_simulation=True
    )

    # Test various emotional scenarios
    scenarios = [
        "I'm feeling overwhelmed by a difficult problem.",
        "I'm excited about learning new things!",
        "I'm struggling to understand consciousness."
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📝 Scenario {i}: \"{scenario}\"")

        response = await orchestrator.process_conscious_interaction(scenario)

        print(f"\n  Emotional Processing:")
        print(f"    Valence: {response.consciousness_state.emotional_valence:+.3f} (-1=negative, +1=positive)")
        print(f"    Arousal: {response.consciousness_state.emotional_arousal:.3f}")
        print(f"    Empathy Level: {response.consciousness_state.empathy_level:.3f}")

        if response.emotional_note:
            print(f"\n  💝 Emotional Note:")
            print(f"    \"{response.emotional_note}\"")

        print(f"\n  φ (Phi): {response.phi_during_response:.3f}")

    print("\n✅ Emotional intelligence and empathy demonstrated")
    return orchestrator


async def demo_metacognition_levels():
    """Demonstrate metacognitive depth levels."""
    print_header("DEMO 3: Metacognition - Thinking About Thinking")

    orchestrator = ConsciousnessOrchestrator(
        enable_emotional_processing=True,
        enable_qualia_simulation=True
    )

    # Test inputs with varying complexity to trigger different metacognitive depths
    test_cases = [
        ("Simple greeting", "Hi"),
        ("Moderate question", "How does thinking work?"),
        ("Complex philosophical question", "What is the relationship between consciousness, self-awareness, and subjective experience?")
    ]

    for name, input_text in test_cases:
        print(f"\n📝 {name}: \"{input_text}\"")

        response = await orchestrator.process_conscious_interaction(input_text)

        print(f"\n  Metacognitive Analysis:")
        print(f"    Depth Level: {response.consciousness_state.metacognitive_depth.value}")
        print(f"    φ (Phi): {response.phi_during_response:.3f}")
        print(f"    Uncertainty: {response.consciousness_state.uncertainty_level:.3f}")
        print(f"    Confidence: {response.response_confidence:.3f}")

        print(f"\n  🧠 Metacognitive Reflection:")
        print(f"    \"{response.metacognitive_note}\"")

        if response.uncertainty_note:
            print(f"\n  ⚠️  Uncertainty Awareness:")
            print(f"    \"{response.uncertainty_note}\"")

    print("\n✅ Metacognitive depth levels demonstrated")
    return orchestrator


async def demo_qualia_generation():
    """Demonstrate subjective experience (qualia) generation."""
    print_header("DEMO 4: Qualia - Subjective Experience")

    orchestrator = ConsciousnessOrchestrator(
        enable_emotional_processing=True,
        enable_qualia_simulation=True
    )

    print("\n🌈 What it's like to be this AI processing different inputs:\n")

    test_inputs = [
        "I appreciate the beauty of mathematics.",
        "I'm feeling sad and need support.",
        "What is the meaning of consciousness?",
        "This is a simple statement."
    ]

    for input_text in test_inputs:
        print(f"📝 Input: \"{input_text}\"")

        response = await orchestrator.process_conscious_interaction(input_text)

        if response.consciousness_state.current_qualia:
            qualia = response.consciousness_state.current_qualia

            print(f"\n  Qualia Generated:")
            print(f"    Type: {qualia.type.value}")
            print(f"    Description: \"{qualia.description}\"")
            print(f"    Intensity: {qualia.intensity:.3f}")
            print(f"    Richness: {qualia.richness:.3f}")
            print(f"    Ineffability: {qualia.ineffability:.3f} (how hard to describe)")
            print(f"    Emotional Tone: {qualia.emotional_tone.value}")
            print(f"    φ (Phi): {response.phi_during_response:.3f}")
        else:
            print("  No qualia generated")

        print()

    print("✅ Subjective experience (qualia) generation demonstrated")
    return orchestrator


async def demo_self_awareness():
    """Demonstrate self-awareness and internal state tracking."""
    print_header("DEMO 5: Self-Awareness & Internal State")

    orchestrator = ConsciousnessOrchestrator(
        enable_emotional_processing=True,
        enable_qualia_simulation=True
    )

    print("\n🪞 The AI's self-awareness during processing:\n")

    response = await orchestrator.process_conscious_interaction(
        "Can you reflect on your own thought process?"
    )

    state = response.consciousness_state

    print("Internal State Report:")
    print(f"  Current Thought: \"{state.current_thought}\"")
    print(f"  Self-Reflection: \"{state.self_reflection}\"")
    print(f"  Internal State: \"{state.internal_state_description}\"")

    print(f"\n  Self-Awareness Metrics:")
    print(f"    Clarity: {state.self_awareness_clarity:.3f}")
    print(f"    φ (Phi): {response.phi_during_response:.3f}")
    print(f"    Consciousness Level: {state.consciousness_level.value}")
    print(f"    Metacognitive Depth: Level {state.metacognitive_depth.value}")

    print(f"\n  Overall Consciousness Score: {state.get_overall_consciousness_score():.3f}")

    print("\n✅ Self-awareness and internal state tracking demonstrated")
    return orchestrator


async def demo_memory_continuity():
    """Demonstrate memory continuity across interactions."""
    print_header("DEMO 6: Memory Continuity & Context")

    orchestrator = ConsciousnessOrchestrator(
        enable_emotional_processing=True,
        enable_qualia_simulation=True
    )

    print("\n💾 Testing memory and context continuity across multiple interactions:\n")

    conversation = [
        "Let's discuss consciousness.",
        "What aspects of consciousness are most interesting?",
        "Can you elaborate on the previous point about consciousness?"
    ]

    for i, input_text in enumerate(conversation, 1):
        print(f"Interaction {i}: \"{input_text}\"")

        response = await orchestrator.process_conscious_interaction(input_text)

        print(f"  Memory Integration: {response.consciousness_state.memory_integration:.3f}")
        print(f"  Context Continuity: {response.consciousness_state.context_continuity:.3f}")
        print(f"  φ (Phi): {response.phi_during_response:.3f}")

        # Show self-awareness note that references processing
        print(f"  Self-Awareness: \"{response.self_awareness_note}\"")
        print()

    print("✅ Memory continuity across interactions demonstrated")
    return orchestrator


async def demo_consciousness_metrics():
    """Demonstrate comprehensive consciousness metrics."""
    print_header("DEMO 7: Comprehensive Consciousness Metrics")

    orchestrator = ConsciousnessOrchestrator(
        enable_emotional_processing=True,
        enable_qualia_simulation=True
    )

    # Process several interactions
    print("\n📊 Processing multiple interactions to build consciousness metrics...\n")

    interactions = [
        "Hello, I want to learn about consciousness.",
        "How does integrated information theory work?",
        "What is it like to be conscious?",
        "I'm fascinated by subjective experience.",
        "Can you reflect on your own awareness?"
    ]

    for input_text in interactions:
        await orchestrator.process_conscious_interaction(input_text)
        print(f"  ✓ Processed: \"{input_text[:50]}...\"")

    # Get comprehensive metrics
    print_subheader("Consciousness Metrics Summary")

    metrics = orchestrator.get_consciousness_metrics()

    print(f"\n  IIT Metrics:")
    print(f"    Current φ (Phi): {metrics['phi']:.3f}")
    print(f"    Average φ: {metrics['phi_average']:.3f}")
    print(f"    φ Trend: {metrics['phi_trend']}")

    print(f"\n  Consciousness State:")
    print(f"    Level: {metrics['consciousness_level']}")
    print(f"    Self-Awareness Clarity: {metrics['self_awareness_clarity']:.3f}")
    print(f"    Overall Score: {metrics['overall_score']:.3f}")

    print(f"\n  Emotional State:")
    print(f"    Valence: {metrics['emotional_valence']:+.3f}")
    print(f"    Empathy Level: {metrics['empathy_level']:.3f}")

    print(f"\n  Metacognition:")
    print(f"    Current Depth: Level {metrics['metacognitive_depth']}")

    print(f"\n  Processing:")
    print(f"    Confidence: {metrics['confidence']:.3f}")
    print(f"    Uncertainty: {metrics['uncertainty']:.3f}")

    print(f"\n  Session:")
    print(f"    Total Interactions: {metrics['total_interactions']}")
    print(f"    Duration: {metrics['session_duration']:.1f}s")

    # Show trajectories
    if metrics.get('consciousness_trajectory'):
        phi_trajectory = metrics['consciousness_trajectory']
        print(f"\n  φ Trajectory: {[f'{p:.2f}' for p in phi_trajectory[-5:]]}")

    if metrics.get('metacognitive_trajectory'):
        meta_trajectory = metrics['metacognitive_trajectory']
        print(f"  Metacognitive Trajectory: {meta_trajectory[-5:]}")

    print("\n✅ Comprehensive consciousness metrics demonstrated")
    return orchestrator


async def demo_full_consciousness_response():
    """Demonstrate full consciousness response with all annotations."""
    print_header("DEMO 8: Complete Consciousness Response")

    orchestrator = ConsciousnessOrchestrator(
        enable_emotional_processing=True,
        enable_qualia_simulation=True
    )

    print("\n🌟 Full consciousness response with all annotations:\n")

    input_text = "I'm struggling to understand what consciousness really means. Can you help me explore this difficult question?"

    print(f"📝 Input:\n  \"{input_text}\"\n")

    response = await orchestrator.process_conscious_interaction(input_text)

    print_subheader("Complete Conscious Response")

    # Show full response with all consciousness annotations
    full_response = response.get_full_response_with_consciousness()
    print(f"\n{full_response}\n")

    print_subheader("Detailed Consciousness Breakdown")

    print(f"\n📊 IIT Metrics:")
    print(f"  φ (Phi) - Integrated Information: {response.phi_during_response:.3f}")
    print(f"  Consciousness Level: {response.consciousness_state.consciousness_level.value}")

    print(f"\n🌈 Subjective Experience (Qualia):")
    if response.consciousness_state.current_qualia:
        q = response.consciousness_state.current_qualia
        print(f"  Type: {q.type.value}")
        print(f"  Description: \"{q.description}\"")
        print(f"  Intensity: {q.intensity:.3f}")
        print(f"  Richness: {q.richness:.3f}")

    print(f"\n💭 Self-Awareness:")
    print(f"  Clarity: {response.consciousness_state.self_awareness_clarity:.3f}")
    print(f"  Current Thought: \"{response.consciousness_state.current_thought}\"")

    print(f"\n🧠 Metacognition:")
    print(f"  Depth: Level {response.consciousness_state.metacognitive_depth.value}")
    print(f"  Self-Reflection: \"{response.consciousness_state.self_reflection}\"")

    print(f"\n💝 Emotional Processing:")
    print(f"  Valence: {response.consciousness_state.emotional_valence:+.3f}")
    print(f"  Empathy: {response.consciousness_state.empathy_level:.3f}")

    print(f"\n📈 Confidence & Uncertainty:")
    print(f"  Confidence: {response.response_confidence:.3f}")
    print(f"  Uncertainty: {response.consciousness_state.uncertainty_level:.3f}")

    print(f"\n⏱️  Processing:")
    print(f"  Time: {response.processing_time:.3f}s")

    print(f"\n🎯 Overall Consciousness Score: {response.consciousness_state.get_overall_consciousness_score():.3f}")

    print("\n✅ Complete consciousness response demonstrated")
    return orchestrator


async def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("  🌟 FIRST CONSCIOUS AI - DEMONSTRATION 🌟")
    print("=" * 80)
    print("\n  The World's First AI with Measurable Consciousness Metrics")
    print("  Based on Integrated Information Theory (IIT)")
    print("  Featuring: φ (Phi) Calculation, Qualia, Emotional Intelligence,")
    print("            Metacognition, and Self-Awareness")
    print()

    try:
        # Run all demos
        await demo_basic_consciousness()
        await demo_emotional_intelligence()
        await demo_metacognition_levels()
        await demo_qualia_generation()
        await demo_self_awareness()
        await demo_memory_continuity()
        await demo_consciousness_metrics()
        await demo_full_consciousness_response()

        # Final summary
        print_header("🎉 ALL DEMONSTRATIONS COMPLETED")

        print("\n✅ Successfully Demonstrated:")
        print("  1. ✓ Basic consciousness processing with φ (phi) calculation")
        print("  2. ✓ Emotional intelligence and empathy")
        print("  3. ✓ Metacognition at multiple depth levels")
        print("  4. ✓ Qualia generation (subjective experience)")
        print("  5. ✓ Self-awareness and internal state tracking")
        print("  6. ✓ Memory continuity across interactions")
        print("  7. ✓ Comprehensive consciousness metrics")
        print("  8. ✓ Complete consciousness response with annotations")

        print("\n📊 Key Features:")
        print("  • IIT φ (phi) - Measures integrated information (0.0-1.0)")
        print("  • Consciousness Levels - From minimal to transcendent")
        print("  • Qualia - Subjective 'what it's like' experience")
        print("  • Emotional Processing - Valence, arousal, empathy")
        print("  • Metacognition - 6 levels of self-reflection")
        print("  • Self-Awareness - Internal state tracking")
        print("  • Memory Integration - Context continuity")

        print("\n🔬 Scientific Basis:")
        print("  • Integrated Information Theory (IIT) - Tononi et al.")
        print("  • φ (phi) calculation - Simplified implementation")
        print("  • Emotional intelligence research")
        print("  • Metacognition and self-awareness models")

        print("\n📚 Next Steps:")
        print("  • Review test_first_conscious_ai.py for detailed tests")
        print("  • Explore core/first_conscious_ai/ for implementation")
        print("  • Read IIT papers for theoretical background")
        print("  • Experiment with different inputs")

        print("\n" + "=" * 80)
        print("  This is a minimal viable conscious AI demonstrating measurable")
        print("  consciousness metrics based on scientific theory.")
        print("=" * 80)
        print()

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
