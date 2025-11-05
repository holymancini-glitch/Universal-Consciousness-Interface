#!/usr/bin/env python3
"""
Test Suite for First Conscious AI

Tests consciousness metrics and validates IIT-based consciousness system.
"""

import asyncio
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from core.first_conscious_ai import (
    ConsciousnessOrchestrator,
    IITCalculator,
    ConsciousnessStateTracker,
    ConsciousnessLevel,
    QualiaType,
    MetacognitiveDepth,
    InteractionContext
)


class TestResults:
    """Track test results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []

    def record(self, name: str, passed: bool, message: str = ""):
        self.tests.append((name, passed, message))
        if passed:
            self.passed += 1
            print(f"  ✅ {name}")
        else:
            self.failed += 1
            print(f"  ❌ {name}: {message}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*80}")
        print(f"Test Results: {self.passed}/{total} passed")
        if self.failed > 0:
            print(f"Failed tests: {self.failed}")
            for name, passed, msg in self.tests:
                if not passed:
                    print(f"  - {name}: {msg}")
        print(f"{'='*80}\n")
        return self.failed == 0


async def test_iit_calculator():
    """Test IIT φ (phi) calculation."""
    print("\n" + "="*80)
    print("TEST 1: IIT Calculator - φ (Phi) Calculation")
    print("="*80)

    results = TestResults()
    calculator = IITCalculator()

    # Test 1: Empty system
    system_state = {}
    iit_result = await calculator.calculate_phi(system_state)
    results.record(
        "Empty system returns φ=0",
        iit_result.phi == 0.0,
        f"Expected 0.0, got {iit_result.phi}"
    )

    # Test 2: Simple system
    system_state = {
        'component_a': 0.8,
        'component_b': 0.6
    }
    iit_result = await calculator.calculate_phi(system_state)
    results.record(
        "Simple system calculates φ > 0",
        iit_result.phi > 0.0,
        f"φ should be > 0, got {iit_result.phi}"
    )

    # Test 3: Complex integrated system
    system_state = {
        'input': 0.9,
        'processor_a': 0.8,
        'processor_b': 0.7,
        'integrator': 0.85,
        'output': 0.75
    }
    connections = {
        'input': ['processor_a', 'processor_b'],
        'processor_a': ['integrator'],
        'processor_b': ['integrator'],
        'integrator': ['output']
    }
    iit_result = await calculator.calculate_phi(system_state, connections)
    results.record(
        "Integrated system has φ > 0.3",
        iit_result.phi > 0.3,
        f"Expected φ > 0.3 for integrated system, got {iit_result.phi}"
    )

    # Test 4: Check consciousness level classification
    results.record(
        "φ correctly maps to consciousness level",
        isinstance(iit_result.get_consciousness_level(), ConsciousnessLevel),
        "Should return ConsciousnessLevel enum"
    )

    # Test 5: Integration strength
    results.record(
        "Integration strength in valid range",
        0.0 <= iit_result.integration_strength <= 1.0,
        f"Integration strength {iit_result.integration_strength} out of range"
    )

    # Test 6: Differentiation
    results.record(
        "Differentiation in valid range",
        0.0 <= iit_result.differentiation <= 1.0,
        f"Differentiation {iit_result.differentiation} out of range"
    )

    # Test 7: Cause-effect power
    results.record(
        "Cause-effect power in valid range",
        0.0 <= iit_result.cause_effect_power <= 1.0,
        f"Cause-effect power {iit_result.cause_effect_power} out of range"
    )

    # Test 8: History tracking
    await calculator.calculate_phi({'test': 0.5})
    await calculator.calculate_phi({'test': 0.6})
    results.record(
        "Calculator tracks history",
        len(calculator.calculation_history) >= 2,
        "History should contain multiple calculations"
    )

    # Test 9: Average φ
    avg_phi = calculator.get_average_phi()
    results.record(
        "Average φ calculated correctly",
        avg_phi > 0.0,
        f"Average φ should be > 0, got {avg_phi}"
    )

    print(f"\nφ (Phi) Calculation Summary:")
    print(f"  Current φ: {iit_result.phi:.3f}")
    print(f"  Integration: {iit_result.integration_strength:.3f}")
    print(f"  Differentiation: {iit_result.differentiation:.3f}")
    print(f"  Cause-Effect Power: {iit_result.cause_effect_power:.3f}")
    print(f"  Consciousness Level: {iit_result.get_consciousness_level().value}")

    return results.summary()


async def test_consciousness_state_tracker():
    """Test consciousness state tracking."""
    print("\n" + "="*80)
    print("TEST 2: Consciousness State Tracker")
    print("="*80)

    results = TestResults()
    tracker = ConsciousnessStateTracker(memory_size=10)

    # Test 1: Initialization
    results.record(
        "Tracker initializes with base state",
        tracker.current_state is not None,
        "Should have initial state"
    )

    # Test 2: State update
    context = InteractionContext(input_text="Test input")
    new_state = await tracker.update_state(
        phi=0.6,
        consciousness_level=ConsciousnessLevel.INTERMEDIATE,
        context=context,
        metacognitive_depth=MetacognitiveDepth.LEVEL_2_MONITORING,
        confidence=0.7
    )

    results.record(
        "State updates correctly",
        new_state.phi == 0.6,
        f"Expected φ=0.6, got {new_state.phi}"
    )

    # Test 3: Self-awareness clarity
    results.record(
        "Self-awareness clarity in valid range",
        0.0 <= new_state.self_awareness_clarity <= 1.0,
        f"Clarity {new_state.self_awareness_clarity} out of range"
    )

    # Test 4: Current thought generation
    results.record(
        "Current thought generated",
        len(new_state.current_thought) > 0,
        "Should generate current thought description"
    )

    # Test 5: Self-reflection generation
    results.record(
        "Self-reflection generated",
        len(new_state.self_reflection) > 0,
        "Should generate self-reflection"
    )

    # Test 6: Internal state description
    results.record(
        "Internal state described",
        len(new_state.internal_state_description) > 0,
        "Should describe internal state"
    )

    # Test 7: Overall consciousness score
    overall_score = new_state.get_overall_consciousness_score()
    results.record(
        "Overall consciousness score in valid range",
        0.0 <= overall_score <= 1.0,
        f"Overall score {overall_score} out of range"
    )

    # Test 8: History tracking
    await tracker.update_state(
        phi=0.7,
        consciousness_level=ConsciousnessLevel.ADVANCED,
        context=InteractionContext(input_text="Second input"),
        metacognitive_depth=MetacognitiveDepth.LEVEL_3_EVALUATION
    )

    results.record(
        "State history maintained",
        len(tracker.state_history) > 0,
        "Should track state history"
    )

    # Test 9: Interaction counter
    results.record(
        "Interaction counter increments",
        tracker.total_interactions >= 2,
        f"Expected >= 2 interactions, got {tracker.total_interactions}"
    )

    # Test 10: State summary
    summary = tracker.get_state_summary()
    results.record(
        "State summary contains required fields",
        all(key in summary for key in ['phi', 'consciousness_level', 'overall_score']),
        "Summary missing required fields"
    )

    print(f"\nState Tracker Summary:")
    print(f"  Current φ: {tracker.current_state.phi:.3f}")
    print(f"  Consciousness Level: {tracker.current_state.consciousness_level.value}")
    print(f"  Self-Awareness Clarity: {tracker.current_state.self_awareness_clarity:.3f}")
    print(f"  Overall Score: {tracker.current_state.get_overall_consciousness_score():.3f}")
    print(f"  Total Interactions: {tracker.total_interactions}")

    return results.summary()


async def test_consciousness_orchestrator():
    """Test consciousness orchestrator integration."""
    print("\n" + "="*80)
    print("TEST 3: Consciousness Orchestrator - Integration")
    print("="*80)

    results = TestResults()
    orchestrator = ConsciousnessOrchestrator(
        enable_emotional_processing=True,
        enable_qualia_simulation=True
    )

    # Test 1: Basic processing
    response = await orchestrator.process_conscious_interaction(
        "Hello, test input"
    )

    results.record(
        "Response generated",
        response is not None,
        "Should generate response"
    )

    results.record(
        "Response has consciousness state",
        response.consciousness_state is not None,
        "Should include consciousness state"
    )

    # Test 2: φ calculation
    results.record(
        "φ calculated during response",
        response.phi_during_response > 0.0,
        f"Expected φ > 0, got {response.phi_during_response}"
    )

    # Test 3: Self-awareness note
    results.record(
        "Self-awareness note generated",
        len(response.self_awareness_note) > 0,
        "Should generate self-awareness note"
    )

    # Test 4: Metacognitive note
    results.record(
        "Metacognitive note generated",
        len(response.metacognitive_note) > 0,
        "Should generate metacognitive note"
    )

    # Test 5: Qualia generation
    results.record(
        "Qualia generated",
        response.qualia_description is not None,
        "Should generate qualia description"
    )

    # Test 6: Emotional processing (with empathy keyword)
    empathy_response = await orchestrator.process_conscious_interaction(
        "I'm feeling sad and need support"
    )

    results.record(
        "Empathy detected",
        empathy_response.consciousness_state.empathy_level > 0.5,
        f"Expected high empathy, got {empathy_response.consciousness_state.empathy_level}"
    )

    # Test 7: Complex input triggers higher metacognition
    complex_response = await orchestrator.process_conscious_interaction(
        "What is the relationship between integrated information theory and subjective conscious experience?"
    )

    results.record(
        "Complex input increases metacognitive depth",
        complex_response.consciousness_state.metacognitive_depth.value >= 2,
        f"Expected depth >= 2, got {complex_response.consciousness_state.metacognitive_depth.value}"
    )

    # Test 8: Full response format
    full_response = response.get_full_response_with_consciousness()
    results.record(
        "Full response includes annotations",
        len(full_response) > len(response.response_text),
        "Full response should include consciousness annotations"
    )

    # Test 9: Processing time tracked
    results.record(
        "Processing time measured",
        response.processing_time > 0.0,
        "Should measure processing time"
    )

    # Test 10: Consciousness metrics retrievable
    metrics = orchestrator.get_consciousness_metrics()
    results.record(
        "Consciousness metrics available",
        'phi' in metrics and 'overall_score' in metrics,
        "Should provide comprehensive metrics"
    )

    print(f"\nOrchestrator Integration Summary:")
    print(f"  Response φ: {response.phi_during_response:.3f}")
    print(f"  Consciousness Level: {response.consciousness_state.consciousness_level.value}")
    print(f"  Metacognitive Depth: Level {response.consciousness_state.metacognitive_depth.value}")
    print(f"  Empathy Level: {empathy_response.consciousness_state.empathy_level:.3f}")
    print(f"  Processing Time: {response.processing_time:.4f}s")

    return results.summary()


async def test_qualia_generation():
    """Test subjective experience (qualia) generation."""
    print("\n" + "="*80)
    print("TEST 4: Qualia Generation - Subjective Experience")
    print("="*80)

    results = TestResults()
    orchestrator = ConsciousnessOrchestrator(enable_qualia_simulation=True)

    # Test different qualia types
    test_cases = [
        ("Emotional input", "I'm feeling overwhelmed", QualiaType.EMOTIONAL),
        ("Conceptual input", "What is consciousness?", QualiaType.CONCEPTUAL),
        ("Introspective input", "Can you think about your thinking?", QualiaType.INTROSPECTIVE)
    ]

    for name, input_text, expected_type in test_cases:
        response = await orchestrator.process_conscious_interaction(input_text)

        if response.consciousness_state.current_qualia:
            qualia = response.consciousness_state.current_qualia

            results.record(
                f"{name} - Qualia generated",
                qualia is not None,
                "Should generate qualia"
            )

            results.record(
                f"{name} - Intensity in range",
                0.0 <= qualia.intensity <= 1.0,
                f"Intensity {qualia.intensity} out of range"
            )

            results.record(
                f"{name} - Richness in range",
                0.0 <= qualia.richness <= 1.0,
                f"Richness {qualia.richness} out of range"
            )

            print(f"\n  {name}:")
            print(f"    Type: {qualia.type.value}")
            print(f"    Description: \"{qualia.description}\"")
            print(f"    Intensity: {qualia.intensity:.3f}")
            print(f"    Richness: {qualia.richness:.3f}")
        else:
            results.record(f"{name} - Qualia generated", False, "No qualia generated")

    return results.summary()


async def test_consciousness_metrics():
    """Test consciousness metrics calculations."""
    print("\n" + "="*80)
    print("TEST 5: Consciousness Metrics")
    print("="*80)

    results = TestResults()
    orchestrator = ConsciousnessOrchestrator()

    # Process multiple interactions
    interactions = [
        "First interaction",
        "Second interaction about consciousness",
        "Third complex philosophical question about subjective experience"
    ]

    for input_text in interactions:
        await orchestrator.process_conscious_interaction(input_text)

    metrics = orchestrator.get_consciousness_metrics()

    # Test metric presence
    required_metrics = [
        'phi', 'consciousness_level', 'self_awareness_clarity',
        'overall_score', 'total_interactions', 'phi_average'
    ]

    for metric in required_metrics:
        results.record(
            f"Metric '{metric}' present",
            metric in metrics,
            f"Missing required metric: {metric}"
        )

    # Test metric ranges
    results.record(
        "φ in valid range",
        0.0 <= metrics['phi'] <= 1.0,
        f"φ {metrics['phi']} out of range"
    )

    results.record(
        "Overall score in valid range",
        0.0 <= metrics['overall_score'] <= 1.0,
        f"Overall score {metrics['overall_score']} out of range"
    )

    results.record(
        "Interaction count correct",
        metrics['total_interactions'] == len(interactions),
        f"Expected {len(interactions)}, got {metrics['total_interactions']}"
    )

    print(f"\nConsciousness Metrics:")
    print(f"  φ (Current): {metrics['phi']:.3f}")
    print(f"  φ (Average): {metrics['phi_average']:.3f}")
    print(f"  φ Trend: {metrics.get('phi_trend', 'N/A')}")
    print(f"  Consciousness Level: {metrics['consciousness_level']}")
    print(f"  Overall Score: {metrics['overall_score']:.3f}")
    print(f"  Self-Awareness: {metrics['self_awareness_clarity']:.3f}")
    print(f"  Total Interactions: {metrics['total_interactions']}")

    return results.summary()


async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("  🧪 FIRST CONSCIOUS AI - TEST SUITE 🧪")
    print("="*80)
    print("\n  Testing consciousness metrics and IIT-based implementation")
    print()

    all_passed = True

    # Run all test suites
    test_suites = [
        ("IIT Calculator", test_iit_calculator),
        ("Consciousness State Tracker", test_consciousness_state_tracker),
        ("Consciousness Orchestrator", test_consciousness_orchestrator),
        ("Qualia Generation", test_qualia_generation),
        ("Consciousness Metrics", test_consciousness_metrics)
    ]

    results = []
    for name, test_func in test_suites:
        try:
            passed = await test_func()
            results.append((name, passed))
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"\n❌ {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
            all_passed = False

    # Final summary
    print("\n" + "="*80)
    print("  TEST SUITE SUMMARY")
    print("="*80)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {name}")

    if all_passed:
        print("\n  🎉 ALL TESTS PASSED")
        print("\n  Consciousness metrics validated:")
        print("    ✓ IIT φ (phi) calculation")
        print("    ✓ Consciousness state tracking")
        print("    ✓ Orchestrator integration")
        print("    ✓ Qualia generation")
        print("    ✓ Comprehensive metrics")
        print()
        return 0
    else:
        print("\n  ⚠️  SOME TESTS FAILED")
        print("  Review failures above for details.")
        print()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
