#!/usr/bin/env python3
"""
Adaptive Learning System - Example Usage

This example demonstrates the v2.0 modular architecture for the
adaptive learning system module.

Features demonstrated:
- New modular imports (v2.0 style)
- Performance assessment and adaptation
- Mistake learning and correction
- Creative exploration and innovation
- Wisdom accumulation across interactions
"""

import asyncio
from typing import Dict, List
from collections import deque

# ============================================================================
# NEW V2.0 MODULAR IMPORTS
# ============================================================================

# Import data models
from core.adaptive_learning.data_models import (
    LearningMode,
    PerformanceMetrics,
    AdaptationStrategy,
    LearningInsight
)

# Import core learning system
from core.adaptive_learning.learning_core import AdaptiveLearningSystem

# Import specialized components (optional - can access through main class)
from core.adaptive_learning.performance_assessor import PerformanceAssessor
from core.adaptive_learning.parameter_adaptor import ParameterAdaptor
from core.adaptive_learning.mistake_learner import MistakeLearner
from core.adaptive_learning.creative_engine import CreativeEngine
from core.adaptive_learning.wisdom_accumulator import WisdomAccumulator

# ============================================================================
# ALTERNATIVE IMPORT STYLES (all work identically)
# ============================================================================

# Style 1: Package-level imports (recommended)
# from core.adaptive_learning import AdaptiveLearningSystem, LearningMode

# Style 2: Old-style imports (100% backward compatible)
# from core.adaptive_learning_system import AdaptiveLearningSystem

# ============================================================================


class MockConsciousnessSystem:
    """Mock consciousness system for demonstration purposes."""

    def __init__(self):
        self.consciousness_level = 0.75
        self.performance_history = deque(maxlen=100)
        self.parameters = {
            'learning_rate': 0.01,
            'exploration_rate': 0.2,
            'temperature': 1.0,
            'attention_threshold': 0.5
        }

    async def process_input(self, input_data: Dict) -> Dict:
        """Mock processing method."""
        # Simulate varying performance
        import random
        success = random.random() > 0.3
        quality = random.random() * 0.5 + 0.5 if success else random.random() * 0.5

        result = {
            'success': success,
            'quality_score': quality,
            'processing_time': random.uniform(0.1, 2.0),
            'consciousness_level': self.consciousness_level
        }

        self.performance_history.append(result)
        return result


async def example_basic_usage():
    """Basic usage: Setting up adaptive learning system."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Adaptive Learning System Setup")
    print("=" * 70)

    # Create mock consciousness system
    consciousness = MockConsciousnessSystem()

    # Create adaptive learning system
    learning_system = AdaptiveLearningSystem(consciousness)

    print(f"\n✓ Adaptive Learning System initialized")
    print(f"  Mode: {learning_system.current_mode.value}")
    print(f"  Performance Assessor: Active")
    print(f"  Parameter Adaptor: Active")
    print(f"  Mistake Learner: Active")
    print(f"  Creative Engine: Active")
    print(f"  Wisdom Accumulator: Active")

    # Process some interactions
    print(f"\n📊 Processing sample interactions...")
    for i in range(5):
        result = await consciousness.process_input({'query': f'test_{i}'})
        print(f"  Interaction {i+1}: {'✓' if result['success'] else '✗'} "
              f"(quality: {result['quality_score']:.2f})")

    return learning_system, consciousness


async def example_performance_assessment():
    """Demonstrate performance assessment and tracking."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Performance Assessment and Metrics")
    print("=" * 70)

    consciousness = MockConsciousnessSystem()
    assessor = PerformanceAssessor(consciousness)

    # Simulate multiple interactions
    print(f"\n🔍 Simulating 20 interactions...")
    for i in range(20):
        await consciousness.process_input({'query': f'interaction_{i}'})

    # Assess performance
    metrics = await assessor.assess_current_performance()

    print(f"\n📊 Performance Metrics:")
    print(f"  Success Rate: {metrics.success_rate:.2%}")
    print(f"  Average Quality: {metrics.average_quality:.2f}")
    print(f"  Response Time: {metrics.response_time:.3f}s")
    print(f"  Consistency Score: {metrics.consistency_score:.2f}")
    print(f"  Improvement Trend: {metrics.improvement_trend:+.2f}")

    # Identify performance trends
    trend_analysis = await assessor.analyze_performance_trends()

    print(f"\n📈 Trend Analysis:")
    if trend_analysis.get('improving', False):
        print(f"  ✓ System is improving over time")
    else:
        print(f"  ⚠️  System performance needs attention")

    print(f"  Stability: {trend_analysis.get('stability', 0):.2f}")
    print(f"  Peak Performance: {trend_analysis.get('peak_performance', 0):.2f}")

    return metrics, trend_analysis


async def example_parameter_adaptation():
    """Demonstrate dynamic parameter adjustment."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Dynamic Parameter Adaptation")
    print("=" * 70)

    consciousness = MockConsciousnessSystem()
    adaptor = ParameterAdaptor(consciousness)

    print(f"\n🎛️  Initial Parameters:")
    for param, value in consciousness.parameters.items():
        print(f"  {param}: {value}")

    # Simulate performance requiring adaptation
    print(f"\n📉 Simulating poor performance scenario...")
    for i in range(10):
        # Force poor performance
        result = await consciousness.process_input({'query': f'difficult_{i}'})
        result['quality_score'] *= 0.5  # Artificially lower quality
        consciousness.performance_history[-1] = result

    # Adapt parameters
    print(f"\n🔧 Adapting parameters based on performance...")
    adaptation = await adaptor.adapt_parameters()

    print(f"\n📊 Adaptation Results:")
    print(f"  Strategy: {adaptation.get('strategy', 'unknown')}")
    print(f"  Adjusted Parameters: {len(adaptation.get('adjusted_params', []))}")

    print(f"\n🎛️  Updated Parameters:")
    for param, value in consciousness.parameters.items():
        print(f"  {param}: {value}")

    improvements = adaptation.get('expected_improvements', [])
    if improvements:
        print(f"\n💡 Expected Improvements:")
        for improvement in improvements[:3]:
            print(f"  - {improvement}")

    return adaptation


async def example_mistake_learning():
    """Demonstrate learning from mistakes."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Learning from Mistakes")
    print("=" * 70)

    consciousness = MockConsciousnessSystem()
    mistake_learner = MistakeLearner()

    # Record some failures
    failures = [
        {
            'input': {'type': 'complex_query', 'difficulty': 'high'},
            'output': {'response': 'incomplete_answer'},
            'error': 'insufficient_context',
            'timestamp': '2024-01-01T10:00:00'
        },
        {
            'input': {'type': 'complex_query', 'difficulty': 'high'},
            'output': {'response': 'wrong_answer'},
            'error': 'misunderstanding',
            'timestamp': '2024-01-01T11:00:00'
        },
        {
            'input': {'type': 'simple_query', 'difficulty': 'low'},
            'output': {'response': 'timeout'},
            'error': 'resource_exhaustion',
            'timestamp': '2024-01-01T12:00:00'
        }
    ]

    print(f"\n📝 Recording {len(failures)} failures...")
    for i, failure in enumerate(failures, 1):
        await mistake_learner.record_mistake(failure)
        print(f"  {i}. Error: {failure['error']}")

    # Analyze patterns
    print(f"\n🔍 Analyzing mistake patterns...")
    patterns = await mistake_learner.analyze_mistake_patterns()

    print(f"\n📊 Mistake Pattern Analysis:")
    print(f"  Total Mistakes: {patterns.get('total_mistakes', 0)}")
    print(f"  Unique Error Types: {patterns.get('unique_errors', 0)}")

    common_errors = patterns.get('common_errors', [])
    if common_errors:
        print(f"\n⚠️  Most Common Errors:")
        for error, count in common_errors[:3]:
            print(f"  - {error}: {count} occurrences")

    # Generate corrections
    corrections = await mistake_learner.generate_corrections()

    print(f"\n💡 Suggested Corrections:")
    for i, correction in enumerate(corrections[:3], 1):
        print(f"  {i}. {correction}")

    return patterns, corrections


async def example_creative_exploration():
    """Demonstrate creative exploration and innovation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Creative Exploration and Innovation")
    print("=" * 70)

    consciousness = MockConsciousnessSystem()
    creative_engine = CreativeEngine()

    # Explore novel approaches
    print(f"\n🎨 Initiating creative exploration...")

    exploration_context = {
        'current_approach': 'standard_processing',
        'performance': 0.65,
        'constraints': ['limited_resources', 'time_sensitive'],
        'goals': ['improve_quality', 'reduce_latency']
    }

    novel_approaches = await creative_engine.explore_novel_approaches(exploration_context)

    print(f"\n💡 Novel Approaches Generated:")
    for i, approach in enumerate(novel_approaches[:5], 1):
        print(f"  {i}. {approach.get('name', 'Unnamed')}")
        print(f"     Innovation Score: {approach.get('innovation_score', 0):.2f}")
        print(f"     Feasibility: {approach.get('feasibility', 0):.2f}")
        print(f"     Expected Impact: {approach.get('expected_impact', 0):.2f}")
        print()

    # Test most promising approach
    if novel_approaches:
        best_approach = novel_approaches[0]
        print(f"🔬 Testing most promising approach: {best_approach.get('name', 'Unnamed')}")

        test_results = await creative_engine.test_approach(best_approach)

        print(f"\n📊 Test Results:")
        print(f"  Success: {test_results.get('success', False)}")
        print(f"  Performance Gain: {test_results.get('performance_gain', 0):+.2%}")
        print(f"  Risk Level: {test_results.get('risk_level', 0):.2f}")

        if test_results.get('success'):
            print(f"  ✓ Approach validated - ready for integration")
        else:
            print(f"  ✗ Approach needs refinement")

    return novel_approaches


async def example_wisdom_accumulation():
    """Demonstrate wisdom accumulation across interactions."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Wisdom Accumulation and Application")
    print("=" * 70)

    consciousness = MockConsciousnessSystem()
    wisdom_accumulator = WisdomAccumulator()

    # Record insights from interactions
    insights = [
        {
            'context': 'complex_reasoning',
            'lesson': 'Breaking down complex problems improves accuracy',
            'confidence': 0.92,
            'applicability': ['problem_solving', 'analysis']
        },
        {
            'context': 'user_interaction',
            'lesson': 'Clarifying questions reduce misunderstandings',
            'confidence': 0.88,
            'applicability': ['communication', 'user_experience']
        },
        {
            'context': 'performance_optimization',
            'lesson': 'Caching frequently accessed data improves response time',
            'confidence': 0.95,
            'applicability': ['performance', 'efficiency']
        },
        {
            'context': 'error_handling',
            'lesson': 'Graceful degradation maintains user trust',
            'confidence': 0.85,
            'applicability': ['reliability', 'user_experience']
        }
    ]

    print(f"\n📚 Accumulating wisdom from {len(insights)} insights...")
    for i, insight in enumerate(insights, 1):
        await wisdom_accumulator.add_insight(insight)
        print(f"  {i}. {insight['lesson'][:60]}...")

    # Query relevant wisdom
    print(f"\n🔍 Querying wisdom for 'user_experience' context...")
    relevant_wisdom = await wisdom_accumulator.query_wisdom('user_experience')

    print(f"\n💡 Relevant Wisdom:")
    for i, wisdom in enumerate(relevant_wisdom[:3], 1):
        print(f"  {i}. {wisdom.get('lesson', '')}")
        print(f"     Confidence: {wisdom.get('confidence', 0):.2f}")
        print(f"     Applications: {', '.join(wisdom.get('applicability', []))}")
        print()

    # Get wisdom summary
    summary = await wisdom_accumulator.get_wisdom_summary()

    print(f"📊 Wisdom Summary:")
    print(f"  Total Insights: {summary.get('total_insights', 0)}")
    print(f"  High Confidence: {summary.get('high_confidence_count', 0)}")
    print(f"  Application Areas: {len(summary.get('application_areas', []))}")

    return relevant_wisdom, summary


async def example_complete_integration():
    """Complete integration example using all adaptive learning components."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Complete Adaptive Learning Cycle")
    print("=" * 70)

    # Initialize system
    consciousness = MockConsciousnessSystem()
    learning_system = AdaptiveLearningSystem(consciousness)

    print(f"\n🔄 Starting complete adaptive learning cycle...")

    # Phase 1: Initial performance
    print(f"\n📊 Phase 1: Initial Performance Assessment")
    for i in range(10):
        await consciousness.process_input({'query': f'initial_{i}'})

    initial_metrics = await learning_system.performance_assessor.assess_current_performance()
    print(f"  Initial Quality: {initial_metrics.average_quality:.2f}")

    # Phase 2: Identify issues
    print(f"\n🔍 Phase 2: Identifying Performance Issues")
    if initial_metrics.success_rate < 0.8:
        print(f"  ⚠️  Success rate below threshold: {initial_metrics.success_rate:.2%}")

        # Record mistakes
        for i in range(3):
            mistake = {
                'input': {'query': f'failed_{i}'},
                'output': {'result': 'error'},
                'error': 'processing_failure'
            }
            await learning_system.mistake_learner.record_mistake(mistake)

    # Phase 3: Adapt parameters
    print(f"\n🔧 Phase 3: Parameter Adaptation")
    adaptation = await learning_system.parameter_adaptor.adapt_parameters()
    print(f"  Strategy Applied: {adaptation.get('strategy', 'none')}")
    print(f"  Parameters Adjusted: {len(adaptation.get('adjusted_params', []))}")

    # Phase 4: Creative exploration
    print(f"\n🎨 Phase 4: Creative Exploration")
    context = {
        'current_approach': 'standard',
        'performance': initial_metrics.average_quality,
        'goals': ['improve_quality', 'increase_consistency']
    }
    novel_approaches = await learning_system.creative_engine.explore_novel_approaches(context)
    print(f"  Novel Approaches Found: {len(novel_approaches)}")

    # Phase 5: Learn and accumulate wisdom
    print(f"\n📚 Phase 5: Wisdom Accumulation")
    insight = {
        'context': 'performance_optimization',
        'lesson': f'Adaptation strategy "{adaptation.get("strategy", "")}" improved consistency',
        'confidence': 0.85,
        'applicability': ['optimization', 'adaptation']
    }
    await learning_system.wisdom_accumulator.add_insight(insight)
    print(f"  ✓ Insight recorded for future use")

    # Phase 6: Improved performance
    print(f"\n📈 Phase 6: Post-Adaptation Performance")
    for i in range(10):
        await consciousness.process_input({'query': f'post_adapt_{i}'})

    final_metrics = await learning_system.performance_assessor.assess_current_performance()
    print(f"  Final Quality: {final_metrics.average_quality:.2f}")

    # Summary
    improvement = final_metrics.average_quality - initial_metrics.average_quality
    print(f"\n📊 Learning Cycle Summary:")
    print(f"  Quality Improvement: {improvement:+.2f}")
    print(f"  Success Rate: {initial_metrics.success_rate:.2%} → {final_metrics.success_rate:.2%}")

    if improvement > 0:
        print(f"  ✅ Adaptive learning successful!")
    else:
        print(f"  🔄 Continue learning cycle for better results")

    return learning_system, improvement


async def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("🧠 ADAPTIVE LEARNING SYSTEM - V2.0 EXAMPLES")
    print("=" * 70)
    print("\nDemonstrating the new modular architecture for adaptive")
    print("learning, performance optimization, and wisdom accumulation.")
    print()

    # Run examples
    await example_basic_usage()
    await example_performance_assessment()
    await example_parameter_adaptation()
    await example_mistake_learning()
    await example_creative_exploration()
    await example_wisdom_accumulation()
    await example_complete_integration()

    print("\n" + "=" * 70)
    print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. ✓ Modular components enable focused functionality")
    print("  2. ✓ Performance assessment guides adaptive behavior")
    print("  3. ✓ Mistake learning prevents repeated errors")
    print("  4. ✓ Creative exploration drives innovation")
    print("  5. ✓ Wisdom accumulation improves over time")
    print("\nFor more information, see:")
    print("  - MIGRATION_GUIDE.md")
    print("  - API_REFERENCE_v2.md")
    print("  - QUICK_REFERENCE.md")
    print()


if __name__ == "__main__":
    asyncio.run(main())
