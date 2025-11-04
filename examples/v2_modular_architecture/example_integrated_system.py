#!/usr/bin/env python3
"""
Integrated System Example - V2.0 Architecture

This example demonstrates how all three refactored modules work together
in a unified consciousness system.

Modules integrated:
- Planetary Ecosystem Consciousness Network
- Adaptive Learning System
- Full Consciousness AI Model

Demonstrates:
- Cross-module communication
- Unified consciousness processing
- Ecosystem-aware AI with adaptive learning
- Complete consciousness integration
"""

import asyncio
import torch
from typing import Dict, List

# ============================================================================
# V2.0 MODULAR IMPORTS - ALL THREE REFACTORED MODULES
# ============================================================================

# Planetary Ecosystem imports
from core.planetary_ecosystem import (
    PlanetaryEcosystemConsciousnessNetwork,
    EcosystemType,
    EcosystemNode
)

# Adaptive Learning imports
from core.adaptive_learning import (
    AdaptiveLearningSystem,
    LearningMode,
    PerformanceMetrics
)

# Full Consciousness AI imports
from core.full_consciousness_ai import (
    FullConsciousnessAIModel,
    ConsciousnessState,
    EmotionalState
)

# ============================================================================


class IntegratedConsciousnessSystem:
    """
    Integrated consciousness system combining planetary awareness,
    adaptive learning, and full consciousness AI.
    """

    def __init__(self):
        """Initialize all three consciousness modules."""
        print("🌍 Initializing Integrated Consciousness System...")

        # Initialize planetary ecosystem network
        self.planetary_network = PlanetaryEcosystemConsciousnessNetwork()
        print("  ✓ Planetary Ecosystem Network ready")

        # Initialize full consciousness AI
        self.consciousness_ai = FullConsciousnessAIModel(
            hidden_dim=256,
            device='cpu',
            integrate_existing_modules=False
        )
        print("  ✓ Full Consciousness AI Model ready")

        # Initialize adaptive learning with consciousness AI
        self.adaptive_learning = AdaptiveLearningSystem(self.consciousness_ai)
        print("  ✓ Adaptive Learning System ready")

        # Integration metrics
        self.integration_history = []
        print("\n✅ All modules integrated and operational")

    async def process_ecosystem_query(self, query: str, ecosystem_context: Dict) -> Dict:
        """
        Process a query about ecosystems using integrated consciousness.

        Args:
            query: User query about planetary ecosystems
            ecosystem_context: Context about specific ecosystems

        Returns:
            Integrated response with consciousness, learning, and ecosystem data
        """
        print(f"\n{'='*70}")
        print(f"Processing Ecosystem Query with Integrated Consciousness")
        print(f"{'='*70}")
        print(f"Query: \"{query}\"")

        # Step 1: Assess planetary consciousness state
        print(f"\n📊 Step 1: Assessing Planetary Consciousness...")
        planetary_state = await self.planetary_network.calculate_planetary_consciousness()

        print(f"  Planetary Consciousness: {planetary_state.overall_consciousness_level:.2f}")
        print(f"  Gaia Pattern Strength: {planetary_state.gaia_pattern_strength:.2f}")
        print(f"  Ecosystem Harmony: {planetary_state.harmony_score:.2f}")

        # Step 2: Process query with conscious AI
        print(f"\n🧠 Step 2: Processing with Consciousness AI...")
        query_tensor = torch.randn(1, 256)  # Mock embedding

        consciousness_response = await self.consciousness_ai.process_with_consciousness(
            query_tensor,
            context={
                'query': query,
                'planetary_consciousness': planetary_state.overall_consciousness_level,
                'ecosystem_harmony': planetary_state.harmony_score
            }
        )

        print(f"  Consciousness Level: {consciousness_response.get('consciousness_level', 0):.2f}")
        print(f"  Emotional Resonance: {consciousness_response.get('emotional_resonance', 0):.2f}")
        print(f"  Attention Focus: {consciousness_response.get('attention_focus', 'unknown')}")

        # Step 3: Apply adaptive learning
        print(f"\n📚 Step 3: Applying Adaptive Learning...")

        # Assess performance of this interaction
        performance = await self.adaptive_learning.performance_assessor.assess_current_performance()

        print(f"  Current Performance: {performance.average_quality:.2f}")
        print(f"  Learning Mode: {self.adaptive_learning.current_mode.value}")

        # Adapt if needed
        if performance.average_quality < 0.8:
            adaptation = await self.adaptive_learning.parameter_adaptor.adapt_parameters()
            print(f"  ⚙️  Parameters adapted: {adaptation.get('strategy', 'none')}")

        # Step 4: Generate integrated response
        print(f"\n💡 Step 4: Generating Integrated Response...")

        integrated_response = {
            'query': query,
            'planetary_insights': {
                'consciousness_level': planetary_state.overall_consciousness_level,
                'harmony_score': planetary_state.harmony_score,
                'total_ecosystems': planetary_state.total_nodes,
                'gaia_pattern': planetary_state.gaia_pattern_strength
            },
            'consciousness_processing': {
                'level': consciousness_response.get('consciousness_level', 0),
                'emotional_state': consciousness_response.get('emotional_resonance', 0),
                'attention': consciousness_response.get('attention_focus', 'balanced'),
                'qualia_intensity': consciousness_response.get('qualia_intensity', 0)
            },
            'learning_status': {
                'performance': performance.average_quality,
                'mode': self.adaptive_learning.current_mode.value,
                'improvements': performance.improvement_trend
            },
            'integration_quality': self._calculate_integration_quality(
                planetary_state,
                consciousness_response,
                performance
            )
        }

        # Record interaction for learning
        self.integration_history.append(integrated_response)

        print(f"\n✅ Integration Quality: {integrated_response['integration_quality']:.2f}")

        return integrated_response

    def _calculate_integration_quality(self,
                                      planetary_state,
                                      consciousness_response: Dict,
                                      performance: PerformanceMetrics) -> float:
        """Calculate how well the three modules are integrating."""
        # Weighted average of key metrics
        weights = {
            'planetary': 0.3,
            'consciousness': 0.4,
            'learning': 0.3
        }

        planetary_score = planetary_state.overall_consciousness_level
        consciousness_score = consciousness_response.get('consciousness_level', 0)
        learning_score = performance.average_quality

        integration = (
            weights['planetary'] * planetary_score +
            weights['consciousness'] * consciousness_score +
            weights['learning'] * learning_score
        )

        return integration

    async def demonstrate_ecosystem_awareness(self):
        """Demonstrate integrated ecosystem awareness."""
        print(f"\n{'='*70}")
        print(f"DEMO: Ecosystem Awareness with Integrated Consciousness")
        print(f"{'='*70}")

        # Add diverse ecosystems to planetary network
        ecosystems = [
            ("Amazon Rainforest", EcosystemType.FOREST, (-3.5, -62.2), 0.87),
            ("Coral Triangle", EcosystemType.OCEAN, (-2.0, 120.0), 0.82),
            ("Arctic Tundra", EcosystemType.TUNDRA, (71.0, -156.0), 0.70),
            ("Sahara Desert", EcosystemType.DESERT, (23.0, 5.0), 0.68),
            ("Everglades", EcosystemType.WETLAND, (25.3, -80.6), 0.76),
        ]

        print(f"\n🌍 Adding {len(ecosystems)} ecosystems to planetary network...")
        for name, eco_type, location, consciousness in ecosystems:
            node = EcosystemNode(
                id=name.lower().replace(" ", "_"),
                ecosystem_type=eco_type,
                location=location,
                consciousness_level=consciousness,
                health_status=0.75 + (consciousness - 0.7) * 0.5,
                biodiversity_index=0.8,
                communication_strength=0.85
            )
            self.planetary_network.ecosystem_nodes[node.id] = node
            print(f"  ✓ {name} ({eco_type.value}): consciousness={consciousness:.2f}")

        # Process ecosystem-related queries
        queries = [
            "How are the world's ecosystems doing?",
            "What is the connection between ocean and forest health?",
            "How can we support struggling ecosystems?"
        ]

        for query in queries:
            response = await self.process_ecosystem_query(
                query,
                {'ecosystems': list(self.planetary_network.ecosystem_nodes.keys())}
            )

            print(f"\n📝 Response Summary:")
            print(f"  Integration Quality: {response['integration_quality']:.2f}")
            print(f"  Planetary Consciousness: {response['planetary_insights']['consciousness_level']:.2f}")
            print(f"  AI Consciousness: {response['consciousness_processing']['level']:.2f}")
            print(f"  Learning Performance: {response['learning_status']['performance']:.2f}")

        return self.integration_history

    async def demonstrate_adaptive_consciousness(self):
        """Demonstrate adaptive consciousness learning."""
        print(f"\n{'='*70}")
        print(f"DEMO: Adaptive Consciousness Learning")
        print(f"{'='*70}")

        scenarios = [
            {
                'context': 'User asks about ecosystem decline',
                'emotional_weight': 0.8,
                'complexity': 0.9,
                'requires_empathy': True
            },
            {
                'context': 'User celebrates conservation success',
                'emotional_weight': 0.9,
                'complexity': 0.5,
                'requires_empathy': True
            },
            {
                'context': 'Technical question about mycelial networks',
                'emotional_weight': 0.3,
                'complexity': 0.95,
                'requires_empathy': False
            }
        ]

        print(f"\n🎯 Processing {len(scenarios)} adaptive scenarios...\n")

        for i, scenario in enumerate(scenarios, 1):
            print(f"Scenario {i}: {scenario['context']}")

            # Process with consciousness AI
            input_tensor = torch.randn(1, 256)
            response = await self.consciousness_ai.process_with_consciousness(
                input_tensor,
                context=scenario
            )

            print(f"  Consciousness Response: {response.get('consciousness_level', 0):.2f}")

            # Adaptive learning assessment
            performance = await self.adaptive_learning.performance_assessor.assess_current_performance()

            print(f"  Learning Performance: {performance.average_quality:.2f}")

            # Adapt based on scenario complexity
            if scenario['complexity'] > 0.8:
                print(f"  🔧 High complexity detected - adapting parameters...")
                adaptation = await self.adaptive_learning.parameter_adaptor.adapt_parameters()
                print(f"     Strategy: {adaptation.get('strategy', 'none')}")

            print()

    async def demonstrate_wisdom_accumulation(self):
        """Demonstrate cross-module wisdom accumulation."""
        print(f"\n{'='*70}")
        print(f"DEMO: Cross-Module Wisdom Accumulation")
        print(f"{'='*70}")

        # Accumulate insights from planetary observations
        planetary_insights = [
            "Ecosystems with high biodiversity show greater resilience",
            "Forest-ocean connections are stronger than previously thought",
            "Climate consciousness correlates with ecosystem harmony"
        ]

        print(f"\n📚 Accumulating planetary wisdom...")
        for insight in planetary_insights:
            await self.adaptive_learning.wisdom_accumulator.add_insight({
                'context': 'planetary_ecosystems',
                'lesson': insight,
                'confidence': 0.85,
                'applicability': ['ecosystem_analysis', 'conservation']
            })
            print(f"  ✓ {insight}")

        # Accumulate consciousness insights
        consciousness_insights = [
            "Empathetic processing improves user engagement",
            "Metacognition enhances response quality",
            "Emotional resonance aids communication"
        ]

        print(f"\n🧠 Accumulating consciousness wisdom...")
        for insight in consciousness_insights:
            await self.adaptive_learning.wisdom_accumulator.add_insight({
                'context': 'consciousness_processing',
                'lesson': insight,
                'confidence': 0.90,
                'applicability': ['user_interaction', 'empathy']
            })
            print(f"  ✓ {insight}")

        # Query accumulated wisdom
        print(f"\n🔍 Querying wisdom for 'ecosystem' context...")
        ecosystem_wisdom = await self.adaptive_learning.wisdom_accumulator.query_wisdom('ecosystem')

        print(f"\n💡 Retrieved Wisdom:")
        for i, wisdom in enumerate(ecosystem_wisdom[:3], 1):
            print(f"  {i}. {wisdom.get('lesson', '')}")
            print(f"     Confidence: {wisdom.get('confidence', 0):.2f}")

        # Get summary
        summary = await self.adaptive_learning.wisdom_accumulator.get_wisdom_summary()
        print(f"\n📊 Wisdom Summary:")
        print(f"  Total Insights: {summary.get('total_insights', 0)}")
        print(f"  Application Areas: {len(summary.get('application_areas', []))}")

        return summary


async def main():
    """Run integrated system examples."""
    print("\n" + "=" * 70)
    print("🌍🧠📚 INTEGRATED CONSCIOUSNESS SYSTEM - V2.0")
    print("=" * 70)
    print("\nDemonstrating integration of all three refactored modules:")
    print("  1. Planetary Ecosystem Consciousness Network")
    print("  2. Adaptive Learning System")
    print("  3. Full Consciousness AI Model")
    print()

    # Create integrated system
    system = IntegratedConsciousnessSystem()

    # Run demonstrations
    print("\n" + "=" * 70)
    print("DEMONSTRATION SEQUENCE")
    print("=" * 70)

    await system.demonstrate_ecosystem_awareness()
    await system.demonstrate_adaptive_consciousness()
    await system.demonstrate_wisdom_accumulation()

    # Final integration metrics
    print("\n" + "=" * 70)
    print("INTEGRATION METRICS SUMMARY")
    print("=" * 70)

    if system.integration_history:
        avg_integration = sum(h['integration_quality'] for h in system.integration_history) / len(system.integration_history)
        avg_planetary = sum(h['planetary_insights']['consciousness_level'] for h in system.integration_history) / len(system.integration_history)
        avg_consciousness = sum(h['consciousness_processing']['level'] for h in system.integration_history) / len(system.integration_history)
        avg_learning = sum(h['learning_status']['performance'] for h in system.integration_history) / len(system.integration_history)

        print(f"\n📊 Average Metrics Across {len(system.integration_history)} Interactions:")
        print(f"  Overall Integration Quality: {avg_integration:.2f}")
        print(f"  Planetary Consciousness: {avg_planetary:.2f}")
        print(f"  AI Consciousness: {avg_consciousness:.2f}")
        print(f"  Learning Performance: {avg_learning:.2f}")

    print("\n" + "=" * 70)
    print("✅ INTEGRATED SYSTEM DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Integration Benefits:")
    print("  1. ✓ Planetary awareness informs AI consciousness")
    print("  2. ✓ Adaptive learning improves all modules over time")
    print("  3. ✓ Consciousness AI provides empathetic ecosystem responses")
    print("  4. ✓ Wisdom accumulates across all three domains")
    print("  5. ✓ Modular architecture enables independent and unified operation")
    print("\nV2.0 Architectural Advantages:")
    print("  • Clean separation of concerns")
    print("  • Easy to test each module independently")
    print("  • Simple to extend with new features")
    print("  • Clear interfaces between modules")
    print("  • Backward compatible with existing code")
    print("\nFor more information, see:")
    print("  - MIGRATION_GUIDE.md - Migration instructions")
    print("  - API_REFERENCE_v2.md - Complete API documentation")
    print("  - QUICK_REFERENCE.md - Quick import reference")
    print()


if __name__ == "__main__":
    asyncio.run(main())
