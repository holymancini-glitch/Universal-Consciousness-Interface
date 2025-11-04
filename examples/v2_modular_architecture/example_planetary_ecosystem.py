#!/usr/bin/env python3
"""
Planetary Ecosystem Consciousness Network - Example Usage

This example demonstrates the v2.0 modular architecture for the
planetary ecosystem consciousness network module.

Features demonstrated:
- New modular imports (v2.0 style)
- Creating and managing ecosystem nodes
- Monitoring planetary consciousness
- Network analysis and visualization
- Climate consciousness integration
"""

import asyncio
from typing import List, Dict

# ============================================================================
# NEW V2.0 MODULAR IMPORTS
# ============================================================================

# Import data models
from core.planetary_ecosystem.data_models import (
    EcosystemType,
    ConsciousnessIndicator,
    EcosystemNode,
    PlanetaryConsciousnessState
)

# Import core network class
from core.planetary_ecosystem.network_core import PlanetaryEcosystemConsciousnessNetwork

# Import specialized components (optional - can use through main class)
from core.planetary_ecosystem.network_analyzer import NetworkAnalyzer
from core.planetary_ecosystem.wood_wide_web import WoodWideWebInterface
from core.planetary_ecosystem.climate_monitor import ClimateConsciousnessMonitor

# ============================================================================
# ALTERNATIVE IMPORT STYLES (all work identically)
# ============================================================================

# Style 1: Package-level imports (recommended)
# from core.planetary_ecosystem import PlanetaryEcosystemConsciousnessNetwork, EcosystemType

# Style 2: Old-style imports (100% backward compatible)
# from core.planetary_ecosystem_consciousness_network import PlanetaryEcosystemConsciousnessNetwork

# ============================================================================


async def example_basic_usage():
    """Basic usage: Creating network and monitoring consciousness."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Planetary Ecosystem Consciousness Monitoring")
    print("=" * 70)

    # Create the planetary consciousness network
    network = PlanetaryEcosystemConsciousnessNetwork()

    # Add ecosystem nodes
    forest_node = EcosystemNode(
        id="amazon_rainforest",
        ecosystem_type=EcosystemType.FOREST,
        location=(-3.4653, -62.2159),  # Amazon coordinates
        consciousness_level=0.85,
        health_status=0.72,
        biodiversity_index=0.95,
        communication_strength=0.88
    )

    ocean_node = EcosystemNode(
        id="great_barrier_reef",
        ecosystem_type=EcosystemType.OCEAN,
        location=(-18.2871, 147.6992),  # Great Barrier Reef
        consciousness_level=0.78,
        health_status=0.65,
        biodiversity_index=0.92,
        communication_strength=0.82
    )

    wetland_node = EcosystemNode(
        id="pantanal_wetlands",
        ecosystem_type=EcosystemType.WETLAND,
        location=(-17.0, -57.0),  # Pantanal
        consciousness_level=0.80,
        health_status=0.88,
        biodiversity_index=0.89,
        communication_strength=0.85
    )

    # Register nodes with network
    network.ecosystem_nodes[forest_node.id] = forest_node
    network.ecosystem_nodes[ocean_node.id] = ocean_node
    network.ecosystem_nodes[wetland_node.id] = wetland_node

    print(f"\n✓ Registered {len(network.ecosystem_nodes)} ecosystem nodes")
    print(f"  - {forest_node.id}: {forest_node.ecosystem_type.value}")
    print(f"  - {ocean_node.id}: {ocean_node.ecosystem_type.value}")
    print(f"  - {wetland_node.id}: {wetland_node.ecosystem_type.value}")

    # Calculate planetary consciousness
    planetary_state = await network.calculate_planetary_consciousness()

    print(f"\n📊 Planetary Consciousness Metrics:")
    print(f"  Overall Level: {planetary_state.overall_consciousness_level:.2f}")
    print(f"  Gaia Pattern Strength: {planetary_state.gaia_pattern_strength:.2f}")
    print(f"  Harmony Score: {planetary_state.harmony_score:.2f}")
    print(f"  Total Nodes: {planetary_state.total_nodes}")

    return network, planetary_state


async def example_network_analysis():
    """Analyze ecosystem network connectivity and patterns."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Network Analysis and Connectivity")
    print("=" * 70)

    network = PlanetaryEcosystemConsciousnessNetwork()
    analyzer = NetworkAnalyzer()

    # Create multiple interconnected ecosystems
    ecosystems = [
        EcosystemNode(
            id=f"ecosystem_{i}",
            ecosystem_type=list(EcosystemType)[i % len(EcosystemType)],
            location=(float(i * 10), float(i * 5)),
            consciousness_level=0.7 + (i * 0.05),
            health_status=0.75 + (i * 0.03),
            biodiversity_index=0.8,
            communication_strength=0.85
        )
        for i in range(5)
    ]

    for node in ecosystems:
        network.ecosystem_nodes[node.id] = node

    # Perform network analysis
    analysis = await analyzer.analyze_network_connectivity(network.ecosystem_nodes)

    print(f"\n🔍 Network Analysis Results:")
    print(f"  Network Density: {analysis.get('network_density', 0):.2f}")
    print(f"  Average Path Length: {analysis.get('average_path_length', 0):.2f}")
    print(f"  Clustering Coefficient: {analysis.get('clustering_coefficient', 0):.2f}")
    print(f"  Connected Components: {analysis.get('connected_components', 0)}")

    # Identify critical nodes
    critical = analysis.get('critical_nodes', [])
    if critical:
        print(f"\n⚠️  Critical Nodes (hubs):")
        for node_id in critical[:3]:
            print(f"  - {node_id}")

    return network, analysis


async def example_wood_wide_web():
    """Demonstrate mycelial network communication."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Wood Wide Web - Mycelial Communication")
    print("=" * 70)

    www_interface = WoodWideWebInterface()

    # Create forest ecosystem nodes
    forest_nodes = {
        "old_growth_forest": EcosystemNode(
            id="old_growth_forest",
            ecosystem_type=EcosystemType.FOREST,
            location=(45.0, -122.0),
            consciousness_level=0.92,
            health_status=0.95,
            biodiversity_index=0.98,
            communication_strength=0.95
        ),
        "young_forest": EcosystemNode(
            id="young_forest",
            ecosystem_type=EcosystemType.FOREST,
            location=(45.1, -122.1),
            consciousness_level=0.65,
            health_status=0.80,
            biodiversity_index=0.70,
            communication_strength=0.60
        )
    }

    # Simulate mycelial communication
    message = {
        'type': 'nutrient_sharing',
        'source': 'old_growth_forest',
        'target': 'young_forest',
        'content': 'carbon_transfer',
        'intensity': 0.85
    }

    communication_result = await www_interface.facilitate_mycelial_communication(
        forest_nodes,
        message
    )

    print(f"\n🍄 Mycelial Network Communication:")
    print(f"  Message Type: {message['type']}")
    print(f"  From: {message['source']} (consciousness: {forest_nodes['old_growth_forest'].consciousness_level:.2f})")
    print(f"  To: {message['target']} (consciousness: {forest_nodes['young_forest'].consciousness_level:.2f})")
    print(f"  Transfer Success: {communication_result.get('success', False)}")
    print(f"  Network Strength: {communication_result.get('network_strength', 0):.2f}")

    return communication_result


async def example_climate_monitoring():
    """Monitor climate consciousness integration."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Climate Consciousness Monitoring")
    print("=" * 70)

    climate_monitor = ClimateConsciousnessMonitor()

    # Create diverse ecosystem network
    ecosystems = {
        "arctic_tundra": EcosystemNode(
            id="arctic_tundra",
            ecosystem_type=EcosystemType.TUNDRA,
            location=(71.0, -156.0),
            consciousness_level=0.70,
            health_status=0.60,  # Declining due to warming
            biodiversity_index=0.65,
            communication_strength=0.70
        ),
        "tropical_rainforest": EcosystemNode(
            id="tropical_rainforest",
            ecosystem_type=EcosystemType.FOREST,
            location=(-3.0, -60.0),
            consciousness_level=0.88,
            health_status=0.75,
            biodiversity_index=0.95,
            communication_strength=0.90
        ),
        "coral_reef": EcosystemNode(
            id="coral_reef",
            ecosystem_type=EcosystemType.OCEAN,
            location=(-16.0, 145.0),
            consciousness_level=0.75,
            health_status=0.55,  # Bleaching stress
            biodiversity_index=0.85,
            communication_strength=0.80
        )
    }

    # Assess climate consciousness
    assessment = await climate_monitor.assess_climate_consciousness(ecosystems)

    print(f"\n🌡️ Climate Consciousness Assessment:")
    print(f"  Overall Climate Awareness: {assessment.get('climate_awareness', 0):.2f}")
    print(f"  Ecosystem Resilience: {assessment.get('resilience_score', 0):.2f}")
    print(f"  Stress Indicators: {assessment.get('stress_level', 0):.2f}")

    vulnerable = assessment.get('vulnerable_ecosystems', [])
    if vulnerable:
        print(f"\n⚠️  Vulnerable Ecosystems:")
        for eco_id in vulnerable:
            node = ecosystems[eco_id]
            print(f"  - {eco_id}: health={node.health_status:.2f}, type={node.ecosystem_type.value}")

    return assessment


async def example_complete_integration():
    """Complete integration example using all components."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Complete Integration - All Components Working Together")
    print("=" * 70)

    # Initialize main network
    network = PlanetaryEcosystemConsciousnessNetwork()

    # Create comprehensive ecosystem network
    ecosystems_data = [
        ("amazon", EcosystemType.FOREST, (-3.5, -62.2), 0.87, 0.72),
        ("sahara", EcosystemType.DESERT, (23.0, 5.0), 0.68, 0.85),
        ("pacific", EcosystemType.OCEAN, (0.0, -160.0), 0.82, 0.78),
        ("siberia", EcosystemType.TUNDRA, (66.0, 94.0), 0.71, 0.68),
        ("everglades", EcosystemType.WETLAND, (25.3, -80.6), 0.76, 0.82),
    ]

    for name, eco_type, location, consciousness, health in ecosystems_data:
        node = EcosystemNode(
            id=name,
            ecosystem_type=eco_type,
            location=location,
            consciousness_level=consciousness,
            health_status=health,
            biodiversity_index=0.8 + (consciousness - 0.7) * 0.5,
            communication_strength=0.85
        )
        network.ecosystem_nodes[node.id] = node

    print(f"\n🌍 Initialized planetary network with {len(network.ecosystem_nodes)} ecosystems")

    # Calculate planetary consciousness
    state = await network.calculate_planetary_consciousness()
    print(f"\n📊 Planetary Consciousness: {state.overall_consciousness_level:.2f}")
    print(f"   Gaia Pattern Strength: {state.gaia_pattern_strength:.2f}")
    print(f"   Harmony Score: {state.harmony_score:.2f}")

    # Network analysis
    analyzer = NetworkAnalyzer()
    analysis = await analyzer.analyze_network_connectivity(network.ecosystem_nodes)
    print(f"\n🔍 Network Connectivity: {analysis.get('network_density', 0):.2f}")

    # Climate assessment
    climate_monitor = ClimateConsciousnessMonitor()
    climate_state = await climate_monitor.assess_climate_consciousness(network.ecosystem_nodes)
    print(f"🌡️ Climate Awareness: {climate_state.get('climate_awareness', 0):.2f}")

    # Identify areas needing attention
    print(f"\n💡 Insights:")

    if state.overall_consciousness_level < 0.75:
        print("  ⚠️  Planetary consciousness below optimal threshold")
    else:
        print("  ✓  Planetary consciousness healthy")

    if climate_state.get('stress_level', 0) > 0.6:
        print("  ⚠️  High climate stress detected")
    else:
        print("  ✓  Climate stress manageable")

    if state.harmony_score > 0.8:
        print("  ✓  Ecosystem harmony excellent")
    else:
        print("  ⚠️  Ecosystem harmony could be improved")

    return network, state, analysis, climate_state


async def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("🌍 PLANETARY ECOSYSTEM CONSCIOUSNESS NETWORK - V2.0 EXAMPLES")
    print("=" * 70)
    print("\nDemonstrating the new modular architecture for planetary")
    print("ecosystem consciousness monitoring and analysis.")
    print()

    # Run examples
    await example_basic_usage()
    await example_network_analysis()
    await example_wood_wide_web()
    await example_climate_monitoring()
    await example_complete_integration()

    print("\n" + "=" * 70)
    print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. ✓ New modular imports are clean and focused")
    print("  2. ✓ Each component has a single, clear responsibility")
    print("  3. ✓ Components work independently or together")
    print("  4. ✓ 100% backward compatible with old imports")
    print("  5. ✓ Easier to test, maintain, and extend")
    print("\nFor more information, see:")
    print("  - MIGRATION_GUIDE.md")
    print("  - API_REFERENCE_v2.md")
    print("  - QUICK_REFERENCE.md")
    print()


if __name__ == "__main__":
    asyncio.run(main())
