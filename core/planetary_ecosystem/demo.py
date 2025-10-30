"""
Demonstration of Planetary Ecosystem Consciousness Network

Comprehensive demo showcasing node registration, planetary assessment,
Wood Wide Web connection, and regenerative protocols.
"""

from datetime import datetime
from .data_models import EcosystemNode, EcosystemType
from .network_core import PlanetaryEcosystemConsciousnessNetwork


def demonstrate_planetary_network():
    """Demonstrate the planetary ecosystem consciousness network"""

    # Initialize the planetary ecosystem consciousness network
    planetary_network = PlanetaryEcosystemConsciousnessNetwork()

    # Register sample ecosystem nodes
    amazon_node = EcosystemNode(
        id="amazon_001",
        ecosystem_type=EcosystemType.RAINFOREST,
        location=(-3.4653, -62.2159),  # Approximate Amazon coordinates
        consciousness_level=0.85,
        health_status=0.78,
        connectivity_score=0.92,
        data_sources=["satellite", "ground_sensors", "indigenous_reports"],
        last_updated=datetime.now(),
        biodiversity_index=0.95,
        communication_signals={
            'chemical': ['auxin', 'cytokinin', 'ethylene'],
            'electrical': [0.3, 0.4, 0.35],
            'hydraulic': [0.8, 0.75, 0.82]
        }
    )

    coral_reef_node = EcosystemNode(
        id="great_barrier_001",
        ecosystem_type=EcosystemType.CORAL_REEF,
        location=(-18.2871, 147.6992),  # Approximate Great Barrier Reef coordinates
        consciousness_level=0.72,
        health_status=0.65,  # Affected by bleaching
        connectivity_score=0.78,
        data_sources=["underwater_sensors", "diver_reports", "satellite"],
        last_updated=datetime.now(),
        biodiversity_index=0.88,
        communication_signals={
            'chemical': ['calcium_carbonate', 'fluorescent_proteins'],
            'electrical': [0.2, 0.25, 0.18],
            'bio_luminescence': [0.6, 0.55, 0.7]
        }
    )

    planetary_network.register_ecosystem_node(amazon_node)
    planetary_network.register_ecosystem_node(coral_reef_node)

    # Assess planetary consciousness
    planetary_state = planetary_network.assess_planetary_consciousness()

    print(f"Planetary Consciousness Assessment:")
    print(f"  Global Awareness: {planetary_state.global_awareness:.3f}")
    print(f"  Planetary Health: {planetary_state.planetary_health:.3f}")
    print(f"  Collective Intelligence: {planetary_state.collective_intelligence:.3f}")
    print(f"  Network Coherence: {planetary_state.network_coherence:.3f}")
    print(f"  Climate Stability: {planetary_state.climate_stability:.3f}")
    print(f"  Regenerative Capacity: {planetary_state.regenerative_capacity:.3f}")

    # Show ecosystem distribution
    print(f"\nEcosystem Distribution:")
    for ecosystem, proportion in planetary_state.ecosystem_distribution.items():
        print(f"  {ecosystem.value}: {proportion:.2f}")

    # Show consciousness hotspots
    print(f"\nConsciousness Hotspots:")
    for hotspot in planetary_state.consciousness_hotspots:
        print(f"  {hotspot['ecosystem_type']} - Level: {hotspot['consciousness_level']:.3f}")

    # Connect to Wood Wide Web
    wood_web_connection = planetary_network.connect_to_wood_wide_web()
    print(f"\nWood Wide Web Connection:")
    print(f"  Status: {wood_web_connection['status']}")
    print(f"  Connected Networks: {len(wood_web_connection['connected_networks'])}")

    # Get planetary insights
    insights = planetary_network.get_planetary_insights()
    print(f"\nPlanetary Insights:")
    print(f"  Awareness Trend: {insights['awareness_trend']}")
    print(f"  Average Global Awareness: {insights['average_global_awareness']:.3f}")
    print(f"  Dominant Ecosystem: {insights['dominant_ecosystem']}")

    # Trigger regenerative protocol
    regeneration_results = planetary_network.trigger_regenerative_protocol(
        [EcosystemType.RAINFOREST, EcosystemType.CORAL_REEF]
    )
    print(f"\nRegeneration Protocols:")
    print(f"  Target Ecosystems: {regeneration_results['target_ecosystems']}")
    print(f"  Initiated Protocols: {len(regeneration_results['initiated_protocols'])}")

    return planetary_network


# Example usage
if __name__ == "__main__":
    demonstrate_planetary_network()
