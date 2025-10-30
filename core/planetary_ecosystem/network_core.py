"""
Planetary Ecosystem Consciousness Network Core

Main coordination class for the planetary ecosystem consciousness network,
connecting to Earth's ecosystem awareness through the "Wood Wide Web".
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Handle optional dependencies with fallbacks
try:
    import numpy as np  # type: ignore
except ImportError:
    import statistics

    class MockNumPy:
        @staticmethod
        def mean(values):
            return statistics.mean(values) if values else 0.0

    np = MockNumPy()

from .data_models import (
    EcosystemNode,
    PlanetaryConsciousnessState,
    EcosystemType
)
from .network_analyzer import NetworkAnalyzer
from .wood_wide_web import WoodWideWebInterface
from .climate_monitor import ClimateConsciousnessMonitor
from .regeneration_engine import RegenerationEngine

logger = logging.getLogger(__name__)


class PlanetaryEcosystemConsciousnessNetwork:
    """Revolutionary Planetary Ecosystem Consciousness Network connecting to Earth's ecosystem awareness"""

    def __init__(self) -> None:
        self.ecosystem_nodes: Dict[str, EcosystemNode] = {}
        self.consciousness_history: List[PlanetaryConsciousnessState] = []
        self.network_analyzer: NetworkAnalyzer = NetworkAnalyzer()
        self.wood_wide_web_interface: WoodWideWebInterface = WoodWideWebInterface()
        self.climate_monitor: ClimateConsciousnessMonitor = ClimateConsciousnessMonitor()
        self.regeneration_engine: RegenerationEngine = RegenerationEngine()

        logger.info("🌍🌐 Planetary Ecosystem Consciousness Network Initialized")
        logger.info("Connecting to Earth's ecosystem awareness through the Wood Wide Web")

    def register_ecosystem_node(self, node: EcosystemNode) -> None:
        """Register an ecosystem node in the planetary network"""
        self.ecosystem_nodes[node.id] = node
        logger.info(f"Registered ecosystem node: {node.id} ({node.ecosystem_type.value})")

    def unregister_ecosystem_node(self, node_id: str) -> bool:
        """Unregister an ecosystem node from the planetary network"""
        if node_id in self.ecosystem_nodes:
            del self.ecosystem_nodes[node_id]
            logger.info(f"Unregistered ecosystem node: {node_id}")
            return True
        return False

    def update_node_data(self, node_id: str, data: Dict[str, Any]) -> bool:
        """Update data for a specific ecosystem node"""
        if node_id not in self.ecosystem_nodes:
            logger.warning(f"Node {node_id} not found in network")
            return False

        node = self.ecosystem_nodes[node_id]

        # Update node properties based on new data
        if 'consciousness_level' in data:
            node.consciousness_level = data['consciousness_level']
        if 'health_status' in data:
            node.health_status = data['health_status']
        if 'connectivity_score' in data:
            node.connectivity_score = data['connectivity_score']
        if 'biodiversity_index' in data:
            node.biodiversity_index = data['biodiversity_index']
        if 'communication_signals' in data:
            node.communication_signals.update(data['communication_signals'])

        node.last_updated = datetime.now()
        return True

    def assess_planetary_consciousness(self) -> PlanetaryConsciousnessState:
        """Assess the overall planetary consciousness state"""
        if not self.ecosystem_nodes:
            return self._create_empty_state()

        # Calculate global awareness
        consciousness_levels = [node.consciousness_level for node in self.ecosystem_nodes.values()]
        global_awareness = np.mean(consciousness_levels) if consciousness_levels else 0.0

        # Calculate ecosystem distribution
        ecosystem_counts = {}
        for node in self.ecosystem_nodes.values():
            ecosystem_type = node.ecosystem_type
            if ecosystem_type not in ecosystem_counts:
                ecosystem_counts[ecosystem_type] = 0
            ecosystem_counts[ecosystem_type] += 1

        total_nodes = len(self.ecosystem_nodes)
        ecosystem_distribution = {
            ecosystem: count / total_nodes
            for ecosystem, count in ecosystem_counts.items()
        }

        # Identify consciousness hotspots
        consciousness_hotspots = self._identify_consciousness_hotspots()

        # Assess environmental stress
        environmental_stress = self._assess_environmental_stress()

        # Calculate collective intelligence
        collective_intelligence = self.network_analyzer.calculate_collective_intelligence(
            list(self.ecosystem_nodes.values())
        )

        # Calculate network coherence
        network_coherence = self.network_analyzer.calculate_network_coherence(
            list(self.ecosystem_nodes.values())
        )

        # Assess planetary health
        planetary_health = self._calculate_planetary_health()

        # Assess climate stability
        climate_stability = self.climate_monitor.assess_climate_stability()

        # Calculate regenerative capacity
        regenerative_capacity = self.regeneration_engine.calculate_regenerative_capacity(
            list(self.ecosystem_nodes.values())
        )

        # Create planetary consciousness state
        planetary_state = PlanetaryConsciousnessState(
            global_awareness=global_awareness,
            ecosystem_distribution=ecosystem_distribution,
            consciousness_hotspots=consciousness_hotspots,
            environmental_stress_indicators=environmental_stress,
            collective_intelligence=collective_intelligence,
            network_coherence=network_coherence,
            timestamp=datetime.now(),
            planetary_health=planetary_health,
            climate_stability=climate_stability,
            regenerative_capacity=regenerative_capacity
        )

        # Add to history
        self.consciousness_history.append(planetary_state)
        if len(self.consciousness_history) > 100:
            self.consciousness_history.pop(0)

        logger.info(f"Planetary consciousness assessed: Awareness {global_awareness:.3f}, Health {planetary_health:.3f}")

        return planetary_state

    def _identify_consciousness_hotspots(self) -> List[Dict[str, Any]]:
        """Identify areas of high consciousness activity"""
        if not self.ecosystem_nodes:
            return []

        # Find nodes with consciousness level above threshold
        threshold = 0.7
        hotspots = []

        for node in self.ecosystem_nodes.values():
            if node.consciousness_level >= threshold:
                hotspots.append({
                    'node_id': node.id,
                    'ecosystem_type': node.ecosystem_type.value,
                    'consciousness_level': node.consciousness_level,
                    'location': node.location,
                    'biodiversity': node.biodiversity_index,
                    'connectivity': node.connectivity_score
                })

        # Sort by consciousness level
        hotspots.sort(key=lambda x: x['consciousness_level'], reverse=True)

        return hotspots[:10]  # Top 10 hotspots

    def _assess_environmental_stress(self) -> Dict[str, float]:
        """Assess environmental stress indicators across the planetary network"""
        if not self.ecosystem_nodes:
            return {}

        stress_indicators = {
            'average_health_decline': 0.0,
            'connectivity_degradation': 0.0,
            'biodiversity_loss': 0.0,
            'communication_disruption': 0.0
        }

        health_levels = [node.health_status for node in self.ecosystem_nodes.values()]
        connectivity_scores = [node.connectivity_score for node in self.ecosystem_nodes.values()]
        biodiversity_indices = [node.biodiversity_index for node in self.ecosystem_nodes.values()]

        if health_levels:
            stress_indicators['average_health_decline'] = 1.0 - np.mean(health_levels)

        if connectivity_scores:
            stress_indicators['connectivity_degradation'] = 1.0 - np.mean(connectivity_scores)

        if biodiversity_indices:
            stress_indicators['biodiversity_loss'] = 1.0 - np.mean(biodiversity_indices)

        # Assess communication disruption
        disrupted_nodes = sum(1 for node in self.ecosystem_nodes.values()
                             if not node.communication_signals)
        stress_indicators['communication_disruption'] = disrupted_nodes / len(self.ecosystem_nodes)

        return stress_indicators

    def _calculate_planetary_health(self) -> float:
        """Calculate overall planetary health based on ecosystem nodes"""
        if not self.ecosystem_nodes:
            return 0.0

        # Weighted average of health indicators
        health_scores = []
        weights = []

        for node in self.ecosystem_nodes.values():
            # Combine multiple health indicators
            combined_health = (
                node.health_status * 0.4 +
                node.biodiversity_index * 0.3 +
                node.connectivity_score * 0.3
            )
            health_scores.append(combined_health)
            weights.append(1.0)  # Equal weights for now

        if not health_scores:
            return 0.0

        weighted_health = sum(score * weight for score, weight in zip(health_scores, weights))
        total_weight = sum(weights)

        return weighted_health / total_weight if total_weight > 0 else 0.0

    def _create_empty_state(self) -> PlanetaryConsciousnessState:
        """Create an empty planetary consciousness state"""
        return PlanetaryConsciousnessState(
            global_awareness=0.0,
            ecosystem_distribution={},
            consciousness_hotspots=[],
            environmental_stress_indicators={},
            collective_intelligence=0.0,
            network_coherence=0.0,
            timestamp=datetime.now(),
            planetary_health=0.0,
            climate_stability=0.0,
            regenerative_capacity=0.0
        )

    def connect_to_wood_wide_web(self) -> Dict[str, Any]:
        """Connect to the Wood Wide Web for plant communication integration"""
        return self.wood_wide_web_interface.connect_to_network()

    def get_planetary_insights(self, time_window_seconds: int = 86400) -> Dict[str, Any]:
        """Get insights from recent planetary consciousness assessments"""
        if not self.consciousness_history:
            return {'insights': 'No planetary consciousness history'}

        # Filter recent assessments
        now = datetime.now()
        cutoff_time = datetime.fromtimestamp(now.timestamp() - time_window_seconds)

        recent_assessments = [
            state for state in self.consciousness_history
            if state.timestamp >= cutoff_time
        ]

        if not recent_assessments:
            return {'insights': 'No recent planetary assessments'}

        # Calculate trends
        if len(recent_assessments) < 2:
            trend = 'insufficient_data'
        else:
            first = recent_assessments[0]
            last = recent_assessments[-1]

            if last.global_awareness > first.global_awareness + 0.05:
                trend = 'increasing'
            elif last.global_awareness < first.global_awareness - 0.05:
                trend = 'decreasing'
            else:
                trend = 'stable'

        # Calculate statistics
        avg_awareness = np.mean([state.global_awareness for state in recent_assessments])
        avg_health = np.mean([state.planetary_health for state in recent_assessments])
        avg_coherence = np.mean([state.network_coherence for state in recent_assessments])

        # Identify most represented ecosystems
        ecosystem_representation = {}
        for state in recent_assessments:
            for ecosystem, proportion in state.ecosystem_distribution.items():
                if ecosystem not in ecosystem_representation:
                    ecosystem_representation[ecosystem] = []
                ecosystem_representation[ecosystem].append(proportion)

        avg_ecosystem_distribution = {
            ecosystem: np.mean(proportions)
            for ecosystem, proportions in ecosystem_representation.items()
        }

        # Find dominant ecosystem
        dominant_ecosystem = max(avg_ecosystem_distribution.items(),
                               key=lambda x: x[1]) if avg_ecosystem_distribution else (None, 0.0)

        return {
            'assessment_count': len(recent_assessments),
            'awareness_trend': trend,
            'average_global_awareness': avg_awareness,
            'average_planetary_health': avg_health,
            'average_network_coherence': avg_coherence,
            'dominant_ecosystem': dominant_ecosystem[0].value if dominant_ecosystem[0] else None,
            'ecosystem_distribution': {k.value: v for k, v in avg_ecosystem_distribution.items()},
            'consciousness_hotspots': self._get_recent_hotspots(recent_assessments),
            'environmental_stress': self._aggregate_stress_indicators(recent_assessments)
        }

    def _get_recent_hotspots(self, assessments: List[PlanetaryConsciousnessState]) -> List[Dict[str, Any]]:
        """Get recent consciousness hotspots"""
        if not assessments:
            return []

        # Get hotspots from most recent assessment
        recent_state = assessments[-1]
        return recent_state.consciousness_hotspots

    def _aggregate_stress_indicators(self, assessments: List[PlanetaryConsciousnessState]) -> Dict[str, float]:
        """Aggregate environmental stress indicators"""
        if not assessments:
            return {}

        # Average all stress indicators
        aggregated = {}
        indicator_names = assessments[0].environmental_stress_indicators.keys()

        for indicator in indicator_names:
            values = [state.environmental_stress_indicators.get(indicator, 0.0)
                     for state in assessments]
            aggregated[indicator] = np.mean(values)

        return aggregated

    def trigger_regenerative_protocol(self, target_ecosystems: Optional[List[EcosystemType]] = None) -> Dict[str, Any]:
        """Trigger regenerative protocols for ecosystem restoration"""
        if not target_ecosystems:
            # Target all ecosystems with low health
            target_ecosystems = [
                node.ecosystem_type for node in self.ecosystem_nodes.values()
                if node.health_status < 0.5
            ]
            # Remove duplicates
            target_ecosystems = list(set(target_ecosystems))

        regeneration_results = self.regeneration_engine.initiate_regeneration(
            target_ecosystems, list(self.ecosystem_nodes.values())
        )

        return regeneration_results


__all__ = ['PlanetaryEcosystemConsciousnessNetwork']
