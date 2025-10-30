"""
Network Processor for Mycelium Language Generator

Manages network topology, signal generation, and network-based linguistic patterns.
"""

import random
from collections import deque
from datetime import datetime
from typing import Dict, List, Any, Deque

from .data_models import MyceliumSignal, MyceliumCommunicationType


class NetworkProcessor:
    """
    Processes mycelium network topology and generates communication signals.

    Manages:
    - 3D network topology with fractal properties
    - Signal generation and tracking
    - Network-based linguistic pattern determination
    - Topology-driven syntactic structures
    """

    def __init__(self, network_size: int = 1000):
        """
        Initialize network processor.

        Args:
            network_size: Number of nodes in the mycelium network
        """
        self.network_size = network_size
        self.network_topology: Dict[str, Any] = self._initialize_network_topology()
        self.active_signals: Deque[MyceliumSignal] = deque(maxlen=1000)

    def _initialize_network_topology(self) -> Dict[str, Any]:
        """
        Initialize 3D mycelium network topology.

        Creates a network with properties typical of mycelial networks:
        - Small-world topology (local clustering + long-range connections)
        - Fractal dimension between 2.3-2.8
        - High clustering coefficient
        - Dynamic growth patterns

        Returns:
            Dictionary describing network topology characteristics
        """
        return {
            'nodes': self.network_size,
            'connection_density': random.uniform(0.3, 0.8),
            'clustering_coefficient': random.uniform(0.6, 0.9),
            'small_world_index': random.uniform(0.7, 0.95),
            'fractal_dimension': random.uniform(2.3, 2.8),
            'growth_rate': random.uniform(0.1, 0.5)
        }

    def generate_sample_signals(self, count: int = 10) -> List[MyceliumSignal]:
        """
        Generate sample mycelium signals for demonstration or testing.

        Creates a balanced mix of:
        - Chemical gradient signals (50%)
        - Electrical pulse signals (30%)
        - Network resonance signals (20%)

        Args:
            count: Number of signals to generate

        Returns:
            List of MyceliumSignal objects
        """
        signals = []

        # Calculate signal distribution
        chemical_count = int(count * 0.5)
        electrical_count = int(count * 0.3)
        resonance_count = count - chemical_count - electrical_count

        # Chemical gradient signals
        for _ in range(chemical_count):
            signal = MyceliumSignal(
                signal_type=MyceliumCommunicationType.CHEMICAL_GRADIENT,
                intensity=random.uniform(0.3, 0.9),
                duration=random.uniform(1.0, 5.0),
                spatial_pattern=random.choice(['radial', 'directional', 'diffuse']),
                chemical_composition={
                    'melanin': random.uniform(0.5, 0.9),
                    'chitin': random.uniform(0.2, 0.7),
                    'enzyme_complex': random.uniform(0.1, 0.6)
                },
                electrical_frequency=random.uniform(0.1, 2.0),
                timestamp=datetime.now(),
                network_location=(
                    random.uniform(-10, 10),
                    random.uniform(-10, 10),
                    random.uniform(-2, 2)
                )
            )
            signals.append(signal)

        # Electrical pulse signals
        for _ in range(electrical_count):
            signal = MyceliumSignal(
                signal_type=MyceliumCommunicationType.ELECTRICAL_PULSE,
                intensity=random.uniform(0.4, 1.0),
                duration=random.uniform(0.5, 2.0),
                spatial_pattern='network_wide',
                chemical_composition={},
                electrical_frequency=random.uniform(2.0, 10.0),
                timestamp=datetime.now(),
                network_location=(
                    random.uniform(-5, 5),
                    random.uniform(-5, 5),
                    random.uniform(-1, 1)
                )
            )
            signals.append(signal)

        # Network resonance signals
        for _ in range(resonance_count):
            signal = MyceliumSignal(
                signal_type=MyceliumCommunicationType.NETWORK_RESONANCE,
                intensity=random.uniform(0.6, 1.0),
                duration=random.uniform(3.0, 8.0),
                spatial_pattern='collective_resonance',
                chemical_composition={
                    'neurotransmitter': random.uniform(0.3, 0.8),
                    'muscimol': random.uniform(0.1, 0.4)  # Amanita muscaria compound
                },
                electrical_frequency=random.uniform(5.0, 15.0),
                timestamp=datetime.now(),
                network_location=(0.0, 0.0, 0.0)  # Network center
            )
            signals.append(signal)

        return signals

    def add_signal(self, signal: MyceliumSignal) -> None:
        """
        Add a signal to the active signals buffer.

        Args:
            signal: MyceliumSignal to add
        """
        self.active_signals.append(signal)

    def get_topology_metrics(self) -> Dict[str, float]:
        """
        Get current network topology metrics.

        Returns:
            Dictionary of topology metrics
        """
        return {
            'nodes': self.network_topology['nodes'],
            'connection_density': self.network_topology['connection_density'],
            'clustering': self.network_topology['clustering_coefficient'],
            'small_world_index': self.network_topology['small_world_index'],
            'fractal_dimension': self.network_topology['fractal_dimension'],
            'growth_rate': self.network_topology['growth_rate']
        }

    def determine_word_order(self) -> str:
        """
        Determine word order based on network topology.

        Uses clustering coefficient to select linguistic pattern:
        - High clustering (>0.8): hub-spoke-peripheral (central concept first)
        - Medium clustering (>0.6): source-pathway-destination (flow-based)
        - Low clustering: gradient-diffusion-response (chemical gradient)

        Returns:
            Word order pattern string
        """
        clustering = self.network_topology.get('clustering_coefficient', 0.5)

        if clustering > 0.8:
            return 'hub-spoke-peripheral'  # Central concept first
        elif clustering > 0.6:
            return 'source-pathway-destination'  # Flow-based order
        else:
            return 'gradient-diffusion-response'  # Chemical gradient order

    def determine_phrase_structure(self) -> str:
        """
        Determine phrase structure from network patterns.

        Uses fractal dimension to select structure type:
        - High fractal (>2.7): recursive-branching (self-similar)
        - Medium fractal (>2.4): hierarchical-clustering (nested)
        - Low fractal: linear-flow (simple sequence)

        Returns:
            Phrase structure pattern string
        """
        fractal_dim = self.network_topology.get('fractal_dimension', 2.5)

        if fractal_dim > 2.7:
            return 'recursive-branching'  # Self-similar structure
        elif fractal_dim > 2.4:
            return 'hierarchical-clustering'  # Nested structure
        else:
            return 'linear-flow'  # Simple sequence

    def determine_temporal_flow(self) -> str:
        """
        Determine temporal flow patterns from network growth.

        Uses growth rate to select temporal pattern:
        - Fast growth (>0.4): rapid-burst-communication
        - Medium growth (>0.2): steady-flow-rhythm
        - Slow growth: slow-accumulation-release

        Returns:
            Temporal flow pattern string
        """
        growth_rate = self.network_topology.get('growth_rate', 0.3)

        if growth_rate > 0.4:
            return 'rapid-burst-communication'
        elif growth_rate > 0.2:
            return 'steady-flow-rhythm'
        else:
            return 'slow-accumulation-release'

    def analyze_network_connectivity(self) -> Dict[str, Any]:
        """
        Analyze network connectivity and communication efficiency.

        Returns:
            Dictionary with connectivity analysis metrics
        """
        return {
            'total_nodes': self.network_topology['nodes'],
            'connection_density': self.network_topology['connection_density'],
            'small_world_property': self.network_topology['small_world_index'] > 0.7,
            'fractal_complexity': self.network_topology['fractal_dimension'],
            'communication_efficiency': (
                self.network_topology['small_world_index'] *
                self.network_topology['clustering_coefficient']
            ),
            'active_signals': len(self.active_signals)
        }


__all__ = ['NetworkProcessor']
