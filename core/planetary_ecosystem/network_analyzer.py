"""
Network Analyzer for Planetary Ecosystem Consciousness Network

Analyzes the collective intelligence and network coherence of the
planetary ecosystem consciousness network.
"""

import logging
from typing import List

# Handle optional dependencies with fallbacks
try:
    import numpy as np  # type: ignore
except ImportError:
    import statistics
    import math

    class MockNumPy:
        @staticmethod
        def mean(values):
            return statistics.mean(values) if values else 0.0

        @staticmethod
        def std(values):
            return statistics.stdev(values) if len(values) > 1 else 0.0

    np = MockNumPy()

from .data_models import EcosystemNode

logger = logging.getLogger(__name__)


class NetworkAnalyzer:
    """Analyzer for the planetary ecosystem consciousness network"""

    def __init__(self) -> None:
        logger.info("🔍 Network Analyzer Initialized")

    def calculate_collective_intelligence(self, nodes: List[EcosystemNode]) -> float:
        """Calculate the collective intelligence of the ecosystem network"""
        if not nodes:
            return 0.0

        # Collective intelligence based on:
        # 1. Average consciousness level
        # 2. Network connectivity
        # 3. Biodiversity
        # 4. Communication activity

        consciousness_levels = [node.consciousness_level for node in nodes]
        connectivity_scores = [node.connectivity_score for node in nodes]
        biodiversity_indices = [node.biodiversity_index for node in nodes]

        avg_consciousness = np.mean(consciousness_levels) if consciousness_levels else 0.0
        avg_connectivity = np.mean(connectivity_scores) if connectivity_scores else 0.0
        avg_biodiversity = np.mean(biodiversity_indices) if biodiversity_indices else 0.0

        # Weighted combination
        collective_intelligence = (
            avg_consciousness * 0.4 +
            avg_connectivity * 0.3 +
            avg_biodiversity * 0.3
        )

        return min(1.0, collective_intelligence)

    def calculate_network_coherence(self, nodes: List[EcosystemNode]) -> float:
        """Calculate the coherence of the ecosystem network"""
        if len(nodes) < 2:
            return 1.0  # Perfect coherence with single node

        # Coherence based on similarity of consciousness levels
        consciousness_levels = [node.consciousness_level for node in nodes]

        if not consciousness_levels:
            return 0.0

        # Calculate standard deviation as inverse of coherence
        std_dev = np.std(consciousness_levels)

        # Convert to coherence score (0-1)
        # Lower standard deviation = higher coherence
        max_expected_std = 0.5  # Maximum expected standard deviation
        coherence = max(0.0, 1.0 - (std_dev / max_expected_std))

        return coherence


__all__ = ['NetworkAnalyzer']
