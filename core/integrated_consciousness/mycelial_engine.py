"""
Mycelial Engine for Integrated Consciousness System

This module provides mycelial network processing with graph algorithms for
experience storage and intelligent connection formation.
"""

import logging
import math
import networkx as nx
from collections import deque
from datetime import datetime
from typing import Dict, List, Any, Optional


logger = logging.getLogger(__name__)


class EnhancedMycelialEngine:
    """
    Enhanced mycelial network with fractal integration.

    Manages a directed graph of experiences with intelligent connection
    formation based on semantic similarity. Automatically prunes weak
    connections to maintain optimal network size.

    Attributes:
        max_nodes: Maximum number of nodes in the network
        graph: NetworkX directed graph of experiences
        experiences: Dictionary of experience data
        connection_strength_threshold: Minimum similarity for connections
        growth_history: History of network growth
    """

    def __init__(self, max_nodes: int = 2000):
        """
        Initialize mycelial engine.

        Args:
            max_nodes: Maximum number of nodes in the network (default: 2000)
        """
        self.max_nodes = max_nodes
        self.graph = nx.DiGraph()
        self.experiences: Dict[int, Dict[str, Any]] = {}
        self.connection_strength_threshold = 0.3
        self.growth_history = deque(maxlen=500)

        logger.info(f"Enhanced MycelialEngine initialized with {max_nodes} max nodes")

    def add_experience(self, experience_id: int, experience_data: Dict[str, Any]):
        """
        Add experience with enhanced metadata.

        Automatically adds timestamp, access tracking, and creates intelligent
        connections to related experiences. Prunes network if size exceeds limit.

        Args:
            experience_id: Unique identifier for the experience
            experience_data: Dictionary containing experience data
        """
        experience_data.update({
            'timestamp': datetime.now(),
            'access_count': 0,
            'last_accessed': datetime.now(),
            'strength': experience_data.get('strength', 0.5)
        })

        self.experiences[experience_id] = experience_data
        self.graph.add_node(experience_id, **experience_data)

        # Connect to related experiences
        self._create_intelligent_connections(experience_id, experience_data)

        # Maintain graph size
        if len(self.graph.nodes) > self.max_nodes:
            self._prune_weak_connections()

    def _create_intelligent_connections(self, new_id: int, new_data: Dict[str, Any]):
        """
        Create intelligent connections based on semantic similarity.

        Compares new experience with all existing experiences and creates
        bidirectional edges when similarity exceeds threshold.

        Args:
            new_id: ID of the new experience
            new_data: Data of the new experience
        """
        for existing_id, existing_data in self.experiences.items():
            if existing_id != new_id:
                similarity = self._calculate_semantic_similarity(new_data, existing_data)

                if similarity > self.connection_strength_threshold:
                    self.graph.add_edge(new_id, existing_id, weight=similarity)
                    self.graph.add_edge(existing_id, new_id, weight=similarity)

    def _calculate_semantic_similarity(self, data1: Dict[str, Any], data2: Dict[str, Any]) -> float:
        """
        Calculate semantic similarity between experiences.

        Uses cosine similarity on extracted numerical features.

        Args:
            data1: First experience data
            data2: Second experience data

        Returns:
            Similarity score (0.0-1.0)
        """
        # Extract numerical features
        features1 = self._extract_numerical_features(data1)
        features2 = self._extract_numerical_features(data2)

        if not features1 or not features2:
            return 0.0

        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(features1, features2))
        magnitude1 = math.sqrt(sum(a * a for a in features1))
        magnitude2 = math.sqrt(sum(b * b for b in features2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _extract_numerical_features(self, data: Dict[str, Any]) -> List[float]:
        """
        Extract numerical features from experience data.

        Converts strings to numerical features using hashing.

        Args:
            data: Experience data dictionary

        Returns:
            List of numerical features
        """
        features = []
        for key, value in data.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                # Simple string hash to numerical feature
                features.append(float(hash(value) % 1000) / 1000.0)
        return features

    def _prune_weak_connections(self):
        """
        Remove weakest connections and nodes.

        Removes bottom 10% of edges by weight and any resulting isolated nodes.
        Maintains network efficiency by eliminating weak connections.
        """
        # Remove edges with lowest weights
        edges_by_weight = sorted(self.graph.edges(data=True),
                               key=lambda x: x[2].get('weight', 0))

        # Remove bottom 10% of edges
        num_to_remove = max(1, len(edges_by_weight) // 10)
        for i in range(num_to_remove):
            edge = edges_by_weight[i]
            self.graph.remove_edge(edge[0], edge[1])

        # Remove isolated nodes
        isolated_nodes = list(nx.isolates(self.graph))
        for node in isolated_nodes:
            if node in self.experiences:
                del self.experiences[node]
            self.graph.remove_node(node)

    def get_experience(self, experience_id: int) -> Optional[Dict[str, Any]]:
        """
        Get experience and update access statistics.

        Increments access count and updates last accessed timestamp.

        Args:
            experience_id: Identifier of the experience

        Returns:
            Experience data dictionary if found, None otherwise
        """
        if experience_id in self.experiences:
            experience = self.experiences[experience_id]
            experience['access_count'] += 1
            experience['last_accessed'] = datetime.now()
            return experience
        return None


__all__ = ['EnhancedMycelialEngine']
