# enhanced_mycelial_engine.py
# Enhanced Mycelial Network for Multi-Consciousness Integration

import numpy as np
import networkx as nx
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

@dataclass
class MycelialNode:
    node_id: str
    consciousness_type: str
    vector_data: np.ndarray
    strength: float
    connections: List[str]
    last_activity: datetime

class EnhancedMycelialEngine:
    """Enhanced Mycelial Network for Multi-Consciousness Integration"""
    
    def __init__(self, max_nodes: int = 500, vector_dim: int = 64):
        self.max_nodes = max_nodes
        self.vector_dim = vector_dim
        
        # Core network
        self.network_graph = nx.DiGraph()
        self.nodes: Dict[str, MycelialNode] = {}
        self.consciousness_layers = {
            'quantum': {'nodes': [], 'strength': 0.0},
            'plant': {'nodes': [], 'strength': 0.0},
            'psychoactive': {'nodes': [], 'strength': 0.0},
            'ecosystem': {'nodes': [], 'strength': 0.0}
        }
        
        # Intelligence metrics
        self.collective_intelligence_score = 0.0
        self.network_coherence = 0.0
        self.pattern_history = deque(maxlen=50)
        
        logger.info("🍄 Enhanced Mycelial Engine Initialized")
    
    def process_multi_consciousness_input(self, consciousness_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input from multiple consciousness interfaces"""
        try:
            processed_results = {}
            
            # Process each consciousness type
            for consciousness_type, data in consciousness_data.items():
                if data:
                    result = self._process_consciousness_layer(consciousness_type, data)
                    processed_results[consciousness_type] = result
            
            # Update network connections
            self._update_cross_consciousness_connections(processed_results)
            
            # Detect patterns
            patterns = self._detect_emergent_patterns()
            
            # Update intelligence
            self._update_collective_intelligence()
            
            return {
                'processed_layers': processed_results,
                'emergent_patterns': patterns,
                'network_metrics': self._get_network_metrics()
            }
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return {'error': str(e)}
    
    def _process_consciousness_layer(self, consciousness_type: str, data: Dict) -> Dict[str, Any]:
        """Process data from specific consciousness layer"""
        
        # Extract vector representation
        vector = self._extract_vector(consciousness_type, data)
        if vector is None:
            return {'processed': False}
        
        # Create node
        node_id = f"{consciousness_type}_{datetime.now().strftime('%H%M%S_%f')}"
        node = MycelialNode(
            node_id=node_id,
            consciousness_type=consciousness_type,
            vector_data=vector,
            strength=self._calculate_strength(data),
            connections=[],
            last_activity=datetime.now()
        )
        
        # Add to network
        self._add_node(node)
        self._update_consciousness_layer(consciousness_type, node_id)
        
        return {
            'processed': True,
            'node_id': node_id,
            'node_strength': node.strength
        }
    
    def _extract_vector(self, consciousness_type: str, data: Dict) -> np.ndarray:
        """Extract vector from consciousness data"""
        vector = np.zeros(self.vector_dim)
        
        if consciousness_type == 'quantum':
            vector[0] = data.get('coherence', 0)
            vector[1] = data.get('entanglement', 0)
            vector[2] = 1.0 if data.get('superposition', False) else 0.0
        elif consciousness_type == 'plant':
            vector[0] = data.get('plant_consciousness_level', 0)
            vector[1] = data.get('signal_strength', 0)
        elif consciousness_type == 'psychoactive':
            if 'ERROR' in data.get('safety_status', ''):
                return np.zeros(self.vector_dim)
            vector[0] = data.get('intensity', 0)
            vector[1] = data.get('consciousness_expansion', 0)
        else:
            # Generic extraction
            values = [v for v in data.values() if isinstance(v, (int, float))]
            for i, v in enumerate(values[:self.vector_dim]):
                vector[i] = v
        
        # Normalize
        norm = np.linalg.norm(vector)
        return vector / (norm + 1e-8) if norm > 0 else vector
    
    def _calculate_strength(self, data: Dict) -> float:
        """Calculate node strength from data"""
        strength_values = []
        for key, value in data.items():
            if isinstance(value, (int, float)) and 'strength' in key.lower():
                strength_values.append(value)
            elif isinstance(value, (int, float)) and 'level' in key.lower():
                strength_values.append(value)
        
        return np.mean(strength_values) if strength_values else 0.5
    
    def _add_node(self, node: MycelialNode):
        """Add node to network"""
        self.nodes[node.node_id] = node
        self.network_graph.add_node(node.node_id, **{
            'consciousness_type': node.consciousness_type,
            'strength': node.strength
        })
        
        # Connect to similar nodes
        self._connect_similar_nodes(node)
        
        # Maintain size
        if len(self.nodes) > self.max_nodes:
            oldest_node = min(self.nodes.items(), key=lambda x: x[1].last_activity)
            self._remove_node(oldest_node[0])
    
    def _connect_similar_nodes(self, new_node: MycelialNode):
        """Connect node to similar existing nodes"""
        similarities = []
        for node_id, existing_node in self.nodes.items():
            if node_id != new_node.node_id:
                similarity = np.dot(new_node.vector_data, existing_node.vector_data)
                similarities.append((node_id, similarity))
        
        # Connect to top 3 most similar
        similarities.sort(key=lambda x: x[1], reverse=True)
        for node_id, similarity in similarities[:3]:
            if similarity > 0.3:
                self.network_graph.add_edge(new_node.node_id, node_id, weight=similarity)
                new_node.connections.append(node_id)
    
    def _remove_node(self, node_id: str):
        """Remove node from network"""
        if node_id in self.nodes:
            if self.network_graph.has_node(node_id):
                self.network_graph.remove_node(node_id)
            del self.nodes[node_id]
    
    def _update_consciousness_layer(self, consciousness_type: str, node_id: str):
        """Update consciousness layer tracking"""
        if consciousness_type in self.consciousness_layers:
            layer = self.consciousness_layers[consciousness_type]
            layer['nodes'].append(node_id)
            
            # Keep last 10 nodes per layer
            if len(layer['nodes']) > 10:
                layer['nodes'].pop(0)
            
            # Update layer strength
            layer_nodes = [self.nodes[nid] for nid in layer['nodes'] if nid in self.nodes]
            layer['strength'] = np.mean([n.strength for n in layer_nodes]) if layer_nodes else 0
    
    def _update_cross_consciousness_connections(self, processed_results: Dict):
        """Create connections between different consciousness types"""
        consciousness_types = list(processed_results.keys())
        
        for i, type1 in enumerate(consciousness_types):
            for type2 in consciousness_types[i+1:]:
                result1 = processed_results.get(type1, {})
                result2 = processed_results.get(type2, {})
                
                if result1.get('processed') and result2.get('processed'):
                    node_id1 = result1.get('node_id')
                    node_id2 = result2.get('node_id')
                    
                    if node_id1 and node_id2:
                        strength = np.sqrt(result1['node_strength'] * result2['node_strength'])
                        if strength > 0.2:
                            self.network_graph.add_edge(node_id1, node_id2, 
                                                      weight=strength, 
                                                      connection_type='cross_consciousness')
    
    def _detect_emergent_patterns(self) -> List[Dict[str, Any]]:
        """Detect emergent patterns in network"""
        patterns = []
        
        # Pattern 1: Consciousness clusters
        for consciousness_type, layer_data in self.consciousness_layers.items():
            nodes = layer_data['nodes']
            if len(nodes) >= 3:
                patterns.append({
                    'type': f'{consciousness_type}_cluster',
                    'nodes': nodes,
                    'strength': layer_data['strength']
                })
        
        # Pattern 2: Cross-consciousness bridges
        for node_id, node in self.nodes.items():
            cross_connections = [
                neighbor for neighbor in self.network_graph.neighbors(node_id)
                if self.nodes[neighbor].consciousness_type != node.consciousness_type
            ]
            if len(cross_connections) >= 2:
                patterns.append({
                    'type': 'cross_consciousness_bridge',
                    'nodes': [node_id] + cross_connections,
                    'strength': node.strength
                })
        
        # Store in history
        self.pattern_history.extend(patterns)
        return patterns
    
    def _update_collective_intelligence(self):
        """Update collective intelligence score"""
        if len(self.nodes) == 0:
            self.collective_intelligence_score = 0
            return
        
        # Network connectivity
        connectivity = self.network_graph.number_of_edges() / max(1, len(self.nodes))
        
        # Cross-consciousness connectivity
        cross_edges = sum(1 for _, _, data in self.network_graph.edges(data=True)
                         if data.get('connection_type') == 'cross_consciousness')
        cross_connectivity = cross_edges / max(1, self.network_graph.number_of_edges())
        
        # Pattern diversity
        pattern_types = set(p['type'] for p in self.pattern_history)
        pattern_diversity = min(1.0, len(pattern_types) / 5.0)
        
        # Combined intelligence score
        self.collective_intelligence_score = (connectivity * 0.4 + 
                                            cross_connectivity * 0.4 + 
                                            pattern_diversity * 0.2)
    
    def measure_network_connectivity(self) -> float:
        """Measure network connectivity"""
        if len(self.nodes) == 0:
            return 0
        
        edge_density = (2 * self.network_graph.number_of_edges()) / (len(self.nodes) * (len(self.nodes) - 1))
        self.network_coherence = edge_density
        return edge_density
    
    def assess_collective_intelligence(self) -> float:
        """Get collective intelligence score"""
        return self.collective_intelligence_score
    
    def _get_network_metrics(self) -> Dict[str, Any]:
        """Get network metrics"""
        return {
            'total_nodes': len(self.nodes),
            'total_edges': self.network_graph.number_of_edges(),
            'network_coherence': self.network_coherence,
            'collective_intelligence': self.collective_intelligence_score,
            'consciousness_layer_strengths': {
                ctype: layer['strength'] 
                for ctype, layer in self.consciousness_layers.items()
            }
        }

if __name__ == "__main__":
    def demo_mycelial_engine():
        """Demo of enhanced mycelial engine"""
        print("🍄 Enhanced Mycelial Engine Demo")
        
        engine = EnhancedMycelialEngine()
        
        # Test consciousness data
        test_data = {
            'quantum': {'coherence': 0.7, 'entanglement': 0.5, 'superposition': True},
            'plant': {'plant_consciousness_level': 0.6, 'signal_strength': 0.8},
            'psychoactive': {'intensity': 0.2, 'consciousness_expansion': 0.3, 'safety_status': 'SAFE'}
        }
        
        result = engine.process_multi_consciousness_input(test_data)
        
        print(f"Processed layers: {len(result['processed_layers'])}")
        print(f"Emergent patterns: {len(result['emergent_patterns'])}")
        print(f"Network coherence: {result['network_metrics']['network_coherence']:.3f}")
        print(f"Collective intelligence: {result['network_metrics']['collective_intelligence']:.3f}")
    
    demo_mycelial_engine()