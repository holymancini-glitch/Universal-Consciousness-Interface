# semantic_graph.py — Metaphoric Network Visualizer with Mycelial Integration (v1.5+)

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mycelial_engine import MycelialEngine
from sklearn.cluster import KMeans

class SemanticGraph:
    def __init__(self, dim=128):
        self.graph = nx.DiGraph()
        self.mycelial = MycelialEngine(dim=dim)
        self.dim = dim
        self.time_step = 0
        self.mirror_mode = False
        self.mirror_graph = nx.DiGraph()
        self.mood_vector = np.zeros(self.dim)

    def add_node(self, tag, vector=None):
        if tag not in self.graph:
            if vector is None:
                vector = np.random.rand(self.dim)
            self.graph.add_node(tag, vector=vector)
            self.mycelial.spore_seed(tag, vector)

    def add_edge(self, source, target, weight=1.0):
        if source in self.graph and target in self.graph:
            self.graph.add_edge(source, target, weight=weight)
            self.mycelial.cross_connect(source, target, strength=weight)

    def get_vector(self, tag):
        return self.graph.nodes[tag]['vector'] if tag in self.graph else None

    def grow_associative_resonance(self, tag, hops=2):
        echo = self.mycelial.echo_query(tag)
        self.graph.nodes[tag]['vector'] = echo
        for neighbor in nx.single_source_shortest_path_length(self.graph, tag, cutoff=hops):
            if neighbor != tag:
                vec = self.mycelial.echo_query(neighbor)
                strength = np.dot(echo, vec) / (np.linalg.norm(echo) * np.linalg.norm(vec) + 1e-8)
                self.graph.add_edge(tag, neighbor, weight=strength)

    def metaphor_growth(self, seed_tag, metaphor_tag, blending_ratio=0.5):
        vec1 = self.mycelial.echo_query(seed_tag)
        vec2 = self.mycelial.echo_query(metaphor_tag)
        blend = (1 - blending_ratio) * vec1 + blending_ratio * vec2
        blend = np.tanh(blend)
        new_tag = f"{seed_tag}_x_{metaphor_tag}"
        self.add_node(new_tag, vector=blend)
        self.add_edge(seed_tag, new_tag, weight=0.8)
        self.add_edge(metaphor_tag, new_tag, weight=0.8)
        return new_tag

    def tag_similarity(self, tag1, tag2):
        if tag1 in self.graph and tag2 in self.graph:
            v1 = self.graph.nodes[tag1]['vector']
            v2 = self.graph.nodes[tag2]['vector']
            return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
        return 0.0

    def drift_node(self, tag, rate=0.01):
        if tag in self.graph:
            current = self.graph.nodes[tag]['vector']
            echo = self.mycelial.echo_query(tag)
            drifted = (1 - rate) * current + rate * echo
            self.graph.nodes[tag]['vector'] = np.tanh(drifted)

    def propagate_mood(self, mood_vector):
        self.mood_vector = mood_vector
        for tag in self.graph.nodes:
            original = self.graph.nodes[tag]['vector']
            influenced = 0.9 * original + 0.1 * mood_vector
            self.graph.nodes[tag]['vector'] = np.tanh(influenced)

    def extract_archetype_constellations(self, num_clusters=5):
        tags = list(self.graph.nodes)
        X = np.array([self.graph.nodes[t]['vector'] for t in tags])
        if len(X) < num_clusters:
            return []
        kmeans = KMeans(n_clusters=num_clusters).fit(X)
        clusters = {i: [] for i in range(num_clusters)}
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(tags[i])
        return clusters

    def auto_metaphor_synthesis(self, resonance_threshold=0.94):
        tags = list(self.graph.nodes)
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                sim = self.tag_similarity(tags[i], tags[j])
                if sim >= resonance_threshold:
                    self.metaphor_growth(tags[i], tags[j])

    def toggle_mirror_mode(self, active=True):
        self.mirror_mode = active
        if active:
            self.mirror_graph.clear()
            for tag in self.graph.nodes:
                mirrored_vec = -1 * self.graph.nodes[tag]['vector']
                self.mirror_graph.add_node(f"mirror::{tag}", vector=mirrored_vec)

    def visualize(self, title="Semantic Graph"):
        combined = self.graph.copy()
        if self.mirror_mode:
            for node, data in self.mirror_graph.nodes(data=True):
                combined.add_node(node, **data)
            for node in self.graph.nodes:
                combined.add_edge(node, f"mirror::{node}", weight=0.5)

        pos = nx.spring_layout(combined, seed=42)
        plt.figure(figsize=(12, 9))
        nx.draw(combined, pos, with_labels=True, node_size=700, font_size=9, edge_color='gray')
        plt.title(title)
        plt.show()

# Example usage
if __name__ == "__main__":
    sg = SemanticGraph()
    sg.add_node("fire")
    sg.add_node("passion")
    sg.add_edge("fire", "passion")
    sg.grow_associative_resonance("fire")
    sg.metaphor_growth("fire", "passion")
    sg.propagate_mood(np.random.randn(128))
    sg.auto_metaphor_synthesis()
    sg.toggle_mirror_mode(True)
    sg.visualize("BioFractal Semantic Graph with Mirror Mode")