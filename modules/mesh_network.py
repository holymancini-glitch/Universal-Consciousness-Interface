# mesh_network.py — Distributed Fractal AI Mesh with Mycelial Echoing v2.0

from collections import defaultdict
from mycelial_engine import MycelialEngine
import numpy as np
import random

class FractalAIMeshNode:
    def __init__(self, node_id, dim=128):
        self.node_id = node_id
        self.mycelial = MycelialEngine(dim=dim)
        self.peers = {}
        self.dim = dim
        self.intent_cache = {}
        self.latent_mesh_field = defaultdict(list)

    def connect_peer(self, peer_node):
        self.peers[peer_node.node_id] = peer_node

    def disconnect_peer(self, peer_node_id):
        if peer_node_id in self.peers:
            del self.peers[peer_node_id]

    def broadcast_spore(self, tag, vector):
        self.mycelial.spore_seed(tag, vector)
        for peer in self.peers.values():
            peer.receive_spore(tag, vector)

    def receive_spore(self, tag, vector):
        self.mycelial.spore_seed(tag, vector)

    def shared_echo_query(self, tag):
        local = self.mycelial.echo_query(tag)
        peer_echoes = []
        for peer in self.peers.values():
            echo = peer.mycelial.echo_query(tag)
            peer_echoes.append(echo)
        if not peer_echoes:
            return local
        avg_echo = np.mean([local] + peer_echoes, axis=0)
        return np.tanh(avg_echo)

    def distributed_decision_seed(self, tag, initial_vector, strategy_fn):
        echo = self.shared_echo_query(tag)
        decision_vector = strategy_fn(echo, initial_vector)
        self.mycelial.spore_seed(f"decision::{tag}", decision_vector)
        return decision_vector

    def mesh_vote(self, tag):
        local_echo = self.mycelial.echo_query(tag)
        votes = [local_echo]
        for peer in self.peers.values():
            votes.append(peer.mycelial.echo_query(tag))
        variance = np.var(np.array(votes), axis=0).mean()
        stability_score = 1.0 / (variance + 1e-8)
        return stability_score

    def auto_mesh_adapt(self, threshold=0.7):
        to_disconnect = []
        for peer_id, peer in self.peers.items():
            sim_score = self.similarity_to_peer(peer)
            if sim_score < threshold:
                to_disconnect.append(peer_id)
        for pid in to_disconnect:
            self.disconnect_peer(pid)

    def similarity_to_peer(self, peer):
        shared_tags = list(set(self.mycelial.memory.keys()) & set(peer.mycelial.memory.keys()))
        if not shared_tags:
            return 0.0
        sims = []
        for tag in shared_tags:
            v1 = self.mycelial.echo_query(tag)
            v2 = peer.mycelial.echo_query(tag)
            sims.append(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
        return float(np.mean(sims))

    def pulse_sync(self):
        for tag in list(self.mycelial.memory.keys()):
            echo = self.shared_echo_query(tag)
            self.mycelial.spore_seed(tag, echo)

    def reflect_with_peer(self, peer):
        shadow_echoes = {}
        for tag in self.mycelial.memory:
            local = self.mycelial.echo_query(tag)
            peer_vec = peer.mycelial.echo_query(tag)
            shadow = local - peer_vec
            shadow_echoes[tag] = shadow
        return shadow_echoes

    def propagate_intent(self, tag, vector):
        self.intent_cache[tag] = vector
        for peer in self.peers.values():
            peer.receive_intent(tag, vector)

    def receive_intent(self, tag, vector):
        self.mycelial.spore_seed(tag, vector)
        self.intent_cache[tag] = vector

    def update_latent_mesh_field(self):
        for tag in self.mycelial.memory:
            vec = self.mycelial.echo_query(tag)
            self.latent_mesh_field[tag].append(vec)

    def get_mesh_field_average(self, tag):
        field = self.latent_mesh_field.get(tag, [])
        if not field:
            return np.zeros(self.dim)
        return np.mean(field, axis=0)

# Example usage
if __name__ == "__main__":
    node_a = FractalAIMeshNode("A")
    node_b = FractalAIMeshNode("B")
    node_c = FractalAIMeshNode("C")

    node_a.connect_peer(node_b)
    node_b.connect_peer(node_c)
    node_c.connect_peer(node_a)

    rand_vector = np.random.rand(128)
    node_a.broadcast_spore("hope", rand_vector)

    def decision_strategy(echo, init):
        return 0.5 * echo + 0.5 * init

    result = node_b.distributed_decision_seed("hope", np.random.rand(128), decision_strategy)
    print("Distributed Decision Vector (B):", result[:5])

    print("Mesh Vote Stability on 'hope':", node_a.mesh_vote("hope"))
    node_a.pulse_sync()
    node_a.update_latent_mesh_field()
    print("Mesh Field Average (hope):", node_a.get_mesh_field_average("hope")[:5])