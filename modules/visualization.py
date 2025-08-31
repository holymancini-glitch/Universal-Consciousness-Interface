# visualization.py — Multimodal Resonance Visualizer (v2.5+)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import seaborn as sns
import networkx as nx
from sklearn.decomposition import PCA

class Visualizer:
    def __init__(self):
        sns.set(style="whitegrid")
        self.persona_colors = {
            'Sage': 'blue',
            'Child': 'orange',
            'Trickster': 'red',
            'Shadow': 'black',
            'Healer': 'green'
        }

    def plot_emotional_wave(self, emotion_sequence, title="Emotional Wave", persona_sequence=None):
        if not emotion_sequence:
            print("No emotion data to visualize.")
            return

        emotion_array = np.array(emotion_sequence)
        plt.figure(figsize=(10, 4))
        for i in range(emotion_array.shape[1]):
            color = self.persona_colors.get(persona_sequence[i] if persona_sequence else 'Sage', 'gray')
            plt.plot(emotion_array[:, i], label=f"E{i}", color=color, alpha=0.8)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Emotion Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_attention_map(self, attention_matrix, tokens=None, title="Attention Map"):
        plt.figure(figsize=(8, 6))
        sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=tokens, cmap="coolwarm", annot=True)
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_latent_projection(self, vectors, labels=None, title="Latent Space Projection", overlay_vectors=None):
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(vectors)
        plt.figure(figsize=(6, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], c='cyan', alpha=0.6, label='Core')

        if overlay_vectors is not None:
            overlay_reduced = pca.transform(overlay_vectors)
            plt.scatter(overlay_reduced[:, 0], overlay_reduced[:, 1], c='magenta', alpha=0.3, label='Overlay')

        if labels:
            for i, label in enumerate(labels):
                plt.text(reduced[i, 0], reduced[i, 1], label, fontsize=9)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_semantic_graph(self, graph_data, title="Semantic Graph", persona_tags=None):
        G = nx.DiGraph()
        for node, edges in graph_data.items():
            G.add_node(node)
            for target in edges:
                G.add_edge(node, target)
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        node_colors = [self.persona_colors.get(persona_tags.get(n, 'Sage'), 'gray') if persona_tags else 'lightblue' for n in G.nodes()]
        nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color="gray", node_size=800, font_size=10)
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_mood_index(self, mood_sequence, title="Mood Index Over Time"):
        if not mood_sequence:
            print("No mood data to visualize.")
            return
        plt.figure(figsize=(8, 3))
        plt.plot(mood_sequence, marker='o', linestyle='-', color='purple')
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Mood Index")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def animate_dream_path(self, dream_sequence):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Dream Projection Path")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        xdata, ydata = [], []
        ln, = plt.plot([], [], 'ro-', alpha=0.5)

        def init():
            ln.set_data([], [])
            return ln,

        def update(frame):
            vec = np.array(frame['content']['signature'])[:2]  # 2D slice
            xdata.append(vec[0])
            ydata.append(vec[1])
            ln.set_data(xdata, ydata)
            return ln,

        ani = animation.FuncAnimation(fig, update, frames=dream_sequence, init_func=init, blit=True, repeat=False)
        plt.show()

# Example usage
if __name__ == "__main__":
    viz = Visualizer()
    emotion_seq = [
        [0.1, 0.3, 0.5],
        [0.2, 0.4, 0.6],
        [0.1, 0.5, 0.7]
    ]
    persona_seq = ['Sage', 'Child', 'Shadow']
    viz.plot_emotional_wave(emotion_seq, persona_sequence=persona_seq)
