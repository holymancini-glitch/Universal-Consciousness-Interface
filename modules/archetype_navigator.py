# archetype_navigator.py — Symbolic Dream Agent & Archetypal Pathing (v0.5+ Full Enhancements)

import random
import time
import numpy as np

class ArchetypeNavigator:
    def __init__(self, self_model=None, phantom_layer=None, semantic_graph=None, mycelial_engine=None, emotional_feedback_loop=None):
        self.self_model = self_model
        self.phantom_layer = phantom_layer
        self.semantic_graph = semantic_graph
        self.mycelial_engine = mycelial_engine
        self.emotional_feedback_loop = emotional_feedback_loop
        self.path_history = []
        self.persona_mode = None

    def assign_persona_mode(self, mode):
        """
        Sets an archetypal bias mode. Affects vector blending and tag priority.
        Possible modes: "sage", "child", "trickster", "shadow", "healer"
        """
        self.persona_mode = mode

    def explore(self, depth=3, tag_filter=None):
        if not self.self_model:
            return []
        log = self.self_model.get_archetype_log(tag_filter=tag_filter)
        if not log:
            return []

        salient_log = self._rank_by_salience(log)
        path = []
        current_vector = self.self_model.get_self_state()

        for _ in range(depth):
            imprint = self._weighted_sample(salient_log)
            vector = np.array(imprint['vector'])
            blend_ratio = self._persona_blend_ratio()
            blended = np.tanh(blend_ratio * current_vector + (1 - blend_ratio) * vector)
            current_vector = blended

            emotional_state = self.emotional_feedback_loop.get_state() if self.emotional_feedback_loop else {}

            step = {
                'timestamp': time.time(),
                'archetype_tags': imprint['tags'],
                'vector': blended.tolist(),
                'source': imprint['source'],
                'emotional_weight': imprint.get('emotional_amplitude', 1.0),
                'persona_mode': self.persona_mode,
                'emotion_context': emotional_state.get('mood', None)
            }

            if self.semantic_graph:
                self.semantic_graph.activate_node_by_vector(blended, tags=imprint['tags'], layer=self.persona_mode)

            if self.phantom_layer:
                phantom = self.phantom_layer.generate(seed=blended, depth=1)
                step['phantom_echo'] = phantom
                step['phantom_emotion'] = phantom[0]['emotion_vector'] if phantom else []

            if self.mycelial_engine:
                trail = self.mycelial_engine.trace(tags=imprint['tags'])
                step['mycelial_trace'] = trail

            path.append(step)

        self.path_history.append(path)
        return path

    def _rank_by_salience(self, log):
        for entry in log:
            amplitude = np.linalg.norm(entry.get('vector', []))
            entry['emotional_amplitude'] = amplitude
        return sorted(log, key=lambda x: x['emotional_amplitude'], reverse=True)

    def _weighted_sample(self, log):
        weights = [entry['emotional_amplitude'] for entry in log]
        total = sum(weights)
        if total == 0:
            return random.choice(log)
        probabilities = [w / total for w in weights]
        return random.choices(log, weights=probabilities, k=1)[0]

    def _persona_blend_ratio(self):
        """Affect blending behavior based on persona mode."""
        if self.persona_mode == "trickster":
            return random.uniform(0.3, 0.5)
        elif self.persona_mode == "sage":
            return 0.7
        elif self.persona_mode == "child":
            return 0.4
        elif self.persona_mode == "shadow":
            return 0.6
        elif self.persona_mode == "healer":
            return 0.65
        return 0.5  # neutral

    def last_path(self):
        return self.path_history[-1] if self.path_history else []

    def switch_persona_if_needed(self, mood):
        """
        Optional: dynamically update persona based on emotional context.
        """
        mood_to_persona = {
            "curious": "child",
            "introspective": "sage",
            "anxious": "shadow",
            "playful": "trickster",
            "soothing": "healer"
        }
        if mood in mood_to_persona:
            self.assign_persona_mode(mood_to_persona[mood])

# Example usage
if __name__ == "__main__":
    nav = ArchetypeNavigator()
    nav.assign_persona_mode("sage")
    journey = nav.explore(depth=3)
    for step in journey:
        print("Symbolic Step:", step)