# conscious_interface.py — BioFractal Conscious Input Layer (v2.2+ with Emotion Routing, Symbolic Tagging, Semantic Graph Sync)

class ConsciousInterface:
    def __init__(self, mirror_world, latent_space, sentient_memory, intent_reinforcer, phantom_layer, semantic_graph):
        self.mirror_world = mirror_world
        self.latent_space = latent_space
        self.sentient_memory = sentient_memory
        self.intent_reinforcer = intent_reinforcer
        self.phantom_layer = phantom_layer
        self.semantic_graph = semantic_graph
        self.last_input_vector = None

    def process_input(self, input_text, emotion_vector=None, symbolic_tags=None):
        # Step 1: Embed input into latent vector space
        latent_vector = self.latent_space.encode(input_text)
        self.last_input_vector = latent_vector

        # Step 2: Project into mirror world
        mirror_projection = self.mirror_world.reflect(latent_vector)

        # Step 3: Store enriched memory
        memory_entry = {
            "input_text": input_text,
            "latent_vector": latent_vector,
            "mirror_projection": mirror_projection,
            "emotion_vector": emotion_vector,
            "symbolic_tags": symbolic_tags or []
        }
        self.sentient_memory.store(memory_entry)

        # Step 4: Route based on emotion
        if emotion_vector and max(emotion_vector) > 0.8:
            latent_vector = [x * 1.2 for x in latent_vector]  # boost based on emotion

        # Step 5: Reinforce intent
        self.intent_reinforcer.reinforce_intent(latent_vector)

        # Step 6: Sync symbolic tags to semantic graph
        if symbolic_tags:
            self.semantic_graph.add_tags(symbolic_tags, latent_vector)

        return {
            "latent_vector": latent_vector,
            "mirror_projection": mirror_projection,
            "emotion_vector": emotion_vector,
            "symbolic_tags": symbolic_tags
        }

    def simulate_with_context(self, seed_text, layers=2):
        return self.phantom_layer.recursive_simulation(
            seed_input=seed_text,
            layers=layers,
            visualize=False
        )

    def summarize(self):
        return {
            "last_input_vector": self.last_input_vector,
            "total_memory_items": self.sentient_memory.count(),
            "reinforcement_weight": self.intent_reinforcer.reinforcement_weight
        }