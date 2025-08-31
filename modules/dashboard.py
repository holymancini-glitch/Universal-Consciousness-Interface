# dashboard.py — Real-Time BioFractal AI Visualization Panel (Enhanced v2.9+)

import streamlit as st
from phantom_layer import PhantomLayer
from visualization import Visualizer
from sentient_memory import SentientMemory
from latent_space import LatentSpace
from mirror_world import MirrorWorld
from mycelial_engine import MycelialEngine
from holographic_memory import HolographicMemory
from intent_reinforcer import IntentReinforcer
from cohesion_layer import CohesionLayer
import json
import datetime

# Initialize components
latent_space = LatentSpace()
sentient_memory = SentientMemory()
mirror_world = MirrorWorld()
mycelial_engine = MycelialEngine()
holographic_memory = HolographicMemory()
visualizer = Visualizer()
cohesion_layer = CohesionLayer()

phantom_layer = PhantomLayer(
    latent_space=latent_space,
    sentient_memory=sentient_memory,
    mirror_world=mirror_world,
    mycelial_engine=mycelial_engine,
    holographic_memory=holographic_memory,
    visualizer=visualizer
)

intent_reinforcer = IntentReinforcer(
    self_model=None,
    sentient_memory=sentient_memory,
    visualizer=visualizer,
    cohesion_layer=cohesion_layer
)

# Streamlit dashboard UI
st.set_page_config(page_title="BioFractal AI Dashboard", layout="wide")
st.title("🌌 BioFractal AI — Dream & Cognition Visualizer")

# Parameters
depth = st.slider("Dream Depth", 1, 10, 3)
layers = st.slider("Recursive Layers", 1, 5, 2)
live_tracking = st.checkbox("Live Intent Tracking", value=False)
run_button = st.button("Generate Dream Sequence")

if live_tracking:
    intent_reinforcer.enable_live_tracking()
else:
    intent_reinforcer.disable_live_tracking()

if run_button:
    dream_trace = phantom_layer.recursive_simulation(layers=layers, visualize=True)
    st.success("Dream simulation completed.")

    # Export JSON
    if st.button("📤 Export Trace to JSON"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dream_trace_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(dream_trace, f, indent=2)
        st.success(f"Exported to {filename}")

    st.subheader("📈 Emotional Wave")
    visualizer.plot_emotional_wave([e['emotion_vector'] for e in dream_trace])

    st.subheader("🧠 Latent Projection")
    latent_vectors = [e['content']['signature'] for e in dream_trace if 'content' in e and 'signature' in e['content']]
    visualizer.plot_latent_projection(latent_vectors)

    st.subheader("🎭 Symbolic Graph & Personas")
    visualizer.plot_semantic_graph(tags=["dream", "phantom"])

    st.subheader("📊 Symbolic Drift & Archetype Density")
    symbolic_tags = [e.get('gru_lstm_feedback', {}).get('symbolic_tags', []) for e in dream_trace]
    visualizer.plot_symbolic_tag_density(symbolic_tags)

    st.subheader("🔁 Cause–Effect Correlation")
    correlation_data = [(e['timestamp'], e.get('gru_lstm_feedback', {}).get('state_classification', [])) for e in dream_trace]
    visualizer.plot_state_class_drift(correlation_data)

    st.subheader("🧬 System Cohesion Over Time")
    cohesion_scores = [cohesion_layer.evaluate(e) for e in dream_trace]
    visualizer.plot_cohesion_drift(cohesion_scores)

    st.subheader("🌀 Dream Trace Explorer")
    for i, event in enumerate(dream_trace):
        with st.expander(f"Dream Frame {i+1}"):
            st.json(event)
            if 'emotion_vector' in event:
                visualizer.plot_emotional_wave([event['emotion_vector']])
            if 'content' in event and 'signature' in event['content']:
                visualizer.plot_latent_projection([event['content']['signature']])
            if 'mycelial_trail' in event:
                st.caption("🌿 Mycelial Trail")
                visualizer.plot_mycelial_trail(event['mycelial_trail'])
            if 'persona_state' in event:
                st.caption("🎭 Archetypal Personas")
                visualizer.plot_persona_resonance(event['persona_state'])
            if 'gru_lstm_feedback' in event:
                st.caption("📍 GRU-LSTM Dream Feedback")
                visualizer.plot_symbolic_tag_density([event['gru_lstm_feedback']['symbolic_tags']])
                visualizer.plot_state_class_drift([(event['timestamp'], event['gru_lstm_feedback']['state_classification'])])