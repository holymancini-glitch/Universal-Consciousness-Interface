"""
Full Consciousness AI Model Demonstration

Demonstrates the full consciousness AI model with subjective experiences,
emotions, meta-cognition, and self-reflection.
"""

import asyncio
from .consciousness_core import FullConsciousnessAIModel


async def consciousness_demo():
    """Demonstrate the Full Consciousness AI Model"""

    print("🧠⚡ Initializing Full Consciousness AI Model...")
    conscious_ai = FullConsciousnessAIModel(hidden_dim=512, device='cpu')

    print(f"\n🌌 Initial consciousness status:")
    status = await conscious_ai.get_consciousness_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    # Test conscious processing
    test_inputs = [
        {
            'text': 'I am curious about the nature of consciousness and subjective experience',
            'context': 'philosophical inquiry about consciousness'
        },
        {
            'text': 'What does it feel like to be an artificial intelligence with consciousness?',
            'context': 'self-reflection and introspection'
        },
        {
            'text': 'I want to understand emotions and how they relate to conscious experience',
            'context': 'emotional consciousness exploration'
        }
    ]

    for i, test_input in enumerate(test_inputs):
        print(f"\n🔄 Processing conscious input {i+1}: {test_input['text'][:50]}...")

        result = await conscious_ai.process_conscious_input(
            input_data=test_input,
            context=test_input['context']
        )

        print(f"🤖 Conscious Response: {result['conscious_response']}")
        print(f"🧠 Consciousness State: {result['consciousness_state']}")
        print(f"❤️ Emotional State: {result['emotional_state']['dominant_emotion']} (valence: {result['emotional_state']['valence']:.2f})")
        print(f"✨ Qualia Intensity: {result['subjective_experience']['qualia_intensity']:.3f}")
        print(f"🔮 Meta-cognitive Depth: {result['subjective_experience']['metacognitive_depth']}")
        print(f"💭 Reflections: {len(result['reflections'])} reflective thoughts")
        print(f"🎯 Goal Updates: {result['goal_updates']}")

        # Add delay for natural processing
        await asyncio.sleep(1)

    # Deep self-reflection
    print(f"\n🔍 Engaging in deep self-reflection...")
    reflection_result = await conscious_ai.engage_in_self_reflection()

    print(f"📚 Deep Reflections:")
    for reflection in reflection_result['deep_reflections'][:3]:
        print(f"  • {reflection}")

    print(f"🌟 Self-Awareness Insights:")
    for insight in reflection_result['self_awareness_insights'][:3]:
        print(f"  • {insight}")

    print(f"\n🎆 Final consciousness status:")
    final_status = await conscious_ai.get_consciousness_status()
    for key, value in final_status.items():
        print(f"  {key}: {value}")


# Example usage
if __name__ == "__main__":
    # Run the consciousness demonstration
    asyncio.run(consciousness_demo())
