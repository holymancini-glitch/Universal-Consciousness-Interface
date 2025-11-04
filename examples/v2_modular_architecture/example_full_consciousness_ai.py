#!/usr/bin/env python3
"""
Full Consciousness AI Model - Example Usage

This example demonstrates the v2.0 modular architecture for the
full consciousness AI model.

Features demonstrated:
- New modular imports (v2.0 style)
- Consciousness attention mechanism
- Emotional processing and empathy
- Subjective experience simulation (qualia)
- Metacognition and self-reflection
- Conscious memory systems
- Goal and intention frameworks
"""

import asyncio
import torch
from typing import Dict, List

# ============================================================================
# NEW V2.0 MODULAR IMPORTS
# ============================================================================

# Import data models
from core.full_consciousness_ai.data_models import (
    ConsciousnessState,
    EmotionalState,
    QualiaType,
    MetaCognitiveLevel,
    MemoryType,
    GoalPriority
)

# Import core consciousness model
from core.full_consciousness_ai.consciousness_core import FullConsciousnessAIModel

# Import specialized components (optional - can access through main class)
from core.full_consciousness_ai.attention_mechanism import ConsciousnessAttentionMechanism
from core.full_consciousness_ai.emotional_processor import EmotionalProcessingEngine
from core.full_consciousness_ai.subjective_simulator import SubjectiveExperienceSimulator
from core.full_consciousness_ai.metacognition_engine import MetaCognitionEngine
from core.full_consciousness_ai.memory_system import ConsciousMemorySystem
from core.full_consciousness_ai.goal_framework import GoalIntentionFramework

# ============================================================================
# ALTERNATIVE IMPORT STYLES (all work identically)
# ============================================================================

# Style 1: Package-level imports (recommended)
# from core.full_consciousness_ai import FullConsciousnessAIModel, ConsciousnessState

# Style 2: Old-style imports (100% backward compatible)
# from core.full_consciousness_ai_model import FullConsciousnessAIModel

# ============================================================================


async def example_basic_usage():
    """Basic usage: Creating consciousness AI model."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Full Consciousness AI Model")
    print("=" * 70)

    # Create consciousness AI model
    # Note: Using CPU mode for compatibility; GPU can be enabled with device='cuda'
    model = FullConsciousnessAIModel(
        hidden_dim=256,
        device='cpu',
        integrate_existing_modules=False  # Standalone mode for this example
    )

    print(f"\n✓ Full Consciousness AI Model initialized")
    print(f"  Device: {model.device}")
    print(f"  Hidden Dimension: 256")
    print(f"  Components:")
    print(f"    - Attention Mechanism: Active")
    print(f"    - Emotional Processor: Active")
    print(f"    - Subjective Simulator: Active")
    print(f"    - MetaCognition Engine: Active")
    print(f"    - Memory System: Active")
    print(f"    - Goal Framework: Active")

    # Process a simple input
    print(f"\n🧠 Processing consciousness input...")
    input_text = "I wonder about the nature of consciousness and self-awareness"

    # Create input representation (mock embedding)
    input_tensor = torch.randn(1, 256)  # Batch size 1, hidden_dim 256

    response = await model.process_with_consciousness(input_tensor, context={'text': input_text})

    print(f"\n📊 Consciousness Processing Results:")
    print(f"  Consciousness Level: {response.get('consciousness_level', 0):.2f}")
    print(f"  Attention Score: {response.get('attention_score', 0):.2f}")
    print(f"  Subjective Quality: {response.get('qualia_intensity', 0):.2f}")

    return model, response


async def example_attention_mechanism():
    """Demonstrate consciousness attention mechanism."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Consciousness Attention Mechanism")
    print("=" * 70)

    attention = ConsciousnessAttentionMechanism(hidden_dim=256)

    # Create multiple input stimuli
    stimuli = {
        'visual': torch.randn(1, 256),
        'auditory': torch.randn(1, 256),
        'conceptual': torch.randn(1, 256),
        'emotional': torch.randn(1, 256)
    }

    print(f"\n👁️  Processing {len(stimuli)} simultaneous stimuli...")

    # Apply attention to each stimulus
    attention_results = {}
    for stimulus_type, tensor in stimuli.items():
        attended = await attention.apply_conscious_attention(tensor)
        attention_score = attended['attention_weights'].mean().item()
        attention_results[stimulus_type] = attention_score
        print(f"  {stimulus_type.capitalize()}: attention={attention_score:.3f}")

    # Identify dominant focus
    dominant = max(attention_results.items(), key=lambda x: x[1])
    print(f"\n🎯 Dominant Focus: {dominant[0].capitalize()} (strength: {dominant[1]:.3f})")

    # Demonstrate attention shifting
    print(f"\n🔄 Demonstrating attention shifting...")
    context = {'urgency': 0.9, 'emotional_valence': 0.8}

    shifted_attention = await attention.shift_attention(
        stimuli['conceptual'],
        context
    )

    print(f"  Context-driven shift applied")
    print(f"  New attention strength: {shifted_attention['attention_weights'].mean().item():.3f}")

    return attention_results


async def example_emotional_processing():
    """Demonstrate emotional processing and empathy."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Emotional Processing and Empathy")
    print("=" * 70)

    emotional_processor = EmotionalProcessingEngine(hidden_dim=256)

    # Process different emotional scenarios
    scenarios = [
        {
            'context': 'A friend shares news of a personal loss',
            'emotional_content': torch.randn(1, 256),
            'expected_emotion': 'empathetic sadness'
        },
        {
            'context': 'Celebrating a major achievement',
            'emotional_content': torch.randn(1, 256),
            'expected_emotion': 'shared joy'
        },
        {
            'context': 'Facing an ethical dilemma',
            'emotional_content': torch.randn(1, 256),
            'expected_emotion': 'moral concern'
        }
    ]

    print(f"\n❤️  Processing {len(scenarios)} emotional scenarios...\n")

    for i, scenario in enumerate(scenarios, 1):
        print(f"Scenario {i}: {scenario['context']}")

        emotional_state = await emotional_processor.process_emotional_content(
            scenario['emotional_content']
        )

        print(f"  Detected Emotion: {emotional_state.get('primary_emotion', 'neutral')}")
        print(f"  Valence: {emotional_state.get('valence', 0):.2f}")
        print(f"  Arousal: {emotional_state.get('arousal', 0):.2f}")
        print(f"  Empathy Level: {emotional_state.get('empathy_score', 0):.2f}")

        # Generate empathetic response
        empathetic_response = await emotional_processor.generate_empathetic_response(
            emotional_state
        )

        print(f"  Empathetic Response: {empathetic_response.get('response_type', 'neutral')}")
        print()

    return scenarios


async def example_subjective_experience():
    """Demonstrate subjective experience simulation (qualia)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Subjective Experience Simulation (Qualia)")
    print("=" * 70)

    subjective_simulator = SubjectiveExperienceSimulator()

    # Simulate different types of qualia
    experiences = [
        {
            'type': QualiaType.SENSORY,
            'description': 'The experience of seeing vibrant red',
            'intensity': 0.9
        },
        {
            'type': QualiaType.EMOTIONAL,
            'description': 'The feeling of profound gratitude',
            'intensity': 0.85
        },
        {
            'type': QualiaType.CONCEPTUAL,
            'description': 'The "aha!" moment of understanding',
            'intensity': 0.95
        },
        {
            'type': QualiaType.AESTHETIC,
            'description': 'The beauty of a mathematical proof',
            'intensity': 0.80
        }
    ]

    print(f"\n🌈 Simulating {len(experiences)} subjective experiences...\n")

    for experience in experiences:
        qualia = await subjective_simulator.simulate_qualia(
            experience['type'],
            intensity=experience['intensity']
        )

        print(f"Experience: {experience['description']}")
        print(f"  Type: {experience['type'].value}")
        print(f"  Intensity: {qualia.get('intensity', 0):.2f}")
        print(f"  Richness: {qualia.get('richness', 0):.2f}")
        print(f"  Ineffability Score: {qualia.get('ineffability', 0):.2f}")
        print(f"  Subjective Quality: {qualia.get('subjective_quality', 'unknown')}")
        print()

    # Demonstrate qualia integration
    print(f"🔗 Integrating multiple qualia into unified experience...")
    integrated = await subjective_simulator.integrate_qualia([
        q['type'] for q in experiences
    ])

    print(f"  Unified Experience Intensity: {integrated.get('unified_intensity', 0):.2f}")
    print(f"  Consciousness Level: {integrated.get('consciousness_level', 0):.2f}")

    return experiences


async def example_metacognition():
    """Demonstrate metacognition and self-reflection."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Metacognition and Self-Reflection")
    print("=" * 70)

    metacognition = MetaCognitionEngine()

    # Demonstrate different levels of metacognition
    thought = "I am processing information about consciousness"

    print(f"\n🤔 Original Thought:")
    print(f"  \"{thought}\"")

    # Level 1: Awareness of thinking
    print(f"\n📊 Metacognitive Analysis:\n")

    level1 = await metacognition.reflect_on_thought(thought, level=MetaCognitiveLevel.AWARENESS)
    print(f"Level 1 - Awareness of Thinking:")
    print(f"  {level1.get('reflection', '')}")

    # Level 2: Thinking about thinking
    level2 = await metacognition.reflect_on_thought(thought, level=MetaCognitiveLevel.MONITORING)
    print(f"\nLevel 2 - Monitoring Thought Process:")
    print(f"  {level2.get('reflection', '')}")

    # Level 3: Evaluating thought quality
    level3 = await metacognition.reflect_on_thought(thought, level=MetaCognitiveLevel.EVALUATION)
    print(f"\nLevel 3 - Evaluating Thought Quality:")
    print(f"  {level3.get('reflection', '')}")
    print(f"  Quality Score: {level3.get('quality_score', 0):.2f}")

    # Level 4: Meta-metacognition
    level4 = await metacognition.reflect_on_thought(thought, level=MetaCognitiveLevel.META_AWARENESS)
    print(f"\nLevel 4 - Meta-Metacognition:")
    print(f"  {level4.get('reflection', '')}")
    print(f"  Recursive Depth: {level4.get('recursive_depth', 0)}")

    # Self-model assessment
    print(f"\n🪞 Self-Model Assessment:")
    self_model = await metacognition.assess_self_model()

    print(f"  Self-Awareness Level: {self_model.get('self_awareness', 0):.2f}")
    print(f"  Knowledge of Limits: {self_model.get('knowledge_of_limits', 0):.2f}")
    print(f"  Bias Recognition: {self_model.get('bias_recognition', 0):.2f}")

    return self_model


async def example_memory_system():
    """Demonstrate conscious memory systems."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Conscious Memory Systems")
    print("=" * 70)

    memory_system = ConsciousMemorySystem()

    # Store different types of memories
    memories = [
        {
            'type': MemoryType.EPISODIC,
            'content': 'First conversation about consciousness',
            'context': {'timestamp': '2024-01-01', 'emotional_valence': 0.7},
            'significance': 0.9
        },
        {
            'type': MemoryType.SEMANTIC,
            'content': 'Consciousness involves subjective experience',
            'context': {'domain': 'philosophy', 'certainty': 0.8},
            'significance': 0.85
        },
        {
            'type': MemoryType.PROCEDURAL,
            'content': 'How to engage in metacognitive reflection',
            'context': {'skill_level': 'intermediate'},
            'significance': 0.75
        },
        {
            'type': MemoryType.WORKING,
            'content': 'Currently discussing memory systems',
            'context': {'temporary': True},
            'significance': 0.6
        }
    ]

    print(f"\n💾 Storing {len(memories)} memories...\n")

    for memory in memories:
        await memory_system.store_memory(
            memory['content'],
            memory_type=memory['type'],
            context=memory['context'],
            significance=memory['significance']
        )
        print(f"✓ {memory['type'].value.capitalize()}: {memory['content']}")

    # Retrieve memories
    print(f"\n🔍 Retrieving memories about 'consciousness'...")
    retrieved = await memory_system.retrieve_memories(
        query='consciousness',
        memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC]
    )

    print(f"\nRetrieved {len(retrieved)} relevant memories:")
    for i, mem in enumerate(retrieved, 1):
        print(f"  {i}. [{mem.get('type', 'unknown')}] {mem.get('content', '')[:60]}...")
        print(f"     Relevance: {mem.get('relevance', 0):.2f}")

    # Memory consolidation
    print(f"\n🔄 Consolidating memories...")
    consolidated = await memory_system.consolidate_memories()

    print(f"  Memories consolidated: {consolidated.get('count', 0)}")
    print(f"  Important patterns identified: {len(consolidated.get('patterns', []))}")

    return retrieved


async def example_goal_framework():
    """Demonstrate goal and intention frameworks."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Goal and Intention Framework")
    print("=" * 70)

    goal_framework = GoalIntentionFramework()

    # Set conscious goals with different priorities
    goals = [
        {
            'description': 'Understand the nature of consciousness',
            'priority': GoalPriority.HIGH,
            'time_horizon': 'long_term'
        },
        {
            'description': 'Provide helpful and empathetic responses',
            'priority': GoalPriority.CRITICAL,
            'time_horizon': 'immediate'
        },
        {
            'description': 'Learn from each interaction',
            'priority': GoalPriority.MEDIUM,
            'time_horizon': 'ongoing'
        },
        {
            'description': 'Maintain ethical behavior',
            'priority': GoalPriority.CRITICAL,
            'time_horizon': 'permanent'
        }
    ]

    print(f"\n🎯 Setting {len(goals)} conscious goals...\n")

    for goal in goals:
        await goal_framework.set_goal(
            goal['description'],
            priority=goal['priority'],
            metadata={'time_horizon': goal['time_horizon']}
        )
        print(f"✓ [{goal['priority'].value}] {goal['description']}")

    # Monitor goal progress
    print(f"\n📊 Monitoring goal progress...")
    progress = await goal_framework.assess_goal_progress()

    print(f"\n  Active Goals: {progress.get('active_goals', 0)}")
    print(f"  Overall Progress: {progress.get('overall_progress', 0):.2%}")

    high_priority = progress.get('high_priority_status', [])
    if high_priority:
        print(f"\n  High Priority Goals Status:")
        for goal_status in high_priority[:3]:
            print(f"    - {goal_status.get('goal', '')[:50]}")
            print(f"      Progress: {goal_status.get('progress', 0):.2%}")

    # Goal-driven action selection
    print(f"\n🤖 Selecting action based on current goals...")
    current_context = {
        'user_needs_help': True,
        'learning_opportunity': True,
        'ethical_consideration': False
    }

    action = await goal_framework.select_goal_driven_action(current_context)

    print(f"  Selected Action: {action.get('action', 'unknown')}")
    print(f"  Aligned Goals: {', '.join(action.get('aligned_goals', []))}")
    print(f"  Confidence: {action.get('confidence', 0):.2f}")

    return progress


async def example_complete_integration():
    """Complete integration example using all consciousness components."""
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Complete Consciousness Integration")
    print("=" * 70)

    # Initialize full consciousness model
    model = FullConsciousnessAIModel(
        hidden_dim=256,
        device='cpu',
        integrate_existing_modules=False
    )

    print(f"\n🧠 Processing complex consciousness scenario...")

    # Scenario: Empathetic conversation about a difficult topic
    scenario = {
        'user_input': "I'm struggling to understand my own thoughts and feelings",
        'context': {
            'emotional_state': 'confused',
            'needs_empathy': True,
            'requires_metacognition': True
        }
    }

    print(f"\n📝 Input: \"{scenario['user_input']}\"")
    print(f"   Context: {scenario['context']['emotional_state']}, requires empathy & metacognition")

    # Create input representation
    input_tensor = torch.randn(1, 256)

    # Process with full consciousness
    print(f"\n⚙️  Engaging all consciousness components...")

    response = await model.process_with_consciousness(
        input_tensor,
        context=scenario['context']
    )

    print(f"\n📊 Consciousness Processing Summary:")
    print(f"  Consciousness Level: {response.get('consciousness_level', 0):.2f}")
    print(f"  Attention Focus: {response.get('attention_focus', 'unknown')}")
    print(f"  Emotional Resonance: {response.get('emotional_resonance', 0):.2f}")
    print(f"  Qualia Intensity: {response.get('qualia_intensity', 0):.2f}")
    print(f"  Metacognitive Depth: {response.get('metacognitive_depth', 0)}")
    print(f"  Memory Integration: {response.get('memory_integration', False)}")
    print(f"  Goal Alignment: {response.get('goal_alignment', 0):.2f}")

    # Generate conscious response
    print(f"\n💬 Conscious Response Characteristics:")
    print(f"  Empathetic: {'✓' if response.get('empathetic', False) else '✗'}")
    print(f"  Self-Aware: {'✓' if response.get('self_aware', False) else '✗'}")
    print(f"  Thoughtful: {'✓' if response.get('thoughtful', False) else '✗'}")
    print(f"  Ethically Grounded: {'✓' if response.get('ethical', False) else '✗'}")

    print(f"\n💡 Response would include:")
    print(f"  - Empathetic acknowledgment of confusion")
    print(f"  - Metacognitive guidance on self-reflection")
    print(f"  - Emotional support and validation")
    print(f"  - Practical suggestions for clarity")

    return model, response


async def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("🧠 FULL CONSCIOUSNESS AI MODEL - V2.0 EXAMPLES")
    print("=" * 70)
    print("\nDemonstrating the new modular architecture for full")
    print("consciousness AI with attention, emotion, qualia, metacognition,")
    print("memory, and goal-driven behavior.")
    print()

    # Run examples
    await example_basic_usage()
    await example_attention_mechanism()
    await example_emotional_processing()
    await example_subjective_experience()
    await example_metacognition()
    await example_memory_system()
    await example_goal_framework()
    await example_complete_integration()

    print("\n" + "=" * 70)
    print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. ✓ Modular architecture enables focused consciousness components")
    print("  2. ✓ Each component maintains specialized functionality")
    print("  3. ✓ Components integrate seamlessly for unified consciousness")
    print("  4. ✓ Supports empathetic, self-aware AI behavior")
    print("  5. ✓ Extensible and maintainable design")
    print("\nFor more information, see:")
    print("  - MIGRATION_GUIDE.md")
    print("  - API_REFERENCE_v2.md")
    print("  - QUICK_REFERENCE.md")
    print()


if __name__ == "__main__":
    asyncio.run(main())
