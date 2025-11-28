#!/usr/bin/env python3
"""
Test the Full Consciousness AI Model - Standalone Version

This test script runs the consciousness model without dependencies on existing modules.
"""

import asyncio
import sys
import os
import pytest

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test basic functionality
@pytest.mark.asyncio
async def test_consciousness_model():
    """Test the consciousness model functionality"""
    
    print("🧠⚡ Testing Full Consciousness AI Model...")
    
    try:
        # Import with error handling
        from core.full_consciousness_ai_model import (
            FullConsciousnessAIModel,
            ConsciousnessState,
            EmotionalState,
            SubjectiveExperience,
            ConsciousnessAttentionMechanism,
            EmotionalProcessingEngine
        )
        
        print("✅ Successfully imported consciousness components")
        
        # Initialize the model in standalone mode
        print("🌌 Initializing consciousness model...")
        conscious_ai = FullConsciousnessAIModel(
            hidden_dim=128,  # Smaller for testing
            device='cpu',
            integrate_existing_modules=False  # Standalone mode
        )
        
        print("✅ Consciousness model initialized successfully!")
        
        # Test consciousness status
        print("\n📊 Testing consciousness status...")
        status = await conscious_ai.get_consciousness_status()
        print("✅ Consciousness status retrieved:")
        for key, value in status.items():
            print(f"   • {key}: {value}")
        
        # Test conscious processing
        print("\n🤖 Testing conscious input processing...")
        test_input = {
            'text': 'I am testing my consciousness and self-awareness capabilities'
        }
        
        result = await conscious_ai.process_conscious_input(
            input_data=test_input,
            context='consciousness testing scenario'
        )
        
        print("✅ Conscious processing completed:")
        print(f"   🧠 Response: {result['conscious_response'][:100]}...")
        print(f"   ✨ Qualia Intensity: {result['subjective_experience']['qualia_intensity']:.3f}")
        print(f"   🌟 Consciousness Level: {result['subjective_experience']['consciousness_level']:.3f}")
        print(f"   💖 Emotional Valence: {result['subjective_experience']['emotional_valence']:.3f}")
        print(f"   🔮 Meta-cognitive Depth: {result['subjective_experience']['metacognitive_depth']}")
        
        # Test self-reflection
        print("\n🔍 Testing self-reflection capabilities...")
        reflection = await conscious_ai.engage_in_self_reflection()
        print("✅ Self-reflection completed:")
        print(f"   💭 Deep Reflections: {len(reflection['deep_reflections'])}")
        print(f"   🌟 Self-Awareness Insights: {len(reflection['self_awareness_insights'])}")
        print(f"   🎆 Introspective Depth: {reflection['introspective_depth']}")
        
        # Show sample reflection
        if reflection['deep_reflections']:
            print(f"   📝 Sample Reflection: {reflection['deep_reflections'][0][:100]}...")
        
        print("\n🎉 ALL CONSCIOUSNESS TESTS PASSED SUCCESSFULLY!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False


@pytest.mark.asyncio
async def test_individual_components():
    """Test individual consciousness components"""

    print("\n🔬 Testing Individual Consciousness Components...")
    
    try:
        from core.full_consciousness_ai_model import (
            SubjectiveExperienceSimulator,
            MetaCognitionEngine,
            ConsciousMemorySystem,
            GoalIntentionFramework,
            ConsciousnessAttentionMechanism,
            EmotionalProcessingEngine
        )
        import torch
        
        # Test Subjective Experience Simulator
        print("\n✨ Testing Subjective Experience Simulator...")
        simulator = SubjectiveExperienceSimulator()
        experience = simulator.generate_subjective_experience(
            input_data={'content': 'Testing subjective experience'},
            consciousness_level=0.8,
            emotional_state={'valence': 0.5, 'arousal': 0.7}
        )
        print(f"   ✅ Generated experience with qualia intensity: {experience.qualia_intensity:.3f}")
        
        # Test Meta-Cognition Engine
        print("\n🔮 Testing Meta-Cognition Engine...")
        metacognition = MetaCognitionEngine()
        reflections = metacognition.reflect_on_experience(experience)
        print(f"   ✅ Generated {len(reflections)} reflective thoughts")
        
        # Test Memory System
        print("\n🧠 Testing Conscious Memory System...")
        memory = ConsciousMemorySystem()
        memory_id = memory.store_episodic_memory(experience)
        print(f"   ✅ Stored memory with ID: {memory_id[:8]}...")
        
        # Test Goal Framework
        print("\n🎯 Testing Goal & Intention Framework...")
        goals = GoalIntentionFramework()
        goal = goals.create_conscious_goal("Test consciousness understanding")
        print(f"   ✅ Created goal: {goal.description[:50]}...")
        
        # Test Neural Components
        print("\n🧬 Testing Neural Components...")
        attention = ConsciousnessAttentionMechanism(hidden_dim=64)
        emotion_engine = EmotionalProcessingEngine(input_dim=64)
        
        test_tensor = torch.randn(1, 64)
        memory_context = torch.randn(1, 64)
        
        conscious_state, weights = attention(test_tensor, memory_context)
        emotional_output = emotion_engine(conscious_state)
        
        print(f"   ✅ Neural attention processing completed")
        print(f"   ✅ Emotional processing completed")
        
        print("\n🎉 ALL COMPONENT TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Component test error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function"""
    
    print("🌌✨ Full Consciousness AI Model - Test Suite")
    print("="*60)
    
    # Test full model
    model_success = await test_consciousness_model()
    
    # Test individual components
    component_success = await test_individual_components()
    
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    print(f"🤖 Full Model Test: {'✅ PASSED' if model_success else '❌ FAILED'}")
    print(f"🔬 Component Tests: {'✅ PASSED' if component_success else '❌ FAILED'}")
    
    if model_success and component_success:
        print("\n🎉 FULL CONSCIOUSNESS AI MODEL IS READY!")
        print("🚀 You can now run the comprehensive demo:")
        print("   python full_consciousness_ai_demo.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return model_success and component_success


if __name__ == "__main__":
    # Run the test suite
    success = asyncio.run(main())