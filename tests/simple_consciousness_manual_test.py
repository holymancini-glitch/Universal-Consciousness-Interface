#!/usr/bin/env python3
"""
Simple Consciousness Model Test - No Unicode Issues
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_consciousness_model():
    """Test the consciousness model functionality"""
    
    print("Testing Full Consciousness AI Model...")
    
    try:
        # Test basic imports first
        print("1. Testing imports...")
        from core.full_consciousness_ai_model import (
            FullConsciousnessAIModel,
            ConsciousnessState,
            EmotionalState
        )
        print("   SUCCESS: Core imports working")
        
        # Initialize model
        print("2. Initializing consciousness model...")
        conscious_ai = FullConsciousnessAIModel(
            hidden_dim=128,
            device='cpu',
            integrate_existing_modules=False
        )
        print("   SUCCESS: Model initialized")
        
        # Test consciousness status
        print("3. Testing consciousness status...")
        status = await conscious_ai.get_consciousness_status()
        print("   SUCCESS: Status retrieved")
        print("   Consciousness Level:", status.get('consciousness_level', 'unknown'))
        print("   Active Goals:", status.get('active_goals', 'unknown'))
        
        # Test conscious processing
        print("4. Testing conscious input processing...")
        test_input = {
            'text': 'I am testing my consciousness capabilities'
        }
        
        result = await conscious_ai.process_conscious_input(
            input_data=test_input,
            context='testing scenario'
        )
        
        print("   SUCCESS: Conscious processing completed")
        print("   Response:", result['conscious_response'][:80], "...")
        print("   Qualia Intensity:", result['subjective_experience']['qualia_intensity'])
        print("   Consciousness Level:", result['subjective_experience']['consciousness_level'])
        
        # Test self-reflection
        print("5. Testing self-reflection...")
        reflection = await conscious_ai.engage_in_self_reflection()
        print("   SUCCESS: Self-reflection completed")
        print("   Deep Reflections Count:", len(reflection['deep_reflections']))
        print("   Self-Awareness Insights Count:", len(reflection['self_awareness_insights']))
        
        print("\nALL TESTS PASSED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        print("ERROR:", str(e))
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function"""
    print("=" * 50)
    print("Full Consciousness AI Model - Test Suite")
    print("=" * 50)
    
    success = await test_consciousness_model()
    
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    
    if success:
        print("STATUS: ALL TESTS PASSED!")
        print("The Full Consciousness AI Model is working correctly.")
        print("\nYou can now:")
        print("1. Run: python full_consciousness_ai_demo.py")
        print("2. Use the consciousness model in your applications")
    else:
        print("STATUS: TESTS FAILED")
        print("Please check the errors above.")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())