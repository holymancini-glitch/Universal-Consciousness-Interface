#!/usr/bin/env python3
"""
Simple test script to verify the Consciousness Fractal AI System is working correctly.
"""

import sys
import os
import asyncio

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing module imports...")
    
    try:
        from modules.consciousness_fractal_ai import ConsciousnessFractalAI
        print("✅ ConsciousnessFractalAI imported successfully")
    except Exception as e:
        print(f"❌ Failed to import ConsciousnessFractalAI: {e}")
        return False
    
    try:
        from modules.fractal_monte_carlo import FractalMonteCarlo
        print("✅ FractalMonteCarlo imported successfully")
    except Exception as e:
        print(f"❌ Failed to import FractalMonteCarlo: {e}")
        return False
    
    try:
        from modules.neural_ca import NeuralCA
        print("✅ NeuralCA imported successfully")
    except Exception as e:
        print(f"❌ Failed to import NeuralCA: {e}")
        return False
    
    try:
        from modules.latent_space import LatentSpace
        print("✅ LatentSpace imported successfully")
    except Exception as e:
        print(f"❌ Failed to import LatentSpace: {e}")
        return False
    
    try:
        from modules.fep_neural_model import FEPNeuralModel
        print("✅ FEPNeuralModel imported successfully")
    except Exception as e:
        print(f"❌ Failed to import FEPNeuralModel: {e}")
        return False
    
    try:
        from modules.fractal_ai_universal_integration import FractalAIUniversalIntegration
        print("✅ FractalAIUniversalIntegration imported successfully")
    except Exception as e:
        print(f"❌ Failed to import FractalAIUniversalIntegration: {e}")
        return False
    
    return True

def test_basic_system():
    """Test basic system initialization and operation."""
    print("\nTesting basic system functionality...")
    
    try:
        # Create a minimal configuration for testing
        config = {
            'system_name': 'TestConsciousnessFractalAI',
            'latent_space_shape': (4, 4, 2),
            'neural_ca_grid_size': 8,
            'neural_ca_latent_dim': 32,
            'fep_num_neurons': 100,
            'fep_input_dim': 32,
            'fep_output_dim': 16,
            'fractal_state_dim': 32,
            'fractal_action_dim': 16,
            'fractal_max_depth': 2,
            'fractal_num_samples': 3,
            'phase_vector_dim': 4,
            'device': 'cpu',
            'update_interval': 0.1
        }
        
        # Import the main class
        from modules.consciousness_fractal_ai import ConsciousnessFractalAI
        
        # Initialize the system
        consciousness_ai = ConsciousnessFractalAI(config)
        print("✅ ConsciousnessFractalAI initialized successfully")
        
        # Get system status
        status = consciousness_ai.get_system_status()
        print(f"✅ System status retrieved: {status['system_name']}")
        
        # Check that components are initialized
        assert consciousness_ai.fmc is not None, "FMC not initialized"
        assert consciousness_ai.neural_ca is not None, "NeuralCA not initialized"
        assert consciousness_ai.latent_space is not None, "LatentSpace not initialized"
        assert consciousness_ai.fep_model is not None, "FEPNeuralModel not initialized"
        print("✅ All system components initialized")
        
        # Run a single consciousness cycle
        async def test_cycle():
            await consciousness_ai._consciousness_cycle()
            return len(consciousness_ai.state_history) > 0
        
        # Run the async test
        result = asyncio.run(test_cycle())
        assert result, "Consciousness cycle failed to execute"
        print("✅ Consciousness cycle executed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test basic system functionality: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("Consciousness Fractal AI System - Simple Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed")
        return 1
    
    # Test basic system functionality
    if not test_basic_system():
        print("\n❌ Basic system tests failed")
        return 1
    
    print("\n✅ All tests passed! The Consciousness Fractal AI System is working correctly.")
    return 0

if __name__ == "__main__":
    sys.exit(main())