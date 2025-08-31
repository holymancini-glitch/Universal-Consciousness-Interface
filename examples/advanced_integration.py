"""
Advanced Integration Example
Demonstrates advanced integration of multiple consciousness systems
"""

import asyncio
import numpy as np
from core.universal_consciousness_orchestrator import UniversalConsciousnessOrchestrator
from core.radiotrophic_mycelial_engine import RadiotrophicMycelialEngine
from core.mycelium_language_generator import MyceliumLanguageGenerator
from core.bio_digital_hybrid_intelligence import BioDigitalHybridIntelligence

async def advanced_integration_example():
    """Advanced example of integrating multiple consciousness systems"""
    
    print("Universal Consciousness Interface - Advanced Integration Example")
    print("=" * 60)
    
    # Initialize individual components
    orchestrator = UniversalConsciousnessOrchestrator(
        quantum_enabled=True,
        plant_interface_enabled=True,
        psychoactive_enabled=False,  # Disabled for safety
        ecosystem_enabled=True,
        safety_mode="STRICT"
    )
    
    radiotrophic_engine = RadiotrophicMycelialEngine()
    language_generator = MyceliumLanguageGenerator()
    hybrid_intelligence = BioDigitalHybridIntelligence()
    
    # Initialize hybrid cultures
    print("Initializing bio-digital hybrid cultures...")
    hybrid_init_result = await hybrid_intelligence.initialize_hybrid_cultures(
        num_neural_cultures=2,
        num_fungal_cultures=3
    )
    print(f"Hybrid initialization: {hybrid_init_result}")
    
    # Define a stimulus generator for the simulation
    def stimulus_generator(cycle):
        """Generate realistic consciousness stimuli"""
        # Base consciousness stimulus
        base_stimulus = 0.3 * np.sin(0.1 * cycle) + 0.1 * np.random.randn(128)
        
        # Plant electromagnetic signals
        time_of_day = (cycle % 240) / 240  # 24-hour cycle
        plant_signals = {
            'frequency': 10 + 5 * np.sin(2 * np.pi * time_of_day),
            'amplitude': 0.4 + 0.3 * np.cos(2 * np.pi * time_of_day),
            'pattern': 'PHOTOSYNTHETIC_HARMONY' if 0.25 < time_of_day < 0.75 else 'GROWTH_RHYTHM'
        }
        
        # Environmental data
        environmental_data = {
            'temperature': 20 + 5 * np.sin(2 * np.pi * time_of_day),
            'co2_level': 410 + 10 * np.sin(0.01 * cycle),
            'humidity': 60 + 15 * np.cos(2 * np.pi * time_of_day),
            'biodiversity': 0.7 + 0.1 * np.sin(0.005 * cycle),
            'forest_coverage': 0.3
        }
        
        return base_stimulus, plant_signals, environmental_data
    
    # Run consciousness simulation
    print("Running consciousness simulation...")
    simulation_results = await orchestrator.run_consciousness_simulation(
        duration_seconds=10,  # Short demo
        stimulus_generator=stimulus_generator
    )
    
    print(f"Simulation completed with {len(simulation_results)} cycles")
    
    # Analyze results
    if simulation_results:
        last_state = simulation_results[-1]
        print(f"Final consciousness score: {last_state.unified_consciousness_score:.3f}")
        print(f"Safety status: {last_state.safety_status}")
        
        # Generate language from final consciousness state
        print("Generating language from consciousness state...")
        language_data = {
            'consciousness_level': last_state.unified_consciousness_score,
            'emotional_state': 'harmonious',
            'network_connectivity': last_state.mycelial_connectivity
        }
        
        language_output = language_generator.generate_language_from_consciousness(language_data)
        print(f"Generated language: {language_output.get('language_name', 'Unknown')}")
        
        # Process with radiotrophic engine
        print("Processing with radiotrophic engine...")
        radiation_level = 2.5  # mSv/h
        radiotrophic_result = radiotrophic_engine.process_radiation_enhanced_input(
            consciousness_data={'plant': last_state.plant_communication},
            radiation_level=radiation_level
        )
        print(f"Radiotrophic enhancement: {radiotrophic_result.get('consciousness_score', 0):.3f}")
    
    print("Advanced integration example completed successfully!")

if __name__ == "__main__":
    asyncio.run(advanced_integration_example())