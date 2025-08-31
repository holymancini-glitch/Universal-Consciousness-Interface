"""
Basic Usage Example
Demonstrates how to use the Universal Consciousness Interface
"""

import asyncio
from core.universal_consciousness_orchestrator import UniversalConsciousnessOrchestrator

async def basic_example():
    """Basic example of using the Universal Consciousness Interface"""
    
    # Initialize the orchestrator
    orchestrator = UniversalConsciousnessOrchestrator(
        quantum_enabled=True,
        plant_interface_enabled=True,
        ecosystem_enabled=True,
        safety_mode="STRICT"
    )
    
    print("Universal Consciousness Interface - Basic Example")
    print("=" * 50)
    
    # Define sample consciousness data
    consciousness_data = {
        'plant': {
            'frequency': 25.0,
            'amplitude': 0.6,
            'pattern': 'PHOTOSYNTHETIC_HARMONY'
        },
        'ecosystem': {
            'temperature': 22.0,
            'humidity': 60.0,
            'biodiversity': 0.7
        },
        'quantum': {
            'coherence': 0.6,
            'entanglement': 0.5
        }
    }
    
    # Process a single consciousness cycle
    print("Processing consciousness cycle...")
    result = await orchestrator.consciousness_cycle(
        base_stimulus=None,
        consciousness_data=consciousness_data
    )
    
    print(f"Unified Consciousness Score: {result.unified_consciousness_score:.3f}")
    print(f"Safety Status: {result.safety_status}")
    print(f"Dimensional State: {result.dimensional_state}")
    
    if result.plant_communication.get('translated_message'):
        print(f"Plant Message: {result.plant_communication['translated_message']}")
    
    print("Basic example completed successfully!")

if __name__ == "__main__":
    asyncio.run(basic_example())