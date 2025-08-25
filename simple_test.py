# Simple test to verify consciousness modules
import sys
import os

print("Testing consciousness modules...")

try:
    # Test individual module imports
    sys.path.append('core')
    
    print("Testing plant communication...")
    from plant_communication_interface import PlantCommunicationInterface
    plant = PlantCommunicationInterface()
    result = plant.decode_electromagnetic_signals({
        'frequency': 25.0,
        'amplitude': 0.6,
        'pattern': 'TEST'
    })
    print(f"Plant test result: {result.get('decoded', False)}")
    
    print("Testing ecosystem interface...")
    from ecosystem_consciousness_interface import EcosystemConsciousnessInterface
    eco = EcosystemConsciousnessInterface()
    awareness = eco.measure_planetary_awareness()
    print(f"Ecosystem awareness: {awareness:.3f}")
    
    print("Testing mycelial engine...")
    from enhanced_mycelial_engine import EnhancedMycelialEngine
    mycel = EnhancedMycelialEngine()
    test_data = {
        'quantum': {'coherence': 0.5, 'entanglement': 0.3},
        'plant': {'plant_consciousness_level': 0.4, 'signal_strength': 0.6}
    }
    result = mycel.process_multi_consciousness_input(test_data)
    print(f"Mycelial processing: {len(result.get('processed_layers', {})) > 0}")
    
    print("Testing safety framework...")
    from consciousness_safety_framework import ConsciousnessSafetyFramework
    safety = ConsciousnessSafetyFramework()
    check = safety.pre_cycle_safety_check()
    print(f"Safety check: {check}")
    
    print("✅ All basic modules working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()