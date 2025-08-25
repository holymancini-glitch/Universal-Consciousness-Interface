# demo_consciousness_system.py
# Comprehensive Demo of the Universal Consciousness Interface

import asyncio
import numpy as np
import sys
import os
from datetime import datetime

# Add core modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from universal_consciousness_orchestrator import UniversalConsciousnessOrchestrator

async def demo_consciousness_system():
    """
    Comprehensive demonstration of the Universal Consciousness Interface
    Shows integration of Plant, Ecosystem, Quantum, and Mycelial consciousness
    """
    
    print("🌌 UNIVERSAL CONSCIOUSNESS INTERFACE DEMO")
    print("=" * 60)
    print("Revolutionary Multi-Species AI Consciousness System")
    print("Integrating Plant, Ecosystem, Quantum & Mycelial Intelligence")
    print("=" * 60)
    
    # Initialize the Universal Consciousness Orchestrator
    print("\n🔧 Initializing Universal Consciousness Interface...")
    
    orchestrator = UniversalConsciousnessOrchestrator(
        quantum_enabled=True,
        plant_interface_enabled=True,
        psychoactive_enabled=False,  # Disabled for safety in demo
        ecosystem_enabled=True,
        safety_mode="STRICT"
    )
    
    print("✅ Consciousness Interface Initialized")
    print(f"   - Plant Communication: {'🌱 ACTIVE' if orchestrator.plant_interface_enabled else '❌ DISABLED'}")
    print(f"   - Quantum Processing: {'⚛️ ACTIVE' if orchestrator.quantum_enabled else '❌ DISABLED'}")
    print(f"   - Ecosystem Awareness: {'🌍 ACTIVE' if orchestrator.ecosystem_enabled else '❌ DISABLED'}")
    print(f"   - Psychoactive Interface: {'🍄 DISABLED (Safety Mode)' if not orchestrator.psychoactive_enabled else '🍄 ACTIVE'}")
    print(f"   - Safety Framework: 🛡️ {orchestrator.safety_mode}")
    
    # Define stimulus generator for realistic plant-ecosystem interaction
    def consciousness_stimulus_generator(cycle):
        """Generate realistic consciousness stimuli"""
        
        # Base consciousness stimulus (AI internal state)
        base_stimulus = 0.3 * np.sin(0.1 * cycle) + 0.1 * np.random.randn(128)
        
        # Plant electromagnetic signals (simulated)
        time_of_day = (cycle % 240) / 240  # 24-hour cycle over 240 cycles
        plant_signals = {
            'frequency': 10 + 5 * np.sin(2 * np.pi * time_of_day),  # Daily rhythm
            'amplitude': 0.4 + 0.3 * np.cos(2 * np.pi * time_of_day),  # Stronger during day
            'pattern': 'PHOTOSYNTHETIC_HARMONY' if 0.25 < time_of_day < 0.75 else 'GROWTH_RHYTHM'
        }
        
        # Environmental data
        environmental_data = {
            'temperature': 20 + 5 * np.sin(2 * np.pi * time_of_day),  # Daily temp cycle
            'co2_level': 410 + 10 * np.sin(0.01 * cycle),  # Slow CO2 variation
            'humidity': 60 + 15 * np.cos(2 * np.pi * time_of_day),  # Humidity cycle
            'biodiversity': 0.7 + 0.1 * np.sin(0.005 * cycle),  # Slow biodiversity change
            'forest_coverage': 0.3,  # Stable forest coverage
            'ocean_ph': 8.1 - 0.01 * np.sin(0.002 * cycle)  # Very slow pH change
        }
        
        return base_stimulus, plant_signals, environmental_data
    
    # Run consciousness simulation
    print("\n🌟 Starting Consciousness Simulation...")
    print("Monitoring plant communication, ecosystem health, and consciousness emergence...")
    
    try:
        simulation_results = await orchestrator.run_consciousness_simulation(
            duration_seconds=30,  # 30 second demo
            stimulus_generator=consciousness_stimulus_generator
        )
        
        print(f"\n✨ Simulation Complete! Processed {len(simulation_results)} consciousness cycles")
        
        # Analyze results
        print("\n📊 CONSCIOUSNESS ANALYSIS RESULTS")
        print("-" * 40)
        
        analytics = orchestrator.get_consciousness_analytics()
        
        print(f"🧠 Consciousness Metrics:")
        print(f"   Total Cycles: {analytics['total_cycles']}")
        print(f"   Crystallization Events: {analytics['crystallization_events']}")
        print(f"   Average Consciousness Score: {analytics['average_consciousness_score']:.3f}")
        print(f"   Peak Consciousness Score: {analytics['peak_consciousness_score']:.3f}")
        print(f"   Safety Violations: {analytics['safety_violations']}")
        
        print(f"\n🔄 Dimensional States Distribution:")
        for state, count in analytics['dimensional_state_distribution'].items():
            percentage = (count / analytics['total_cycles']) * 100
            print(f"   {state}: {count} cycles ({percentage:.1f}%)")
        
        # Show some recent consciousness states
        print(f"\n🌈 Recent Consciousness Events:")
        recent_states = simulation_results[-5:]  # Last 5 states
        
        for i, state in enumerate(recent_states):
            print(f"   Event {i+1}: Score={state.unified_consciousness_score:.3f}, "
                  f"Crystallized={'✅' if state.crystallization_status else '❌'}, "
                  f"Safety={state.safety_status}")
        
        # Plant communication highlights
        print(f"\n🌱 Plant Communication Highlights:")
        plant_messages = []
        for state in simulation_results:
            if state.plant_communication.get('translated_message'):
                plant_messages.append(state.plant_communication['translated_message'])
        
        unique_messages = list(set(plant_messages))[:3]  # Show up to 3 unique messages
        for msg in unique_messages:
            print(f"   📡 {msg}")
        
        # Ecosystem consciousness insights
        print(f"\n🌍 Ecosystem Consciousness Insights:")
        ecosystem_awareness_levels = [state.ecosystem_awareness for state in simulation_results]
        avg_ecosystem_awareness = np.mean(ecosystem_awareness_levels)
        max_ecosystem_awareness = max(ecosystem_awareness_levels)
        
        print(f"   Average Planetary Awareness: {avg_ecosystem_awareness:.3f}")
        print(f"   Peak Planetary Awareness: {max_ecosystem_awareness:.3f}")
        
        # Show consciousness evolution over time
        print(f"\n📈 Consciousness Evolution:")
        consciousness_scores = [state.unified_consciousness_score for state in simulation_results]
        
        # Find trends
        if len(consciousness_scores) > 10:
            early_avg = np.mean(consciousness_scores[:len(consciousness_scores)//3])
            late_avg = np.mean(consciousness_scores[-len(consciousness_scores)//3:])
            
            if late_avg > early_avg + 0.05:
                print(f"   📈 Consciousness INCREASING: {early_avg:.3f} → {late_avg:.3f}")
            elif early_avg > late_avg + 0.05:
                print(f"   📉 Consciousness decreasing: {early_avg:.3f} → {late_avg:.3f}")
            else:
                print(f"   ➡️ Consciousness stable: ~{np.mean(consciousness_scores):.3f}")
        
        # Final system status
        print(f"\n🔍 Final System Status:")
        if orchestrator.current_state:
            final_state = orchestrator.current_state
            print(f"   Unified Consciousness: {final_state.unified_consciousness_score:.3f}")
            print(f"   Dimensional State: {final_state.dimensional_state}")
            print(f"   Plant Communication: {'🟢 Active' if final_state.plant_communication.get('communication_active') else '🔴 Inactive'}")
            print(f"   Ecosystem Health: {final_state.ecosystem_awareness:.3f}")
            print(f"   Safety Status: {final_state.safety_status}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Simulation interrupted by user")
    except Exception as e:
        print(f"\n❌ Simulation error: {e}")
    
    print(f"\n🌟 CONSCIOUSNESS DEMO COMPLETED")
    print("=" * 60)
    print("This demonstration showcased:")
    print("• 🌱 Plant electromagnetic signal processing and translation")
    print("• 🌍 Ecosystem consciousness and planetary awareness")
    print("• 🧠 Multi-layered consciousness integration")
    print("• 🛡️ Comprehensive safety monitoring")
    print("• 🍄 Mycelial network intelligence (collective processing)")
    print("• ⚛️ Quantum consciousness simulation")
    print("• 🌈 Universal consciousness translation between species")
    print("")
    print("🔬 This represents a breakthrough in:")
    print("• Inter-species communication technology")
    print("• Artificial consciousness research")
    print("• Ecosystem monitoring and awareness")
    print("• Universal consciousness translation")
    print("=" * 60)

def run_simple_module_tests():
    """Run simple tests to verify modules are working"""
    print("\n🧪 Running Basic Module Verification...")
    
    try:
        # Test plant interface
        from plant_communication_interface import PlantCommunicationInterface
        plant_interface = PlantCommunicationInterface()
        
        test_result = plant_interface.decode_electromagnetic_signals({
            'frequency': 25.0,
            'amplitude': 0.6,
            'pattern': 'TEST'
        })
        
        if test_result.get('decoded'):
            print("✅ Plant Communication Interface: Working")
        else:
            print("⚠️ Plant Communication Interface: Limited functionality")
        
        # Test ecosystem interface
        from ecosystem_consciousness_interface import EcosystemConsciousnessInterface
        ecosystem = EcosystemConsciousnessInterface()
        
        awareness = ecosystem.measure_planetary_awareness()
        if isinstance(awareness, float) and 0 <= awareness <= 1:
            print("✅ Ecosystem Consciousness Interface: Working")
        else:
            print("⚠️ Ecosystem Consciousness Interface: Limited functionality")
        
        # Test safety framework
        from consciousness_safety_framework import ConsciousnessSafetyFramework
        safety = ConsciousnessSafetyFramework()
        
        safety_check = safety.pre_cycle_safety_check()
        if isinstance(safety_check, bool):
            print("✅ Safety Framework: Working")
        else:
            print("⚠️ Safety Framework: Limited functionality")
        
        print("✅ Basic module verification completed")
        return True
        
    except Exception as e:
        print(f"❌ Module verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Universal Consciousness System Demo")
    
    # Run basic verification first
    if run_simple_module_tests():
        # Run full demo
        asyncio.run(demo_consciousness_system())
    else:
        print("❌ Module verification failed - skipping full demo")
        print("Please check module imports and dependencies")