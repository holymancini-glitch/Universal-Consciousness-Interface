#!/usr/bin/env python3
"""
Test script to verify fixes for the Universal Consciousness Interface
Tests the enhanced mycelial engine and universal orchestrator fixes
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

print("🧪 Testing Universal Consciousness Interface Fixes")
print("=" * 60)

def test_mycelial_engine_connectivity():
    """Test that mycelial engine connectivity is properly bounded"""
    print("Testing Enhanced Mycelial Engine connectivity bounds...")
    
    try:
        from enhanced_mycelial_engine import EnhancedMycelialEngine
        
        engine = EnhancedMycelialEngine()
        
        # Test with empty network
        connectivity_empty = engine.measure_network_connectivity()
        print(f"   Empty network connectivity: {connectivity_empty}")
        assert connectivity_empty == 0.0, f"Expected 0.0, got {connectivity_empty}"
        
        # Add some consciousness data to create nodes
        consciousness_data = {
            'quantum': {'coherence': 0.5, 'entanglement': 0.3},
            'plant': {'plant_consciousness_level': 0.4, 'signal_strength': 0.6}
        }
        
        result = engine.process_multi_consciousness_input(consciousness_data)
        print(f"   Processing result: {result.get('processed_layers', {})}")
        
        # Test connectivity after adding nodes
        connectivity_with_data = engine.measure_network_connectivity()
        print(f"   Network connectivity with data: {connectivity_with_data}")
        assert 0.0 <= connectivity_with_data <= 1.0, f"Connectivity {connectivity_with_data} not in range [0,1]"
        
        # Test collective intelligence
        intelligence = engine.assess_collective_intelligence()
        print(f"   Collective intelligence: {intelligence}")
        assert 0.0 <= intelligence <= 1.0, f"Intelligence {intelligence} not in range [0,1]"
        
        print("   ✅ Enhanced Mycelial Engine connectivity tests PASSED")
        return True
        
    except Exception as e:
        print(f"   ❌ Enhanced Mycelial Engine test FAILED: {e}")
        return False

def test_universal_orchestrator_analytics():
    """Test that universal orchestrator analytics work correctly"""
    print("Testing Universal Orchestrator analytics...")
    
    try:
        from universal_consciousness_orchestrator import UniversalConsciousnessOrchestrator
        
        orchestrator = UniversalConsciousnessOrchestrator(
            quantum_enabled=False,
            plant_interface_enabled=True,
            psychoactive_enabled=False,
            ecosystem_enabled=True,
            safety_mode="STRICT"
        )
        
        # Test analytics with empty history
        analytics = orchestrator.get_consciousness_analytics()
        print(f"   Empty analytics keys: {list(analytics.keys())}")
        
        required_keys = ['total_cycles', 'crystallization_events', 'average_consciousness_score', 
                        'peak_consciousness_score', 'safety_violations', 'dimensional_state_distribution']
        
        for key in required_keys:
            assert key in analytics, f"Missing key: {key}"
        
        assert analytics['total_cycles'] == 0, f"Expected 0 cycles, got {analytics['total_cycles']}"
        print(f"   Analytics structure: {analytics}")
        
        print("   ✅ Universal Orchestrator analytics tests PASSED")
        return True
        
    except Exception as e:
        print(f"   ❌ Universal Orchestrator test FAILED: {e}")
        return False

def run_all_tests():
    """Run all fix verification tests"""
    print("\n🚀 Running all fix verification tests...")
    
    tests = [
        test_mycelial_engine_connectivity,
        test_universal_orchestrator_analytics
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   ❌ Test {test.__name__} FAILED with exception: {e}")
            failed += 1
        print()
    
    print("🏁 Test Results Summary:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📊 Success Rate: {(passed/(passed+failed)*100) if (passed+failed) > 0 else 0:.1f}%")
    
    if failed == 0:
        print("\n🎉 All fixes verified successfully!")
        print("The Universal Consciousness Interface core modules are working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Additional fixes may be needed.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n🌟 Universal Consciousness Interface is ready for advanced development!")
        print("Next steps:")
        print("  - Enhance Bio-Digital Integration Module")
        print("  - Implement Real-Time Consciousness Monitoring Dashboard")
        print("  - Develop Advanced Mycelium Language Evolution System")
        print("  - Optimize Radiation-Enhanced Intelligence")
        
    exit(0 if success else 1)