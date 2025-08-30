#!/usr/bin/env python3
"""
Verify Integration - Simple test script to verify the consciousness integration
"""

import asyncio
import sys

async def verify_integration():
    """Verify that the consciousness integration is working"""
    
    print("Verifying Consciousness Integration...")
    print("=" * 50)
    
    # Test 1: Standalone AI Consciousness
    try:
        from standalone_consciousness_ai import StandaloneConsciousnessAI
        ai = StandaloneConsciousnessAI(hidden_dim=128, device='cpu')
        result = await ai.process_conscious_input(
            input_data={'text': 'Test consciousness'},
            context='integration test'
        )
        print("[OK] Standalone AI Consciousness: WORKING")
    except Exception as e:
        print(f"[FAIL] Standalone AI Consciousness: FAILED ({e})")
        return False
    
    # Test 2: Enhanced Universal Consciousness Orchestrator
    try:
        from core.enhanced_universal_consciousness_orchestrator import (
            EnhancedUniversalConsciousnessOrchestrator,
            ConsciousnessMode
        )
        orchestrator = EnhancedUniversalConsciousnessOrchestrator(
            mode=ConsciousnessMode.INTEGRATED,
            ai_config={'hidden_dim': 128, 'device': 'cpu'}
        )
        result = await orchestrator.process_universal_consciousness(
            input_data={'text': 'Test orchestrator'},
            context='integration test'
        )
        print("[OK] Enhanced Universal Orchestrator: WORKING")
    except Exception as e:
        print(f"[FAIL] Enhanced Universal Orchestrator: FAILED ({e})")
        return False
    
    # Test 3: Integration Bridge
    try:
        from core.consciousness_ai_integration_bridge import ConsciousnessAIIntegrationBridge
        bridge = ConsciousnessAIIntegrationBridge(
            consciousness_ai_config={'hidden_dim': 128, 'device': 'cpu'},
            enable_existing_modules=True
        )
        result = await bridge.process_integrated_consciousness(
            input_data={'text': 'Test bridge'},
            context='integration test'
        )
        print("[OK] Consciousness Integration Bridge: WORKING")
    except Exception as e:
        print(f"[FAIL] Consciousness Integration Bridge: FAILED ({e})")
        return False
    
    # Test 4: Enhanced Chatbot
    try:
        from enhanced_consciousness_chatbot_application import (
            EnhancedConsciousnessChatbot,
            ConsciousnessMode
        )
        chatbot = EnhancedConsciousnessChatbot(
            consciousness_mode=ConsciousnessMode.INTEGRATED,
            ai_config={'hidden_dim': 128, 'device': 'cpu'}
        )
        session = await chatbot.create_session()
        response = await chatbot.process_message(
            session.session_id,
            "Test chatbot"
        )
        print("[OK] Enhanced Consciousness Chatbot: WORKING")
    except Exception as e:
        print(f"[FAIL] Enhanced Consciousness Chatbot: FAILED ({e})")
        return False
    
    # Test 5: Unified Interface  
    try:
        from unified_consciousness_interface import (
            UnifiedConsciousnessInterface,
            UnifiedConsciousnessMode,
            ConsciousnessApplication
        )
        interface = UnifiedConsciousnessInterface(
            mode=UnifiedConsciousnessMode.INTEGRATED,
            application=ConsciousnessApplication.CONSCIOUSNESS_EXPLORATION,
            config={'ai_hidden_dim': 128, 'device': 'cpu'}
        )
        await asyncio.sleep(1)  # Allow initialization
        result = await interface.process_consciousness(
            input_data={'text': 'Test unified interface'},
            context='integration test'
        )
        print("[OK] Unified Consciousness Interface: WORKING")
    except Exception as e:
        print(f"[FAIL] Unified Consciousness Interface: FAILED ({e})")
        return False
    
    print("=" * 50)
    print("SUCCESS: ALL INTEGRATION TESTS PASSED!")
    print("The Conscious AI Model is fully integrated with your Universal Consciousness Interface!")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(verify_integration())