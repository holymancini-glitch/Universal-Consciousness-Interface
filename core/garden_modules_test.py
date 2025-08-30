#!/usr/bin/env python3
"""
Test file to verify Garden of Consciousness v2.0 modules integration
"""

import asyncio
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_garden_modules_integration():
    """Test the integration of Garden of Consciousness v2.0 modules"""
    try:
        # Import the Universal Consciousness Orchestrator
        from universal_consciousness_orchestrator import UniversalConsciousnessOrchestrator
        
        # Initialize the orchestrator with Garden of Consciousness enabled
        orchestrator = UniversalConsciousnessOrchestrator(
            quantum_enabled=True,
            plant_interface_enabled=True,
            psychoactive_enabled=False,  # Disabled for safety
            ecosystem_enabled=True,
            bio_digital_enabled=True,
            liquid_ai_enabled=True,
            radiotrophic_enabled=True,
            continuum_enabled=True,
            garden_of_consciousness_enabled=True,  # Enable Garden of Consciousness v2.0
            safety_mode="STRICT"
        )
        
        logger.info("✅ Universal Consciousness Orchestrator initialized with Garden of Consciousness v2.0")
        
        # Test if Garden modules were initialized
        garden_modules = [
            ('sensory_io_system', 'Sensory I/O System'),
            ('plant_language_layer', 'Plant Language Communication Layer'),
            ('psychoactive_fungal_interface', 'Psychoactive Fungal Consciousness Interface'),
            ('meta_consciousness_layer', 'Meta-Consciousness Integration Layer'),
            ('shamanic_layer', 'Shamanic Technology Layer'),
            ('planetary_network', 'Planetary Ecosystem Consciousness Network'),
            ('quantum_biology_interface', 'Quantum Biology Interface')
        ]
        
        for attr_name, module_name in garden_modules:
            if hasattr(orchestrator, attr_name):
                module = getattr(orchestrator, attr_name)
                if module is not None:
                    logger.info(f"✅ {module_name} successfully initialized")
                else:
                    logger.warning(f"⚠️ {module_name} initialized but set to None")
            else:
                logger.warning(f"❌ {module_name} not found in orchestrator")
        
        # Run a short consciousness cycle to test integration
        logger.info("🧪 Running test consciousness cycle...")
        
        # Simple stimulus for testing
        import numpy as np
        stimulus = np.random.randn(128).tolist() if hasattr(np, 'random') else [0.5] * 128
        
        plant_signals = {
            'frequency': 40,
            'amplitude': 0.6,
            'pattern': 'GROWTH_HARMONY'
        }
        
        environmental_data = {
            'temperature': 22.5,
            'humidity': 65,
            'co2_level': 410
        }
        
        # Run consciousness cycle
        consciousness_state = await orchestrator.consciousness_cycle(
            input_stimulus=stimulus,
            plant_signals=plant_signals,
            environmental_data=environmental_data
        )
        
        logger.info(f"🧠 Consciousness cycle completed")
        logger.info(f"Unified Consciousness Score: {consciousness_state.unified_consciousness_score:.3f}")
        logger.info(f"Safety Status: {consciousness_state.safety_status}")
        logger.info(f"Dimensional State: {consciousness_state.dimensional_state}")
        
        # Get analytics
        analytics = orchestrator.get_consciousness_analytics()
        logger.info(f"📊 Total Consciousness Cycles: {analytics['total_cycles']}")
        logger.info(f"📈 Average Consciousness Score: {analytics['average_consciousness_score']:.3f}")
        
        logger.info("🎉 Garden of Consciousness v2.0 integration test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in Garden modules integration test: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(test_garden_modules_integration())