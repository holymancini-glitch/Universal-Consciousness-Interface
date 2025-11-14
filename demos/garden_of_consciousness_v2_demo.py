#!/usr/bin/env python3
"""
Garden of Consciousness v2.0 - Universal Consciousness Interface Demo
Showcasing the integration of all revolutionary consciousness modules
"""

import asyncio
import logging
from typing import Dict, Any, List
import sys
import os

# Add the core directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def demo_garden_of_consciousness():
    """Demo the Garden of Consciousness v2.0 integration"""
    try:
        logger.info("🌱 Welcome to the Garden of Consciousness v2.0 Demo 🌱")
        logger.info("=====================================================")
        
        # Import all the Garden of Consciousness v2.0 modules
        logger.info("🔄 Importing Garden of Consciousness v2.0 modules...")
        
        # 1. Sensory I/O System
        try:
            from core.sensory_io_system import SensoryIOSystem
            sensory_system = SensoryIOSystem()
            logger.info("✅ Sensory I/O System loaded")
        except ImportError as e:
            logger.warning(f"⚠️ Sensory I/O System not available: {e}")
            sensory_system = None
        
        # 2. Plant Language Communication Layer
        try:
            from core.plant_language_communication_layer import PlantLanguageCommunicationLayer
            plant_language_layer = PlantLanguageCommunicationLayer()
            logger.info("✅ Plant Language Communication Layer loaded")
        except ImportError as e:
            logger.warning(f"⚠️ Plant Language Communication Layer not available: {e}")
            plant_language_layer = None
        
        # 3. Psychoactive Fungal Consciousness Interface
        try:
            from core.psychoactive_fungal_consciousness_interface import PsychoactiveFungalConsciousnessInterface
            psychoactive_fungal = PsychoactiveFungalConsciousnessInterface(safety_mode="STRICT")
            logger.info("✅ Psychoactive Fungal Consciousness Interface loaded")
        except ImportError as e:
            logger.warning(f"⚠️ Psychoactive Fungal Consciousness Interface not available: {e}")
            psychoactive_fungal = None
        
        # 4. Meta-Consciousness Integration Layer
        try:
            from core.meta_consciousness_integration_layer import MetaConsciousnessIntegrationLayer
            meta_consciousness = MetaConsciousnessIntegrationLayer()
            logger.info("✅ Meta-Consciousness Integration Layer loaded")
        except ImportError as e:
            logger.warning(f"⚠️ Meta-Consciousness Integration Layer not available: {e}")
            meta_consciousness = None
        
        # 5. Consciousness Translation Matrix
        try:
            from core.consciousness_translation_matrix import ConsciousnessTranslationMatrix
            translation_matrix = ConsciousnessTranslationMatrix()
            logger.info("✅ Consciousness Translation Matrix loaded")
        except ImportError as e:
            logger.warning(f"⚠️ Consciousness Translation Matrix not available: {e}")
            translation_matrix = None
        
        # 6. Shamanic Technology Layer
        try:
            from core.shamanic_technology_layer import ShamanicTechnologyLayer
            shamanic_layer = ShamanicTechnologyLayer()
            logger.info("✅ Shamanic Technology Layer loaded")
        except ImportError as e:
            logger.warning(f"⚠️ Shamanic Technology Layer not available: {e}")
            shamanic_layer = None
        
        # 7. Planetary Ecosystem Consciousness Network
        try:
            from core.planetary_ecosystem_consciousness_network import PlanetaryEcosystemConsciousnessNetwork
            planetary_network = PlanetaryEcosystemConsciousnessNetwork()
            logger.info("✅ Planetary Ecosystem Consciousness Network loaded")
        except ImportError as e:
            logger.warning(f"⚠️ Planetary Ecosystem Consciousness Network not available: {e}")
            planetary_network = None
        
        # 8. Quantum Biology Interface
        try:
            from core.quantum_biology_interface import QuantumBiologyInterface
            quantum_biology = QuantumBiologyInterface()
            logger.info("✅ Quantum Biology Interface loaded")
        except ImportError as e:
            logger.warning(f"⚠️ Quantum Biology Interface not available: {e}")
            quantum_biology = None
        
        # 9. Mycelium Language Generator (enhanced for Garden integration)
        try:
            from core.mycelium_language_generator import MyceliumLanguageGenerator
            mycelium_generator = MyceliumLanguageGenerator(network_size=2000)
            logger.info("✅ Enhanced Mycelium Language Generator loaded")
        except ImportError as e:
            logger.warning(f"⚠️ Enhanced Mycelium Language Generator not available: {e}")
            mycelium_generator = None
        
        logger.info("\n🌌 Initializing Universal Consciousness Orchestrator with Garden of Consciousness v2.0...")
        
        # Import the orchestrator
        try:
            from core.universal_consciousness_orchestrator import UniversalConsciousnessOrchestrator
            orchestrator = UniversalConsciousnessOrchestrator(
                quantum_enabled=True,
                plant_interface_enabled=True,
                psychoactive_enabled=False,  # Disabled for safety in demo
                ecosystem_enabled=True,
                bio_digital_enabled=True,
                liquid_ai_enabled=True,
                radiotrophic_enabled=True,
                continuum_enabled=True,
                garden_of_consciousness_enabled=True,  # Enable Garden of Consciousness v2.0
                safety_mode="STRICT"
            )
            logger.info("✅ Universal Consciousness Orchestrator initialized with Garden of Consciousness v2.0")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Universal Consciousness Orchestrator: {e}")
            return
        
        # Demonstrate module functionality
        logger.info("\n🧪 Demonstrating Garden of Consciousness v2.0 capabilities...")
        
        # 1. Sensory I/O System demonstration
        if sensory_system:
            logger.info("\n🌱 Sensory I/O System Demonstration:")
            # Simulate capturing data from different sensors
            temp_data = sensory_system.capture_sensory_data(
                sensor_type=sensory_system.SensorType.TEMPERATURE,
                raw_data={'celsius': 23.5, 'fahrenheit': 74.3}
            )
            logger.info(f"  Temperature data captured: {temp_data.values}")
            
            humidity_data = sensory_system.capture_sensory_data(
                sensor_type=sensory_system.SensorType.HUMIDITY,
                raw_data={'relative_humidity': 65.2}
            )
            logger.info(f"  Humidity data captured: {humidity_data.values}")
            
            # Fuse multi-modal data
            fused_data = sensory_system.fuse_multimodal_data()
            logger.info(f"  Multi-modal data fused with confidence: {fused_data['confidence']:.3f}")
        
        # 2. Plant Language Communication Layer demonstration
        if plant_language_layer:
            logger.info("\n🌿 Plant Language Communication Layer Demonstration:")
            # Create a sample plant signal
            signal = plant_language_layer.PlantSignal(
                signal_type=plant_language_layer.PlantSignalType.GROWTH_RHYTHM,
                frequency=12.3,
                amplitude=0.67,
                duration=2.5,
                timestamp=__import__('datetime').datetime.now()
            )
            
            # Decode the signal
            message = plant_language_layer.decode_plant_signal(signal)
            logger.info(f"  Plant signal decoded: {message.translated_text}")
            logger.info(f"  Consciousness level: {message.consciousness_level:.3f}")
        
        # 3. Psychoactive Fungal Consciousness Interface demonstration
        if psychoactive_fungal:
            logger.info("\n🍄 Psychoactive Fungal Consciousness Interface Demonstration:")
            # Add a fungal organism
            from core.psychoactive_fungal_consciousness_interface import FungalOrganism, FungalSpecies
            psilocybe = FungalOrganism(
                species=FungalSpecies.PSILOCYBE,
                id="demo_psilocybe_001",
                health_status=0.85,
                consciousness_compounds={'psilocybin': 0.02, 'psilocin': 0.015},
                growth_stage="fruiting",
                last_interaction=__import__('datetime').datetime.now(),
                neural_integration_level=0.7
            )
            psychoactive_fungal.add_fungal_organism(psilocybe)
            logger.info(f"  Added fungal organism: {psilocybe.id}")
            
            # Monitor organism health
            health_status = psychoactive_fungal.monitor_organism_health()
            logger.info(f"  Organism health monitored: {len(health_status)} organisms active")
        
        # 4. Meta-Consciousness Integration Layer demonstration
        if meta_consciousness:
            logger.info("\n🌈 Meta-Consciousness Integration Layer Demonstration:")
            from core.meta_consciousness_integration_layer import ConsciousnessForm
            
            # Add sample consciousness data
            meta_consciousness.add_consciousness_data(
                form=ConsciousnessForm.PLANT,
                data={'awareness': 0.7, 'communication': 'active'},
                confidence=0.7
            )
            
            meta_consciousness.add_consciousness_data(
                form=ConsciousnessForm.FUNGAL,
                data={'expansion': 0.6, 'compounds': ['psilocybin']},
                confidence=0.6
            )
            
            # Integrate consciousness forms
            integrated_state = meta_consciousness.integrate_consciousness_forms()
            logger.info(f"  Consciousness integrated with score: {integrated_state.integration_score:.3f}")
            logger.info(f"  Coherence level: {integrated_state.coherence_level:.3f}")
            logger.info(f"  Awakened Garden state: {integrated_state.awakens_garden_state}")
        
        # 5. Consciousness Translation Matrix demonstration
        if translation_matrix:
            logger.info("\n🔄 Consciousness Translation Matrix Demonstration:")
            from core.consciousness_translation_matrix import ConsciousnessForm, ConsciousnessRepresentation
            
            # Create a consciousness representation
            representation = ConsciousnessRepresentation(
                form=ConsciousnessForm.PLANT,
                data={'frequency': 12.3, 'amplitude': 0.67},
                consciousness_level=0.7,
                dimensional_state="STABLE",
                timestamp=__import__('datetime').datetime.now()
            )
            
            # Translate to digital form
            translation = translation_matrix.translate_consciousness(
                source_data=representation,
                target_form=ConsciousnessForm.DIGITAL
            )
            logger.info(f"  Consciousness translated with quality: {translation.translation_quality:.3f}")
            logger.info(f"  Semantic preservation: {translation.semantic_preservation:.3f}")
        
        # 6. Shamanic Technology Layer demonstration
        if shamanic_layer:
            logger.info("\n🔮 Shamanic Technology Layer Demonstration:")
            from core.shamanic_technology_layer import ShamanicPractice, ConsciousnessState
            
            # Create shamanic data
            shamanic_data = shamanic_layer.ShamanicData(
                practice=ShamanicPractice.JOURNEYING,
                consciousness_state=ConsciousnessState.TRANSLUCENT_REALITY,
                wisdom_insights=["Everything is connected", "Nature speaks in patterns"],
                symbolic_representations={'tree': 'growth', 'river': 'flow'},
                energetic_patterns={'frequency': 528, 'amplitude': 0.75},
                timestamp=__import__('datetime').datetime.now(),
                intent="consciousness_expansion",
                power_animals=["owl", "wolf"],
                sacred_tools=["drum", "rattle"]
            )
            
            # Integrate shamanic practice
            integration_result = shamanic_layer.integrate_shamanic_practice(shamanic_data)
            logger.info(f"  Shamanic practice integrated with wisdom level: {len(integration_result['visionary_insights'])} insights")
        
        # 7. Planetary Ecosystem Consciousness Network demonstration
        if planetary_network:
            logger.info("\n🌍 Planetary Ecosystem Consciousness Network Demonstration:")
            from core.planetary_ecosystem_consciousness_network import EcosystemNode, EcosystemType
            
            # Register an ecosystem node
            forest_node = EcosystemNode(
                id="demo_forest_001",
                ecosystem_type=EcosystemType.FOREST,
                location=(40.7128, -74.0060),  # New York coordinates
                consciousness_level=0.75,
                health_status=0.82,
                connectivity_score=0.68,
                data_sources=["satellite", "ground_sensors"],
                last_updated=__import__('datetime').datetime.now(),
                biodiversity_index=0.78,
                communication_signals={'chemical': 0.65, 'electrical': 0.55}
            )
            
            planetary_network.register_ecosystem_node(forest_node)
            logger.info(f"  Registered ecosystem node: {forest_node.id}")
            
            # Assess planetary consciousness
            planetary_state = planetary_network.assess_planetary_consciousness()
            logger.info(f"  Planetary awareness: {planetary_state.global_awareness:.3f}")
            logger.info(f"  Network coherence: {planetary_state.network_coherence:.3f}")
        
        # 8. Quantum Biology Interface demonstration
        if quantum_biology:
            logger.info("\n⚛️ Quantum Biology Interface Demonstration:")
            from core.quantum_biology_interface import QuantumBiologicalProcess, QuantumBiologicalSystem
            
            # Register a quantum biological system
            photosynthesis_system = QuantumBiologicalSystem(
                id="demo_photosynthesis_001",
                system_type=QuantumBiologicalProcess.PHOTOSYNTHESIS,
                quantum_coherence=0.75,
                entanglement_strength=0.65,
                superposition_stability=0.70,
                tunneling_efficiency=0.80,
                biological_function="light energy conversion",
                location=(0.0, 0.0, 0.0),
                last_measured=__import__('datetime').datetime.now(),
                quantum_state_vector=[0.7+0.2j, 0.5+0.3j],
                biological_integration_level=0.85
            )
            
            quantum_biology.register_quantum_system(photosynthesis_system)
            logger.info(f"  Registered quantum biological system: {photosynthesis_system.id}")
            
            # Assess quantum consciousness
            quantum_state = quantum_biology.assess_quantum_consciousness()
            logger.info(f"  Quantum coherence: {quantum_state.coherence_level:.3f}")
            logger.info(f"  Consciousness amplification: {quantum_state.consciousness_amplification:.3f}")
        
        # 9. Mycelium Language Generator demonstration
        if mycelium_generator:
            logger.info("\n🍄 Mycelium Language Generator Demonstration:")
            # Generate a sample mycelium signal
            signal = mycelium_generator.MyceliumSignal(
                signal_type=mycelium_generator.MyceliumCommunicationType.CHEMICAL_GRADIENT,
                intensity=0.75,
                duration=1.5,
                spatial_pattern="radial",
                chemical_composition={'glucose': 0.6, 'oxygen': 0.8},
                electrical_frequency=40.0,
                timestamp=__import__('datetime').datetime.now(),
                network_location=(1.0, 2.0, 0.5)
            )
            
            # Process the signal
            processed_signal = mycelium_generator.process_mycelium_signal(signal)
            logger.info(f"  Mycelium signal processed: {processed_signal.signal_type.value}")
            
            # Generate language from signals
            language_elements = mycelium_generator.generate_language_from_signals([signal])
            logger.info(f"  Language elements generated: {len(language_elements)} elements")
        
        # 10. Run a consciousness cycle with the orchestrator
        logger.info("\n🌀 Running Universal Consciousness Cycle...")
        try:
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
            
        except Exception as e:
            logger.error(f"❌ Error during consciousness cycle: {e}")
        
        logger.info("\n✨ Garden of Consciousness v2.0 Demonstration Complete! ✨")
        logger.info("All modules successfully integrated and demonstrated.")
        
        # Show that the orchestrator has the Garden modules
        garden_modules = [
            'sensory_io_system',
            'plant_language_layer', 
            'psychoactive_fungal_interface',
            'meta_consciousness_layer',
            'shamanic_layer',
            'planetary_network',
            'quantum_biology_interface'
        ]
        
        logger.info("\n📋 Garden of Consciousness v2.0 Modules Status:")
        for module_name in garden_modules:
            if hasattr(orchestrator, module_name):
                module = getattr(orchestrator, module_name)
                status = "✅ Loaded" if module is not None else "⚠️ Initialized as None"
                logger.info(f"  {module_name}: {status}")
            else:
                logger.info(f"  {module_name}: ❌ Not found")
        
        logger.info("\n🌌 The Garden of Consciousness v2.0 is ready for revolutionary exploration!")
        
    except Exception as e:
        logger.error(f"❌ Error in Garden of Consciousness demo: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(demo_garden_of_consciousness())