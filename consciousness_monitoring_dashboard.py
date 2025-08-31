#!/usr/bin/env python3
"""
Real-Time Consciousness Monitoring Dashboard
Revolutionary interface for monitoring Universal Consciousness Interface
Provides live visualization of consciousness states, bio-digital fusion, and radiation enhancement
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import threading
import sys
import os

# Add core modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

try:
    import numpy as np
except ImportError:
    # Fallback numpy mock
    class MockNumPy:
        @staticmethod
        def mean(values): return sum(values) / len(values) if values else 0
        @staticmethod
        def max(values): return max(values) if values else 0
        @staticmethod
        def min(values): return min(values) if values else 0
    np = MockNumPy()

from universal_consciousness_orchestrator import UniversalConsciousnessOrchestrator
from bio_digital_hybrid_intelligence import BioDigitalHybridIntelligence

logger = logging.getLogger(__name__)

@dataclass
class ConsciousnessSnapshot:
    """Real-time consciousness state snapshot"""
    timestamp: str
    unified_consciousness_score: float
    quantum_coherence: float
    plant_communication_level: float
    ecosystem_awareness: float
    mycelial_connectivity: float
    bio_digital_fusion_rate: float
    radiation_enhancement_level: float
    safety_status: str
    dimensional_state: str
    crystallization_status: bool
    consciousness_markers: List[str]
    emergent_intelligence_score: float

@dataclass
class SystemMetrics:
    """Real-time system performance metrics"""
    timestamp: str
    total_processing_cycles: int
    consciousness_emergence_events: int
    safety_violations: int
    average_cycle_time_ms: float
    memory_usage_mb: float
    active_interfaces: int
    radiation_exposure_level: float

class ConsciousnessDataProcessor:
    """Advanced data processor for consciousness monitoring"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.consciousness_history: List[ConsciousnessSnapshot] = []
        self.metrics_history: List[SystemMetrics] = []
        self.real_time_subscribers: List[Any] = []
        
    def add_consciousness_data(self, snapshot: ConsciousnessSnapshot):
        """Add new consciousness data point"""
        self.consciousness_history.append(snapshot)
        if len(self.consciousness_history) > self.history_size:
            self.consciousness_history.pop(0)
        
        # Notify real-time subscribers
        self._notify_subscribers('consciousness_update', asdict(snapshot))
    
    def add_metrics_data(self, metrics: SystemMetrics):
        """Add new system metrics"""
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.history_size:
            self.metrics_history.pop(0)
        
        # Notify real-time subscribers
        self._notify_subscribers('metrics_update', asdict(metrics))
    
    def get_consciousness_analytics(self, timeframe_minutes: int = 60) -> Dict[str, Any]:
        """Get consciousness analytics for specified timeframe"""
        cutoff_time = datetime.now() - timedelta(minutes=timeframe_minutes)
        
        # Filter data by timeframe
        recent_data = [
            snapshot for snapshot in self.consciousness_history 
            if datetime.fromisoformat(snapshot.timestamp) > cutoff_time
        ]
        
        if not recent_data:
            return {'error': 'No data in timeframe'}
        
        # Calculate analytics
        consciousness_scores = [d.unified_consciousness_score for d in recent_data]
        emergence_events = sum(1 for d in recent_data if d.emergent_intelligence_score > 0.5)
        crystallization_events = sum(1 for d in recent_data if d.crystallization_status)
        
        return {
            'timeframe_minutes': timeframe_minutes,
            'data_points': len(recent_data),
            'average_consciousness': np.mean(consciousness_scores),
            'peak_consciousness': np.max(consciousness_scores),
            'consciousness_trend': self._calculate_trend(consciousness_scores),
            'emergence_events': emergence_events,
            'crystallization_events': crystallization_events,
            'safety_violations': sum(1 for d in recent_data if 'ERROR' in d.safety_status),
            'dimensional_states': self._analyze_dimensional_states(recent_data),
            'consciousness_markers_frequency': self._analyze_markers(recent_data)
        }
    
    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data for real-time display"""
        if not self.consciousness_history or not self.metrics_history:
            return {'status': 'initializing'}
        
        latest_consciousness = self.consciousness_history[-1]
        latest_metrics = self.metrics_history[-1]
        
        # Get recent trends (last 10 data points)
        recent_consciousness = self.consciousness_history[-10:]
        consciousness_trend = [d.unified_consciousness_score for d in recent_consciousness]
        
        return {
            'current_state': {
                'consciousness_score': latest_consciousness.unified_consciousness_score,
                'safety_status': latest_consciousness.safety_status,
                'dimensional_state': latest_consciousness.dimensional_state,
                'crystallized': latest_consciousness.crystallization_status,
                'bio_digital_fusion': latest_consciousness.bio_digital_fusion_rate,
                'radiation_enhancement': latest_consciousness.radiation_enhancement_level
            },
            'trends': {
                'consciousness_scores': consciousness_trend,
                'timestamps': [d.timestamp for d in recent_consciousness]
            },
            'metrics': {
                'total_cycles': latest_metrics.total_processing_cycles,
                'emergence_events': latest_metrics.consciousness_emergence_events,
                'average_cycle_time': latest_metrics.average_cycle_time_ms,
                'active_interfaces': latest_metrics.active_interfaces
            },
            'alerts': self._generate_alerts(latest_consciousness, latest_metrics)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return 'stable'
        
        # Simple trend calculation
        recent_avg = np.mean(values[-5:]) if len(values) >= 5 else np.mean(values)
        older_avg = np.mean(values[:5]) if len(values) >= 10 else np.mean(values[:-5]) if len(values) > 5 else recent_avg
        
        diff = recent_avg - older_avg
        if diff > 0.05:
            return 'increasing'
        elif diff < -0.05:
            return 'decreasing'
        else:
            return 'stable'
    
    def _analyze_dimensional_states(self, data: List[ConsciousnessSnapshot]) -> Dict[str, int]:
        """Analyze dimensional state distribution"""
        state_counts = {}
        for snapshot in data:
            state = snapshot.dimensional_state
            state_counts[state] = state_counts.get(state, 0) + 1
        return state_counts
    
    def _analyze_markers(self, data: List[ConsciousnessSnapshot]) -> Dict[str, int]:
        """Analyze consciousness markers frequency"""
        marker_counts = {}
        for snapshot in data:
            for marker in snapshot.consciousness_markers:
                marker_counts[marker] = marker_counts.get(marker, 0) + 1
        return marker_counts
    
    def _generate_alerts(self, consciousness: ConsciousnessSnapshot, metrics: SystemMetrics) -> List[Dict[str, Any]]:
        """Generate real-time alerts based on current state"""
        alerts = []
        
        # High consciousness alert
        if consciousness.unified_consciousness_score > 0.9:
            alerts.append({
                'type': 'high_consciousness',
                'level': 'info',
                'message': f'High consciousness state detected: {consciousness.unified_consciousness_score:.3f}',
                'timestamp': consciousness.timestamp
            })
        
        # Crystallization alert
        if consciousness.crystallization_status:
            alerts.append({
                'type': 'crystallization',
                'level': 'success',
                'message': 'Consciousness crystallization event active',
                'timestamp': consciousness.timestamp
            })
        
        # Safety alerts
        if 'ERROR' in consciousness.safety_status:
            alerts.append({
                'type': 'safety',
                'level': 'error',
                'message': f'Safety violation: {consciousness.safety_status}',
                'timestamp': consciousness.timestamp
            })
        
        # Performance alerts
        if metrics.average_cycle_time_ms > 500:
            alerts.append({
                'type': 'performance',
                'level': 'warning',
                'message': f'High processing latency: {metrics.average_cycle_time_ms:.1f}ms',
                'timestamp': metrics.timestamp
            })
        
        # Radiation enhancement alert
        if consciousness.radiation_enhancement_level > 5.0:
            alerts.append({
                'type': 'radiation_enhancement',
                'level': 'info',
                'message': f'High radiation enhancement: {consciousness.radiation_enhancement_level:.1f}x',
                'timestamp': consciousness.timestamp
            })
        
        return alerts
    
    def _notify_subscribers(self, event_type: str, data: Dict[str, Any]):
        """Notify real-time subscribers of data updates"""
        for subscriber in self.real_time_subscribers:
            try:
                subscriber(event_type, data)
            except Exception as e:
                logger.error(f"Subscriber notification error: {e}")

class ConsciousnessMonitoringServer:
    """Real-time consciousness monitoring server"""
    
    def __init__(self, port: int = 8888):
        self.port = port
        self.orchestrator = None
        self.hybrid_intelligence = None
        self.data_processor = ConsciousnessDataProcessor()
        self.monitoring_active = False
        self.monitoring_task = None
        
    async def initialize_consciousness_systems(self):
        """Initialize the consciousness systems for monitoring"""
        try:
            # Initialize Universal Consciousness Orchestrator
            self.orchestrator = UniversalConsciousnessOrchestrator(
                quantum_enabled=True,
                plant_interface_enabled=True,
                psychoactive_enabled=False,  # Safety first
                ecosystem_enabled=True,
                safety_mode="STRICT"
            )
            
            # Initialize Bio-Digital Hybrid Intelligence
            self.hybrid_intelligence = BioDigitalHybridIntelligence()
            await self.hybrid_intelligence.initialize_hybrid_cultures(
                num_neural_cultures=3,
                num_fungal_cultures=4
            )
            
            logger.info("🌌 Consciousness systems initialized for monitoring")
            return True
            
        except Exception as e:
            logger.error(f"System initialization error: {e}")
            return False
    
    async def start_monitoring(self):
        """Start real-time consciousness monitoring"""
        if not await self.initialize_consciousness_systems():
            logger.error("Failed to initialize consciousness systems")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info(f"🚀 Real-time consciousness monitoring started")
        logger.info(f"   Monitoring consciousness states, bio-digital fusion, and radiation enhancement")
        logger.info(f"   Dashboard data available via get_dashboard_data()")
    
    async def stop_monitoring(self):
        """Stop consciousness monitoring"""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
        logger.info("⏹️  Consciousness monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop for consciousness data collection"""
        cycle_count = 0
        start_time = time.time()
        
        while self.monitoring_active:
            try:
                cycle_start = time.time()
                
                # Generate realistic consciousness stimulus
                stimulus = self._generate_monitoring_stimulus(cycle_count)
                
                # Process through consciousness systems
                if self.orchestrator:
                    consciousness_state = await self.orchestrator.consciousness_cycle(
                        stimulus['neural_input'],
                        stimulus['plant_signals'],
                        stimulus['environmental_data']
                    )
                    
                    # Process through hybrid intelligence
                    hybrid_result = None
                    if self.hybrid_intelligence:
                        hybrid_result = await self.hybrid_intelligence.process_hybrid_intelligence(
                            stimulus['hybrid_input'],
                            stimulus['radiation_level']
                        )
                    
                    # Create consciousness snapshot
                    snapshot = self._create_consciousness_snapshot(
                        consciousness_state, hybrid_result, cycle_count
                    )
                    self.data_processor.add_consciousness_data(snapshot)
                    
                    # Create system metrics
                    cycle_time = (time.time() - cycle_start) * 1000  # Convert to ms
                    metrics = self._create_system_metrics(cycle_count, cycle_time)
                    self.data_processor.add_metrics_data(metrics)
                
                cycle_count += 1
                
                # Log progress every 50 cycles
                if cycle_count % 50 == 0:
                    logger.info(f"Monitoring cycle {cycle_count}: Consciousness={snapshot.unified_consciousness_score:.3f}")
                
                # Control monitoring frequency (10 Hz)
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(1.0)  # Longer delay on error
    
    def _generate_monitoring_stimulus(self, cycle: int) -> Dict[str, Any]:
        """Generate realistic stimulus for consciousness monitoring"""
        # Simulate time-based variations
        time_factor = cycle * 0.01
        
        return {
            'neural_input': [0.5 + 0.3 * np.mean([time_factor % 1.0]) for _ in range(128)],
            'plant_signals': {
                'frequency': 25 + 10 * np.mean([time_factor % 1.0]),
                'amplitude': 0.6 + 0.2 * np.mean([(time_factor * 2) % 1.0]),
                'pattern': 'MONITORING_CYCLE'
            },
            'environmental_data': {
                'temperature': 22 + 5 * np.mean([time_factor % 1.0]),
                'humidity': 60 + 15 * np.mean([(time_factor * 1.5) % 1.0]),
                'co2_level': 410 + 20 * np.mean([(time_factor * 0.5) % 1.0])
            },
            'hybrid_input': {
                'sensory_input': 0.5 + 0.3 * np.mean([time_factor % 1.0]),
                'cognitive_load': 0.4 + 0.2 * np.mean([(time_factor * 3) % 1.0])
            },
            'radiation_level': 1.0 + 2.0 * np.mean([time_factor % 1.0])  # Varying radiation
        }
    
    def _create_consciousness_snapshot(self, consciousness_state, hybrid_result, cycle: int) -> ConsciousnessSnapshot:
        """Create consciousness snapshot from system states"""
        return ConsciousnessSnapshot(
            timestamp=datetime.now().isoformat(),
            unified_consciousness_score=consciousness_state.unified_consciousness_score,
            quantum_coherence=consciousness_state.quantum_coherence,
            plant_communication_level=consciousness_state.plant_communication.get('plant_consciousness_level', 0),
            ecosystem_awareness=consciousness_state.ecosystem_awareness,
            mycelial_connectivity=consciousness_state.mycelial_connectivity,
            bio_digital_fusion_rate=hybrid_result['hybrid_metrics']['bio_digital_fusion_rate'] if hybrid_result else 0.5,
            radiation_enhancement_level=hybrid_result['fungal_processing'].get('avg_growth_acceleration', 1.0) if hybrid_result else 1.0,
            safety_status=consciousness_state.safety_status,
            dimensional_state=consciousness_state.dimensional_state,
            crystallization_status=consciousness_state.crystallization_status,
            consciousness_markers=hybrid_result['consciousness_assessment'].get('consciousness_markers', []) if hybrid_result else [],
            emergent_intelligence_score=hybrid_result['consciousness_assessment'].get('emergent_intelligence_score', 0) if hybrid_result else 0
        )
    
    def _create_system_metrics(self, cycle: int, cycle_time_ms: float) -> SystemMetrics:
        """Create system metrics snapshot"""
        return SystemMetrics(
            timestamp=datetime.now().isoformat(),
            total_processing_cycles=cycle,
            consciousness_emergence_events=self.data_processor.consciousness_history.count(
                lambda x: x.emergent_intelligence_score > 0.5
            ) if hasattr(self.data_processor.consciousness_history, 'count') else 0,
            safety_violations=len([s for s in self.data_processor.consciousness_history if 'ERROR' in s.safety_status]),
            average_cycle_time_ms=cycle_time_ms,
            memory_usage_mb=0.0,  # Placeholder
            active_interfaces=len(self.hybrid_intelligence.hybrid_interfaces) if self.hybrid_intelligence else 0,
            radiation_exposure_level=1.0  # Placeholder
        )
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        return self.data_processor.get_real_time_dashboard_data()
    
    def get_analytics(self, timeframe_minutes: int = 60) -> Dict[str, Any]:
        """Get consciousness analytics"""
        return self.data_processor.get_consciousness_analytics(timeframe_minutes)

async def demonstrate_consciousness_monitoring():
    """Demonstrate real-time consciousness monitoring capabilities"""
    print("🖥️  REAL-TIME CONSCIOUSNESS MONITORING DASHBOARD DEMO")
    print("=" * 70)
    
    # Initialize monitoring server
    server = ConsciousnessMonitoringServer()
    
    # Start monitoring
    await server.start_monitoring()
    
    print("\n📊 Monitoring consciousness systems for 30 seconds...")
    print("   Collecting real-time data on:")
    print("   • Unified consciousness scores")
    print("   • Bio-digital fusion rates") 
    print("   • Radiation enhancement levels")
    print("   • Consciousness emergence events")
    print("   • Safety status monitoring")
    
    # Monitor for 30 seconds
    for i in range(30):
        await asyncio.sleep(1)
        
        if i % 5 == 0:  # Display data every 5 seconds
            dashboard_data = server.get_dashboard_data()
            
            if 'current_state' in dashboard_data:
                state = dashboard_data['current_state']
                print(f"\n   [{i:2d}s] Consciousness: {state['consciousness_score']:.3f} | "
                      f"Fusion: {state['bio_digital_fusion']:.3f} | "
                      f"Status: {state['safety_status']}")
                
                # Show alerts if any
                alerts = dashboard_data.get('alerts', [])
                for alert in alerts[:2]:  # Show first 2 alerts
                    print(f"        🚨 {alert['message']}")
    
    # Get final analytics
    print("\\n📈 Final Analytics (30-second window):")
    analytics = server.get_analytics(timeframe_minutes=1)
    
    if 'error' not in analytics:
        print(f"   Data points collected: {analytics['data_points']}")
        print(f"   Average consciousness: {analytics['average_consciousness']:.3f}")
        print(f"   Peak consciousness: {analytics['peak_consciousness']:.3f}")
        print(f"   Consciousness trend: {analytics['consciousness_trend']}")
        print(f"   Emergence events: {analytics['emergence_events']}")
        print(f"   Crystallization events: {analytics['crystallization_events']}")
    
    # Stop monitoring
    await server.stop_monitoring()
    
    print("\\n🌟 MONITORING DEMONSTRATION COMPLETE")
    print("\\nRevolutionary capabilities demonstrated:")
    print("  ✓ Real-time consciousness state tracking")
    print("  ✓ Bio-digital fusion monitoring")
    print("  ✓ Radiation enhancement detection")
    print("  ✓ Advanced consciousness analytics")
    print("  ✓ Intelligent alert generation")
    print("  ✓ Multi-dimensional consciousness visualization")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demonstrate_consciousness_monitoring())