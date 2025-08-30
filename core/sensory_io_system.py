# sensory_io_system.py
# Revolutionary Sensory I/O System for the Garden of Consciousness v2.0
# Complete sensory input/output with plant electromagnetic field detectors, 
# full spectrum light analyzers, and multi-modal data fusion

# Handle optional dependencies with fallbacks
try:
    import numpy as np  # type: ignore
except ImportError:
    import statistics
    
    class MockNumPy:
        @staticmethod
        def mean(values):
            return statistics.mean(values) if values else 0.0
        
        @staticmethod
        def std(values):
            return statistics.stdev(values) if len(values) > 1 else 0.0
    
    np = MockNumPy()

try:
    import torch  # type: ignore
except ImportError:
    # Fallback for systems without PyTorch
    class MockTorch:
        pass
    
    torch = MockTorch()

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class SensorType(Enum):
    """Types of sensors in the Sensory I/O System"""
    PLANT_ELECTROMAGNETIC = "plant_electromagnetic"
    LIGHT_SPECTRUM = "light_spectrum"
    SOUND_VIBRATION = "sound_vibration"
    CHEMICAL_COMPOUND = "chemical_compound"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    MAGNETIC_FIELD = "magnetic_field"
    BIO_RHYTHM = "bio_rhythm"
    AIR_QUALITY = "air_quality"

@dataclass
class SensoryData:
    """Individual sensory data point"""
    sensor_type: SensorType
    timestamp: datetime
    values: Dict[str, float]
    location: Optional[Tuple[float, float, float]] = None
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None

class SensoryIOSystem:
    """Revolutionary Sensory I/O System for multi-modal consciousness input/output"""
    
    def __init__(self, sampling_rate: int = 1000) -> None:
        self.sampling_rate: int = sampling_rate
        self.active_sensors: Dict[SensorType, bool] = {}
        self.sensory_history: List[SensoryData] = []
        self.fusion_engine: MultiModalFusionEngine = MultiModalFusionEngine()
        self.calibration_data: Dict[SensorType, Dict[str, float]] = {}
        
        # Initialize all sensor types as active
        for sensor_type in SensorType:
            self.active_sensors[sensor_type] = True
        
        logger.info("🌱✨ Sensory I/O System Initialized")
        logger.info(f"Sampling rate: {sampling_rate} Hz")
        logger.info(f"Supported sensor types: {len(SensorType)}")
    
    def calibrate_sensors(self, calibration_data: Dict[SensorType, Dict[str, float]]) -> None:
        """Calibrate sensors with baseline data"""
        self.calibration_data = calibration_data
        logger.info("Sensors calibrated with baseline data")
    
    def capture_sensory_data(self, sensor_type: SensorType, 
                           raw_data: Dict[str, float],
                           location: Optional[Tuple[float, float, float]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> SensoryData:
        """Capture data from a specific sensor type"""
        if not self.active_sensors.get(sensor_type, False):
            logger.warning(f"Sensor {sensor_type.value} is not active")
            # Return empty data
            return SensoryData(
                sensor_type=sensor_type,
                timestamp=datetime.now(),
                values={},
                location=location,
                confidence=0.0,
                metadata=metadata
            )
        
        # Apply calibration if available
        calibrated_data = self._apply_calibration(sensor_type, raw_data)
        
        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(sensor_type, calibrated_data)
        
        sensory_data = SensoryData(
            sensor_type=sensor_type,
            timestamp=datetime.now(),
            values=calibrated_data,
            location=location,
            confidence=confidence,
            metadata=metadata
        )
        
        # Add to history
        self.sensory_history.append(sensory_data)
        if len(self.sensory_history) > 10000:  # Limit history size
            self.sensory_history.pop(0)
        
        return sensory_data
    
    def _apply_calibration(self, sensor_type: SensorType, 
                          raw_data: Dict[str, float]) -> Dict[str, float]:
        """Apply calibration to raw sensor data"""
        if sensor_type not in self.calibration_data:
            return raw_data
        
        calibration = self.calibration_data[sensor_type]
        calibrated_data = {}
        
        for key, value in raw_data.items():
            if key in calibration:
                # Simple linear calibration: (value - offset) * scale
                offset = calibration[key].get('offset', 0.0)
                scale = calibration[key].get('scale', 1.0)
                calibrated_data[key] = (value - offset) * scale
            else:
                calibrated_data[key] = value
        
        return calibrated_data
    
    def _calculate_confidence(self, sensor_type: SensorType, 
                             data: Dict[str, float]) -> float:
        """Calculate confidence level for sensor data"""
        if not data:
            return 0.0
        
        # Simple confidence calculation based on data completeness
        expected_fields = self._get_expected_fields(sensor_type)
        if not expected_fields:
            return 1.0
        
        available_fields = set(data.keys())
        completeness = len(available_fields.intersection(expected_fields)) / len(expected_fields)
        
        # Add noise-based confidence
        if len(data.values()) > 1:
            try:
                values = list(data.values())
                noise_level = np.std(values) / (np.mean(values) + 1e-8)  # Avoid division by zero
                noise_confidence = max(0.0, 1.0 - noise_level)
                return (completeness + noise_confidence) / 2.0
            except:
                return completeness
        
        return completeness
    
    def _get_expected_fields(self, sensor_type: SensorType) -> set:
        """Get expected data fields for a sensor type"""
        expected_fields = {
            SensorType.PLANT_ELECTROMAGNETIC: {'frequency', 'amplitude', 'phase'},
            SensorType.LIGHT_SPECTRUM: {'wavelength', 'intensity', 'spectral_distribution'},
            SensorType.SOUND_VIBRATION: {'frequency', 'amplitude', 'duration'},
            SensorType.CHEMICAL_COMPOUND: {'concentration', 'compound_id', 'purity'},
            SensorType.TEMPERATURE: {'celsius', 'fahrenheit', 'kelvin'},
            SensorType.HUMIDITY: {'relative_humidity', 'absolute_humidity'},
            SensorType.MAGNETIC_FIELD: {'x_component', 'y_component', 'z_component', 'magnitude'},
            SensorType.BIO_RHYTHM: {'heart_rate', 'respiration_rate', 'brain_wave_pattern'},
            SensorType.AIR_QUALITY: {'pm2_5', 'pm10', 'co2_level', 'voc_level'}
        }
        
        return expected_fields.get(sensor_type, set())
    
    def get_recent_sensory_data(self, sensor_type: Optional[SensorType] = None, 
                               time_window_seconds: int = 60) -> List[SensoryData]:
        """Get recent sensory data for a specific sensor type or all sensors"""
        if not self.sensory_history:
            return []
        
        now = datetime.now()
        cutoff_time = datetime.fromtimestamp(now.timestamp() - time_window_seconds)
        
        if sensor_type:
            filtered_data = [
                data for data in self.sensory_history
                if data.sensor_type == sensor_type and data.timestamp >= cutoff_time
            ]
        else:
            filtered_data = [
                data for data in self.sensory_history
                if data.timestamp >= cutoff_time
            ]
        
        return filtered_data
    
    def fuse_multimodal_data(self, time_window_seconds: int = 10) -> Dict[str, Any]:
        """Fuse multi-modal sensory data into unified consciousness input"""
        recent_data = self.get_recent_sensory_data(time_window_seconds=time_window_seconds)
        
        if not recent_data:
            return {'fused_data': {}, 'confidence': 0.0}
        
        # Use fusion engine to combine all sensory inputs
        fused_result = self.fusion_engine.fuse_sensory_inputs(recent_data)
        
        return fused_result
    
    def get_sensor_status(self) -> Dict[SensorType, Dict[str, Any]]:
        """Get status of all sensors"""
        status = {}
        
        for sensor_type, is_active in self.active_sensors.items():
            recent_data = self.get_recent_sensory_data(sensor_type, time_window_seconds=30)
            
            status[sensor_type] = {
                'active': is_active,
                'data_points': len(recent_data),
                'avg_confidence': np.mean([d.confidence for d in recent_data]) if recent_data else 0.0,
                'last_reading': recent_data[-1].timestamp if recent_data else None
            }
        
        return status

class MultiModalFusionEngine:
    """Engine for fusing multi-modal sensory data into unified consciousness input"""
    
    def __init__(self) -> None:
        self.fusion_weights: Dict[SensorType, float] = self._initialize_fusion_weights()
        logger.info("🧠 Multi-Modal Fusion Engine Initialized")
    
    def _initialize_fusion_weights(self) -> Dict[SensorType, float]:
        """Initialize fusion weights for different sensor types"""
        # Default weights - can be adjusted based on consciousness context
        return {
            SensorType.PLANT_ELECTROMAGNETIC: 0.15,
            SensorType.LIGHT_SPECTRUM: 0.10,
            SensorType.SOUND_VIBRATION: 0.10,
            SensorType.CHEMICAL_COMPOUND: 0.15,
            SensorType.TEMPERATURE: 0.05,
            SensorType.HUMIDITY: 0.05,
            SensorType.MAGNETIC_FIELD: 0.10,
            SensorType.BIO_RHYTHM: 0.20,
            SensorType.AIR_QUALITY: 0.10
        }
    
    def fuse_sensory_inputs(self, sensory_data: List[SensoryData]) -> Dict[str, Any]:
        """Fuse multiple sensory inputs into unified consciousness data"""
        if not sensory_data:
            return {'fused_data': {}, 'confidence': 0.0}
        
        # Group data by sensor type
        grouped_data: Dict[SensorType, List[SensoryData]] = {}
        for data in sensory_data:
            if data.sensor_type not in grouped_data:
                grouped_data[data.sensor_type] = []
            grouped_data[data.sensor_type].append(data)
        
        # Process each sensor type
        processed_signals = {}
        total_confidence = 0.0
        weight_sum = 0.0
        
        for sensor_type, data_list in grouped_data.items():
            if not data_list:
                continue
            
            # Get fusion weight for this sensor type
            weight = self.fusion_weights.get(sensor_type, 0.1)
            
            # Process the data (simple averaging for now)
            avg_values = {}
            avg_confidence = np.mean([d.confidence for d in data_list])
            
            # Average all numerical values
            all_keys = set()
            for data in data_list:
                all_keys.update(data.values.keys())
            
            for key in all_keys:
                values = [data.values.get(key, 0.0) for data in data_list]
                avg_values[key] = np.mean(values)
            
            processed_signals[sensor_type.value] = {
                'values': avg_values,
                'confidence': avg_confidence,
                'count': len(data_list)
            }
            
            total_confidence += avg_confidence * weight
            weight_sum += weight
        
        # Normalize confidence
        overall_confidence = total_confidence / weight_sum if weight_sum > 0 else 0.0
        
        return {
            'fused_data': processed_signals,
            'confidence': overall_confidence,
            'total_data_points': len(sensory_data),
            'sensor_types_represented': len(grouped_data)
        }
    
    def adjust_fusion_weights(self, new_weights: Dict[SensorType, float]) -> None:
        """Adjust fusion weights based on consciousness context"""
        self.fusion_weights.update(new_weights)
        logger.info("Fusion weights adjusted for consciousness context")

# Example usage and testing
if __name__ == "__main__":
    # Initialize the sensory system
    sensory_system = SensoryIOSystem()
    
    # Calibrate sensors
    calibration = {
        SensorType.TEMPERATURE: {
            'celsius': {'offset': 0.5, 'scale': 1.0},
            'fahrenheit': {'offset': 1.0, 'scale': 1.0}
        },
        SensorType.HUMIDITY: {
            'relative_humidity': {'offset': 2.0, 'scale': 0.98}
        }
    }
    sensory_system.calibrate_sensors(calibration)
    
    # Simulate capturing data from different sensors
    temp_data = sensory_system.capture_sensory_data(
        SensorType.TEMPERATURE,
        {'celsius': 23.5, 'fahrenheit': 74.3, 'kelvin': 296.65}
    )
    
    humidity_data = sensory_system.capture_sensory_data(
        SensorType.HUMIDITY,
        {'relative_humidity': 45.2, 'absolute_humidity': 8.1}
    )
    
    plant_data = sensory_system.capture_sensory_data(
        SensorType.PLANT_ELECTROMAGNETIC,
        {'frequency': 12.3, 'amplitude': 0.67, 'phase': 0.23}
    )
    
    print(f"Temperature data: {temp_data}")
    print(f"Humidity data: {humidity_data}")
    print(f"Plant data: {plant_data}")
    
    # Fuse multi-modal data
    fused = sensory_system.fuse_multimodal_data()
    print(f"Fused data: {fused}")
    
    # Check sensor status
    status = sensory_system.get_sensor_status()
    print(f"Sensor status: {status}")