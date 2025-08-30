#!/usr/bin/env python3
"""
Unit tests for the Sensory I/O System module
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch
import logging

# Add the core directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestSensoryIOSystem(unittest.TestCase):
    """Test cases for the Sensory I/O System"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        try:
            from core.sensory_io_system import SensoryIOSystem, SensorType
            self.sensory_system = SensoryIOSystem(sampling_rate=100)
            self.SensorType = SensorType
        except ImportError as e:
            self.skipTest(f"Sensory I/O System not available: {e}")
    
    def test_initialization(self):
        """Test that the Sensory I/O System initializes correctly"""
        self.assertIsNotNone(self.sensory_system)
        self.assertEqual(self.sensory_system.sampling_rate, 100)
        self.assertIsInstance(self.sensory_system.active_sensors, dict)
        self.assertIsInstance(self.sensory_system.sensory_history, list)
        
        # Check that all sensor types are active by default
        for sensor_type in self.SensorType:
            self.assertTrue(self.sensory_system.active_sensors.get(sensor_type, False))
    
    def test_calibration(self):
        """Test sensor calibration functionality"""
        calibration_data = {
            self.SensorType.TEMPERATURE: {
                'celsius': {'offset': 0.5, 'scale': 1.0},
                'fahrenheit': {'offset': 1.0, 'scale': 1.0}
            }
        }
        
        self.sensory_system.calibrate_sensors(calibration_data)
        self.assertEqual(self.sensory_system.calibration_data, calibration_data)
    
    def test_capture_sensory_data(self):
        """Test capturing sensory data from different sensor types"""
        # Test temperature data capture
        temp_data = {
            'celsius': 23.5,
            'fahrenheit': 74.3,
            'kelvin': 296.65
        }
        
        sensory_data = self.sensory_system.capture_sensory_data(
            sensor_type=self.SensorType.TEMPERATURE,
            raw_data=temp_data
        )
        
        self.assertEqual(sensory_data.sensor_type, self.SensorType.TEMPERATURE)
        self.assertEqual(sensory_data.values, temp_data)
        self.assertEqual(sensory_data.confidence, 1.0)  # Complete data should have full confidence
        
        # Test that data is added to history
        self.assertEqual(len(self.sensory_system.sensory_history), 1)
        self.assertEqual(self.sensory_system.sensory_history[0], sensory_data)
    
    def test_capture_sensory_data_with_calibration(self):
        """Test capturing sensory data with calibration applied"""
        # Set up calibration
        calibration_data = {
            self.SensorType.TEMPERATURE: {
                'celsius': {'offset': 0.5, 'scale': 1.0}
            }
        }
        self.sensory_system.calibrate_sensors(calibration_data)
        
        # Capture data
        raw_temp_data = {'celsius': 23.5}
        sensory_data = self.sensory_system.capture_sensory_data(
            sensor_type=self.SensorType.TEMPERATURE,
            raw_data=raw_temp_data
        )
        
        # Check that calibration was applied
        # calibrated_value = (raw_value - offset) * scale = (23.5 - 0.5) * 1.0 = 23.0
        self.assertEqual(sensory_data.values['celsius'], 23.0)
    
    def test_capture_sensory_data_inactive_sensor(self):
        """Test capturing data from an inactive sensor"""
        # Deactivate temperature sensor
        self.sensory_system.active_sensors[self.SensorType.TEMPERATURE] = False
        
        # Try to capture data
        temp_data = {'celsius': 23.5}
        sensory_data = self.sensory_system.capture_sensory_data(
            sensor_type=self.SensorType.TEMPERATURE,
            raw_data=temp_data
        )
        
        # Should return empty data with zero confidence
        self.assertEqual(sensory_data.values, {})
        self.assertEqual(sensory_data.confidence, 0.0)
    
    def test_confidence_calculation(self):
        """Test confidence calculation for sensor data"""
        # Test complete data (should have high confidence)
        complete_data = {
            'celsius': 23.5,
            'fahrenheit': 74.3,
            'kelvin': 296.65
        }
        
        sensory_data = self.sensory_system.capture_sensory_data(
            sensor_type=self.SensorType.TEMPERATURE,
            raw_data=complete_data
        )
        
        # Confidence should be high for complete data
        self.assertGreater(sensory_data.confidence, 0.8)
        
        # Test incomplete data (should have lower confidence)
        incomplete_data = {
            'celsius': 23.5
            # Missing fahrenheit and kelvin
        }
        
        sensory_data_incomplete = self.sensory_system.capture_sensory_data(
            sensor_type=self.SensorType.TEMPERATURE,
            raw_data=incomplete_data
        )
        
        # Confidence should be lower for incomplete data
        self.assertLess(sensory_data_incomplete.confidence, sensory_data.confidence)
    
    def test_get_recent_sensory_data(self):
        """Test retrieving recent sensory data"""
        # Add some test data
        temp_data = {'celsius': 23.5}
        humidity_data = {'relative_humidity': 65.2}
        
        self.sensory_system.capture_sensory_data(self.SensorType.TEMPERATURE, temp_data)
        self.sensory_system.capture_sensory_data(self.SensorType.HUMIDITY, humidity_data)
        
        # Get all recent data
        recent_data = self.sensory_system.get_recent_sensory_data()
        self.assertEqual(len(recent_data), 2)
        
        # Get specific sensor type data
        temp_recent = self.sensory_system.get_recent_sensory_data(self.SensorType.TEMPERATURE)
        self.assertEqual(len(temp_recent), 1)
        self.assertEqual(temp_recent[0].sensor_type, self.SensorType.TEMPERATURE)
    
    def test_fuse_multimodal_data(self):
        """Test fusing multi-modal sensory data"""
        # Add some test data
        temp_data = {'celsius': 23.5, 'fahrenheit': 74.3}
        humidity_data = {'relative_humidity': 65.2}
        
        self.sensory_system.capture_sensory_data(self.SensorType.TEMPERATURE, temp_data)
        self.sensory_system.capture_sensory_data(self.SensorType.HUMIDITY, humidity_data)
        
        # Fuse the data
        fused_result = self.sensory_system.fuse_multimodal_data()
        
        self.assertIn('fused_data', fused_result)
        self.assertIn('confidence', fused_result)
        self.assertGreater(fused_result['confidence'], 0)
        
        # Check that both sensor types are represented
        fused_data = fused_result['fused_data']
        self.assertIn('temperature', fused_data)
        self.assertIn('humidity', fused_data)
    
    def test_sensor_status(self):
        """Test getting sensor status"""
        # Add some test data
        temp_data = {'celsius': 23.5}
        self.sensory_system.capture_sensory_data(self.SensorType.TEMPERATURE, temp_data)
        
        # Get sensor status
        status = self.sensory_system.get_sensor_status()
        
        # Check that all sensor types are reported
        for sensor_type in self.SensorType:
            self.assertIn(sensor_type, status)
            self.assertIsInstance(status[sensor_type], dict)
            self.assertIn('active', status[sensor_type])
            self.assertIn('data_points', status[sensor_type])
            self.assertIn('avg_confidence', status[sensor_type])

class TestMultiModalFusionEngine(unittest.TestCase):
    """Test cases for the Multi-Modal Fusion Engine"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        try:
            from core.sensory_io_system import MultiModalFusionEngine, SensorType, SensoryData
            self.fusion_engine = MultiModalFusionEngine()
            self.SensorType = SensorType
            self.SensoryData = SensoryData
        except ImportError as e:
            self.skipTest(f"Multi-Modal Fusion Engine not available: {e}")
    
    def test_initialization(self):
        """Test that the Fusion Engine initializes correctly"""
        self.assertIsNotNone(self.fusion_engine)
        self.assertIsInstance(self.fusion_engine.fusion_weights, dict)
        
        # Check that all sensor types have weights
        for sensor_type in self.SensorType:
            self.assertIn(sensor_type, self.fusion_engine.fusion_weights)
    
    def test_fuse_sensory_inputs(self):
        """Test fusing sensory inputs"""
        from datetime import datetime
        
        # Create test sensory data
        temp_data = self.SensoryData(
            sensor_type=self.SensorType.TEMPERATURE,
            timestamp=datetime.now(),
            values={'celsius': 23.5, 'fahrenheit': 74.3},
            confidence=0.9
        )
        
        humidity_data = self.SensoryData(
            sensor_type=self.SensorType.HUMIDITY,
            timestamp=datetime.now(),
            values={'relative_humidity': 65.2},
            confidence=0.8
        )
        
        sensory_data_list = [temp_data, humidity_data]
        
        # Fuse the data
        fused_result = self.fusion_engine.fuse_sensory_inputs(sensory_data_list)
        
        self.assertIn('fused_data', fused_result)
        self.assertIn('confidence', fused_result)
        self.assertIn('total_data_points', fused_result)
        self.assertIn('sensor_types_represented', fused_result)
        
        self.assertEqual(fused_result['total_data_points'], 2)
        self.assertEqual(fused_result['sensor_types_represented'], 2)
    
    def test_fuse_empty_inputs(self):
        """Test fusing with empty inputs"""
        fused_result = self.fusion_engine.fuse_sensory_inputs([])
        
        self.assertEqual(fused_result['fused_data'], {})
        self.assertEqual(fused_result['confidence'], 0.0)
    
    def test_adjust_fusion_weights(self):
        """Test adjusting fusion weights"""
        original_weight = self.fusion_engine.fusion_weights[self.SensorType.TEMPERATURE]
        
        # Adjust weight
        new_weights = {self.SensorType.TEMPERATURE: 0.5}
        self.fusion_engine.adjust_fusion_weights(new_weights)
        
        # Check that weight was updated
        self.assertEqual(self.fusion_engine.fusion_weights[self.SensorType.TEMPERATURE], 0.5)

def main():
    """Run the tests"""
    unittest.main()

if __name__ == '__main__':
    main()