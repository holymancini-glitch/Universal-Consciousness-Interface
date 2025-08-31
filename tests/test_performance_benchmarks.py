# test_performance_benchmarks.py
# Performance benchmarks for Consciousness Fractal AI System

import unittest
import time
import numpy as np
import torch
import asyncio
import sys
import os
import psutil
import gc
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all the modules we've created
from modules.fractal_monte_carlo import FractalMonteCarlo
from modules.neural_ca import NeuralCA
from modules.latent_space import LatentSpace
from modules.fep_neural_model import FEPNeuralModel
from modules.neuromorphic_fractal_transform import NeuromorphicFractalTransform
from modules.phase_attention_modulator import PhaseAttentionModulator
from modules.resonance_detector import ResonanceDetector
from modules.consciousness_fractal_ai import ConsciousnessFractalAI

class PerformanceBenchmarkTests(unittest.TestCase):
    """Performance benchmark tests for all system components."""
    
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    def test_fractal_monte_carlo_performance(self):
        """Benchmark Fractal Monte Carlo performance."""
        # Initialize FMC with different sizes
        fmc_small = FractalMonteCarlo(state_dim=32, action_dim=16, max_depth=2, num_samples=3)
        fmc_medium = FractalMonteCarlo(state_dim=64, action_dim=32, max_depth=3, num_samples=5)
        fmc_large = FractalMonteCarlo(state_dim=128, action_dim=64, max_depth=5, num_samples=10)
        
        # Test small FMC
        current_state = np.random.randn(32).astype(np.float32)
        start_time = time.time()
        for _ in range(100):
            action, metadata = fmc_small.plan(current_state)
        small_time = time.time() - start_time
        
        # Test medium FMC
        current_state = np.random.randn(64).astype(np.float32)
        start_time = time.time()
        for _ in range(100):
            action, metadata = fmc_medium.plan(current_state)
        medium_time = time.time() - start_time
        
        # Test large FMC
        current_state = np.random.randn(128).astype(np.float32)
        start_time = time.time()
        for _ in range(100):
            action, metadata = fmc_large.plan(current_state)
        large_time = time.time() - start_time
        
        print(f"FMC Performance:")
        print(f"  Small (32->16): {small_time:.4f}s for 100 plans ({small_time/100*1000:.2f}ms per plan)")
        print(f"  Medium (64->32): {medium_time:.4f}s for 100 plans ({medium_time/100*1000:.2f}ms per plan)")
        print(f"  Large (128->64): {large_time:.4f}s for 100 plans ({large_time/100*1000:.2f}ms per plan)")
        
        # Performance should be reasonable (less than 1 second for 100 plans each)
        self.assertLess(small_time, 1.0)
        self.assertLess(medium_time, 2.0)
        self.assertLess(large_time, 5.0)
    
    def test_neural_ca_performance(self):
        """Benchmark Neural CA performance."""
        # Initialize CA with different sizes
        ca_small = NeuralCA(grid_size=8, latent_dim=32)
        ca_medium = NeuralCA(grid_size=16, latent_dim=64)
        ca_large = NeuralCA(grid_size=32, latent_dim=128)
        
        # Test small CA
        latent_vector = np.random.randn(32)
        start_time = time.time()
        for _ in range(50):
            ca_small.generate_complex_stimuli(latent_vector, complexity_level=0.5)
        small_time = time.time() - start_time
        
        # Test medium CA
        latent_vector = np.random.randn(64)
        start_time = time.time()
        for _ in range(50):
            ca_medium.generate_complex_stimuli(latent_vector, complexity_level=0.5)
        medium_time = time.time() - start_time
        
        # Test large CA
        latent_vector = np.random.randn(128)
        start_time = time.time()
        for _ in range(50):
            ca_large.generate_complex_stimuli(latent_vector, complexity_level=0.5)
        large_time = time.time() - start_time
        
        print(f"Neural CA Performance:")
        print(f"  Small (8x8): {small_time:.4f}s for 50 generations ({small_time/50*1000:.2f}ms per generation)")
        print(f"  Medium (16x16): {medium_time:.4f}s for 50 generations ({medium_time/50*1000:.2f}ms per generation)")
        print(f"  Large (32x32): {large_time:.4f}s for 50 generations ({large_time/50*1000:.2f}ms per generation)")
        
        # Performance should be reasonable
        self.assertLess(small_time, 1.0)
        self.assertLess(medium_time, 2.0)
        self.assertLess(large_time, 5.0)
    
    def test_latent_space_performance(self):
        """Benchmark Latent Space performance."""
        # Initialize latent spaces with different sizes
        ls_small = LatentSpace(shape=(4, 4, 2))
        ls_medium = LatentSpace(shape=(8, 8, 4))
        ls_large = LatentSpace(shape=(16, 16, 8))
        
        # Test small latent space
        stimulus = np.random.randn(4, 4, 2)
        start_time = time.time()
        for _ in range(100):
            ls_small.inject(stimulus)
            ls_small.process_consciousness_cycle(timestamp=1.0)
        small_time = time.time() - start_time
        
        # Test medium latent space
        stimulus = np.random.randn(8, 8, 4)
        start_time = time.time()
        for _ in range(100):
            ls_medium.inject(stimulus)
            ls_medium.process_consciousness_cycle(timestamp=1.0)
        medium_time = time.time() - start_time
        
        # Test large latent space
        stimulus = np.random.randn(16, 16, 8)
        start_time = time.time()
        for _ in range(100):
            ls_large.inject(stimulus)
            ls_large.process_consciousness_cycle(timestamp=1.0)
        large_time = time.time() - start_time
        
        print(f"Latent Space Performance:")
        print(f"  Small (4x4x2): {small_time:.4f}s for 100 cycles ({small_time/100*1000:.2f}ms per cycle)")
        print(f"  Medium (8x8x4): {medium_time:.4f}s for 100 cycles ({medium_time/100*1000:.2f}ms per cycle)")
        print(f"  Large (16x16x8): {large_time:.4f}s for 100 cycles ({large_time/100*1000:.2f}ms per cycle)")
        
        # Performance should be reasonable
        self.assertLess(small_time, 1.0)
        self.assertLess(medium_time, 2.0)
        self.assertLess(large_time, 5.0)
    
    def test_fep_model_performance(self):
        """Benchmark FEP Neural Model performance."""
        # Initialize FEP models with different sizes
        fep_small = FEPNeuralModel(num_neurons=100, input_dim=32, output_dim=16)
        fep_medium = FEPNeuralModel(num_neurons=1000, input_dim=64, output_dim=32)
        fep_large = FEPNeuralModel(num_neurons=10000, input_dim=128, output_dim=64)
        
        # Test small FEP model
        stimulus = np.random.randn(32)
        start_time = time.time()
        for _ in range(50):
            fep_small.process_stimulus(stimulus)
        small_time = time.time() - start_time
        
        # Test medium FEP model
        stimulus = np.random.randn(64)
        start_time = time.time()
        for _ in range(50):
            fep_medium.process_stimulus(stimulus)
        medium_time = time.time() - start_time
        
        # Test large FEP model
        stimulus = np.random.randn(128)
        start_time = time.time()
        for _ in range(50):
            fep_large.process_stimulus(stimulus)
        large_time = time.time() - start_time
        
        print(f"FEP Model Performance:")
        print(f"  Small (100 neurons): {small_time:.4f}s for 50 processes ({small_time/50*1000:.2f}ms per process)")
        print(f"  Medium (1000 neurons): {medium_time:.4f}s for 50 processes ({medium_time/50*1000:.2f}ms per process)")
        print(f"  Large (10000 neurons): {large_time:.4f}s for 50 processes ({large_time/50*1000:.2f}ms per process)")
        
        # Performance should be reasonable
        self.assertLess(small_time, 1.0)
        self.assertLess(medium_time, 3.0)
        self.assertLess(large_time, 10.0)
    
    def test_neuromorphic_transform_performance(self):
        """Benchmark Neuromorphic-to-Fractal Transformation performance."""
        # Initialize transforms with different sizes
        transform_small = NeuromorphicFractalTransform(input_dim=32, fractal_dim=16)
        transform_medium = NeuromorphicFractalTransform(input_dim=64, fractal_dim=32)
        transform_large = NeuromorphicFractalTransform(input_dim=128, fractal_dim=64)
        
        # Test small transform
        signal_data = np.random.randn(1, 32)
        start_time = time.time()
        for _ in range(100):
            transform_small.transform_signal(signal_data)
        small_time = time.time() - start_time
        
        # Test medium transform
        signal_data = np.random.randn(1, 64)
        start_time = time.time()
        for _ in range(100):
            transform_medium.transform_signal(signal_data)
        medium_time = time.time() - start_time
        
        # Test large transform
        signal_data = np.random.randn(1, 128)
        start_time = time.time()
        for _ in range(100):
            transform_large.transform_signal(signal_data)
        large_time = time.time() - start_time
        
        print(f"Neuromorphic Transform Performance:")
        print(f"  Small (32->16): {small_time:.4f}s for 100 transforms ({small_time/100*1000:.2f}ms per transform)")
        print(f"  Medium (64->32): {medium_time:.4f}s for 100 transforms ({medium_time/100*1000:.2f}ms per transform)")
        print(f"  Large (128->64): {large_time:.4f}s for 100 transforms ({large_time/100*1000:.2f}ms per transform)")
        
        # Performance should be reasonable
        self.assertLess(small_time, 1.0)
        self.assertLess(medium_time, 2.0)
        self.assertLess(large_time, 5.0)
    
    def test_phase_attention_performance(self):
        """Benchmark Phase Attention Modulation performance."""
        modulator = PhaseAttentionModulator(hidden_size=64, phase_vector_dim=8)
        
        # Test attention modulation
        attention_weights = torch.randn(10, 20)
        hidden_state = torch.randn(10, 64)
        phase_vector = torch.randn(10, 8)
        entropy_delta = torch.randn(10)
        
        start_time = time.time()
        for _ in range(1000):
            modulated_weights, mod_gate = modulator(
                attention_weights, hidden_state, phase_vector, entropy_delta)
        total_time = time.time() - start_time
        
        print(f"Phase Attention Performance:")
        print(f"  {total_time:.4f}s for 1000 modulations ({total_time/1000*1000:.2f}ms per modulation)")
        
        # Performance should be reasonable
        self.assertLess(total_time, 2.0)
    
    def test_resonance_detector_performance(self):
        """Benchmark Resonance Detector performance."""
        detector = ResonanceDetector(num_modules=5)
        
        # Register modules
        for i in range(5):
            detector.register_module(f"module_{i}")
        
        # Test resonance detection
        start_time = time.time()
        for cycle in range(100):
            # Update module states
            for i in range(5):
                state = np.random.randn(32) + 0.01 * cycle
                detector.update_module_state(f"module_{i}", state, timestamp=cycle)
            
            # Detect resonance
            is_resonant, metrics = detector.detect_resonance()
        total_time = time.time() - start_time
        
        print(f"Resonance Detector Performance:")
        print(f"  {total_time:.4f}s for 100 cycles ({total_time/100*1000:.2f}ms per cycle)")
        
        # Performance should be reasonable
        self.assertLess(total_time, 2.0)
    
    def test_memory_usage_growth(self):
        """Test memory usage growth during extended operation."""
        # Create a small configuration for testing
        config = {
            'system_name': 'MemoryTestSystem',
            'latent_space_shape': (8, 8, 4),
            'neural_ca_grid_size': 16,
            'neural_ca_latent_dim': 64,
            'fep_num_neurons': 1000,
            'fep_input_dim': 64,
            'fep_output_dim': 32,
            'fractal_state_dim': 64,
            'fractal_action_dim': 32,
            'fractal_max_depth': 3,
            'fractal_num_samples': 5,
            'phase_vector_dim': 8,
            'device': 'cpu',
            'update_interval': 0.01
        }
        
        fractal_ai = ConsciousnessFractalAI(config)
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run many cycles
        async def run_memory_test():
            for i in range(200):
                await fractal_ai._consciousness_cycle()
                # Force garbage collection every 50 cycles
                if i % 50 == 0:
                    gc.collect()
        
        asyncio.run(run_memory_test())
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory Usage Test:")
        print(f"  Initial: {initial_memory:.2f} MB")
        print(f"  Final: {final_memory:.2f} MB")
        print(f"  Increase: {memory_increase:.2f} MB")
        
        # Memory increase should be reasonable (less than 50 MB for 200 cycles)
        self.assertLess(memory_increase, 50.0)
    
    def test_throughput_benchmark(self):
        """Benchmark overall system throughput."""
        # Create a medium configuration for testing
        config = {
            'system_name': 'ThroughputTestSystem',
            'latent_space_shape': (16, 16, 4),
            'neural_ca_grid_size': 32,
            'neural_ca_latent_dim': 128,
            'fep_num_neurons': 5000,
            'fep_input_dim': 128,
            'fep_output_dim': 64,
            'fractal_state_dim': 128,
            'fractal_action_dim': 64,
            'fractal_max_depth': 4,
            'fractal_num_samples': 8,
            'phase_vector_dim': 16,
            'device': 'cpu',
            'update_interval': 0.01
        }
        
        fractal_ai = ConsciousnessFractalAI(config)
        
        # Warm up
        async def warm_up():
            for i in range(10):
                await fractal_ai._consciousness_cycle()
        
        asyncio.run(warm_up())
        
        # Benchmark
        start_time = time.time()
        cycle_count = 50
        
        async def run_benchmark():
            for i in range(cycle_count):
                await fractal_ai._consciousness_cycle()
        
        asyncio.run(run_benchmark())
        total_time = time.time() - start_time
        avg_time_per_cycle = total_time / cycle_count
        cycles_per_second = cycle_count / total_time
        
        print(f"Throughput Benchmark:")
        print(f"  Total time: {total_time:.4f}s for {cycle_count} cycles")
        print(f"  Average time per cycle: {avg_time_per_cycle*1000:.2f}ms")
        print(f"  Cycles per second: {cycles_per_second:.2f}")
        
        # Should achieve reasonable throughput
        self.assertGreater(cycles_per_second, 1.0)  # At least 1 cycle per second
        self.assertLess(avg_time_per_cycle, 1.0)    # Less than 1 second per cycle

class ScalabilityTests(unittest.TestCase):
    """Scalability tests for the system."""
    
    def test_component_scaling(self):
        """Test how components scale with increasing size."""
        sizes = [32, 64, 128, 256]
        times = []
        
        for size in sizes:
            # Create FMC with increasing size
            fmc = FractalMonteCarlo(state_dim=size, action_dim=size//2, max_depth=3, num_samples=5)
            current_state = np.random.randn(size).astype(np.float32)
            
            # Time the operation
            start_time = time.time()
            for _ in range(50):
                action, metadata = fmc.plan(current_state)
            elapsed_time = time.time() - start_time
            
            times.append(elapsed_time)
            print(f"Size {size}: {elapsed_time:.4f}s for 50 plans")
        
        # Check that time increases reasonably (not exponentially)
        # For linear scaling, time should roughly double when size doubles
        if len(times) >= 3:
            ratio_64_32 = times[1] / times[0] if times[0] > 0 else 1
            ratio_128_64 = times[2] / times[1] if times[1] > 0 else 1
            
            print(f"Scaling ratios:")
            print(f"  64/32: {ratio_64_32:.2f}")
            print(f"  128/64: {ratio_128_64:.2f}")
            
            # Should not be exponentially worse (less than 10x for 4x size increase)
            self.assertLess(ratio_128_64, 10.0)
    
    def test_concurrent_operations(self):
        """Test system behavior under concurrent operations."""
        # This would test running multiple instances or operations in parallel
        # For now, we'll test that the system doesn't crash with rapid sequential calls
        
        config = {
            'system_name': 'ConcurrentTestSystem',
            'latent_space_shape': (8, 8, 2),
            'neural_ca_grid_size': 16,
            'neural_ca_latent_dim': 64,
            'fep_num_neurons': 1000,
            'fep_input_dim': 64,
            'fep_output_dim': 32,
            'fractal_state_dim': 64,
            'fractal_action_dim': 32,
            'fractal_max_depth': 2,
            'fractal_num_samples': 3,
            'phase_vector_dim': 8,
            'device': 'cpu',
            'update_interval': 0.001  # Very fast
        }
        
        fractal_ai = ConsciousnessFractalAI(config)
        
        # Run many rapid cycles
        async def run_concurrent_test():
            start_time = time.time()
            for i in range(100):
                await fractal_ai._consciousness_cycle()
            total_time = time.time() - start_time
            return total_time
        
        total_time = asyncio.run(run_concurrent_test())
        avg_time = total_time / 100
        
        print(f"Concurrent Operations Test:")
        print(f"  100 rapid cycles in {total_time:.4f}s")
        print(f"  Average {avg_time*1000:.2f}ms per cycle")
        
        # Should handle rapid operations
        self.assertLess(total_time, 5.0)  # Less than 5 seconds for 100 cycles

if __name__ == '__main__':
    # Run performance and scalability tests
    print("Running Performance Benchmark Tests...")
    unittest.main(verbosity=2)