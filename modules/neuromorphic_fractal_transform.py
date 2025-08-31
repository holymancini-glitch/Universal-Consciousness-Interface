# neuromorphic_fractal_transform.py
# Neuromorphic-to-fractal transformation for biological signal conversion

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, Union
from scipy import signal
from scipy.fftpack import fft, ifft
import warnings

class NeuromorphicFractalTransform:
    """Converts neuromorphic signals to fractal representations for analysis and decision-making."""
    
    def __init__(self, input_dim: int = 512, fractal_dim: int = 256, device: str = "cpu"):
        """
        Initialize the neuromorphic-to-fractal transformation system.
        
        Args:
            input_dim: Dimension of input neuromorphic signals
            fractal_dim: Dimension of output fractal representations
            device: Computing device ('cpu' or 'cuda')
        """
        self.input_dim = input_dim
        self.fractal_dim = fractal_dim
        self.device = torch.device(device)
        
        # Transformation networks
        self.signal_encoder = self._create_signal_encoder()
        self.fractal_generator = self._create_fractal_generator()
        self.attention_mechanism = self._create_attention_mechanism()
        
        # Fractal parameters
        self.fractal_parameters = {
            'scale': 1.0,
            'rotation': 0.0,
            'complexity': 1.0
        }
        
        # History tracking
        self.transformation_history = []
        self.coherence_metrics = []
        
    def _create_signal_encoder(self) -> nn.Module:
        """Create neural network for encoding neuromorphic signals."""
        encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, self.fractal_dim)
        )
        return encoder.to(self.device)
    
    def _create_fractal_generator(self) -> nn.Module:
        """Create neural network for generating fractal patterns."""
        generator = nn.Sequential(
            nn.Linear(self.fractal_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, self.fractal_dim),
            nn.Tanh()
        )
        return generator.to(self.device)
    
    def _create_attention_mechanism(self) -> nn.Module:
        """Create attention mechanism for focusing on relevant signal components."""
        attention = nn.Sequential(
            nn.Linear(self.fractal_dim, 64),
            nn.Tanh(),
            nn.Linear(64, self.fractal_dim),
            nn.Softmax(dim=-1)
        )
        return attention.to(self.device)
    
    def preprocess_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Preprocess neuromorphic signal data.
        
        Args:
            signal_data: Raw neuromorphic signal data
            
        Returns:
            processed_signal: Preprocessed signal
        """
        # Ensure correct dimensionality
        if len(signal_data.shape) == 1:
            signal_data = signal_data.reshape(1, -1)
        
        # Pad or truncate to input_dim
        if signal_data.shape[1] < self.input_dim:
            padding = np.zeros((signal_data.shape[0], self.input_dim - signal_data.shape[1]))
            signal_data = np.concatenate([signal_data, padding], axis=1)
        elif signal_data.shape[1] > self.input_dim:
            signal_data = signal_data[:, :self.input_dim]
        
        # Normalize signal
        signal_data = (signal_data - np.mean(signal_data, axis=1, keepdims=True)) / (
            np.std(signal_data, axis=1, keepdims=True) + 1e-8)
        
        return signal_data
    
    def extract_features(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Extract relevant features from neuromorphic signals.
        
        Args:
            signal_data: Preprocessed signal data
            
        Returns:
            features: Extracted feature vector
        """
        # Convert to torch tensor
        signal_tensor = torch.FloatTensor(signal_data).to(self.device)
        
        # Extract features through encoder
        with torch.no_grad():
            features = self.signal_encoder(signal_tensor)
            
        return features.cpu().numpy()
    
    def generate_fractal_representation(self, features: np.ndarray) -> np.ndarray:
        """
        Generate fractal representation from extracted features.
        
        Args:
            features: Extracted feature vector
            
        Returns:
            fractal_repr: Fractal representation
        """
        # Convert to torch tensor
        feature_tensor = torch.FloatTensor(features).to(self.device)
        
        # Generate fractal pattern
        with torch.no_grad():
            fractal_pattern = self.fractal_generator(feature_tensor)
            
        # Apply attention mechanism
        attention_weights = self.attention_mechanism(fractal_pattern)
        fractal_repr = fractal_pattern * attention_weights
        
        return fractal_repr.cpu().numpy()
    
    def apply_fractal_transformations(self, fractal_data: np.ndarray) -> np.ndarray:
        """
        Apply fractal transformations to enhance representation.
        
        Args:
            fractal_data: Base fractal representation
            
        Returns:
            transformed_data: Transformed fractal data
        """
        # Apply scale transformation
        scaled_data = fractal_data * self.fractal_parameters['scale']
        
        # Apply rotation in complex plane (if applicable)
        if fractal_data.shape[-1] % 2 == 0:
            # Treat pairs as complex numbers for rotation
            complex_data = scaled_data.reshape(-1, 2)
            angle = self.fractal_parameters['rotation']
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            rotated_data = complex_data @ rotation_matrix.T
            scaled_data = rotated_data.reshape(fractal_data.shape)
        
        # Apply complexity modulation through power transformation
        complexity = self.fractal_parameters['complexity']
        transformed_data = np.sign(scaled_data) * np.power(np.abs(scaled_data), complexity)
        
        return transformed_data
    
    def compute_fractal_coherence(self, fractal_data: np.ndarray) -> float:
        """
        Compute coherence of fractal representation.
        
        Args:
            fractal_data: Fractal representation
            
        Returns:
            coherence: Coherence metric
        """
        # Compute self-similarity as coherence measure
        if len(fractal_data.shape) == 1:
            fractal_data = fractal_data.reshape(1, -1)
        
        # Normalize data
        norm_data = fractal_data / (np.linalg.norm(fractal_data, axis=1, keepdims=True) + 1e-8)
        
        # Compute autocorrelation as coherence measure
        autocorr = np.correlate(norm_data[0], norm_data[0], mode='full')
        coherence = np.max(autocorr[len(autocorr)//2+1:])  # Max correlation at non-zero lag
        
        return float(coherence)
    
    def transform_signal(self, signal_data: np.ndarray, 
                        update_parameters: bool = True) -> Dict[str, Union[np.ndarray, float]]:
        """
        Complete transformation pipeline from neuromorphic signal to fractal representation.
        
        Args:
            signal_data: Raw neuromorphic signal data
            update_parameters: Whether to update transformation parameters based on coherence
            
        Returns:
            result: Dictionary containing transformation results
        """
        # Preprocess signal
        processed_signal = self.preprocess_signal(signal_data)
        
        # Extract features
        features = self.extract_features(processed_signal)
        
        # Generate fractal representation
        fractal_repr = self.generate_fractal_representation(features)
        
        # Apply fractal transformations
        transformed_fractal = self.apply_fractal_transformations(fractal_repr)
        
        # Compute coherence
        coherence = self.compute_fractal_coherence(transformed_fractal)
        
        # Update parameters if requested
        if update_parameters:
            self._update_parameters(coherence)
        
        # Store in history
        transformation_record = {
            'input_signal': processed_signal.copy(),
            'features': features.copy(),
            'fractal_representation': transformed_fractal.copy(),
            'coherence': coherence
        }
        self.transformation_history.append(transformation_record)
        self.coherence_metrics.append(coherence)
        
        return {
            'fractal_representation': transformed_fractal,
            'coherence': coherence,
            'features': features,
            'preprocessed_signal': processed_signal
        }
    
    def _update_parameters(self, coherence: float):
        """
        Update transformation parameters based on coherence feedback.
        
        Args:
            coherence: Current coherence metric
        """
        # Simple adaptive parameter update
        if len(self.coherence_metrics) > 1:
            coherence_trend = coherence - self.coherence_metrics[-2]
            
            # Adjust scale based on coherence trend
            if coherence_trend > 0.01:  # Improving coherence
                self.fractal_parameters['scale'] *= 1.01
            elif coherence_trend < -0.01:  # Degrading coherence
                self.fractal_parameters['scale'] *= 0.99
            
            # Adjust complexity based on current coherence
            if coherence > 0.8:  # High coherence
                self.fractal_parameters['complexity'] = min(1.2, self.fractal_parameters['complexity'] * 1.005)
            elif coherence < 0.3:  # Low coherence
                self.fractal_parameters['complexity'] = max(0.8, self.fractal_parameters['complexity'] * 0.995)
        
        # Keep parameters within reasonable bounds
        self.fractal_parameters['scale'] = np.clip(self.fractal_parameters['scale'], 0.1, 10.0)
        self.fractal_parameters['complexity'] = np.clip(self.fractal_parameters['complexity'], 0.5, 2.0)
    
    def batch_transform(self, signal_batch: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Transform a batch of neuromorphic signals.
        
        Args:
            signal_batch: Batch of signal data (batch_size, signal_dim)
            
        Returns:
            batch_results: Dictionary containing batch transformation results
        """
        batch_size = signal_batch.shape[0]
        fractal_representations = []
        coherences = []
        features_list = []
        
        for i in range(batch_size):
            signal_slice = signal_batch[i:i+1]  # Maintain batch dimension
            result = self.transform_signal(signal_slice, update_parameters=False)
            fractal_representations.append(result['fractal_representation'])
            coherences.append(result['coherence'])
            features_list.append(result['features'])
        
        # Update parameters based on batch coherence
        avg_coherence = np.mean(coherences)
        self._update_parameters(avg_coherence)
        
        return {
            'fractal_representations': np.concatenate(fractal_representations, axis=0),
            'coherences': np.array(coherences),
            'features': np.concatenate(features_list, axis=0)
        }
    
    def get_transformation_metrics(self) -> Dict[str, float]:
        """Get current transformation metrics."""
        if not self.coherence_metrics:
            return {'avg_coherence': 0.0, 'coherence_trend': 0.0}
        
        avg_coherence = np.mean(self.coherence_metrics[-100:])  # Average of last 100
        if len(self.coherence_metrics) > 1:
            coherence_trend = self.coherence_metrics[-1] - np.mean(self.coherence_metrics[-10:-1])
        else:
            coherence_trend = 0.0
            
        return {
            'avg_coherence': float(avg_coherence),
            'coherence_trend': float(coherence_trend),
            'current_coherence': self.coherence_metrics[-1] if self.coherence_metrics else 0.0,
            'scale_parameter': self.fractal_parameters['scale'],
            'complexity_parameter': self.fractal_parameters['complexity']
        }
    
    def reset(self):
        """Reset the transformation system."""
        self.transformation_history.clear()
        self.coherence_metrics.clear()
        self.fractal_parameters = {
            'scale': 1.0,
            'rotation': 0.0,
            'complexity': 1.0
        }

# Example usage
if __name__ == "__main__":
    # Initialize transformation system
    transform = NeuromorphicFractalTransform(input_dim=256, fractal_dim=128)
    
    # Create test signal
    test_signal = np.random.randn(1, 256).astype(np.float32)
    
    # Transform signal
    result = transform.transform_signal(test_signal)
    
    print(f"Input signal shape: {test_signal.shape}")
    print(f"Fractal representation shape: {result['fractal_representation'].shape}")
    print(f"Coherence: {result['coherence']:.4f}")
    print(f"Features shape: {result['features'].shape}")
    
    # Test batch transformation
    batch_signal = np.random.randn(5, 256).astype(np.float32)
    batch_result = transform.batch_transform(batch_signal)
    
    print(f"\nBatch transformation:")
    print(f"Batch fractal representations shape: {batch_result['fractal_representations'].shape}")
    print(f"Batch coherences: {batch_result['coherences']}")
    
    # Test metrics
    metrics = transform.get_transformation_metrics()
    print(f"\nTransformation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")