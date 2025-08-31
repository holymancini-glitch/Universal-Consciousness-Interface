# fractal_monte_carlo.py
# Fractal Monte Carlo (FMC) implementation for forward-thinking planning

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from collections import deque

@dataclass
class TrajectoryNode:
    """Represents a node in a fractal trajectory."""
    state: np.ndarray
    action: Optional[np.ndarray]
    reward: float
    depth: int
    parent: Optional['TrajectoryNode'] = None
    children: List['TrajectoryNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

class FractalTrajectorySampler:
    """Samples fractal trajectories for forward-thinking planning."""
    
    def __init__(self, state_dim: int, action_dim: int, max_depth: int = 5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_depth = max_depth
        self.trajectory_cache = deque(maxlen=1000)
        
    def sample_action(self, state: np.ndarray) -> np.ndarray:
        """Sample a random action based on the current state."""
        # Simple random action sampling
        return np.random.randn(self.action_dim).astype(np.float32)
    
    def simulate_step(self, state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, float]:
        """Simulate a single step in the environment."""
        # Simple transition model - in a real implementation, this would interface with the environment
        next_state = np.tanh(state + 0.1 * action + 0.01 * np.random.randn(*state.shape))
        reward = -np.mean(np.abs(next_state))  # Simple reward function
        return next_state, reward
    
    def generate_fractal_trajectory(self, initial_state: np.ndarray) -> TrajectoryNode:
        """Generate a fractal trajectory starting from the initial state."""
        root = TrajectoryNode(
            state=initial_state.copy(),
            action=None,
            reward=0.0,
            depth=0
        )
        
        def expand_node(node: TrajectoryNode):
            if node.depth >= self.max_depth:
                return
            
            # Sample multiple actions to create fractal branching
            for _ in range(3):  # Fractal branching factor
                action = self.sample_action(node.state)
                next_state, reward = self.simulate_step(node.state, action)
                
                child = TrajectoryNode(
                    state=next_state,
                    action=action,
                    reward=reward,
                    depth=node.depth + 1,
                    parent=node
                )
                
                node.children.append(child)
                expand_node(child)
        
        expand_node(root)
        self.trajectory_cache.append(root)
        return root
    
    def evaluate_trajectory(self, trajectory: TrajectoryNode) -> float:
        """Evaluate a trajectory using coherence metrics."""
        # Simple evaluation based on cumulative reward and state coherence
        total_reward = 0.0
        node_count = 0
        current = trajectory
        
        # Traverse the trajectory and calculate metrics
        while current:
            total_reward += current.reward
            node_count += 1
            
            # Calculate coherence with parent (if exists)
            if current.parent:
                coherence = 1.0 - np.mean(np.abs(current.state - current.parent.state))
                total_reward += 0.1 * coherence  # Reward coherence
            
            # Move to children (take first child for simplicity)
            current = current.children[0] if current.children else None
        
        return total_reward / max(node_count, 1)

class FractalMonteCarlo:
    """Fractal Monte Carlo (FMC) implementation for forward-thinking planning."""
    
    def __init__(self, state_dim: int, action_dim: int, max_depth: int = 5, num_samples: int = 10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_depth = max_depth
        self.num_samples = num_samples
        self.trajectory_sampler = FractalTrajectorySampler(state_dim, action_dim, max_depth)
        self.best_trajectories = deque(maxlen=100)
        
    def plan(self, current_state: np.ndarray) -> Tuple[np.ndarray, Dict[str, Union[float, List[float]]]]:
        """Plan the next action using Fractal Monte Carlo."""
        trajectories = []
        evaluations = []
        
        # Generate multiple fractal trajectories
        for _ in range(self.num_samples):
            trajectory = self.trajectory_sampler.generate_fractal_trajectory(current_state)
            evaluation = self.trajectory_sampler.evaluate_trajectory(trajectory)
            trajectories.append(trajectory)
            evaluations.append(evaluation)
        
        # Find the best trajectory
        best_idx = np.argmax(evaluations)
        best_trajectory = trajectories[best_idx]
        best_evaluation = evaluations[best_idx]
        
        # Store best trajectories
        self.best_trajectories.append((best_trajectory, best_evaluation))
        
        # Extract the first action from the best trajectory
        first_action = best_trajectory.children[0].action if best_trajectory.children else np.zeros(self.action_dim)
        
        # Return action and metadata
        metadata = {
            'evaluation_score': best_evaluation,
            'all_evaluations': evaluations,
            'trajectory_depth': best_trajectory.depth if best_trajectory else 0
        }
        
        return first_action, metadata
    
    def adapt_horizon(self, recent_performance: List[float]) -> int:
        """Adaptively adjust the planning horizon based on recent performance."""
        if len(recent_performance) < 5:
            return self.max_depth
        
        # Calculate performance trend
        trend = np.mean(recent_performance[-3:]) - np.mean(recent_performance[-5:-3])
        
        # Adjust horizon based on trend
        if trend > 0.1:  # Improving performance
            new_depth = min(self.max_depth + 1, 10)
        elif trend < -0.1:  # Degrading performance
            new_depth = max(self.max_depth - 1, 2)
        else:
            new_depth = self.max_depth
            
        self.max_depth = new_depth
        self.trajectory_sampler.max_depth = new_depth
        return new_depth

# Example usage
if __name__ == "__main__":
    # Initialize FMC planner
    fmc = FractalMonteCarlo(state_dim=64, action_dim=16, max_depth=4, num_samples=8)
    
    # Create initial state
    initial_state = np.random.randn(64).astype(np.float32)
    
    # Plan action
    action, metadata = fmc.plan(initial_state)
    
    print(f"Planned action shape: {action.shape}")
    print(f"Evaluation score: {metadata['evaluation_score']:.4f}")
    print(f"Trajectory depth: {metadata['trajectory_depth']}")
    print(f"Number of evaluations: {len(metadata['all_evaluations'])}")