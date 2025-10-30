"""
Biological Interfaces for Consciousness Systems

Interfaces to biological neural systems including Cortical Labs DishBrain,
Neural Cellular Automata, and Fungal Neuroglia networks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

class CorticalLabsInterface:
    """
    Інтерфейс з Cortical Labs DishBrain
    Емуляція взаємодії з живими нейронними культурами
    """
    
    def __init__(self, electrode_count: int = 8000):
        self.electrode_count = electrode_count
        self.neuron_positions = self._generate_neuron_positions()
        self.synaptic_weights = torch.randn(electrode_count, electrode_count) * 0.1
        self.neuron_states = torch.zeros(electrode_count)
        self.firing_threshold = 0.5
        
    def _generate_neuron_positions(self) -> torch.Tensor:
        """Генерація позицій нейронів у 2D культурі"""
        positions = torch.rand(self.electrode_count, 2) * 100  # 100x100 мікрон
        return positions
        
    def electrical_stimulation(self, 
                             stimulus_pattern: torch.Tensor,
                             electrode_indices: List[int]) -> torch.Tensor:
        """
        Електрична стимуляція нейронів
        """
        # Застосування стимулу до вибраних електродів
        self.neuron_states[electrode_indices] += stimulus_pattern
        
        # Поширення активності через синаптичні з'єднання
        propagated_activity = torch.matmul(
            self.synaptic_weights, 
            self.neuron_states
        )
        
        # Функція активації (спрощена модель спайків)
        spikes = torch.sigmoid(propagated_activity - self.firing_threshold)
        
        # Оновлення станів нейронів
        self.neuron_states = self.neuron_states * 0.9 + spikes * 0.1
        
        return spikes
    
    def record_activity(self) -> Dict[str, torch.Tensor]:
        """
        Запис активності нейронів
        """
        return {
            'spike_trains': self.neuron_states,
            'synaptic_weights': self.synaptic_weights,
            'population_activity': torch.mean(self.neuron_states),
            'synchronization_index': self._calculate_synchronization()
        }
    
    def _calculate_synchronization(self) -> float:
        """
        Обчислення індексу синхронізації нейронної активності
        """
        # Кореляційна матриця активності
        correlation_matrix = torch.corrcoef(self.neuron_states.unsqueeze(0))
        # Середня кореляція як міра синхронізації
        sync_index = torch.mean(torch.abs(correlation_matrix)).item()
        return sync_index

class NeuralCellularAutomata:

class NeuralCellularAutomata:
    """
    Нейронні клітинні автомати для емерджентних патернів
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (64, 64)):
        self.grid_size = grid_size
        self.state_grid = torch.rand(grid_size)
        self.rules = self._initialize_rules()
        
    def _initialize_rules(self) -> Dict[str, float]:
        """Ініціалізація правил клітинних автоматів"""
        return {
            'survival_min': 2,
            'survival_max': 3,
            'birth_count': 3,
            'excitation_threshold': 0.3,
            'inhibition_factor': 0.1
        }
    
    def update_step(self) -> torch.Tensor:
        """
        Один крок оновлення клітинних автоматів
        """
        new_state = torch.zeros_like(self.state_grid)
        
        for i in range(1, self.grid_size[0] - 1):
            for j in range(1, self.grid_size[1] - 1):
                # Отримання сусідів
                neighbors = self.state_grid[i-1:i+2, j-1:j+2]
                neighbor_sum = torch.sum(neighbors) - self.state_grid[i, j]
                
                # Застосування правил
                current_state = self.state_grid[i, j]
                
                if current_state > self.rules['excitation_threshold']:
                    # Жива клітина
                    if (neighbor_sum >= self.rules['survival_min'] and 
                        neighbor_sum <= self.rules['survival_max']):
                        new_state[i, j] = current_state * 0.95  # Поступове згасання
                    else:
                        new_state[i, j] = current_state * 0.5  # Швидке згасання
                else:
                    # Мертва клітина
                    if abs(neighbor_sum - self.rules['birth_count']) < 0.5:
                        new_state[i, j] = 0.8  # Народження
                
                # Додавання шуму для реалістичності
                new_state[i, j] += torch.randn(1).item() * 0.01
                
        self.state_grid = torch.clamp(new_state, 0, 1)
        return self.state_grid
    
    def extract_patterns(self) -> Dict[str, Any]:
        """
        Вилучення емерджентних патернів
        """
        # Аналіз Фур'є для частотних компонентів
        fft_pattern = torch.abs(torch.fft.fft2(self.state_grid))
        
        # Фрактальна розмірність
        fractal_dim = self._calculate_fractal_dimension()
        
        # Структурні паттерни (кластери, цикли)
        clusters = self._detect_clusters()
        
        return {
            'frequency_pattern': fft_pattern,
            'fractal_dimension': fractal_dim,
            'cluster_count': len(clusters),
            'pattern_complexity': torch.std(self.state_grid).item()
        }
    
    def _calculate_fractal_dimension(self) -> float:
        """Обчислення фрактальної розмірності патерну"""
        # Спрощений алгоритм box-counting
        binary_grid = (self.state_grid > 0.5).float()
        sizes = [2, 4, 8, 16]
        counts = []
        
        for size in sizes:
            count = 0
            for i in range(0, self.grid_size[0], size):
                for j in range(0, self.grid_size[1], size):
                    box = binary_grid[i:i+size, j:j+size]
                    if torch.sum(box) > 0:
                        count += 1
            counts.append(count)
        
        # Лінійна регресія log(count) vs log(1/size)
        log_sizes = [np.log(1/s) for s in sizes]
        log_counts = [np.log(c + 1) for c in counts]
        
        # Простий алгоритм найменших квадратів
        if len(log_counts) > 1:
            slope = (log_counts[-1] - log_counts[0]) / (log_sizes[-1] - log_sizes[0])
            return abs(slope)
        return 1.0
    
    def _detect_clusters(self) -> List[Dict]:
        """Виявлення кластерів активності"""
        # Спрощене виявлення кластерів через threshold
        active_cells = self.state_grid > 0.7
        clusters = []
        
        # Групування сусідніх активних клітин
        visited = torch.zeros_like(active_cells, dtype=torch.bool)
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if active_cells[i, j] and not visited[i, j]:
                    cluster = self._flood_fill(active_cells, visited, i, j)
                    if len(cluster) > 3:  # Мінімальний розмір кластеру
                        clusters.append({
                            'center': (i, j),
                            'size': len(cluster),
                            'cells': cluster
                        })
        
        return clusters
    
    def _flood_fill(self, grid, visited, start_i, start_j) -> List[Tuple[int, int]]:
        """Алгоритм заливки для знаходження з'єднаних компонентів"""
        stack = [(start_i, start_j)]
        cluster = []
        
        while stack:
            i, j = stack.pop()
            if (i < 0 or i >= self.grid_size[0] or 
                j < 0 or j >= self.grid_size[1] or 
                visited[i, j] or not grid[i, j]):
                continue
                
            visited[i, j] = True
            cluster.append((i, j))
            
            # Додавання сусідів
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    stack.append((i + di, j + dj))
        
        return cluster

# ===================================================================
# 🌀 3. FRACTAL AI ENGINE - Фрактальний AI Двигун
# ===================================================================

class FractalMonteCarloAgent:

class FungalNeuroglia:
    """
    Грибна нейроглія - розподілена мережа обробки інформації
    """
    
    def __init__(self, network_size: int = 100):
        self.network_size = network_size
        self.nodes = self._create_network()
        self.global_state = torch.zeros(64)
        self.collective_memory = []
        self.synchronization_frequency = 0.1
        
    def _create_network(self) -> Dict[str, MycelialNode]:
        """
        Створення міцелієвої мережі
        """
        nodes = {}
        
        # Створення вузлів
        for i in range(self.network_size):
            node_id = f"node_{i}"
            # Випадкові позиції в 2D просторі
            position = (np.random.uniform(0, 100), np.random.uniform(0, 100))
            nodes[node_id] = MycelialNode(node_id, position)
        
        # Створення з'єднань на основі відстані
        node_list = list(nodes.values())
        for i, node1 in enumerate(node_list):
            for j, node2 in enumerate(node_list[i+1:], i+1):
                distance = self._calculate_distance(node1.position, node2.position)
                
                # Ймовірність з'єднання залежить від відстані
                connection_probability = np.exp(-distance / 20)  # Експоненційне спадання
                
                if np.random.random() < connection_probability:
                    # Сила з'єднання обернено пропорційна відстані
                    strength = max(0.1, 1.0 - distance / 100)
                    node1.connect_to(node2, strength)
        
        return nodes
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """
        Обчислення Евклідової відстані між вузлами
        """
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def propagate_signal(self, source_node_id: str, signal: torch.Tensor, max_hops: int = 5):
        """
        Поширення сигналу через мережу
        """
        if source_node_id not in self.nodes:
            return
        
        # Ініціалізація поширення
        visited = set()
        current_layer = {source_node_id: signal}
        
        for hop in range(max_hops):
            if not current_layer:
                break
                
            next_layer = {}
            
            for node_id, node_signal in current_layer.items():
                if node_id in visited:
                    continue
                    
                visited.add(node_id)
                current_node = self.nodes[node_id]
                
                # Відправка сигналу до сусідів
                for neighbor_id, connection_strength in current_node.connections.items():
                    if neighbor_id not in visited:
                        # Ослаблення сигналу з відстанню
                        attenuated_signal = node_signal * connection_strength * (0.8 ** hop)
                        
                        # Додавання шуму для реалістичності
                        noise = torch.randn_like(attenuated_signal) * 0.01
                        final_signal = attenuated_signal + noise
                        
                        # Накопичення сигналів від різних джерел
                        if neighbor_id in next_layer:
                            next_layer[neighbor_id] = next_layer[neighbor_id] + final_signal
                        else:
                            next_layer[neighbor_id] = final_signal
                        
                        # Доставка сигналу до вузла
                        self.nodes[neighbor_id].receive_signal(final_signal, node_id)
            
            current_layer = next_layer
    
    def collective_decision_making(self, decision_options: List[torch.Tensor]) -> int:
        """
        Колективне прийняття рішень через консенсус мережі
        """
        if not decision_options:
            return 0
        
        # Голосування кожного вузла
        votes = torch.zeros(len(decision_options))
        
        for node in self.nodes.values():
            # Кожен вузол оцінює опції на основі свого стану
            node_votes = torch.zeros(len(decision_options))
            
            for i, option in enumerate(decision_options):
                # Схожість опції зі станом вузла
                if len(option) == len(node.state):
                    similarity = torch.cosine_similarity(
                        option.unsqueeze(0), 
                        node.state.unsqueeze(0)
                    ).item()
                    node_votes[i] = similarity
                else:
                    # Випадкове голосування, якщо розміри не співпадають
                    node_votes[i] = np.random.random()
            
            # Зважування голосу за енергією вузла
            weight = node.resources['energy']
            votes += node_votes * weight
        
        # Вибір опції з найбільшою кількістю голосів
        best_option = torch.argmax(votes).item()
        
        # Збереження рішення в колективній пам'яті
        decision_record = {
            'timestamp': len(self.collective_memory),
            'options': decision_options,
            'votes': votes,
            'chosen_option': best_option,
            'consensus_strength': torch.max(votes).item() / torch.sum(votes).item()
        }
        self.collective_memory.append(decision_record)
        
        return best_option
    
    def synchronize_network(self):
        """
        Синхронізація всієї мережі
        """
        # Збір глобального стану
        all_states = torch.stack([node.state for node in self.nodes.values()])
        self.global_state = torch.mean(all_states, dim=0)
        
        # Синхронізація частоти
        for node in self.nodes.values():
            # Підтягування стану вузла до глобального
            sync_factor = self.synchronization_frequency
            node.state = node.state * (1 - sync_factor) + self.global_state * sync_factor
            
            # Оновлення стану вузла
            node.update_state()
        
        # Розподіл ресурсів
        self._redistribute_resources()
    
    def _redistribute_resources(self):
        """
        Перерозподіл ресурсів у мережі
        """
        node_list = list(self.nodes.values())
        
        for resource_type in ['energy', 'information', 'nutrients']:
            # Знаходження вузлів з надлишком та нестачею
            excess_nodes = [node for node in node_list if node.resources[resource_type] > 1.5]
            deficit_nodes = [node for node in node_list if node.resources[resource_type] < 0.5]
            
            # Перерозподіл від надлишкових до дефіцитних
            for excess_node in excess_nodes:
                excess_node.share_resources(deficit_nodes, resource_type)
    
    def get_network_metrics(self) -> Dict[str, float]:
        """
        Отримання метрик мережі
        """
        node_list = list(self.nodes.values())
        
        # Зв'язність мережі
        total_connections = sum(len(node.connections) for node in node_list)
        avg_connectivity = total_connections / len(node_list) if node_list else 0
        
        # Синхронізація мережі
        states = torch.stack([node.state for node in node_list])
        synchronization = 1.0 - torch.std(states).item()
        
        # Розподіл ресурсів
        energies = [node.resources['energy'] for node in node_list]
        energy_balance = 1.0 - np.std(energies) / (np.mean(energies) + 1e-8)
        
        # Активність мережі
        total_memory = sum(len(node.memory) for node in node_list)
        network_activity = min(1.0, total_memory / (len(node_list) * 50))
        
        return {
            'connectivity': avg_connectivity,
            'synchronization': max(0.0, min(1.0, synchronization)),
            'energy_balance': max(0.0, min(1.0, energy_balance)),
            'network_activity': network_activity,
            'collective_decisions': len(self.collective_memory)
        }

class CollectiveIntelligence:


__all__ = ['CorticalLabsInterface', 'NeuralCellularAutomata', 'FungalNeuroglia']
