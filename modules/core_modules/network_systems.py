"""
Network Systems for Consciousness

Mycelial network nodes and collective intelligence systems for
distributed consciousness processing.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict, deque
import random

class MycelialNode:
    """
    Вузол міцелієвої мережі
    """
    
    def __init__(self, node_id: str, position: Tuple[float, float]):
        self.node_id = node_id
        self.position = position
        self.connections = {}  # node_id -> connection_strength
        self.resources = {
            'energy': 1.0,
            'information': 0.0,
            'nutrients': 1.0
        }
        self.state = torch.zeros(32)  # Внутрішній стан вузла
        self.memory = []
        self.processing_capacity = 1.0
        
    def connect_to(self, other_node: 'MycelialNode', strength: float = 0.5):
        """
        Створення з'єднання з іншим вузлом
        """
        self.connections[other_node.node_id] = strength
        other_node.connections[self.node_id] = strength
        
    def send_signal(self, target_node_id: str, signal: torch.Tensor) -> bool:
        """
        Відправка сигналу до цільового вузла
        """
        if target_node_id in self.connections:
            connection_strength = self.connections[target_node_id]
            
            # Ослаблення сигналу залежно від сили з'єднання
            attenuated_signal = signal * connection_strength
            
            # Витрати енергії на передачу
            energy_cost = torch.norm(signal).item() * 0.1
            self.resources['energy'] = max(0, self.resources['energy'] - energy_cost)
            
            return True
        return False
    
    def receive_signal(self, signal: torch.Tensor, sender_id: str):
        """
        Отримання сигналу від іншого вузла
        """
        if sender_id in self.connections:
            connection_strength = self.connections[sender_id]
            
            # Обробка сигналу
            processed_signal = self._process_signal(signal, connection_strength)
            
            # Оновлення стану
            self.state = self.state * 0.9 + processed_signal * 0.1
            
            # Збереження в пам'яті
            self.memory.append({
                'timestamp': len(self.memory),
                'sender': sender_id,
                'signal': signal.clone(),
                'processed': processed_signal
            })
            
            # Обмеження розміру пам'яті
            if len(self.memory) > 100:
                self.memory.pop(0)
    
    def _process_signal(self, signal: torch.Tensor, connection_strength: float) -> torch.Tensor:
        """
        Обробка отриманого сигналу
        """
        # Нелінійна обробка сигналу
        processed = torch.tanh(signal * connection_strength)
        
        # Додавання внутрішнього стану
        if len(processed) == len(self.state):
            processed = processed + self.state * 0.1
        
        # Нормалізація
        processed = processed / (torch.norm(processed) + 1e-8)
        
        return processed
    
    def share_resources(self, other_nodes: List['MycelialNode'], resource_type: str):
        """
        Розподіл ресурсів з іншими вузлами
        """
        if resource_type not in self.resources:
            return
        
        total_resource = self.resources[resource_type]
        connected_nodes = [node for node in other_nodes if node.node_id in self.connections]
        
        if not connected_nodes:
            return
        
        # Розрахунок розподілу на основі сили з'єднань
        total_connection_strength = sum(self.connections[node.node_id] for node in connected_nodes)
        
        for node in connected_nodes:
            connection_strength = self.connections[node.node_id]
            share_ratio = connection_strength / total_connection_strength
            shared_amount = total_resource * share_ratio * 0.1  # 10% розподіл
            
            # Передача ресурсу
            self.resources[resource_type] -= shared_amount
            node.resources[resource_type] += shared_amount
    
    def update_state(self):
        """
        Оновлення стану вузла
        """
        # Природне згасання
        self.state = self.state * 0.99
        
        # Регенерація ресурсів
        for resource_type in self.resources:
            if self.resources[resource_type] < 1.0:
                self.resources[resource_type] += 0.01
        
        # Обмеження ресурсів
        for resource_type in self.resources:
            self.resources[resource_type] = max(0.0, min(2.0, self.resources[resource_type]))

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
    """
    Система колективного інтелекту
    """
    
    def __init__(self, mycelial_network: FungalNeuroglia):
        self.network = mycelial_network
        self.swarm_behaviors = {}
        self.emergent_patterns = []
        self.consensus_threshold = 0.7
        
    def swarm_optimization(self, 
                          objective_function: callable, 
                          search_space: Tuple[torch.Tensor, torch.Tensor],
                          max_iterations: int = 100) -> torch.Tensor:
        """
        Роєва оптимізація через міцелієву мережу
        """
        # Ініціалізація позицій "часток" (вузлів)
        dim = search_space[0].shape[0]
        positions = {}
        velocities = {}
        personal_best = {}
        personal_best_values = {}
        
        # Ініціалізація для кожного вузла
        for node_id, node in self.network.nodes.items():
            # Випадкова позиція в межах пошукового простору
            position = search_space[0] + torch.rand(dim) * (search_space[1] - search_space[0])
            positions[node_id] = position
            velocities[node_id] = torch.randn(dim) * 0.1
            
            # Оцінка початкової позиції
            value = objective_function(position)
            personal_best[node_id] = position.clone()
            personal_best_values[node_id] = value
        
        # Глобальний найкращий результат
        global_best_id = max(personal_best_values.keys(), key=lambda k: personal_best_values[k])
        global_best = personal_best[global_best_id].clone()
        global_best_value = personal_best_values[global_best_id]
        
        # Основний цикл оптимізації
        for iteration in range(max_iterations):
            for node_id, node in self.network.nodes.items():
                current_pos = positions[node_id]
                current_vel = velocities[node_id]
                
                # Соціальна складова від сусідів
                social_influence = torch.zeros(dim)
                neighbor_count = 0
                
                for neighbor_id, connection_strength in node.connections.items():
                    if neighbor_id in positions:
                        neighbor_pos = positions[neighbor_id]
                        social_influence += (neighbor_pos - current_pos) * connection_strength
                        neighbor_count += 1
                
                if neighbor_count > 0:
                    social_influence /= neighbor_count
                
                # Оновлення швидкості (PSO з соціальною компонентою)
                inertia = 0.7
                cognitive_factor = 1.5
                social_factor = 1.5
                
                cognitive_component = cognitive_factor * torch.rand(1) * (personal_best[node_id] - current_pos)
                social_component = social_factor * torch.rand(1) * social_influence
                
                new_velocity = (inertia * current_vel + 
                              cognitive_component + 
                              social_component)
                
                # Обмеження швидкості
                max_velocity = torch.norm(search_space[1] - search_space[0]) * 0.1
                if torch.norm(new_velocity) > max_velocity:
                    new_velocity = new_velocity * max_velocity / torch.norm(new_velocity)
                
                velocities[node_id] = new_velocity
                
                # Оновлення позиції
                new_position = current_pos + new_velocity
                
                # Обмеження межами пошукового простору
                new_position = torch.clamp(new_position, search_space[0], search_space[1])
                positions[node_id] = new_position
                
                # Оцінка нової позиції
                new_value = objective_function(new_position)
                
                # Оновлення особистого найкращого
                if new_value > personal_best_values[node_id]:
                    personal_best[node_id] = new_position.clone()
                    personal_best_values[node_id] = new_value
                    
                    # Оновлення глобального найкращого
                    if new_value > global_best_value:
                        global_best = new_position.clone()
                        global_best_value = new_value
            
            # Синхронізація мережі кожні 10 ітерацій
            if iteration % 10 == 0:
                self.network.synchronize_network()
        
        return global_best
    
    def detect_emergent_patterns(self) -> List[Dict[str, Any]]:
        """
        Виявлення емерджентних патернів у мережі
        """
        patterns = []
        
        # Аналіз топологічних патернів
        topology_pattern = self._analyze_network_topology()
        if topology_pattern:
            patterns.append(topology_pattern)
        
        # Аналіз динамічних патернів
        dynamic_pattern = self._analyze_dynamic_patterns()
        if dynamic_pattern:
            patterns.append(dynamic_pattern)
        
        # Аналіз інформаційних потоків
        information_pattern = self._analyze_information_flows()
        if information_pattern:
            patterns.append(information_pattern)
        
        # Збереження знайдених патернів
        self.emergent_patterns.extend(patterns)
        
        return patterns
    
    def _analyze_network_topology(self) -> Optional[Dict[str, Any]]:
        """
        Аналіз топологічних властивостей мережі
        """
        # Створення графу NetworkX для аналізу
        G = nx.Graph()
        
        # Додавання вузлів та ребер
        for node_id, node in self.network.nodes.items():
            G.add_node(node_id)
            for neighbor_id, strength in node.connections.items():
                G.add_edge(node_id, neighbor_id, weight=strength)
        
        if G.number_of_edges() == 0:
            return None
        
        # Обчислення топологічних метрик
        try:
            clustering_coeff = nx.average_clustering(G)
            if G.number_of_nodes() > 1:
                path_length = nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf')
            else:
                path_length = 0
            
            # Малий світ властивості
            random_clustering = 1.0 / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
            small_world_coeff = clustering_coeff / (random_clustering + 1e-8)
            
            pattern = {
                'type': 'topology',
                'clustering_coefficient': clustering_coeff,
                'average_path_length': path_length,
                'small_world_coefficient': small_world_coeff,
                'is_small_world': small_world_coeff > 1 and path_length < np.log(G.number_of_nodes()),
                'node_count': G.number_of_nodes(),
                'edge_count': G.number_of_edges()
            }
            
            return pattern
            
        except:
            return None
    
    def _analyze_dynamic_patterns(self) -> Optional[Dict[str, Any]]:
        """
        Аналіз динамічних патернів у мережі
        """
        # Збір історії станів вузлів
        recent_states = []
        for node in self.network.nodes.values():
            if len(node.memory) > 0:
                recent_states.extend([mem['processed'] for mem in node.memory[-5:]])
        
        if len(recent_states) < 2:
            return None
        
        # Аналіз темпоральних патернів
        states_tensor = torch.stack(recent_states)
        
        # Автокореляція для виявлення циклічних патернів
        autocorr = self._calculate_autocorrelation(states_tensor)
        
        # Виявлення домінуючих частот
        fft_result = torch.abs(torch.fft.fft(states_tensor, dim=0))
        dominant_frequencies = torch.topk(torch.mean(fft_result, dim=1), k=3).indices
        
        pattern = {
            'type': 'dynamic',
            'autocorrelation': autocorr.tolist(),
            'dominant_frequencies': dominant_frequencies.tolist(),
            'temporal_complexity': torch.std(states_tensor).item(),
            'pattern_stability': 1.0 - torch.std(autocorr).item()
        }
        
        return pattern
    
    def _analyze_information_flows(self) -> Optional[Dict[str, Any]]:
        """
        Аналіз потоків інформації у мережі
        """
        # Збір статистики передач сигналів
        transmission_counts = defaultdict(int)
        transmission_strengths = defaultdict(list)
        
        for node in self.network.nodes.values():
            for memory_item in node.memory[-10:]:  # Останні 10 записів
                sender = memory_item['sender']
                signal_strength = torch.norm(memory_item['signal']).item()
                
                transmission_counts[sender] += 1
                transmission_strengths[sender].append(signal_strength)
        
        if not transmission_counts:
            return None
        
        # Аналіз патернів потоків
        total_transmissions = sum(transmission_counts.values())
        
        # Ентропія розподілу передач
        probabilities = [count / total_transmissions for count in transmission_counts.values()]
        information_entropy = -sum(p * np.log2(p + 1e-8) for p in probabilities)
        
        # Середня сила сигналів
        avg_signal_strengths = {
            sender: np.mean(strengths) 
            for sender, strengths in transmission_strengths.items()
        }
        
        pattern = {
            'type': 'information_flow',
            'transmission_entropy': information_entropy,
            'total_transmissions': total_transmissions,
            'active_senders': len(transmission_counts),
            'avg_signal_strength': np.mean(list(avg_signal_strengths.values())),
            'flow_diversity': len(transmission_counts) / len(self.network.nodes)
        }
        
        return pattern
    
    def _calculate_autocorrelation(self, signal: torch.Tensor, max_lag: int = 10) -> torch.Tensor:
        """
        Обчислення автокореляції сигналу
        """
        autocorr = torch.zeros(max_lag)
        signal_mean = torch.mean(signal, dim=0)
        signal_centered = signal - signal_mean
        
        for lag in range(max_lag):
            if lag < signal.shape[0]:
                if lag == 0:
                    autocorr[lag] = 1.0
                else:
                    # Обчислення кореляції зі зсувом
                    shifted_signal = torch.roll(signal_centered, lag, dims=0)
                    correlation = torch.mean(
                        torch.sum(signal_centered * shifted_signal, dim=1)
                    )
                    variance = torch.mean(torch.sum(signal_centered ** 2, dim=1))
                    autocorr[lag] = correlation / (variance + 1e-8)
        
        return autocorr
    
    def achieve_consensus(self, proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Досягнення консенсусу щодо пропозицій
        """
        if not proposals:
            return {'consensus_reached': False, 'reason': 'no_proposals'}
        
        # Голосування кожного вузла за кожну пропозицію
        votes = torch.zeros(len(proposals))
        detailed_votes = {}
        
        for node_id, node in self.network.nodes.items():
            node_votes = torch.zeros(len(proposals))
            detailed_votes[node_id] = {}
            
            for i, proposal in enumerate(proposals):
                # Оцінка пропозиції вузлом
                score = self._evaluate_proposal(node, proposal)
                node_votes[i] = score
                detailed_votes[node_id][f'proposal_{i}'] = score
            
            # Зважування голосу за енергією та кількістю з'єднань
            weight = node.resources['energy'] * (len(node.connections) + 1)
            votes += node_votes * weight
        
        # Нормалізація голосів
        if torch.sum(votes) > 0:
            votes = votes / torch.sum(votes)
        
        # Перевірка досягнення консенсусу
        max_vote = torch.max(votes)
        consensus_reached = max_vote >= self.consensus_threshold
        
        winning_proposal = torch.argmax(votes).item() if consensus_reached else None
        
        consensus_result = {
            'consensus_reached': consensus_reached,
            'winning_proposal': winning_proposal,
            'vote_distribution': votes.tolist(),
            'consensus_strength': max_vote.item(),
            'detailed_votes': detailed_votes,
            'threshold': self.consensus_threshold
        }
        
        return consensus_result
    
    def _evaluate_proposal(self, node: MycelialNode, proposal: Dict[str, Any]) -> float:
        """
        Оцінка пропозиції окремим вузлом
        """
        # Базова оцінка
        score = 0.5
        
        # Оцінка на основі типу пропозиції
        if 'type' in proposal:
            proposal_type = proposal['type']
            
            # Різні типи пропозицій мають різні критерії оцінки
            if proposal_type == 'resource_allocation':
                # Оцінка пропозицій розподілу ресурсів
                if 'target_resource' in proposal:
                    target_resource = proposal['target_resource']
                    if target_resource in node.resources:
                        current_level = node.resources[target_resource]
                        if current_level < 0.7:  # Потреба в ресурсі
                            score += 0.3
                        elif current_level > 1.3:  # Надлишок ресурсу
                            score -= 0.2
            
            elif proposal_type == 'network_restructure':
                # Оцінка пропозицій зміни структури мережі
                current_connections = len(node.connections)
                if current_connections < 3:  # Мало з'єднань
                    score += 0.4
                elif current_connections > 10:  # Багато з'єднань
                    score -= 0.1
            
            elif proposal_type == 'behavior_change':
                # Оцінка пропозицій зміни поведінки
                if 'urgency' in proposal:
                    urgency = proposal['urgency']
                    score += urgency * 0.2
        
        # Оцінка на основі історії взаємодій
        if len(node.memory) > 0:
            recent_activity = len([m for m in node.memory[-10:] if m])
            activity_factor = min(1.0, recent_activity / 10.0)
            score *= (0.5 + 0.5 * activity_factor)
        
        # Додавання випадковості для реалістичності
        noise = np.random.normal(0, 0.1)
        score += noise
        
        # Обмеження оцінки
        return max(0.0, min(1.0, score))

# ===================================================================
# 🌅 5. META-CONSCIOUSNESS LAYER - Рівень Метасвідомості
# ===================================================================


__all__ = ['MycelialNode', 'CollectiveIntelligence']
