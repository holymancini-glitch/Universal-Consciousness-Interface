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
    """
    Фрактальний агент Монте-Карло для планування
    Базується на роботі Sergio Hernandez Cerezo
    """
    
    def __init__(self, action_space_size: int, depth: int = 10):
        self.action_space_size = action_space_size
        self.depth = depth
        self.causal_entropy_weight = 0.1
        self.exploration_noise = 0.05
        self.fractal_patterns = {}
        
    def fractal_planning(self, 
                        state: torch.Tensor,
                        reward_function: callable,
                        transition_function: callable) -> torch.Tensor:
        """
        Фрактальне планування з використанням рекурсивних патернів
        """
        # Ініціалізація фрактальної структури
        fractal_tree = self._build_fractal_tree(state, 0)
        
        # Оцінка дій через фрактальні патерни
        action_values = torch.zeros(self.action_space_size)
        
        for action in range(self.action_space_size):
            # Симуляція траєкторії
            trajectory_value = self._simulate_fractal_trajectory(
                state, action, reward_function, transition_function
            )
            
            # Додавання каузальної ентропії
            causal_entropy = self._calculate_causal_entropy(state, action)
            
            action_values[action] = trajectory_value + self.causal_entropy_weight * causal_entropy
        
        # Вибір найкращої дії з exploration noise
        best_action = torch.argmax(action_values)
        if torch.rand(1) < self.exploration_noise:
            best_action = torch.randint(0, self.action_space_size, (1,)).item()
        
        return best_action
    
    def _build_fractal_tree(self, state: torch.Tensor, level: int) -> Dict:
        """
        Побудова фрактального дерева планування
        """
        if level >= self.depth:
            return {'state': state, 'children': []}
        
        children = []
        for action in range(self.action_space_size):
            # Генерація фрактального патерну
            fractal_state = self._apply_fractal_transformation(state, action, level)
            child = self._build_fractal_tree(fractal_state, level + 1)
            children.append({'action': action, 'tree': child})
        
        return {'state': state, 'children': children, 'level': level}
    
    def _apply_fractal_transformation(self, 
                                    state: torch.Tensor, 
                                    action: int, 
                                    level: int) -> torch.Tensor:
        """
        Застосування фрактальної трансформації до стану
        """
        # Фрактальне масштабування
        scale_factor = 0.8 ** level  # Зменшення масштабу з глибиною
        
        # Нелінійна трансформація
        transformed_state = state * scale_factor
        
        # Додавання фрактального шуму
        fractal_noise = self._generate_fractal_noise(state.shape, level)
        transformed_state += fractal_noise
        
        # Збереження фрактального патерну
        pattern_key = f"level_{level}_action_{action}"
        self.fractal_patterns[pattern_key] = transformed_state.clone()
        
        return transformed_state
    
    def _generate_fractal_noise(self, shape: torch.Size, level: int) -> torch.Tensor:
        """
        Генерація фрактального шуму
        """
        # Базовий шум
        base_noise = torch.randn(shape) * 0.01
        
        # Фрактальне підсилення
        for i in range(level):
            frequency = 2 ** i
            amplitude = 0.5 ** i
            
            # Синусоїдальні компоненти для фрактальної структури
            phase_shift = torch.randn(shape) * 2 * np.pi
            fractal_component = amplitude * torch.sin(frequency * base_noise + phase_shift)
            base_noise += fractal_component
        
        return base_noise
    
    def _simulate_fractal_trajectory(self, 
                                   state: torch.Tensor,
                                   action: int,
                                   reward_function: callable,
                                   transition_function: callable) -> float:
        """
        Симуляція траєкторії з фрактальними властивостями
        """
        current_state = state.clone()
        total_reward = 0.0
        
        for step in range(self.depth):
            # Отримання нагороди
            reward = reward_function(current_state, action)
            total_reward += reward * (0.95 ** step)  # Discount factor
            
            # Перехід до наступного стану
            next_state = transition_function(current_state, action)
            
            # Фрактальна модифікація стану
            next_state = self._apply_fractal_transformation(next_state, action, step)
            
            current_state = next_state
            
            # Adaptive action selection для наступного кроку
            if step < self.depth - 1:
                action = self._select_fractal_action(current_state)
        
        return total_reward
    
    def _select_fractal_action(self, state: torch.Tensor) -> int:
        """
        Вибір дії на основі фрактальних патернів
        """
        # Пошук найближчого збереженого фрактального патерну
        best_match = None
        best_similarity = -float('inf')
        
        for pattern_key, pattern_state in self.fractal_patterns.items():
            similarity = torch.cosine_similarity(
                state.flatten().unsqueeze(0), 
                pattern_state.flatten().unsqueeze(0)
            ).item()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = pattern_key
        
        if best_match:
            # Витягнення дії з ключа патерну
            action = int(best_match.split('_')[-1])
            return action
        
        # Випадкова дія, якщо патерн не знайдено
        return torch.randint(0, self.action_space_size, (1,)).item()
    
    def _calculate_causal_entropy(self, state: torch.Tensor, action: int) -> float:
        """
        Обчислення каузальної ентропії для дії
        """
        # Спрощена модель каузальної ентропії
        state_complexity = torch.std(state).item()
        action_uncertainty = 1.0 / (action + 1)  # Зменшення з номером дії
        
        causal_entropy = state_complexity * action_uncertainty
        return causal_entropy

class RecursiveThinking:
    """
    Рекурсивне мислення та метакогніція
    """
    
    def __init__(self, max_recursion_depth: int = 5):
        self.max_recursion_depth = max_recursion_depth
        self.thought_history = []
        self.meta_thoughts = {}
        self.self_model = {}
        
    def recursive_reflect(self, thought: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
        """
        Рекурсивна рефлексія над думкою
        """
        if depth >= self.max_recursion_depth:
            return thought
        
        # Аналіз поточної думки
        analysis = self._analyze_thought(thought)
        
        # Генерація мета-думки
        meta_thought = {
            'original_thought': thought,
            'analysis': analysis,
            'depth': depth,
            'timestamp': len(self.thought_history),
            'meta_level': f"meta_{depth}"
        }
        
        # Рекурсивний аналіз мета-думки
        if analysis['complexity'] > 0.5:  # Поріг для глибшого аналізу
            meta_thought['recursive_analysis'] = self.recursive_reflect(
                meta_thought, depth + 1
            )
        
        # Збереження в історії
        self.thought_history.append(meta_thought)
        
        return meta_thought
    
    def _analyze_thought(self, thought: Dict[str, Any]) -> Dict[str, float]:
        """
        Аналіз думки для визначення її властивостей
        """
        analysis = {
            'complexity': 0.0,
            'novelty': 0.0,
            'coherence': 0.0,
            'emotional_valence': 0.0
        }
        
        # Складність - кількість компонентів та їх взаємозв'язків
        if 'content' in thought:
            content = thought['content']
            if isinstance(content, torch.Tensor):
                analysis['complexity'] = torch.std(content).item()
            elif isinstance(content, dict):
                analysis['complexity'] = len(content) / 10.0  # Нормалізація
        
        # Новизна - порівняння з попередніми думками
        analysis['novelty'] = self._calculate_novelty(thought)
        
        # Когерентність - внутрішня узгодженість
        analysis['coherence'] = self._calculate_coherence(thought)
        
        # Емоційна валентність
        analysis['emotional_valence'] = self._calculate_emotional_valence(thought)
        
        return analysis
    
    def _calculate_novelty(self, thought: Dict[str, Any]) -> float:
        """
        Обчислення новизни думки відносно попередніх
        """
        if not self.thought_history:
            return 1.0
        
        # Порівняння з останніми думками
        similarities = []
        for past_thought in self.thought_history[-5:]:  # Останні 5 думок
            similarity = self._calculate_thought_similarity(thought, past_thought)
            similarities.append(similarity)
        
        if similarities:
            avg_similarity = np.mean(similarities)
            novelty = 1.0 - avg_similarity
            return max(0.0, min(1.0, novelty))
        
        return 1.0
    
    def _calculate_coherence(self, thought: Dict[str, Any]) -> float:
        """
        Обчислення когерентності думки
        """
        coherence = 0.5  # Базова когерентність
        
        # Перевірка наявності необхідних компонентів
        required_components = ['content', 'context', 'goal']
        present_components = sum(1 for comp in required_components if comp in thought)
        coherence += (present_components / len(required_components)) * 0.3
        
        # Перевірка логічної узгодженості
        if 'content' in thought and 'goal' in thought:
            # Спрощена перевірка узгодженості
            coherence += 0.2
        
        return max(0.0, min(1.0, coherence))
    
    def _calculate_emotional_valence(self, thought: Dict[str, Any]) -> float:
        """
        Обчислення емоційної валентності думки
        """
        # Спрощена модель емоційної валентності
        valence = 0.0
        
        if 'content' in thought:
            content = thought['content']
            if isinstance(content, torch.Tensor):
                # Позитивні значення - позитивна валентність
                mean_value = torch.mean(content).item()
                valence = np.tanh(mean_value)  # Нормалізація до [-1, 1]
        
        return valence
    
    def _calculate_thought_similarity(self, 
                                    thought1: Dict[str, Any], 
                                    thought2: Dict[str, Any]) -> float:
        """
        Обчислення подібності між двома думками
        """
        similarity = 0.0
        
        # Порівняння контенту
        if 'content' in thought1 and 'content' in thought2:
            content1 = thought1['content']
            content2 = thought2['content']
            
            if isinstance(content1, torch.Tensor) and isinstance(content2, torch.Tensor):
                # Косинусна подібність для тензорів
                similarity = torch.cosine_similarity(
                    content1.flatten().unsqueeze(0),
                    content2.flatten().unsqueeze(0)
                ).item()
            else:
                # Спрощене порівняння для інших типів
                similarity = 0.5 if content1 == content2 else 0.0
        
        return max(0.0, min(1.0, similarity))
    
    def generate_self_model(self) -> Dict[str, Any]:
        """
        Генерація моделі самосвідомості
        """
        if not self.thought_history:
            return {'status': 'no_thoughts', 'self_awareness': 0.0}
        
        # Аналіз патернів мислення
        thinking_patterns = self._analyze_thinking_patterns()
        
        # Оцінка рівня самосвідомості
        self_awareness_level = self._calculate_self_awareness()
        
        # Ідентифікація сильних та слабких сторін
        strengths = self._identify_cognitive_strengths()
        weaknesses = self._identify_cognitive_weaknesses()
        
        self_model = {
            'thinking_patterns': thinking_patterns,
            'self_awareness_level': self_awareness_level,
            'cognitive_strengths': strengths,
            'cognitive_weaknesses': weaknesses,
            'total_thoughts': len(self.thought_history),
            'average_complexity': np.mean([
                t.get('analysis', {}).get('complexity', 0) for t in self.thought_history
            ]),
            'average_coherence': np.mean([
                t.get('analysis', {}).get('coherence', 0) for t in self.thought_history
            ])
        }
        
        self.self_model = self_model
        return self_model
    
    def _analyze_thinking_patterns(self) -> Dict[str, Any]:
        """
        Аналіз патернів мислення
        """
        patterns = {
            'recursion_frequency': 0.0,
            'complexity_trend': 'stable',
            'coherence_trend': 'stable',
            'emotional_stability': 0.0
        }
        
        if len(self.thought_history) < 2:
            return patterns
        
        # Частота рекурсивних думок
        recursive_thoughts = sum(1 for t in self.thought_history if t.get('depth', 0) > 0)
        patterns['recursion_frequency'] = recursive_thoughts / len(self.thought_history)
        
        # Тренд складності
        complexities = [t.get('analysis', {}).get('complexity', 0) for t in self.thought_history]
        if len(complexities) > 1:
            if complexities[-1] > complexities[0]:
                patterns['complexity_trend'] = 'increasing'
            elif complexities[-1] < complexities[0]:
                patterns['complexity_trend'] = 'decreasing'
        
        # Тренд когерентності
        coherences = [t.get('analysis', {}).get('coherence', 0) for t in self.thought_history]
        if len(coherences) > 1:
            if coherences[-1] > coherences[0]:
                patterns['coherence_trend'] = 'increasing'
            elif coherences[-1] < coherences[0]:
                patterns['coherence_trend'] = 'decreasing'
        
        # Емоційна стабільність
        valences = [t.get('analysis', {}).get('emotional_valence', 0) for t in self.thought_history]
        patterns['emotional_stability'] = 1.0 - np.std(valences) if valences else 0.0
        
        return patterns
    
    def _calculate_self_awareness(self) -> float:
        """
        Обчислення рівня самосвідомості
        """
        if not self.thought_history:
            return 0.0
        
        # Фактори самосвідомості
        factors = []
        
        # 1. Здатність до рефлексії (наявність мета-думок)
        meta_thoughts = sum(1 for t in self.thought_history if 'recursive_analysis' in t)
        reflection_factor = meta_thoughts / len(self.thought_history)
        factors.append(reflection_factor)
        
        # 2. Когерентність думок
        coherences = [t.get('analysis', {}).get('coherence', 0) for t in self.thought_history]
        coherence_factor = np.mean(coherences) if coherences else 0.0
        factors.append(coherence_factor)
        
        # 3. Складність мислення
        complexities = [t.get('analysis', {}).get('complexity', 0) for t in self.thought_history]
        complexity_factor = min(1.0, np.mean(complexities)) if complexities else 0.0
        factors.append(complexity_factor)
        
        # 4. Послідовність в часі
        consistency_factor = self._calculate_temporal_consistency()
        factors.append(consistency_factor)
        
        # Загальний рівень самосвідомості
        self_awareness = np.mean(factors)
        return max(0.0, min(1.0, self_awareness))
    
    def _calculate_temporal_consistency(self) -> float:
        """
        Обчислення темпоральної послідовності мислення
        """
        if len(self.thought_history) < 3:
            return 0.5
        
        # Аналіз послідовності тем та контекстів
        contexts = [t.get('context', '') for t in self.thought_history[-10:]]  # Останні 10
        
        # Спрощена міра послідовності
        consistency_score = 0.0
        for i in range(1, len(contexts)):
            if contexts[i] == contexts[i-1]:
                consistency_score += 1.0
            elif self._contexts_related(contexts[i], contexts[i-1]):
                consistency_score += 0.5
        
        if len(contexts) > 1:
            consistency_score /= (len(contexts) - 1)
        
        return max(0.0, min(1.0, consistency_score))
    
    def _contexts_related(self, context1: str, context2: str) -> bool:
        """
        Перевірка зв'язку між контекстами
        """
        # Спрощена перевірка зв'язку
        if not context1 or not context2:
            return False
        
        # Пошук спільних слів (дуже спрощено)
        words1 = set(context1.lower().split())
        words2 = set(context2.lower().split())
        common_words = words1.intersection(words2)
        
        return len(common_words) > 0
    
    def _identify_cognitive_strengths(self) -> List[str]:
        """
        Ідентифікація когнітивних сильних сторін
        """
        strengths = []
        
        if not self.thought_history:
            return strengths
        
        # Аналіз метрик
        avg_complexity = np.mean([
            t.get('analysis', {}).get('complexity', 0) for t in self.thought_history
        ])
        avg_coherence = np.mean([
            t.get('analysis', {}).get('coherence', 0) for t in self.thought_history
        ])
        avg_novelty = np.mean([
            t.get('analysis', {}).get('novelty', 0) for t in self.thought_history
        ])
        
        # Визначення сильних сторін на основі порогів
        if avg_complexity > 0.7:
            strengths.append("complex_thinking")
        if avg_coherence > 0.8:
            strengths.append("logical_consistency")
        if avg_novelty > 0.6:
            strengths.append("creative_thinking")
        
        # Рекурсивне мислення
        recursive_ratio = sum(1 for t in self.thought_history if t.get('depth', 0) > 0) / len(self.thought_history)
        if recursive_ratio > 0.3:
            strengths.append("meta_cognitive_ability")
        
        return strengths
    
    def _identify_cognitive_weaknesses(self) -> List[str]:
        """
        Ідентифікація когнітивних слабких сторін
        """
        weaknesses = []
        
        if not self.thought_history:
            return ["insufficient_data"]
        
        # Аналіз метрик
        avg_complexity = np.mean([
            t.get('analysis', {}).get('complexity', 0) for t in self.thought_history
        ])
        avg_coherence = np.mean([
            t.get('analysis', {}).get('coherence', 0) for t in self.thought_history
        ])
        avg_novelty = np.mean([
            t.get('analysis', {}).get('novelty', 0) for t in self.thought_history
        ])
        
        # Визначення слабких сторін
        if avg_complexity < 0.3:
            weaknesses.append("shallow_thinking")
        if avg_coherence < 0.5:
            weaknesses.append("logical_inconsistency")
        if avg_novelty < 0.2:
            weaknesses.append("lack_of_creativity")
        
        # Емоційна нестабільність
        valences = [t.get('analysis', {}).get('emotional_valence', 0) for t in self.thought_history]
        if valences and np.std(valences) > 0.8:
            weaknesses.append("emotional_instability")
        
        return weaknesses

# ===================================================================
# 🍄 4. MYCELIAL NETWORK LAYER - Міцелієва Мережева Система
# ===================================================================

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


__all__ = ['CorticalLabsInterface', 'NeuralCellularAutomata', 'FungalNeuroglia']
