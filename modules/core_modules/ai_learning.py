"""
AI and Learning Systems for Consciousness

Advanced AI agents including Fractal Monte Carlo learning and 
recursive thinking/metacognition capabilities.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict, deque
import random

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


__all__ = ['FractalMonteCarloAgent', 'RecursiveThinking']
