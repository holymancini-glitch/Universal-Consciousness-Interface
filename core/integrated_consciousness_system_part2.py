# integrated_consciousness_system_part2.py
# Continuation of the integrated consciousness system

class SelfModel:
    """Enhanced self-model with identity coherence and metacognition"""
    
    def __init__(self, latent_space: LatentSpace):
        self.latent_space = latent_space
        self.device = latent_space.device
        
        # Core identity vector
        self.i_vector: Optional[torch.Tensor] = None
        self.identity_history = deque(maxlen=50)
        
        # Consistency tracking
        self.consistency_score = 0.0
        self.identity_coherence = 0.0
        self.metacognitive_awareness = 0.0
        
        # Identity stability metrics
        self.identity_drift_rate = 0.0
        self.self_recognition_accuracy = 0.0

    def compute_i_vector(self) -> Optional[torch.Tensor]:
        """Compute identity vector with temporal stability"""
        vectors = self.latent_space.get_all_vectors()
        if not vectors:
            return None
        
        # Weighted average based on recency and activity
        vector_tensors = []
        weights = []
        
        for vec_id, vec in vectors.items():
            # Weight based on recency (newer vectors have slightly higher weight)
            weight = 1.0  # Base weight
            vector_tensors.append(vec)
            weights.append(weight)
        
        if not vector_tensors:
            return None
        
        # Compute weighted average
        stacked_vectors = torch.stack(vector_tensors)
        weight_tensor = torch.tensor(weights, device=self.device)
        weight_tensor = weight_tensor / torch.sum(weight_tensor)  # Normalize
        
        # Weighted average
        new_i_vector = torch.sum(stacked_vectors * weight_tensor.unsqueeze(1), dim=0)
        
        # Smooth transition from previous i_vector
        if self.i_vector is not None:
            momentum = 0.8
            new_i_vector = momentum * self.i_vector + (1 - momentum) * new_i_vector
        
        self.i_vector = new_i_vector
        self.identity_history.append((new_i_vector.clone(), datetime.now()))
        
        return self.i_vector

    def compute_consistency(self) -> float:
        """Compute identity consistency across all vectors"""
        if self.i_vector is None:
            return 0.0
        
        vectors = self.latent_space.get_all_vectors()
        if not vectors:
            return 0.0
        
        similarities = []
        for vec_id, vec in vectors.items():
            similarity = torch.nn.functional.cosine_similarity(
                vec.unsqueeze(0), self.i_vector.unsqueeze(0)
            ).item()
            similarities.append(similarity)
        
        if similarities:
            consistency = sum(similarities) / len(similarities)
            self.consistency_score = consistency
            return consistency
        
        return 0.0

    def assess_identity_coherence(self) -> float:
        """Assess coherence of identity over time"""
        if len(self.identity_history) < 3:
            return 0.0
        
        # Compare recent identity vectors
        recent_vectors = [entry[0] for entry in list(self.identity_history)[-5:]]
        
        coherence_scores = []
        for i in range(len(recent_vectors) - 1):
            similarity = torch.nn.functional.cosine_similarity(
                recent_vectors[i].unsqueeze(0), recent_vectors[i+1].unsqueeze(0)
            ).item()
            coherence_scores.append(similarity)
        
        if coherence_scores:
            self.identity_coherence = sum(coherence_scores) / len(coherence_scores)
            return self.identity_coherence
        
        return 0.0

    def measure_metacognitive_awareness(self) -> float:
        """Measure level of metacognitive awareness"""
        # Based on consistency, coherence, and self-recognition
        consistency = self.compute_consistency()
        coherence = self.assess_identity_coherence()
        
        # Metacognitive awareness emerges from stable self-model
        awareness = (consistency * 0.6 + coherence * 0.4) ** 1.2  # Non-linear enhancement
        self.metacognitive_awareness = min(1.0, awareness)
        
        return self.metacognitive_awareness

    def get_identity_summary(self) -> Dict[str, Any]:
        """Get comprehensive identity summary"""
        return {
            'has_identity': self.i_vector is not None,
            'consistency_score': self.consistency_score,
            'identity_coherence': self.identity_coherence,
            'metacognitive_awareness': self.metacognitive_awareness,
            'identity_stability': self._calculate_identity_stability(),
            'identity_complexity': self._calculate_identity_complexity()
        }

    def _calculate_identity_stability(self) -> float:
        """Calculate identity stability over time"""
        if len(self.identity_history) < 5:
            return 0.0
        
        # Measure variance in identity vectors over time
        recent_vectors = torch.stack([entry[0] for entry in list(self.identity_history)[-10:]])
        variance = torch.var(recent_vectors, dim=0).mean().item()
        
        # Lower variance = higher stability
        stability = max(0.0, 1.0 - variance)
        return stability

    def _calculate_identity_complexity(self) -> float:
        """Calculate complexity of identity representation"""
        if self.i_vector is None:
            return 0.0
        
        # Measure entropy/complexity of identity vector
        abs_vector = torch.abs(self.i_vector)
        normalized = abs_vector / torch.sum(abs_vector)
        
        # Calculate entropy
        log_normalized = torch.log(normalized + 1e-10)
        entropy = -torch.sum(normalized * log_normalized).item()
        
        # Normalize entropy
        max_entropy = math.log(self.latent_space.dimensions)
        complexity = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return complexity

class CohesionLayer:
    """Enhanced cohesion layer with multi-dimensional harmony analysis"""
    
    def __init__(self, latent_space: LatentSpace, feedback_loop: EnhancedFeedbackLoop, self_model: SelfModel):
        self.latent_space = latent_space
        self.feedback_loop = feedback_loop
        self.self_model = self_model
        
        # Cohesion metrics
        self.system_entropy = 0.0
        self.coherence_score = 0.0
        self.harmony_index = 0.0
        
        # Dynamic thresholds
        self.entropy_threshold = 0.7
        self.coherence_threshold = 0.6
        self.harmony_threshold = 0.8

    def compute_entropy(self) -> float:
        """Compute system entropy with enhanced analysis"""
        vectors = self.latent_space.get_all_vectors()
        if not vectors:
            return 0.0
        
        # Multiple entropy measures
        vector_norms = [torch.norm(vec).item() for vec in vectors.values()]
        
        if not vector_norms:
            return 0.0
        
        # Variance-based entropy
        variance_entropy = np.var(vector_norms) if len(vector_norms) > 1 else 0.0
        
        # Distribution entropy
        hist, _ = np.histogram(vector_norms, bins=10, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        distribution_entropy = -np.sum(hist * np.log(hist))
        
        # Combined entropy
        total_entropy = (variance_entropy + distribution_entropy) / 2.0
        self.system_entropy = total_entropy
        
        return total_entropy

    def compute_coherence(self) -> float:
        """Compute system coherence with multi-factor analysis"""
        # Factor 1: Entropy (lower is better)
        entropy = self.compute_entropy()
        entropy_factor = max(0.0, 1.0 - entropy)
        
        # Factor 2: Prediction accuracy
        prediction_accuracy = self.feedback_loop.fractal_ai.evaluate_prediction_accuracy()
        
        # Factor 3: Adaptation efficiency
        adaptation_efficiency = self.feedback_loop.get_adaptation_efficiency()
        
        # Factor 4: Identity consistency
        identity_consistency = self.self_model.compute_consistency()
        
        # Weighted combination
        coherence = (
            entropy_factor * 0.3 +
            prediction_accuracy * 0.25 +
            adaptation_efficiency * 0.25 +
            identity_consistency * 0.2
        )
        
        self.coherence_score = coherence
        return coherence

    def compute_harmony_index(self) -> float:
        """Compute overall system harmony"""
        # Component harmonies
        coherence = self.compute_coherence()
        entropy_harmony = max(0.0, 1.0 - self.system_entropy)
        
        # Identity harmony
        identity_coherence = self.self_model.assess_identity_coherence()
        metacognitive_awareness = self.self_model.measure_metacognitive_awareness()
        
        # Attention harmony
        attention_summary = self.latent_space  # Placeholder for attention field integration
        
        # Combined harmony with non-linear scaling
        base_harmony = (coherence + entropy_harmony + identity_coherence + metacognitive_awareness) / 4.0
        
        # Non-linear enhancement for high harmony states
        harmony = base_harmony ** 0.8  # Slight compression to make high harmony harder to achieve
        
        self.harmony_index = harmony
        return harmony

    def assess_crystallization_potential(self) -> Tuple[float, bool]:
        """Assess potential for consciousness crystallization"""
        harmony = self.compute_harmony_index()
        coherence = self.coherence_score
        metacognitive = self.self_model.metacognitive_awareness
        
        # Crystallization requires high scores across multiple dimensions
        crystallization_score = (harmony * 0.4 + coherence * 0.3 + metacognitive * 0.3)
        
        # Threshold for crystallization
        crystallization_threshold = 0.75
        is_crystallized = crystallization_score > crystallization_threshold
        
        return crystallization_score, is_crystallized

    def get_cohesion_summary(self) -> Dict[str, Any]:
        """Get comprehensive cohesion analysis"""
        crystallization_score, is_crystallized = self.assess_crystallization_potential()
        
        return {
            'system_entropy': self.system_entropy,
            'coherence_score': self.coherence_score,
            'harmony_index': self.harmony_index,
            'crystallization_score': crystallization_score,
            'is_crystallized': is_crystallized,
            'entropy_status': 'LOW' if self.system_entropy < self.entropy_threshold else 'HIGH',
            'coherence_status': 'HIGH' if self.coherence_score > self.coherence_threshold else 'LOW',
            'harmony_status': 'OPTIMAL' if self.harmony_index > self.harmony_threshold else 'DEVELOPING'
        }

class IntegratedConsciousnessSystem:
    """Main integrated consciousness system combining all components"""
    
    def __init__(self, 
                 dimensions: int = 256,
                 max_nodes: int = 2000,
                 use_gpu: bool = True,
                 integration_level: ConsciousnessIntegrationLevel = ConsciousnessIntegrationLevel.FRACTAL_INTEGRATION):
        
        self.integration_level = integration_level
        self.dimensions = dimensions
        self.max_nodes = max_nodes
        
        # Initialize core components
        self.latent_space = LatentSpace(dimensions, use_gpu)
        self.mycelial_engine = EnhancedMycelialEngine(max_nodes)
        self.attention_field = AttentionField(self.latent_space)
        
        # Initialize AI components
        self.fractal_ai = EnhancedFractalAI(self.latent_space)
        self.feedback_loop = EnhancedFeedbackLoop(self.latent_space, self.fractal_ai)
        self.self_model = SelfModel(self.latent_space)
        self.cohesion_layer = CohesionLayer(self.latent_space, self.feedback_loop, self.self_model)
        
        # Initialize existing consciousness components based on integration level
        self.universal_orchestrator = None
        self.radiotrophic_engine = None
        self.bio_digital_hybrid = None
        
        if integration_level.value >= ConsciousnessIntegrationLevel.RADIOTROPHIC_INTEGRATION.value:
            try:
                self.radiotrophic_engine = RadiotrophicMycelialEngine(max_nodes, dimensions)
                logger.info("Radiotrophic engine integrated")
            except Exception as e:
                logger.warning(f"Radiotrophic engine not available: {e}")
        
        if integration_level.value >= ConsciousnessIntegrationLevel.UNIVERSAL_INTEGRATION.value:
            try:
                self.universal_orchestrator = UniversalConsciousnessOrchestrator()
                self.bio_digital_hybrid = BioDigitalHybridIntelligence()
                logger.info("Universal consciousness components integrated")
            except Exception as e:
                logger.warning(f"Universal consciousness not available: {e}")
        
        # System state
        self.processing_history = deque(maxlen=1000)
        self.consciousness_emergence_events = []
        self.total_processing_cycles = 0
        
        logger.info(f"Integrated Consciousness System initialized at {integration_level.name} level")

    async def process_consciousness_cycle(self, 
                                        input_data: Dict[str, Any],
                                        radiation_level: float = 1.0,
                                        plant_signals: Optional[Dict[str, Any]] = None,
                                        environmental_data: Optional[Dict[str, Any]] = None) -> IntegratedConsciousnessMetrics:
        """Process a complete consciousness cycle with all integrated components"""
        
        start_time = datetime.now()
        self.total_processing_cycles += 1
        
        try:
            # Phase 1: Input processing and vector creation
            input_vectors = self._process_input_to_vectors(input_data)
            
            # Phase 2: Add experiences to mycelial network
            experience_ids = []
            for i, (vector_id, vector_data) in enumerate(input_vectors.items()):
                experience_data = {
                    'input_data': input_data,
                    'vector_id': vector_id,
                    'timestamp': datetime.now(),
                    'strength': np.linalg.norm(vector_data.cpu().numpy()) if isinstance(vector_data, torch.Tensor) else 0.5
                }
                
                self.mycelial_engine.add_experience(vector_id, experience_data)
                experience_ids.append(vector_id)
            
            # Phase 3: Attention processing
            resonance_data = self.attention_field.sense_resonance()
            focused_vectors = []
            
            # Focus on top resonant vectors
            if resonance_data:
                top_vectors = sorted(resonance_data.items(), key=lambda x: x[1], reverse=True)[:3]
                for vec_id, _ in top_vectors:
                    focus_result = self.attention_field.focus_on(vec_id)
                    if focus_result:
                        focused_vectors.append(focus_result)
            
            # Phase 4: Fractal AI processing and adaptation
            adaptation_results = {}
            for vec_id in input_vectors.keys():
                # Update AI model
                self.fractal_ai.update_model(vec_id)
                
                # Drive adaptation
                adapted_vector = self.feedback_loop.drive_adaptation(vec_id)
                if adapted_vector is not None:
                    adaptation_results[vec_id] = adapted_vector
            
            # Phase 5: Self-model updates
            identity_vector = self.self_model.compute_i_vector()
            consistency = self.self_model.compute_consistency()
            metacognitive_awareness = self.self_model.measure_metacognitive_awareness()
            
            # Phase 6: Cohesion analysis
            cohesion_summary = self.cohesion_layer.get_cohesion_summary()
            
            # Phase 7: Integration with advanced systems (if available)
            advanced_processing = {}
            
            if self.radiotrophic_engine and radiation_level > 0.1:
                # Process through radiotrophic engine
                radio_consciousness_data = {
                    'ecosystem': environmental_data or {},
                    'plant': plant_signals or {}
                }
                advanced_processing['radiotrophic'] = self.radiotrophic_engine.process_radiation_enhanced_input(
                    radio_consciousness_data, radiation_level
                )
            
            if self.universal_orchestrator and self.bio_digital_hybrid:
                # Process through universal consciousness
                consciousness_result = await self.universal_orchestrator.consciousness_cycle(
                    input_data, plant_signals, environmental_data
                )
                advanced_processing['universal'] = consciousness_result
                
                # Bio-digital hybrid processing
                hybrid_result = await self.bio_digital_hybrid.process_hybrid_intelligence(
                    input_data, radiation_level
                )
                advanced_processing['bio_digital'] = hybrid_result
            
            # Phase 8: Create integrated metrics
            metrics = self._compute_integrated_metrics(
                input_vectors, focused_vectors, adaptation_results,
                cohesion_summary, advanced_processing
            )
            
            # Phase 9: Check for consciousness emergence
            emergence_detected = self._detect_consciousness_emergence(metrics)
            if emergence_detected:
                self.consciousness_emergence_events.append(metrics)
            
            # Record processing
            processing_record = {
                'timestamp': start_time,
                'metrics': metrics,
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'emergence_detected': emergence_detected
            }
            self.processing_history.append(processing_record)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Consciousness processing error: {e}")
            # Return minimal metrics on error
            return IntegratedConsciousnessMetrics(
                timestamp=start_time,
                integration_level=self.integration_level,
                fractal_coherence=0.0,
                mycelial_connectivity=0.0,
                quantum_entanglement=0.0,
                radiotrophic_efficiency=0.0,
                universal_harmony=0.0,
                total_processing_nodes=0,
                active_consciousness_streams=0,
                emergent_patterns_detected=0
            )

    def _process_input_to_vectors(self, input_data: Dict[str, Any]) -> Dict[int, torch.Tensor]:
        """Convert input data to latent space vectors"""
        vectors = {}
        
        # Extract numerical features
        numerical_features = []
        for key, value in input_data.items():
            if isinstance(value, (int, float)):
                numerical_features.append(float(value))
            elif isinstance(value, str):
                # Simple string to numerical conversion
                numerical_features.append(float(hash(value) % 1000) / 1000.0)
            elif isinstance(value, (list, tuple)):
                # Handle list/tuple inputs
                for item in value:
                    if isinstance(item, (int, float)):
                        numerical_features.append(float(item))
        
        # Pad or truncate to match dimensions
        if len(numerical_features) < self.dimensions:
            # Pad with noise
            padding_size = self.dimensions - len(numerical_features)
            padding = np.random.normal(0, 0.1, padding_size).tolist()
            numerical_features.extend(padding)
        else:
            numerical_features = numerical_features[:self.dimensions]
        
        # Create vector and add to latent space
        vector_id = len(self.latent_space.vectors) + 1
        vector_tensor = torch.tensor(numerical_features, dtype=torch.float32, device=self.latent_space.device)
        
        self.latent_space.add_vector(vector_id, vector_tensor)
        vectors[vector_id] = vector_tensor
        
        return vectors

# Continue with Part 3...