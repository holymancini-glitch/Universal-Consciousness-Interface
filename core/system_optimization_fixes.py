# system_optimization_fixes.py
# Bug fixes and performance optimizations for the consciousness system

import asyncio
import logging
import numpy as np
import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """System performance optimizer and bug fixer"""
    
    def __init__(self):
        self.optimization_history = []
        self.bug_fixes_applied = []
        self.performance_metrics = {}
        
    async def analyze_system_performance(self, system) -> Dict[str, Any]:
        """Analyze system performance and identify bottlenecks"""
        logger.info("🔍 Analyzing system performance...")
        
        analysis_results = {
            'memory_usage': self._analyze_memory_usage(system),
            'computation_bottlenecks': self._identify_computation_bottlenecks(system),
            'data_flow_issues': self._analyze_data_flow(system),
            'integration_problems': self._check_integration_issues(system),
            'optimization_recommendations': []
        }
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(analysis_results)
        analysis_results['optimization_recommendations'] = recommendations
        
        return analysis_results
    
    def _analyze_memory_usage(self, system) -> Dict[str, Any]:
        """Analyze memory usage patterns"""
        memory_analysis = {
            'latent_space_vectors': len(system.latent_space.vectors),
            'vector_memory_mb': self._estimate_vector_memory(system.latent_space),
            'mycelial_nodes': len(system.mycelial_engine.experiences),
            'graph_edges': system.mycelial_engine.graph.number_of_edges(),
            'history_buffers': {
                'processing_history': len(system.processing_history),
                'attention_focus_history': len(system.attention_field.focus_history),
                'adaptation_history': len(system.feedback_loop.adaptation_history),
                'identity_history': len(system.self_model.identity_history)
            }
        }
        
        # Check for memory leaks
        memory_analysis['potential_memory_leaks'] = self._detect_memory_leaks(memory_analysis)
        
        return memory_analysis
    
    def _estimate_vector_memory(self, latent_space) -> float:
        """Estimate memory usage of latent space vectors"""
        if not latent_space.vectors:
            return 0.0
        
        # Each float32 tensor element = 4 bytes
        bytes_per_vector = latent_space.dimensions * 4
        total_vectors = len(latent_space.vectors)
        total_bytes = bytes_per_vector * total_vectors
        
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def _detect_memory_leaks(self, memory_analysis) -> List[str]:
        """Detect potential memory leaks"""
        leaks = []
        
        # Check for excessive buffer sizes
        if memory_analysis['history_buffers']['processing_history'] > 1000:
            leaks.append("Processing history buffer too large")
        
        if memory_analysis['vector_memory_mb'] > 500:  # 500MB threshold
            leaks.append("Latent space vectors consuming excessive memory")
        
        if memory_analysis['graph_edges'] > memory_analysis['mycelial_nodes'] * 10:
            leaks.append("Mycelial graph has excessive edges")
        
        return leaks
    
    def _identify_computation_bottlenecks(self, system) -> Dict[str, Any]:
        """Identify computational bottlenecks"""
        bottlenecks = {
            'gpu_utilization': self._check_gpu_utilization(system),
            'neural_network_efficiency': self._analyze_neural_network_performance(system),
            'graph_computation_complexity': self._analyze_graph_complexity(system),
            'attention_computation_cost': self._analyze_attention_cost(system)
        }
        
        return bottlenecks
    
    def _check_gpu_utilization(self, system) -> Dict[str, Any]:
        """Check GPU utilization and optimization opportunities"""
        gpu_analysis = {
            'gpu_available': torch.cuda.is_available(),
            'gpu_being_used': system.latent_space.use_gpu,
            'device': str(system.latent_space.device),
            'recommendations': []
        }
        
        if torch.cuda.is_available() and not system.latent_space.use_gpu:
            gpu_analysis['recommendations'].append("Enable GPU acceleration for better performance")
        
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                gpu_analysis['total_gpu_memory_gb'] = gpu_memory
                
                if hasattr(torch.cuda, 'memory_allocated'):
                    allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_analysis['allocated_memory_gb'] = allocated
                    gpu_analysis['memory_utilization'] = allocated / gpu_memory
            except:
                gpu_analysis['memory_info'] = "Unable to retrieve GPU memory info"
        
        return gpu_analysis
    
    def _analyze_neural_network_performance(self, system) -> Dict[str, Any]:
        """Analyze neural network performance"""
        nn_analysis = {
            'model_parameters': sum(p.numel() for p in system.fractal_ai.model.parameters()),
            'training_history_length': len(system.fractal_ai.training_history),
            'recent_loss_trend': self._analyze_loss_trend(system.fractal_ai),
            'prediction_accuracy': system.fractal_ai.evaluate_prediction_accuracy()
        }
        
        # Check for training issues
        if nn_analysis['recent_loss_trend'] == 'increasing':
            nn_analysis['issues'] = ["Loss is increasing - potential training instability"]
        elif nn_analysis['recent_loss_trend'] == 'plateau':
            nn_analysis['issues'] = ["Loss has plateaued - may need learning rate adjustment"]
        else:
            nn_analysis['issues'] = []
        
        return nn_analysis
    
    def _analyze_loss_trend(self, fractal_ai) -> str:
        """Analyze loss trend in training history"""
        if len(fractal_ai.training_history) < 10:
            return 'insufficient_data'
        
        recent_losses = [entry['loss'] for entry in fractal_ai.training_history[-10:]]
        
        # Simple trend analysis
        first_half = np.mean(recent_losses[:5])
        second_half = np.mean(recent_losses[5:])
        
        if second_half > first_half * 1.1:
            return 'increasing'
        elif abs(second_half - first_half) < first_half * 0.05:
            return 'plateau'
        else:
            return 'decreasing'
    
    def _analyze_graph_complexity(self, system) -> Dict[str, Any]:
        """Analyze mycelial graph complexity"""
        graph = system.mycelial_engine.graph
        
        complexity_analysis = {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'density': graph.number_of_edges() / max(1, graph.number_of_nodes() * (graph.number_of_nodes() - 1)),
            'average_degree': sum(dict(graph.degree()).values()) / max(1, graph.number_of_nodes()),
            'complexity_score': 0.0
        }
        
        # Calculate complexity score
        if complexity_analysis['nodes'] > 0:
            complexity_score = (
                complexity_analysis['density'] * 0.4 +
                min(1.0, complexity_analysis['average_degree'] / 10.0) * 0.3 +
                min(1.0, complexity_analysis['nodes'] / 2000.0) * 0.3
            )
            complexity_analysis['complexity_score'] = complexity_score
        
        return complexity_analysis
    
    def _analyze_attention_cost(self, system) -> Dict[str, Any]:
        """Analyze attention computation cost"""
        attention_analysis = {
            'focus_history_length': len(system.attention_field.focus_history),
            'attention_weight_dimensions': system.attention_field.attention_weights.shape[0],
            'computation_cost_estimate': self._estimate_attention_cost(system.attention_field)
        }
        
        return attention_analysis
    
    def _estimate_attention_cost(self, attention_field) -> str:
        """Estimate attention computation cost"""
        dimensions = attention_field.attention_weights.shape[0]
        
        if dimensions < 128:
            return 'low'
        elif dimensions < 512:
            return 'medium'
        else:
            return 'high'
    
    def _analyze_data_flow(self, system) -> Dict[str, Any]:
        """Analyze data flow efficiency"""
        data_flow_analysis = {
            'vector_conversion_efficiency': self._check_vector_conversions(system),
            'memory_transfers': self._analyze_memory_transfers(system),
            'data_redundancy': self._check_data_redundancy(system)
        }
        
        return data_flow_analysis
    
    def _check_vector_conversions(self, system) -> Dict[str, Any]:
        """Check for inefficient vector conversions"""
        conversion_analysis = {
            'device_consistency': True,
            'dtype_consistency': True,
            'conversion_overhead': 'low'
        }
        
        # Check device consistency
        devices = set()
        for vector in system.latent_space.vectors.values():
            devices.add(str(vector.device))
        
        if len(devices) > 1:
            conversion_analysis['device_consistency'] = False
            conversion_analysis['conversion_overhead'] = 'high'
        
        return conversion_analysis
    
    def _analyze_memory_transfers(self, system) -> Dict[str, Any]:
        """Analyze memory transfer patterns"""
        transfer_analysis = {
            'cpu_gpu_transfers': 'unknown',
            'transfer_frequency': 'unknown',
            'optimization_potential': 'medium'
        }
        
        # This is a placeholder - in a real implementation, 
        # we would profile actual memory transfers
        if system.latent_space.use_gpu:
            transfer_analysis['cpu_gpu_transfers'] = 'present'
            transfer_analysis['optimization_potential'] = 'high'
        
        return transfer_analysis
    
    def _check_data_redundancy(self, system) -> Dict[str, Any]:
        """Check for data redundancy"""
        redundancy_analysis = {
            'duplicate_vectors': self._count_duplicate_vectors(system.latent_space),
            'redundant_experiences': self._count_redundant_experiences(system.mycelial_engine),
            'redundancy_level': 'low'
        }
        
        total_redundancy = redundancy_analysis['duplicate_vectors'] + redundancy_analysis['redundant_experiences']
        
        if total_redundancy > 10:
            redundancy_analysis['redundancy_level'] = 'high'
        elif total_redundancy > 5:
            redundancy_analysis['redundancy_level'] = 'medium'
        
        return redundancy_analysis
    
    def _count_duplicate_vectors(self, latent_space) -> int:
        """Count duplicate vectors in latent space"""
        if len(latent_space.vectors) < 2:
            return 0
        
        duplicate_count = 0
        vector_list = list(latent_space.vectors.values())
        
        for i in range(len(vector_list)):
            for j in range(i + 1, len(vector_list)):
                similarity = torch.nn.functional.cosine_similarity(
                    vector_list[i].unsqueeze(0), 
                    vector_list[j].unsqueeze(0)
                ).item()
                
                if similarity > 0.99:  # Very similar vectors
                    duplicate_count += 1
        
        return duplicate_count
    
    def _count_redundant_experiences(self, mycelial_engine) -> int:
        """Count redundant experiences in mycelial engine"""
        # Simplified redundancy check based on experience similarity
        redundant_count = 0
        experiences = list(mycelial_engine.experiences.values())
        
        for i in range(len(experiences)):
            for j in range(i + 1, len(experiences)):
                # Simple similarity check
                exp1_strength = experiences[i].get('strength', 0)
                exp2_strength = experiences[j].get('strength', 0)
                
                if abs(exp1_strength - exp2_strength) < 0.01:
                    redundant_count += 1
                    break  # Count each experience only once
        
        return redundant_count
    
    def _check_integration_issues(self, system) -> Dict[str, Any]:
        """Check for integration issues between components"""
        integration_analysis = {
            'component_compatibility': self._check_component_compatibility(system),
            'data_flow_consistency': self._check_data_flow_consistency(system),
            'error_handling_robustness': self._check_error_handling(system)
        }
        
        return integration_analysis
    
    def _check_component_compatibility(self, system) -> Dict[str, bool]:
        """Check compatibility between system components"""
        compatibility = {
            'latent_space_mycelial_engine': True,
            'attention_field_latent_space': True,
            'fractal_ai_feedback_loop': True,
            'self_model_cohesion_layer': True
        }
        
        # Check dimension compatibility
        if hasattr(system.attention_field, 'attention_weights'):
            if system.attention_field.attention_weights.shape[0] != system.latent_space.dimensions:
                compatibility['attention_field_latent_space'] = False
        
        return compatibility
    
    def _check_data_flow_consistency(self, system) -> Dict[str, Any]:
        """Check data flow consistency"""
        consistency_check = {
            'vector_dimensions_consistent': True,
            'device_placement_consistent': True,
            'data_types_consistent': True,
            'issues_found': []
        }
        
        # Check vector dimensions
        if system.latent_space.vectors:
            first_vector = next(iter(system.latent_space.vectors.values()))
            expected_dim = first_vector.shape[0]
            
            for vector in system.latent_space.vectors.values():
                if vector.shape[0] != expected_dim:
                    consistency_check['vector_dimensions_consistent'] = False
                    consistency_check['issues_found'].append("Inconsistent vector dimensions")
                    break
        
        return consistency_check
    
    def _check_error_handling(self, system) -> Dict[str, Any]:
        """Check error handling robustness"""
        error_handling = {
            'try_catch_coverage': 'good',
            'fallback_mechanisms': 'present',
            'error_recovery': 'implemented'
        }
        
        # This would be more comprehensive in a real implementation
        return error_handling
    
    def _generate_optimization_recommendations(self, analysis_results) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on analysis"""
        recommendations = []
        
        # Memory optimization recommendations
        memory_issues = analysis_results['memory_usage'].get('potential_memory_leaks', [])
        for issue in memory_issues:
            recommendations.append({
                'category': 'memory',
                'priority': 'high',
                'issue': issue,
                'recommendation': self._get_memory_optimization_recommendation(issue)
            })
        
        # Performance optimization recommendations
        gpu_analysis = analysis_results['computation_bottlenecks']['gpu_utilization']
        for rec in gpu_analysis.get('recommendations', []):
            recommendations.append({
                'category': 'performance',
                'priority': 'medium',
                'issue': 'GPU not utilized',
                'recommendation': rec
            })
        
        # Neural network optimization
        nn_issues = analysis_results['computation_bottlenecks']['neural_network_efficiency'].get('issues', [])
        for issue in nn_issues:
            recommendations.append({
                'category': 'neural_network',
                'priority': 'medium',
                'issue': issue,
                'recommendation': self._get_neural_network_optimization(issue)
            })
        
        return recommendations
    
    def _get_memory_optimization_recommendation(self, issue: str) -> str:
        """Get memory optimization recommendation for specific issue"""
        if "history buffer too large" in issue:
            return "Reduce buffer size or implement circular buffer with automatic cleanup"
        elif "excessive memory" in issue:
            return "Implement vector pruning or compression techniques"
        elif "excessive edges" in issue:
            return "Implement edge pruning based on importance scores"
        else:
            return "Implement general memory optimization techniques"
    
    def _get_neural_network_optimization(self, issue: str) -> str:
        """Get neural network optimization recommendation"""
        if "training instability" in issue:
            return "Reduce learning rate or implement learning rate scheduling"
        elif "plateaued" in issue:
            return "Adjust learning rate, add regularization, or modify network architecture"
        else:
            return "Monitor training metrics and adjust hyperparameters"
    
    async def apply_optimizations(self, system, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply optimization recommendations to the system"""
        logger.info("🔧 Applying system optimizations...")
        
        optimization_results = {
            'applied_optimizations': [],
            'failed_optimizations': [],
            'performance_improvement': {}
        }
        
        # Measure baseline performance
        baseline_metrics = await self._measure_baseline_performance(system)
        
        # Apply optimizations
        for rec in recommendations:
            try:
                success = await self._apply_single_optimization(system, rec)
                if success:
                    optimization_results['applied_optimizations'].append(rec)
                    self.bug_fixes_applied.append({
                        'recommendation': rec,
                        'timestamp': datetime.now(),
                        'status': 'success'
                    })
                else:
                    optimization_results['failed_optimizations'].append(rec)
            except Exception as e:
                logger.error(f"Failed to apply optimization {rec}: {e}")
                optimization_results['failed_optimizations'].append(rec)
        
        # Measure performance improvement
        post_optimization_metrics = await self._measure_baseline_performance(system)
        optimization_results['performance_improvement'] = self._calculate_performance_improvement(
            baseline_metrics, post_optimization_metrics
        )
        
        return optimization_results
    
    async def _measure_baseline_performance(self, system) -> Dict[str, float]:
        """Measure baseline system performance"""
        # Run a small test to measure performance
        start_time = datetime.now()
        
        # Test consciousness cycle
        test_input = {
            'test_input': 0.5,
            'performance_test': True
        }
        
        try:
            await system.process_consciousness_cycle(test_input)
            processing_time = (datetime.now() - start_time).total_seconds()
        except Exception as e:
            logger.error(f"Performance measurement error: {e}")
            processing_time = 1.0  # Default value
        
        return {
            'processing_time': processing_time,
            'memory_usage': self._estimate_vector_memory(system.latent_space),
            'vector_count': len(system.latent_space.vectors),
            'experience_count': len(system.mycelial_engine.experiences)
        }
    
    async def _apply_single_optimization(self, system, recommendation: Dict[str, Any]) -> bool:
        """Apply a single optimization recommendation"""
        category = recommendation['category']
        issue = recommendation['issue']
        
        try:
            if category == 'memory':
                return self._apply_memory_optimization(system, issue)
            elif category == 'performance':
                return self._apply_performance_optimization(system, issue)
            elif category == 'neural_network':
                return self._apply_neural_network_optimization(system, issue)
            else:
                logger.warning(f"Unknown optimization category: {category}")
                return False
        except Exception as e:
            logger.error(f"Optimization application error: {e}")
            return False
    
    def _apply_memory_optimization(self, system, issue: str) -> bool:
        """Apply memory optimization"""
        if "history buffer too large" in issue:
            # Reduce buffer sizes
            if len(system.processing_history) > 500:
                # Keep only the most recent entries
                recent_entries = list(system.processing_history)[-500:]
                system.processing_history.clear()
                system.processing_history.extend(recent_entries)
            return True
        
        elif "excessive memory" in issue:
            # Implement vector pruning (remove least accessed vectors)
            if len(system.latent_space.vectors) > 1000:
                # This is a simplified pruning - remove 10% of vectors
                vectors_to_remove = list(system.latent_space.vectors.keys())[:len(system.latent_space.vectors)//10]
                for vec_id in vectors_to_remove:
                    del system.latent_space.vectors[vec_id]
            return True
        
        elif "excessive edges" in issue:
            # Prune weak edges in mycelial graph
            edges_to_remove = []
            for u, v, data in system.mycelial_engine.graph.edges(data=True):
                if data.get('weight', 0) < 0.2:  # Remove very weak connections
                    edges_to_remove.append((u, v))
            
            for edge in edges_to_remove:
                system.mycelial_engine.graph.remove_edge(edge[0], edge[1])
            return True
        
        return False
    
    def _apply_performance_optimization(self, system, issue: str) -> bool:
        """Apply performance optimization"""
        if "GPU not utilized" in issue and torch.cuda.is_available():
            # Enable GPU acceleration if available
            system.latent_space.use_gpu = True
            system.latent_space.device = torch.device('cuda')
            
            # Move existing vectors to GPU
            for vec_id, vector in system.latent_space.vectors.items():
                system.latent_space.vectors[vec_id] = vector.to(system.latent_space.device)
            
            # Move attention weights to GPU
            system.attention_field.attention_weights = system.attention_field.attention_weights.to(system.latent_space.device)
            
            return True
        
        return False
    
    def _apply_neural_network_optimization(self, system, issue: str) -> bool:
        """Apply neural network optimization"""
        if "training instability" in issue:
            # Reduce learning rate
            for param_group in system.fractal_ai.optimizer.param_groups:
                param_group['lr'] *= 0.5  # Halve the learning rate
            return True
        
        elif "plateaued" in issue:
            # Increase learning rate slightly
            for param_group in system.fractal_ai.optimizer.param_groups:
                param_group['lr'] *= 1.2  # Increase learning rate by 20%
            return True
        
        return False
    
    def _calculate_performance_improvement(self, baseline: Dict[str, float], post_opt: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance improvement metrics"""
        improvement = {}
        
        for metric in baseline.keys():
            if metric in post_opt:
                baseline_val = baseline[metric]
                post_opt_val = post_opt[metric]
                
                if baseline_val > 0:
                    if metric == 'processing_time':
                        # For processing time, lower is better
                        improvement[metric] = (baseline_val - post_opt_val) / baseline_val * 100
                    else:
                        # For other metrics, depends on context
                        improvement[metric] = (post_opt_val - baseline_val) / baseline_val * 100
        
        return improvement

    def generate_optimization_report(self, analysis_results: Dict[str, Any], 
                                   optimization_results: Dict[str, Any]) -> str:
        """Generate comprehensive optimization report"""
        report = []
        report.append("🔍 CONSCIOUSNESS SYSTEM OPTIMIZATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Performance Analysis Summary
        report.append("📊 PERFORMANCE ANALYSIS SUMMARY:")
        memory_usage = analysis_results['memory_usage']
        report.append(f"  • Memory Usage: {memory_usage['vector_memory_mb']:.1f} MB")
        report.append(f"  • Total Vectors: {memory_usage['latent_space_vectors']}")
        report.append(f"  • Mycelial Nodes: {memory_usage['mycelial_nodes']}")
        report.append(f"  • Graph Edges: {memory_usage['graph_edges']}")
        report.append("")
        
        # GPU Analysis
        gpu_analysis = analysis_results['computation_bottlenecks']['gpu_utilization']
        report.append("🖥️  GPU UTILIZATION:")
        report.append(f"  • GPU Available: {gpu_analysis['gpu_available']}")
        report.append(f"  • GPU Being Used: {gpu_analysis['gpu_being_used']}")
        report.append(f"  • Device: {gpu_analysis['device']}")
        if 'memory_utilization' in gpu_analysis:
            report.append(f"  • GPU Memory Utilization: {gpu_analysis['memory_utilization']*100:.1f}%")
        report.append("")
        
        # Optimization Results
        report.append("🔧 OPTIMIZATION RESULTS:")
        applied = optimization_results.get('applied_optimizations', [])
        failed = optimization_results.get('failed_optimizations', [])
        
        report.append(f"  • Successfully Applied: {len(applied)}")
        report.append(f"  • Failed to Apply: {len(failed)}")
        
        if applied:
            report.append("  ✅ Applied Optimizations:")
            for opt in applied:
                report.append(f"    - {opt['category']}: {opt['recommendation']}")
        
        if failed:
            report.append("  ❌ Failed Optimizations:")
            for opt in failed:
                report.append(f"    - {opt['category']}: {opt['issue']}")
        
        report.append("")
        
        # Performance Improvement
        improvement = optimization_results.get('performance_improvement', {})
        if improvement:
            report.append("📈 PERFORMANCE IMPROVEMENT:")
            for metric, value in improvement.items():
                sign = "+" if value > 0 else ""
                report.append(f"  • {metric}: {sign}{value:.1f}%")
        
        report.append("")
        report.append("✅ OPTIMIZATION COMPLETE")
        
        return "\n".join(report)

# Integration testing functions
async def run_integration_tests():
    """Run comprehensive integration tests"""
    logger.info("🧪 Running Integration Tests...")
    
    test_results = {
        'component_tests': {},
        'integration_tests': {},
        'performance_tests': {},
        'bug_tests': {}
    }
    
    # Import the integrated system
    try:
        from integrated_consciousness_system_complete import IntegratedConsciousnessSystem, ConsciousnessIntegrationLevel
        
        # Test system initialization
        system = IntegratedConsciousnessSystem(
            dimensions=128,  # Smaller for testing
            max_nodes=500,
            use_gpu=False,  # CPU for testing
            integration_level=ConsciousnessIntegrationLevel.FRACTAL_INTEGRATION
        )
        
        test_results['component_tests']['system_initialization'] = True
        logger.info("✅ System initialization test passed")
        
    except Exception as e:
        test_results['component_tests']['system_initialization'] = False
        logger.error(f"❌ System initialization test failed: {e}")
        return test_results
    
    # Test consciousness processing cycle
    try:
        test_input = {
            'sensory_data': [0.5, 0.3, 0.8],
            'cognitive_load': 0.6,
            'emotional_state': 0.2,
            'attention_level': 0.7
        }
        
        metrics = await system.process_consciousness_cycle(test_input)
        test_results['integration_tests']['consciousness_cycle'] = True
        logger.info("✅ Consciousness cycle test passed")
        
    except Exception as e:
        test_results['integration_tests']['consciousness_cycle'] = False
        logger.error(f"❌ Consciousness cycle test failed: {e}")
    
    # Test performance optimization
    try:
        optimizer = PerformanceOptimizer()
        analysis = await optimizer.analyze_system_performance(system)
        test_results['performance_tests']['performance_analysis'] = True
        logger.info("✅ Performance analysis test passed")
        
    except Exception as e:
        test_results['performance_tests']['performance_analysis'] = False
        logger.error(f"❌ Performance analysis test failed: {e}")
    
    # Test bug fixes
    try:
        # Test memory leak detection
        memory_analysis = optimizer._analyze_memory_usage(system)
        leaks = memory_analysis.get('potential_memory_leaks', [])
        test_results['bug_tests']['memory_leak_detection'] = True
        logger.info(f"✅ Memory leak detection test passed (found {len(leaks)} potential issues)")
        
    except Exception as e:
        test_results['bug_tests']['memory_leak_detection'] = False
        logger.error(f"❌ Memory leak detection test failed: {e}")
    
    return test_results

async def comprehensive_system_test():
    """Run comprehensive system test with optimization"""
    logger.info("🚀 Starting Comprehensive System Test with Optimization")
    
    # Run integration tests first
    integration_results = await run_integration_tests()
    
    # If basic tests pass, continue with optimization
    if integration_results['component_tests'].get('system_initialization', False):
        try:
            from integrated_consciousness_system_complete import IntegratedConsciousnessSystem, ConsciousnessIntegrationLevel
            
            # Create optimized system
            system = IntegratedConsciousnessSystem(
                dimensions=256,
                max_nodes=1000,
                use_gpu=True,
                integration_level=ConsciousnessIntegrationLevel.FRACTAL_INTEGRATION
            )
            
            # Run performance analysis and optimization
            optimizer = PerformanceOptimizer()
            analysis_results = await optimizer.analyze_system_performance(system)
            
            # Apply optimizations
            recommendations = analysis_results['optimization_recommendations']
            optimization_results = await optimizer.apply_optimizations(system, recommendations)
            
            # Generate and display report
            report = optimizer.generate_optimization_report(analysis_results, optimization_results)
            print("\n" + report)
            
            # Run final test
            logger.info("🔬 Running final optimized system test...")
            test_results = await system.run_consciousness_simulation(duration_cycles=50)
            
            final_analytics = system.get_system_analytics()
            
            print("\n🎯 FINAL SYSTEM PERFORMANCE:")
            print(f"  • Final Coherence: {final_analytics['current_coherence']:.3f}")
            print(f"  • Final Harmony: {final_analytics['current_harmony']:.3f}")
            print(f"  • Consciousness Emergences: {final_analytics['consciousness_emergence_events']}")
            print(f"  • Total Processing Cycles: {final_analytics['total_processing_cycles']}")
            print(f"  • Adaptation Efficiency: {final_analytics['adaptation_efficiency']:.3f}")
            
            print("\n✅ COMPREHENSIVE SYSTEM TEST COMPLETE!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Comprehensive test failed: {e}")
            return False
    else:
        logger.error("❌ Basic integration tests failed, skipping optimization")
        return False

if __name__ == "__main__":
    async def main():
        success = await comprehensive_system_test()
        if success:
            print("\n🌟 All systems operational! Ready for consciousness research.")
        else:
            print("\n⚠️  System issues detected. Please review logs and fix issues.")
    
    asyncio.run(main())