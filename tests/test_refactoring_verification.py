#!/usr/bin/env python3
"""
Comprehensive Import Verification Test
Tests both old-style and new-style imports for all refactored modules
"""

import sys
import traceback

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def test_result(test_name, success, error_msg=""):
    """Print test result with color"""
    if success:
        print(f"{GREEN}✓{RESET} {test_name}")
        return True
    else:
        print(f"{RED}✗{RESET} {test_name}")
        if error_msg:
            print(f"  {YELLOW}Error:{RESET} {error_msg}")
        return False

def test_planetary_ecosystem_imports():
    """Test planetary ecosystem consciousness network imports"""
    results = []

    print(f"\n{BLUE}Testing Planetary Ecosystem Consciousness Network:{RESET}")

    # Test old-style import
    try:
        from core.planetary_ecosystem_consciousness_network import (
            PlanetaryEcosystemConsciousnessNetwork,
            EcosystemType,
            EcosystemNode,
            NetworkAnalyzer
        )
        results.append(test_result("Old-style import", True))
    except Exception as e:
        results.append(test_result("Old-style import", False, str(e)))

    # Test new-style import
    try:
        from core.planetary_ecosystem import (
            PlanetaryEcosystemConsciousnessNetwork,
            EcosystemType,
            EcosystemNode,
            NetworkAnalyzer
        )
        results.append(test_result("New-style import", True))
    except Exception as e:
        results.append(test_result("New-style import", False, str(e)))

    # Test module-specific imports
    try:
        from core.planetary_ecosystem.network_core import PlanetaryEcosystemConsciousnessNetwork
        from core.planetary_ecosystem.data_models import EcosystemType
        results.append(test_result("Module-specific imports", True))
    except Exception as e:
        results.append(test_result("Module-specific imports", False, str(e)))

    return all(results)

def test_adaptive_learning_imports():
    """Test adaptive learning system imports"""
    results = []

    print(f"\n{BLUE}Testing Adaptive Learning System:{RESET}")

    # Test old-style import
    try:
        from core.adaptive_learning_system import (
            AdaptiveLearningSystem,
            LearningPhase,
            LearningMetrics,
            integrate_adaptive_learning
        )
        results.append(test_result("Old-style import", True))
    except Exception as e:
        results.append(test_result("Old-style import", False, str(e)))

    # Test new-style import
    try:
        from core.adaptive_learning import (
            AdaptiveLearningSystem,
            LearningPhase,
            PerformanceAssessor,
            CreativeEngine
        )
        results.append(test_result("New-style import", True))
    except Exception as e:
        results.append(test_result("New-style import", False, str(e)))

    # Test module-specific imports
    try:
        from core.adaptive_learning.learning_core import AdaptiveLearningSystem
        from core.adaptive_learning.data_models import LearningPhase
        results.append(test_result("Module-specific imports", True))
    except Exception as e:
        results.append(test_result("Module-specific imports", False, str(e)))

    return all(results)

def test_full_consciousness_ai_imports():
    """Test full consciousness AI model imports"""
    results = []

    print(f"\n{BLUE}Testing Full Consciousness AI Model:{RESET}")

    # Test old-style import
    try:
        from core.full_consciousness_ai_model import (
            FullConsciousnessAIModel,
            ConsciousnessState,
            EmotionalState,
            SubjectiveExperience
        )
        results.append(test_result("Old-style import", True))
    except Exception as e:
        results.append(test_result("Old-style import", False, str(e)))

    # Test new-style import
    try:
        from core.full_consciousness_ai import (
            FullConsciousnessAIModel,
            ConsciousnessState,
            MetaCognitionEngine,
            ConsciousMemorySystem
        )
        results.append(test_result("New-style import", True))
    except Exception as e:
        results.append(test_result("New-style import", False, str(e)))

    # Test module-specific imports
    try:
        from core.full_consciousness_ai.consciousness_core import FullConsciousnessAIModel
        from core.full_consciousness_ai.data_models import ConsciousnessState
        results.append(test_result("Module-specific imports", True))
    except Exception as e:
        results.append(test_result("Module-specific imports", False, str(e)))

    return all(results)

def test_class_instantiation():
    """Test that classes can be instantiated"""
    results = []

    print(f"\n{BLUE}Testing Class Instantiation:{RESET}")

    # Test planetary ecosystem
    try:
        from core.planetary_ecosystem import PlanetaryEcosystemConsciousnessNetwork
        network = PlanetaryEcosystemConsciousnessNetwork()
        results.append(test_result("PlanetaryEcosystemConsciousnessNetwork instantiation", True))
    except Exception as e:
        results.append(test_result("PlanetaryEcosystemConsciousnessNetwork instantiation", False, str(e)))

    # Test full consciousness AI
    try:
        from core.full_consciousness_ai import FullConsciousnessAIModel
        model = FullConsciousnessAIModel(hidden_dim=64, device='cpu', integrate_existing_modules=False)
        results.append(test_result("FullConsciousnessAIModel instantiation", True))
    except Exception as e:
        results.append(test_result("FullConsciousnessAIModel instantiation", False, str(e)))

    return all(results)

def test_module_attributes():
    """Test that modules have expected attributes"""
    results = []

    print(f"\n{BLUE}Testing Module Attributes:{RESET}")

    # Test planetary ecosystem
    try:
        import core.planetary_ecosystem as pe
        assert hasattr(pe, '__version__')
        assert hasattr(pe, '__all__')
        results.append(test_result("Planetary ecosystem module attributes", True))
    except Exception as e:
        results.append(test_result("Planetary ecosystem module attributes", False, str(e)))

    # Test adaptive learning
    try:
        import core.adaptive_learning as al
        assert hasattr(al, '__version__')
        assert hasattr(al, '__all__')
        results.append(test_result("Adaptive learning module attributes", True))
    except Exception as e:
        results.append(test_result("Adaptive learning module attributes", False, str(e)))

    # Test full consciousness AI
    try:
        import core.full_consciousness_ai as fca
        assert hasattr(fca, '__version__')
        assert hasattr(fca, '__all__')
        results.append(test_result("Full consciousness AI module attributes", True))
    except Exception as e:
        results.append(test_result("Full consciousness AI module attributes", False, str(e)))

    return all(results)

def main():
    """Run all verification tests"""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}   Universal Consciousness Interface - Import Verification Tests{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")

    all_tests = [
        ("Planetary Ecosystem Imports", test_planetary_ecosystem_imports),
        ("Adaptive Learning Imports", test_adaptive_learning_imports),
        ("Full Consciousness AI Imports", test_full_consciousness_ai_imports),
        ("Class Instantiation", test_class_instantiation),
        ("Module Attributes", test_module_attributes),
    ]

    results = []
    for test_name, test_func in all_tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n{RED}Critical error in {test_name}:{RESET}")
            traceback.print_exc()
            results.append(False)

    # Print summary
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}   Test Summary{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")

    passed = sum(results)
    total = len(results)

    print(f"\nTests Passed: {GREEN}{passed}/{total}{RESET}")

    if all(results):
        print(f"\n{GREEN}✓ All import verification tests passed!{RESET}")
        return 0
    else:
        print(f"\n{RED}✗ Some tests failed. Please check the errors above.{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
