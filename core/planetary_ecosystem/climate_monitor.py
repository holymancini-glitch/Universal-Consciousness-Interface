"""
Climate Consciousness Monitor for Planetary Ecosystem

Monitors climate-related consciousness indicators and stability metrics
for the planetary ecosystem network.
"""

import logging
from typing import Dict, List

# Handle optional dependencies with fallbacks
try:
    import numpy as np  # type: ignore
except ImportError:
    import statistics

    class MockNumPy:
        @staticmethod
        def mean(values):
            return statistics.mean(values) if values else 0.0

        @staticmethod
        def std(values):
            return statistics.stdev(values) if len(values) > 1 else 0.0

    np = MockNumPy()

logger = logging.getLogger(__name__)


class ClimateConsciousnessMonitor:
    """Monitor for climate-related consciousness indicators"""

    def __init__(self) -> None:
        self.climate_data: Dict[str, List[float]] = {}
        logger.info("🌡️ Climate Consciousness Monitor Initialized")

    def update_climate_data(self, data: Dict[str, float]) -> None:
        """Update climate consciousness data"""
        for key, value in data.items():
            if key not in self.climate_data:
                self.climate_data[key] = []
            self.climate_data[key].append(value)

            # Keep only recent data (last 100 points)
            if len(self.climate_data[key]) > 100:
                self.climate_data[key].pop(0)

    def assess_climate_stability(self) -> float:
        """Assess climate stability based on recent data"""
        if not self.climate_data:
            return 0.5  # Neutral stability

        stability_scores = []

        # Assess stability for each climate parameter
        for param, values in self.climate_data.items():
            if len(values) < 10:
                continue  # Need sufficient data

            # Calculate variance as inverse of stability
            variance = np.std(values)

            # Convert to stability score (0-1)
            # Lower variance = higher stability
            max_expected_variance = 2.0  # Adjust based on parameter type
            stability = max(0.0, 1.0 - (variance / max_expected_variance))
            stability_scores.append(stability)

        return np.mean(stability_scores) if stability_scores else 0.5

    def predict_climate_trends(self) -> Dict[str, str]:
        """Predict climate trends based on consciousness data"""
        predictions = {}

        for param, values in self.climate_data.items():
            if len(values) < 5:
                predictions[param] = 'insufficient_data'
                continue

            # Simple trend analysis
            recent_values = values[-5:]
            if len(recent_values) < 2:
                predictions[param] = 'stable'
                continue

            # Calculate trend
            first = recent_values[0]
            last = recent_values[-1]

            if last > first + 0.5:
                trend = 'increasing'
            elif last < first - 0.5:
                trend = 'decreasing'
            else:
                trend = 'stable'

            predictions[param] = trend

        return predictions


__all__ = ['ClimateConsciousnessMonitor']
