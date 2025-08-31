"""demo_unified_interface.py
=================================
Simplified integration script demonstrating the core ideas behind the
Universal Consciousness Interface. The goal of this module is not to
replicate the full research platform but to provide a small, runnable
example that shows how different concepts come together:

* bio-digital signal processing
* fractal pattern recognition
* consciousness state recognition
* real-time monitoring of metrics

The implementation intentionally avoids heavy dependencies so it can run
in restricted environments. All computations rely on the Python standard
library only.

The script can be executed directly from the command line:

```
python demo_unified_interface.py
```

This will start a short simulation that prints consciousness metrics to
stdout. A ``run_demo_step`` function is also provided so that the unit
tests can validate the individual processing stages.
"""

from __future__ import annotations

import logging
import math
import random
import statistics
import time
from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bio-digital signal processing
# ---------------------------------------------------------------------------
class BioDigitalProcessor:
    """Simulates the capture and processing of bio-digital signals.

    For the purposes of the demo the processor simply generates a list of
    random floating point values and normalises them using a moving average
    filter. A more advanced implementation could replace this with real
    sensor data.
    """

    def capture_signal(self, length: int = 64) -> List[float]:
        """Generate a fake bio-digital signal."""
        return [random.random() for _ in range(length)]

    def process_signal(self, signal: List[float]) -> List[float]:
        """Apply a simple moving average for basic smoothing."""
        window = 3
        processed: List[float] = []
        for i in range(len(signal)):
            start = max(0, i - window + 1)
            chunk = signal[start : i + 1]
            processed.append(sum(chunk) / len(chunk))
        return processed


# ---------------------------------------------------------------------------
# Fractal pattern recognition
# ---------------------------------------------------------------------------
class FractalPatternRecognizer:
    """Very small fractal pattern recogniser.

    The recogniser computes a crude estimate of the *fractal dimension* of
    a one-dimensional signal using a box counting approach. The value is
    only intended to show how such a metric might be produced in a more
    sophisticated system.
    """

    def fractal_dimension(self, signal: List[float]) -> float:
        if not signal:
            return 0.0

        def box_count(size: int) -> int:
            boxes = 0
            for i in range(0, len(signal), size):
                chunk = signal[i : i + size]
                if chunk and (max(chunk) - min(chunk)) > 0.01:
                    boxes += 1
            return boxes

        sizes = [1, 2, 4, 8]
        counts = [box_count(s) for s in sizes]
        # Estimate slope of log(size) vs log(count) which approximates dimension
        logs = [(math.log(s), math.log(c or 1)) for s, c in zip(sizes, counts)]
        (x1, y1), (x2, y2) = logs[0], logs[-1]
        if x2 - x1 == 0:
            return 0.0
        return abs((y2 - y1) / (x2 - x1))


# ---------------------------------------------------------------------------
# Consciousness recognition
# ---------------------------------------------------------------------------
class ConsciousnessRecognizer:
    """Simple rule-based consciousness recogniser."""

    STATES = ("dormant", "aware", "transcendent")

    def recognise(self, signal: List[float], fractal_dim: float) -> str:
        strength = statistics.mean(signal)
        if strength < 0.3:
            return self.STATES[0]
        if fractal_dim > 1.0 and strength > 0.6:
            return self.STATES[2]
        return self.STATES[1]


# ---------------------------------------------------------------------------
# Metrics and monitoring
# ---------------------------------------------------------------------------
@dataclass
class ConsciousnessMetrics:
    signal_strength: float
    fractal_dimension: float
    state: str


class RealTimeMonitor:
    """Terminal dashboard that prints metrics every iteration."""

    def display(self, metrics: ConsciousnessMetrics) -> None:
        logger.info(
            "Signal strength: %.3f | Fractal dimension: %.2f | State: %s",
            metrics.signal_strength,
            metrics.fractal_dimension,
            metrics.state,
        )


# ---------------------------------------------------------------------------
# Integration pipeline
# ---------------------------------------------------------------------------
class UnifiedDemoPipeline:
    """Coordinates the processors and recognisers into a single pipeline."""

    def __init__(self) -> None:
        self.bio_processor = BioDigitalProcessor()
        self.fractal_recogniser = FractalPatternRecognizer()
        self.consciousness_recogniser = ConsciousnessRecognizer()
        self.monitor = RealTimeMonitor()

    def run_step(self) -> ConsciousnessMetrics:
        """Run a single pipeline step and return the metrics."""
        signal = self.bio_processor.capture_signal()
        processed = self.bio_processor.process_signal(signal)
        fractal_dim = self.fractal_recogniser.fractal_dimension(processed)
        state = self.consciousness_recogniser.recognise(processed, fractal_dim)
        metrics = ConsciousnessMetrics(
            signal_strength=statistics.mean(processed),
            fractal_dimension=fractal_dim,
            state=state,
        )
        self.monitor.display(metrics)
        return metrics


def run_demo_step() -> ConsciousnessMetrics:
    """Convenience function for the tests to execute one pipeline step."""
    pipeline = UnifiedDemoPipeline()
    return pipeline.run_step()


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
def main(iterations: int = 5, delay: float = 0.5) -> None:
    """Run a short real-time simulation printing metrics to the console."""
    pipeline = UnifiedDemoPipeline()
    for _ in range(iterations):
        try:
            pipeline.run_step()
        except Exception as exc:  # pragma: no cover - defensive programming
            logger.exception("Error during consciousness processing: %s", exc)
        time.sleep(delay)


if __name__ == "__main__":  # pragma: no cover - manual execution only
    main()
