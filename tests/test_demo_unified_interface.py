"""Tests for the simplified unified consciousness demo."""

import os
import sys
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from demo_unified_interface import (
    BioDigitalProcessor,
    FractalPatternRecognizer,
    ConsciousnessRecognizer,
    run_demo_step,
)


def test_bio_digital_processing():
    processor = BioDigitalProcessor()
    signal = processor.capture_signal(length=10)
    processed = processor.process_signal(signal)
    assert len(signal) == 10
    assert len(processed) == 10
    # processed values should be within [0,1]
    assert all(0.0 <= x <= 1.0 for x in processed)


def test_fractal_recognition_value_range():
    recogniser = FractalPatternRecognizer()
    signal = [0.1 * i for i in range(16)]
    dimension = recogniser.fractal_dimension(signal)
    assert 0.0 <= dimension <= 5.0


def test_pipeline_state_output():
    metrics = run_demo_step()
    assert metrics.state in ConsciousnessRecognizer.STATES
    assert 0.0 <= metrics.signal_strength <= 1.0


def test_cli_argument_parsing():
    script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo_unified_interface.py")
    result = subprocess.run(
        [sys.executable, script, "--iterations", "2", "--delay", "0"],
        capture_output=True,
        text=True,
        check=True,
    )
    # Logging is directed to stderr; count iteration lines
    lines = [line for line in result.stderr.splitlines() if "Signal strength" in line]
    assert len(lines) == 2
