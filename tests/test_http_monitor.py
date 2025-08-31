"""Integration test for the HTTP monitor."""

import json
import os
import sys
import urllib.request

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from demo_unified_interface import UnifiedDemoPipeline
from http_monitor import HTTPMonitor


def test_http_monitor_serves_metrics():
    monitor = HTTPMonitor()
    pipeline = UnifiedDemoPipeline(monitor=monitor)
    metrics = pipeline.run_step()

    with urllib.request.urlopen(monitor.url) as response:
        data = json.load(response)

    monitor.shutdown()

    assert data["state"] == metrics.state
    assert "signal_strength" in data
    assert "fractal_dimension" in data
