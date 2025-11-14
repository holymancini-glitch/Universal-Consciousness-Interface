"""Simple HTTP monitor to expose pipeline metrics as JSON."""

from __future__ import annotations

import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any, Dict


class MetricsHTTPRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler that returns the latest metrics in JSON format."""

    metrics: Dict[str, Any] | None = None

    def do_GET(self) -> None:  # pragma: no cover - trivial network glue
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        payload = json.dumps(self.metrics or {})
        self.wfile.write(payload.encode())

    def log_message(self, format: str, *args: Any) -> None:  # pragma: no cover
        """Suppress default logging to keep test output clean."""
        return


class HTTPMonitor:
    """Monitor that serves metrics over HTTP.

    The monitor starts a simple HTTP server in a background thread. Each time
    ``display`` is called the metrics are stored and can be fetched via ``GET``
    requests.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._server = HTTPServer((host, port), MetricsHTTPRequestHandler)
        self.host, self.port = self._server.server_address
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def display(self, metrics: Any) -> None:
        MetricsHTTPRequestHandler.metrics = asdict(metrics)

    def shutdown(self) -> None:
        self._server.shutdown()
        self._thread.join()
