"""Thread-safe metrics collectors shared across the RAG News application."""

from __future__ import annotations

import threading


class AppMetrics:
    """Track cumulative timings for external services."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_api_time = 0.0
        self.congress_api_time = 0.0

    def add_llm_api_time(self, duration: float) -> None:
        with self._lock:
            self.llm_api_time += duration

    def add_congress_api_time(self, duration: float) -> None:
        with self._lock:
            self.congress_api_time += duration


metrics = AppMetrics()
file_write_lock = threading.Lock()
