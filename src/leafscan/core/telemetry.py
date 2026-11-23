"""
src/leafscan/core/telemetry.py
Lightweight Prometheus-compatible metrics and OpenTelemetry trace helpers.
"""

import threading
from collections import defaultdict


class LeafScanMetrics:
    """
    In-memory Prometheus-compatible metrics collector.
    Avoids external dependencies while providing standard prometheus exposition.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.request_count = defaultdict(int)  # (endpoint, status) -> count
        self.error_count = defaultdict(int)  # (error_code) -> count
        self.safety_status_count = defaultdict(int)  # (status) -> count
        self.active_requests = 0
        self.inference_latencies = []
        self.model_load_time_ms = 0.0
        self.bundle_version = "unknown"

    def record_request(self, endpoint: str, status_code: int):
        with self._lock:
            self.request_count[(endpoint, str(status_code))] += 1

    def record_error(self, error_code: str):
        with self._lock:
            self.error_count[error_code] += 1

    def record_safety_status(self, safety_status: str):
        with self._lock:
            self.safety_status_count[safety_status] += 1

    def record_latency(self, latency_ms: float):
        with self._lock:
            self.inference_latencies.append(latency_ms)
            if len(self.inference_latencies) > 1000:
                self.inference_latencies.pop(0)

    def generate_prometheus_output(self) -> str:
        with self._lock:
            lines = [
                "# HELP leafscan_requests_total Total HTTP requests handled",
                "# TYPE leafscan_requests_total counter",
            ]
            for (ep, status), cnt in self.request_count.items():
                lines.append(f'leafscan_requests_total{{endpoint="{ep}",status="{status}"}} {cnt}')

            lines.extend(["# HELP leafscan_errors_total Total errors by code", "# TYPE leafscan_errors_total counter"])
            for code, cnt in self.error_count.items():
                lines.append(f'leafscan_errors_total{{code="{code}"}} {cnt}')

            lines.extend(
                [
                    "# HELP leafscan_safety_status_total Distribution of safety status outcomes",
                    "# TYPE leafscan_safety_status_total counter",
                ]
            )
            for status, cnt in self.safety_status_count.items():
                lines.append(f'leafscan_safety_status_total{{status="{status}"}} {cnt}')

            lines.extend(
                [
                    "# HELP leafscan_active_requests Currently in-flight requests",
                    "# TYPE leafscan_active_requests gauge",
                    f"leafscan_active_requests {self.active_requests}",
                ]
            )

            lines.extend(
                [
                    "# HELP leafscan_model_load_duration_ms Model initialization duration in milliseconds",
                    "# TYPE leafscan_model_load_duration_ms gauge",
                    f'leafscan_model_load_duration_ms{{bundle="{self.bundle_version}"}} {self.model_load_time_ms:.2f}',
                ]
            )

            if self.inference_latencies:
                lats = sorted(self.inference_latencies)
                p50 = lats[len(lats) // 2]
                p95 = lats[int(len(lats) * 0.95)]
                p99 = lats[int(len(lats) * 0.99)]
                lines.extend(
                    [
                        "# HELP leafscan_inference_latency_ms Inference latency summary in milliseconds",
                        "# TYPE leafscan_inference_latency_ms summary",
                        f'leafscan_inference_latency_ms{{quantile="0.5"}} {p50:.2f}',
                        f'leafscan_inference_latency_ms{{quantile="0.95"}} {p95:.2f}',
                        f'leafscan_inference_latency_ms{{quantile="0.99"}} {p99:.2f}',
                        f"leafscan_inference_latency_ms_sum {sum(lats):.2f}",
                        f"leafscan_inference_latency_ms_count {len(lats)}",
                    ]
                )

            return "\n".join(lines) + "\n"


metrics = LeafScanMetrics()
