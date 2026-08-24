"""
FastAPI REST API Performance Benchmark.
Measures latency and throughput across /api/health, /api/system/provenance,
/api/incidents, and /api/blockchain/stats endpoints.
"""

import sys
import time
import json
import statistics
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from dashboard.api import app
from config.settings import BASE_DIR


def benchmark_api_endpoints(iterations: int = 100) -> Dict[str, Any]:
    client = TestClient(app)
    endpoints = [
        ("GET", "/api/health"),
        ("GET", "/api/system/provenance"),
        ("GET", "/api/incidents"),
        ("GET", "/api/blockchain/stats"),
    ]

    benchmark_results = {}

    for method, path in endpoints:
        latencies = []
        status_codes = []

        # Warmup
        for _ in range(5):
            client.get(path)

        for _ in range(iterations):
            t0 = time.perf_counter()
            resp = client.get(path)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms
            status_codes.append(resp.status_code)

        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.50)]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        mean_lat = statistics.mean(latencies)

        benchmark_results[path] = {
            "method": method,
            "iterations": iterations,
            "success_rate_pct": (status_codes.count(200) / len(status_codes)) * 100.0,
            "mean_latency_ms": round(mean_lat, 3),
            "p50_latency_ms": round(p50, 3),
            "p95_latency_ms": round(p95, 3),
            "p99_latency_ms": round(p99, 3),
            "rps": round(1000.0 / mean_lat, 1) if mean_lat > 0 else 0
        }

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoints": benchmark_results,
        "summary": {
            "tested_endpoints": len(endpoints),
            "total_requests": len(endpoints) * iterations,
            "overall_status": "MEASURED (Local ASGI In-Memory TestClient)"
        }
    }

    out_file = BASE_DIR / "results" / "api_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 75)
    print(" [FASTAPI REST API LATENCY & PERFORMANCE BENCHMARK REPORT] ")
    print("=" * 75)
    for path, data in benchmark_results.items():
        print(f" • {path:<26} -> Mean: {data['mean_latency_ms']:>6.3f} ms | p95: {data['p95_latency_ms']:>6.3f} ms | {data['rps']:>6.1f} req/s")
    print("=" * 75)
    print(f"Report saved to: {out_file}\n")

    return report


if __name__ == "__main__":
    benchmark_api_endpoints()
