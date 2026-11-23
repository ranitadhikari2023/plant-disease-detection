# LeafScan V3 — Production Deployment Guide

**Target Architecture:** LeafScan V3 (ONNX Runtime CPU Microservice)  
**Document Version:** 3.0.0  
**Status:** Production Standard  

---

## 1. Deployment Topology

LeafScan V3 is packaged as a lightweight, non-root, multi-stage Docker container optimized for CPU execution.

### Architectural Defaults:
- **Base Image:** `python:3.12-slim`
- **Execution User:** `leafscan:leafscan` (UID 10001)
- **Serving Runtime:** ONNX Runtime 1.20.0 (CPUExecutionProvider)
- **Web Server:** Uvicorn with 2 worker processes
- **Target Container Size:** < 400 MB
- **Dependencies Excluded:** TensorFlow, Keras, CUDA, cuDNN, development compilers

---

## 2. Docker Deployment

### 2.1 Building the Production Container
```bash
docker build -t leafscan:3.0.0 -f Dockerfile .
```

Verify that the built image is under 500 MB:
```bash
docker image inspect leafscan:3.0.0 --format='{{.Size}}' | awk '{print $1/1024/1024 " MB"}'
```

### 2.2 Running with Docker Compose
```bash
docker compose up -d
```

Verify service readiness via the readiness probe:
```bash
curl -f http://localhost:8000/ready
```

Expected response:
```json
{
  "status": "ready",
  "bundle_version": "v3.0.0",
  "engine": "OnnxEngine",
  "execution_provider": "CPUExecutionProvider",
  "models_verified": ["species_expert", "potato_expert", "tomato_expert"]
}
```

---

## 3. Environment Variable Configuration

All configuration is externalized via `pydantic-settings` with the `LEAFSCAN_` prefix:

| Environment Variable | Default | Purpose |
|---|---|---|
| `LEAFSCAN_APP_ENV` | `production` | Operational environment mode |
| `LEAFSCAN_HOST` | `0.0.0.0` | Bind interface address |
| `LEAFSCAN_PORT` | `8000` | Bind port number |
| `LEAFSCAN_MODEL_BUNDLE_PATH` | `/app/models/bundles/v3` | Path to active model bundle directory |
| `LEAFSCAN_VERIFY_BUNDLE_CHECKSUMS` | `true` | Enforces SHA-256 verification on startup |
| `LEAFSCAN_MAX_CONCURRENT_INFERENCES` | `4` | Max concurrent inference calls before 503 load shedding |
| `LEAFSCAN_RATE_LIMIT_PER_MINUTE` | `60` | Max requests per minute per IP |
| `LEAFSCAN_INTRA_OP_THREADS` | `2` | ONNX session threadpool allocation per worker |
| `LEAFSCAN_INTER_OP_THREADS` | `1` | ONNX inter-op thread count |
| `LEAFSCAN_LOG_JSON` | `true` | Structured JSON log output |
| `LEAFSCAN_LOG_LEVEL` | `INFO` | Logging threshold level |

---

## 4. Container Health & Readiness Checks

### Docker / Kubernetes Probes:
- **Liveness Probe:**
  ```http
  GET /health
  ```
  Returns 200 OK as long as the web process is running. Does not inspect models.
- **Readiness Probe:**
  ```http
  GET /ready
  ```
  Returns 200 OK only when:
  1. The model bundle manifest is cryptographically verified against all SHA-256 digests.
  2. All three ONNX sessions are loaded and optimized.
  3. Warmup inference has completed successfully.
- **Prometheus Metrics:**
  ```http
  GET /metrics
  ```
  Exposes request counts, latency distributions (p50, p95, p99), in-flight concurrency, and safety status distributions.
