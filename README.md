# 🚁 SkyGuard UAV: Edge AI + Blockchain Traffic Monitoring System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00dc82.svg)](https://github.com/ultralytics/ultralytics)
[![Solidity](https://img.shields.io/badge/Solidity-0.8.20-363636.svg)](https://soliditylang.org/)
[![IPFS](https://img.shields.io/badge/IPFS-Decentralized%20Storage-65c2cb.svg)](https://ipfs.tech/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42.0-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 92/92 Passing](https://img.shields.io/badge/Tests-92%2F92%20Passing-brightgreen.svg)](tests/)

**SkyGuard UAV** is an intelligent aerial traffic monitoring research and demonstration platform. It combines **Edge AI computer vision (YOLOv8 + ByteTrack)** on UAVs with **EVM Smart Contracts (Solidity)** and **IPFS content addressing** for real-time traffic incident detection, immutable evidentiary audit trails, offline-first fault tolerance, and interactive operator command center dashboards.

---

## 📑 Table of Contents
1. [System Architecture](#1-system-architecture)
2. [Technology Stack & Verified Environment](#2-technology-stack--verified-environment)
3. [Quick Start & Reproduction Guide](#3-quick-start--reproduction-guide)
4. [Environment Variables Configuration](#4-environment-variables-configuration)
5. [Dataset Pipeline (KaggleHub)](#5-dataset-pipeline-kagglehub)
6. [Model Training & Evaluation](#6-model-training--evaluation)
7. [Edge AI, ONNX Runtime & Kinematics](#7-edge-ai-onnx-runtime--kinematics)
8. [Blockchain Smart Contracts (EVM)](#8-blockchain-smart-contracts-evm)
9. [IPFS Evidence Packaging](#9-ipfs-evidence-packaging)
10. [Off-Chain Emergency Notifications](#10-off-chain-emergency-notifications)
11. [Offline-First Resilience & Retry Queue](#11-offline-first-resilience--retry-queue)
12. [FastAPI REST API & Streamlit Dashboard](#12-fastapi-rest-api--streamlit-dashboard)
13. [System Verification & Automated Testing](#13-system-verification--automated-testing)
14. [Verified Performance Results](#14-verified-performance-results)
15. [Known Limitations & Viva-Safe Claims](#15-known-limitations--viva-safe-claims)
16. [Comprehensive Documentation Index](#16-comprehensive-documentation-index)
17. [Troubleshooting Guide](#17-troubleshooting-guide)
18. [License](#18-license)

---

## 1. System Architecture

```text
               ┌─────────────────────────────────────────────────────────┐
               │                  UAV Camera Video Feed                  │
               └────────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
               ┌─────────────────────────────────────────────────────────┐
               │                Edge AI Inference Engine                 │
               │                                                         │
               │  1. YOLOv8 Detection (Vehicle, Pedestrian, Cyclist)     │
               │  2. ByteTrack Multi-Object Tracking (Persistent IDs)    │
               │  3. Kinematics Estimator (Speed & Deceleration vectors) │
               │  4. Heuristic Anomaly Detector (Collisions/Congestion)  │
               └────────────────────────────┬────────────────────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     │ Incident Detected (Bounding Box + Frame)    │
                     ▼                                             ▼
       ┌───────────────────────────┐                 ┌───────────────────────────┐
       │   IPFS Evidence Storage   │                 │   Offline SQLite Buffer   │
       │                           │                 │                           │
       │ - Encodes JPEG evidence   │                 │ - WAL Mode Persistence    │
       │ - Bundles structured JSON │                 │ - Exponential Backoff     │
       │ - Yields Content CID      │                 │ - Dead-Letter Archiving   │
       └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                     │                                             │
                     └──────────────────────┬──────────────────────┘
                                            │
                                            ▼
               ┌─────────────────────────────────────────────────────────┐
               │                  EVM Blockchain Layer                   │
               │                                                         │
               │  • TrafficIncidentRegistry.sol (onlyActiveDrone)        │
               │  • EmergencyNotificationService.sol (Alert Event)       │
               │  • Lifecycle: REPORTED ──► ESCALATED ──► RESOLVED       │
               └────────────────────────────┬────────────────────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     │                                             │
                     ▼                                             ▼
       ┌───────────────────────────┐                 ┌───────────────────────────┐
       │   Streamlit Command GUI   │                 │     FastAPI REST API      │
       │                           │                 │                           │
       │ - Provenance badges       │                 │ - Machine REST endpoints  │
       │ - Geospatial live map     │                 │ - JWT Bearer auth guards  │
       │ - Model comparison table  │                 │ - Health check scorecard  │
       └───────────────────────────┘                 └───────────────────────────┘
```

For complete architectural specifications, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 2. Technology Stack & Verified Environment

| Component | Technology | Verified Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Language** | Python | `3.12.x` | Core application and testing runtime |
| **Object Detection** | Ultralytics YOLOv8 | `8.4.126` | Aerial object detection (`model/weights/yolov8_uav_best.pt`) |
| **Deep Learning** | PyTorch | `2.13.0+cpu` | Tensor computation and PyTorch model execution |
| **Optimized Edge** | ONNX Runtime | `1.20.1` | High-throughput CPU inference engine |
| **Datasets** | KaggleHub | `1.0.2` | Automated dataset acquisition & registry |
| **Smart Contracts** | Solidity | `^0.8.20` | EVM incident registry and access control |
| **Storage** | IPFS (Mock / Kubo) | `mock` (tested) | Tamper-proof content-addressed evidence storage |
| **Operator GUI** | Streamlit | `1.42.0` | Human operator dashboard with source attribution |
| **REST Backend** | FastAPI / Uvicorn | `0.109.0` | Machine-to-machine API & JWT authentication |
| **Testing** | Pytest | `9.1.1` | Complete regression test suite (92 passed tests) |

---

## 3. Quick Start & Reproduction Guide

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/SaiNandhan06/trafficMontioring.git
cd trafficMonitoring

# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell:
.venv\Scripts\activate

# Activate on Linux/macOS:
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Configure Environment
```bash
cp .env.example .env
```

### Step 4: Run Master Verification Suite
```bash
python scripts/verify_all.py
```
*Executes all 10 component checks (environment, dependencies, model, Kaggle download, inference, contracts, API, security, pytest).*

---

## 4. Environment Variables Configuration

Copy `.env.example` to `.env`. The default values work out-of-the-box in development simulation mode:

```env
# Application Settings
ENVIRONMENT=development
LOG_LEVEL=INFO

# Drone Identifier & Coordinates
DRONE_ID=UAV-ALPHA-01
DRONE_LAT=37.7749
DRONE_LNG=-122.4194

# Deep Learning Model
MODEL_WEIGHTS_PATH=model/weights/yolov8_uav_best.pt
DEVICE=cpu
CONF_THRESHOLD=0.35
IOU_THRESHOLD=0.45

# Blockchain (Simulated in development)
ETH_RPC_URL=http://127.0.0.1:8545
CHAIN_ID=1337
OPERATOR_PRIVATE_KEY=<0x_YOUR_OPERATOR_PRIVATE_KEY_HEX_64_CHARS>

# IPFS Storage
IPFS_MODE=mock
PINATA_JWT=

# Emergency Notifications
NOTIFICATION_MODE=mock
NOTIFICATION_WEBHOOK_URL=
NOTIFICATION_MAX_RETRIES=3

# Security & JWT
SECRET_KEY=dev-secret-key-change-in-production-uav-2026!
```

---

## 5. Dataset Pipeline (KaggleHub)

The repository integrates 3 standardized UAV aerial traffic datasets via KaggleHub:

1. **VisDrone** (`kushagrapandya/visdrone-dataset`): Primary training benchmark.
2. **UAVDT** (`shakaibkaggle/uavdt-dataset`): UAV benchmark for vehicle tracking.
3. **UA-DETRAC** (`bratjay/ua-detrac-orig`): Roadway surveillance benchmark.

### Commands:
```bash
# Download and prepare VisDrone dataset
python data/prepare_visdrone.py

# Validate bounding boxes and check for data leakage
python data/validate_dataset.py
```

---

## 6. Model Training & Evaluation

The active model (`model/weights/yolov8_uav_best.pt`) was trained on VisDrone UAV imagery.

### Training Command:
```bash
python model/train.py --epochs 5 --imgsz 640 --batch 8 --device cpu
```

### Held-Out Test Evaluation:
```bash
python scripts/evaluate_model.py --split test
```

### Verified Test Metrics:
| Metric | 1-Epoch Baseline | 5-Epoch Trained Model | Relative Improvement |
| :--- | :--- | :--- | :--- |
| **Precision** | `0.00128` (0.13%) | **`0.349129` (34.91%)** | **+272.7x** |
| **Recall** | `0.01050` (1.05%) | **`0.255468` (25.55%)** | **+24.3x** |
| **mAP@50** | `0.0000587` (0.006%) | **`0.229574` (22.96%)** | **+3,910.9x** |
| **mAP@50-95** | `0.0000174` (0.002%) | **`0.093745` (9.37%)** | **+5,387.6x** |

---

## 7. Edge AI, ONNX Runtime & Kinematics

### Edge Engine Benchmark:
```bash
python scripts/benchmark_edge_models.py --iterations 50
```

### Measured Host CPU Latency:
* **PyTorch CPU**: `67.63 ms / frame` (**14.79 FPS**)
* **ONNX Runtime CPU**: `40.50 ms / frame` (**24.69 FPS**, **1.67x Speedup**)
* **TensorRT / Jetson**: `BLOCKED` (No NVIDIA CUDA GPU on current host)

### Kinematics & Incident Logic:
* **Speed Estimation**: Screen-space centroid displacement over time: $v = \frac{\Delta d}{\text{PPM} \cdot \Delta t} \times 3.6$ km/h.
* **Sudden Braking**: $\Delta v > 25$ km/h deceleration within 1 second.
* **Collisions**: Spatial IoU overlap combined with abrupt post-collision velocity drops.
* **Traffic Congestion**: Vehicle density per lane area.

---

## 8. Blockchain Smart Contracts (EVM)

* **Contracts**: [`blockchain/contracts/TrafficIncidentRegistry.sol`](blockchain/contracts/TrafficIncidentRegistry.sol) & [`blockchain/contracts/EmergencyNotificationService.sol`](blockchain/contracts/EmergencyNotificationService.sol).
* **Access Control**: Strict `onlyOwner` drone registration, `onlyActiveDrone` incident reporting.
* **Lifecycle**: `REPORTED (0)` $\to$ `ESCALATED (1)` $\to$ `RESOLVED (2)`.
* **Execution Mode**: `SIMULATED` in development.

```bash
# Benchmark smart contract transaction throughput
python scripts/benchmark_blockchain.py --txs 20
```

---

## 9. IPFS Evidence Packaging

* **Mock Local Store**: Located at `ipfs/mock_store/`. Provides bit-exact byte retrieval and deterministic SHA-256 content addressing.
* **Master Bundle**: Packages JPEG evidence frame, timestamp, GPS coordinates, and vehicle telemetry into a cryptographically tamper-evident JSON payload.

```bash
# Benchmark IPFS upload & retrieval
python scripts/benchmark_ipfs.py --samples 20
```

---

## 10. Off-Chain Emergency Notifications

* **Notification Service**: [`src/notifications/notification_service.py`](src/notifications/notification_service.py).
* **Event Listener**: [`src/notifications/event_listener.py`](src/notifications/event_listener.py).
* **Features**: In-memory deduplication cooldowns, bounded webhook retry backoff, and mock delivery logs in `results/notification_audit.json`.

```bash
# Validate emergency notification dispatches
python scripts/validate_notifications.py
```

---

## 11. Offline-First Resilience & Retry Queue

* **Persistent SQLite Queue**: [`edge/retry_queue.py`](edge/retry_queue.py) (`edge/offline_queue.db`).
* **Lifecycle States**: `PENDING` $\to$ `RETRYING` $\to$ `SUCCESS` (or `DEAD_LETTER` upon exhausting max retries).
* **Crash Resilience**: Replay engine automatically resyncs pending incidents to IPFS and Blockchain upon network restoration.

```bash
# Validate offline-first recovery and dead-letter archiving
python scripts/validate_resilience.py
```

---

## 12. FastAPI REST API & Streamlit Dashboard

### Launching FastAPI Backend:
```bash
python -m uvicorn dashboard.api:app --host 0.0.0.0 --port 8000
```
* **Health Check**: `GET http://localhost:8000/api/health`
* **Swagger Docs**: `http://localhost:8000/docs`

### Launching Streamlit Command Center:
```bash
python -m streamlit run dashboard/app.py
```
* **URL**: `http://localhost:8501`
* **Default Credentials**: `admin` / `Admin@UAV2026!`

---

## 13. System Verification & Automated Testing

### Run Complete Pytest Suite:
```bash
python -m pytest tests/ -v
```

### Test Suite Structure (92 Passed Tests):
* `tests/test_api_endpoints.py`: 6 tests (FastAPI routes, JWT auth guards, health check)
* `tests/test_blockchain_contracts.py`: 8 tests (Access control, lifecycle, simulation mode)
* `tests/test_contracts.py`: 2 tests (Contract compilation & incident reporting)
* `tests/test_dashboard_truthfulness.py`: 5 tests (Provenance attribution & fake metric prevention)
* `tests/test_dataset_pipeline.py`: 5 tests (VisDrone, UAVDT, UA-DETRAC converters & splits)
* `tests/test_edge_detection.py`: 3 tests (IoU, ByteTrack, collision detection)
* `tests/test_edge_export.py`: 4 tests (ONNX export & ONNX Runtime execution)
* `tests/test_evaluation_truthfulness.py`: 4 tests (Verified evaluation JSON schema)
* `tests/test_incident_kinematics.py`: 11 tests (Kinematics formulas, sudden braking, congestion)
* `tests/test_ipfs.py`: 2 tests (CID generation & incident package)
* `tests/test_ipfs_validation.py`: 8 tests (Bit-exact retrieval, JPEG compression, mutation resistance)
* `tests/test_logger_unification.py`: 5 tests (Idempotency, rotation, secret redaction)
* `tests/test_model_training.py`: 5 tests (KaggleHub registry & training checkpoints)
* `tests/test_notifications.py`: 7 tests (Mock delivery, deduplication, retry limits)
* `tests/test_reproducibility_documentation.py`: 5 tests (Docs structure, weights, configs, handles, env vars)
* `tests/test_retry_queue.py`: 1 test (Lifecycle queue test)
* `tests/test_retry_resilience.py`: 7 tests (Schema creation, restart persistence, replay engine)
* `tests/test_security_config.py`: 4 tests (Gitignore, secrets isolation, dataset portability)

---

## 14. Verified Performance Results

| Subsystem | Measurement | Verified Host Value | Provenance Status |
| :--- | :--- | :--- | :--- |
| **Object Detection** | mAP@50 (Held-Out Test) | **`0.229574` (22.96%)** | **`[MEASURED]`** |
| **Object Detection** | Precision / Recall | **`0.3491` / `0.2555`** | **`[MEASURED]`** |
| **Edge Compute** | PyTorch CPU FPS | **`14.79 FPS`** | **`[MEASURED]`** |
| **Edge Compute** | ONNX Runtime CPU FPS | **`24.69 FPS` (1.67x)** | **`[MEASURED]`** |
| **Edge Compute** | TensorRT / Jetson | Unavailable / Blocked | **`[BLOCKED]` / `[NOT MEASURED]`** |
| **Blockchain** | Settlement Latency | In-Memory Instantaneous | **`[SIMULATED]`** |
| **IPFS Storage** | Local Byte Retrieval | 100% Bit-Exact Match | **`[MOCK]`** |
| **Emergency Alerts**| Notification Dispatch | In-Memory Mock Delivery | **`[MOCK]`** |

---

## 15. Known Limitations & Viva-Safe Claims

### ✅ What SkyGuard UAV Demonstrates:
1. End-to-end aerial object detection pipeline using YOLOv8 trained on VisDrone imagery.
2. ByteTrack-style multi-object tracking and kinematic anomaly detection (speed, sudden braking, collisions).
3. 1.67x edge acceleration using ONNX Runtime on CPU.
4. Smart contract architecture in Solidity with access control and state lifecycle transitions.
5. Content-addressed tamper-evident IPFS evidence packaging.
6. Offline-first SQLite WAL queue with exponential backoff and replay engine.
7. Clean Option A architecture separating Human UI (Streamlit) from External Integrations (FastAPI).
8. Unified thread-safe logging engine with automatic private key / secret redaction.

### ⚠️ What is NOT Currently Demonstrated:
1. **Physical Jetson Hardware**: Benchmarked on host Windows CPU, not physical Jetson Orin/Nano.
2. **Live Ethereum Mainnet**: Blockchain executes in high-fidelity simulated in-memory EVM mode.
3. **Global IPFS Network**: Utilizes local deterministic mock store for reproducible offline testing.
4. **Real 911 / Emergency Services**: Dispatches to mock logging and generic webhooks.
5. **Real-World Metric Speed**: Speed is screen-space estimated unless calibrated with ground camera parameters.

---

## 16. Comprehensive Documentation Index

* 📜 **[CHANGELOG_BY_PHASE.md](docs/CHANGELOG_BY_PHASE.md)**: Full narrative and technical history of all 16 development phases.
* 🏛️ **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Detailed 4-tier edge-to-blockchain system architecture and sequence diagrams.
* 💻 **[DEVELOPMENT.md](docs/DEVELOPMENT.md)**: Developer guide, testing procedures, and contribution guidelines.
* 🚀 **[DEPLOYMENT.md](docs/DEPLOYMENT.md)**: Deployment instructions for Docker, Jetson Edge, EVM, and IPFS.
* 🎓 **[VIVA_GUIDE.md](docs/VIVA_GUIDE.md)**: Academic defense notes, questions, and viva preparation guide.

---

## 17. Troubleshooting Guide

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| `KaggleHub download fails` | Network timeout or missing cache | Verify internet access; dataset will cache in `~/.cache/kagglehub/` |
| `No module named 'web3'` | Optional EVM dependency missing | System automatically falls back to high-fidelity simulated state |
| `No module named 'solcx'` | Native compiler missing | System automatically utilizes embedded contract ABI artifacts |
| `TensorRT export blocked` | No NVIDIA GPU / CUDA toolkit | Use ONNX Runtime engine (`model/weights/yolov8_uav_best.onnx`) |
| `Streamlit port in use` | Another instance running on 8501 | Run `python -m streamlit run dashboard/app.py --server.port 8502` |

---

## 18. License

This project is open-source and available under the terms of the [MIT License](LICENSE).