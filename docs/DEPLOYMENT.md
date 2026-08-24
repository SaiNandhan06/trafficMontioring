# SkyGuard UAV: Deployment & Operations Guide

This guide details deployment options for the **SkyGuard UAV** system across Docker containerized environments, physical edge hardware (NVIDIA Jetson), local/testnet EVM blockchains, and decentralized IPFS storage.

---

## 1. Docker Compose Deployment

The repository includes a multi-container Docker Compose stack deploying the IPFS daemon, local Ganache EVM blockchain, Streamlit/FastAPI verification dashboard, and Edge AI inference engine.

### Prerequisites
* **Docker Engine:** Version `24.0+`
* **Docker Compose:** Version `v2.20+`

### Deploying the Multi-Container Stack

1. **Configure Environment:**
   Ensure `.env` exists with appropriate network settings:
   ```bash
   cp .env.example .env
   ```

2. **Build and Launch Services:**
   ```bash
   docker-compose up -d --build
   ```

3. **Verify Container Health:**
   ```bash
   docker-compose ps
   ```

### Service Map
| Container Name | Service | Ports | Description |
| :--- | :--- | :--- | :--- |
| `uav_ipfs_node` | IPFS Kubo | `4001`, `5001`, `8080` | Local content-addressed decentralized storage daemon |
| `uav_ganache_node` | Ganache EVM | `8545` | Local deterministic EVM testnet (Chain ID 1337) |
| `uav_dashboard` | Dashboard & API | `8501`, `8000` | Streamlit user interface and FastAPI REST backend |
| `uav_edge_node` | Edge AI Engine | — | YOLOv8 + ByteTrack video processing pipeline |

4. **Shutdown Stack:**
   ```bash
   docker-compose down
   ```

---

## 2. Physical Edge Hardware Deployment (NVIDIA Jetson)

When deploying on an embedded UAV edge companion computer (e.g., NVIDIA Jetson Nano, Jetson Orin Nano, or Jetson Xavier NX):

### Recommended Configuration
* **OS:** NVIDIA JetPack `5.1.2` / Ubuntu `20.04 LTS` or `22.04 LTS`
* **Inference Runtime:** ONNX Runtime (`onnxruntime-gpu`) or TensorRT
* **Video Input:** Hardware-accelerated GStreamer pipeline or RTSP camera stream

### Edge Startup Command
```bash
python edge/edge_pipeline.py \
  --source "rtsp://192.168.1.100:554/stream" \
  --weights "model/weights/yolov8_uav_best.onnx" \
  --device "cuda:0"
```

---

## 3. Blockchain Deployment & Smart Contracts

### Compiling Smart Contracts
```bash
python blockchain/compile.py
```
Compiled JSON artifacts (ABI and bytecode) are written to `blockchain/build/`.

### Deploying to Local EVM Network
1. Ensure Ganache, Anvil, or Hardhat is running on `http://127.0.0.1:8545`.
2. Execute the deployment script:
   ```bash
   python blockchain/deploy.py
   ```
3. Update `CONTRACT_REGISTRY_ADDRESS` and `CONTRACT_EMERGENCY_ADDRESS` in `.env`.

---

## 4. Decentralized IPFS Storage Configuration

### Mode 1: Local Development Mock (Default)
Operates with zero external dependencies using content-addressed local hashing:
```ini
IPFS_MODE=mock
```

### Mode 2: Local Kubo IPFS Daemon
Connects to an active IPFS Kubo daemon running on port 5001:
```ini
IPFS_MODE=local
IPFS_HOST=127.0.0.1
IPFS_PORT=5001
IPFS_GATEWAY=http://127.0.0.1:8080/ipfs/
```

### Mode 3: Pinata Cloud Pinning Service
Uploads immutable incident packages directly to Pinata IPFS gateway:
```ini
IPFS_MODE=pinata
PINATA_JWT=your_pinata_jwt_token_here
```

---

## 5. Emergency Notification Webhook Setup

To bridge smart contract emergency alerts to external incident response services:
```ini
NOTIFICATION_MODE=webhook
NOTIFICATION_WEBHOOK_URL=https://api.emergency-dispatch.example.com/v1/alerts
NOTIFICATION_MAX_RETRIES=3
```
*Note: All alerts dispatched are deduplicated on-chain and recorded to `results/notification_audit.json`.*

---

## 6. Production Security Hardening Checklist

- [ ] Set `ENVIRONMENT=production` in `.env`.
- [ ] Configure a strong `SECRET_KEY` (minimum 32 characters, non-default).
- [ ] Use genuine encrypted keystore (`security/key_vault.py`) for private keys instead of plain-text `.env` keys.
- [ ] Enable HTTPS/TLS using signed SSL certificates on FastAPI (`dashboard/api.py`).
- [ ] Verify that `edge/offline_queue.db` has write permissions on the persistent edge filesystem.
