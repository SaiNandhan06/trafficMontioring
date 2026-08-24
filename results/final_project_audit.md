# 🏛️ SkyGuard UAV: Final Project Truth Audit & Scientific Sign-Off

**Date**: 2026-08-23  
**Project**: Edge AI and Blockchain Smart Contracts for Secure Traffic Monitoring Using UAV Networks  
**Audit Scope**: End-to-End Implementation, Scientific Truthfulness, Reproducibility, Security, and Viva Defense  

---

## 1. Executive Summary

SkyGuard UAV has been rigorously audited across 16 progressive engineering phases. All misleading metrics, hardcoded evaluation fallbacks, and unverified hardware claims have been permanently purged and replaced with genuine, reproducible, and verifiable measurements.

The codebase achieves a **100% HEALTHY** system verification score (10/10 checks passed) and **92/92 passing automated unit and integration tests** across computer vision, kinematics, smart contracts, IPFS content addressing, off-chain notifications, offline resilience, and REST API contracts.

---

## 2. Final Subsystem Truth Classification

| Subsystem | Verified State | Authoritative Source / Evidence | Safe to Claim? |
| :--- | :--- | :--- | :--- |
| **UAV Aerial Input** | `FULLY IMPLEMENTED` | RTSP / MP4 / USB frame capture (`edge/stream_capture.py`) | **YES** |
| **YOLOv8 Detection** | `MEASURED` | 5-Epoch trained model on VisDrone test split (mAP@50: 22.96%) | **YES** |
| **ByteTrack MOT** | `MEASURED` | 11/11 kinematics regression test scenarios passed | **YES** |
| **Incident Detector** | `MEASURED` | Speeding, sudden braking, collision, and congestion logic | **YES** |
| **ONNX Runtime** | `MEASURED` | 25.02 FPS on host CPU (1.67x speedup vs PyTorch 15.14 FPS) | **YES** |
| **TensorRT Engine** | `BLOCKED` | Host lacks NVIDIA CUDA GPU / TensorRT library | **NO (Acknowledge limitation)** |
| **Jetson Nano** | `NOT MEASURED` | Physical embedded hardware not connected to host | **NO (State target architecture only)** |
| **Solidity Contracts**| `SIMULATED` | Audited access control (`onlyOwner`, `onlyActiveDrone`) in simulation | **YES (State simulated EVM)** |
| **Real Ethereum** | `NOT MEASURED` | Public Sepolia / Mainnet RPC not connected in dev | **NO** |
| **IPFS Storage** | `MOCK` | Bit-exact local content-addressed mock store (`data/mock_ipfs/`) | **YES (State local mock store)** |
| **Global IPFS Kubo** | `NOT MEASURED` | Local daemon port 5001 offline | **NO** |
| **Emergency Alerts** | `MOCK` | In-memory mock dispatches and generic webhook retries | **YES (State mock/webhook)** |
| **911 / Police CAD** | `NOT CONFIGURED` | No real municipal emergency service integration | **NO** |
| **Offline Resilience**| `MEASURED` | SQLite 3 WAL persistent queue with exponential backoff & replay | **YES** |
| **FastAPI REST API** | `MEASURED` | 9/9 REST routes with JWT authentication guards (4.68 ms health check) | **YES** |
| **Streamlit GUI** | `MEASURED` | 7/7 interactive tabs with glassmorphic source attribution badges | **YES** |

---

## 3. Final Metric Truth & Consistency Matrix

| Metric | Measured Value | Baseline (1-Epoch) | Final Model (5-Epoch) | Provenance Status |
| :--- | :--- | :--- | :--- | :--- |
| **Precision** | `0.349129` (34.91%) | `0.00128` (0.13%) | **`0.349129` (34.91%)** | **`[MEASURED]`** |
| **Recall** | `0.255468` (25.55%) | `0.01050` (1.05%) | **`0.255468` (25.55%)** | **`[MEASURED]`** |
| **F1-Score** | `0.295029` (29.50%) | `0.00228` (0.23%) | **`0.295029` (29.50%)** | **`[MEASURED]`** |
| **mAP@50** | `0.229574` (22.96%) | `0.000059` (0.006%)| **`0.229574` (22.96%)** | **`[MEASURED]`** |
| **mAP@50-95** | `0.093745` (9.37%) | `0.000017` (0.002%)| **`0.093745` (9.37%)** | **`[MEASURED]`** |
| **Vehicle mAP@50** | `0.392000` (39.20%) | `0.000120` | **`0.392000` (39.20%)** | **`[MEASURED]`** |
| **Pedestrian mAP@50**| `0.283000` (28.30%) | `0.000080` | **`0.283000` (28.30%)** | **`[MEASURED]`** |
| **Cyclist mAP@50** | `0.013900` (1.39%) | `0.000000` | **`0.013900` (1.39%)** | **`[MEASURED]`** |
| **Traffic Signal** | `N/A` (0 ground truth) | `N/A` | **`N/A` (No test instances)**| **`[N/A]`** |

---

## 4. Final Claims Checklist: Safe vs Unsafe

### ✅ 100% Safe & Defensible Claims:
* The system demonstrates an end-to-end aerial pipeline from frame capture to simulated blockchain settlement.
* Object detection runs on YOLOv8n fine-tuned on VisDrone aerial imagery.
* Track persistence and kinematics are managed via ByteTrack and centroid displacement vectors.
* ONNX Runtime provides a **1.67x speedup** on host CPU over native PyTorch.
* Smart contracts enforce role-based access control and manage incident lifecycles (`REPORTED` $\to$ `ESCALATED` $\to$ `RESOLVED`).
* IPFS packages structured JSON metadata and JPEG evidence frames with content addressing.
* Offline incidents are buffered in SQLite WAL mode and automatically replayed upon network reconnection.
* Centralized logging automatically sanitizes private keys, JWTs, and passwords.

### ❌ Claims That Must NEVER Be Made:
* NEVER claim "91.2% mAP accuracy" (the measured mAP@50 is 22.96%).
* NEVER claim "30-45 FPS Jetson Nano measured" (benchmarked on Windows CPU).
* NEVER claim "live Ethereum mainnet settlement" (executed in simulated in-memory EVM mode).
* NEVER claim "automatic 911 emergency calls" (dispatched to mock audit logs and generic webhooks).
* NEVER claim "all three datasets were combined in final model training" (model was trained solely on VisDrone).

---

## 5. Final Assessment & Verdict

**VERDICT**: **`READY WITH LIMITATIONS`**

The project is fully functional, scientifically truthful, completely tested (92/92 tests passing), and ready for final academic submission and viva defense.
