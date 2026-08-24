# 🎓 SkyGuard UAV: Comprehensive Viva Defense & Technical Interview Guide

This guide provides authoritative, mathematically rigorous, and scientifically defensible answers for viva voce, technical presentations, and project examinations.

---

## 📑 Section Index
1. [Project Overview & System Architecture (Q1–Q10)](#1-project-overview--system-architecture-q1q10)
2. [Machine Learning & Model Evaluation (Q11–Q20)](#2-machine-learning--model-evaluation-q11q20)
3. [Edge AI, ONNX Runtime & Acceleration (Q21–Q27)](#3-edge-ai-onnx-runtime--acceleration-q21q27)
4. [EVM Blockchain & Smart Contracts (Q28–Q36)](#4-evm-blockchain--smart-contracts-q28q36)
5. [IPFS Decentralized Evidence Storage (Q37–Q41)](#5-ipfs-decentralized-evidence-storage-q37q41)
6. [Off-Chain Emergency Notifications (Q42–Q45)](#6-off-chain-emergency-notifications-q42q45)
7. [Offline Resilience & SQLite Retry Queue (Q46–Q50)](#7-offline-resilience--sqlite-retry-queue-q46q50)
8. [Software Architecture, API & Security (Q51–Q55)](#8-software-architecture-api--security-q51q55)

---

## 1. Project Overview & System Architecture (Q1–Q10)

### Q1: What core problem does SkyGuard UAV solve?
* **Short Answer**: It resolves the trade-off between massive cellular bandwidth costs from continuous UAV video streaming and the need for tamper-proof, auditable traffic incident reporting.
* **Technical Detail**: Instead of streaming continuous 4K video feeds to the cloud, edge AI processes the feed locally onboard the UAV. Only when an incident is detected is cryptographic evidence packaged, pinned to IPFS, and permanently recorded on the blockchain.
* **Evidence**: [`edge/edge_pipeline.py`](file:///c:/Users/M%20Sainandhan/OneDrive/Documents/Projects/trafficMonitoring/edge/edge_pipeline.py), [`blockchain/contracts/TrafficIncidentRegistry.sol`](file:///c:/Users/M%20Sainandhan/OneDrive/Documents/Projects/trafficMonitoring/blockchain/contracts/TrafficIncidentRegistry.sol).
* **Safe Claim**: An edge-to-blockchain prototype demonstrating automated incident filtering and decentralized proof of capture.

### Q2: Why use Unmanned Aerial Vehicles (UAVs) instead of fixed roadside CCTV?
* **Short Answer**: UAVs eliminate road occlusion blind spots, offer dynamic vantage points, and can be rapidly dispatched to temporary congestion hotspots or highway construction zones.
* **Technical Detail**: Fixed CCTV cameras have stationary perspective constraints and high cabling/installation costs. UAVs offer nadir (top-down) and oblique visual coverage of entire multi-lane intersections.

### Q3: Why is Edge AI essential for UAV operations?
* **Short Answer**: Real-time incident detection cannot tolerate high transmission latencies or intermittent 4G/5G connection drops in flight.
* **Technical Detail**: Cellular uplink from drones is bandwidth-limited and power-intensive. Running inference onboard enables sub-50ms incident detection and zero raw video egress.

### Q4: Why choose YOLOv8 as the detection backbone?
* **Short Answer**: YOLOv8 provides state-of-the-art single-stage detection efficiency with an anchor-free design and C2f feature extraction modules optimized for real-time edge hardware.
* **Technical Detail**: Anchor-free detection reduces hyperparameter tuning and post-processing latency (NMS), achieving high recall on small, densely packed aerial objects.

### Q5: Why YOLOv8n (Nano) specifically?
* **Short Answer**: YOLOv8n is the smallest model in the YOLOv8 family (3.0M parameters, 8.1 GFLOPs), fitting comfortably into low-power edge memory budgets.
* **Technical Detail**: On CPU/embedded edge devices, larger models (YOLOv8x/8l) drop below 5 FPS, whereas YOLOv8n reaches 25+ FPS with ONNX Runtime.

### Q6: Why use ByteTrack for multi-object tracking?
* **Short Answer**: ByteTrack associates both high-confidence and low-confidence detection boxes using Kalman filters and IoU matching, maintaining persistent IDs even through occlusion.
* **Technical Detail**: Standard trackers discard low-score detections, causing track fragmentation when vehicles pass under trees or overpasses. ByteTrack retains them in a second-stage association pass.

### Q7: What constitutes an "incident" in the system?
* **Short Answer**: An anomalous kinematic or spatial event exceeding predefined roadway safety thresholds.
* **Technical Detail**: Specifically: Speeding ($v > \text{limit}$), Sudden Braking ($\Delta v > 25\text{ km/h/s}$), Collisions (spatial IoU overlap + post-impact halt), and Congestion (high vehicle-to-lane area ratio).

### Q8: How is vehicle speed estimated from 2D aerial video?
* **Short Answer**: Centroid Euclidean displacement over consecutive frames divided by the time delta, scaled by a pixels-per-meter (PPM) factor.
* **Technical Detail**: $v = \left(\frac{\sqrt{\Delta x^2 + \Delta y^2}}{\text{PPM} \cdot \Delta t}\right) \times 3.6\text{ km/h}$.
* **Safe Claim**: Screen-space velocity estimation; requires camera calibration parameters for real-world physical accuracy.

### Q9: How is a vehicular collision detected?
* **Short Answer**: Pairwise bounding box IoU overlap between two tracked vehicles combined with an immediate drop in post-contact velocity.
* **Technical Detail**: High-speed parallel vehicles have low IoU and non-zero velocity. A collision triggers when $\text{IoU}(B_1, B_2) > 0.15$ and $\min(v_1, v_2) \approx 0$.

### Q10: How is traffic congestion density calculated?
* **Short Answer**: Ratio of total vehicle bounding box pixel area to designated roadway region-of-interest (ROI) area: $D = \frac{\sum \text{Area}_{\text{vehicles}}}{\text{Area}_{\text{roadway}}}$.

---

## 2. Machine Learning & Model Evaluation (Q11–Q20)

### Q11: What is Precision?
* **Definition**: $\text{Precision} = \frac{TP}{TP + FP}$. Of all positive vehicle detections predicted by the model, how many were actually vehicles.

### Q12: What is Recall?
* **Definition**: $\text{Recall} = \frac{TP}{TP + FN}$. Of all true ground-truth vehicles present in the aerial image, how many did the model successfully find.

### Q13: What is mAP@50?
* **Definition**: Mean Average Precision calculated at a fixed Intersection-over-Union (IoU) threshold of 0.50 across all target classes. It represents the area under the Precision-Recall curve.

### Q14: What is mAP@50-95?
* **Definition**: The primary COCO benchmark metric: the average of mAP values evaluated across 10 IoU thresholds from 0.50 to 0.95 in steps of 0.05.

### Q15: What is Intersection over Union (IoU)?
* **Definition**: $\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$ between the predicted bounding box and the ground-truth annotation.

### Q16: Why is mAP more scientifically rigorous than simple classification "Accuracy"?
* **Short Answer**: Object detection involves both classification and spatial localization. Accuracy does not penalize misplaced bounding boxes, false positives, or class imbalance.

### Q17: Why is the model's test mAP@50 22.96% instead of 90%+?
* **Short Answer**: The model was trained for only 5 epochs on CPU on a small subset of VisDrone aerial imagery as a proof-of-concept.
* **Technical Detail**: Aerial drone datasets feature tiny objects (10–30px), extreme scale variation, and nadir angles. 5 epochs is sufficient to demonstrate pipeline functionality, but full convergence requires 100–300 epochs on GPU.

### Q18: Why did you train for only 5 epochs in this development phase?
* **Short Answer**: CPU training constraint. 5 epochs took ~3.5 hours on CPU while proving that the loss curves converged and mAP improved from 0.006% (1-epoch baseline) to 22.96%.

### Q19: What concrete steps would improve model accuracy to production grade?
* **Short Answer**: 
  1. Train for 150+ epochs on NVIDIA GPUs with higher resolution input (`imgsz=1280`).
  2. Combine all three supported datasets (VisDrone + UAVDT + UA-DETRAC).
  3. Integrate small-object feature augmentations (SAHI: Slicing Aided Hyper Inference).

### Q20: What are the main dataset limitations?
* **Short Answer**: VisDrone has class imbalance (more vehicles and pedestrians than cyclists or traffic signals) and variable drone altitudes that challenge fixed-size convolution kernels.

---

## 3. Edge AI, ONNX Runtime & Acceleration (Q21–Q27)

### Q21: What does "Edge AI" mean in this system?
* **Short Answer**: Running deep learning neural network inference directly on the drone's onboard compute module rather than streaming video to a remote server.

### Q22: Why export the PyTorch model to ONNX?
* **Short Answer**: Open Neural Network Exchange (ONNX) decouples the model graph from the Python interpreter, allowing optimized C++ execution engines to run graph-level optimizations.

### Q23: What is TensorRT and why is it currently marked BLOCKED?
* **Short Answer**: NVIDIA TensorRT is a proprietary GPU inference optimizer. It is marked `BLOCKED` because the host development environment is CPU-only and lacks an NVIDIA GPU/CUDA toolkit.

### Q24: Why is ONNX Runtime 1.67x faster than PyTorch CPU on the same host?
* **Short Answer**: ONNX Runtime applies layer fusion (e.g., Conv + BatchNorm + SiLU into single kernel calls), constant folding, memory arena reuse, and optimized CPU vector instructions (AVX-512).

### Q25: Did you test on physical NVIDIA Jetson Nano hardware?
* **Honest Answer**: No. Jetson Nano is the target production hardware architecture, but benchmarks in this report were measured on the host CPU.

### Q26: What prevents you from claiming measured Jetson FPS?
* **Honest Answer**: Scientific truthfulness. Claiming Jetson performance without physical hardware connection is academic fabrication. We report host CPU numbers honestly.

### Q27: What is the primary compute bottleneck in the edge pipeline?
* **Short Answer**: YOLOv8 neural network forward pass accounts for **~99.5%** of latency (~39.4 ms), while ByteTrack and heuristics account for **<0.5%** (~0.25 ms).

---

## 4. EVM Blockchain & Smart Contracts (Q28–Q36)

### Q28: Why use blockchain in a traffic monitoring system?
* **Short Answer**: To create an immutable, decentralized, and tamper-proof legal chain of custody for traffic incident evidence that cannot be altered by rogue operators or compromised drones.

### Q29: What is stored on-chain vs off-chain?
* **On-Chain**: IPFS Hash (CID), Drone ID, Incident Type, Severity Level, GPS Coordinates, Timestamp, Resolution Status.
* **Off-Chain**: High-resolution video frames, raw telemetry, bounding box arrays (stored in IPFS).

### Q30: Why not store high-resolution video frames directly on the Ethereum blockchain?
* **Short Answer**: Ethereum storage costs (gas) are prohibitively high (~20,000 gas per 32 bytes). Storing a 1 MB image would cost thousands of dollars in transaction fees.

### Q31: What is a Smart Contract?
* **Short Answer**: A self-executing program deployed to an EVM blockchain that enforces immutable business logic and state transitions without intermediaries.

### Q32: What does the `onlyOwner` modifier do?
* **Short Answer**: Restricts administrative functions (e.g., registering new drones or resolving incidents) strictly to the contract owner wallet address.

### Q33: What does the `onlyActiveDrone` modifier do?
* **Short Answer**: Rejects incident submissions from unauthorized or deactivated drone wallet addresses, preventing spoofed data injection.

### Q34: What is the lifecycle of an on-chain incident?
* **Short Answer**: `REPORTED (0)` $\to$ `ESCALATED (1)` (if critical severity) $\to$ `RESOLVED (2)` (upon operator clearance).

### Q35: What happens when the blockchain RPC network is unavailable?
* **Short Answer**: The edge node buffers the transaction in the offline SQLite retry queue and continues video monitoring without crashing.

### Q36: Did you test on a live Ethereum public testnet (Sepolia/Mainnet)?
* **Honest Answer**: No. The blockchain runs in an in-memory simulated EVM container. Web3 transactions execute with zero gas costs in development.

---

## 5. IPFS Decentralized Evidence Storage (Q37–Q41)

### Q37: Why use IPFS (InterPlanetary File System)?
* **Short Answer**: IPFS uses content addressing where files are identified by the cryptographic hash of their content (CID), ensuring mathematical tamper-evidence.

### Q38: What is a CID (Content Identifier)?
* **Short Answer**: A unique SHA-256 / multihash digest of the uploaded data. If even one pixel in an evidence image is altered, the CID changes completely.

### Q39: Why is IPFS superior to traditional AWS S3 cloud storage for legal evidence?
* **Short Answer**: AWS S3 uses location addressing (URLs) where file contents can be overwritten at the same URL. IPFS guarantees immutability through cryptographic hashing.

### Q40: Did you test on the public distributed IPFS network?
* **Honest Answer**: No. The verified implementation uses a local deterministic mock store located at `data/mock_ipfs/` to guarantee reproducible offline test execution.

### Q41: What happens if IPFS fails during incident capture?
* **Short Answer**: The raw image and metadata are stored in the local SQLite retry queue; the replay engine uploads and pins them once the store is available.

---

## 6. Off-Chain Emergency Notifications (Q42–Q45)

### Q42: How does the emergency notification mechanism work?
* **Short Answer**: When a high-severity incident is recorded, `EmergencyNotificationService.sol` emits an `EmergencyAlertDispatched` event. An off-chain listener catches this event and dispatches alerts.

### Q43: Does the system actually call 911 or dispatch police vehicles?
* **Honest Answer**: No. The system dispatches to an in-memory mock log and generic webhook adapter. Real municipal dispatch would require authorized CAD (Computer Aided Dispatch) API access.

### Q44: What is the role of `src/notifications/event_listener.py`?
* **Short Answer**: It polls or subscribes to smart contract event logs, parses the `EmergencyAlertDispatched` parameters, and routes them to the notification service.

### Q45: How are duplicate notifications prevented?
* **Short Answer**: The notification service maintains an in-memory cooldown cache keyed by `(incident_id, severity)` to suppress duplicate alerts within a 60-second window.

---

## 7. Offline Resilience & SQLite Retry Queue (Q46–Q50)

### Q46: What happens if network connectivity drops entirely during flight?
* **Short Answer**: The offline-first edge pipeline queues incident payloads into `edge/offline_queue.db` (SQLite) and continues real-time object detection uninterrupted.

### Q47: How does SQLite provide crash resilience?
* **Short Answer**: SQLite Write-Ahead Logging (WAL) mode guarantees ACID transactions. If the UAV abruptly reboots, pending items remain in the database upon startup.

### Q48: What is Exponential Backoff?
* **Short Answer**: A retry strategy where delay increases exponentially between attempts ($\text{delay} = 2^{\text{retries}} \times 1.5\text{s}$) to avoid flooding recovering network endpoints.

### Q49: What is a Dead-Letter state?
* **Short Answer**: A permanent error archive state where incidents that have exceeded maximum retry attempts (e.g. 5) are preserved with error diagnostics rather than silently dropped.

### Q50: How do you prevent duplicate incidents in the queue?
* **Short Answer**: The queue enforces a `UNIQUE` index on `incident_id` and performs idempotent checks before insertion.

---

## 8. Software Architecture, API & Security (Q51–Q55)

### Q51: Why does the system provide both Streamlit and FastAPI?
* **Short Answer**: Architecture Option A: Streamlit serves human operators through an interactive GUI, while FastAPI provides a REST API for automated UAV fleets and external municipal services. Both consume shared services directly.

### Q52: Where does core business logic live?
* **Short Answer**: In authoritative shared backend modules (`blockchain/contract_client.py`, `ipfs/ipfs_client.py`, `edge/retry_queue.py`, `src/notifications/notification_service.py`), avoiding duplication in UI or API layers.

### Q53: How does FastAPI authenticate users and drones?
* **Short Answer**: Stateless RFC 7519 JWT (JSON Web Token) bearer tokens with role-based access control (`ADMIN`, `OPERATOR`) and HMAC-SHA256 telemetry message signatures.

### Q54: How is logging centralized across the system?
* **Short Answer**: Single authoritative logging engine in `config/logging_config.py` with colored console output, rotating structured JSON files (10MB cap), and idempotency guards.

### Q55: How are sensitive secrets protected in logs and evidence?
* **Short Answer**: Automated `SecretRedactionFilter` intercepts log records and redacts private keys (`0x...`), JWT tokens (`eyJ...`), and passwords with `[REDACTED_SECRET]`.
