# SkyGuard UAV: Developer Guide & Contribution Workflow

This guide details the setup, development workflow, testing procedures, and coding standards for contributing to the **SkyGuard UAV** codebase.

---

## 1. Local Environment Setup

### Prerequisites
* **Python:** Version `3.10`, `3.11`, or `3.12`
* **Git:** Version `2.30+`
* **Node / Ganache (Optional):** For live Ethereum EVM testing
* **IPFS Kubo (Optional):** For local IPFS daemon testing

### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/SaiNandhan06/trafficMontioring.git
   cd trafficMonitoring
   ```

2. **Create and Activate a Virtual Environment:**
   * **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   * **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Initialize Configuration:**
   ```bash
   cp .env.example .env
   ```
   *Note: In development mode, default settings in `.env.example` will operate out-of-the-box using built-in simulated blockchain and mock IPFS storage.*

---

## 2. Running Automated Tests

The test suite covers unit tests, integration tests, contract simulations, and documentation integrity across 18 test files.

### Run All Pytest Tests
```bash
python -m pytest tests/ -v
```
*Expected baseline: 92/92 tests passing.*

### Run System Verification Scorecard
```bash
python scripts/verify_all.py
```
*Expected output: 10/10 checks passing (100% HEALTHY).*

### Run Specific Test Suites
* **Edge Kinematics & Incident Detection:**
  ```bash
  python -m pytest tests/test_incident_kinematics.py tests/test_edge_detection.py -v
  ```
* **Blockchain Contracts & Lifecycle:**
  ```bash
  python -m pytest tests/test_blockchain_contracts.py tests/test_contracts.py -v
  ```
* **IPFS & Evidence Integrity:**
  ```bash
  python -m pytest tests/test_ipfs_validation.py -v
  ```
* **Offline Resilience & Retry Queue:**
  ```bash
  python -m pytest tests/test_retry_resilience.py -v
  ```

---

## 3. Running Benchmarks & Validation Scripts

| Component | Benchmark Script | Output Artifact |
| :--- | :--- | :--- |
| **Dataset Split & Leakage** | `python data/validate_dataset.py` | `results/dataset_manifest.json` |
| **YOLOv8 Model Evaluation** | `python scripts/evaluate_model.py --split test` | `results/model_evaluation_report.json` |
| **Host CPU / ONNX Latency** | `python scripts/benchmark_edge_models.py` | `results/edge_benchmark.json` |
| **Blockchain Transactions** | `python scripts/benchmark_blockchain.py` | `results/blockchain_validation.json` |
| **IPFS Storage Drivers** | `python scripts/benchmark_ipfs.py` | `results/ipfs_validation.json` |
| **Emergency Notifications** | `python scripts/validate_notifications.py` | `results/notification_validation.json` |
| **Offline Resilience** | `python scripts/validate_resilience.py` | `results/resilience_validation.json` |
| **Real UAV Batch Inference**| `python src/inference/real_data_inference.py` | `results/real_data_test/` |

---

## 4. Running the Local Application

### Start the FastAPI REST Backend
```bash
python -m uvicorn dashboard.api:app --host 0.0.0.0 --port 8000 --reload
```
* Interactive Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* Health Check: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

### Start the Streamlit Verification Dashboard
```bash
python -m streamlit run dashboard/app.py --server.port 8501
```
* Dashboard URL: [http://127.0.0.1:8501](http://127.0.0.1:8501)

---

## 5. Coding Standards & Conventions

1. **Python Package Imports:**
   Always use absolute imports rooted at the project root:
   ```python
   from config.settings import settings
   from edge.incident_detector import IncidentDetector
   from ipfs.ipfs_client import IPFSClient
   ```
2. **Explicit Package Initialization:**
   Every package directory must contain an `__init__.py` file with clear docstrings and explicitly defined `__all__` exports where applicable.
3. **Structured Logging:**
   Always instantiate loggers using `config.logging_config.setup_logger()`:
   ```python
   from config.logging_config import setup_logger
   logger = setup_logger("my_module")
   ```
   Do not use raw `print()` for production system logs.
4. **Secret Safety:**
   Never commit private keys, real API tokens, or credentials to Git. Ensure `.gitignore` ignores all `.env`, `*.db`, and `logs/` files.
5. **Compilation Verification:**
   Before submitting code changes, run:
   ```bash
   python -m compileall .
   ```
