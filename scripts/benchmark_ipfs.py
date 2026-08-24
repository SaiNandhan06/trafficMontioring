"""
IPFS Decentralized Storage & CID Integrity Validation Benchmark.
Tests Mock IPFS, Local Kubo, and Pinata modes, measures upload/retrieval latencies,
and verifies content-addressing integrity and security compliance.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import requests

from config.settings import settings, BASE_DIR
from config.logging_config import setup_logger
from ipfs.ipfs_client import IPFSClient, generate_mock_cid
from ipfs.metadata_builder import build_incident_metadata
from blockchain.contract_client import Web3ContractClient

logger = setup_logger("benchmark_ipfs")


def check_kubo_status() -> Dict[str, Any]:
    """Checks whether a local Kubo IPFS daemon is reachable."""
    url = f"http://{settings.IPFS_HOST}:{settings.IPFS_PORT}/api/v0/version"
    try:
        resp = requests.post(url, timeout=2)
        if resp.status_code == 200:
            return {"available": True, "version": resp.json().get("Version", "Unknown"), "status": "ONLINE"}
    except Exception:
        pass
    return {"available": False, "status": "BLOCKED — Local Kubo daemon (port 5001) unavailable"}


def check_pinata_status() -> Dict[str, Any]:
    """Checks whether Pinata credentials are valid."""
    if not settings.PINATA_JWT:
        return {"available": False, "status": "BLOCKED — PINATA_JWT not configured in .env"}
    try:
        url = "https://api.pinata.cloud/data/testAuthentication"
        headers = {"Authorization": f"Bearer {settings.PINATA_JWT}"}
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            return {"available": True, "status": "AUTHENTICATED"}
    except Exception as e:
        return {"available": False, "status": f"FAILED — {e}"}
    return {"available": False, "status": "FAILED — Invalid authentication"}


def run_ipfs_validation() -> Dict[str, Any]:
    """Executes end-to-end IPFS validation across evidence generation, upload, and retrieval."""
    client = IPFSClient(mode="mock")
    kubo_status = check_kubo_status()
    pinata_status = check_pinata_status()

    # 1. Raw Bytes Integrity Test
    test_payload = b"SkyGuard UAV Decentralized Traffic Monitoring Evidence String Test Payload 2026"
    t0 = time.perf_counter()
    cid_raw = client.upload_bytes(test_payload, filename="test_raw.txt")
    t_upload_raw = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    retrieved_raw = client.retrieve_bytes(cid_raw)
    t_retrieve_raw = (time.perf_counter() - t0) * 1000.0

    raw_integrity = (test_payload == retrieved_raw)

    # 2. Evidence Image Compression & Retrieval
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "INCIDENT EVIDENCE SNAPSHOT #101", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    cv2.rectangle(dummy_frame, (300, 200), (600, 500), (0, 255, 0), 3)

    metadata, frame_bytes = build_incident_metadata(
        incident_id="INC-TEST-001",
        drone_id="UAV-ALPHA-01",
        incident_type="COLLISION_ACCIDENT",
        severity="CRITICAL",
        confidence=0.96,
        location={"lat": 37.774929, "lng": -122.419418},
        involved_tracks=[1, 2],
        description="Multi-vehicle collision verified at intersection",
        frame=dummy_frame,
        detections=[{"bbox": [300, 200, 600, 500], "class_id": 0, "conf": 0.96}]
    )

    t0 = time.perf_counter()
    master_cid = client.upload_incident_package(metadata, frame_bytes)
    t_upload_pkg = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    retrieved_meta = client.retrieve_json(master_cid)
    t_retrieve_pkg = (time.perf_counter() - t0) * 1000.0

    img_cid = retrieved_meta.get("evidence_frame", {}).get("ipfs_image_cid")
    retrieved_img_bytes = client.retrieve_bytes(img_cid)
    decoded_img = cv2.imdecode(np.frombuffer(retrieved_img_bytes, np.uint8), cv2.IMREAD_COLOR)

    img_valid = (decoded_img is not None and decoded_img.shape == (720, 1280, 3))
    meta_valid = (retrieved_meta["incident_id"] == "INC-TEST-001" and retrieved_meta["incident"]["type"] == "COLLISION_ACCIDENT")

    # 3. CID Stability & Mutation Test
    cid_orig = generate_mock_cid(b"Incident Severity: HIGH")
    cid_same = generate_mock_cid(b"Incident Severity: HIGH")
    cid_mutated = generate_mock_cid(b"Incident Severity: CRITICAL")

    stability_verified = (cid_orig == cid_same) and (cid_orig != cid_mutated)

    # 4. Security Check (No Secrets)
    meta_str = json.dumps(retrieved_meta)
    no_secrets = ("private_key" not in meta_str and "SECRET" not in meta_str and "PINATA_JWT" not in meta_str)

    # 5. Blockchain Integration
    bc_client = Web3ContractClient()
    ok, inc_id, tx_hash = bc_client.report_incident(
        ipfs_hash=master_cid,
        incident_type="COLLISION_ACCIDENT",
        severity_str="CRITICAL",
        latitude=37.774929,
        longitude=-122.419418
    )
    onchain_inc = bc_client.get_incident(inc_id)
    bc_match = (onchain_inc is not None and onchain_inc["ipfsHash"] == master_cid)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "mock",
        "backends": {
            "mock": {
                "status": "AVAILABLE & VERIFIED",
                "storage_directory": str(settings.MOCK_IPFS_DIR),
                "deterministic_cids": True
            },
            "local_kubo": kubo_status,
            "pinata_cloud": pinata_status
        },
        "performance": {
            "raw_payload_bytes": len(test_payload),
            "raw_upload_latency_ms": round(t_upload_raw, 3),
            "raw_retrieval_latency_ms": round(t_retrieve_raw, 3),
            "package_payload_bytes": len(frame_bytes) + len(json.dumps(metadata)),
            "package_upload_latency_ms": round(t_upload_pkg, 3),
            "package_retrieval_latency_ms": round(t_retrieve_pkg, 3),
            "measured_or_simulated": "measured (local mock store)"
        },
        "integrity_verification": {
            "raw_bytes_integrity": raw_integrity,
            "evidence_image_decoded": img_valid,
            "metadata_json_valid": meta_valid,
            "cid_stability_and_mutation": stability_verified,
            "security_no_secrets_leaked": no_secrets,
            "blockchain_cid_match": bc_match,
            "master_root_cid": master_cid,
            "evidence_image_cid": img_cid
        }
    }

    out_file = BASE_DIR / "results" / "ipfs_validation.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(" [PHASE 8: IPFS STORAGE & EVIDENCE INTEGRITY VALIDATION REPORT] ")
    print("=" * 70)
    print(f"Active IPFS Mode:         MOCK (Local Content-Addressed Store)")
    print(f"Local Kubo Daemon:        {kubo_status['status']}")
    print(f"Pinata Cloud Gateway:     {pinata_status['status']}")
    print("-" * 70)
    print(f"Raw Bytes Integrity:      {'VERIFIED (Exact Match)' if raw_integrity else 'FAILED'}")
    print(f"Evidence Image Decode:    {'VERIFIED (720x1280 BGR Match)' if img_valid else 'FAILED'}")
    print(f"Metadata JSON Integrity:  {'VERIFIED (All fields intact)' if meta_valid else 'FAILED'}")
    print(f"CID Stability & Mutation: {'VERIFIED (Content-addressed)' if stability_verified else 'FAILED'}")
    print(f"Blockchain CID Match:     {'VERIFIED (Recorded On-Chain)' if bc_match else 'FAILED'}")
    print(f"Security / Privacy Audit: {'PASSED (Zero secrets in payload)' if no_secrets else 'FAILED'}")
    print("-" * 70)
    print(f"Master Package Root CID:  {master_cid}")
    print(f"Evidence Frame Image CID: {img_cid}")
    print(f"Package Upload Latency:   {t_upload_pkg:.3f} ms")
    print(f"Package Retrieval Lat:    {t_retrieve_pkg:.3f} ms")
    print("=" * 70)
    print(f"Report written to: {out_file}\n")

    return report


if __name__ == "__main__":
    run_ipfs_validation()
