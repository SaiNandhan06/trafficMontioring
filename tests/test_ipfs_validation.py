"""
Comprehensive IPFS Storage & Evidence CID Integrity Regression Tests.
Tests Mock IPFS storage, deterministic CID generation, content mutation detection,
evidence frame compression/decoding, master package bundling,
blockchain CID linkage, and security payload compliance.
"""

import json
import numpy as np
import cv2
import pytest

from ipfs.ipfs_client import IPFSClient, generate_mock_cid
from ipfs.metadata_builder import build_incident_metadata
from blockchain.contract_client import Web3ContractClient
from config.settings import settings


def test_ipfs_mock_byte_upload_and_exact_retrieval():
    """Verifies that uploaded raw bytes match retrieved bytes exactly."""
    client = IPFSClient(mode="mock")
    payload = b"Traffic Incident High-Speed Braking Telemetry Telemetry 2026"

    cid = client.upload_bytes(payload, filename="telemetry.bin")
    assert cid.startswith("Qm")
    assert len(cid) == 46

    retrieved = client.retrieve_bytes(cid)
    assert retrieved == payload, "Retrieved bytes must be 100% bit-exact"


def test_ipfs_deterministic_cid_generation():
    """Verifies that identical content generates identical CIDs."""
    payload1 = b"Identical drone incident snapshot data"
    payload2 = b"Identical drone incident snapshot data"

    cid1 = generate_mock_cid(payload1)
    cid2 = generate_mock_cid(payload2)
    assert cid1 == cid2, "Deterministic CIDs must match for identical content"


def test_ipfs_mutated_content_produces_different_cid():
    """Verifies that mutated content produces a distinct CID."""
    payload_a = b"Severity: HIGH"
    payload_b = b"Severity: CRITICAL"

    cid_a = generate_mock_cid(payload_a)
    cid_b = generate_mock_cid(payload_b)
    assert cid_a != cid_b, "Mutated content must produce a distinct content-addressed identifier"


def test_ipfs_evidence_frame_encoding_and_image_decode():
    """Verifies evidence frame JPEG compression, upload, retrieval, and OpenCV decoding."""
    client = IPFSClient(mode="mock")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), -1)

    _, meta_bytes = build_incident_metadata(
        incident_id="INC-EVID-01",
        drone_id="UAV-ALPHA-01",
        incident_type="SPEEDING",
        severity="MEDIUM",
        confidence=0.89,
        location={"lat": 37.7749, "lng": -122.4194},
        involved_tracks=[1],
        description="Speeding test",
        frame=frame
    )

    img_cid = client.upload_bytes(meta_bytes, filename="frame.jpg")
    retrieved_bytes = client.retrieve_bytes(img_cid)
    assert retrieved_bytes is not None

    decoded = cv2.imdecode(np.frombuffer(retrieved_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (480, 640, 3)


def test_ipfs_master_incident_package_bundle():
    """Verifies that upload_incident_package embeds frame CID and returns a valid root metadata CID."""
    client = IPFSClient(mode="mock")
    frame = np.zeros((300, 300, 3), dtype=np.uint8)

    metadata, frame_bytes = build_incident_metadata(
        incident_id="INC-PKG-02",
        drone_id="UAV-BETA-02",
        incident_type="COLLISION_ACCIDENT",
        severity="CRITICAL",
        confidence=0.98,
        location={"lat": 37.77, "lng": -122.42},
        involved_tracks=[1, 2],
        description="Accident bundle test",
        frame=frame
    )

    master_cid = client.upload_incident_package(metadata, frame_bytes)
    assert master_cid.startswith("Qm")

    retrieved_json = client.retrieve_json(master_cid)
    assert retrieved_json is not None
    assert retrieved_json["incident_id"] == "INC-PKG-02"
    assert "ipfs_image_cid" in retrieved_json["evidence_frame"]
    assert retrieved_json["evidence_frame"]["ipfs_image_cid"].startswith("Qm")


def test_ipfs_nonexistent_cid_returns_none():
    """Verifies that requesting an invalid or missing CID returns None cleanly."""
    client = IPFSClient(mode="mock")
    res = client.retrieve_bytes("QmInvalidNonExistentHash0000000000000000000000")
    assert res is None


def test_ipfs_payload_security_no_secrets():
    """Verifies that serialized IPFS payloads contain no credentials or private keys."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    metadata, _ = build_incident_metadata(
        incident_id="INC-SEC-01",
        drone_id="UAV-ALPHA-01",
        incident_type="CONGESTION",
        severity="LOW",
        confidence=0.92,
        location={"lat": 37.77, "lng": -122.42},
        involved_tracks=[1],
        description="Security test",
        frame=frame
    )

    meta_str = json.dumps(metadata)
    assert "private_key" not in meta_str.lower()
    assert "jwt" not in meta_str.lower()
    assert "password" not in meta_str.lower()


def test_ipfs_blockchain_cid_linkage():
    """Verifies that the Master CID from IPFS matches the on-chain IncidentRecord.ipfsHash."""
    client = IPFSClient(mode="mock")
    bc_client = Web3ContractClient()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    meta, frame_bytes = build_incident_metadata(
        incident_id="INC-BC-LINK",
        drone_id="UAV-ALPHA-01",
        incident_type="SPEEDING",
        severity="HIGH",
        confidence=0.94,
        location={"lat": 37.77, "lng": -122.42},
        involved_tracks=[5],
        description="Blockchain link test",
        frame=frame
    )

    master_cid = client.upload_incident_package(meta, frame_bytes)
    ok, inc_id, _ = bc_client.report_incident(
        ipfs_hash=master_cid,
        incident_type="SPEEDING",
        severity_str="HIGH",
        latitude=37.77,
        longitude=-122.42
    )
    assert ok is True

    onchain = bc_client.get_incident(inc_id)
    assert onchain is not None
    assert onchain["ipfsHash"] == master_cid
