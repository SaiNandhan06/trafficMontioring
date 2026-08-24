"""
Unit Tests for IPFS Client and Metadata Builder.
"""

import numpy as np
import pytest
from ipfs.ipfs_client import IPFSClient, generate_mock_cid
from ipfs.metadata_builder import build_incident_metadata


def test_cid_generation():
    data = b"Hello UAV Traffic Monitoring"
    cid = generate_mock_cid(data)
    assert cid.startswith("Qm")
    assert len(cid) == 46


def test_ipfs_incident_package():
    client = IPFSClient(mode="mock")
    frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

    meta, frame_bytes = build_incident_metadata(
        incident_id="INC-TEST-001",
        drone_id="UAV-TEST-01",
        incident_type="SPEEDING",
        severity="HIGH",
        confidence=0.96,
        location={"lat": 37.77, "lng": -122.41},
        involved_tracks=[1],
        description="Speeding test event",
        frame=frame
    )

    master_cid = client.upload_incident_package(meta, frame_bytes)
    assert master_cid.startswith("Qm")

    # Verify retrieval
    retrieved = client.retrieve_json(master_cid)
    assert retrieved is not None
    assert retrieved["incident_id"] == "INC-TEST-001"
    assert retrieved["drone_id"] == "UAV-TEST-01"
