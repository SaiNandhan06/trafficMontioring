"""
Off-Chain Emergency Notification & Blockchain Event Listener Regression Tests.
Validates event parsing, mock delivery, duplicate suppression,
bounded webhook retries, payload security, and audit logging.
"""

import json
import time
import pytest
from pathlib import Path

from src.notifications.notification_service import (
    NotificationService,
    NotificationAlert,
    NotificationStatus
)
from src.notifications.event_listener import BlockchainEventListener
from blockchain.contract_client import Web3ContractClient
from config.settings import BASE_DIR


@pytest.fixture
def temp_audit_file(tmp_path):
    return tmp_path / "test_notification_audit.json"


def test_mock_notification_delivery_and_audit(temp_audit_file):
    """Verifies that mock notification delivers cleanly and records to audit log."""
    service = NotificationService(mode="mock", audit_file=temp_audit_file)
    alert = NotificationAlert(
        incident_id=1,
        ipfs_cid="QmTestCID12345",
        severity="CRITICAL",
        message="Collision detected",
        timestamp=time.time(),
        tx_hash="0xabcd1234",
        block_number=1001
    )

    record = service.dispatch_alert(alert)
    assert record.status == NotificationStatus.SENT.value
    assert record.attempts == 1
    assert record.notification_mode == "mock"

    # Verify audit file was written
    assert temp_audit_file.exists()
    with open(temp_audit_file, "r", encoding="utf-8") as f:
        logs = json.load(f)
    assert len(logs) == 1
    assert logs[0]["incident_id"] == 1


def test_duplicate_notification_suppression(temp_audit_file):
    """Verifies that identical events are suppressed as SUPPRESSED_DUPLICATE."""
    service = NotificationService(mode="mock", audit_file=temp_audit_file)
    alert = NotificationAlert(
        incident_id=2,
        ipfs_cid="QmTestCID67890",
        severity="HIGH",
        message="Speeding alert",
        timestamp=time.time(),
        tx_hash="0xefgh5678",
        block_number=1002
    )

    # 1st dispatch
    rec1 = service.dispatch_alert(alert)
    assert rec1.status == NotificationStatus.SENT.value

    # 2nd dispatch of same event
    rec2 = service.dispatch_alert(alert)
    assert rec2.status == NotificationStatus.SUPPRESSED_DUPLICATE.value


def test_webhook_missing_url_fails_gracefully(temp_audit_file):
    """Verifies that webhook mode with unconfigured URL returns FAILED without crashing."""
    service = NotificationService(mode="webhook", webhook_url="", audit_file=temp_audit_file)
    alert = NotificationAlert(
        incident_id=3,
        ipfs_cid="QmTestCID11111",
        severity="MEDIUM",
        message="Congestion",
        timestamp=time.time(),
        tx_hash="0xijkl9999",
        block_number=1003
    )

    record = service.dispatch_alert(alert)
    assert record.status == NotificationStatus.FAILED.value
    assert "not configured" in record.error


def test_webhook_bounded_retries(temp_audit_file):
    """Verifies that webhook retries are strictly bounded on connection failure."""
    service = NotificationService(
        mode="webhook",
        webhook_url="http://127.0.0.1:9998/offline_endpoint",
        audit_file=temp_audit_file
    )
    alert = NotificationAlert(
        incident_id=4,
        ipfs_cid="QmTestCID22222",
        severity="CRITICAL",
        message="Braking",
        timestamp=time.time(),
        tx_hash="0xmnop0000",
        block_number=1004
    )

    record = service.dispatch_alert(alert, max_retries=2)
    assert record.status == NotificationStatus.FAILED.value
    assert record.attempts == 2


def test_event_listener_parses_contract_event():
    """Verifies that BlockchainEventListener normalizes raw contract events into NotificationAlert."""
    listener = BlockchainEventListener()
    raw_event = {
        "event": "EmergencyAlertDispatched",
        "incidentId": 10,
        "ipfsHash": "QmRawEventCID12345",
        "severity": 3,
        "message": "Major pileup",
        "dispatchedAt": 1724427900,
        "txHash": "0x1234567890abcdef",
        "blockNumber": 1010
    }

    alert = listener.parse_event(raw_event)
    assert alert is not None
    assert alert.incident_id == 10
    assert alert.ipfs_cid == "QmRawEventCID12345"
    assert alert.severity == "CRITICAL"
    assert alert.message == "Major pileup"
    assert alert.block_number == 1010


def test_event_listener_rejects_malformed_event():
    """Verifies that event listener rejects malformed events missing required identifiers."""
    listener = BlockchainEventListener()
    bad_event = {"event": "EmergencyAlertDispatched", "message": "Incomplete"}
    alert = listener.parse_event(bad_event)
    assert alert is None


def test_notification_payload_privacy_no_secrets():
    """Verifies that notification payload does not contain private keys or JWT tokens."""
    alert = NotificationAlert(
        incident_id=5,
        ipfs_cid="QmTestCID33333",
        severity="LOW",
        message="Clean alert",
        timestamp=time.time(),
        tx_hash="0xqrst1111",
        block_number=1005
    )
    service = NotificationService(mode="mock")
    record = service.dispatch_alert(alert)

    payload_str = json.dumps(record.payload)
    assert "private_key" not in payload_str.lower()
    assert "jwt" not in payload_str.lower()
    assert "password" not in payload_str.lower()
