"""
Offline-First Resilience & Failure Recovery Regression Tests.
Verifies SQLite queue initialization, state transitions, duplicate suppression,
dead-letter archiving, restart persistence, and multi-dependency replay engine.
"""

import time
import json
import pytest
from pathlib import Path
from edge.retry_queue import RetryQueue
from ipfs.ipfs_client import IPFSClient
from blockchain.contract_client import Web3ContractClient
from src.notifications.notification_service import NotificationService
from config.settings import BASE_DIR


@pytest.fixture
def temp_queue(tmp_path):
    db_path = tmp_path / "test_isolated_queue.db"
    return RetryQueue(db_path=db_path)


def test_retry_queue_initialization_on_missing_db(tmp_path):
    """Verifies that RetryQueue initializes schema even when DB file does not exist."""
    db_path = tmp_path / "new_dir" / "fresh_queue.db"
    assert not db_path.exists()
    q = RetryQueue(db_path=db_path)
    assert db_path.exists()
    stats = q.get_stats()
    assert stats["total"] == 0


def test_retry_queue_normal_lifecycle(temp_queue):
    """Verifies push -> pending -> mark_success state flow."""
    inc = {
        "incident_id": "INC-TEST-001",
        "incident_type": "COLLISION_ACCIDENT",
        "severity": 3,
        "latitude": 37.7749,
        "longitude": -122.4194,
        "timestamp": time.time()
    }
    assert temp_queue.push(inc) is True

    pending = temp_queue.get_pending(ignore_backoff=True)
    assert len(pending) == 1
    assert pending[0]["incident_id"] == "INC-TEST-001"

    temp_queue.mark_success("INC-TEST-001", ipfs_hash="QmTestCID01", tx_hash="0xabc123")
    stats = temp_queue.get_stats()
    assert stats["success"] == 1
    assert stats["pending"] == 0


def test_retry_queue_duplicate_prevention(temp_queue):
    """Verifies that pushing the exact same incident ID is rejected to avoid duplicate work."""
    inc = {
        "incident_id": "INC-TEST-DUP",
        "incident_type": "SPEEDING",
        "severity": 2,
        "latitude": 37.77,
        "longitude": -122.41
    }
    assert temp_queue.push(inc) is True
    assert temp_queue.push(inc) is False  # Duplicate ignored

    stats = temp_queue.get_stats()
    assert stats["total"] == 1


def test_retry_queue_dead_letter_transition(temp_queue):
    """Verifies that items exceeding max_retries transition to DEAD_LETTER."""
    inc = {
        "incident_id": "INC-TEST-FAIL",
        "incident_type": "TRAFFIC_CONGESTION",
        "severity": 1
    }
    temp_queue.push(inc)
    pending = temp_queue.get_pending(ignore_backoff=True)
    item_id = pending[0]["id"]

    # Fail 3 times with max_retries=3
    temp_queue.mark_failed_attempt(item_id, "Fail 1", max_retries=3)
    temp_queue.mark_failed_attempt(item_id, "Fail 2", max_retries=3)
    temp_queue.mark_failed_attempt(item_id, "Fail 3", max_retries=3)

    stats = temp_queue.get_stats()
    assert stats["dead_letter"] == 1
    assert stats["pending"] == 0


def test_retry_queue_restart_persistence(tmp_path):
    """Verifies SQLite persistence across simulated application restarts."""
    db_path = tmp_path / "persist_queue.db"
    q1 = RetryQueue(db_path=db_path)
    q1.push({"incident_id": "INC-PERSIST-1", "incident_type": "SPEEDING", "severity": 1})
    del q1

    # Simulate app reboot
    q2 = RetryQueue(db_path=db_path)
    pending = q2.get_pending(ignore_backoff=True)
    assert len(pending) == 1
    assert pending[0]["incident_id"] == "INC-PERSIST-1"


def test_retry_queue_replay_pending_success(temp_queue):
    """Verifies replay_pending integrates IPFS, Blockchain, and Notification services."""
    temp_queue.push({
        "incident_id": "INC-REPLAY-001",
        "incident_type": "COLLISION_ACCIDENT",
        "severity": 3,
        "latitude": 37.77,
        "longitude": -122.41
    })

    ipfs_c = IPFSClient(mode="mock")
    bc_c = Web3ContractClient()
    notif_s = NotificationService(mode="mock")

    res = temp_queue.replay_pending(ipfs_c, bc_c, notif_s)
    assert res["replayed"] == 1
    assert res["succeeded"] == 1
    assert res["failed"] == 0

    stats = temp_queue.get_stats()
    assert stats["success"] == 1


def test_resilience_validation_report_exists_and_valid():
    """Verifies results/resilience_validation.json schema."""
    report_file = BASE_DIR / "results" / "resilience_validation.json"
    assert report_file.exists()

    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "scenarios" in data
    assert len(data["scenarios"]) >= 5
    assert data["summary"]["passed_scenarios"] == len(data["scenarios"])
