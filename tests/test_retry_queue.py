"""
Unit Tests for Offline-First SQLite Retry Queue.
"""

import time
from pathlib import Path
import pytest
from edge.retry_queue import RetryQueue


def test_retry_queue_lifecycle(tmp_path: Path):
    db_file = tmp_path / "test_queue.db"
    queue = RetryQueue(db_path=db_file)

    sample_inc = {
        "incident_id": "INC-QUEUE-001",
        "type": "ACCIDENT",
        "severity": "CRITICAL",
        "ipfs_hash": "QmQueueHashTest123"
    }

    # Push to queue
    success = queue.push(sample_inc)
    assert success is True

    # Check pending
    pending = queue.get_pending()
    assert len(pending) == 1
    assert pending[0]["incident_id"] == "INC-QUEUE-001"

    # Mark success
    queue.mark_success("INC-QUEUE-001", tx_hash="0xabcdef123456")
    pending_after = queue.get_pending()
    assert len(pending_after) == 0

    stats = queue.get_stats()
    assert stats["completed"] == 1
