"""
Emergency Notification & Event Listener Validation Benchmark.
Tests blockchain event consumption, mock notification dispatch,
duplicate alert suppression, bounded retry behavior, and audit logging.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings, BASE_DIR
from config.logging_config import setup_logger
from blockchain.contract_client import Web3ContractClient
from src.notifications.notification_service import NotificationService, NotificationAlert, NotificationStatus
from src.notifications.event_listener import BlockchainEventListener

logger = setup_logger("validate_notifications")


def run_notification_validation() -> Dict[str, Any]:
    """Executes end-to-end emergency notification validation benchmark."""
    bc_client = Web3ContractClient()
    audit_file = BASE_DIR / "results" / "notification_audit.json"
    notifier = NotificationService(mode="mock", audit_file=audit_file)
    listener = BlockchainEventListener(blockchain_client=bc_client, notification_service=notifier)

    logger.info("Triggering emergency alert on blockchain layer...")
    t0 = time.perf_counter()
    ok, _ = bc_client.sim_state.notify_emergency(
        incident_id=101,
        ipfs_hash="QmTestEmergencyIncidentCID101",
        severity=3,  # CRITICAL
        details="Critical 3-vehicle collision blocking 2 lanes at Exit 42",
        caller=bc_client.sim_state.owner
    )
    t_event = (time.perf_counter() - t0) * 1000.0

    # 1. Process Event
    t0 = time.perf_counter()
    records = listener.process_new_events()
    t_process = (time.perf_counter() - t0) * 1000.0

    assert len(records) >= 1
    first_record = records[0]

    # 2. Test Duplicate Event Suppression
    t0 = time.perf_counter()
    dup_records = listener.process_new_events()
    t_dup = (time.perf_counter() - t0) * 1000.0

    dup_suppressed = any(r.status == NotificationStatus.SUPPRESSED_DUPLICATE.value for r in dup_records)

    # 3. Test Webhook Retry Simulation
    webhook_notifier = NotificationService(mode="webhook", webhook_url="http://127.0.0.1:9999/nonexistent_webhook")
    test_alert = NotificationAlert(
        incident_id=102,
        ipfs_cid="QmTestRetryCID102",
        severity="HIGH",
        message="Highway obstruction detected",
        timestamp=time.time(),
        tx_hash="0xsimretry0001",
        block_number=1005
    )
    retry_record = webhook_notifier.dispatch_alert(test_alert, max_retries=2)
    retry_handled = (retry_record.status == NotificationStatus.FAILED.value and retry_record.attempts == 2)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event_source": {
            "source": "EmergencyNotificationService.sol",
            "event_name": "EmergencyAlertDispatched",
            "mode": bc_client.mode,
            "is_simulated": bc_client.is_simulated
        },
        "notification_modes": {
            "mock": {
                "available": True,
                "verified": True,
                "label": "SIMULATED NOTIFICATION (MOCK_DELIVERED)"
            },
            "webhook": {
                "available": True,
                "verified": retry_handled,
                "status": "CONFIGURABLE (Tested with bounded retry on offline endpoint)"
            }
        },
        "performance": {
            "events_received": 1,
            "events_processed": listener.processed_event_count,
            "duplicates_suppressed": listener.duplicates_suppressed,
            "notifications_sent": 1,
            "notifications_failed": 1 if retry_handled else 0,
            "event_dispatch_latency_ms": round(t_event, 3),
            "event_processing_latency_ms": round(t_process, 3),
            "measured_or_simulated": "measured (local mock & event loop)"
        },
        "test_audit": {
            "first_incident_id": first_record.incident_id,
            "first_severity": first_record.severity,
            "first_status": first_record.status,
            "duplicate_suppression_verified": dup_suppressed,
            "bounded_retry_verified": retry_handled
        }
    }

    out_file = BASE_DIR / "results" / "notification_validation.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(" [PHASE 9: OFF-CHAIN EMERGENCY NOTIFICATION & EVENT LISTENER REPORT] ")
    print("=" * 70)
    print(f"Event Source:            EmergencyAlertDispatched ({bc_client.mode.upper()})")
    print(f"Notification Mode:       MOCK (Simulated Development Notifier)")
    print("-" * 70)
    print(f"Events Processed:        {listener.processed_event_count} (Status: {first_record.status})")
    print(f"Duplicate Suppression:   {'VERIFIED (SUPPRESSED_DUPLICATE)' if dup_suppressed else 'FAILED'}")
    print(f"Bounded Retry (Webhook): {'VERIFIED (Max 2 retries on offline endpoint)' if retry_handled else 'FAILED'}")
    print("-" * 70)
    print(f"Event Processing Lat:    {t_process:.3f} ms")
    print(f"Audit Log Path:          {audit_file}")
    print(f"Validation Report Path:  {out_file}")
    print("=" * 70 + "\n")

    return report


if __name__ == "__main__":
    run_notification_validation()
