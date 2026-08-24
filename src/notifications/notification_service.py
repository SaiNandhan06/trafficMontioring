"""
Off-Chain Emergency Notification Dispatcher.
Bridges smart contract events to external dispatch channels (Mock, Webhook),
with deterministic deduplication, bounded retries, and auditable JSON logging.
"""

import sys
import time
import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
from config.settings import settings, BASE_DIR
from config.logging_config import setup_logger

logger = setup_logger("notification_service")


class NotificationStatus(str, Enum):
    PENDING = "PENDING"
    SENT = "SENT"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    SUPPRESSED_DUPLICATE = "SUPPRESSED_DUPLICATE"


@dataclass
class NotificationAlert:
    """Normalized internal emergency incident alert payload."""
    incident_id: int
    ipfs_cid: str
    severity: str
    message: str
    timestamp: float
    tx_hash: str
    block_number: int
    network: str = "simulated"


@dataclass
class NotificationRecord:
    """Auditable dispatch record."""
    alert_id: str
    incident_id: int
    severity: str
    notification_mode: str
    status: str
    attempts: int
    created_at: float
    sent_at: Optional[float] = None
    error: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None


class NotificationService:
    """Manages off-chain emergency alert delivery."""

    def __init__(
        self,
        mode: str = None,
        webhook_url: str = None,
        audit_file: Path = None
    ):
        self.mode = mode or settings.NOTIFICATION_MODE
        self.webhook_url = webhook_url or settings.NOTIFICATION_WEBHOOK_URL
        self.audit_file = audit_file or (BASE_DIR / "results" / "notification_audit.json")
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)

        self.processed_events: Dict[str, NotificationRecord] = {}
        self.audit_log: List[Dict[str, Any]] = []

    def _generate_event_id(self, alert: NotificationAlert) -> str:
        """Constructs a deterministic deduplication key from transaction hash and incident parameters."""
        return f"{alert.tx_hash}_{alert.incident_id}_{alert.ipfs_cid}"

    def dispatch_alert(self, alert: NotificationAlert, max_retries: int = None) -> NotificationRecord:
        """Dispatches an alert to configured notification channel with deduplication and bounded retries."""
        event_id = self._generate_event_id(alert)
        retries = max_retries if max_retries is not None else settings.NOTIFICATION_MAX_RETRIES

        # 1. Deduplication check
        if event_id in self.processed_events:
            prev_record = self.processed_events[event_id]
            logger.info(f"[Notification] Duplicate event detected for Incident #{alert.incident_id}. Suppressing re-dispatch.")
            dup_record = NotificationRecord(
                alert_id=event_id,
                incident_id=alert.incident_id,
                severity=alert.severity,
                notification_mode=self.mode,
                status=NotificationStatus.SUPPRESSED_DUPLICATE.value,
                attempts=prev_record.attempts,
                created_at=time.time(),
                sent_at=prev_record.sent_at,
                error="Duplicate event ignored",
                payload={"incident_id": alert.incident_id, "ipfs_cid": alert.ipfs_cid}
            )
            self._record_audit(dup_record)
            return dup_record

        # Prepare payload
        payload = {
            "incident_id": alert.incident_id,
            "severity": alert.severity,
            "ipfs_cid": alert.ipfs_cid,
            "message": alert.message,
            "timestamp": alert.timestamp,
            "tx_hash": alert.tx_hash,
            "block_number": alert.block_number,
            "network": alert.network
        }

        # 2. Mock Notification Mode
        if self.mode == "mock":
            logger.info(
                f"[SIMULATED NOTIFICATION] Dispatching alert for Incident #{alert.incident_id} "
                f"| Severity: {alert.severity} | CID: {alert.ipfs_cid} | Status: MOCK_DELIVERED"
            )
            record = NotificationRecord(
                alert_id=event_id,
                incident_id=alert.incident_id,
                severity=alert.severity,
                notification_mode="mock",
                status=NotificationStatus.SENT.value,
                attempts=1,
                created_at=time.time(),
                sent_at=time.time(),
                payload=payload
            )
            self.processed_events[event_id] = record
            self._record_audit(record)
            return record

        # 3. Webhook Notification Mode
        if self.mode == "webhook":
            if not self.webhook_url:
                logger.warning("[Notification] Webhook mode configured but NOTIFICATION_WEBHOOK_URL is empty.")
                record = NotificationRecord(
                    alert_id=event_id,
                    incident_id=alert.incident_id,
                    severity=alert.severity,
                    notification_mode="webhook",
                    status=NotificationStatus.FAILED.value,
                    attempts=1,
                    created_at=time.time(),
                    error="Webhook URL not configured in environment",
                    payload=payload
                )
                self.processed_events[event_id] = record
                self._record_audit(record)
                return record

            # Execute HTTP POST with bounded retries
            attempts = 0
            last_error = None
            for attempt in range(1, retries + 1):
                attempts = attempt
                try:
                    logger.info(f"[Notification] Attempting webhook POST to {self.webhook_url} (Attempt {attempt}/{retries})...")
                    resp = requests.post(self.webhook_url, json=payload, timeout=5)
                    if 200 <= resp.status_code < 300:
                        logger.info(f"[Notification] Webhook delivered successfully (HTTP {resp.status_code})")
                        record = NotificationRecord(
                            alert_id=event_id,
                            incident_id=alert.incident_id,
                            severity=alert.severity,
                            notification_mode="webhook",
                            status=NotificationStatus.SENT.value,
                            attempts=attempts,
                            created_at=time.time(),
                            sent_at=time.time(),
                            payload=payload
                        )
                        self.processed_events[event_id] = record
                        self._record_audit(record)
                        return record
                    elif 400 <= resp.status_code < 500:
                        # Client error - permanent failure
                        last_error = f"HTTP {resp.status_code}: {resp.text[:100]}"
                        logger.error(f"[Notification] Webhook client error: {last_error}")
                        break
                    else:
                        last_error = f"HTTP {resp.status_code}: {resp.text[:100]}"
                        logger.warning(f"[Notification] Webhook server error: {last_error}")
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"[Notification] Webhook network error: {e}")

                time.sleep(0.05 * (2 ** (attempt - 1)))  # Micro backoff

            record = NotificationRecord(
                alert_id=event_id,
                incident_id=alert.incident_id,
                severity=alert.severity,
                notification_mode="webhook",
                status=NotificationStatus.FAILED.value,
                attempts=attempts,
                created_at=time.time(),
                error=last_error,
                payload=payload
            )
            self.processed_events[event_id] = record
            self._record_audit(record)
            return record

        # Unknown mode fallback
        record = NotificationRecord(
            alert_id=event_id,
            incident_id=alert.incident_id,
            severity=alert.severity,
            notification_mode=self.mode,
            status=NotificationStatus.FAILED.value,
            attempts=1,
            created_at=time.time(),
            error=f"Unsupported notification mode: {self.mode}"
        )
        self.processed_events[event_id] = record
        self._record_audit(record)
        return record

    def _record_audit(self, record: NotificationRecord):
        """Persists notification record to audit log JSON."""
        rec_dict = asdict(record)
        self.audit_log.append(rec_dict)
        try:
            with open(self.audit_file, "w", encoding="utf-8") as f:
                json.dump(self.audit_log, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write notification audit log: {e}")
