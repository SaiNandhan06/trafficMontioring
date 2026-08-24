"""
Blockchain Emergency Event Listener.
Listens for on-chain EmergencyAlertDispatched events, normalizes payloads,
and routes alerts to the off-chain NotificationService.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from config.logging_config import setup_logger
from blockchain.contract_client import Web3ContractClient
from src.notifications.notification_service import (
    NotificationService,
    NotificationAlert,
    NotificationRecord,
    NotificationStatus
)

logger = setup_logger("event_listener")


class BlockchainEventListener:
    """Consumes EmergencyAlertDispatched events and dispatches notifications."""

    def __init__(
        self,
        blockchain_client: Optional[Web3ContractClient] = None,
        notification_service: Optional[NotificationService] = None
    ):
        self.bc_client = blockchain_client or Web3ContractClient()
        self.notifier = notification_service or NotificationService()
        self.processed_event_count = 0
        self.rejected_event_count = 0
        self.duplicates_suppressed = 0

    def parse_event(self, raw_event: Dict[str, Any]) -> Optional[NotificationAlert]:
        """Validates and normalizes raw contract event dictionary into NotificationAlert."""
        try:
            inc_id = raw_event.get("incidentId") or raw_event.get("args", {}).get("incidentId")
            ipfs_hash = raw_event.get("ipfsHash") or raw_event.get("args", {}).get("ipfsHash")
            sev = raw_event.get("severity") or raw_event.get("args", {}).get("severity", 1)
            msg = raw_event.get("message") or raw_event.get("args", {}).get("message", "Emergency Alert")
            ts = raw_event.get("dispatchedAt") or raw_event.get("args", {}).get("dispatchedAt", time.time())
            tx_hash = raw_event.get("txHash") or raw_event.get("transactionHash", f"0xsim{int(time.time())}")
            blk = raw_event.get("blockNumber", 1000)

            if isinstance(tx_hash, bytes):
                tx_hash = "0x" + tx_hash.hex()

            severity_names = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}
            severity_str = severity_names.get(sev, "CRITICAL") if isinstance(sev, int) else str(sev)

            if not inc_id or not ipfs_hash:
                logger.warning(f"Invalid emergency event payload missing incidentId or ipfsHash: {raw_event}")
                return None

            return NotificationAlert(
                incident_id=int(inc_id),
                ipfs_cid=str(ipfs_hash),
                severity=severity_str,
                message=str(msg),
                timestamp=float(ts),
                tx_hash=str(tx_hash),
                block_number=int(blk),
                network=self.bc_client.mode
            )
        except Exception as e:
            logger.error(f"Error parsing event dictionary: {e}")
            return None

    def process_new_events(self) -> List[NotificationRecord]:
        """Polls for new events from blockchain client and dispatches alerts."""
        records = []
        raw_events = self.bc_client.get_events()
        emergency_events = [e for e in raw_events if e.get("event") == "EmergencyAlertDispatched"]

        logger.info(f"EventListener discovered {len(emergency_events)} EmergencyAlertDispatched event(s).")

        for ev in emergency_events:
            alert = self.parse_event(ev)
            if alert is None:
                self.rejected_event_count += 1
                continue

            record = self.notifier.dispatch_alert(alert)
            if record.status == NotificationStatus.SUPPRESSED_DUPLICATE.value:
                self.duplicates_suppressed += 1
            else:
                self.processed_event_count += 1
            records.append(record)

        return records
