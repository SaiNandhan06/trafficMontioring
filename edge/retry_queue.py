"""
Offline-First Persistent Incident and Transaction Retry Queue.
Stores failed or pending IPFS uploads and Web3 blockchain transactions in SQLite
with exponential backoff, dead-letter archiving, and end-to-end replay mechanisms.
"""

import sys
import json
import time
import sqlite3
import threading
from pathlib import Path
from typing import List, Dict, Optional, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger("retry_queue")


class RetryQueue:
    """Manages persistent SQLite queue for offline-first resilience."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.QUEUE_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self):
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS pending_incidents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        incident_id TEXT UNIQUE NOT NULL,
                        payload_json TEXT NOT NULL,
                        ipfs_hash TEXT,
                        tx_hash TEXT,
                        status TEXT NOT NULL DEFAULT 'PENDING',
                        retry_count INTEGER DEFAULT 0,
                        last_attempt REAL DEFAULT 0,
                        last_error TEXT,
                        created_at REAL NOT NULL
                    )
                """)
                conn.commit()

    def push(self, incident_dict: Dict[str, Any]) -> bool:
        """Pushes an incident into the queue. Idempotent against duplicate incident_ids."""
        inc_id = incident_dict.get("incident_id")
        if not inc_id:
            logger.warning("Cannot push incident without incident_id")
            return False

        with self._lock:
            try:
                with self._get_connection() as conn:
                    # Check if already processed
                    cursor = conn.cursor()
                    cursor.execute("SELECT status FROM pending_incidents WHERE incident_id = ?", (inc_id,))
                    existing = cursor.fetchone()
                    if existing:
                        logger.info(f"[RetryQueue] Incident {inc_id} already exists in queue (Status: {existing[0]}). Skipping duplicate push.")
                        return False

                    conn.execute(
                        """
                        INSERT INTO pending_incidents
                        (incident_id, payload_json, ipfs_hash, status, retry_count, last_attempt, created_at)
                        VALUES (?, ?, ?, 'PENDING', 0, 0.0, ?)
                        """,
                        (
                            inc_id,
                            json.dumps(incident_dict),
                            incident_dict.get("ipfs_hash"),
                            time.time()
                        )
                    )
                    conn.commit()
                logger.info(f"[RetryQueue] Queued incident {inc_id} for persistent dispatch.")
                return True
            except Exception as e:
                logger.error(f"[RetryQueue] Failed to queue incident: {e}")
                return False

    def get_pending(self, max_retries: int = 5, ignore_backoff: bool = False) -> List[Dict[str, Any]]:
        """Fetches items that are eligible for retry based on exponential backoff."""
        items = []
        now = time.time()
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, incident_id, payload_json, ipfs_hash, retry_count, last_attempt, status
                    FROM pending_incidents
                    WHERE status IN ('PENDING', 'RETRYING') AND retry_count < ?
                    ORDER BY id ASC
                    """,
                    (max_retries,)
                )
                for row in cursor.fetchall():
                    row_id, inc_id, payload_json, ipfs_h, retries, last_att, st_val = row
                    backoff = 0.0 if ignore_backoff else (2 ** retries) * 1.5
                    if (now - last_att) >= backoff:
                        try:
                            payload = json.loads(payload_json)
                        except Exception:
                            payload = {}
                        items.append({
                            "id": row_id,
                            "incident_id": inc_id,
                            "payload": payload,
                            "ipfs_hash": ipfs_h,
                            "retry_count": retries,
                            "status": st_val
                        })
        return items

    def get_pending_incidents(self, max_retries: int = 5) -> List[Dict[str, Any]]:
        """Alias for get_pending with unpacked payload fields."""
        items = []
        for item in self.get_pending(max_retries=max_retries, ignore_backoff=True):
            payload = item.get("payload", {})
            items.append({
                "id": item["id"],
                "incident_id": item["incident_id"],
                "incident_type": payload.get("incident_type", payload.get("incident", {}).get("type", "UNKNOWN")),
                "severity": payload.get("severity", payload.get("incident", {}).get("severity", 1)),
                "latitude": payload.get("latitude", payload.get("location", {}).get("latitude", settings.DRONE_LAT)),
                "longitude": payload.get("longitude", payload.get("location", {}).get("longitude", settings.DRONE_LNG)),
                "timestamp": payload.get("timestamp", time.time()),
                "ipfs_hash": item.get("ipfs_hash") or payload.get("ipfs_hash", "N/A"),
                "status": 1
            })
        return items

    def get_all_incidents(self) -> List[Dict[str, Any]]:
        """Returns all records regardless of status."""
        items = []
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, incident_id, payload_json, ipfs_hash, status FROM pending_incidents ORDER BY id DESC")
                for row in cursor.fetchall():
                    row_id, inc_id, payload_json, ipfs_h, status = row
                    try:
                        payload = json.loads(payload_json)
                    except Exception:
                        payload = {}
                    items.append({
                        "id": row_id,
                        "incident_id": inc_id,
                        "incident_type": payload.get("incident_type", payload.get("incident", {}).get("type", "UNKNOWN")),
                        "severity": payload.get("severity", payload.get("incident", {}).get("severity", 1)),
                        "latitude": payload.get("latitude", payload.get("location", {}).get("latitude", settings.DRONE_LAT)),
                        "longitude": payload.get("longitude", payload.get("location", {}).get("longitude", settings.DRONE_LNG)),
                        "timestamp": payload.get("timestamp", time.time()),
                        "ipfs_hash": ipfs_h or payload.get("ipfs_hash", "N/A"),
                        "status": 3 if status in ("SUCCESS", "COMPLETED") else 1
                    })
        return items

    def mark_success(self, incident_id: str, ipfs_hash: Optional[str] = None, tx_hash: Optional[str] = None):
        """Marks an item as successfully completed."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE pending_incidents
                    SET status = 'SUCCESS', ipfs_hash = COALESCE(?, ipfs_hash), tx_hash = ?, last_attempt = ?
                    WHERE incident_id = ?
                    """,
                    (ipfs_hash, tx_hash, time.time(), incident_id)
                )
                conn.commit()
            logger.info(f"[RetryQueue] Marked incident {incident_id} as successfully synchronized (Status: SUCCESS).")

    def mark_failed_attempt(self, item_id: int, error_msg: str = "", max_retries: int = 5):
        """Increments retry count on failure. Transitions to DEAD_LETTER when retries are exhausted."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT retry_count FROM pending_incidents WHERE id = ?", (item_id,))
                row = cursor.fetchone()
                if not row:
                    return

                new_count = row[0] + 1
                new_status = "DEAD_LETTER" if new_count >= max_retries else "RETRYING"

                conn.execute(
                    """
                    UPDATE pending_incidents
                    SET retry_count = ?, status = ?, last_error = ?, last_attempt = ?
                    WHERE id = ?
                    """,
                    (new_count, new_status, error_msg, time.time(), item_id)
                )
                conn.commit()
            if new_status == "DEAD_LETTER":
                logger.error(f"[RetryQueue] Item #{item_id} exhausted max retries ({max_retries}). Transitioned to DEAD_LETTER.")
            else:
                logger.warning(f"[RetryQueue] Item #{item_id} attempt failed ({new_count}/{max_retries}). Status: RETRYING.")

    def replay_pending(
        self,
        ipfs_client: Any,
        blockchain_client: Any,
        notification_service: Optional[Any] = None,
        max_retries: int = 5
    ) -> Dict[str, int]:
        """Processes and replays eligible pending queue items through IPFS, Blockchain, and Notifications."""
        pending_items = self.get_pending(max_retries=max_retries, ignore_backoff=True)
        results = {"replayed": 0, "succeeded": 0, "failed": 0}

        for item in pending_items:
            results["replayed"] += 1
            inc_id = item["incident_id"]
            payload = item["payload"]
            cid = item.get("ipfs_hash")

            try:
                # 1. IPFS Pinning if CID missing
                if not cid or cid == "N/A":
                    cid = ipfs_client.upload_json(payload, f"{inc_id}_meta.json")

                # 2. Blockchain submission
                sev_val = payload.get("severity", payload.get("incident", {}).get("severity", 1))
                sev_str = "CRITICAL" if sev_val in (3, "CRITICAL") else "HIGH" if sev_val in (2, "HIGH") else "MEDIUM"
                lat = payload.get("latitude", payload.get("location", {}).get("latitude", settings.DRONE_LAT))
                lng = payload.get("longitude", payload.get("location", {}).get("longitude", settings.DRONE_LNG))
                inc_type = payload.get("incident_type", payload.get("incident", {}).get("type", "UNKNOWN"))

                ok, onchain_id, tx_hash = blockchain_client.report_incident(
                    ipfs_hash=cid,
                    incident_type=inc_type,
                    severity_str=sev_str,
                    latitude=lat,
                    longitude=lng
                )

                if ok:
                    # 3. Notification dispatch for high/critical incidents
                    if notification_service and sev_str in ("HIGH", "CRITICAL"):
                        from src.notifications.notification_service import NotificationAlert
                        alert = NotificationAlert(
                            incident_id=onchain_id or 1,
                            ipfs_cid=cid,
                            severity=sev_str,
                            message=f"Replayed offline incident: {inc_type}",
                            timestamp=time.time(),
                            tx_hash=tx_hash or f"0xsim{inc_id}",
                            block_number=1000
                        )
                        notification_service.dispatch_alert(alert)

                    self.mark_success(inc_id, ipfs_hash=cid, tx_hash=tx_hash)
                    results["succeeded"] += 1
                else:
                    self.mark_failed_attempt(item["id"], "Blockchain transaction rejected", max_retries=max_retries)
                    results["failed"] += 1
            except Exception as e:
                self.mark_failed_attempt(item["id"], str(e), max_retries=max_retries)
                results["failed"] += 1

        return results

    def get_stats(self) -> Dict[str, int]:
        """Returns comprehensive queue statistics."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT status, COUNT(*) FROM pending_incidents GROUP BY status")
                stats = dict(cursor.fetchall())
            success_count = stats.get("SUCCESS", 0) + stats.get("COMPLETED", 0)
            return {
                "pending": stats.get("PENDING", 0),
                "retrying": stats.get("RETRYING", 0),
                "success": success_count,
                "completed": success_count,
                "failed": stats.get("FAILED", 0),
                "dead_letter": stats.get("DEAD_LETTER", 0),
                "total": sum(stats.values())
            }
