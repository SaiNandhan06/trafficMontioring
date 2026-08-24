"""
Security Authentication & Cryptographic Message Verification.
Provides HMAC-SHA256 telemetry signing for drone-to-edge channels and JWT tokens for operators.
"""

import sys
import hmac
import hashlib
import time
import json
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger("security_auth")


class UAVSecurityManager:
    """Manages cryptographic message integrity and drone identity verification."""

    def __init__(self, secret_key: str = None):
        self.secret_key = (secret_key or settings.SECRET_KEY).encode("utf-8")

    def sign_telemetry_payload(self, payload: Dict) -> str:
        """Generates HMAC-SHA256 signature for a dictionary payload."""
        # Canonical JSON string
        canonical_str = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            self.secret_key,
            canonical_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return signature

    def verify_telemetry_payload(self, payload: Dict, signature: str) -> bool:
        """Verifies HMAC signature of received drone payload."""
        expected_sig = self.sign_telemetry_payload(payload)
        is_valid = hmac.compare_digest(expected_sig, signature)
        if not is_valid:
            logger.warning("Telemetry signature verification failed!")
        return is_valid

    # Aliases
    sign_telemetry = sign_telemetry_payload
    verify_telemetry = verify_telemetry_payload


if __name__ == "__main__":
    sec = UAVSecurityManager()
    sample_data = {"drone_id": "UAV-ALPHA-01", "lat": 37.77, "lng": -122.41, "ts": time.time()}
    sig = sec.sign_telemetry_payload(sample_data)
    print(f"Generated HMAC Signature: {sig}")
    valid = sec.verify_telemetry_payload(sample_data, sig)
    print(f"Signature Verification: {valid}")
