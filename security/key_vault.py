"""
Encrypted Local Keystore & Secret Vault.
Provides AES-GCM-256 encrypted storage for Ethereum private keys and API credentials.
"""

import sys
import os
import base64
import json
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger("key_vault")


class KeyVault:
    """Secure local keystore protected by a master passphrase."""

    def __init__(self, vault_path: Path = None):
        self.vault_path = vault_path or (settings.BASE_DIR / "security" / ".vault.enc")

    def _derive_key(self, passphrase: str, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(passphrase.encode("utf-8"))

    def encrypt_and_save_secrets(self, secrets: dict, passphrase: str):
        """Encrypts secrets dictionary and saves to disk."""
        salt = os.urandom(16)
        key = self._derive_key(passphrase, salt)
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)

        data = json.dumps(secrets).encode("utf-8")
        ciphertext = aesgcm.encrypt(nonce, data, None)

        payload = {
            "salt": base64.b64encode(salt).decode("utf-8"),
            "nonce": base64.b64encode(nonce).decode("utf-8"),
            "ciphertext": base64.b64encode(ciphertext).decode("utf-8")
        }

        self.vault_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.vault_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Encrypted secrets successfully saved to {self.vault_path}")

    def load_and_decrypt_secrets(self, passphrase: str) -> Optional[dict]:
        """Decrypts and returns secrets from disk."""
        if not self.vault_path.exists():
            logger.warning("Vault file does not exist.")
            return None

        with open(self.vault_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        salt = base64.b64decode(payload["salt"])
        nonce = base64.b64decode(payload["nonce"])
        ciphertext = base64.b64decode(payload["ciphertext"])

        key = self._derive_key(passphrase, salt)
        aesgcm = AESGCM(key)

        try:
            decrypted_bytes = aesgcm.decrypt(nonce, ciphertext, None)
            return json.loads(decrypted_bytes.decode("utf-8"))
        except Exception as e:
            logger.error(f"Decryption failed: {e}. Incorrect passphrase or corrupted vault.")
            return None
