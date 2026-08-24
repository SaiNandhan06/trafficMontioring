"""
Multi-Backend IPFS Client for Decentralized Incident Storage.
Supports Local Mock Storage, Local IPFS Daemon Node, and Pinata Cloud Gateway.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import requests
from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger("ipfs_client")


def generate_mock_cid(data: bytes) -> str:
    """Generates a deterministic IPFS v0 CID-like string (Qm...) from data bytes."""
    sha256_hash = hashlib.sha256(data).hexdigest()
    # Mock Qm CID (46 characters starting with Qm)
    return "Qm" + sha256_hash[:44]


class IPFSClient:
    """Unified IPFS Client interface."""

    def __init__(self, mode: str = None):
        self.mode = mode or settings.IPFS_MODE
        self.mock_store_dir = settings.MOCK_IPFS_DIR
        self.mock_store_dir.mkdir(parents=True, exist_ok=True)

    def upload_bytes(self, data: bytes, filename: str = "file.bin") -> str:
        """Pins raw bytes to IPFS and returns the content CID."""
        if self.mode == "pinata" and settings.PINATA_JWT:
            try:
                url = "https://api.pinata.cloud/pinning/pinFileToIPFS"
                headers = {"Authorization": f"Bearer {settings.PINATA_JWT}"}
                files = {"file": (filename, data)}
                response = requests.post(url, files=files, headers=headers, timeout=15)
                if response.status_code == 200:
                    cid = response.json()["IpfsHash"]
                    logger.info(f"[Pinata IPFS] Pinned {filename} -> CID: {cid}")
                    return cid
            except Exception as e:
                logger.warning(f"Pinata upload failed: {e}. Falling back to mock.")

        elif self.mode == "local":
            try:
                url = f"http://{settings.IPFS_HOST}:{settings.IPFS_PORT}/api/v0/add"
                files = {"file": (filename, data)}
                response = requests.post(url, files=files, timeout=10)
                if response.status_code == 200:
                    cid = response.json()["Hash"]
                    logger.info(f"[Local IPFS] Pinned {filename} -> CID: {cid}")
                    return cid
            except Exception as e:
                logger.warning(f"Local IPFS daemon upload failed: {e}. Falling back to mock.")

        # Default: Mock Local IPFS Storage
        cid = generate_mock_cid(data)
        out_file = self.mock_store_dir / cid
        with open(out_file, "wb") as f:
            f.write(data)
        logger.info(f"[Mock IPFS] Stored {filename} locally -> CID: {cid}")
        return cid

    def upload_json(self, data: Dict[str, Any], filename: str = "metadata.json") -> str:
        """Pins structured JSON document to IPFS."""
        json_bytes = json.dumps(data, indent=2).encode("utf-8")
        return self.upload_bytes(json_bytes, filename=filename)

    def upload_incident_package(self, metadata: Dict[str, Any], frame_bytes: bytes) -> str:
        """Uploads evidence frame and metadata bundle, returning root IPFS CID."""
        # 1. Upload frame image
        frame_cid = self.upload_bytes(frame_bytes, filename=f"{metadata['incident_id']}.jpg")

        # 2. Embed image CID into metadata and upload metadata
        metadata["evidence_frame"]["ipfs_image_cid"] = frame_cid
        metadata["evidence_frame"]["gateway_url"] = self.get_gateway_url(frame_cid)

        metadata_cid = self.upload_json(metadata, filename=f"{metadata['incident_id']}_meta.json")
        logger.info(f"Published Incident Package to IPFS: Master CID = {metadata_cid}")
        return metadata_cid

    def retrieve_bytes(self, cid: str) -> Optional[bytes]:
        """Retrieves raw content bytes by CID."""
        # Check mock store first
        mock_file = self.mock_store_dir / cid
        if mock_file.exists():
            with open(mock_file, "rb") as f:
                return f.read()

        try:
            gateway_url = f"{settings.IPFS_GATEWAY.rstrip('/')}/{cid}"
            resp = requests.get(gateway_url, timeout=10)
            if resp.status_code == 200:
                return resp.content
        except Exception as e:
            logger.error(f"Failed to fetch CID {cid} from gateway: {e}")

        return None

    def retrieve_json(self, cid: str) -> Optional[Dict]:
        """Retrieves and parses JSON document by CID."""
        raw = self.retrieve_bytes(cid)
        if raw:
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception as e:
                logger.error(f"Failed to parse JSON for CID {cid}: {e}")
        return None

    def get_gateway_url(self, cid: str) -> str:
        """Returns public or local HTTP URL for the CID."""
        return f"{settings.IPFS_GATEWAY.rstrip('/')}/{cid}"
