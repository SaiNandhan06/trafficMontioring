"""
Central Configuration Module using Pydantic Settings.
Loads configuration from environment variables and .env file with robust defaults.
"""

from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Environment Mode
    ENVIRONMENT: Literal["development", "testing", "production"] = Field(
        default="development",
        description="Runtime environment mode: development, testing, or production"
    )

    # Edge AI & UAV Settings
    DRONE_ID: str = Field(default="UAV-ALPHA-01", description="Unique Identifier for UAV")
    DRONE_LAT: float = Field(default=37.774929, description="Base Latitude")
    DRONE_LNG: float = Field(default=-122.419418, description="Base Longitude")
    STREAM_SOURCE: str = Field(default="0", description="RTSP URL, video file path, or webcam index")
    STREAM_FPS: int = Field(default=30, description="Target Processing FPS")
    MODEL_WEIGHTS_PATH: str = Field(default="model/weights/yolov8n.pt", description="Path to YOLO weights")
    CONFIDENCE_THRESHOLD: float = Field(default=0.45, description="Detection confidence threshold")
    IOU_THRESHOLD: float = Field(default=0.50, description="NMS IoU threshold")
    USE_TENSORRT: bool = Field(default=False, description="Enable TensorRT acceleration")
    DEVICE: str = Field(default="cpu", description="Inference device: cpu, cuda:0, etc.")

    # Incident Detection Thresholds
    SPEED_LIMIT_KMH: float = Field(default=80.0, description="Speed limit in km/h for speeding violation")
    SPEED_DECELERATION_THRESHOLD: float = Field(default=25.0, description="Deceleration threshold km/h per second for collision")
    ACCIDENT_IOU_THRESHOLD: float = Field(default=0.35, description="Bounding box overlap threshold for accident")
    CONGESTION_DENSITY_THRESHOLD: float = Field(default=0.60, description="Vehicle area density threshold for congestion")
    INCIDENT_COOLDOWN_SECONDS: int = Field(default=10, description="Minimum seconds between reporting duplicate incidents")

    # Blockchain Settings
    ETH_RPC_URL: str = Field(default="http://127.0.0.1:8545", description="Ethereum JSON-RPC Endpoint")
    CHAIN_ID: int = Field(default=1337, description="EVM Chain ID")
    CONTRACT_REGISTRY_ADDRESS: str = Field(
        default="0x0000000000000000000000000000000000000000",
        description="Address of deployed TrafficIncidentRegistry.sol"
    )
    CONTRACT_EMERGENCY_ADDRESS: str = Field(
        default="0x0000000000000000000000000000000000000000",
        description="Address of deployed EmergencyNotificationService.sol"
    )
    OPERATOR_PRIVATE_KEY: str = Field(
        default="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        description="Private key of contract owner/operator"
    )
    DRONE_PRIVATE_KEY: str = Field(
        default="0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
        description="Private key of the UAV edge node"
    )
    GAS_LIMIT: int = Field(default=500000, description="Default transaction gas limit")
    GAS_PRICE_GWEI: int = Field(default=20, description="Gas price in Gwei")

    # IPFS Settings
    IPFS_MODE: Literal["mock", "local", "pinata"] = Field(default="mock", description="IPFS storage mode")
    IPFS_HOST: str = Field(default="127.0.0.1", description="IPFS node host")
    IPFS_PORT: int = Field(default=5001, description="IPFS node port")
    IPFS_GATEWAY: str = Field(default="https://ipfs.io/ipfs/", description="Public IPFS Gateway")
    PINATA_API_KEY: str = Field(default="", description="Pinata API Key")
    PINATA_SECRET_API_KEY: str = Field(default="", description="Pinata Secret API Key")
    PINATA_JWT: str = Field(default="", description="Pinata JWT")

    # Emergency Notification Settings
    NOTIFICATION_MODE: Literal["mock", "webhook"] = Field(default="mock", description="Off-chain emergency notification mode: mock or webhook")
    NOTIFICATION_WEBHOOK_URL: str = Field(default="", description="Webhook URL for external incident alert dispatch")
    NOTIFICATION_MAX_RETRIES: int = Field(default=3, description="Maximum retry attempts for failed webhook notifications")

    # Dashboard & Security Settings
    DASHBOARD_PORT: int = Field(default=8501, description="Streamlit dashboard port")
    API_HOST: str = Field(default="0.0.0.0", description="FastAPI host")
    API_PORT: int = Field(default=8000, description="FastAPI port")
    SECRET_KEY: str = Field(default="uav-traffic-super-secret-key-change-in-production", description="Secret key for JWT")
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT hashing algorithm")
    JWT_EXPIRATION_MINUTES: int = Field(default=1440, description="JWT validity duration in minutes")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    MODEL_DIR: Path = BASE_DIR / "model"
    WEIGHTS_DIR: Path = BASE_DIR / "model" / "weights"
    CONTRACTS_DIR: Path = BASE_DIR / "blockchain" / "contracts"
    ARTIFACTS_DIR: Path = BASE_DIR / "blockchain" / "build"
    QUEUE_DB_PATH: Path = BASE_DIR / "edge" / "offline_queue.db"
    MOCK_IPFS_DIR: Path = BASE_DIR / "ipfs" / "mock_store"

    def validate_security(self) -> None:
        """Validates that security-critical settings are safe for production."""
        insecure_defaults = {
            "uav-traffic-super-secret-key-change-in-production",
            "change-me",
            "secret",
            "<GENERATE_A_SECURE_JWT_SECRET_FOR_PRODUCTION>",
            ""
        }
        if self.ENVIRONMENT == "production":
            if not self.SECRET_KEY or self.SECRET_KEY in insecure_defaults or len(self.SECRET_KEY) < 32:
                raise ValueError(
                    "Production environment detected: Insecure or default SECRET_KEY is prohibited. "
                    "Please configure a strong SECRET_KEY (min 32 characters) via environment variable."
                )


settings = Settings()
settings.validate_security()

