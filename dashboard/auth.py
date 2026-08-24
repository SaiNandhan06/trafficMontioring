"""
Authentication & Role-Based Access Control (RBAC) Module for Verification Dashboard.
Implements JWT tokens, password hashing, and role permissions.
"""

import time
import hashlib
from typing import Dict, Optional
import jwt
from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger("auth")

# Default User Credentials Store (In-memory / Configurable)
USERS_DB = {
    "admin": {
        "password_hash": hashlib.sha256("Admin@UAV2026!".encode()).hexdigest(),
        "role": "ADMIN",
        "name": "Chief Traffic Controller"
    },
    "operator": {
        "password_hash": hashlib.sha256("DroneOps@2026".encode()).hexdigest(),
        "role": "OPERATOR",
        "name": "Fleet Flight Operator"
    },
    "responder": {
        "password_hash": hashlib.sha256("FirstResponders@911".encode()).hexdigest(),
        "role": "EMERGENCY_RESPONDER",
        "name": "Emergency Services Dispatcher"
    }
}


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hash_password(plain_password) == hashed_password


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticates credentials against database."""
    user = USERS_DB.get(username)
    if not user:
        return None
    if verify_password(password, user["password_hash"]):
        return {
            "username": username,
            "role": user["role"],
            "name": user["name"]
        }
    return None


def create_access_token(data: Dict, expires_delta_minutes: int = None) -> str:
    """Generates a signed JWT token."""
    to_encode = data.copy()
    expire_minutes = expires_delta_minutes or settings.JWT_EXPIRATION_MINUTES
    to_encode.update({"exp": time.time() + (expire_minutes * 60)})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def verify_access_token(token: str) -> Optional[Dict]:
    """Decodes and validates a JWT token."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except jwt.PyJWTError as e:
        logger.warning(f"Invalid JWT Token: {e}")
        return None
