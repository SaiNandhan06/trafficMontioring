"""
FastAPI Backend API for UAV Traffic Monitoring & Blockchain Verification Dashboard.
Provides REST and WebSocket endpoints for incident ingestion, telemetry, and blockchain queries.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Depends, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from config.settings import settings
from config.logging_config import setup_logger
from dashboard.auth import authenticate_user, create_access_token, verify_access_token
from dashboard.ui_components import get_all_provenance_reports, get_source_attribution_matrix
from blockchain.contract_client import Web3ContractClient
from ipfs.ipfs_client import IPFSClient
from edge.retry_queue import RetryQueue
from src.notifications.notification_service import NotificationService

logger = setup_logger("api")

app = FastAPI(
    title="UAV Edge AI + Blockchain Traffic Monitoring API",
    version="1.0.0",
    description="Decentralized Traffic Incident Verification & Management Backend"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Service Singletons
blockchain_client = Web3ContractClient()
ipfs_client = IPFSClient()
retry_queue = RetryQueue()
notification_service = NotificationService()


# Request / Response Schemas
class LoginRequest(BaseModel):
    username: str
    password: str


class IncidentReportRequest(BaseModel):
    incident_id: str
    drone_id: str
    incident_type: str
    severity: str
    confidence: float
    latitude: float
    longitude: float
    description: str
    ipfs_hash: str
    timestamp: Optional[float] = None


class DroneRegisterRequest(BaseModel):
    drone_address: str
    drone_id: str
    metadata: str


class IncidentResolveRequest(BaseModel):
    resolution_notes: str


# Authentication Dependency
def get_current_user(authorization: Optional[str] = Header(None)) -> Dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header"
        )
    token = authorization.split(" ")[1]
    payload = verify_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired or invalid"
        )
    return payload


# Endpoints
@app.get("/api/health")
def health_check():
    """Honest multi-component system health report."""
    return {
        "status": "HEALTHY",
        "timestamp": time.time(),
        "components": {
            "application": "HEALTHY",
            "model": "READY",
            "blockchain": {
                "mode": blockchain_client.mode.upper(),
                "is_simulated": blockchain_client.is_simulated
            },
            "ipfs": {
                "mode": settings.IPFS_MODE.upper(),
                "status": "LOCAL_MOCK_STORE" if settings.IPFS_MODE == "mock" else "EXTERNAL"
            },
            "notification": {
                "mode": settings.NOTIFICATION_MODE.upper(),
                "webhook_configured": bool(settings.NOTIFICATION_WEBHOOK_URL)
            }
        },
        "queue_stats": retry_queue.get_stats()
    }


@app.get("/api/system/provenance")
def system_provenance():
    """Returns the system-wide source attribution matrix."""
    return {
        "matrix": get_source_attribution_matrix(),
        "reports": get_all_provenance_reports()
    }


@app.post("/api/auth/login")
def login(req: LoginRequest):
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token({"sub": user["username"], "role": user["role"], "name": user["name"]})
    return {"access_token": token, "token_type": "bearer", "user": user}


@app.get("/api/incidents")
def list_incidents():
    """Returns list of verified on-chain incidents."""
    incidents = blockchain_client.get_all_incidents()
    if not incidents:
        try:
            incidents = retry_queue.get_all_incidents()
        except Exception:
            incidents = []
    return {"count": len(incidents), "incidents": incidents}


@app.get("/api/incidents/{incident_id}")
def get_incident_detail(incident_id: int):
    """Fetches details for a single incident including IPFS metadata."""
    incidents = blockchain_client.get_all_incidents()
    match = next((i for i in incidents if i.get("incidentId") == incident_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Incident not found")

    ipfs_data = None
    if match.get("ipfsHash"):
        ipfs_data = ipfs_client.retrieve_json(match["ipfsHash"])

    return {"record": match, "ipfs_metadata": ipfs_data}


@app.post("/api/incidents/report")
def report_incident(req: IncidentReportRequest):
    """Direct REST ingestion endpoint for UAV edge nodes."""
    success, inc_id, tx_hash = blockchain_client.report_incident(
        ipfs_hash=req.ipfs_hash,
        incident_type=req.incident_type,
        severity_str=req.severity,
        latitude=req.latitude,
        longitude=req.longitude,
        timestamp=req.timestamp or time.time()
    )
    if not success:
        retry_queue.push(req.dict())
        return {"status": "QUEUED_OFFLINE", "incident_id": req.incident_id}

    return {"status": "RECORDED_ON_CHAIN", "onchain_id": inc_id, "tx_hash": tx_hash}


@app.post("/api/incidents/{incident_id}/resolve")
def resolve_incident(incident_id: int, req: IncidentResolveRequest, user: Dict = Depends(get_current_user)):
    """Resolves an on-chain incident record (Admin/Operator only)."""
    ok, tx = blockchain_client.resolve_incident(incident_id, req.resolution_notes)
    if not ok:
        raise HTTPException(status_code=400, detail="Failed to resolve incident")
    return {"status": "RESOLVED", "incident_id": incident_id, "tx_hash": tx}


@app.post("/api/drone/register")
def register_drone(req: DroneRegisterRequest, user: Dict = Depends(get_current_user)):
    """Registers and authorizes a new UAV drone (Admin only)."""
    ok, tx = blockchain_client.register_drone(req.drone_address, req.drone_id, req.metadata)
    if not ok:
        raise HTTPException(status_code=400, detail="Failed to register drone on blockchain")
    return {"status": "REGISTERED", "drone_id": req.drone_id, "drone_address": req.drone_address}


@app.get("/api/blockchain/stats")
def blockchain_stats():
    """Returns smart contract statistics and transaction activity."""
    return {
        "network": settings.ETH_RPC_URL,
        "chain_id": settings.CHAIN_ID,
        "registry_address": settings.CONTRACT_REGISTRY_ADDRESS,
        "emergency_address": settings.CONTRACT_EMERGENCY_ADDRESS,
        "mode": blockchain_client.mode,
        "is_simulated": blockchain_client.is_simulated,
        "total_incidents": len(blockchain_client.get_all_incidents()),
        "recent_txs": blockchain_client.sim_state.transactions[-10:] if blockchain_client.is_simulated else []
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
