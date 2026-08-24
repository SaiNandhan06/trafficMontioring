"""
FastAPI REST API Contract & Service Integration Regression Tests.
Validates health reporting, JWT authentication guards, incident reporting,
blockchain queries, and provenance matrix delivery.
"""

import pytest
from fastapi.testclient import TestClient
from dashboard.api import app
from config.settings import settings


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health_check_honesty(client):
    """Verifies that GET /api/health returns honest component statuses."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "HEALTHY"
    assert "components" in data
    assert data["components"]["blockchain"]["mode"] in ["SIMULATED", "LOCAL_EVM", "TESTNET"]
    assert data["components"]["ipfs"]["mode"] in ["MOCK", "LOCAL", "PINATA"]


def test_api_system_provenance(client):
    """Verifies that GET /api/system/provenance serves the system attribution matrix."""
    resp = client.get("/api/system/provenance")
    assert resp.status_code == 200
    data = resp.json()

    assert "matrix" in data
    assert len(data["matrix"]) >= 8
    assert "reports" in data


def test_api_auth_login_success_and_failure(client):
    """Verifies POST /api/auth/login credentials handling and JWT token generation."""
    # 1. Valid login
    resp_ok = client.post("/api/auth/login", json={"username": "admin", "password": "Admin@UAV2026!"})
    assert resp_ok.status_code == 200
    token_data = resp_ok.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"

    # 2. Invalid login
    resp_bad = client.post("/api/auth/login", json={"username": "admin", "password": "WrongPassword!"})
    assert resp_bad.status_code == 401


def test_api_list_and_report_incidents(client):
    """Verifies listing and reporting incidents via REST API."""
    # 1. Report new incident
    report_payload = {
        "incident_id": "INC-API-TEST-01",
        "drone_id": "UAV-ALPHA-01",
        "incident_type": "SPEEDING",
        "severity": "HIGH",
        "confidence": 0.95,
        "latitude": 37.7749,
        "longitude": -122.4194,
        "description": "API Ingestion Test",
        "ipfs_hash": "QmApiTestCID123456789"
    }
    resp_rep = client.post("/api/incidents/report", json=report_payload)
    assert resp_rep.status_code == 200
    assert resp_rep.json()["status"] in ["RECORDED_ON_CHAIN", "QUEUED_OFFLINE"]

    # 2. List incidents
    resp_list = client.get("/api/incidents")
    assert resp_list.status_code == 200
    assert "incidents" in resp_list.json()


def test_api_protected_endpoints_auth_guards(client):
    """Verifies that admin/operator endpoints require valid Bearer token."""
    # 1. Reject without token
    resp_unauth = client.post("/api/drone/register", json={
        "drone_address": "0x1234567890123456789012345678901234567890",
        "drone_id": "UAV-TEST-01",
        "metadata": "Test Drone"
    })
    assert resp_unauth.status_code == 401

    # 2. Accept with valid token
    login_resp = client.post("/api/auth/login", json={"username": "admin", "password": "Admin@UAV2026!"})
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    resp_auth = client.post("/api/drone/register", json={
        "drone_address": "0x1234567890123456789012345678901234567890",
        "drone_id": "UAV-TEST-01",
        "metadata": "Test Drone"
    }, headers=headers)
    assert resp_auth.status_code == 200
    assert resp_auth.json()["status"] == "REGISTERED"


def test_api_blockchain_stats(client):
    """Verifies GET /api/blockchain/stats."""
    resp = client.get("/api/blockchain/stats")
    assert resp.status_code == 200
    stats = resp.json()
    assert "network" in stats
    assert "chain_id" in stats
    assert "mode" in stats
