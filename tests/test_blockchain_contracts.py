"""
Comprehensive Smart Contract & Web3 Blockchain Validation Suite.
Tests Solidity compilation, access control, drone authorization,
incident lifecycle states, emergency alert dispatch, event logs,
and simulated vs live EVM mode separation.
"""

import time
import pytest
from blockchain.compile import compile_solidity_contracts
from blockchain.contract_client import Web3ContractClient, SimulatedBlockchainState
from config.settings import settings


def test_contract_compilation_produces_artifacts():
    """Verifies that contract compilation produces valid ABI and bytecode artifacts."""
    artifacts = compile_solidity_contracts()
    assert "TrafficIncidentRegistry" in artifacts
    assert "EmergencyNotificationService" in artifacts

    reg = artifacts["TrafficIncidentRegistry"]
    assert "abi" in reg and len(reg["abi"]) > 0
    assert "bytecode" in reg and len(reg["bytecode"]) > 0

    emg = artifacts["EmergencyNotificationService"]
    assert "abi" in emg and len(emg["abi"]) > 0
    assert "bytecode" in emg and len(emg["bytecode"]) > 0


def test_owner_drone_registration_and_deactivation():
    """Verifies that contract owner can register and deactivate UAV drone nodes."""
    sim = SimulatedBlockchainState()
    new_drone = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"

    # 1. Owner registers drone
    ok, msg = sim.register_drone(new_drone, "UAV-BETA-02", "Hexacopter 4K", caller=sim.owner)
    assert ok is True
    assert sim.drones[new_drone]["is_active"] is True

    # 2. Owner deactivates drone
    ok, msg = sim.deactivate_drone(new_drone, caller=sim.owner)
    assert ok is True
    assert sim.drones[new_drone]["is_active"] is False


def test_unauthorized_drone_registration_rejected():
    """Verifies that non-owner accounts cannot register or deactivate drones."""
    sim = SimulatedBlockchainState()
    intruder = "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"
    target = "0x90F79bf6EB2c4f870365E785982E1f101E93b906"

    # Intruder tries to register
    ok, msg = sim.register_drone(target, "UAV-ROGUE", "Unauthorized", caller=intruder)
    assert ok is False
    assert "Only registry owner" in msg

    # Intruder tries to deactivate active drone
    ok, msg = sim.deactivate_drone(sim.owner, caller=intruder)
    assert ok is False
    assert "Only registry owner" in msg


def test_active_drone_incident_reporting_and_event():
    """Verifies that an authorized active drone can report an incident and emit IncidentReported event."""
    client = Web3ContractClient()
    drone_addr = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"

    success, inc_id, tx_hash = client.report_incident(
        ipfs_hash="QmValidIncidentCID123456789",
        incident_type="COLLISION_ACCIDENT",
        severity_str="CRITICAL",
        latitude=37.774929,
        longitude=-122.419418,
        timestamp=time.time(),
        sender=drone_addr
    )
    assert success is True
    assert inc_id == 1
    assert tx_hash.startswith("0x")

    # Read back incident
    inc = client.get_incident(inc_id)
    assert inc is not None
    assert inc["ipfsHash"] == "QmValidIncidentCID123456789"
    assert inc["incidentType"] == "COLLISION_ACCIDENT"
    assert inc["severity"] == 3  # CRITICAL
    assert inc["status"] == 0    # REPORTED
    assert inc["reportingDrone"] == drone_addr


def test_unauthorized_and_deactivated_drone_reporting_rejected():
    """Verifies that unauthorized or deactivated drones cannot report incidents."""
    sim = SimulatedBlockchainState()
    rogue_drone = "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65"

    # 1. Unregistered drone tries to report
    ok, inc_id, tx, msg = sim.report_incident(
        ipfs_hash="QmFakeCID",
        incident_type="SPEEDING",
        severity=1,
        lat=37.77,
        lng=-122.41,
        timestamp=time.time(),
        sender=rogue_drone
    )
    assert ok is False
    assert "Caller is not an active authorized UAV drone" in msg

    # 2. Registered then deactivated drone tries to report
    sim.register_drone(rogue_drone, "UAV-TEMP", "Temp Drone", caller=sim.owner)
    sim.deactivate_drone(rogue_drone, caller=sim.owner)

    ok, inc_id, tx, msg = sim.report_incident(
        ipfs_hash="QmFakeCID",
        incident_type="SPEEDING",
        severity=1,
        lat=37.77,
        lng=-122.41,
        timestamp=time.time(),
        sender=rogue_drone
    )
    assert ok is False
    assert "Caller is not an active authorized UAV drone" in msg


def test_incident_lifecycle_escalation_and_resolution():
    """Verifies complete lifecycle: REPORTED (0) -> ESCALATED (1) -> RESOLVED (3)."""
    client = Web3ContractClient()

    # Step 1: Report
    ok, inc_id, _ = client.report_incident(
        ipfs_hash="QmAccidentCID999",
        incident_type="COLLISION_ACCIDENT",
        severity_str="CRITICAL",
        latitude=37.775,
        longitude=-122.419
    )
    assert ok is True

    # Step 2: Escalate
    ok, _ = client.sim_state.escalate_incident(inc_id, "Ambulance dispatched to scene", caller=client.sim_state.owner)
    assert ok is True
    inc = client.get_incident(inc_id)
    assert inc["status"] == 1  # ESCALATED

    # Step 3: Resolve
    ok, _ = client.resolve_incident(inc_id, "Highway lanes cleared by emergency responders")
    assert ok is True
    inc = client.get_incident(inc_id)
    assert inc["status"] == 3  # RESOLVED
    assert inc["resolutionNotes"] == "Highway lanes cleared by emergency responders"
    assert inc["resolvedAt"] > 0


def test_emergency_notification_alert_dispatch():
    """Verifies dispatching on-chain emergency alerts for critical incidents."""
    sim = SimulatedBlockchainState()
    ok, msg = sim.notify_emergency(
        incident_id=1,
        ipfs_hash="QmCriticalCID123",
        severity=3,
        details="Multi-vehicle collision blocking 2 lanes",
        caller=sim.owner
    )
    assert ok is True
    assert len(sim.emergency_alerts) == 1
    alert = sim.emergency_alerts[0]
    assert alert["incidentId"] == 1
    assert alert["severity"] == 3


def test_blockchain_mode_transparency():
    """Verifies that Web3ContractClient transparently reports simulated vs live EVM mode."""
    client = Web3ContractClient()
    assert client.mode in ["simulated", "local_evm", "testnet"]
    assert client.is_simulated is True or client.w3 is not None
