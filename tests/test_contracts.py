"""
Unit & Integration Tests for Solidity Contracts & Web3 Client.
"""

import pytest
from blockchain.compile import compile_solidity_contracts
from blockchain.contract_client import Web3ContractClient


def test_contract_compilation():
    """Verifies that compilation succeeds and creates ABI artifacts."""
    compiled = compile_solidity_contracts()
    assert "TrafficIncidentRegistry" in compiled
    assert "EmergencyNotificationService" in compiled
    assert "abi" in compiled["TrafficIncidentRegistry"]
    assert "bytecode" in compiled["TrafficIncidentRegistry"]


def test_blockchain_incident_reporting():
    """Verifies that incident reporting succeeds on-chain or via simulation."""
    client = Web3ContractClient()
    success, inc_id, tx_hash = client.report_incident(
        ipfs_hash="QmTestHash1234567890abcdef",
        incident_type="COLLISION_ACCIDENT",
        severity_str="CRITICAL",
        latitude=37.7749,
        longitude=-122.4194
    )
    assert success is True
    assert tx_hash is not None
    assert inc_id is not None

    # Verify incident is in state
    all_inc = client.get_all_incidents()
    assert len(all_inc) >= 1
    latest = all_inc[-1]
    assert latest["incidentType"] == "COLLISION_ACCIDENT"
    assert latest["ipfsHash"] == "QmTestHash1234567890abcdef"
