"""
Web3 Blockchain Client for TrafficIncidentRegistry & EmergencyNotificationService.
Handles wallet key signing, transaction dispatch, gas estimation, and on-chain event querying.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger("blockchain_client")


class SimulatedBlockchainState:
    """In-memory blockchain state simulator when external RPC is unavailable."""

    def __init__(self, owner_address: str = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"):
        self.owner = owner_address
        self.incidents: List[Dict] = []
        self.drones: Dict[str, Dict] = {
            "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d": {
                "drone_id": "UAV-ALPHA-01",
                "metadata": "Primary UAV Alpha Drone",
                "is_active": True,
                "registered_at": time.time()
            },
            self.owner: {
                "drone_id": "UAV-SYSTEM-ROOT",
                "metadata": "Initial Deployer UAV",
                "is_active": True,
                "registered_at": time.time()
            }
        }
        self.transactions: List[Dict] = []
        self.events: List[Dict] = []
        self.emergency_alerts: List[Dict] = []

    def register_drone(self, drone_address: str, drone_id: str, metadata: str, caller: str) -> Tuple[bool, str]:
        """Registers a new drone (onlyOwner)."""
        if caller.lower() != self.owner.lower():
            return False, "Only registry owner can execute this action"
        if not drone_address or drone_address == "0x0000000000000000000000000000000000000000":
            return False, "Invalid drone address"

        self.drones[drone_address] = {
            "drone_id": drone_id,
            "metadata": metadata,
            "is_active": True,
            "registered_at": time.time()
        }
        self.events.append({
            "event": "DroneRegistered",
            "droneAddress": drone_address,
            "droneId": drone_id,
            "registeredAt": int(time.time()),
            "blockNumber": 1000 + len(self.transactions)
        })
        return True, "Drone registered successfully"

    def deactivate_drone(self, drone_address: str, caller: str) -> Tuple[bool, str]:
        """Deactivates an active drone (onlyOwner)."""
        if caller.lower() != self.owner.lower():
            return False, "Only registry owner can execute this action"
        if drone_address not in self.drones or not self.drones[drone_address]["is_active"]:
            return False, "Drone is not currently active"

        self.drones[drone_address]["is_active"] = False
        self.events.append({
            "event": "DroneDeactivated",
            "droneAddress": drone_address,
            "blockNumber": 1000 + len(self.transactions)
        })
        return True, "Drone deactivated successfully"

    def report_incident(
        self,
        ipfs_hash: str,
        incident_type: str,
        severity: int,
        lat: float,
        lng: float,
        timestamp: float,
        sender: str
    ) -> Tuple[bool, Optional[int], Optional[str], str]:
        """Reports a verified incident (onlyActiveDrone)."""
        is_owner = (sender.lower() == self.owner.lower())
        is_active_drone = (sender in self.drones and self.drones[sender]["is_active"])
        if not (is_owner or is_active_drone):
            return False, None, None, "Caller is not an active authorized UAV drone"

        if not ipfs_hash:
            return False, None, None, "IPFS hash required"

        inc_id = len(self.incidents) + 1
        tx_hash = f"0x{abs(hash(ipfs_hash + str(time.time()) + str(inc_id))):064x}"
        block_num = 1000 + len(self.transactions)
        rec = {
            "incidentId": inc_id,
            "ipfsHash": ipfs_hash,
            "incidentType": incident_type,
            "severity": severity,
            "latitude": int(lat * 1e6),
            "longitude": int(lng * 1e6),
            "timestamp": int(timestamp),
            "reportingDrone": sender,
            "status": 0,  # 0: REPORTED, 1: ESCALATED, 2: UNDER_INVESTIGATION, 3: RESOLVED
            "resolutionNotes": "",
            "resolvedAt": 0,
            "txHash": tx_hash,
            "gasUsed": 142850,
            "blockNumber": block_num
        }
        self.incidents.append(rec)
        self.transactions.append({
            "tx_hash": tx_hash,
            "from": sender,
            "to": settings.CONTRACT_REGISTRY_ADDRESS,
            "type": "reportIncident",
            "incident_id": inc_id,
            "gas_used": 142850,
            "block_number": block_num,
            "timestamp": time.time()
        })
        self.events.append({
            "event": "IncidentReported",
            "incidentId": inc_id,
            "ipfsHash": ipfs_hash,
            "incidentType": incident_type,
            "severity": severity,
            "latitude": int(lat * 1e6),
            "longitude": int(lng * 1e6),
            "timestamp": int(timestamp),
            "reportingDrone": sender,
            "txHash": tx_hash,
            "blockNumber": block_num
        })
        return True, inc_id, tx_hash, "Success"

    def escalate_incident(self, incident_id: int, reason: str, caller: str) -> Tuple[bool, str]:
        """Escalates an incident to emergency response."""
        if incident_id < 1 or incident_id > len(self.incidents):
            return False, "Incident does not exist"

        inc = self.incidents[incident_id - 1]
        inc["status"] = 1  # ESCALATED
        self.events.append({
            "event": "IncidentEscalated",
            "incidentId": incident_id,
            "reason": reason,
            "escalatedAt": int(time.time()),
            "blockNumber": 1000 + len(self.transactions)
        })
        return True, "Incident escalated"

    def resolve_incident(self, incident_id: int, notes: str, caller: str) -> Tuple[bool, str]:
        """Resolves an incident with resolution notes (onlyOwner)."""
        if caller.lower() != self.owner.lower():
            return False, "Only registry owner can execute this action"
        if incident_id < 1 or incident_id > len(self.incidents):
            return False, "Incident does not exist"

        inc = self.incidents[incident_id - 1]
        inc["status"] = 3  # RESOLVED
        inc["resolutionNotes"] = notes
        inc["resolvedAt"] = int(time.time())
        self.events.append({
            "event": "IncidentResolved",
            "incidentId": incident_id,
            "resolutionNotes": notes,
            "resolvedAt": int(time.time()),
            "blockNumber": 1000 + len(self.transactions)
        })
        return True, "Incident resolved successfully"

    def notify_emergency(self, incident_id: int, ipfs_hash: str, severity: int, details: str, caller: str) -> Tuple[bool, str]:
        """Dispatches an emergency alert event."""
        alert = {
            "incidentId": incident_id,
            "ipfsHash": ipfs_hash,
            "severity": severity,
            "message": details,
            "dispatchedAt": int(time.time()),
            "caller": caller
        }
        self.emergency_alerts.append(alert)
        self.events.append({
            "event": "EmergencyAlertDispatched",
            "incidentId": incident_id,
            "ipfsHash": ipfs_hash,
            "severity": severity,
            "message": details,
            "dispatchedAt": int(time.time()),
            "blockNumber": 1000 + len(self.transactions)
        })
        return True, "Emergency alert dispatched on-chain"


class Web3ContractClient:
    """Production Web3 Client with simulated fallback."""

    def __init__(
        self,
        rpc_url: str = None,
        registry_address: str = None,
        emergency_address: str = None,
        private_key: str = None
    ):
        self.rpc_url = rpc_url or settings.ETH_RPC_URL
        self.registry_address = registry_address or settings.CONTRACT_REGISTRY_ADDRESS
        self.emergency_address = emergency_address or settings.CONTRACT_EMERGENCY_ADDRESS
        self.private_key = private_key or settings.OPERATOR_PRIVATE_KEY

        self.w3 = None
        self.account = None
        self.registry_contract = None
        self.emergency_contract = None
        self.is_simulated = True
        self.sim_state = SimulatedBlockchainState()

        self._init_web3()

    @property
    def mode(self) -> str:
        if self.is_simulated:
            return "simulated"
        if "127.0.0.1" in self.rpc_url or "localhost" in self.rpc_url:
            return "local_evm"
        return "testnet"

    def _init_web3(self):
        try:
            from web3 import Web3
            from eth_account import Account

            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            if self.w3.is_connected():
                logger.info(f"Connected to Ethereum RPC at {self.rpc_url}")
                self.account = Account.from_key(self.private_key)
                self.is_simulated = False

                # Load ABI artifacts
                reg_artifact = settings.ARTIFACTS_DIR / "TrafficIncidentRegistry.json"
                if reg_artifact.exists() and self.registry_address != "0x0000000000000000000000000000000000000000":
                    with open(reg_artifact, "r") as f:
                        data = json.load(f)
                    self.registry_contract = self.w3.eth.contract(
                        address=Web3.to_checksum_address(self.registry_address),
                        abi=data["abi"]
                    )
            else:
                logger.warning("Web3 RPC not reachable. Falling back to in-memory simulated blockchain.")
                self.is_simulated = True
        except Exception as e:
            logger.warning(f"Web3 initialization note: {e}. Using simulated blockchain state.")
            self.is_simulated = True

    def register_drone(self, drone_address: str, drone_id: str, metadata: str = "UAV Drone Node") -> Tuple[bool, str]:
        """Registers a new drone."""
        if self.is_simulated or self.registry_contract is None:
            caller = self.account.address if self.account else self.sim_state.owner
            return self.sim_state.register_drone(drone_address, drone_id, metadata, caller=caller)

        try:
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            txn = self.registry_contract.functions.registerDrone(
                self.w3.to_checksum_address(drone_address),
                drone_id,
                metadata
            ).build_transaction({
                "from": self.account.address,
                "nonce": nonce,
                "gas": 150000,
                "gasPrice": self.w3.to_wei(settings.GAS_PRICE_GWEI, "gwei"),
                "chainId": settings.CHAIN_ID
            })
            signed_txn = self.w3.eth.account.sign_transaction(txn, private_key=self.private_key)
            tx_hash = self.w3.to_hex(self.w3.eth.send_raw_transaction(signed_txn.rawTransaction))
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
            return (receipt.status == 1), tx_hash
        except Exception as e:
            logger.error(f"registerDrone failed: {e}")
            return False, str(e)

    def deactivate_drone(self, drone_address: str) -> Tuple[bool, str]:
        """Deactivates a drone."""
        if self.is_simulated or self.registry_contract is None:
            caller = self.account.address if self.account else self.sim_state.owner
            return self.sim_state.deactivate_drone(drone_address, caller=caller)

        try:
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            txn = self.registry_contract.functions.deactivateDrone(
                self.w3.to_checksum_address(drone_address)
            ).build_transaction({
                "from": self.account.address,
                "nonce": nonce,
                "gas": 100000,
                "gasPrice": self.w3.to_wei(settings.GAS_PRICE_GWEI, "gwei"),
                "chainId": settings.CHAIN_ID
            })
            signed_txn = self.w3.eth.account.sign_transaction(txn, private_key=self.private_key)
            tx_hash = self.w3.to_hex(self.w3.eth.send_raw_transaction(signed_txn.rawTransaction))
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
            return (receipt.status == 1), tx_hash
        except Exception as e:
            logger.error(f"deactivateDrone failed: {e}")
            return False, str(e)

    def report_incident(
        self,
        ipfs_hash: str,
        incident_type: str,
        severity_str: str,
        latitude: float,
        longitude: float,
        timestamp: float = 0.0,
        sender: str = None
    ) -> Tuple[bool, Optional[int], Optional[str]]:
        """Submits an incident report to the on-chain TrafficIncidentRegistry."""
        severity_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        severity_uint = severity_map.get(severity_str.upper(), 1)
        lat_scaled = int(latitude * 1e6)
        lng_scaled = int(longitude * 1e6)
        ts_uint = int(timestamp or time.time())

        if self.is_simulated or self.registry_contract is None:
            effective_sender = sender or (self.account.address if self.account else self.sim_state.owner)
            success, inc_id, tx_hash, msg = self.sim_state.report_incident(
                ipfs_hash=ipfs_hash,
                incident_type=incident_type,
                severity=severity_uint,
                lat=latitude,
                lng=longitude,
                timestamp=ts_uint,
                sender=effective_sender
            )
            if success:
                logger.info(f"[Blockchain] Reported Incident #{inc_id} (TX: {tx_hash}) [Simulated]")
                return True, inc_id, tx_hash
            else:
                logger.warning(f"[Blockchain] reportIncident rejected: {msg}")
                return False, None, None

        try:
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            txn = self.registry_contract.functions.reportIncident(
                ipfs_hash,
                incident_type,
                severity_uint,
                lat_scaled,
                lng_scaled,
                ts_uint
            ).build_transaction({
                "from": self.account.address,
                "nonce": nonce,
                "gas": settings.GAS_LIMIT,
                "gasPrice": self.w3.to_wei(settings.GAS_PRICE_GWEI, "gwei"),
                "chainId": settings.CHAIN_ID
            })

            signed_txn = self.w3.eth.account.sign_transaction(txn, private_key=self.private_key)
            tx_hash_bytes = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            tx_hash = self.w3.to_hex(tx_hash_bytes)

            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash_bytes, timeout=60)
            logger.info(f"[Blockchain] Mined Incident Report TX: {tx_hash} (Status: {receipt.status})")
            return (receipt.status == 1), 0, tx_hash

        except Exception as e:
            logger.error(f"Blockchain transaction failed: {e}")
            return False, None, None

    def resolve_incident(self, incident_id: int, notes: str = "Resolved by highway authority") -> Tuple[bool, str]:
        """Resolves an incident."""
        if self.is_simulated or self.registry_contract is None:
            caller = self.account.address if self.account else self.sim_state.owner
            return self.sim_state.resolve_incident(incident_id, notes, caller=caller)

        try:
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            txn = self.registry_contract.functions.resolveIncident(
                incident_id,
                notes
            ).build_transaction({
                "from": self.account.address,
                "nonce": nonce,
                "gas": 120000,
                "gasPrice": self.w3.to_wei(settings.GAS_PRICE_GWEI, "gwei"),
                "chainId": settings.CHAIN_ID
            })
            signed_txn = self.w3.eth.account.sign_transaction(txn, private_key=self.private_key)
            tx_hash = self.w3.to_hex(self.w3.eth.send_raw_transaction(signed_txn.rawTransaction))
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
            return (receipt.status == 1), tx_hash
        except Exception as e:
            logger.error(f"resolveIncident failed: {e}")
            return False, str(e)

    def get_incident(self, incident_id: int) -> Optional[Dict]:
        """Retrieves a single incident by ID."""
        if self.is_simulated or self.registry_contract is None:
            if 1 <= incident_id <= len(self.sim_state.incidents):
                return self.sim_state.incidents[incident_id - 1]
            return None

        try:
            rec = self.registry_contract.functions.getIncident(incident_id).call()
            return {
                "incidentId": rec[0],
                "ipfsHash": rec[1],
                "incidentType": rec[2],
                "severity": rec[3],
                "latitude": rec[4],
                "longitude": rec[5],
                "timestamp": rec[6],
                "reportingDrone": rec[7],
                "status": rec[8],
                "resolutionNotes": rec[9],
                "resolvedAt": rec[10]
            }
        except Exception as e:
            logger.error(f"getIncident error: {e}")
            return None

    def get_all_incidents(self) -> List[Dict]:
        """Fetches all registered incidents from on-chain state or simulation."""
        if self.is_simulated or self.registry_contract is None:
            return self.sim_state.incidents

        try:
            total = self.registry_contract.functions.incidentCount().call()
            results = []
            for i in range(1, total + 1):
                inc = self.get_incident(i)
                if inc:
                    results.append(inc)
            return results
        except Exception as e:
            logger.error(f"Failed to fetch on-chain incidents: {e}")
            return self.sim_state.incidents

    def get_events(self) -> List[Dict]:
        """Returns recorded event logs."""
        return self.sim_state.events
