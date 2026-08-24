"""
Multi-UAV Fleet Simulation Engine.
Simulates concurrent UAV drone patrol streams across multiple GPS coordinates,
generating realistic traffic anomalies and broadcasting on-chain incident transactions.
"""

import time
import sys
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from concurrent.futures import ThreadPoolExecutor
from typing import Dict
import cv2
import numpy as np
from config.settings import settings
from config.logging_config import setup_logger
from ipfs.ipfs_client import IPFSClient
from ipfs.metadata_builder import build_incident_metadata
from blockchain.contract_client import Web3ContractClient

logger = setup_logger("fleet_simulator")

INCIDENT_TYPES = ["SPEEDING", "COLLISION_ACCIDENT", "LANE_VIOLATION", "TRAFFIC_CONGESTION"]
SEVERITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def simulate_single_drone(
    drone_idx: int,
    drone_id: str,
    base_lat: float,
    base_lng: float,
    num_incidents: int,
    ipfs_client: IPFSClient,
    bc_client: Web3ContractClient
):
    """Simulates a single UAV patrol node generating and reporting incidents."""
    logger.info(f"Starting Drone [{drone_id}] patrol loop...")

    for i in range(num_incidents):
        time.sleep(random.uniform(0.5, 1.5))

        inc_type = random.choice(INCIDENT_TYPES)
        severity = random.choice(SEVERITIES)
        lat = base_lat + random.uniform(-0.015, 0.015)
        lng = base_lng + random.uniform(-0.015, 0.015)
        conf = random.uniform(0.85, 0.98)

        # Generate synthetic evidence frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (40, 40, 40)
        cv2.rectangle(frame, (100, 100), (540, 380), (70, 70, 70), -1)
        cv2.putText(
            frame,
            f"{drone_id} EVIDENCE: {inc_type}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 100),
            2
        )

        inc_id = f"INC-SIM-{drone_idx}-{i}-{int(time.time())}"

        # 1. Package Metadata & Frame to IPFS
        meta, frame_bytes = build_incident_metadata(
            incident_id=inc_id,
            drone_id=drone_id,
            incident_type=inc_type,
            severity=severity,
            confidence=conf,
            location={"lat": lat, "lng": lng},
            involved_tracks=[random.randint(1, 20), random.randint(21, 40)],
            description=f"Automated fleet detection: {inc_type} at coordinates ({lat:.4f}, {lng:.4f})",
            frame=frame,
            detections=[{"bbox": [120, 120, 240, 280]}]
        )

        master_cid = ipfs_client.upload_incident_package(meta, frame_bytes)

        # 2. Publish to Blockchain
        success, onchain_id, tx_hash = bc_client.report_incident(
            ipfs_hash=master_cid,
            incident_type=inc_type,
            severity_str=severity,
            latitude=lat,
            longitude=lng,
            timestamp=time.time()
        )

        logger.info(
            f"[{drone_id}] Reported Incident #{onchain_id} | Type: {inc_type} | "
            f"CID: {master_cid[:16]}... | TX: {tx_hash[:16]}..."
        )

    logger.info(f"Drone [{drone_id}] patrol simulation complete.")


def run_fleet_simulation(num_drones: int = 4, incidents_per_drone: int = 3):
    """Orchestrates multi-drone fleet simulation."""
    logger.info(f"Initiating Fleet Simulation with {num_drones} UAV nodes...")

    ipfs_client = IPFSClient()
    bc_client = Web3ContractClient()

    with ThreadPoolExecutor(max_workers=num_drones) as executor:
        futures = []
        for d in range(num_drones):
            drone_id = f"UAV-FLEET-NODE-{d+1:02d}"
            f = executor.submit(
                simulate_single_drone,
                d + 1,
                drone_id,
                settings.DRONE_LAT,
                settings.DRONE_LNG,
                incidents_per_drone,
                ipfs_client,
                bc_client
            )
            futures.append(f)

        for f in futures:
            f.result()

    logger.info("All UAV fleet patrol missions concluded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Multi-UAV Drone Fleet")
    parser.add_argument("--drones", type=int, default=4, help="Number of concurrent UAV drones")
    parser.add_argument("--incidents", type=int, default=3, help="Incidents generated per drone")
    args = parser.parse_args()

    run_fleet_simulation(num_drones=args.drones, incidents_per_drone=args.incidents)
