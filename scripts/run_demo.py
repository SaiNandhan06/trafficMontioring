"""
One-Command End-to-End UAV Traffic AI + Blockchain Demonstration Runner.
Compiles smart contracts, generates synthetic UAV video, runs Edge AI inference,
pins incident evidence to IPFS, records on-chain, and displays full audit trail.
"""

import time
import sys
import subprocess
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from config.logging_config import setup_logger
from data.synthetic_generator import generate_synthetic_uav_dataset
from blockchain.compile import compile_solidity_contracts
from blockchain.deploy import deploy_contracts
from edge.edge_pipeline import EdgeInferencePipeline
from blockchain.contract_client import Web3ContractClient
from ipfs.ipfs_client import IPFSClient

logger = setup_logger("run_demo")


def run_full_demo(synthetic_frames: int = 80, launch_dashboard: bool = False):
    print("\n" + "=" * 80)
    print(" [SKYGUARD UAV: EDGE AI + BLOCKCHAIN TRAFFIC MONITORING PLATFORM] ")
    print("=" * 80)

    # 1. Generate Synthetic Video & Ground Truth Annotations
    video_path = settings.DATA_DIR / "sample_drone_feed.mp4"
    if not video_path.exists():
        logger.info("[STEP 1/4] Generating synthetic aerial drone footage and YOLO annotations...")
        generate_synthetic_uav_dataset(
            output_dir=settings.DATA_DIR / "processed",
            num_frames=synthetic_frames,
            generate_video=True,
            video_path=video_path,
            fps=25
        )
    else:
        logger.info(f"[STEP 1/4] Synthetic video already available at {video_path}")

    # 2. Compile & Deploy Smart Contracts
    logger.info("[STEP 2/4] Compiling and deploying Solidity contracts (TrafficIncidentRegistry & EmergencyNotificationService)...")
    compile_solidity_contracts()
    reg_addr, emg_addr = deploy_contracts(network="local")
    logger.info(f"Deployed TrafficIncidentRegistry at: {reg_addr}")
    logger.info(f"Deployed EmergencyNotificationService at: {emg_addr}")

    # 3. Execute Edge AI Inference Pipeline on Drone Feed
    logger.info("[STEP 3/4] Starting Edge AI Inference Engine (YOLOv8 + ByteTrack + IPFS + Web3)...")
    pipeline = EdgeInferencePipeline(
        source=str(video_path),
        drone_id=settings.DRONE_ID,
        display=False
    )
    # Process frames through the video
    pipeline.run(max_frames=synthetic_frames)

    # 4. Display Verification & On-Chain Audit Summary
    logger.info("[STEP 4/4] Fetching on-chain verified records & IPFS audit package...")
    incidents = pipeline.blockchain_client.get_all_incidents()

    print("\n" + "=" * 80)
    print(" [END-TO-END VERIFICATION & AUDIT TRAIL SUMMARY] ")
    print("=" * 80)
    print(f"Total Video Frames Processed:   {pipeline.processed_frames}")
    print(f"Incidents Detected & Reported:  {len(incidents)}")
    print("-" * 80)
    print(f"{'ID':<6} | {'Type':<20} | {'Severity':<10} | {'IPFS Master CID':<22} | {'TX Hash':<18}")
    print("-" * 80)
    for inc in incidents:
        inc_id = inc.get("incidentId", 1)
        inc_type = inc.get("incidentType", "UNKNOWN")
        sev = inc.get("severity", 1)
        sev_str = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}.get(sev, str(sev))
        cid = inc.get("ipfsHash", "N/A")
        tx = inc.get("txHash", "0x" + "0" * 16)
        print(f"#{inc_id:<5} | {inc_type:<20} | {sev_str:<10} | {cid[:20]}.. | {tx[:16]}..")
    print("=" * 80 + "\n")

    if launch_dashboard:
        logger.info("Launching Streamlit Verification Dashboard on http://localhost:8501 ...")
        subprocess.run(["streamlit", "run", "dashboard/app.py"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full End-to-End UAV Traffic AI Demo")
    parser.add_argument("--frames", type=int, default=80, help="Number of frames to process in demo")
    parser.add_argument("--launch-dashboard", action="store_true", help="Launch Streamlit dashboard after run")
    args = parser.parse_args()

    run_full_demo(synthetic_frames=args.frames, launch_dashboard=args.launch_dashboard)
