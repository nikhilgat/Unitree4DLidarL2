"""
Capture N seconds of LiDAR point cloud data and save it as a single .ply
file, ready to use as a SMCODE scan (data/scans/<name>.ply).

Prereq: the relay must be running (see start_bridge.ps1).

Usage:
    python save_scan.py --name 203
    python save_scan.py --name 203 --seconds 20
    python save_scan.py --name 203 --out-dir "../native_bridge/outputs"
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from lidar_client import connect, read_frame

DEFAULT_OUT_DIR = Path(r"../native_bridge/outputs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="output file name, without extension")
    parser.add_argument("--seconds", type=float, default=10.0, help="how long to capture (default: 10)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="output directory")
    args = parser.parse_args()

    try:
        sock = connect()
    except (ConnectionRefusedError, TimeoutError, OSError) as e:
        print(f"Could not connect to lidar_bridge on 127.0.0.1:9899 ({e}).")
        print("Start it first: .\\start_bridge.ps1")
        sys.exit(1)

    print(f"Capturing for {args.seconds:.1f} seconds...")
    chunks = []
    n_frames = 0
    deadline = time.time() + args.seconds
    try:
        while time.time() < deadline:
            _, xyz, _ = read_frame(sock)
            if xyz.shape[0] > 0:
                chunks.append(xyz)
                n_frames += 1
    finally:
        sock.close()

    if not chunks:
        print("No points captured. Is the sensor connected and lidar_bridge actually receiving data?")
        sys.exit(1)

    points = np.concatenate(chunks, axis=0)
    print(f"Captured {n_frames} frames, {points.shape[0]} points total.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.name}.ply"

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(str(out_path), pcd)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
