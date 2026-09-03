"""
Live 3D visualization of the LiDAR point cloud, reading from the
lidar_bridge relay. No Docker, ROS2, or RViz needed.

Keeps a short rolling window of recent frames overlaid (like RViz's
"Decay Time" display option) instead of showing only the single latest
frame -- a single frame from this sensor is fairly sparse (~5000 points,
~cloud_scan_num=18 sub-scans), so overlaying the last ~0.6s of frames
gives a denser, more RViz-like view. The sensor's own internal
accumulation window is what causes the inherent quarter-to-half-second
lag you'll see when moving something in front of it -- that's on the
sensor/driver side, not something the viewer controls.

Prereq: the pipeline must be running (see start_lidar.ps1).

Usage:
    python visualize_lidar.py
    python visualize_lidar.py --decay 1.0   # keep more history overlaid
"""

import argparse
import collections
import sys
import time

import numpy as np
import open3d as o3d

from lidar_client import connect, read_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decay", type=float, default=0.6,
                         help="seconds of recent frames to keep overlaid (default: 0.6)")
    args = parser.parse_args()

    try:
        sock = connect()
    except (ConnectionRefusedError, TimeoutError, OSError) as e:
        print(f"Could not connect to lidar_bridge on 127.0.0.1:9899 ({e}).")
        print("Start it first: .\\start_lidar.ps1")
        sys.exit(1)

    print("Connected to lidar_bridge. Opening viewer...")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR Live View", width=1024, height=768)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    vis.add_geometry(pcd)

    render_opt = vis.get_render_option()
    render_opt.point_size = 2.0
    render_opt.background_color = np.array([0.05, 0.05, 0.05])

    # Rolling buffer of (arrival_time, xyz) for recent frames.
    buffer = collections.deque()

    first_frame = True
    try:
        while True:
            _, xyz, _ = read_frame(sock)
            now = time.time()
            if xyz.shape[0] > 0:
                buffer.append((now, xyz))

            cutoff = now - args.decay
            while buffer and buffer[0][0] < cutoff:
                buffer.popleft()

            if not buffer:
                continue

            all_xyz = np.concatenate([b[1] for b in buffer], axis=0)
            pcd.points = o3d.utility.Vector3dVector(all_xyz)

            # Color by height (z) so structure is easy to read visually.
            z = all_xyz[:, 2]
            z_range = max(z.max() - z.min(), 1e-6)
            norm = (z - z.min()) / z_range
            colors = np.stack([norm, 0.4 * np.ones_like(norm), 1.0 - norm], axis=1)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            vis.update_geometry(pcd)
            if first_frame:
                vis.reset_view_point(True)
                first_frame = False

            if not vis.poll_events():
                break
            vis.update_renderer()
    except KeyboardInterrupt:
        pass
    finally:
        vis.destroy_window()
        sock.close()


if __name__ == "__main__":
    main()
