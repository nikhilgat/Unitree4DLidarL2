"""
Shared client for talking to the lidar_bridge relay (127.0.0.1:9899).

The relay (a small native C++ program running in WSL, built from
lidar_bridge.cpp) talks to the LiDAR over UDP using Unitree's SDK and
re-broadcasts each parsed point cloud frame over this local TCP socket.

Wire format per frame, little-endian:
    4 bytes  magic "PCLD"
    8 bytes  double stamp
    4 bytes  uint32 num_points (N)
    N * 16 bytes  float32 x, y, z, intensity
"""

import socket
import struct
import numpy as np

HOST = "127.0.0.1"
PORT = 9899
HEADER = struct.Struct("<4sdI")  # magic, stamp, num_points


def connect(timeout=5.0):
    sock = socket.create_connection((HOST, PORT), timeout=timeout)
    sock.settimeout(None)
    return sock


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("lidar_bridge closed the connection")
        buf.extend(chunk)
    return bytes(buf)


def read_frame(sock):
    """Read one point cloud frame. Returns (stamp, xyz, intensity)."""
    header = _recv_exact(sock, HEADER.size)
    magic, stamp, n = HEADER.unpack(header)
    if magic != b"PCLD":
        raise ValueError(f"bad frame magic: {magic!r} (relay/client out of sync)")

    payload = _recv_exact(sock, n * 16)
    points = np.frombuffer(payload, dtype="<f4").reshape(-1, 4)
    xyz = points[:, :3].copy()
    intensity = points[:, 3].copy()
    return stamp, xyz, intensity
