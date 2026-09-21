"""Read Unitree L2 PointCloud2 messages out of a rosbag2 (mcap) and accumulate them."""
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

_DTYPES = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
           5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def find_mcap(bag_dir):
    bag_dir = Path(bag_dir)
    if bag_dir.suffix == ".mcap":
        return bag_dir
    files = sorted(bag_dir.glob("*.mcap"))
    if not files:
        raise FileNotFoundError(f"no .mcap in {bag_dir}")
    return files[0]


def unpack_pointcloud2(msg):
    """Return (N,3) float32 xyz and a dict of the other scalar fields, NaNs dropped."""
    step = msg.point_step
    n = len(msg.data) // step
    if n == 0:
        return np.empty((0, 3), np.float32), {}
    rec = np.frombuffer(bytes(msg.data)[: n * step], np.uint8).reshape(n, step)
    cols = {}
    for f in msg.fields:
        dt = np.dtype(_DTYPES[f.datatype])
        cols[f.name] = rec[:, f.offset:f.offset + dt.itemsize].copy().view(dt).ravel()
    xyz = np.column_stack([cols.pop(k).astype(np.float32) for k in "xyz"])
    ok = np.isfinite(xyz).all(1)
    return xyz[ok], {k: v[ok] for k, v in cols.items()}


def read_bag(bag, topic="/unilidar/cloud", max_frames=None, skip_frames=0):
    """Accumulate frames of a static scene into one cloud.

    Returns xyz (N,3) float32, intensity (N,) float32 and the number of frames used.
    """
    xyz, inten, used = [], [], 0
    with open(find_mcap(bag), "rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        for i, (_, ch, _, msg) in enumerate(reader.iter_decoded_messages(topics=[topic])):
            if i < skip_frames:
                continue
            if max_frames is not None and used >= max_frames:
                break
            p, sc = unpack_pointcloud2(msg)
            if len(p):
                xyz.append(p)
                inten.append(sc.get("intensity", np.zeros(len(p))).astype(np.float32))
                used += 1
    return np.vstack(xyz), np.concatenate(inten), used


def apply_range_model(xyz, kappa=1.0, delta=0.0):
    """Map measured to corrected points along each ray: r' = (r - delta) / kappa."""
    r = np.linalg.norm(xyz, axis=1, keepdims=True)
    return (xyz / r * ((r - delta) / kappa)).astype(np.float32)
