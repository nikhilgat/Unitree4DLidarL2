"""Batch driver.

Use from code (see main.py):   process_bags(bags, out, spec, cfg, ...)
or from the shell:             python -m calib_target.extract <rosbags dir | bag dir ...> --out results
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .bag_io import apply_range_model, read_bag
from .board import BoardSpec
from .detect import Config, extract_target
from .report import plot_detection
from .viz import board_mask, save_scene_png, save_scene_ply, scene_colors


def list_bags(paths, names=None):
    """Bag folders under `paths` (a rosbags root or bag folders), optionally filtered by name."""
    out = []
    for p in map(Path, paths):
        if p.is_dir() and not list(p.glob("*.mcap")):
            out += sorted((d for d in p.iterdir() if d.is_dir() and list(d.glob("*.mcap"))),
                          key=lambda d: (len(d.name), d.name))
        else:
            out.append(p)
    if names:
        out = [b for b in out if b.name in set(names)]
    return out


def _save_ply(path, pts, inten):
    g = np.clip(inten, 0, 255).astype(np.uint8)
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    a = np.zeros(len(pts), dt)
    a["x"], a["y"], a["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    a["r"] = a["g"] = a["b"] = g
    head = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
            "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % len(pts))
    with open(path, "wb") as f:
        f.write(head.encode())
        f.write(a.tobytes())


def process_bags(bags, out, spec=None, cfg=None, topic="/unilidar/cloud", frames=None,
                 range_offset=0.0, range_scale=1.0, save_scene=True, save_board_ply=True,
                 show=False, log=print):
    """Extract the checkerboard from every bag; write results to `out`; return the summary rows."""
    spec, cfg = spec or BoardSpec(), cfg or Config()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for bag in bags:
        bag = Path(bag)
        name = bag.parent.name if bag.suffix == ".mcap" else bag.name
        t = time.time()
        xyz, inten, used = read_bag(bag, topic, frames)
        if range_offset or range_scale != 1.0:
            xyz = apply_range_model(xyz, range_scale, range_offset)
        det = extract_target(xyz, inten, spec, cfg)
        d = out / name
        d.mkdir(exist_ok=True)
        if det is None:
            log(f"{name}: NO TARGET FOUND ({used} frames)")
            rows.append(dict(bag=name, frames=used, found=False))
            continue
        T = det.T_lidar_target
        rec = dict(bag=name, frames=used, found=True, accepted=bool(det.accepted), ncc=round(det.ncc, 3),
                   coverage=round(det.coverage, 2), square_mm=round(det.spec.square * 1000, 1),
                   scale=round(det.scale, 3), plane_rms_mm=round(det.plane_rms_mm, 1),
                   walk_mm=round(det.walk_mm, 1), n_points=det.n_points,
                   distance_m=round(float(np.linalg.norm(T[:3, 3])), 3), note=det.note)
        (d / "target.json").write_text(json.dumps(dict(
            rec, dark_first=det.dark_first,
            T_lidar_target=T.tolist(), T_lidar_target_alt=det.T_lidar_target_alt.tolist(),
            T_target_lidar=np.linalg.inv(T).tolist(),
            board=dict(cols=det.spec.cols, rows=det.spec.rows, square_m=det.spec.square),
            range_model=dict(offset_m=range_offset, scale=range_scale),
            walk_curve_intensity_mm=det.walk_curve.tolist()), indent=2))
        if save_board_ply:
            _save_ply(d / "board.ply", det.board_points, det.board_intensity)
        plot_detection(det, det.spec, d / "detection.png", name)
        if save_scene:
            mask = board_mask(xyz, inten, det)
            rgb = scene_colors(inten, mask)
            save_scene_ply(d / "scene.ply", xyz, rgb)
            save_scene_png(d / "scene.png", xyz, rgb, mask, det, title=name)
            rec["board_points_in_scene"] = int(mask.sum())
            if show:
                subprocess.Popen([sys.executable, "-m", "calib_target.view", str(d)],
                                 cwd=Path(__file__).resolve().parents[1])
        rows.append(rec)
        log(f"{name}: {'OK ' if det.accepted else 'LOW'} ncc={det.ncc:.2f} sq={det.spec.square*1000:.0f}mm "
            f"dist={rec['distance_m']}m rms={det.plane_rms_mm:.1f}mm pts={det.n_points} ({time.time()-t:.0f}s)")
    keys = sorted({k for r in rows for k in r}, key=lambda k: ["bag", "frames", "found", "accepted"].index(k)
                  if k in ("bag", "frames", "found", "accepted") else 9)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--out", default="results")
    ap.add_argument("--topic", default="/unilidar/cloud")
    ap.add_argument("--frames", type=int, default=None, help="max frames to accumulate per bag")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square", type=float, default=0.095, help="nominal square size (m)")
    ap.add_argument("--range-offset", type=float, default=0.0, help="delta (m): r' = (r - delta) / kappa")
    ap.add_argument("--range-scale", type=float, default=1.0, help="kappa in r' = (r - delta) / kappa")
    ap.add_argument("--show", action="store_true", help="open the interactive 3D viewer per bag")
    ap.add_argument("--no-scene", action="store_true", help="skip scene.ply / scene.png / viewer")
    a = ap.parse_args(argv)
    process_bags(list_bags(a.bags), a.out, BoardSpec(a.cols, a.rows, a.square), Config(), a.topic, a.frames,
                 a.range_offset, a.range_scale, save_scene=not a.no_scene, show=a.show)


if __name__ == "__main__":
    main()
