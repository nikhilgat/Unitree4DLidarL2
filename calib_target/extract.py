"""Batch driver:  python -m calib_target.extract <rosbags dir | bag dir ...> --out results"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .bag_io import apply_range_model, find_mcap, read_bag
from .board import BoardSpec
from .detect import Config, extract_target
from .report import plot_detection
from .viz import board_mask, save_scene_png, save_scene_ply, scene_colors


def _bags(paths):
    out = []
    for p in map(Path, paths):
        if p.is_dir() and not list(p.glob("*.mcap")):
            out += sorted((d for d in p.iterdir() if d.is_dir() and list(d.glob("*.mcap"))),
                          key=lambda d: (len(d.name), d.name))
        else:
            out.append(p)
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
    ap.add_argument("--no-show", action="store_true", help="do not open the interactive 3D viewer per bag")
    ap.add_argument("--no-scene", action="store_true", help="skip scene.ply / scene.png / viewer (faster, smaller)")
    args = ap.parse_args(argv)

    spec = BoardSpec(args.cols, args.rows, args.square)
    cfg = Config()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for bag in _bags(args.bags):
        name = bag.parent.name if bag.suffix == ".mcap" else bag.name
        t = time.time()
        xyz, inten, used = read_bag(bag, args.topic, args.frames)
        if args.range_offset or args.range_scale != 1.0:
            xyz = apply_range_model(xyz, args.range_scale, args.range_offset)
        det = extract_target(xyz, inten, spec, cfg)
        d = out / name
        d.mkdir(exist_ok=True)
        if det is None:
            print(f"{name}: NO TARGET FOUND ({used} frames)")
            rows.append(dict(bag=name, frames=used, found=False))
            continue
        T = det.T_lidar_target
        rec = dict(bag=name, frames=used, found=True, accepted=bool(det.accepted), ncc=round(det.ncc, 3),
                   coverage=round(det.coverage, 2), square_mm=round(det.spec.square * 1000, 1),
                   scale=round(det.scale, 3), plane_rms_mm=round(det.plane_rms_mm, 1),
                   walk_mm=round(det.walk_mm, 1), n_points=det.n_points,
                   distance_m=round(float(np.linalg.norm(T[:3, 3])), 3))
        (d / "target.json").write_text(json.dumps(dict(
            rec, note=det.note, dark_first=det.dark_first,
            T_lidar_target=T.tolist(), T_lidar_target_alt=det.T_lidar_target_alt.tolist(),
            T_target_lidar=np.linalg.inv(T).tolist(),
            board=dict(cols=det.spec.cols, rows=det.spec.rows, square_m=det.spec.square),
            walk_curve_intensity_mm=det.walk_curve.tolist()), indent=2))
        _save_ply(d / "board.ply", det.board_points, det.board_intensity)
        plot_detection(det, det.spec, d / "detection.png", name)
        if not args.no_scene:
            mask = board_mask(xyz, inten, det)
            rgb = scene_colors(inten, mask)
            save_scene_ply(d / "scene.ply", xyz, rgb)
            save_scene_png(d / "scene.png", xyz, rgb, mask, det, title=name)
            rec["board_points_in_scene"] = int(mask.sum())
        rows.append(rec)
        if not args.no_show and not args.no_scene:
            # separate process so extraction keeps running while the window is open
            subprocess.Popen([sys.executable, "-m", "calib_target.view", str(d)],
                             cwd=Path(__file__).resolve().parents[1])
        print(f"{name}: {'OK ' if det.accepted else 'LOW'} ncc={det.ncc:.2f} sq={det.spec.square*1000:.0f}mm "
              f"dist={rec['distance_m']}m rms={det.plane_rms_mm:.1f}mm pts={det.n_points} ({time.time()-t:.0f}s)")
    keys = sorted({k for r in rows for k in r}, key=lambda k: ["bag", "frames", "found", "accepted"].index(k)
                  if k in ("bag", "frames", "found", "accepted") else 9)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
