"""3D verification views: whole cloud in grey, only the detected checkerboard in colour.

  python -m calib_target.view results/my_lidar_bag6          # interactive Open3D window
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .board import BoardSpec
from .detect import _apply_walk, _inside

GREY = 140


def board_mask(xyz, inten, det):
    """Boolean mask over the raw scene points that belong to the detected board.

    Uses the detection's walk curve so that dark squares (measured nearer than the white ones)
    are still attributed to the board.
    """
    T = det.T_lidar_target
    o, n, R = T[:3, 3], T[:3, 2], T[:3, :3]
    cand = np.nonzero((np.abs((xyz - o) @ n) < 0.2) & (np.linalg.norm(xyz - o, axis=1) < 1.3)
                      & (inten >= 100))[0]
    P, I = xyz[cand], inten[cand]
    Pc, ok = _apply_walk(P, I, (det.walk_curve[:, 0], det.walk_curve[:, 1] / 1000.0), n)
    q = (Pc - o) @ R
    keep = ok & (np.abs(q[:, 2]) < 0.03) & _inside(q[:, :2], det.spec, margin=0.01)
    mask = np.zeros(len(xyz), bool)
    mask[cand[keep]] = True
    return mask


def scene_colors(inten, mask):
    """Grey everywhere, turbo-coloured intensity on the board."""
    rgb = np.full((len(inten), 3), GREY, np.uint8)
    t = np.clip((inten[mask] - 100.0) / 155.0, 0, 1)
    rgb[mask] = (plt.get_cmap("turbo")(t)[:, :3] * 255).astype(np.uint8)
    return rgb


def save_scene_ply(path, xyz, rgb):
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    a = np.empty(len(xyz), dt)
    a["x"], a["y"], a["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    a["r"], a["g"], a["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    head = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
            "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
            % len(xyz))
    with open(path, "wb") as f:
        f.write(head.encode())
        f.write(a.tobytes())


def _outline_world(T, spec):
    xy = spec.corners_xy()
    loc = np.column_stack([xy, np.zeros(len(xy))])
    return loc @ T[:3, :3].T + T[:3, 3]


def save_scene_png(path, xyz, rgb, mask, det, max_grey=60000, title=""):
    """Static views (looking from the sensor at the board, and a zoom on the board)."""
    rng = np.random.default_rng(0)
    grey = np.nonzero(~mask)[0]
    grey = grey[rng.permutation(len(grey))[:max_grey]]
    T = det.T_lidar_target
    o = T[:3, 3]
    out = _outline_world(T, det.spec)
    fig = plt.figure(figsize=(18, 8))
    for k, (zoom, name) in enumerate(((None, "whole cloud"), (0.9, "board close-up"))):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        g, b = xyz[grey], xyz[mask]
        if zoom:
            near = np.linalg.norm(g - o, axis=1) < 1.4
            g = g[near]
        ax.scatter(g[:, 0], g[:, 1], g[:, 2], c="#8c8c8c", s=1.0 if zoom else 0.3, alpha=0.35,
                   depthshade=False)
        ax.scatter(b[::4, 0], b[::4, 1], b[::4, 2], c=rgb[mask][::4] / 255.0, s=1.5, depthshade=False)
        ax.plot(out[:, 0], out[:, 1], out[:, 2], "k-", lw=1.5)
        for i, col in enumerate("rgb"):
            ax.plot(*np.column_stack([o, o + 0.25 * T[:3, i]]), col, lw=2)
        ax.scatter(0, 0, 0, c="k", marker="^", s=60)               # sensor origin
        pts = np.vstack([g, b, [[0, 0, 0]]]) if not zoom else np.vstack([g, b])
        c, r = pts.mean(0), np.ptp(pts, axis=0).max() / 2
        ax.set_xlim(c[0] - r, c[0] + r), ax.set_ylim(c[1] - r, c[1] + r), ax.set_zlim(c[2] - r, c[2] + r)
        ax.view_init(elev=25, azim=-60)
        ax.set_title(name)
    fig.suptitle(f"{title}  board points coloured by intensity, everything else grey  "
                 f"(triangle = sensor, axes: x=red y=green z=blue)")
    plt.tight_layout()
    plt.savefig(path, dpi=60)
    plt.close(fig)


def show(result_dir, point_size=2.0, max_grey_voxel=0.01):
    """Interactive Open3D viewer for a results/<bag> folder (needs scene.ply + target.json)."""
    import open3d as o3d

    d = Path(result_dir)
    info = json.loads((d / "target.json").read_text())
    T = np.array(info["T_lidar_target"])
    spec = BoardSpec(info["board"]["cols"], info["board"]["rows"], info["board"]["square_m"])
    pcd = o3d.io.read_point_cloud(str(d / "scene.ply"))
    rgb = np.asarray(pcd.colors)
    is_board = np.abs(rgb[:, 0] - rgb[:, 1]) + np.abs(rgb[:, 1] - rgb[:, 2]) > 0.05   # non-grey
    xyz = np.asarray(pcd.points)
    # thin the grey background for speed, keep every board point
    grey = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[~is_board]))
    grey.colors = o3d.utility.Vector3dVector(rgb[~is_board])
    grey = grey.voxel_down_sample(max_grey_voxel)
    board = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[is_board]))
    board.colors = o3d.utility.Vector3dVector(rgb[is_board])

    out = _outline_world(T, spec)
    lines = o3d.geometry.LineSet(o3d.utility.Vector3dVector(out),
                                 o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(out) - 1)]))
    lines.paint_uniform_color([1, 1, 1])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    frame.transform(T)
    sensor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)

    vis = o3d.visualization.Visualizer()
    if not vis.create_window(f"{d.name}  |  grey = scene, colour = detected checkerboard", 1400, 900):
        print(f"[view] could not open a 3D window (no OpenGL display?). "
              f"Open {d / 'scene.png'} or {d / 'scene.ply'} (CloudCompare) instead.")
        return
    for g in (grey, board, lines, frame, sensor):
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array([0.06, 0.06, 0.08])
    ctr = vis.get_view_control()
    o = T[:3, 3]
    ctr.set_lookat(o)
    ctr.set_front(-o / np.linalg.norm(o))          # look from the sensor toward the board
    ctr.set_up(T[:3, 1])
    ctr.set_zoom(0.3)
    vis.run()
    vis.destroy_window()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result_dir", help="results/<bag> folder written by calib_target.extract")
    ap.add_argument("--point-size", type=float, default=2.0)
    a = ap.parse_args(argv)
    show(a.result_dir, a.point_size)


if __name__ == "__main__":
    main()
