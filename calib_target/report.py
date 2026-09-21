"""Diagnostic figures for a Detection."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def to_target_frame(det, pts):
    T = det.T_lidar_target
    return (pts - T[:3, 3]) @ T[:3, :3]


def plot_detection(det, spec, path, title=""):
    """Left: on-sheet points in the target frame with the expected grid. Right: walk curve."""
    q = to_target_frame(det, det.board_points)
    fig, axs = plt.subplots(1, 3, figsize=(21, 7), gridspec_kw=dict(width_ratios=[2, 2, 1.2]))
    axs[0].scatter(q[:, 0], q[:, 1], c=det.board_intensity, s=2, cmap="gray", vmin=100, vmax=255)
    axs[1].scatter(q[:, 0], q[:, 1], c=np.clip(q[:, 2] * 1000, -30, 30), s=2, cmap="coolwarm")
    for ax, t in zip(axs[:2], ("intensity + expected pattern", "plane residual (mm, walk-corrected)")):
        for poly, col in ((spec.corners_xy(), "lime"), (spec.corners_xy(True), "cyan")):
            ax.plot(poly[:, 0], poly[:, 1], col, lw=1.5)
        ic = spec.inner_corners_xy()
        ax.plot(ic[:, 0], ic[:, 1], "r.", ms=5)
        ax.set_aspect("equal")
        ax.set_title(t)
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.5, 0.5)
    axs[2].plot(det.walk_curve[:, 0], det.walk_curve[:, 1], "o-")
    axs[2].set_xlabel("intensity")
    axs[2].set_ylabel("offset toward sensor (mm)")
    axs[2].set_title("range-walk curve")
    fig.suptitle(f"{title}  NCC={det.ncc:.3f} cov={det.coverage:.2f} pts={det.n_points} "
                 f"rms={det.plane_rms_mm:.1f}mm dark_first={det.dark_first}")
    plt.tight_layout()
    plt.savefig(path, dpi=50)
    plt.close(fig)
