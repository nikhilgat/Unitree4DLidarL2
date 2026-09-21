"""Extract the calibration checkerboard from every rosbag.  Edit the settings below, then press F5
(or run `python main.py`).  No windows are opened; everything is written to OUT_DIR.

Per bag, in OUT_DIR/<bag>/:
    target.json   T_lidar_target (target -> lidar), its inverse, the 180-degree alternative, metrics
    scene.ply     whole point cloud in grey, ONLY the detected checkerboard coloured (open anywhere)
    scene.png     static 3D views of the same
    board.ply     just the board points
    detection.png checker fit diagnostics
OUT_DIR/summary.csv lists every bag (check the `accepted` column and `note`).
"""
from pathlib import Path

from calib_target.board import BoardSpec
from calib_target.detect import Config
from calib_target.extract import list_bags, process_bags

# ----------------------------------------------------------------------------- INPUT / OUTPUT
ROSBAGS_DIR = r"C:\Users\nikhi\Documents\SMART-MARK\Unitree4DLidarL2\rosbags"   # folder of bag folders
BAG_NAMES = None            # None = every bag, or e.g. ["my_lidar_bag3", "my_lidar_bag6"]
OUT_DIR = Path(__file__).resolve().parent / "results"
TOPIC = "/unilidar/cloud"
FRAMES = None               # max frames accumulated per bag (None = all; fewer = faster, sparser)

# ----------------------------------------------------------------------------- CHECKERBOARD
COLS = 8                    # squares along the long side
ROWS = 6                    # squares along the short side
SQUARE_M = 0.095            # printed square size in metres

# ----------------------------------------------------------------------------- RANGE CORRECTION
# The L2 cloud shows a distance-dependent scale error (board looks ~100-120 mm instead of 95 mm).
# r' = (r - RANGE_OFFSET_M) / RANGE_SCALE was fitted to the board size (delta=0.245, kappa=0.956).
# It is NOT independently verified - tape-measure a sensor-to-board distance before trusting it.
USE_RANGE_CORRECTION = False
RANGE_OFFSET_M = 0.245
RANGE_SCALE = 0.956

# ----------------------------------------------------------------------------- OUTPUT FILES
SAVE_SCENE = True           # scene.ply + scene.png (grey cloud, coloured board)
SAVE_BOARD_PLY = True       # board.ply

# ----------------------------------------------------------------------------- DETECTOR TUNING
CONFIG_OVERRIDES = dict(
    # min_ncc=0.4,           # accept threshold on the checker match (lower = more permissive)
    # min_seg_points=400,    # smallest planar segment considered as a board candidate
    # max_range=6.0,         # ignore candidates further than this (m)
    # gather_radius=0.9,     # region around a candidate that is searched (m)
    # theta_step=3.0,        # coarse rotation step (deg): smaller = slower, more thorough
)


def main():
    spec = BoardSpec(COLS, ROWS, SQUARE_M)
    cfg = Config(**CONFIG_OVERRIDES)
    bags = list_bags([ROSBAGS_DIR], BAG_NAMES)
    if not bags:
        raise SystemExit(f"No bags found in {ROSBAGS_DIR}")
    print(f"{len(bags)} bag(s) -> {OUT_DIR}")
    rows = process_bags(
        bags, OUT_DIR, spec, cfg, TOPIC, FRAMES,
        range_offset=RANGE_OFFSET_M if USE_RANGE_CORRECTION else 0.0,
        range_scale=RANGE_SCALE if USE_RANGE_CORRECTION else 1.0,
        save_scene=SAVE_SCENE, save_board_ply=SAVE_BOARD_PLY, show=False)
    ok = sum(bool(r.get("accepted")) for r in rows)
    print(f"\nDone: {ok}/{len(rows)} accepted. Results: {OUT_DIR}\\summary.csv")


if __name__ == "__main__":
    main()
