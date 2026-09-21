"""Browse the extraction results in one window.  Press F5 (or `python view_results.py`) after main.py.

Left: list of all bags (click to load).  The whole point cloud is grey, only the detected
checkerboard is coloured; toggle the grey scene, the outline/axes and the point size on the left.
Mouse: left-drag rotate, wheel zoom, right-drag / shift-drag pan.  "Reset view" looks from just
behind the sensor toward the board.
"""
from pathlib import Path

from calib_target.gui import run

RESULTS_DIR = Path(__file__).resolve().parent / "results"     # the OUT_DIR used by main.py
POINT_SIZE = 3.0        # pixels
GREY_VOXEL = 0.01       # metres; thins the grey background for speed (0.005 = denser, slower)

if __name__ == "__main__":
    run(RESULTS_DIR, POINT_SIZE, GREY_VOXEL)
