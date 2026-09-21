"""Geometry of the calibration board and its intensity template."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BoardSpec:
    """A checkerboard printed on an A1 sheet.

    `cols` x `rows` squares of `square` metres, centred on a `sheet` = (long, short) metre sheet.
    Target frame: origin at the pattern centre, x along the `cols` direction, y along the `rows`
    direction, z = x cross y (the pipeline orients it toward the sensor).
    """
    cols: int = 8
    rows: int = 6
    square: float = 0.095
    sheet: tuple = (0.841, 0.594)

    @property
    def pattern(self):
        return self.cols * self.square, self.rows * self.square

    def scaled(self, k):
        """Same board with every dimension multiplied by `k`."""
        return BoardSpec(self.cols, self.rows, self.square * k, (self.sheet[0] * k, self.sheet[1] * k))

    def template(self, px, dark_first=True, with_margin=True):
        """Return (template, mask) at `px` metres/pixel, +1 = light, -1 = dark.

        With `with_margin` the white sheet border around the pattern is included and the mask
        covers the whole sheet, otherwise only the pattern.
        """
        pw, ph = self.pattern
        w, h = self.sheet if with_margin else self.pattern
        W, H = int(round(w / px)), int(round(h / px))
        xs = (np.arange(W) + 0.5) * px - w / 2
        ys = (np.arange(H) + 0.5) * px - h / 2
        X, Y = np.meshgrid(xs, ys)
        ci = np.floor((X + pw / 2) / self.square).astype(int)
        ri = np.floor((Y + ph / 2) / self.square).astype(int)
        inside = (ci >= 0) & (ci < self.cols) & (ri >= 0) & (ri < self.rows)
        parity = ((ci + ri) % 2 == 0)
        t = np.where(inside, np.where(parity == dark_first, -1.0, 1.0), 1.0).astype(np.float32)
        return t, np.ones_like(t)

    def corners_xy(self, with_margin=False):
        """Outline of the pattern (or sheet) in the target frame, closed polygon (5,2)."""
        w, h = self.sheet if with_margin else self.pattern
        return np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2], [-w / 2, -h / 2]])

    def inner_corners_xy(self):
        """(rows-1)*(cols-1) inner checker corners in the target frame, row-major."""
        pw, ph = self.pattern
        xs = -pw / 2 + self.square * np.arange(1, self.cols)
        ys = -ph / 2 + self.square * np.arange(1, self.rows)
        X, Y = np.meshgrid(xs, ys)
        return np.column_stack([X.ravel(), Y.ravel()])


def rotate_template(t, m, angle_deg):
    """Rotate a template+mask by angle (deg, counter-clockwise) onto a tight canvas."""
    h, w = t.shape
    a = np.radians(angle_deg)
    cw = int(np.ceil(abs(w * np.cos(a)) + abs(h * np.sin(a))))
    ch = int(np.ceil(abs(w * np.sin(a)) + abs(h * np.cos(a))))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    M[0, 2] += cw / 2 - w / 2
    M[1, 2] += ch / 2 - h / 2
    tr = cv2.warpAffine(t, M, (cw, ch), flags=cv2.INTER_LINEAR, borderValue=0)
    mr = cv2.warpAffine(m, M, (cw, ch), flags=cv2.INTER_NEAREST, borderValue=0)
    return tr, (mr > 0.5).astype(np.float32)
