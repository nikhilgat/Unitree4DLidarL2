"""Find the checkerboard in an accumulated LiDAR cloud and estimate T_lidar_target.

Stages
  1. propose  - planar segments (normal region growing) whose size could be the A1 sheet
  2. verify   - plane from the saturated (white) returns, intensity-dependent range-walk
                correction, then masked-NCC match of the checker template on the intensity
                image in the plane frame (rotation x translation x colour polarity)
  3. refine   - continuous (rotation, translation) fit on the raw points, then a final
                plane fit on the walk-corrected points that lie on the sheet

The L2 measures dark returns several centimetres nearer than white ones on the same plane, so
the reference surface is the white (saturated, intensity 255) squares.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import minimize

from .board import BoardSpec, rotate_template
from .segment import estimate_normals, planar_segments, voxelize

WHITE_I = 254.0            # intensity at/above which a return is treated as saturated white
_WALK_EDGES = np.array([100, 115, 130, 145, 160, 175, 190, 205, 220, 235, 248, 254, 256], float)


@dataclass
class Config:
    voxel: float = 0.01               # segmentation voxel size (m)
    seg_radius: float = 0.03
    seg_angle_deg: float = 10.0
    seg_plane_dist: float = 0.015
    min_seg_points: int = 400         # voxels in a proposal segment
    max_range: float = 6.0            # ignore far segments
    n_proposals: int = 12
    gather_radius: float = 0.9        # around the seed centre (m)
    plane_band: float = 0.14          # |distance to seed plane| gathered, wide because of range walk
    inlier_band: float = 0.03         # |distance| kept after walk correction
    px: float = 0.01                  # intensity image resolution (m/px)
    theta_step: float = 3.0           # coarse rotation step (deg), search covers [0,180)
    scales: tuple = (1.0, 1.1, 1.2)   # coarse board-scale multipliers on the nominal square size
    scale_bounds: tuple = (0.9, 1.3)  # allowed range of the fitted scale
    n_refine: int = 5                 # coarse peaks refined continuously
    min_coverage: float = 0.7         # min fraction of the pattern area that has points
    min_ncc: float = 0.4              # accept threshold on the refined NCC
    walk_gain: float = 1.0            # scale of the range-walk correction (1 = full)


@dataclass
class Detection:
    T_lidar_target: np.ndarray        # 4x4, target -> lidar (origin = pattern centre)
    T_lidar_target_alt: np.ndarray    # same board rotated 180 deg about its normal
    ncc: float
    coverage: float
    n_points: int
    plane_rms_mm: float               # after walk correction
    walk_mm: float                    # plane offset of typical dark-square returns (I=130) vs white, toward sensor +
    walk_curve: np.ndarray            # (K,2) intensity, offset mm
    dark_first: bool
    spec: BoardSpec                   # board with the fitted square size
    scale: float                      # fitted square size / nominal square size
    seed_centre: np.ndarray
    sheet_dims: tuple                 # (long, short) extent of the connected on-plane region (m)
    sheet_fill: float                 # occupancy of that region's bounding rectangle
    dark_frac: float                  # fraction of on-pattern points that are dark
    board_points: np.ndarray          # (N,3) on-sheet points, walk-corrected, lidar frame
    board_intensity: np.ndarray
    accepted: bool = True
    note: str = ""
    problems: list = field(default_factory=list)
    rank: float = 0.0


# ---------------------------------------------------------------- geometry helpers
def _fit_plane(pts):
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, vt


def _trimmed_plane(pts, tol, iters=6):
    keep = np.ones(len(pts), bool)
    c, vt = _fit_plane(pts)
    for _ in range(iters):
        new = np.abs((pts - c) @ vt[2]) < tol
        if new.sum() < 50 or (new == keep).all():
            break
        keep = new
        c, vt = _fit_plane(pts[keep])
    return c, vt


def _basis(centre, vt):
    """Right-handed (e1, e2, n) with n facing the sensor (origin)."""
    n = vt[2] if np.dot(vt[2], -centre) > 0 else -vt[2]
    e1 = vt[0] - np.dot(vt[0], n) * n
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1), n


def _walk_curve(inten, d):
    """Median plane offset (m, toward-sensor positive) per intensity bin.  Returns (I_k, d_k)."""
    xs, ys = [], []
    for lo, hi in zip(_WALK_EDGES[:-1], _WALK_EDGES[1:]):
        m = (inten >= lo) & (inten < hi)
        if m.sum() >= 200:
            xs.append(np.median(inten[m]))
            ys.append(np.median(d[m]))
    if not xs or xs[-1] < WHITE_I:            # pin the saturated reference to zero
        xs.append(255.0)
        ys.append(0.0)
    else:
        ys[-1] = 0.0
    return np.array(xs), np.array(ys)


def _apply_walk(P, I, curve, n, min_cos=0.3, gain=1.0):
    """Shift each point along its ray so that its plane offset no longer depends on intensity."""
    bias = gain * np.interp(I, curve[0], curve[1])
    r = P / np.linalg.norm(P, axis=1, keepdims=True)
    rn = r @ n                                   # < 0: ray runs against the sensor-facing normal
    ok = np.abs(rn) > min_cos
    s = np.where(ok, -bias / np.where(ok, rn, 1.0), 0.0)
    return P + s[:, None] * r, ok


# ---------------------------------------------------------------- template matching
def _intensity_image(uv, val, px, pad):
    lo = uv.min(0) - pad
    W, H = np.ceil((uv.max(0) + pad - lo) / px).astype(int)
    ij = np.floor((uv - lo) / px).astype(int)
    s = np.zeros((H, W), np.float32)
    n = np.zeros((H, W), np.float32)
    np.add.at(s, (ij[:, 1], ij[:, 0]), val)
    np.add.at(n, (ij[:, 1], ij[:, 0]), 1)
    return np.where(n > 0, s / np.maximum(n, 1), 0).astype(np.float32), (n > 0).astype(np.float32), lo


def _masked_ncc(img, valid, tmpl, mask):
    """NCC of `tmpl` over `img` using only valid pixels inside `mask`.

    Returns (ncc, coverage) maps indexed by the template's top-left corner.
    """
    tm = tmpl * mask
    corr = lambda a, b: cv2.matchTemplate(a, b, cv2.TM_CCORR)
    N = np.maximum(corr(valid, mask), 1)
    sI, sII = corr(img * valid, mask), corr(img * img * valid, mask)
    sT, sTT, sIT = corr(valid, tm), corr(valid, tm * tm), corr(img * valid, tm)
    num = sIT - sI * sT / N
    den = np.sqrt(np.maximum(sII - sI * sI / N, 1e-9) * np.maximum(sTT - sT * sT / N, 1e-9))
    return num / den, corr(valid, mask) / mask.sum()


def _to_target(uv, phi, tu, tv):
    c, s = np.cos(phi), np.sin(phi)
    d = uv - np.array([tu, tv])
    return np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]])


def _score(inten):
    """Soft light(+1)/dark(-1) score of an intensity; white paper saturates at 255."""
    return np.clip((inten - 215.0) / 40.0, -1.0, 1.0).astype(np.float32)


def _checker_signal(xy, spec, dark_first, k=6.0):
    """Smooth template value (+1 light, -1 dark) at target-frame coordinates."""
    pw, ph = spec.pattern
    sig = np.tanh(k * np.sin(np.pi * (xy[:, 0] + pw / 2) / spec.square)) *           np.tanh(k * np.sin(np.pi * (xy[:, 1] + ph / 2) / spec.square))   # +1 if col+row even
    return -sig if dark_first else sig


def _inside(xy, spec, margin=0.0):
    pw, ph = spec.pattern
    return (np.abs(xy[:, 0]) <= pw / 2 + margin) & (np.abs(xy[:, 1]) <= ph / 2 + margin)


def _pose_ncc(params, uv, val, spec0, dark_first, bounds):
    phi, tu, tv, k = params
    if not bounds[0] <= k <= bounds[1]:
        return 0.0
    spec = spec0.scaled(k)
    xy = _to_target(uv, phi, tu, tv)
    m = _inside(xy, spec)
    if m.sum() < 200:
        return 0.0
    a = _checker_signal(xy[m], spec, dark_first)
    b = val[m]
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _coarse_search(img, valid, origin, spec0, cfg):
    """Coarse peaks [(ncc, theta_deg, cx, cy, scale, dark_first, coverage)], best first."""
    peaks = []
    for k in cfg.scales:
        spec = spec0.scaled(k)
        for dark_first in (True, False):
            t0, m0 = spec.template(cfg.px, dark_first, with_margin=False)
            rows = []
            for theta in np.arange(0, 180, cfg.theta_step):
                tr, mr = rotate_template(t0, m0, theta)
                if tr.shape[0] > img.shape[0] or tr.shape[1] > img.shape[1]:
                    rows.append((-1, theta, 0, 0, k, dark_first, 0))
                    continue
                ncc, cov = _masked_ncc(img, valid, tr, mr)
                ncc = np.where(cov >= cfg.min_coverage, ncc, -1)
                i = np.unravel_index(np.argmax(ncc), ncc.shape)
                rows.append((ncc[i], theta, (i[1] + tr.shape[1] / 2) * cfg.px + origin[0],
                             (i[0] + tr.shape[0] / 2) * cfg.px + origin[1], k, dark_first, cov[i]))
            s = np.array([r[0] for r in rows])
            for i in range(len(rows)):                   # local maxima over theta (circular)
                if s[i] > 0 and s[i] >= s[i - 1] and s[i] >= s[(i + 1) % len(rows)]:
                    peaks.append(rows[i])
    return sorted(peaks, key=lambda r: -r[0])


# ---------------------------------------------------------------- stage 1
def propose(xyz, cfg: Config):
    """Seed point arrays (voxel points of plausible sheet-sized planar segments), largest first."""
    pcd = voxelize(xyz, cfg.voxel)
    pts = np.asarray(pcd.points)
    lab = planar_segments(pts, estimate_normals(pcd, 24), cfg.seg_radius, cfg.seg_angle_deg,
                          cfg.seg_plane_dist)
    sizes = np.bincount(lab)
    seeds = []
    for l in np.argsort(-sizes):
        if sizes[l] < cfg.min_seg_points or len(seeds) >= cfg.n_proposals:
            break
        c = pts[lab == l]
        ctr = c.mean(0)
        if np.linalg.norm(ctr) > cfg.max_range:
            continue
        _, _, vt = np.linalg.svd(c - ctr, full_matrices=False)
        (_, _), (w, h), _ = cv2.minAreaRect(((c - ctr) @ vt[:2].T).astype(np.float32))
        if min(w, h) >= 0.25 and max(w, h) >= 0.4:      # could be (part of) the sheet
            seeds.append(c)
    return seeds


# ---------------------------------------------------------------- stage 2/3
def verify(xyz, inten, seed, spec: BoardSpec, cfg: Config):
    """Match the checker to one proposal.  Returns a Detection or None."""
    ctr0 = seed.mean(0)
    c, vt = _trimmed_plane(seed, 0.03)
    dist = (xyz - c) @ vt[2]
    near = (np.abs(dist) < cfg.plane_band) & (np.linalg.norm(xyz - ctr0, axis=1) < cfg.gather_radius) \
           & (inten >= 100)
    P, I = xyz[near], inten[near]
    white = I >= WHITE_I
    if white.sum() < 300:
        return None
    # reference plane from the saturated returns; re-gather once around it
    c, vt = _trimmed_plane(P[white], 0.02)
    e1, e2, n = _basis(c, vt)
    d = (P - c) @ n
    # range-walk curve from the wide band, then flatten the points onto the white plane
    curve = _walk_curve(I, d)
    Pc, ok = _apply_walk(P, I, curve, n, gain=cfg.walk_gain)
    d = (Pc - c) @ n
    keep = ok & (np.abs(d) < cfg.inlier_band)
    Pc, Ic = Pc[keep], I[keep]
    if len(Pc) < 500:
        return None
    uv = np.column_stack([(Pc - c) @ e1, (Pc - c) @ e2])
    val = _score(Ic)

    img, valid, origin = _intensity_image(uv, val, cfg.px, pad=max(spec.sheet) * max(cfg.scales))
    peaks = _coarse_search(img, valid, origin, spec, cfg)[:cfg.n_refine]
    if not peaks:
        return None
    best = None
    for ncc0, theta, cx, cy, k, dark_first, cov in peaks:
        # cv2 rotates counter-clockwise on a y-down image, i.e. uv = R(-theta) xy
        res = minimize(lambda p: -_pose_ncc(p, uv, val, spec, dark_first, cfg.scale_bounds),
                       [-np.radians(theta), cx, cy, k], method="Nelder-Mead",
                       options=dict(xatol=1e-4, fatol=1e-7, maxiter=800))
        if best is None or -res.fun > best[0]:
            best = (-res.fun, res.x, dark_first, cov)
    ncc, (phi, tu, tv, k), dark_first, cov = best
    return _finalize(P, I, c, e1, e2, n, phi, tu, tv, spec.scaled(k), k, ncc, cov, dark_first, ctr0, cfg)


def _plane_region(uv_near, centre_uv, px=0.02):
    """Connected on-plane region around `centre_uv`: (long, short) extent and rectangle fill."""
    lo = uv_near.min(0)
    ij = np.floor((uv_near - lo) / px).astype(int)
    W, H = ij.max(0) + 1
    occ = np.zeros((H, W), np.uint8)
    occ[ij[:, 1], ij[:, 0]] = 1
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, lab = cv2.connectedComponents(occ)
    ci = np.clip(np.floor((centre_uv - lo) / px).astype(int), 0, [W - 1, H - 1])
    l = lab[ci[1], ci[0]]
    if l == 0:
        return (0.0, 0.0), 0.0
    ys, xs = np.nonzero(lab == l)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    a, b = max(w, h) * px, min(w, h) * px
    return (a, b), float(len(xs) * px * px / max(a * b, 1e-9))


def _finalize(P, I, c, e1, e2, n, phi, tu, tv, spec, k, ncc, cov, dark_first, ctr0, cfg):
    """Re-estimate walk curve + plane on the on-sheet points and build the transform."""
    for _ in range(3):
        curve = _walk_curve(I, (P - c) @ n)
        Pc, ok = _apply_walk(P, I, curve, n, gain=cfg.walk_gain)
        uv = np.column_stack([(Pc - c) @ e1, (Pc - c) @ e2])
        xy = _to_target(uv, phi, tu, tv)
        sel = _inside(xy, spec, margin=0.01) & ok & (np.abs((Pc - c) @ n) < cfg.inlier_band)
        c2, vt2 = _trimmed_plane(Pc[sel], 0.02)
        e1b, e2b, nb = _basis(c2, vt2)
        # carry the in-plane pose to the refined plane
        X0 = np.cos(phi) * e1 + np.sin(phi) * e2
        X = X0 - np.dot(X0, nb) * nb
        X /= np.linalg.norm(X)
        origin = c + tu * e1 + tv * e2
        origin = origin - np.dot(origin - c2, nb) * nb
        # express the pose in the refined plane's own basis for the next pass
        phi = np.arctan2(np.dot(X, e2b), np.dot(X, e1b))
        tu, tv = np.dot(origin - c2, e1b), np.dot(origin - c2, e2b)
        c, e1, e2, n = c2, e1b, e2b, nb
    Y = np.cross(n, X)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = X, Y, n, origin
    Talt = T.copy()
    Talt[:3, 0], Talt[:3, 1] = -X, -Y
    resid = ((Pc[sel] - c) @ n) * 1000
    near = ok & (np.abs((Pc - c) @ n) < cfg.inlier_band)
    dims, fill = _plane_region(uv[near], np.array([tu, tv]))
    dark_frac = float(np.mean(_score(I[sel]) < 0))
    return Detection(
        T_lidar_target=T, T_lidar_target_alt=Talt, ncc=float(ncc), coverage=float(cov),
        n_points=int(sel.sum()), plane_rms_mm=float(np.sqrt(np.mean(resid ** 2))),
        walk_mm=float(1000 * np.interp(130.0, curve[0], curve[1])), walk_curve=np.column_stack([curve[0], curve[1] * 1000]),
        dark_first=bool(dark_first), spec=spec, scale=float(k), seed_centre=ctr0,
        sheet_dims=dims, sheet_fill=fill, dark_frac=dark_frac,
        board_points=Pc[sel], board_intensity=I[sel])


def plausible(det, spec: BoardSpec):
    """Sanity checks that a real printed checker sheet satisfies; returns list of failures."""
    bad = []
    pl, ps = max(spec.pattern) * det.scale, min(spec.pattern) * det.scale
    a, b = det.sheet_dims
    if not (ps * 0.95 <= b <= ps * 2.0 and pl * 0.95 <= a <= pl * 2.0):
        bad.append(f"plane region {a:.2f}x{b:.2f} m does not fit the board")
    if det.sheet_fill < 0.6:
        bad.append(f"plane region fill {det.sheet_fill:.2f} < 0.6")
    if not 0.2 <= det.dark_frac <= 0.7:
        bad.append(f"dark fraction {det.dark_frac:.2f} outside 0.2-0.7")
    return bad


def extract_target(xyz, inten, spec: BoardSpec = BoardSpec(), cfg: Config = Config()):
    """Full pipeline on one accumulated cloud.  Returns the best Detection (or None)."""
    best = None
    for seed in propose(xyz, cfg):
        det = verify(xyz, inten, seed, spec, cfg)
        if det is None:
            continue
        det.problems = plausible(det, spec)
        det.rank = det.ncc * (0.5 if det.problems else 1.0)
        if best is None or det.rank > best.rank:
            best = det
    if best is not None:
        why = list(best.problems)
        if best.ncc < cfg.min_ncc:
            why.append(f"NCC {best.ncc:.2f} < {cfg.min_ncc}")
        best.accepted = not why
        best.note = "; ".join(why)
    return best
