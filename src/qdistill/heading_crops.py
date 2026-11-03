"""image crops of 3D boxes and the observation angle, for the pedestrian / bicycle heading-flip refiner.

Boxes are ego frame [x, y, z_centre, length (along the heading), width, h, yaw, vx, vy] (; GT and exported
predictions alike). Cameras: packed calib (cam_intrinsic for the original 1600 x 900 JPEGs, sensor2lidar R / t,
lidar2ego).

Observation angle (camera frame, x right / y down / z forward): heading azimuth beta = atan2(d_x, d_z) of the heading
direction, ray azimuth theta = atan2(c_x, c_z) of the box centre, alpha = wrap(beta - theta). It is what appearance
determines (the same person seen from the same side gives the same alpha anywhere in the image). A horizontal mirror
of the crop maps x -> -x, so alpha -> -alpha (``mirror_alpha``; checked by ``_self_test``).
"""

from __future__ import annotations

import numpy as np

CROP_H, CROP_W = 128, 64


def wrap(a):
    return np.angle(np.exp(1j * np.asarray(a, np.float64)))


class Rig:
    """Per-frame camera geometry from a PerceptionFrame (or anything with the packed calib attributes)."""

    def __init__(self, frame):
        self.cams = list(frame.cam_order)
        self.K = np.asarray(frame.cam_intrinsic, np.float64)                       # [6, 3, 3]
        self.R = np.asarray(frame.sensor2lidar_rotation, np.float64)                # cam -> lidar
        self.t = np.asarray(frame.sensor2lidar_translation, np.float64)
        e2l = np.linalg.inv(np.asarray(frame.lidar2ego, np.float64))
        self.e2l_R, self.e2l_t = e2l[:3, :3], e2l[:3, 3]

    def to_cam(self, p_ego: np.ndarray, c: int) -> np.ndarray:
        """[N, 3] ego points -> camera c frame."""
        p_l = p_ego @ self.e2l_R.T + self.e2l_t
        return (p_l - self.t[c]) @ self.R[c]                                        # R^T (p - t), row form

    def dir_to_cam(self, d_ego: np.ndarray, c: int) -> np.ndarray:
        return d_ego @ self.e2l_R.T @ self.R[c]


def corners(box) -> np.ndarray:
    """[8, 3] ego-frame corners of one box."""
    x, y, z, ln, wd, h, yaw = box[:7]
    dx, dy, dz = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * ln / 2, np.array([1, -1, -1, 1, 1, -1, -1, 1]) * wd / 2, \
        np.array([1, 1, 1, 1, -1, -1, -1, -1]) * h / 2
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([x + c * dx - s * dy, y + s * dx + c * dy, z + dz], 1)


def best_view(rig: Rig, box, img_wh=(1600, 900), min_in=0.5):
    """Camera index and enlarged pixel rect (x0, y0, x1, y1) where the box's projected hull has the largest
    in-image area; None if no camera sees >= ``min_in`` of the hull with every corner in front."""
    W, H = img_wh
    best = None
    for c in range(len(rig.cams)):
        pc = rig.to_cam(corners(box), c)
        if (pc[:, 2] < 0.3).any():
            continue
        uv = pc @ rig.K[c].T
        uv = uv[:, :2] / uv[:, 2:3]
        x0, y0 = uv.min(0)
        x1, y1 = uv.max(0)
        area = (x1 - x0) * (y1 - y0)
        inside = max(0.0, min(x1, W) - max(x0, 0)) * max(0.0, min(y1, H) - max(y0, 0))
        if area <= 0 or inside / area < min_in:
            continue
        if best is None or inside > best[0]:
            cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * 0.6, (y1 - y0) * 0.6   # x1.2
            best = (inside, c, (cx - hw, cy - hh, cx + hw, cy + hh))
    return None if best is None else best[1:]


def alpha(rig: Rig, box, c: int, yaw=None) -> float:
    """Observation angle of ``box`` (or of heading ``yaw`` at its centre) in camera c."""
    yaw = box[6] if yaw is None else yaw
    cc = rig.to_cam(np.asarray(box[:3], np.float64)[None], c)[0]
    d = rig.dir_to_cam(np.array([[np.cos(yaw), np.sin(yaw), 0.0]]), c)[0]
    return float(wrap(np.arctan2(d[0], d[2]) - np.arctan2(cc[0], cc[2])))


def mirror_alpha(a):
    return wrap(-np.asarray(a))


def crop(img: np.ndarray, rect) -> np.ndarray:
    """uint8 [H, W, 3] image, float rect -> letterboxed [CROP_H, CROP_W, 3] uint8 (zero padding outside the image)."""
    from PIL import Image
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    s = min(CROP_W / w, CROP_H / h)
    ow, oh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    H, W = img.shape[:2]
    ix0, iy0, ix1, iy1 = int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1))
    pad = np.zeros((iy1 - iy0, ix1 - ix0, 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(ix0, 0), max(iy0, 0), min(ix1, W), min(iy1, H)
    if sx1 > sx0 and sy1 > sy0:
        pad[sy0 - iy0:sy1 - iy0, sx0 - ix0:sx1 - ix0] = img[sy0:sy1, sx0:sx1]
    r = np.asarray(Image.fromarray(pad).resize((ow, oh), Image.BILINEAR))
    out = np.zeros((CROP_H, CROP_W, 3), np.uint8)
    oy, ox = (CROP_H - oh) // 2, (CROP_W - ow) // 2
    out[oy:oy + oh, ox:ox + ow] = r
    return out


def _self_test():
    """Mirror check: a box mirrored across the camera's y-z plane has alpha -> -alpha."""
    class F:  # one level camera looking along ego +x, identity lidar2ego
        cam_order = ["CAM_FRONT"]
        cam_intrinsic = np.array([[[1000.0, 0, 800], [0, 1000, 450], [0, 0, 1]]])
        sensor2lidar_rotation = np.array([[[0.0, 0, 1], [-1, 0, 0], [0, -1, 0]]])  # cam z -> ego x, cam x -> -ego y
        sensor2lidar_translation = np.zeros((1, 3))
        lidar2ego = np.eye(4)
    rig = Rig(F())
    rng = np.random.default_rng(0)
    for _ in range(100):
        x, y, yaw = rng.uniform(5, 30), rng.uniform(-10, 10), rng.uniform(-np.pi, np.pi)
        a = alpha(rig, np.array([x, y, 0, 0.6, 0.6, 1.7, yaw]), 0)
        am = alpha(rig, np.array([x, -y, 0, 0.6, 0.6, 1.7, -yaw]), 0)                # mirror: ego y -> -y
        assert abs(wrap(am - mirror_alpha(a))) < 1e-9, (a, am)
    # a box facing straight away from the camera on the optical axis has alpha 0; facing the camera, pi
    assert abs(alpha(rig, np.array([10, 0, 0, 1, 1, 1, 0.0]), 0)) < 1e-9
    assert abs(abs(alpha(rig, np.array([10, 0, 0, 1, 1, 1, np.pi]), 0)) - np.pi) < 1e-9
    return "heading_crops self-test ok"


if __name__ == "__main__":
    print(_self_test())
