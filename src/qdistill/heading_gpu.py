"""the pedestrian / bicycle heading-flip refiner on the GPU, inside the deployed pipeline.

    hr = HeadingGPU("weights/heading_refiner.pt", engine="weights/engines/heading_refiner_fp16_b64.engine")
    rig = hr.rig(PerceptionFrame(...))           # per camera rig, before the clock (like img_metas)
    images, raw = load_images_gpu(frame_dir, return_raw=True)
    det = bev.head.get_bboxes(outs, [metas])[0]
    flip = hr(det, raw, rig)                     # flips det["boxes"] in place, returns the flipped mask

Same rule as scripts/heading/apply_heading_refiner.py: candidates are pedestrian / bicycle boxes
with score >= 0.1 within 40 m of the ego; the crop camera is the one with the largest in-image hull
(qdistill.heading_crops.best_view, vectorised here), the crop the x1.2 hull letterboxed to 128 x 64 from the full
1600 x 900 image (bilinear antialias resize as PIL); flip (yaw + pi) where the refiner's observation angle differs
from the box's by > 90 deg with confidence >= the class tau; pedestrians also negate velocity, bicycles keep it
(--keep-vel 2) and are not flipped above 1 m/s predicted speed (--max-speed 2:1.0).
Boxes stay in the lidar frame (yaw + pi and velocity negation commute with the lidar -> ego rotation).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from qdistill.heading_crops import CROP_H, CROP_W, Rig

PED, BIKE = 4, 2
CLS_IDX = {PED: 0, BIKE: 1}
MIN_SCORE, MAX_DIST = 0.1, 40.0
IMG_W, IMG_H = 1600, 900
# box corner signs (qdistill.heading_crops.corners)
_SX = torch.tensor([1, 1, -1, -1, 1, 1, -1, -1], dtype=torch.float64) / 2
_SY = torch.tensor([1, -1, -1, 1, 1, -1, -1, 1], dtype=torch.float64) / 2
_SZ = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], dtype=torch.float64) / 2


class GpuRig:
    """Rig (qdistill.heading_crops) as float64 CUDA tensors, plus lidar -> ego."""

    def __init__(self, frame):
        r = Rig(frame)
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64, device="cuda")
        self.K, self.R, self.t = t(r.K), t(r.R), t(r.t)
        self.e2l_R, self.e2l_t = t(r.e2l_R), t(r.e2l_t)
        l2e = t(frame.lidar2ego)
        self.l2e_R, self.l2e_t = l2e[:3, :3], l2e[:3, 3]

    def to_cam(self, p_ego):
        """[..., 3] ego points -> [6, ..., 3] in every camera."""
        p_l = p_ego @ self.e2l_R.T + self.e2l_t
        return torch.einsum("...j,cjk->c...k", p_l, self.R) - torch.einsum("cj,cjk->ck", self.t, self.R).view(
            -1, *([1] * (p_ego.dim() - 1)), 3)


def best_view(rig: GpuRig, ego):
    """ego boxes [N, >= 7] (z at centre) -> (cam [N] long, -1 if none; rect [N, 4] x1.2 hull), as best_view."""
    x, y, z, ln, wd, h, yaw = ego[:, :7].T
    dx, dy, dz = _SX.cuda() * ln[:, None], _SY.cuda() * wd[:, None], _SZ.cuda() * h[:, None]
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    corners = torch.stack([x[:, None] + c * dx - s * dy, y[:, None] + s * dx + c * dy, z[:, None] + dz], -1)  # [N, 8, 3]
    pc = rig.to_cam(corners)                                                     # [6, N, 8, 3]
    front = (pc[..., 2] >= 0.3).all(-1)
    uv = torch.einsum("cnkj,cij->cnki", pc, rig.K)
    uv = uv[..., :2] / uv[..., 2:3].clamp(min=1e-6)
    lo, hi = uv.amin(2), uv.amax(2)                                              # [6, N, 2]
    area = (hi - lo).prod(-1)
    inside = ((hi[..., 0].clamp(max=IMG_W) - lo[..., 0].clamp(min=0)).clamp(min=0)
              * (hi[..., 1].clamp(max=IMG_H) - lo[..., 1].clamp(min=0)).clamp(min=0))
    ok = front & (area > 0) & (inside >= 0.5 * area)
    score = torch.where(ok, inside, torch.full_like(inside, -1.0))
    best = score.argmax(0)                                                       # first max = best_view's strict >
    cam = torch.where(ok.any(0), best, torch.full_like(best, -1))
    n = torch.arange(len(ego), device=ego.device)
    lo, hi = lo[best, n], hi[best, n]
    ctr, half = (lo + hi) / 2, (hi - lo) * 0.6
    return cam, torch.cat([ctr - half, ctr + half], 1)


def alpha(rig: GpuRig, ego, cam):
    """Observation angle of each box in its camera (heading_crops.alpha)."""
    n = torch.arange(len(ego), device=ego.device)
    cc = rig.to_cam(ego[:, :3])[cam, n]
    d = torch.stack([torch.cos(ego[:, 6]), torch.sin(ego[:, 6]), torch.zeros_like(ego[:, 6])], 1) @ rig.e2l_R.T
    d = torch.einsum("nj,njk->nk", d, rig.R[cam])
    return wrap(torch.atan2(d[:, 0], d[:, 2]) - torch.atan2(cc[:, 0], cc[:, 2]))


def wrap(a):
    return torch.remainder(a + np.pi, 2 * np.pi) - np.pi


def crops(raw, cam, rect):
    """raw [6, 3, 900, 1600] (uint8 or float 0-255), cam [N], rect [N, 4] -> float [N, 3, 128, 64]
    (heading_crops.crop on the GPU, one box at a time: the reference for crops_fast)."""
    out = torch.zeros((len(cam), 3, CROP_H, CROP_W), dtype=torch.float32, device=raw.device)
    for i, (c, (x0, y0, x1, y1)) in enumerate(zip(cam.tolist(), rect.tolist())):
        w, h = x1 - x0, y1 - y0
        s = min(CROP_W / w, CROP_H / h)
        ow, oh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        ix0, iy0, ix1, iy1 = int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1))
        pad = torch.zeros((3, iy1 - iy0, ix1 - ix0), dtype=torch.float32, device=raw.device)
        sx0, sy0, sx1, sy1 = max(ix0, 0), max(iy0, 0), min(ix1, IMG_W), min(iy1, IMG_H)
        if sx1 > sx0 and sy1 > sy0:
            pad[:, sy0 - iy0:sy1 - iy0, sx0 - ix0:sx1 - ix0] = raw[c, :, sy0:sy1, sx0:sx1]
        r = F.interpolate(pad[None], size=(oh, ow), mode="bilinear", antialias=True, align_corners=False)[0]
        oy, ox = (CROP_H - oh) // 2, (CROP_W - ow) // 2
        out[i, :, oy:oy + oh, ox:ox + ow] = r.round().clamp(0, 255)
    return out


def crops_fast(raw, cam, rect):
    """crops() as one roi_align: the 128 x 64 canvas maps to the letterboxed source window (adaptive sampling =
    box-filter antialias), zero outside the resized rect (the letterbox bars) and outside the image."""
    from torchvision.ops import roi_align

    x0, y0, x1, y1 = rect.T
    w, h = x1 - x0, y1 - y0
    s = torch.minimum(CROP_W / w, CROP_H / h)
    ow, oh = (w * s).round().clamp(min=1), (h * s).round().clamp(min=1)
    ix0, iy0, ix1, iy1 = x0.floor(), y0.floor(), x1.ceil(), y1.ceil()
    sx, sy = (ix1 - ix0) / ow, (iy1 - iy0) / oh                                  # source px per canvas px (as PIL)
    ox, oy = ((CROP_W - ow) / 2).floor(), ((CROP_H - oh) / 2).floor()
    rois = torch.stack([cam.to(rect.dtype), ix0 - ox * sx, iy0 - oy * sy, ix0 + (CROP_W - ox) * sx,
                        iy0 + (CROP_H - oy) * sy], 1).float()
    out = roi_align(raw.float(), rois, (CROP_H, CROP_W), 1.0, -1, aligned=True)
    u = torch.arange(CROP_W, device=raw.device)[None]
    v = torch.arange(CROP_H, device=raw.device)[None]
    m = (((u >= ox[:, None]) & (u < (ox + ow)[:, None]))[:, None, :] & ((v >= oy[:, None]) & (v < (oy + oh)[:, None]))[:, :, None])
    return out.round().clamp(0, 255) * m[:, None].float()


def best_view_np(rig: Rig, ego):
    """best_view / alpha vectorised in numpy (CPU; ~30 boxes a frame is launch-bound on the GPU):
    ego [N, >= 7] -> cam [N] (-1 if none), rect [N, 4], alpha [N]."""
    x, y, z, ln, wd, h, yaw = ego[:, :7].T
    sx, sy, sz = (t.numpy() for t in (_SX, _SY, _SZ))
    dx, dy, dz = sx * ln[:, None], sy * wd[:, None], sz * h[:, None]
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    corners = np.stack([x[:, None] + c * dx - s * dy, y[:, None] + s * dx + c * dy, z[:, None] + dz], -1)  # [N, 8, 3]
    p_l = corners @ rig.e2l_R.T + rig.e2l_t
    pc = np.einsum("nkj,cjl->cnkl", p_l, rig.R) - np.einsum("cj,cjl->cl", rig.t, rig.R)[:, None, None]
    front = (pc[..., 2] >= 0.3).all(-1)
    uv = np.einsum("cnkj,cij->cnki", pc, rig.K)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uv[..., :2] / uv[..., 2:3]
    lo, hi = uv.min(2), uv.max(2)
    area = (hi - lo).prod(-1)
    inside = (np.clip(np.minimum(hi[..., 0], IMG_W) - np.maximum(lo[..., 0], 0), 0, None)
              * np.clip(np.minimum(hi[..., 1], IMG_H) - np.maximum(lo[..., 1], 0), 0, None))
    ok = front & (area > 0) & (inside >= 0.5 * area)
    best = np.where(ok, inside, -1.0).argmax(0)
    n = np.arange(len(ego))
    cam = np.where(ok.any(0), best, -1)
    lo, hi = lo[best, n], hi[best, n]
    ctr, half = (lo + hi) / 2, (hi - lo) * 0.6
    cc = (ego[:, :3] @ rig.e2l_R.T + rig.e2l_t - rig.t[best])
    cc = np.einsum("nj,njk->nk", cc, rig.R[best])
    d = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], 1) @ rig.e2l_R.T
    d = np.einsum("nj,njk->nk", d, rig.R[best])
    a = np.angle(np.exp(1j * (np.arctan2(d[:, 0], d[:, 2]) - np.arctan2(cc[:, 0], cc[:, 2]))))
    return cam, np.concatenate([ctr - half, ctr + half], 1), a


class HeadingGPU:
    def __init__(self, ckpt: str | Path, engine: str | Path | None = None, batch: int = 64,
                 keep_vel=(BIKE,), max_speed: dict | None = None, fast: bool = True):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        ev = ck["eval"]
        self.tau = {PED: ev["tau"], BIKE: ev.get("tau_bike", ev["tau"])}
        self.tau = {c: np.inf if t is None else float(t) for c, t in self.tau.items()}
        self.keep_vel, self.max_speed = set(keep_vel), {BIKE: 1.0} if max_speed is None else max_speed
        self.batch, self.fast = batch, fast  # an engine's batch comes from its file name (..._b<batch>.engine)
        if engine is None:
            from train_heading_refiner import HeadingRefiner
            self.model = HeadingRefiner(pretrained=False).cuda().eval()
            self.model.load_state_dict(ck["model"])
            self.engine = None
        else:
            from qdistill.trt_utils import TrtRunner
            import re
            self.batch = batch = int(re.search(r"_b(\d+)\.engine$", str(engine)).group(1))
            self.engine = TrtRunner(engine, "x", (batch, 3, CROP_H, CROP_W), "y", (batch, 6))
        self.ck = ck
        self.prof = None  # {step: [ms per frame]} when set to {} (synced steps; e2e_latency --heading-profile)
        from train_heading_refiner import MEAN, STD
        self.mean, self.std = MEAN.cuda(), STD.cuda()

    def rig(self, frame) -> Rig:
        r = Rig(frame)
        r.l2e = np.asarray(frame.lidar2ego, np.float64)
        return r

    def refine(self, x, cls):
        """x float [N, 3, 128, 64] (0-255), cls [N] (0 ped, 1 bike) -> (alpha [N], confidence [N])."""
        if self.engine is None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a, c = self.model(x, cls)
            return a.float(), torch.sigmoid(c.float())
        A, C = [], []
        for i in range(0, len(x), self.batch):
            xb, cb = x[i:i + self.batch], cls[i:i + self.batch]
            n = len(xb)
            inp = torch.zeros((self.batch, 3, CROP_H, CROP_W), dtype=torch.float16, device=x.device)
            inp[:n] = (xb - self.mean) / self.std  # the engine takes normalised crops (trt_heading_refiner.py)
            o = self.engine(inp)[:n].float().view(n, 2, 3)[torch.arange(n, device=x.device), cb]  # both classes out
            A.append(torch.atan2(o[:, 0], o[:, 1])); C.append(torch.sigmoid(o[:, 2]))
        return torch.cat(A), torch.cat(C)

    @torch.no_grad()
    def __call__(self, det: dict, raw: torch.Tensor, rig: Rig) -> np.ndarray:
        """det: get_bboxes output for one frame (boxes [N, 9] lidar frame, scores, labels; tensors or arrays);
        flips det["boxes"] in place; returns the flipped mask [N] (numpy)."""
        import time
        clock = {"t": None}

        def tick(name):  # : synced step times
            if self.prof is None:
                return
            torch.cuda.synchronize()
            t = time.perf_counter()
            if clock["t"] is not None:
                self.prof.setdefault(name, []).append(1000 * (t - clock["t"]))
            clock["t"] = t
        tick("start")
        to_np = lambda v: v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
        b, sc, lab = to_np(det["boxes"]).astype(np.float64), to_np(det["scores"]), to_np(det["labels"])
        tick("det to host")
        flip = np.zeros(len(b), bool)
        self.last = {"cand": 0, "cropped": 0, "flipped": 0}
        if not len(b):
            return flip
        ego = b.copy()
        ego[:, :3] = b[:, :3] @ rig.l2e[:3, :3].T + rig.l2e[:3, 3]
        ego[:, 6] = b[:, 6] + np.arctan2(rig.l2e[1, 0], rig.l2e[0, 0])
        ego[:, 2] += ego[:, 5] / 2
        cand = np.isin(lab, (PED, BIKE)) & (sc >= MIN_SCORE) & (np.hypot(ego[:, 0], ego[:, 1]) <= MAX_DIST)
        idx = np.nonzero(cand)[0]
        if len(idx):
            cam, rect, a_dec = best_view_np(rig, ego[idx])
            tick("geometry (numpy)")
            seen = cam >= 0
            idx, cam, rect, a_dec = idx[seen], cam[seen], rect[seen], a_dec[seen]
        if not len(idx):
            return flip
        dev = raw.device
        g = torch.from_numpy(np.concatenate([cam[:, None], rect, (lab[idx] == BIKE)[:, None]], 1)).to(dev)  # one copy
        cam_t, rect_t, cls = g[:, 0].long(), g[:, 1:5], g[:, 5].long()
        tick("upload")
        x = (crops_fast if self.fast else crops)(raw, cam_t, rect_t)
        tick("crops (roi_align)")
        a_ref, conf = self.refine(x, cls)
        tick("refiner engine")
        a_ref, conf = a_ref.cpu().numpy().astype(np.float64), conf.cpu().numpy()
        tau = np.where(lab[idx] == PED, self.tau[PED], self.tau[BIKE])
        f = (np.abs(np.angle(np.exp(1j * (a_ref - a_dec)))) > np.pi / 2) & (conf >= tau)
        speed = np.hypot(b[idx, 7], b[idx, 8])
        for c, v in self.max_speed.items():
            f &= ~((lab[idx] == c) & (speed > v))
        flip[idx[f]] = True
        self.last = {"cand": int(cand.sum()), "cropped": int(len(idx)), "flipped": int(f.sum())}
        self._apply(det, flip)
        tick("flip rule + apply")
        return flip

    def _apply(self, det, flip):
        boxes = det["boxes"]
        lab = det["labels"].cpu().numpy() if torch.is_tensor(det["labels"]) else np.asarray(det["labels"])
        nv = flip & ~np.isin(lab, sorted(self.keep_vel))
        if torch.is_tensor(boxes):
            fl, nv = torch.from_numpy(flip).to(boxes.device), torch.from_numpy(nv).to(boxes.device)
            boxes[fl, 6] = torch.remainder(boxes[fl, 6] + 2 * np.pi, 2 * np.pi) - np.pi
        else:
            fl = flip
            boxes[fl, 6] = np.remainder(boxes[fl, 6] + 2 * np.pi, 2 * np.pi) - np.pi
        boxes[nv, 7:9] = -boxes[nv, 7:9]
