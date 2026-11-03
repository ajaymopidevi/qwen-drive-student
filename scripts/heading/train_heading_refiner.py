"""Train the heading refiner on GT crops."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from qdistill.heading_crops import mirror_alpha

ROOT = Path(__file__).resolve().parents[2]
CLS_IDX = {4: 0, 2: 1}  # pedestrian, bicycle
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255
DIST_BINS = (0, 10, 20, 30, 40)


class Crops(torch.utils.data.Dataset):
    def __init__(self, root: Path, part: str, train: bool):
        self.x = [np.load(f, mmap_mode="r") for f in sorted(root.glob(f"{part}_*_x.npy"))]
        metas = [np.load(str(f).replace("_x.npy", "_meta.npz")) for f in sorted(root.glob(f"{part}_*_x.npy"))]
        self.meta = {k: np.concatenate([m[k] for m in metas]) for k in ("alpha", "cls", "dist")}
        self.index = [(i, j) for i, x in enumerate(self.x) for j in range(len(x))]
        self.train = train

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, j = self.index[k]
        x = np.array(self.x[i][j], dtype=np.float32)
        a = float(self.meta["alpha"][k])
        if self.train:
            if np.random.rand() < 0.5:
                x, a = x[:, ::-1].copy(), float(mirror_alpha(a))
            x = np.clip((x - 128) * np.random.uniform(0.7, 1.3) + 128 + np.random.uniform(-25, 25), 0, 255)
        return (torch.from_numpy(x).permute(2, 0, 1), torch.tensor(a, dtype=torch.float32),
                torch.tensor(CLS_IDX[int(self.meta["cls"][k])]), torch.tensor(float(self.meta["dist"][k])))


class HeadingRefiner(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        r = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.backbone = nn.Sequential(*list(r.children())[:-1])
        self.cls = nn.Embedding(2, 32)
        self.out = nn.Sequential(nn.Linear(512 + 32, 256), nn.ReLU(inplace=True), nn.Linear(256, 3))

    def forward(self, x, cls):
        """x: uint8-range float [B, 3, 128, 64]; cls: [B] (0 ped, 1 bike) -> (alpha [B], confidence logit [B])."""
        f = self.backbone((x - MEAN.to(x.device)) / STD.to(x.device)).flatten(1)
        o = self.out(torch.cat([f, self.cls(cls)], 1)).float()
        return torch.atan2(o[:, 0], o[:, 1]), o[:, 2]


def loss_fn(alpha_p, conf, alpha_t):
    err = torch.remainder(alpha_p - alpha_t + math.pi, 2 * math.pi) - math.pi
    ok = (err.abs() < math.pi / 2).float().detach()
    return (1 - torch.cos(err)).mean() + F.binary_cross_entropy_with_logits(conf, ok), err


@torch.no_grad()
def evaluate(model, loader, tau_acc):
    model.eval()
    E, C, K, D = [], [], [], []
    for x, a, c, d in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ap, cf = model(x.cuda(non_blocking=True), c.cuda())
        _, err = loss_fn(ap, cf, a.cuda())
        E.append(err.abs().cpu()); C.append(torch.sigmoid(cf).cpu()); K.append(c); D.append(d)
    E, C, K, D = (torch.cat(t).numpy() for t in (E, C, K, D))
    ok = E < math.pi / 2
    res = {"n": int(len(E)), "mean_abs_err": float(E.mean())}
    for k, name in ((0, "pedestrian"), (1, "bicycle")):
        m = K == k
        res[f"{name}_fb_acc"] = float(ok[m].mean()) if m.any() else None
        res[f"{name}_by_dist"] = {f"{lo}-{hi}": float(ok[m & (D >= lo) & (D < hi)].mean())
                                  for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:]) if (m & (D >= lo) & (D < hi)).any()}
    for k, name in ((0, "ped"), (1, "bike")):
        m = K == k
        taus = [t for t in np.linspace(0, 0.99, 100) if (m & (C >= t)).sum() >= 50 and ok[m & (C >= t)].mean() >= tau_acc]
        tau = float(taus[0]) if taus else None
        res["tau" if k == 0 else "tau_bike"] = tau
        res[f"tau_coverage_{name}"] = float((m & (C >= tau)).sum() / m.sum()) if tau is not None else 0.0
    model.train()
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crops", type=Path, default=ROOT / "outputs/heading_crops/gt")
    p.add_argument("--run-name", default="heading_refiner")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tau-acc", type=float, default=0.9)
    p.add_argument("--max-steps", type=int, default=0, help="smoke: stop after N steps (then one eval)")
    args = p.parse_args()
    out = ROOT / "runs" / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)

    tr, ho = Crops(args.crops, "train", True), Crops(args.crops, "hold", False)
    print(f"crops: train {len(tr)}, held out {len(ho)}", flush=True)
    dl = torch.utils.data.DataLoader(tr, args.batch, shuffle=True, num_workers=1, pin_memory=True, drop_last=True,
                                     persistent_workers=True)
    dh = torch.utils.data.DataLoader(ho, 512, num_workers=1, pin_memory=True)
    model = HeadingRefiner().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.max_steps or args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total, pct_start=0.05)
    hist, best, step = [], -1.0, 0
    t0 = time.time()
    for ep in range(args.epochs):
        for x, a, c, _ in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ap, cf = model(x.cuda(non_blocking=True), c.cuda())
            loss, _ = loss_fn(ap, cf, a.cuda())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
            if step % 100 == 0:
                print(f"  step {step}/{total} loss {loss.item():.4f} {(time.time() - t0) / step:.3f} s/step "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        r = evaluate(model, dh, args.tau_acc)
        r.update(epoch=ep, step=step, minutes=round((time.time() - t0) / 60, 1))
        hist.append(r)
        print("hold " + json.dumps(r), flush=True)
        if r["pedestrian_fb_acc"] > best:
            best = r["pedestrian_fb_acc"]
            torch.save({"model": model.state_dict(), "eval": r, "args": vars(args)}, out / "best.pt")
        (out / "history.json").write_text(json.dumps(hist, indent=1))
        if args.max_steps and step >= args.max_steps:
            break
    print(f"done: best held-out pedestrian front / back accuracy {best:.4f} -> {out / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
