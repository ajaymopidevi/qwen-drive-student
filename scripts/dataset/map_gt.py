"""Rasterise the nuScenes map expansion into the teacher map taxonomy (eval env)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
RES, H, W = 0.15, 200, 400
DEMO = ROOT / "third_party/Qwen-Drive-1.0/data/demo/perception"
DEMO_TOKENS = ("f41862469f33424eb660258e9392c4f4", "90162f90eceb4ada9e595bc1adb71b5f")
CLASSES = ("background", "driveable_surface", "road_line", "road_edge", "crosswalk", "walkway")
DEFAULT = {"line_px": 8, "edge_px": 4, "order": (1, 5, 4, 3, 2), "flip": "none"}


def yaw_of(q) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class Rasteriser:
    def __init__(self, dataroot: Path):
        self.dataroot = dataroot
        self.maps: dict[str, NuScenesMap] = {}

    def nmap(self, location: str) -> NuScenesMap:
        if location not in self.maps:
            self.maps[location] = NuScenesMap(dataroot=str(self.dataroot), map_name=location)
        return self.maps[location]

    def geoms(self, pose: dict) -> dict:
        """Global-frame shapely geometries near the ego, per source layer."""
        m = self.nmap(pose["location"])
        x, y = pose["translation"][:2]
        box = (x - 40, y - 40, x + 40, y + 40)
        rec = m.get_records_in_patch(box, ["drivable_area", "ped_crossing", "walkway", "lane_divider", "road_divider"],
                                     mode="intersect")
        drivable = [m.extract_polygon(pt) for tok in rec["drivable_area"] for pt in m.get("drivable_area", tok)["polygon_tokens"]]
        return {
            "drivable": drivable,
            "crosswalk": [m.extract_polygon(m.get("ped_crossing", t)["polygon_token"]) for t in rec["ped_crossing"]],
            "walkway": [m.extract_polygon(m.get("walkway", t)["polygon_token"]) for t in rec["walkway"]],
            "lines": [m.extract_line(m.get(layer, t)["line_token"]) for layer in ("lane_divider", "road_divider")
                      for t in rec[layer]],
        }

    @staticmethod
    def to_px(coords: np.ndarray, pose: dict) -> np.ndarray:
        yaw = yaw_of(pose["rotation"])
        c, s = np.cos(yaw), np.sin(yaw)
        d = np.asarray(coords)[:, :2] - np.asarray(pose["translation"][:2])
        xe, ye = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
        return np.stack([(xe + 30) / RES, (ye + 15) / RES], 1)  # (col, row) float pixels

    def render(self, geoms: dict, pose: dict, cfg: dict) -> np.ndarray:
        layers = {k: np.zeros((H, W), np.uint8) for k in range(1, 6)}

        def polys(g):
            if isinstance(g, MultiPolygon):
                return list(g.geoms)
            return [g] if isinstance(g, Polygon) and not g.is_empty else []

        def fill(mask, geom_list):
            for g in geom_list:
                for p in polys(g):
                    ext = np.round(self.to_px(np.asarray(p.exterior.coords), pose) * 8).astype(np.int32)
                    cv2.fillPoly(mask, [ext], 1, shift=3)
                    for hole in p.interiors:
                        cv2.fillPoly(mask, [np.round(self.to_px(np.asarray(hole.coords), pose) * 8).astype(np.int32)], 0, shift=3)

        def stroke(mask, lines, px):
            for ln in lines:
                parts = ln.geoms if isinstance(ln, MultiLineString) else [ln]
                for part in parts:
                    if isinstance(part, LineString) and not part.is_empty:
                        pts = np.round(self.to_px(np.asarray(part.coords), pose) * 8).astype(np.int32)
                        cv2.polylines(mask, [pts], False, 1, thickness=px, shift=3)

        fill(layers[1], geoms["drivable"])
        fill(layers[5], geoms["walkway"])
        fill(layers[4], geoms["crosswalk"])
        edge = unary_union([p for g in geoms["drivable"] for p in polys(g)]).boundary if geoms["drivable"] else None
        if edge is not None:
            stroke(layers[3], [edge], cfg["edge_px"])
        stroke(layers[2], geoms["lines"], cfg["line_px"])
        out = np.zeros((H, W), np.uint8)
        for k in cfg["order"]:
            out[layers[k] > 0] = k
        if cfg["flip"] == "y":
            out = out[::-1]
        elif cfg["flip"] == "x":
            out = out[:, ::-1]
        return out


def iou(a: np.ndarray, b: np.ndarray) -> dict:
    r = {}
    for k in range(6):
        u = np.sum((a == k) | (b == k))
        r[CLASSES[k]] = float(np.sum((a == k) & (b == k)) / u) if u else None
    r["miou_fg"] = float(np.mean([v for k, v in r.items() if k != "background" and v is not None]))
    r["pixel_acc"] = float(np.mean(a == b))
    return r


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataroot", type=Path, default=ROOT / "data/nuscenes")
    p.add_argument("--poses", type=Path, default=ROOT / "data/gt/ego_poses_val.json")
    p.add_argument("--split", default="val_quick")
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-trainval.json")
    p.add_argument("--config", type=Path, default=ROOT / "data/gt/map_gt_config.json")
    p.add_argument("--output", type=Path, default=ROOT / "data/gt/map")
    args = p.parse_args()
    poses = json.loads(args.poses.read_text())
    ras = Rasteriser(args.dataroot)


    cfg = json.loads(args.config.read_text())
    cfg["order"] = tuple(cfg["order"])
    args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.splits.read_text())  # same expansion as qdistill.data.split_tokens (no torch in this env)
    if args.split in data.get("frame_lists", {}):  # explicit frame lists
        tokens = data["frame_lists"][args.split][:: args.stride]
    else:
        tokens = [t for sc in data[args.split] for t in data["scenes"][sc]["samples"]][:: args.stride]
    for i, t in enumerate(tokens):
        out = args.output / f"{t}.npy"
        if not out.exists():
            np.save(out, ras.render(ras.geoms(poses[t]), poses[t], cfg))
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(tokens)}", flush=True)
    print(f"map gt done: {len(tokens)} frames -> {args.output}")


if __name__ == "__main__":
    main()
