"""Pack nuScenes keyframes into the teacher's self-contained frame format.

One directory per sample token, readable by ``qwen_drive_perception.dataset.PerceptionFrame``::

    <token>/frame.json   prompt content (view tag / image pairs), cam_order, dataset_type
    <token>/images/      symlinks to the original camera JPEGs
    <token>/calib.npz    cam_intrinsic, sensor2lidar_rotation, sensor2lidar_translation, lidar2ego
    <token>/gt.npz       boxes (N, 9) in the ego frame, 7-class labels (-1 = unmapped), nuScenes
                         category names, attributes and lidar+radar point counts

Reads the raw nuScenes JSON tables directly: nuscenes-devkit pins numpy<2, which conflicts with
the teacher's numpy 2.2.6.

Conventions follow the teacher's demo frames and mmdet3d's nuScenes infos:
  * ``sensor2lidar_*`` maps camera coordinates to the LIDAR_TOP frame at the lidar timestamp,
    going through the global frame so the camera's own ego pose is used (``obtain_sensor2top``).
  * GT boxes are ``[x, y, z, w, l, h, yaw, vx, vy]`` in the ego frame at the lidar timestamp, z at
    the box centre, ``w`` along the heading (= nuScenes length), velocity from neighbouring annotations.
    Box conventions are checked against the teacher's predictions, not assumed.
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import numpy as np

from qdistill.json_stream import iter_json_array
from qwen_drive_perception.geometry import _quaternion_rotation

CAM_ORDER = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT")
VIEW_TAGS = {
    "CAM_FRONT": "<FRONT VIEW>",
    "CAM_FRONT_RIGHT": "<FRONT RIGHT VIEW>",
    "CAM_BACK_RIGHT": "<BACK RIGHT VIEW>",
    "CAM_BACK": "<BACK VIEW>",
    "CAM_BACK_LEFT": "<BACK LEFT VIEW>",
    "CAM_FRONT_LEFT": "<FRONT LEFT VIEW>",
}
PROMPT_TEXT = "Analyze the scene."  # verbatim from the demo nuScenes frames

# nuScenes category -> teacher DET_CLASS_NAMES index, derived from which label the teacher predicts
# for each category on mini. czone_sign (1) has no nuScenes counterpart (nuPlan only).
# generic_object <- debris, pushable_pullable, bicycle_rack, animal as stated in the teacher's report
#; emergency vehicles and animal are absent from mini.
LABEL_MAP = {
    "vehicle.car": 0,
    "vehicle.truck": 0,
    "vehicle.bus.rigid": 0,
    "vehicle.bus.bendy": 0,
    "vehicle.construction": 0,
    "vehicle.trailer": 0,
    "vehicle.emergency.ambulance": 0,
    "vehicle.emergency.police": 0,
    "vehicle.bicycle": 2,
    "vehicle.motorcycle": 2,
    "movable_object.pushable_pullable": 3,
    "movable_object.debris": 3,
    "static_object.bicycle_rack": 3,
    "animal": 3,
    "human.pedestrian.adult": 4,
    "human.pedestrian.child": 4,
    "human.pedestrian.construction_worker": 4,
    "human.pedestrian.personal_mobility": 4,
    "human.pedestrian.police_officer": 4,
    "human.pedestrian.stroller": 4,
    "human.pedestrian.wheelchair": 4,
    "movable_object.trafficcone": 5,
    "movable_object.barrier": 6,
}


def _pose(record: dict) -> tuple[np.ndarray, np.ndarray]:
    return _quaternion_rotation(record["rotation"]), np.asarray(record["translation"], dtype=np.float64)


def _yaw(rotation: np.ndarray) -> float:
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


class NuScenesTables:
    """Minimal index over the nuScenes JSON tables.

    ``scene_names`` restricts every table to those scenes, and the tables are streamed
    (``json_stream``) so only the kept rows are held in memory: trainval's sample_data.json (1.28 GB),
    ego_pose.json and sample_annotation.json do not fit in 7.6 GB of RAM as Python objects.
    ``keyframes_only`` additionally drops non-keyframe sample_data (sweeps).
    """

    def __init__(self, dataroot: str | Path, version: str = "v1.0-mini", scene_names=None, keyframes_only: bool = False):
        self.dataroot = Path(dataroot)
        self.version = version
        self.scene_names = None if scene_names is None else set(scene_names)
        self.keyframes_only = keyframes_only

    def _table(self, name: str):
        return iter_json_array(self.dataroot / self.version / f"{name}.json")

    def _index(self, name: str, keep=None) -> dict[str, dict]:
        return {r["token"]: r for r in self._table(name) if keep is None or keep(r)}

    @cached_property
    def scene(self):
        return self._index("scene", None if self.scene_names is None else lambda r: r["name"] in self.scene_names)

    @cached_property
    def sample(self):
        return self._index("sample", lambda r: r["scene_token"] in self.scene)

    @cached_property
    def sample_data(self):
        return self._index("sample_data", lambda r: r["sample_token"] in self.sample
                           and (r["is_key_frame"] or not self.keyframes_only))

    @cached_property
    def calibrated_sensor(self):
        return self._index("calibrated_sensor")

    @cached_property
    def ego_pose(self):
        used = {sd["ego_pose_token"] for sd in self.sample_data.values()}
        return self._index("ego_pose", lambda r: r["token"] in used)

    @cached_property
    def sample_annotation(self):
        return self._index("sample_annotation", lambda r: r["sample_token"] in self.sample)

    @cached_property
    def instance(self):
        return self._index("instance")

    @cached_property
    def category(self):
        return self._index("category")

    @cached_property
    def attribute(self):
        return self._index("attribute")

    @cached_property
    def sample_anns(self) -> dict[str, list[str]]:
        """sample_token -> annotation tokens (the devkit adds this as ``sample['anns']``)."""
        out: dict[str, list[str]] = {t: [] for t in self.sample}
        for ann in self.sample_annotation.values():
            out[ann["sample_token"]].append(ann["token"])
        return out

    @cached_property
    def sd_channel(self) -> dict[str, str]:
        """sample_data token -> sensor channel."""
        sensor_channel = {r["token"]: r["channel"] for r in self._table("sensor")}
        cs_channel = {t: sensor_channel[r["sensor_token"]] for t, r in self.calibrated_sensor.items()}
        return {t: cs_channel[sd["calibrated_sensor_token"]] for t, sd in self.sample_data.items()}

    @cached_property
    def keyframe_data(self) -> dict[tuple[str, str], dict]:
        """(sample_token, channel) -> keyframe sample_data record."""
        return {
            (sd["sample_token"], self.sd_channel[t]): sd for t, sd in self.sample_data.items() if sd["is_key_frame"]
        }

    def samples_in_order(self) -> list[str]:
        """Sample tokens scene by scene, in time order."""
        tokens = []
        for scene in sorted(self.scene.values(), key=lambda s: s["name"]):
            tok = scene["first_sample_token"]
            while tok:
                tokens.append(tok)
                tok = self.sample[tok]["next"]
        return tokens

    def sensor_to_global(self, sd: dict) -> tuple[np.ndarray, np.ndarray]:
        """(R, t) such that p_global = R @ p_sensor + t."""
        cs_r, cs_t = _pose(self.calibrated_sensor[sd["calibrated_sensor_token"]])
        ep_r, ep_t = _pose(self.ego_pose[sd["ego_pose_token"]])
        return ep_r @ cs_r, ep_r @ cs_t + ep_t

    def annotation_velocity(self, ann: dict, max_time_diff: float = 1.5) -> np.ndarray:
        """Global-frame xy velocity from neighbouring annotations, as nuscenes-devkit ``box_velocity``."""
        has_prev, has_next = bool(ann["prev"]), bool(ann["next"])
        if not (has_prev or has_next):
            return np.full(2, np.nan)
        first = self.sample_annotation[ann["prev"]] if has_prev else ann
        last = self.sample_annotation[ann["next"]] if has_next else ann
        t_first = self.sample[first["sample_token"]]["timestamp"] * 1e-6
        t_last = self.sample[last["sample_token"]]["timestamp"] * 1e-6
        dt = t_last - t_first
        if has_prev and has_next:
            max_time_diff *= 2
        if dt <= 0 or dt > max_time_diff:
            return np.full(2, np.nan)
        return (np.asarray(last["translation"][:2]) - np.asarray(first["translation"][:2])) / dt


def _write_rig(tables: NuScenesTables, out: Path, lidar_sd: dict, cam_sds: dict[str, dict]):
    """Write images (symlinks) + calib.npz for one 6-camera frame; return the ego->global pose."""
    (out / "images").mkdir(parents=True, exist_ok=True)
    l2e_r, l2e_t = _pose(tables.calibrated_sensor[lidar_sd["calibrated_sensor_token"]])
    e2g_r, e2g_t = _pose(tables.ego_pose[lidar_sd["ego_pose_token"]])
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3], lidar2ego[:3, 3] = l2e_r, l2e_t
    lidar_to_global_r = e2g_r @ l2e_r
    lidar_to_global_t = e2g_r @ l2e_t + e2g_t

    intrinsics, s2l_r, s2l_t = [], [], []
    for cam in CAM_ORDER:
        sd = cam_sds[cam]
        intrinsics.append(np.asarray(tables.calibrated_sensor[sd["calibrated_sensor_token"]]["camera_intrinsic"]))
        cam_r, cam_t = tables.sensor_to_global(sd)
        # camera -> global -> lidar: p_lidar = Rl^T (Rc p + tc - tl)
        s2l_r.append(lidar_to_global_r.T @ cam_r)
        s2l_t.append(lidar_to_global_r.T @ (cam_t - lidar_to_global_t))
        link = out / "images" / f"{cam}.jpg"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to((tables.dataroot / sd["filename"]).resolve())

    np.savez(
        out / "calib.npz",
        cam_intrinsic=np.stack(intrinsics).astype(np.float32),
        sensor2lidar_rotation=np.stack(s2l_r).astype(np.float32),
        sensor2lidar_translation=np.stack(s2l_t).astype(np.float32),
        lidar2ego=lidar2ego.astype(np.float32),
    )
    return e2g_r, e2g_t


def _write_frame_json(out: Path, **extra) -> None:
    content = []
    for cam in CAM_ORDER:
        content += [{"text": VIEW_TAGS[cam]}, {"image": cam}]
    content.append({"text": PROMPT_TEXT})
    frame = {"dataset_type": "nuscenes", "cam_order": list(CAM_ORDER), "content": content, **extra}
    (out / "frame.json").write_text(json.dumps(frame, indent=2))


def _write_empty_gt(out: Path) -> None:
    np.savez(out / "gt.npz", boxes=np.zeros((0, 9), np.float32), labels=np.zeros(0, np.int64),
             categories=np.zeros(0, dtype=str), attributes=np.zeros(0, dtype=str), num_pts=np.zeros(0, np.int64))


def pack_sample(tables: NuScenesTables, sample_token: str, out_root: str | Path) -> Path:
    """Write one packed keyframe directory (with GT boxes) and return its path."""
    out = Path(out_root) / sample_token
    lidar_sd = tables.keyframe_data[(sample_token, "LIDAR_TOP")]
    cam_sds = {cam: tables.keyframe_data[(sample_token, cam)] for cam in CAM_ORDER}
    e2g_r, e2g_t = _write_rig(tables, out, lidar_sd, cam_sds)

    # GT boxes in the ego frame at the lidar timestamp.
    boxes, categories, attributes, num_pts = [], [], [], []
    for ann_token in tables.sample_anns[sample_token]:
        ann = tables.sample_annotation[ann_token]
        rot_g = _quaternion_rotation(ann["rotation"])
        center = e2g_r.T @ (np.asarray(ann["translation"]) - e2g_t)
        yaw = _yaw(e2g_r.T @ rot_g)
        vel_g = tables.annotation_velocity(ann)
        vel = (e2g_r.T @ np.array([vel_g[0], vel_g[1], 0.0]))[:2]
        # nuScenes size is (width, length, height); the teacher's w is the extent along the heading,
        # i.e. nuScenes length (verified in size error 0.25 m swapped vs 3.2 m unswapped).
        width, length, height = ann["size"]
        boxes.append([*center, length, width, height, yaw, *vel])
        categories.append(tables.category[tables.instance[ann["instance_token"]]["category_token"]]["name"])
        attributes.append(",".join(tables.attribute[a]["name"] for a in ann["attribute_tokens"]))
        num_pts.append(ann["num_lidar_pts"] + ann["num_radar_pts"])
    np.savez(
        out / "gt.npz",
        boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 9),
        labels=np.asarray([LABEL_MAP.get(c, -1) for c in categories], dtype=np.int64),
        categories=np.asarray(categories, dtype=str),
        attributes=np.asarray(attributes, dtype=str),
        num_pts=np.asarray(num_pts, dtype=np.int64),
    )
    sample = tables.sample[sample_token]
    _write_frame_json(out, scene_token=sample["scene_token"], timestamp=sample["timestamp"], labeled=True)
    return out
