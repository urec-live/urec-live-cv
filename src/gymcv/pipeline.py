from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from gymcv.motion import MotionConfig, compute_pose_motion_score, is_motion_active
from gymcv.usage import EquipmentState, EquipmentStatus, build_status_payload, init_states, update_equipment_state
from gymcv.zones import Zone, load_zones_from_config


@dataclass
class PipelineConfig:
    camera_id: str
    frame_interval_sec: float = 0.5
    in_use_persist_sec: float = 3.0
    motion: MotionConfig = field(default_factory=MotionConfig)
    zones: List[Zone] = field(default_factory=list)


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    zones = load_zones_from_config(data.get("equipment_zones", []))
    motion_cfg = data.get("motion", {}) or {}
    return PipelineConfig(
        camera_id=str(data.get("camera_id", "cam_01")),
        frame_interval_sec=float(data.get("frame_interval_sec", 0.5)),
        in_use_persist_sec=float(data.get("in_use_persist_sec", 3.0)),
        motion=MotionConfig(
            keypoints_to_use=tuple(motion_cfg.get("keypoints_to_use", MotionConfig().keypoints_to_use)),
            min_mean_pixel_movement=float(motion_cfg.get("min_mean_pixel_movement", 10.0)),
        ),
        zones=zones,
    )


def _xyxy_center(xyxy: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy.tolist()
    return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)


def _iter_video_frames(
    video_path: str | Path,
    *,
    frame_interval_sec: float,
) -> Iterator[Tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0
    frame_step = max(1, int(round(frame_interval_sec * fps)))
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_step == 0:
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            yield ts, frame
        frame_idx += 1

    cap.release()


def _extract_person_boxes(det_result) -> List[np.ndarray]:
    """
    Returns list of person boxes (xyxy) as float arrays shape (4,).
    Works with Ultralytics YOLO Results.
    """
    if det_result is None or det_result.boxes is None:
        return []
    boxes = det_result.boxes
    if boxes.cls is None or boxes.xyxy is None:
        return []
    cls = boxes.cls.cpu().numpy().astype(int)
    xyxy = boxes.xyxy.cpu().numpy()
    # COCO class 0 = person
    persons = [xyxy[i] for i in range(len(cls)) if cls[i] == 0]
    return persons


def _extract_first_pose_keypoints(pose_result) -> Optional[np.ndarray]:
    """
    Returns keypoints (17,2) for the most confident detected person, or None.
    """
    if pose_result is None or pose_result.keypoints is None:
        return None
    kps = pose_result.keypoints
    if kps.xy is None:
        return None
    xy = kps.xy
    if len(xy) == 0:
        return None
    # Select person with highest mean keypoint confidence if available.
    if hasattr(kps, "conf") and kps.conf is not None and len(kps.conf) == len(xy):
        conf = kps.conf.cpu().numpy()
        scores = conf.mean(axis=1)
        best = int(np.argmax(scores))
    else:
        best = 0
    return xy[best].cpu().numpy()


def run_usage_pipeline_on_video(
    *,
    video_path: str | Path,
    cfg: PipelineConfig,
    det_model_path: str = "yolov8x.pt",
    pose_model_path: str = "yolov8x-pose.pt",
    device: Optional[str] = None,
) -> Iterator[EquipmentStatus]:
    """
    Yields EquipmentStatus updates whenever a zone's status changes.

    This MVP determines "in-zone" via person bbox center in the zone rectangle.
    It determines "motion" via mean keypoint displacement between sampled frames.
    """
    det_model = YOLO(det_model_path)
    pose_model = YOLO(pose_model_path)

    equipment_ids = [z.equipment_id for z in cfg.zones]
    states: Dict[str, EquipmentState] = init_states(equipment_ids)
    last_payloads: Dict[str, Optional[EquipmentStatus]] = {eid: None for eid in equipment_ids}

    prev_pose_kps: Optional[np.ndarray] = None

    for video_ts, frame in _iter_video_frames(video_path, frame_interval_sec=cfg.frame_interval_sec):
        # In dev, treat video timestamp as "now"; in production you would use time.time().
        now_ts = float(video_ts)

        det_results = det_model.predict(source=frame, verbose=False, device=device)
        pose_results = pose_model.predict(source=frame, verbose=False, device=device)

        det_r0 = det_results[0] if det_results else None
        pose_r0 = pose_results[0] if pose_results else None

        person_boxes = _extract_person_boxes(det_r0)
        curr_pose_kps = _extract_first_pose_keypoints(pose_r0)

        motion_score = compute_pose_motion_score(prev_pose_kps, curr_pose_kps, cfg.motion)
        motion_active = is_motion_active(motion_score, cfg.motion)
        prev_pose_kps = curr_pose_kps

        # Zone occupancy: any detected person center inside zone.
        zone_occupied: Dict[str, bool] = {z.equipment_id: False for z in cfg.zones}
        for pb in person_boxes:
            cx, cy = _xyxy_center(pb)
            for z in cfg.zones:
                if z.contains_point(cx, cy):
                    zone_occupied[z.equipment_id] = True

        for z in cfg.zones:
            st = states[z.equipment_id]
            states[z.equipment_id] = update_equipment_state(
                state=st,
                now_ts=now_ts,
                person_in_zone=zone_occupied[z.equipment_id],
                motion_active=motion_active,
                persist_sec=cfg.in_use_persist_sec,
            )

            # Confidence: simple heuristic based on motion_score and occupancy.
            conf = 0.5
            if zone_occupied[z.equipment_id]:
                conf += 0.25
            conf += min(0.25, motion_score / max(cfg.motion.min_mean_pixel_movement, 1.0) * 0.25)
            conf = float(max(0.0, min(1.0, conf)))

            payload = build_status_payload(
                equipment_id=z.equipment_id,
                equipment_type=z.equipment_type,
                status=states[z.equipment_id].status,
                confidence=conf,
            )
            if last_payloads[z.equipment_id] is None or payload.status != last_payloads[z.equipment_id].status:
                last_payloads[z.equipment_id] = payload
                yield payload

