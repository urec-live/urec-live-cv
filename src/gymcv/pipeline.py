from __future__ import annotations

import json
import math
import time
import warnings
import requests
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

from gymcv.motion import MotionConfig, compute_pose_motion_score, is_motion_active
from gymcv.usage import EquipmentState, EquipmentStatus, build_status_payload, init_states, update_equipment_state
from gymcv.zones import Zone, load_zones_from_config


BACKEND_URL = "http://localhost:8080/api/equipment/status"

def send_to_backend(payload):
    try:
        response = requests.post(BACKEND_URL, json={
            "equipmentId": payload.equipment_id,
            "equipmentType": payload.equipment_type,
            "status": payload.status,
            "confidence": payload.confidence,
            "timestamp": payload.timestamp
        })
        print(f"Sent: {payload.equipment_id} -> {payload.status}")
    except Exception as e:
        print(f"Error sending to backend: {e}")  # Example backend URL for status updates

@dataclass
class PipelineConfig:
    camera_id: str
    frame_interval_sec: float = 0.3
    in_use_persist_sec: float = 3.0
    available_cooldown_sec: float = 2.0
    # Optional: override video_path when set (webcam index, rtsp URL, or file path string).
    input_source: Optional[str] = None
    # Extra frames to discard between processed frames (0 = none). Improves live FPS.
    frame_skip: int = 0
    debug_mode: bool = False
    # If True, show camera preview on every captured frame, while running inference at frame_interval_sec.
    preview_every_frame: bool = False
    # Demo mode: treat "person in zone" as active interaction (no motion needed).
    presence_only_mode: bool = False
    # If set in JSON, overrides run_usage_pipeline_on_video() defaults for that run.
    det_model_path: Optional[str] = None
    pose_model_path: Optional[str] = None
    motion: MotionConfig = field(default_factory=MotionConfig)
    zones: List[Zone] = field(default_factory=list)


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    zones = load_zones_from_config(data.get("equipment_zones", []))
    motion_cfg = data.get("motion", {}) or {}
    inf = data.get("inference") or {}
    profile = str(data.get("inference_profile", inf.get("profile", ""))).lower()
    det_default = "yolov8n.pt" if profile == "fast" else None
    pose_default = "yolov8n-pose.pt" if profile == "fast" else None
    _raw_input = data.get("input_source") or data.get("video_source") or inf.get("input_source")
    _input_source = str(_raw_input).strip() if _raw_input is not None else ""
    return PipelineConfig(
        camera_id=str(data.get("camera_id", "cam_01")),
        frame_interval_sec=float(data.get("frame_interval_sec", 0.5)),
        in_use_persist_sec=float(data.get("in_use_persist_sec", 3.0)),
        available_cooldown_sec=float(data.get("available_cooldown_sec", 2.0)),
        input_source=_input_source if _input_source else None,
        frame_skip=int(data.get("frame_skip", inf.get("frame_skip", 0))),
        debug_mode=bool(data.get("debug_mode", inf.get("debug_mode", False))),
        preview_every_frame=bool(data.get("preview_every_frame", inf.get("preview_every_frame", False))),
        presence_only_mode=bool(data.get("presence_only_mode", inf.get("presence_only_mode", False))),
        det_model_path=(data.get("det_model_path") or inf.get("det_model") or det_default),
        pose_model_path=(data.get("pose_model_path") or inf.get("pose_model") or pose_default),
        motion=MotionConfig(
            keypoints_to_use=tuple(motion_cfg.get("keypoints_to_use", MotionConfig().keypoints_to_use)),
            min_mean_pixel_movement=float(motion_cfg.get("min_mean_pixel_movement", 10.0)),
        ),
        zones=zones,
    )


def _xyxy_center(xyxy: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy.tolist()
    return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)


def _normalize_capture_source(video_path: Union[str, Path, int]) -> Tuple[Union[str, int], bool]:
    """
    Returns (source_for_cv2.VideoCapture, is_live).
    Live: webcam index, RTSP/HTTP URL, or common aliases.
    File: existing file path as string.
    """
    if isinstance(video_path, int):
        return video_path, True
    if isinstance(video_path, Path):
        if video_path.is_file():
            return str(video_path.resolve()), False
        s = str(video_path).strip()
    else:
        s = str(video_path).strip()
    if not s:
        raise ValueError("video source is empty")
    low = s.lower()
    if low in ("0", "webcam", "camera"):
        return 0, True
    if s.isdigit():
        return int(s), True
    if low.startswith(("rtsp://", "http://", "https://")):
        return s, True
    if s.startswith("/dev/"):
        return s, True
    p = Path(s)
    if p.is_file():
        return str(p.resolve()), False
    return s, False


def _read_frame_with_skip(cap: cv2.VideoCapture, frame_skip: int) -> Tuple[bool, Optional[np.ndarray]]:
    """Read 1 + frame_skip frames; return last good frame (performance knob for live streams)."""
    last: Optional[np.ndarray] = None
    for _ in range(1 + max(0, int(frame_skip))):
        ok, frame = cap.read()
        if not ok:
            return False, None
        last = frame
    return True, last


def _open_capture_with_fallbacks(cap_source: Union[str, int], is_live: bool) -> cv2.VideoCapture:
    """
    Windows note: OpenCV's default MSMF backend can intermittently fail grabbing webcam frames.
    Prefer DirectShow for webcam indices when available.
    """
    # Webcam index: try DirectShow first on Windows, then default.
    if isinstance(cap_source, int) and is_live:
        backends: list[Optional[int]] = []
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(int(cv2.CAP_DSHOW))
        backends.append(None)

        for b in backends:
            cap = cv2.VideoCapture(cap_source) if b is None else cv2.VideoCapture(cap_source, b)
            if cap.isOpened():
                return cap
            cap.release()

        return cv2.VideoCapture(cap_source)

    # URL/file: let OpenCV pick the best backend (FFmpeg if built).
    return cv2.VideoCapture(cap_source)


def _iter_video_frames(
    video_path: Union[str, Path, int],
    *,
    frame_interval_sec: float,
    frame_skip: int = 0,
    max_live_read_failures: int = 300,
) -> Iterator[Tuple[float, np.ndarray]]:
    cap_source, is_live = _normalize_capture_source(video_path)
    cap = _open_capture_with_fallbacks(cap_source, is_live=is_live)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video/stream: {cap_source!r}")

    try:
        # Time-based sampling is more stable than FPS-based stepping for live streams.
        interval = max(0.0, float(frame_interval_sec))
        next_emit_ts: Optional[float] = None
        consecutive_failures = 0

        while True:
            ok, frame = _read_frame_with_skip(cap, frame_skip)
            if not ok:
                consecutive_failures += 1
                if is_live:
                    if consecutive_failures == 1 or consecutive_failures % 30 == 0:
                        warnings.warn(
                            f"Frame read failed ({consecutive_failures}x); stream may be unstable.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    if consecutive_failures >= max_live_read_failures:
                        warnings.warn(
                            "Too many consecutive read failures; stopping capture loop.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        break
                    time.sleep(0.01)
                    continue
                break

            consecutive_failures = 0
            if is_live:
                ts = float(time.time())
            else:
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                ts = float(pos_ms) / 1000.0 if pos_ms and pos_ms > 0 else float(time.time())

            if interval <= 0.0:
                yield ts, frame
                continue

            if next_emit_ts is None:
                next_emit_ts = ts

            if ts >= next_emit_ts:
                yield ts, frame
                # Avoid drift if ts jumps (e.g. RTSP hiccup)
                steps = max(1, int((ts - next_emit_ts) / interval) + 1)
                next_emit_ts = next_emit_ts + steps * interval
    finally:
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


def _extract_pose_keypoints_list(pose_result) -> List[np.ndarray]:
    if pose_result is None or pose_result.keypoints is None:
        return []
    kps = pose_result.keypoints
    if kps.xy is None or len(kps.xy) == 0:
        return []
    return [kps.xy[i].cpu().numpy() for i in range(len(kps.xy))]


def _keypoints_center(kps_xy: np.ndarray) -> Tuple[float, float]:
    # Center from valid keypoints only.
    if kps_xy.size == 0:
        return 0.0, 0.0
    valid = np.isfinite(kps_xy).all(axis=1)
    pts = kps_xy[valid]
    if pts.size == 0:
        return 0.0, 0.0
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def _zone_center(z: Zone) -> Tuple[float, float]:
    return (z.x1 + z.x2) / 2.0, (z.y1 + z.y2) / 2.0


def _extract_pose_people_with_scores(pose_result) -> List[Tuple[np.ndarray, float]]:
    """All detected people as (keypoints_xy, mean_keypoint_confidence)."""
    if pose_result is None or pose_result.keypoints is None:
        return []
    kps = pose_result.keypoints
    if kps.xy is None or len(kps.xy) == 0:
        return []
    out: List[Tuple[np.ndarray, float]] = []
    for i in range(len(kps.xy)):
        xy = kps.xy[i].cpu().numpy()
        score = 0.0
        if hasattr(kps, "conf") and kps.conf is not None and len(kps.conf) > i:
            score = float(np.nanmean(kps.conf[i].cpu().numpy()))
        out.append((xy, score))
    return out


def _best_pose_keypoints_in_zone(z: Zone, people: List[Tuple[np.ndarray, float]]) -> Optional[np.ndarray]:
    """
    Pick the best pose inside a zone: closest keypoint-center to zone center,
    tie-break by higher mean keypoint confidence.
    """
    zcx, zcy = _zone_center(z)
    candidates: List[Tuple[float, float, np.ndarray]] = []
    for kps_xy, pconf in people:
        cx, cy = _keypoints_center(kps_xy)
        if not z.contains_point(cx, cy):
            continue
        dist = math.hypot(cx - zcx, cy - zcy)
        candidates.append((dist, -float(pconf), kps_xy))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[0][2]


def _draw_debug_overlay(
    frame: np.ndarray,
    *,
    zones: List[Zone],
    status_by_equipment: Dict[str, str],
    conf_by_equipment: Dict[str, float],
    fps_text: str,
) -> np.ndarray:
    out = frame.copy()
    for z in zones:
        x1, y1, x2, y2 = z.as_xyxy()
        st = status_by_equipment.get(z.equipment_id, "?")
        cf = float(conf_by_equipment.get(z.equipment_id, 0.0))
        is_in_use = st == "IN_USE"
        color = (0, 80, 240) if is_in_use else (0, 200, 0)
        # Semi-transparent fill + crisp border for more precise visual guidance.
        fill = out.copy()
        cv2.rectangle(fill, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
        out = cv2.addWeighted(fill, 0.12, out, 0.88, 0)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        # Corner ticks improve perceived rectangle precision on projected screens.
        tick = 12
        cv2.line(out, (int(x1), int(y1)), (int(x1 + tick), int(y1)), color, 2)
        cv2.line(out, (int(x1), int(y1)), (int(x1), int(y1 + tick)), color, 2)
        cv2.line(out, (int(x2), int(y1)), (int(x2 - tick), int(y1)), color, 2)
        cv2.line(out, (int(x2), int(y1)), (int(x2), int(y1 + tick)), color, 2)
        cv2.line(out, (int(x1), int(y2)), (int(x1 + tick), int(y2)), color, 2)
        cv2.line(out, (int(x1), int(y2)), (int(x1), int(y2 - tick)), color, 2)
        cv2.line(out, (int(x2), int(y2)), (int(x2 - tick), int(y2)), color, 2)
        cv2.line(out, (int(x2), int(y2)), (int(x2), int(y2 - tick)), color, 2)
        label = f"{z.equipment_id}: {st} {cf:.2f}"
        cv2.putText(
            out,
            label,
            (int(x1), max(22, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        out,
        fps_text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )
    return out


@lru_cache(maxsize=8)
def _load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def _iso_from_epoch(epoch_ts: float) -> str:
    return datetime.fromtimestamp(epoch_ts, tz=timezone.utc).isoformat()


def run_usage_pipeline_on_video(
    *,
    video_path: Union[str, Path, int],
    cfg: PipelineConfig,
    det_model_path: str = "yolov8x.pt",
    pose_model_path: str = "yolov8x-pose.pt",
    device: Optional[str] = None,
) -> Iterator[EquipmentStatus]:
    """
    Yields EquipmentStatus updates whenever a zone's status changes.

    This MVP determines "in-zone" via person bbox center in the zone rectangle.
    It determines "motion" via mean keypoint displacement between sampled frames.

    ``video_path`` may be a file path, ``Path``, webcam index (``0``), ``"webcam"``, or an RTSP/HTTP URL string.
    If ``cfg.input_source`` is set, it overrides ``video_path`` for capture only.
    """
    capture_arg: Union[str, Path, int] = cfg.input_source if cfg.input_source else video_path
    try:
        _normalize_capture_source(capture_arg)
    except Exception as e:
        raise RuntimeError(f"Invalid video/stream source: {capture_arg!r}") from e

    dm = cfg.det_model_path or det_model_path
    pm = cfg.pose_model_path or pose_model_path
    det_model = _load_model(dm)
    pose_model = _load_model(pm)

    equipment_ids = [z.equipment_id for z in cfg.zones]
    states: Dict[str, EquipmentState] = init_states(equipment_ids)
    last_payloads: Dict[str, Optional[EquipmentStatus]] = {eid: None for eid in equipment_ids}

    prev_pose_kps_by_zone: Dict[str, Optional[np.ndarray]] = {z.equipment_id: None for z in cfg.zones}

    fps_ema = 0.0
    last_loop_t = time.perf_counter()
    debug_window = "gymcv debug"
    next_infer_ts: Optional[float] = None
    last_status_by_id: Dict[str, str] = {z.equipment_id: "AVAILABLE" for z in cfg.zones}
    last_conf_by_id: Dict[str, float] = {z.equipment_id: 0.2 for z in cfg.zones}
    
    # Track when the last person was detected to handle 10-second no-person timeout
    last_person_detected_ts: Optional[float] = None
    no_person_timeout_sec = 10.0

    try:
        # If preview_every_frame is enabled, capture every frame but gate inference by frame_interval_sec.
        capture_interval = 0.0 if (cfg.debug_mode and cfg.preview_every_frame) else cfg.frame_interval_sec
        for video_ts, frame in _iter_video_frames(
            capture_arg,
            frame_interval_sec=capture_interval,
            frame_skip=cfg.frame_skip,
        ):
            # Keep state and payload timestamps consistent on system time.
            now_ts = float(time.time())
            payload_timestamp = _iso_from_epoch(now_ts)

            do_infer = True
            if cfg.debug_mode and cfg.preview_every_frame and cfg.frame_interval_sec > 0:
                if next_infer_ts is None:
                    next_infer_ts = now_ts
                do_infer = now_ts >= next_infer_ts
                if do_infer:
                    next_infer_ts = now_ts + float(cfg.frame_interval_sec)

            if do_infer:
                det_results = det_model.predict(source=frame, verbose=False, device=device)
                pose_results = pose_model.predict(source=frame, verbose=False, device=device)

                det_r0 = det_results[0] if det_results else None
                pose_r0 = pose_results[0] if pose_results else None

                person_boxes = _extract_person_boxes(det_r0)
                pose_people_scored = _extract_pose_people_with_scores(pose_r0)

                # Track when the last person was detected
                if len(person_boxes) > 0:
                    last_person_detected_ts = now_ts
                
                # Check if no person detected for more than 10 seconds
                person_disappeared = False
                if last_person_detected_ts is not None and (now_ts - last_person_detected_ts) > no_person_timeout_sec:
                    person_disappeared = True

                # Zone occupancy: any detected person center inside zone.
                zone_occupied: Dict[str, bool] = {z.equipment_id: False for z in cfg.zones}
                for pb in person_boxes:
                    cx, cy = _xyxy_center(pb)
                    for z in cfg.zones:
                        if z.contains_point(cx, cy):
                            zone_occupied[z.equipment_id] = True

                zone_motion_active: Dict[str, bool] = {z.equipment_id: False for z in cfg.zones}
                for z in cfg.zones:
                    curr_zone_kps = _best_pose_keypoints_in_zone(z, pose_people_scored)
                    prev_zone_kps = prev_pose_kps_by_zone[z.equipment_id]
                    motion_score = compute_pose_motion_score(prev_zone_kps, curr_zone_kps, cfg.motion)
                    zone_motion_active[z.equipment_id] = is_motion_active(motion_score, cfg.motion)
                    prev_pose_kps_by_zone[z.equipment_id] = curr_zone_kps

                conf_by_id: Dict[str, float] = {}
                for z in cfg.zones:
                    st = states[z.equipment_id]
                    person_here = zone_occupied[z.equipment_id]
                    motion_here = zone_motion_active[z.equipment_id]
                    interaction_active = person_here if cfg.presence_only_mode else motion_here
                    states[z.equipment_id] = update_equipment_state(
                        state=st,
                        now_ts=now_ts,
                        person_in_zone=person_here,
                        motion_active=interaction_active,
                        persist_sec=cfg.in_use_persist_sec,
                        available_cooldown_sec=cfg.available_cooldown_sec,
                        presence_only_mode=cfg.presence_only_mode,
                    )
                    
                    # Force AVAILABLE if no person detected for >10 seconds
                    if person_disappeared:
                        states[z.equipment_id].status = "AVAILABLE"
                        states[z.equipment_id].in_use_since_ts = None

                    if person_here and interaction_active:
                        conf = 0.9
                    elif person_here or interaction_active:
                        conf = 0.6
                    else:
                        conf = 0.2
                    conf_by_id[z.equipment_id] = float(max(0.0, min(1.0, conf)))

                last_status_by_id = {z.equipment_id: states[z.equipment_id].status for z in cfg.zones}
                last_conf_by_id = conf_by_id

            if cfg.debug_mode:
                now_loop = time.perf_counter()
                dt = now_loop - last_loop_t
                last_loop_t = now_loop
                if dt > 1e-6:
                    inst = 1.0 / dt
                    fps_ema = inst if fps_ema <= 0 else 0.9 * fps_ema + 0.1 * inst
                vis = _draw_debug_overlay(
                    frame,
                    zones=cfg.zones,
                    status_by_equipment=last_status_by_id,
                    conf_by_equipment=last_conf_by_id,
                    fps_text=f"FPS~{fps_ema:.1f}",
                )
                cv2.imshow(debug_window, vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if do_infer:
                for z in cfg.zones:
                    conf = last_conf_by_id.get(z.equipment_id, 0.2)
                    payload = build_status_payload(
                        equipment_id=z.equipment_id,
                        equipment_type=z.equipment_type,
                        status=states[z.equipment_id].status,
                        confidence=conf,
                        timestamp=payload_timestamp,
                    )
                    if last_payloads[z.equipment_id] is None or payload.status != last_payloads[z.equipment_id].status:
                        last_payloads[z.equipment_id] = payload
                        send_to_backend(payload)
                        yield payload
    finally:
        if cfg.debug_mode:
            try:
                cv2.destroyWindow(debug_window)
            except Exception:
                pass

