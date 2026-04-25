from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gymcv.pipeline import PipelineConfig, load_pipeline_config, run_usage_pipeline_on_camera
from gymcv.usage import EquipmentState
from gymcv.zones import Zone

DEFAULT_ZONES = ROOT / "configs" / "laptop_zones.json"

# Sentinel to break out of the pipeline loop from within on_frame callback
class _Quit(Exception):
    pass


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post_json(url: str, data: dict, timeout_sec: float = 2.0) -> None:
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        resp.read()


def _put_springboot_status(springboot_url: str, equipment_id: str, cv_status: str) -> None:
    """Direct PUT to Spring Boot. cv_status is 'IN_USE' or 'AVAILABLE'."""
    sb_status = "In Use" if cv_status == "IN_USE" else "Available"
    url = f"{springboot_url.rstrip('/')}/api/machines/code/{equipment_id}/status"
    body = json.dumps({"status": sb_status}).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    with urllib.request.urlopen(req, timeout=2.0) as resp:
        resp.read()


# ---------------------------------------------------------------------------
# Frame annotation
# ---------------------------------------------------------------------------

def _draw_annotations(
    frame: np.ndarray,
    zones: List[Zone],
    states: Dict[str, EquipmentState],
    person_boxes: List[np.ndarray],
) -> None:
    # Person bounding boxes — yellow
    for pb in person_boxes:
        x1, y1, x2, y2 = (int(v) for v in pb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Zone rectangles: green = AVAILABLE, red = IN_USE
    for z in zones:
        st = states[z.equipment_id].status
        color = (0, 0, 255) if st == "IN_USE" else (0, 200, 0)
        cv2.rectangle(frame, (z.x1, z.y1), (z.x2, z.y2), color, 2)
        label = f"{z.equipment_id}: {st}"
        cv2.putText(frame, label, (z.x1 + 5, z.y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Status bar
    cv2.putText(frame, "q = quit", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# ---------------------------------------------------------------------------
# Calibrate mode
# ---------------------------------------------------------------------------

def _run_calibrate(camera_index: int) -> None:
    """Show live camera feed with mouse XY overlay — no YOLO. Use to discover zone coords."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    mouse_pos = [0, 0]

    def _on_mouse(event, x, y, flags, param):  # noqa: ANN001
        mouse_pos[0] = x
        mouse_pos[1] = y

    win = "Calibrate — move mouse, note coords, press q to quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, _on_mouse)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            x, y = mouse_pos
            cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)
            cv2.putText(frame, f"x={x}  y={y}", (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(
                frame,
                "Move mouse over zone corners | q=quit",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.imshow(win, frame)
            print(f"\rx={x:4d}  y={y:4d}", end="", flush=True)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        print()  # newline after \r coords
        cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main pipeline loop
# ---------------------------------------------------------------------------

def _run_pipeline(
    cfg: PipelineConfig,
    camera_index: int,
    det_model: str,
    device: Optional[str],
    post_url: Optional[str],
    springboot_url: Optional[str],
) -> None:
    win = "UREC Live CV"
    cv2.namedWindow(win)

    def on_frame(frame: np.ndarray, zones: List[Zone], states: Dict[str, EquipmentState], person_boxes: List[np.ndarray]) -> None:
        _draw_annotations(frame, zones, states, person_boxes)
        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise _Quit()

    try:
        for status in run_usage_pipeline_on_camera(
            cfg=cfg,
            det_model_path=det_model,
            device=device,
            camera_index=camera_index,
            on_frame=on_frame,
        ):
            payload = asdict(status)
            print(json.dumps(payload, ensure_ascii=False))

            if post_url:
                try:
                    _post_json(post_url, payload)
                except Exception as exc:
                    print(f"POST failed: {exc}", file=sys.stderr)

            if springboot_url:
                try:
                    _put_springboot_status(springboot_url, status.equipment_id, status.status)
                except Exception as exc:
                    print(f"Spring Boot PUT failed: {exc}", file=sys.stderr)

    except _Quit:
        pass
    finally:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="UREC Live — real-time webcam CV pipeline")
    parser.add_argument("--zones", default=str(DEFAULT_ZONES), help="Zones JSON config (default: configs/laptop_zones.json)")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument("--det-model", default="yolov8n.pt", help="Ultralytics model (default: yolov8n.pt)")
    parser.add_argument("--device", default=None, help="Inference device, e.g. 'cpu' or '0' for GPU")
    parser.add_argument("--post-url", default=None, help="POST status changes here, e.g. http://127.0.0.1:8000/equipment/status")
    parser.add_argument(
        "--springboot-url",
        default=None,
        help="Optional: direct PUT to Spring Boot, e.g. http://172.20.1.229:8080 (bypasses polling for immediate DB write)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Show camera feed with live mouse XY coords — no YOLO. Use to discover zone coordinates.",
    )
    args = parser.parse_args(argv)

    if args.calibrate:
        _run_calibrate(args.camera)
        return 0

    zones_path = Path(args.zones)
    if not zones_path.exists():
        print(f"Zones config not found: {zones_path}\nRun with --calibrate to discover coordinates, then create the file.", file=sys.stderr)
        return 2

    cfg = load_pipeline_config(zones_path)
    print(f"Loaded {len(cfg.zones)} zone(s) from {zones_path}")
    print(f"Timing: in_use_persist={cfg.in_use_persist_sec}s  cooldown={cfg.available_cooldown_sec}s")

    _run_pipeline(
        cfg=cfg,
        camera_index=args.camera,
        det_model=args.det_model,
        device=args.device,
        post_url=args.post_url,
        springboot_url=args.springboot_url,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
