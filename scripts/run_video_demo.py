from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gymcv.pipeline import load_pipeline_config, run_usage_pipeline_on_video

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm"}


def _is_live_cli_input(video_input: str) -> bool:
    s = video_input.strip().lower()
    if s in ("0", "webcam", "camera"):
        return True
    if s.isdigit():
        return True
    if s.startswith(("rtsp://", "http://", "https://")):
        return True
    if s.startswith("/dev/"):
        return True
    return False


def _apply_runtime_overrides(
    cfg,
    *,
    frame_skip: Optional[int],
    debug: bool,
    presence_only_mode: bool,
) -> None:
    if frame_skip is not None:
        cfg.frame_skip = int(frame_skip)
    if debug:
        cfg.debug_mode = True
    if presence_only_mode:
        cfg.presence_only_mode = True


def _apply_preview_overrides(cfg, *, preview_every_frame: bool) -> None:
    if preview_every_frame:
        cfg.preview_every_frame = True


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


def _resolve_default_zones_path() -> Optional[Path]:
    p1 = ROOT / "configs" / "zones.json"
    if p1.exists():
        return p1
    p2 = ROOT / "configs" / "zones.example.json"
    if p2.exists():
        return p2
    return None


def _collect_videos_from_input(video_input: str) -> list[Path]:
    """
    Accepts:
      - a direct file path
      - a directory path (recursively scans video files)
      - a file prefix without extension (tries matching siblings and recursive parent)
    """
    p = Path(video_input)
    if p.is_file():
        return [p]

    if p.is_dir():
        vids = [x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in VIDEO_EXTS]
        return sorted(vids)

    # Prefix mode: e.g. gym_videos/.../push-up (no extension)
    parent = p.parent if str(p.parent) else Path(".")
    stem = p.name
    matches: list[Path] = []
    if parent.exists():
        for x in parent.rglob("*"):
            if not x.is_file():
                continue
            if x.suffix.lower() not in VIDEO_EXTS:
                continue
            if x.stem == stem or x.name.startswith(stem):
                matches.append(x)
    return sorted(matches)


def _load_zones_map(path: Optional[str]) -> dict:
    if not path:
        return {"rules": [], "default": None}
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    rules = data.get("rules", []) if isinstance(data, dict) else []
    default = data.get("default") if isinstance(data, dict) else None
    return {"rules": rules, "default": default}


def _pick_zones_for_video(video: Path, zones_map: dict, fallback_zones: Optional[Path]) -> Path:
    video_str = str(video).lower()
    for rule in zones_map.get("rules", []):
        match = str(rule.get("match", "")).lower()
        zones = rule.get("zones")
        if match and zones and match in video_str:
            zp = Path(zones)
            if zp.exists():
                return zp
    default_from_map = zones_map.get("default")
    if default_from_map:
        dp = Path(default_from_map)
        if dp.exists():
            return dp
    if fallback_zones and fallback_zones.exists():
        return fallback_zones
    raise FileNotFoundError("No usable zones config found. Provide --zones and/or --zones-map.")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default="gym_videos",
        help="Video input path: file, directory, or name prefix without extension",
    )
    parser.add_argument("--zones", default=None, help="Default zones config path (optional if configs/zones.json exists)")
    parser.add_argument(
        "--zones-map",
        default=None,
        help="Optional JSON mapping for multi-camera views. See configs/zones_map.example.json",
    )
    parser.add_argument("--det-model", default="yolov8x.pt", help="Ultralytics detection model path/name")
    parser.add_argument("--pose-model", default="yolov8x-pose.pt", help="Ultralytics pose model path/name")
    parser.add_argument("--device", default=None, help="e.g. 'cpu', '0' for GPU")
    parser.add_argument("--post-url", default=None, help="Optional API URL, e.g. http://127.0.0.1:8000/equipment/status")
    parser.add_argument("--frame-skip", type=int, default=None, help="Extra frames to discard between processed frames (live/RTSP)")
    parser.add_argument("--debug", action="store_true", help="Draw zones, status, confidence, FPS (press q in window to stop)")
    parser.add_argument(
        "--presence-only-mode",
        action="store_true",
        help="Mark zone IN_USE based on person presence (no motion required), useful for class demos",
    )
    parser.add_argument(
        "--preview-every-frame",
        action="store_true",
        help="Show preview on every captured frame while running inference at frame_interval_sec",
    )
    args = parser.parse_args(argv)

    default_zones = Path(args.zones) if args.zones else _resolve_default_zones_path()
    zones_map = _load_zones_map(args.zones_map)

    if _is_live_cli_input(args.video):
        if not default_zones or not default_zones.exists():
            print("Live/RTSP mode requires --zones or configs/zones.json", file=sys.stderr)
            return 2
        cfg = load_pipeline_config(default_zones)
        _apply_runtime_overrides(
            cfg,
            frame_skip=args.frame_skip,
            debug=args.debug,
            presence_only_mode=args.presence_only_mode,
        )
        _apply_preview_overrides(cfg, preview_every_frame=args.preview_every_frame)
        print(f"Live input: {args.video!r} | zones: {default_zones}")
        for status in run_usage_pipeline_on_video(
            video_path=args.video,
            cfg=cfg,
            det_model_path=args.det_model,
            pose_model_path=args.pose_model,
            device=args.device,
        ):
            payload = asdict(status)
            print(json.dumps(payload, ensure_ascii=False))
            if args.post_url:
                try:
                    _post_json(args.post_url, payload)
                except Exception as e:
                    print(f"POST failed: {e}", file=sys.stderr)
        return 0

    videos = _collect_videos_from_input(args.video)
    if not videos:
        print(
            "No video files found for input: "
            f"{args.video}\n"
            "Tip: pass a full filename with extension, or pass a folder containing videos.",
            file=sys.stderr,
        )
        return 2

    for video in videos:
        zones_path = _pick_zones_for_video(video, zones_map, default_zones)
        cfg = load_pipeline_config(zones_path)
        _apply_runtime_overrides(
            cfg,
            frame_skip=args.frame_skip,
            debug=args.debug,
            presence_only_mode=args.presence_only_mode,
        )
        _apply_preview_overrides(cfg, preview_every_frame=args.preview_every_frame)
        print(f"Processing video: {video} | zones: {zones_path}")
        for status in run_usage_pipeline_on_video(
            video_path=video,
            cfg=cfg,
            det_model_path=args.det_model,
            pose_model_path=args.pose_model,
            device=args.device,
        ):
            payload = asdict(status)
            print(json.dumps(payload, ensure_ascii=False))
            if args.post_url:
                try:
                    _post_json(args.post_url, payload)
                except Exception as e:
                    print(f"POST failed: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

